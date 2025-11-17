import os
import cv2
import json
from tqdm import tqdm
import subprocess
import tempfile
import re


def convert_labelstudio_to_coco(
    labelstudio_json: dict,
    output_coco_file: str,
    output_frames_dir: str = None,
    video_path: str = None,
    jpeg_quality: int = 95,
    class_names=None,
    reencode_video: bool = False,
    reencode_fps: float = None
):
    """
    Converts Label Studio video annotations (given as a Python dict) into COCO format 
    and optionally extracts video frames.

    Args:
        labelstudio_json (dict): Parsed Label Studio JSON data (not a file path).
        output_coco_file (str): Path to save COCO format JSON file.
        output_frames_dir (str, optional): Directory to save extracted video frames. Ignored if `video_path` is None.
        video_path (str, optional): Path to the source video for frame extraction.
        jpeg_quality (int, optional): JPEG quality for frame extraction (0-100). Default is 95.
        class_names (list[str], optional): List of class names. Default is ["Person"].
        reencode_video (bool, optional): Whether to re-encode the video before frame extraction. Default False.
        reencode_fps (float, optional): Target FPS for re-encoding. If None, original FPS is preserved.

    Returns:
        tuple[str, Optional[str]]: Paths to (output_coco_file, output_frames_dir)
    """
    if class_names is None:
        class_names = ["Person"]

    # --- Helper: linear interpolation ---
    def interpolate(frame1, frame2):
        """Interpolate only if both frames are enabled."""
        f1, f2 = frame1["frame"], frame2["frame"]
        if not frame1["enabled"] or not frame2["enabled"]:
            return []
        n_frames = f2 - f1 - 1
        interpolated = []
        for i in range(1, n_frames + 1):
            t = i / (n_frames + 1)
            x = frame1["x"] + t * (frame2["x"] - frame1["x"])
            y = frame1["y"] + t * (frame2["y"] - frame1["y"])
            w = frame1["width"] + t * (frame2["width"] - frame1["width"])
            h = frame1["height"] + t * (frame2["height"] - frame1["height"])
            interpolated.append({
                "frame": f1 + i,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "enabled": True
            })
        return interpolated

    # --- Prepare output directory for COCO file ---
    os.makedirs(os.path.dirname(output_coco_file), exist_ok=True)
    if video_path and output_frames_dir:
        os.makedirs(output_frames_dir, exist_ok=True)

    # --- Get video properties ---
    reencoded_path = None  # Track re-encoded video path
    if video_path:
        cap = cv2.VideoCapture(video_path)
        image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if reencode_video:
            print("🔄 Re-encoding video for consistent frame extraction...")
            temp_dir = tempfile.mkdtemp()
            reencoded_path = os.path.join(temp_dir, "reencoded_video.mp4")
            target_fps = reencode_fps if reencode_fps else fps

            # Use -progress option for real-time updates
            progress_file = os.path.join(temp_dir, "ffmpeg_progress.log")
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={target_fps}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an", reencoded_path,
                "-progress", progress_file
            ]

            # Launch ffmpeg
            process = subprocess.Popen(
                ffmpeg_cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                universal_newlines=True,
                bufsize=1
            )

            # tqdm progress bar
            pbar = tqdm(total=total_frames, desc="Re-encoding", unit="frame")
            last_frame = 0  # Track the last frame number to ensure sequential updates

            while process.poll() is None:
                if os.path.exists(progress_file):
                    with open(progress_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            if "frame=" in line:
                                try:
                                    frame = int(line.split("=")[1].strip())
                                    if frame > last_frame:  # Only update if the frame number is greater
                                        pbar.n = frame
                                        pbar.refresh()
                                        last_frame = frame
                                except ValueError:
                                    continue

            pbar.n = total_frames  # Ensure progress bar completes
            pbar.close()

            if process.returncode != 0:
                raise RuntimeError("❌ ffmpeg re-encoding failed")

            # Update properties after re-encoding and use re-encoded video for extraction
            cap = cv2.VideoCapture(reencoded_path)
            image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"✅ Video re-encoded to {target_fps:.2f} FPS ({total_frames} frames)")
            
            # Use re-encoded video for frame extraction
            video_path = reencoded_path
    else:
        image_width = 1920
        image_height = 1080
        total_frames = None
        fps = None

    # --- Initialize COCO format structure ---
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for i, class_name in enumerate(class_names):
        coco_data["categories"].append({"id": i + 1, "name": class_name})

    # --- Use provided Label Studio JSON data ---
    data = labelstudio_json
    annotations = data["result"]
    frames_dict_all = {}  # frame_number -> list of annotation dicts
    annotation_id = 1

    # --- Process annotations ---
    for ann in tqdm(annotations, desc="Processing tracks"):
        # Skip tracks that don't have labels
        if "labels" not in ann["value"] or not ann["value"]["labels"]:
            continue
        
        seq = ann["value"]["sequence"]
        seq = sorted(seq, key=lambda x: x["frame"])
        last_enabled_frame = None

        for frame in seq:
            if not frame["enabled"]:
                last_enabled_frame = None
                continue

            # Interpolate between frames
            if last_enabled_frame is not None:
                for f in interpolate(last_enabled_frame, frame):
                    class_id = class_names.index(ann["value"]["labels"][0]) + 1
                    # Convert from percentage to pixel coordinates
                    x = f["x"] * image_width / 100
                    y = f["y"] * image_height / 100
                    w = f["width"] * image_width / 100
                    h = f["height"] * image_height / 100
                    
                    annotation = {
                        "id": annotation_id,
                        "image_id": f["frame"],
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    }
                    frames_dict_all.setdefault(f["frame"], []).append(annotation)
                    annotation_id += 1

            # Current frame
            class_id = class_names.index(ann["value"]["labels"][0]) + 1
            # Convert from percentage to pixel coordinates
            x = frame["x"] * image_width / 100
            y = frame["y"] * image_height / 100
            w = frame["width"] * image_width / 100
            h = frame["height"] * image_height / 100
            
            annotation = {
                "id": annotation_id,
                "image_id": frame["frame"],
                "category_id": class_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            frames_dict_all.setdefault(frame["frame"], []).append(annotation)
            annotation_id += 1

            last_enabled_frame = frame

    # Determine total frames if missing
    if total_frames is None:
        total_frames = max(frames_dict_all.keys()) if frames_dict_all else 0

    # --- Build COCO images and annotations ---
    for frame_number in tqdm(range(1, total_frames + 1), desc="Building COCO annotations"):
        # Add image info
        coco_data["images"].append({
            "id": frame_number,
            "file_name": f"frame_{frame_number:06d}.jpg",
            "width": image_width,
            "height": image_height
        })
        
        # Add annotations for this frame
        annotations_for_frame = frames_dict_all.get(frame_number, [])
        coco_data["annotations"].extend(annotations_for_frame)

    # --- Write COCO JSON file ---
    with open(output_coco_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # --- Extract frames using time-based approach for Label Studio compatibility ---
    if video_path and output_frames_dir:
        def _save_frame(frame_number, img):
            frame_file = os.path.join(output_frames_dir, f"frame_{frame_number:06d}.jpg")
            cv2.imwrite(frame_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        cap = cv2.VideoCapture(video_path)
        orig_frame_rate = cap.get(cv2.CAP_PROP_FPS)
        orig_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use original frame rate if no reencode_fps specified
        target_frame_rate = reencode_fps if reencode_fps else orig_frame_rate
        target_length = round(orig_length / orig_frame_rate * target_frame_rate)
        
        print(f'Original frames: {orig_length}, Original FPS: {orig_frame_rate:.2f}')
        print(f'Target frames: {target_length}, Target FPS: {target_frame_rate:.2f}')
        
        pbar = tqdm(total=target_length, desc="Extracting frames")
        
        # Start from first frame
        frame_id = 1  # Start from 1 to match COCO image IDs
        success, frame = cap.read()
        if success:
            _save_frame(frame_id, frame)
            pbar.update(1)
        
        last_position_s = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            position_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            delta_s = position_s - last_position_s
            if delta_s >= (1 / target_frame_rate):
                frame_id += 1
                if frame_id > target_length:  # Don't exceed expected frame count
                    break
                _save_frame(frame_id, frame)
                pbar.update(1)
                last_position_s += 1 / target_frame_rate
        
        cap.release()
        pbar.close()
        print(f'Done. Extracted {frame_id} frames to {output_frames_dir}/')
        
        # Update total_frames to match actual extracted frames
        total_frames = frame_id

    print(f"✅ COCO annotations saved to {output_coco_file}")
    if video_path and output_frames_dir:
        print(f"✅ Video frames saved to {output_frames_dir}/")

    # --- Return output paths ---
    return output_coco_file, output_frames_dir