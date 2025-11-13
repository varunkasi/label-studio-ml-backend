import os
import cv2
from tqdm import tqdm
import subprocess
import tempfile
import re


def convert_labelstudio_to_yolo(
    labelstudio_json: dict,
    output_labels_dir: str,
    output_frames_dir: str = None,
    video_path: str = None,
    jpeg_quality: int = 95,
    class_names=None,
    save_empty_labels: bool = True,
    reencode_video: bool = False,
    reencode_fps: float = None
):
    """
    Converts Label Studio video annotations (given as a Python dict) into YOLO format 
    and optionally extracts video frames.

    Args:
        labelstudio_json (dict): Parsed Label Studio JSON data (not a file path).
        output_labels_dir (str): Directory to save YOLO label .txt files.
        output_frames_dir (str, optional): Directory to save extracted video frames. Ignored if `video_path` is None.
        video_path (str, optional): Path to the source video for frame extraction.
        jpeg_quality (int, optional): JPEG quality for frame extraction (0-100). Default is 95.
        class_names (list[str], optional): List of YOLO class names. Default is ["Person"].
        save_empty_labels (bool, optional): Whether to save empty .txt files for frames without labels.
        reencode_video (bool, optional): Whether to re-encode the video before frame extraction. Default False.
        reencode_fps (float, optional): Target FPS for re-encoding. If None, original FPS is preserved.

    Returns:
        tuple[str, Optional[str]]: Paths to (output_labels_dir, output_frames_dir)
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

    # --- Prepare output directories ---
    os.makedirs(output_labels_dir, exist_ok=True)
    if video_path and output_frames_dir:
        os.makedirs(output_frames_dir, exist_ok=True)

    # --- Get video properties ---
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

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={target_fps}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an", reencoded_path
            ]

            # Launch ffmpeg and capture stderr for progress updates
            process = subprocess.Popen(
                ffmpeg_cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                universal_newlines=True,
                bufsize=1
            )

            # tqdm progress bar
            pbar = tqdm(total=total_frames, desc="Re-encoding", unit="frame")
            frame_pattern = re.compile(r"frame=\s*(\d+)")

            for line in process.stderr:
                match = frame_pattern.search(line)
                if match:
                    frame = int(match.group(1))
                    pbar.n = frame
                    pbar.refresh()

            process.wait()
            pbar.close()

            if process.returncode != 0:
                raise RuntimeError("❌ ffmpeg re-encoding failed")

            # Update properties after re-encoding
            cap = cv2.VideoCapture(reencoded_path)
            image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"✅ Video re-encoded to {target_fps:.2f} FPS ({total_frames} frames)")
    else:
        image_width = 1920
        image_height = 1080
        total_frames = None
        fps = None

    # --- Use provided Label Studio JSON data ---
    data = labelstudio_json
    annotations = data["result"]
    frames_dict_all = {}  # frame_number -> list of YOLO lines

    # --- Process annotations ---
    for ann in tqdm(annotations, desc="Processing tracks"):
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
                    class_id = class_names.index(ann["value"]["labels"][0])
                    x_center = (f["x"] + f["width"] / 2) / 100
                    y_center = (f["y"] + f["height"] / 2) / 100
                    w = f["width"] / 100
                    h = f["height"] / 100
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                    frames_dict_all.setdefault(f["frame"], []).append(yolo_line)

            # Current frame
            class_id = class_names.index(ann["value"]["labels"][0])
            x_center = (frame["x"] + frame["width"] / 2) / 100
            y_center = (frame["y"] + frame["height"] / 2) / 100
            w = frame["width"] / 100
            h = frame["height"] / 100
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            frames_dict_all.setdefault(frame["frame"], []).append(yolo_line)

            last_enabled_frame = frame

    # Determine total frames if missing
    if total_frames is None:
        total_frames = max(frames_dict_all.keys()) if frames_dict_all else 0

    # --- Write YOLO annotations ---
    for frame_number in tqdm(range(1, total_frames + 1), desc="Writing YOLO annotations"):
        lines = frames_dict_all.get(frame_number, [])
        if lines or save_empty_labels:
            txt_file = os.path.join(output_labels_dir, f"frame_{frame_number:06d}.txt")
            with open(txt_file, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    # --- Extract frames ---
    if video_path and output_frames_dir:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break
            frame_idx += 1
            frame_file = os.path.join(output_frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            pbar.update(1)
        cap.release()
        pbar.close()

    print(f"✅ YOLO annotations saved to {output_labels_dir}/")
    if video_path and output_frames_dir:
        print(f"✅ Video frames saved to {output_frames_dir}/")

    # --- Return output directories ---
    return output_labels_dir, output_frames_dir
