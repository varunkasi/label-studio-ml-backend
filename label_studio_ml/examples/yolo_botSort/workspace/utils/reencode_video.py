import os
import shutil
import subprocess
import cv2
from tqdm import tqdm
import re

def reencode_video(video_path: str) -> str:
    """
    Reencodes the video at video_path using ffmpeg and replaces the original file.
    Uses the FPS read from the video using OpenCV.
    Shows a progress bar based on frame count.
    Returns the path to the reencoded video (same as input).
    """
    # Read FPS and frame count using OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 0:
        fps = 30  # fallback to 30 if unable to read
    print(f"Reencoding video at {video_path} with FPS: {fps}, Total frames: {total_frames}")

    temp_path = video_path + ".reencoded.mp4"
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-r", str(fps),
        temp_path
    ]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pbar = tqdm(total=total_frames, desc="Reencoding", unit="frame")
        frame_pattern = re.compile(r"frame=\s*(\d+)")
        current_frame = 0
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            match = frame_pattern.search(line)
            if match:
                frame_num = int(match.group(1))
                if frame_num > current_frame:
                    pbar.update(frame_num - current_frame)
                    current_frame = frame_num
        process.wait()
        pbar.close()
        if process.returncode != 0:
            raise Exception("ffmpeg failed")
        shutil.move(temp_path, video_path)
        return video_path
    except Exception as e:
        print(f"Failed to reencode video {video_path}: {e}")
        return video_path
