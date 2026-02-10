"""Extract per-frame instance segmentation masks from snippet videos using SAM3.

End-to-end pipeline:
  1. SAM3 inference — per-frame mask extraction using Sam3Model with text+box prompts
  2. ffmpeg encoding — mask-only and overlay videos (same encoding as overlay_snippet_bboxes.sh)
  3. Permission fix — chown outputs to host user when running inside Docker

Usage (single snippet):
  python extract_snippet_masks.py \
    --snippet snippets_proj.../casualty_2_f25425-25601_fps25.mp4 \
    --bbox-json snippets_proj.../casualty_2_f25425-25601_fps25.json

Usage (batch — all pairs in a folder):
  python extract_snippet_masks.py --batch-dir snippets_proj.../
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def load_bbox_json(path: str) -> List[Dict[str, Any]]:
    """Load per-frame bounding box JSON (array of {x, y, width, height, snippet_frame})."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"BBox JSON is empty or not an array: {path}")
    return data


def percent_xywh_to_xyxy(
    x_pct: float, y_pct: float, w_pct: float, h_pct: float,
    img_w: int, img_h: int,
) -> List[int]:
    """Convert percent [0,100] xywh bbox to pixel [x1, y1, x2, y2]."""
    x1 = int(round((x_pct / 100.0) * img_w))
    y1 = int(round((y_pct / 100.0) * img_h))
    x2 = int(round(((x_pct + w_pct) / 100.0) * img_w))
    y2 = int(round(((y_pct + h_pct) / 100.0) * img_h))
    return [
        max(0, min(img_w, x1)), max(0, min(img_h, y1)),
        max(0, min(img_w, x2)), max(0, min(img_h, y2)),
    ]


def build_frame_bbox_map(
    entries: List[Dict[str, Any]], img_w: int, img_h: int,
) -> Dict[int, List[int]]:
    """Build 0-based frame index -> pixel xyxy bbox mapping."""
    frame_map: Dict[int, List[int]] = {}
    for entry in entries:
        sf = int(entry["snippet_frame"])
        frame_map[sf - 1] = percent_xywh_to_xyxy(
            float(entry["x"]), float(entry["y"]),
            float(entry["width"]), float(entry["height"]),
            img_w, img_h,
        )
    return frame_map


# ---------------------------------------------------------------------------
# SAM3 mask extraction
# ---------------------------------------------------------------------------

def extract_mask_for_frame(
    frame: Image.Image,
    box_xyxy: List[int],
    *,
    model,
    processor,
    text_prompt: str = "person",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cpu",
    dtype=None,
) -> Tuple[np.ndarray, float]:
    """Run Sam3Model on a single frame and return (mask_uint8, score)."""
    w, h = frame.size
    blank = np.zeros((h, w), dtype=np.uint8)

    try:
        inputs = processor(
            images=frame,
            text=text_prompt,
            input_boxes=[[box_xyxy]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            if dtype is not None and device != "cpu":
                with torch.autocast(device_type=device, dtype=dtype):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        original_sizes = inputs.get("original_sizes")
        if hasattr(original_sizes, "tolist"):
            original_sizes = original_sizes.tolist()
        results = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=mask_threshold,
            target_sizes=original_sizes,
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])
        if len(masks) == 0:
            return blank, 0.0

        score_vals = [
            s.item() if hasattr(s, "item") else float(s) for s in scores
        ] if len(scores) > 0 else [0.5]
        best_idx = int(np.argmax(score_vals))
        best_score = score_vals[best_idx]

        best_mask = masks[best_idx]
        # bfloat16 safety: always .float() before .numpy()
        if hasattr(best_mask, "cpu"):
            best_mask = best_mask.cpu().float().numpy()
        elif hasattr(best_mask, "numpy"):
            best_mask = best_mask.float().numpy()

        mask_uint8 = (best_mask.astype(bool).astype(np.uint8)) * 255
        if mask_uint8.ndim > 2:
            mask_uint8 = mask_uint8.squeeze()
        if mask_uint8.shape != (h, w):
            mask_uint8 = np.array(
                Image.fromarray(mask_uint8, mode="L").resize((w, h), Image.NEAREST)
            )
        return mask_uint8, best_score

    except Exception as exc:
        logger.warning("SAM3 mask extraction failed: %s", exc)
        return blank, 0.0


def process_snippet(
    video_path: str,
    bbox_json_path: str,
    output_dir: str,
    *,
    model=None,
    processor=None,
    text_prompt: str = "person",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cpu",
    dtype=None,
) -> Dict[str, Any]:
    """Extract SAM3 masks for every frame and save as PNGs + scores.json."""
    import av

    if model is None or processor is None:
        import seeding_common as base
        model, processor = base._get_sam3_image_model()
        device = base.DEVICE
        dtype = base.DTYPE

    entries = load_bbox_json(bbox_json_path)

    container = av.open(video_path)
    stream = container.streams.video[0]
    img_w = stream.codec_context.width
    img_h = stream.codec_context.height
    total_frames = stream.frames
    if not total_frames and stream.duration and stream.time_base:
        fps_est = float(stream.average_rate) if stream.average_rate else 30.0
        total_frames = int(float(stream.duration * stream.time_base) * fps_est)
    container.close()

    frame_map = build_frame_bbox_map(entries, img_w, img_h)
    os.makedirs(output_dir, exist_ok=True)

    container = av.open(video_path)
    scores_map: Dict[str, float] = {}
    frames_processed = 0
    frames_blank = 0
    frame_idx = 0

    try:
        for av_frame in container.decode(video=0):
            if total_frames > 0 and frame_idx >= total_frames:
                break

            snippet_frame = frame_idx + 1
            mask_path = os.path.join(output_dir, f"mask_{snippet_frame:06d}.png")

            if frame_idx in frame_map:
                mask, score = extract_mask_for_frame(
                    av_frame.to_image(), frame_map[frame_idx],
                    model=model, processor=processor,
                    text_prompt=text_prompt, threshold=threshold,
                    mask_threshold=mask_threshold, device=device, dtype=dtype,
                )
                frames_processed += 1
                scores_map[str(snippet_frame)] = round(float(score), 4)
            else:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                frames_blank += 1
                scores_map[str(snippet_frame)] = 0.0

            Image.fromarray(mask, mode="L").save(mask_path)
            frame_idx += 1

            if frame_idx % 50 == 0 or frame_idx == total_frames:
                logger.info(
                    "Progress: %d/%d frames (processed=%d, blank=%d)",
                    frame_idx, total_frames, frames_processed, frames_blank,
                )
    finally:
        container.close()

    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(scores_map, f, indent=2)

    avg_score = (
        sum(v for v in scores_map.values() if v > 0) / max(frames_processed, 1)
    )
    stats = {
        "frames_processed": frames_processed,
        "frames_blank": frames_blank,
        "total_frames": frame_idx,
        "avg_score": round(avg_score, 4),
    }
    logger.info("Masks done: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# ffmpeg video encoding (matches overlay_snippet_bboxes.sh)
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd: List[str]) -> None:
    """Run an ffmpeg command, raising on failure."""
    logger.debug("ffmpeg: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {r.stderr[-500:]}")


def _get_video_fps(video_path: str) -> int:
    """Get video FPS via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    # r_frame_rate comes as "25/1" or "30000/1001"
    num, den = r.stdout.strip().split("/")
    return round(int(num) / int(den))


def _get_video_dims(video_path: str) -> Tuple[int, int]:
    """Get video (width, height) via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
        capture_output=True, text=True,
    )
    w, h = r.stdout.strip().split("x")
    return int(w), int(h)


def encode_mask_video(mask_dir: str, output_path: str, fps: int) -> None:
    """Encode mask PNGs -> MP4 using ffmpeg (same flags as overlay_snippet_bboxes.sh)."""
    _run_ffmpeg([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-start_number", "1",
        "-i", os.path.join(mask_dir, "mask_%06d.png"),
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        output_path,
    ])
    logger.info("Wrote mask video: %s", output_path)


def encode_overlay_video(
    source_video: str, mask_dir: str, output_path: str,
    fps: int, width: int, height: int,
) -> None:
    """Composite masks onto video as translucent green overlay using ffmpeg."""
    # Write filter to temp file (avoids shell escaping in docker exec)
    filter_text = (
        f"color=c=0x00FF64:s={width}x{height}:r={fps},format=rgba[green];\n"
        f"[2:v]format=gray,lut=c0=val/4[alpha];\n"
        f"[green][alpha]alphamerge[green_overlay];\n"
        f"[0:v][green_overlay]overlay=shortest=1:format=auto\n"
    )
    fd, filter_path = tempfile.mkstemp(suffix=".txt", prefix="overlay_filter_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(filter_text)
        _run_ffmpeg([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", source_video,
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}",
            "-framerate", str(fps), "-start_number", "1",
            "-i", os.path.join(mask_dir, "mask_%06d.png"),
            "-filter_complex_script", filter_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            output_path,
        ])
    finally:
        os.unlink(filter_path)
    logger.info("Wrote overlay video: %s", output_path)


# ---------------------------------------------------------------------------
# Permission fix
# ---------------------------------------------------------------------------

def fix_ownership(path: str, uid: int = 1000, gid: int = 1000) -> None:
    """Recursively chown outputs to host user (no-op if not root)."""
    if os.getuid() != 0:
        return
    for root, dirs, files in os.walk(path):
        os.chown(root, uid, gid)
        for name in files:
            os.chown(os.path.join(root, name), uid, gid)
    # Also chown any sibling video files
    parent = os.path.dirname(path)
    base = os.path.basename(path)
    if base.endswith("_masks"):
        stem = base[:-6]  # strip _masks suffix
        for suffix in ("_masks.mp4", "_overlaid_masks.mp4"):
            fp = os.path.join(parent, stem + suffix)
            if os.path.isfile(fp):
                os.chown(fp, uid, gid)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_pairs(directory: str) -> List[Tuple[str, str]]:
    """Find all MP4+JSON pairs in a snippet folder."""
    import glob
    pairs = []
    for mp4 in sorted(glob.glob(os.path.join(directory, "*_fps*.mp4"))):
        if any(tag in mp4 for tag in ("_bbox_overlaid", "_masks", "_overlaid_masks")):
            continue
        json_path = os.path.splitext(mp4)[0] + ".json"
        if os.path.isfile(json_path):
            pairs.append((mp4, json_path))
    return pairs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SAM3 instance segmentation masks from snippet videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--snippet",
        help="Path to a single snippet MP4 video file.",
    )
    group.add_argument(
        "--batch-dir",
        help="Path to a snippet folder — processes all MP4+JSON pairs.",
    )
    parser.add_argument("--bbox-json", help="Per-frame bbox JSON (required with --snippet).")
    parser.add_argument("--text-prompt", default="person")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-mask-video", action="store_true")
    parser.add_argument("--no-overlay-video", action="store_true")
    parser.add_argument("--host-uid", type=int, default=1000, help="Host UID for chown.")
    parser.add_argument("--host-gid", type=int, default=1000, help="Host GID for chown.")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )
    return parser.parse_args()


def process_one(
    snippet: str, bbox_json: str, args, model=None, processor=None,
    device: str = "cpu", dtype=None,
) -> Dict[str, Any]:
    """Full pipeline for a single snippet: SAM3 masks -> ffmpeg videos -> chown."""
    snippet_base = os.path.splitext(snippet)[0]
    mask_dir = f"{snippet_base}_masks"

    logger.info("  Snippet:   %s", snippet)
    logger.info("  BBox JSON: %s", bbox_json)
    logger.info("  Mask dir:  %s", mask_dir)

    # Step 1: SAM3 mask extraction
    stats = process_snippet(
        video_path=snippet, bbox_json_path=bbox_json, output_dir=mask_dir,
        model=model, processor=processor,
        text_prompt=args.text_prompt, threshold=args.threshold,
        device=device, dtype=dtype,
    )

    # Step 2: ffmpeg video encoding
    fps = _get_video_fps(snippet)
    width, height = _get_video_dims(snippet)

    if not args.no_mask_video:
        encode_mask_video(mask_dir, f"{snippet_base}_masks.mp4", fps)

    if not args.no_overlay_video:
        encode_overlay_video(
            snippet, mask_dir, f"{snippet_base}_overlaid_masks.mp4",
            fps, width, height,
        )

    # Step 3: fix ownership
    fix_ownership(mask_dir, args.host_uid, args.host_gid)

    return stats


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load model once
    import time
    logger.info("Loading SAM3 model...")
    t0 = time.time()
    import seeding_common as base
    model, processor = base._get_sam3_image_model()
    device = base.DEVICE
    dtype = base.DTYPE
    logger.info("Model loaded in %.1fs (device=%s)", time.time() - t0, device)

    # Build list of (snippet, bbox_json) pairs
    if args.batch_dir:
        pairs = _find_pairs(args.batch_dir)
        if not pairs:
            logger.error("No MP4+JSON pairs found in %s", args.batch_dir)
            sys.exit(1)
    else:
        if not args.bbox_json:
            logger.error("--bbox-json is required with --snippet")
            sys.exit(1)
        pairs = [(args.snippet, args.bbox_json)]

    logger.info("=" * 70)
    logger.info("EXTRACT SNIPPET MASKS — %d pair(s)", len(pairs))
    logger.info("=" * 70)

    results = []
    for i, (snippet, bbox_json) in enumerate(pairs, 1):
        logger.info("")
        logger.info("[%d/%d] %s", i, len(pairs), os.path.basename(snippet))
        logger.info("-" * 70)
        t1 = time.time()
        try:
            stats = process_one(
                snippet, bbox_json, args,
                model=model, processor=processor, device=device, dtype=dtype,
            )
            stats["elapsed_sec"] = round(time.time() - t1, 1)
            stats["name"] = os.path.basename(snippet)
            results.append(stats)
        except Exception as exc:
            logger.error("FAILED: %s", exc, exc_info=True)
            results.append({"name": os.path.basename(snippet), "error": str(exc)})

    # Summary
    ok = sum(1 for r in results if "error" not in r)
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPLETE — %d/%d succeeded", ok, len(results))
    logger.info("=" * 70)
    for r in results:
        if "error" in r:
            logger.info("  FAIL  %s: %s", r["name"], r["error"])
        else:
            logger.info(
                "  OK    %s: %d frames, avg_score=%.3f, %.1fs",
                r["name"], r["total_frames"], r["avg_score"], r["elapsed_sec"],
            )


if __name__ == "__main__":
    main()
