"""Unified annotation-to-artifacts pipeline.

Combines export (with interpolation), annotation extraction, snippet cutting,
bbox JSON generation, and optional SAM3 mask extraction into one script.

Usage (no GPU needed):
  python process_annotation.py \\
    --ls-url http://localhost:8080 --ls-api-key TOKEN \\
    --project 225156 --task 245567455 --annotation 85565349 \\
    --skip-masks

Usage (full pipeline with SAM3):
  python process_annotation.py \\
    --ls-url http://localhost:8080 --ls-api-key TOKEN \\
    --project 225156 --task 245567455 --annotation 85565349

Replaces the multi-tool workflow of:
  1. export_interpolated_annotation.sh  (export + snippets)
  2. extract_snippet_masks.py           (SAM3 masks + video encoding)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Export with interpolation
# ---------------------------------------------------------------------------

def _build_ls_session(ls_url: str, ls_api_key: str) -> Tuple[str, "requests.Session"]:
    """Build an authenticated requests.Session for the Label Studio API.

    NOTE: Duplicated from video_tools._build_ls_api intentionally.
    process_annotation.py is designed to run standalone without depending
    on video_tools internals (which use a different return type).
    """
    import requests

    base_url = ls_url.rstrip("/")
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Token {ls_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    return base_url, session


def create_export_snapshot(
    base_url: str, session: "requests.Session", project_id: int,
) -> int:
    """POST /api/projects/{PID}/exports with interpolated keyframes enabled."""
    title = (
        f"Interpolated Export proj{project_id}"
        f"_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    payload = {
        "title": title,
        "serialization_options": {"interpolate_key_frames": True},
    }
    url = f"{base_url}/api/projects/{project_id}/exports"
    resp = session.post(url, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Failed to create export snapshot (status={resp.status_code}): "
            f"{resp.text[:300]}"
        )
    data = resp.json()
    export_id = data.get("id") or data.get("pk")
    if export_id is None:
        raise RuntimeError(f"Could not parse export id from response: {data}")
    return int(export_id)


def wait_for_export(
    base_url: str,
    session: "requests.Session",
    project_id: int,
    export_id: int,
    *,
    poll_interval: int = 5,
    timeout: int = 300,
) -> None:
    """Poll GET until export status is 'completed'."""
    url = f"{base_url}/api/projects/{project_id}/exports/{export_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = session.get(url, timeout=30)
        data = resp.json()
        status = data.get("status") or data.get("state") or "unknown"
        if status == "completed":
            logger.info("Export %d completed", export_id)
            return
        if status in ("failed", "error"):
            raise RuntimeError(f"Export {export_id} failed: {data}")
        logger.debug("Export status: %s — polling in %ds", status, poll_interval)
        time.sleep(poll_interval)
    raise RuntimeError(f"Timed out waiting for export {export_id} after {timeout}s")


def download_export(
    base_url: str,
    session: "requests.Session",
    project_id: int,
    export_id: int,
    tmp_dir: str,
) -> str:
    """Download export JSON (or ZIP containing JSON). Returns path to JSON file."""
    url = f"{base_url}/api/projects/{project_id}/exports/{export_id}/download?exportType=JSON"
    # Drop Content-Type for download — the server rejects it on GET endpoints
    resp = session.get(url, timeout=120, headers={"Content-Type": None})
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    body = resp.content

    if "application/zip" in content_type:
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            first_entry = zf.namelist()[0]
            json_bytes = zf.read(first_entry)
        json_path = os.path.join(tmp_dir, "export.json")
        with open(json_path, "wb") as f:
            f.write(json_bytes)
        return json_path

    # application/json or anything else — treat as raw JSON
    json_path = os.path.join(tmp_dir, "export.json")
    with open(json_path, "wb") as f:
        f.write(body)
    return json_path


# ---------------------------------------------------------------------------
# Phase 2: Extract annotation + build summary
# ---------------------------------------------------------------------------

def extract_annotation(
    export_data: Any,
    project_id: int,
    task_id: int,
    annotation_id: int,
) -> Dict[str, Any]:
    """Normalize export format, find task+annotation, extract video_url+fps.

    Handles four LS export shapes:
      - top-level array of tasks
      - {"tasks": [...]}
      - {"data": [...]}
      - {"results": [...]}
    """
    # Normalize to list of task objects
    if isinstance(export_data, list):
        tasks = export_data
    elif isinstance(export_data, dict):
        for key in ("tasks", "data", "results"):
            if key in export_data and isinstance(export_data[key], list):
                tasks = export_data[key]
                break
        else:
            raise RuntimeError("Unsupported export structure (no tasks/data/results key)")
    else:
        raise RuntimeError(f"Unexpected export type: {type(export_data)}")

    # Find the target task
    task_obj = None
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = t.get("id") or t.get("task_id")
        if tid is not None and int(tid) == task_id:
            task_obj = t
            break
    if task_obj is None:
        raise RuntimeError(f"Task {task_id} not found in export")

    # Extract video URL from task data
    task_data = task_obj.get("data", {})
    video_url = None
    for key in ("video", "video_url", "videoUrl", "video_path", "videoPath",
                "source", "video_source"):
        if key in task_data and task_data[key]:
            video_url = task_data[key]
            break
    fps = task_data.get("fps")

    # Find annotation or prediction by ID
    ann_id_str = str(annotation_id)
    source_type = None
    entry_data = None

    for ann in (task_obj.get("annotations") or []):
        if not isinstance(ann, dict):
            continue
        aid = ann.get("id") or ann.get("annotation_id")
        if aid is not None and str(aid) == ann_id_str:
            source_type = "annotation"
            entry_data = ann
            break

    if entry_data is None:
        for pred in (task_obj.get("predictions") or []):
            if not isinstance(pred, dict):
                continue
            pid = pred.get("id") or pred.get("prediction_id")
            if pid is not None and str(pid) == ann_id_str:
                source_type = "prediction"
                entry_data = pred
                break

    if entry_data is None:
        raise RuntimeError(
            f"Annotation/prediction {annotation_id} not found within task {task_id}"
        )

    result = {
        "project_id": project_id,
        "task_id": task_id,
        "source_type": source_type,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "video_url": video_url,
        "fps": fps,
    }
    if source_type == "annotation":
        result["annotation_id"] = annotation_id
        result["annotation"] = entry_data
    else:
        result["prediction_id"] = annotation_id
        result["prediction"] = entry_data

    return result


def _parse_casualty_id(result_item: Dict[str, Any]) -> Optional[str]:
    """Extract casualty ID from meta.text via pattern 'id:\\d+'."""
    meta = result_item.get("meta", {})
    if not isinstance(meta, dict):
        return None
    text_list = meta.get("text", [])
    if isinstance(text_list, str):
        text_list = [text_list]
    if not isinstance(text_list, list):
        return None
    for text in text_list:
        if not isinstance(text, str):
            continue
        m = re.search(r"id:(\d+)", text)
        if m:
            return m.group(1)
    return None


def _merge_consecutive_frames(
    sequence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge consecutive frames into {start_frame, end_frame, start_time, end_time} ranges."""
    if not sequence:
        return []

    sorted_seq = sorted(sequence, key=lambda s: s.get("frame", 0))
    ranges: List[Dict[str, Any]] = []
    current = None

    for item in sorted_seq:
        frame = item.get("frame")
        if frame is None:
            continue
        frame = int(frame)
        t = item.get("time")

        if current is None:
            current = {
                "start_frame": frame,
                "end_frame": frame,
                "start_time": t,
                "end_time": t,
            }
        elif frame == current["end_frame"] + 1:
            current["end_frame"] = frame
            if t is not None:
                current["end_time"] = t
        else:
            ranges.append(current)
            current = {
                "start_frame": frame,
                "end_frame": frame,
                "start_time": t,
                "end_time": t,
            }

    if current is not None:
        ranges.append(current)

    return ranges


def generate_summary(annotation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse casualty IDs and group videorectangle sequences into frame ranges."""
    if annotation_data["source_type"] == "annotation":
        results = (annotation_data.get("annotation") or {}).get("result", [])
    else:
        results = (annotation_data.get("prediction") or {}).get("result", [])

    casualties: Dict[str, Dict[str, Any]] = {}

    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "videorectangle":
            continue

        casualty_id = _parse_casualty_id(item)
        if casualty_id is None:
            continue

        sequence = (item.get("value") or {}).get("sequence", [])
        ranges = _merge_consecutive_frames(sequence)

        if casualty_id not in casualties:
            casualties[casualty_id] = {"ranges": []}
        casualties[casualty_id]["ranges"].extend(ranges)

    return {
        "project_id": annotation_data.get("project_id"),
        "task_id": annotation_data.get("task_id"),
        "annotation_id": annotation_data.get("annotation_id"),
        "prediction_id": annotation_data.get("prediction_id"),
        "source_type": annotation_data.get("source_type"),
        "video_url": annotation_data.get("video_url"),
        "fps": annotation_data.get("fps"),
        "casualties": casualties,
    }


# ---------------------------------------------------------------------------
# Phase 3: Download source video
# ---------------------------------------------------------------------------

def download_video(
    base_url: str,
    session: "requests.Session",
    video_url: str,
    dest_path: str,
) -> None:
    """Streaming download of source video with auth header."""
    if video_url.startswith("http"):
        full_url = video_url
    elif video_url.startswith("/"):
        full_url = base_url + video_url
    else:
        full_url = base_url + "/" + video_url

    logger.info("Downloading video: %s", full_url)
    # Drop Content-Type for file download — server rejects it on GET endpoints
    resp = session.get(full_url, stream=True, timeout=600, headers={"Content-Type": None})
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    logger.info("Saved video (%d bytes) to %s", os.path.getsize(dest_path), dest_path)


# ---------------------------------------------------------------------------
# Phase 4: Cut snippets
# ---------------------------------------------------------------------------

def compute_keep_frames(
    start_frame: int,
    end_frame: int,
    start_time: Optional[float],
    end_time: Optional[float],
    source_fps: float,
    target_fps: float,
) -> List[int]:
    """Compute which source frames to keep for the target FPS.

    If FPS matches: simple range(start, end+1).
    Otherwise: resample by stepping through time at 1/target_fps intervals.

    Note: when upsampling (target_fps > source_fps), the returned list may
    contain duplicate frame numbers. This is intentional — ffmpeg -r creates
    real duplicate frames in the output video, so each output frame needs a
    corresponding bbox entry. Matches the bash script's awk behavior.
    """
    if abs(target_fps - source_fps) < 1e-6:
        return list(range(start_frame, end_frame + 1))

    if start_time is None or end_time is None:
        # Fall back to frame-based time computation
        start_time = start_frame / source_fps
        end_time = end_frame / source_fps

    frames: List[int] = []
    step = 1.0 / target_fps
    t = start_time
    while t <= end_time + 1e-9:
        frame = round(t * source_fps)
        frame = max(start_frame, min(end_frame, frame))
        frames.append(frame)
        t += step

    return frames


def write_bbox_json(
    annotation_data: Dict[str, Any],
    casualty_id: str,
    start_frame: int,
    end_frame: int,
    keep_frames: List[int],
    source_fps: float,
    output_path: str,
) -> None:
    """Write per-frame bbox JSON for a snippet, with 1-based snippet_frame numbering."""
    if annotation_data["source_type"] == "annotation":
        results = (annotation_data.get("annotation") or {}).get("result", [])
    else:
        results = (annotation_data.get("prediction") or {}).get("result", [])

    # Collect all sequence entries for this casualty
    all_seq: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict) or item.get("type") != "videorectangle":
            continue
        if _parse_casualty_id(item) != casualty_id:
            continue
        seq = (item.get("value") or {}).get("sequence", [])
        all_seq.extend(seq)

    # Build frame -> entry lookup (filter to range)
    by_frame: Dict[int, Dict[str, Any]] = {}
    for entry in sorted(all_seq, key=lambda e: e.get("frame", 0)):
        f = entry.get("frame")
        if f is not None and start_frame <= int(f) <= end_frame:
            by_frame[int(f)] = entry

    # Build output array
    output: List[Dict[str, Any]] = []
    snippet_frame = 1
    for frame in keep_frames:
        entry = by_frame.get(frame)
        if entry is None:
            continue
        t = entry.get("time")
        if t is None:
            t = frame / source_fps
        output.append({
            "original_frame": frame,
            "snippet_frame": snippet_frame,
            "time": t,
            "x": entry.get("x"),
            "y": entry.get("y"),
            "width": entry.get("width"),
            "height": entry.get("height"),
            "rotation": entry.get("rotation"),
            "score": entry.get("score"),
            "enabled": entry.get("enabled"),
            "auto": entry.get("auto"),
        })
        snippet_frame += 1

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote bbox JSON (%d frames) to %s", len(output), output_path)


def cut_snippet_video(
    video_path: str,
    start_time: float,
    end_time: float,
    source_fps: float,
    target_fps: float,
    output_path: str,
) -> None:
    """Cut a snippet video using ffmpeg. Stream-copy when FPS matches, re-encode otherwise."""
    fps_same = abs(target_fps - source_fps) < 1e-6

    if fps_same:
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_time:.6f}", "-to", f"{end_time:.6f}",
            "-i", video_path,
            "-c", "copy",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_time:.6f}", "-to", f"{end_time:.6f}",
            "-i", video_path,
            "-r", str(target_fps),
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-c:a", "copy",
            output_path,
        ]

    logger.debug("ffmpeg: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {r.stderr[-500:]}")
    logger.info("Wrote snippet video: %s", output_path)


# ---------------------------------------------------------------------------
# Phase 5: SAM3 masks (lazy-imported, skipped with --skip-masks)
# ---------------------------------------------------------------------------

def run_sam3_pipeline(
    snippets_dir: str,
    pairs: List[Tuple[str, str]],
    *,
    skip_mask_video: bool = False,
    skip_overlay_video: bool = False,
    skip_bbox_video: bool = False,
    chunk_size: int = 1000,
    host_uid: int = 1000,
    host_gid: int = 1000,
) -> List[Dict[str, Any]]:
    """Run SAM3 mask extraction + video encoding for each (snippet, bbox_json) pair.

    Lazily imports extract_snippet_masks to avoid requiring torch when --skip-masks.
    """
    from extract_snippet_masks import (
        process_snippet,
        encode_mask_video,
        encode_overlay_video,
        encode_bbox_video,
        fix_ownership,
        _load_tracker_model,
        _get_video_fps,
        _get_video_dims,
    )
    import torch

    logger.info("Loading Sam3TrackerModel...")
    t0 = time.time()
    model, processor = _load_tracker_model()
    device = next(model.parameters()).device.type
    dtype = torch.bfloat16
    logger.info("Model loaded in %.1fs (device=%s)", time.time() - t0, device)

    results = []
    for i, (snippet, bbox_json) in enumerate(pairs, 1):
        logger.info("[SAM3 %d/%d] %s", i, len(pairs), os.path.basename(snippet))
        t1 = time.time()

        snippet_base = os.path.splitext(snippet)[0]
        mask_dir = f"{snippet_base}_masks"

        try:
            stats = process_snippet(
                video_path=snippet,
                bbox_json_path=bbox_json,
                output_dir=mask_dir,
                model=model,
                processor=processor,
                device=device,
                dtype=dtype,
            )

            fps = _get_video_fps(snippet)
            width, height = _get_video_dims(snippet)

            if not skip_mask_video:
                encode_mask_video(mask_dir, f"{snippet_base}_masks.mp4", fps)
            if not skip_overlay_video:
                encode_overlay_video(
                    snippet, mask_dir, f"{snippet_base}_overlaid_masks.mp4",
                    fps, width, height,
                )
            if not skip_bbox_video:
                encode_bbox_video(
                    snippet, bbox_json, f"{snippet_base}_bbox_overlaid.mp4",
                    chunk_size=chunk_size,
                )

            fix_ownership(mask_dir, host_uid, host_gid)

            stats["elapsed_sec"] = round(time.time() - t1, 1)
            stats["name"] = os.path.basename(snippet)
            results.append(stats)
        except Exception as exc:
            logger.error("SAM3 FAILED for %s: %s", os.path.basename(snippet), exc)
            results.append({"name": os.path.basename(snippet), "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _env_first(*names: str) -> Optional[str]:
    """Return the first non-empty env var value from the given names."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value
    return None


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified annotation-to-artifacts pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--ls-url", default=None,
        help="Label Studio URL (fallback: LABEL_STUDIO_URL / LABEL_STUDIO_HOST)",
    )
    parser.add_argument(
        "--ls-api-key", default=None,
        help="Label Studio API key (fallback: LABEL_STUDIO_API_KEY)",
    )
    parser.add_argument("--project", type=int, required=True, help="Project ID")
    parser.add_argument("--task", type=int, required=True, help="Task ID")
    parser.add_argument("--annotation", type=int, required=True, help="Annotation ID")

    # Optional
    parser.add_argument("--snippets-dir", default=None, help="Output directory")
    parser.add_argument("--person-id", default=None, help="Filter to single casualty")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--min-frames", type=int, default=None,
                       help="Min snippet length in frames")
    group.add_argument("--min-seconds", type=float, default=None,
                       help="Min snippet duration in seconds")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS")
    parser.add_argument("--skip-snippets", action="store_true",
                        help="Export + summary only, no snippet videos")
    parser.add_argument("--skip-masks", action="store_true",
                        help="No SAM3 — still cuts snippets + bbox JSON")
    parser.add_argument("--skip-mask-video", action="store_true",
                        help="Skip grayscale mask MP4")
    parser.add_argument("--skip-overlay-video", action="store_true",
                        help="Skip green overlay MP4")
    parser.add_argument("--skip-bbox-video", action="store_true",
                        help="Skip red bbox overlay MP4")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="ffmpeg drawbox chunk size (default: 1000)")
    parser.add_argument("--host-uid", type=int, default=1000,
                        help="Docker ownership fix: UID (default: 1000)")
    parser.add_argument("--host-gid", type=int, default=1000,
                        help="Docker ownership fix: GID (default: 1000)")
    parser.add_argument("--poll-interval", type=int, default=5,
                        help="Export poll interval in seconds (default: 5)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Export poll timeout in seconds (default: 300)")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Logging level",
    )

    args = parser.parse_args(argv)

    # Resolve from env vars
    args.ls_url = args.ls_url or _env_first("LABEL_STUDIO_URL", "LABEL_STUDIO_HOST")
    args.ls_api_key = args.ls_api_key or _env_first("LABEL_STUDIO_API_KEY")

    if not args.ls_url:
        parser.error("--ls-url is required (or set LABEL_STUDIO_URL / LABEL_STUDIO_HOST)")
    if not args.ls_api_key:
        parser.error("--ls-api-key is required (or set LABEL_STUDIO_API_KEY)")

    return args


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    base_url, session = _build_ls_session(args.ls_url, args.ls_api_key)
    project_id = args.project
    task_id = args.task
    annotation_id = args.annotation

    tmp_dir = tempfile.mkdtemp(prefix="process_ann_")

    try:
        # ── Phase 1: Export with interpolation ──────────────────────────
        logger.info("Phase 1: Creating export snapshot with interpolated keyframes...")
        export_id = create_export_snapshot(base_url, session, project_id)
        wait_for_export(
            base_url, session, project_id, export_id,
            poll_interval=args.poll_interval, timeout=args.timeout,
        )
        json_path = download_export(base_url, session, project_id, export_id, tmp_dir)
        logger.info("Export downloaded to %s", json_path)

        with open(json_path) as f:
            export_data = json.load(f)

        # ── Phase 2: Extract annotation + summary ──────────────────────
        logger.info("Phase 2: Extracting annotation and building summary...")
        annotation_data = extract_annotation(
            export_data, project_id, task_id, annotation_id,
        )

        # Determine output directory
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if args.snippets_dir:
            snippets_dir = args.snippets_dir
        else:
            snippets_dir = os.path.join(
                os.getcwd(),
                f"snippets_proj{project_id}_task{task_id}_ann{annotation_id}_{ts}",
            )
        os.makedirs(snippets_dir, exist_ok=True)

        # Save annotation JSON
        ann_json_path = os.path.join(
            snippets_dir,
            f"project{project_id}_task{task_id}_ann{annotation_id}.json",
        )
        with open(ann_json_path, "w") as f:
            json.dump(annotation_data, f, indent=2)
        logger.info("Saved annotation to %s", ann_json_path)

        # Generate and save summary
        summary = generate_summary(annotation_data)
        summary_path = os.path.join(
            snippets_dir,
            f"project{project_id}_task{task_id}_ann{annotation_id}.summary.json",
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary to %s", summary_path)

        if args.skip_snippets:
            logger.info("--skip-snippets: stopping after export + summary")
            _print_casualty_stats(summary)
            return

        # Validate video_url and fps
        video_url = summary.get("video_url")
        source_fps = summary.get("fps")
        if not video_url:
            raise RuntimeError("Missing video_url in annotation data")
        if not source_fps:
            raise RuntimeError("Missing fps in annotation data")
        source_fps = float(source_fps)
        target_fps = args.fps if args.fps else source_fps

        # ── Phase 3: Download source video ─────────────────────────────
        logger.info("Phase 3: Downloading source video...")
        video_ext = os.path.splitext(video_url.split("?")[0])[-1] or ".mp4"
        video_path = os.path.join(tmp_dir, f"source_video{video_ext}")
        download_video(base_url, session, video_url, video_path)

        # ── Phase 4: Cut snippets ──────────────────────────────────────
        logger.info("Phase 4: Cutting snippets...")
        snippet_pairs = _cut_all_snippets(
            annotation_data=annotation_data,
            summary=summary,
            video_path=video_path,
            snippets_dir=snippets_dir,
            source_fps=source_fps,
            target_fps=target_fps,
            person_id=args.person_id,
            min_frames=args.min_frames,
            min_seconds=args.min_seconds,
        )

        if not snippet_pairs:
            logger.info("No snippet ranges matched the filters.")
        else:
            logger.info("Cut %d snippet(s)", len(snippet_pairs))

        # ── Phase 5: SAM3 masks ────────────────────────────────────────
        sam3_results = []
        if not args.skip_masks and snippet_pairs:
            logger.info("Phase 5: Running SAM3 mask extraction...")
            sam3_results = run_sam3_pipeline(
                snippets_dir,
                snippet_pairs,
                skip_mask_video=args.skip_mask_video,
                skip_overlay_video=args.skip_overlay_video,
                skip_bbox_video=args.skip_bbox_video,
                chunk_size=args.chunk_size,
                host_uid=args.host_uid,
                host_gid=args.host_gid,
            )
        elif args.skip_masks:
            logger.info("Phase 5: Skipped (--skip-masks)")

        # ── Phase 6: Cleanup + summary ─────────────────────────────────
        _write_readme(
            snippets_dir, project_id, task_id, annotation_id,
            video_url, target_fps, args.person_id,
            args.min_frames, args.min_seconds,
            len(snippet_pairs), args.skip_masks,
        )
        _print_casualty_stats(summary, sam3_results)
        logger.info("All artifacts saved to %s", snippets_dir)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _cut_all_snippets(
    *,
    annotation_data: Dict[str, Any],
    summary: Dict[str, Any],
    video_path: str,
    snippets_dir: str,
    source_fps: float,
    target_fps: float,
    person_id: Optional[str],
    min_frames: Optional[int],
    min_seconds: Optional[float],
) -> List[Tuple[str, str]]:
    """Cut snippet videos and write bbox JSONs. Returns list of (video, json) pairs."""
    casualties = summary.get("casualties", {})
    fps_name = f"{target_fps:.0f}"
    pairs: List[Tuple[str, str]] = []

    for cid in sorted(casualties.keys(), key=lambda x: int(x)):
        if person_id is not None and cid != person_id:
            continue

        for rng in casualties[cid].get("ranges", []):
            sf = rng["start_frame"]
            ef = rng["end_frame"]
            st = rng.get("start_time")
            et = rng.get("end_time")

            # Compute start/end time from frames if missing (needed for filters and ffmpeg)
            if st is None:
                st = sf / source_fps
            if et is None:
                et = ef / source_fps

            # Apply filters
            frame_count = ef - sf + 1
            if min_frames is not None and frame_count < min_frames:
                continue
            if min_seconds is not None and (et - st) < min_seconds:
                continue

            base = f"casualty_{cid}_f{sf}-{ef}_fps{fps_name}"
            snippet_subdir = os.path.join(
                snippets_dir, f"casualty_{cid}", f"casualty_{cid}_f{sf}-{ef}",
            )
            os.makedirs(snippet_subdir, exist_ok=True)
            video_out = os.path.join(snippet_subdir, f"{base}.mp4")
            json_out = os.path.join(snippet_subdir, f"{base}_frame_bbox.json")

            keep_frames = compute_keep_frames(
                sf, ef, st, et, source_fps, target_fps,
            )

            write_bbox_json(
                annotation_data, cid, sf, ef,
                keep_frames, source_fps, json_out,
            )

            # ffmpeg -to must be strictly > -ss. A frame at time T occupies
            # [T, T+1/fps), so extend end_time past the last frame.
            ffmpeg_et = max(et + 1.0 / source_fps, st + 1.0 / source_fps)
            cut_snippet_video(
                video_path, st, ffmpeg_et, source_fps, target_fps, video_out,
            )

            pairs.append((video_out, json_out))

    return pairs


def _write_readme(
    snippets_dir: str,
    project_id: int,
    task_id: int,
    annotation_id: int,
    video_url: str,
    target_fps: float,
    person_id: Optional[str],
    min_frames: Optional[int],
    min_seconds: Optional[float],
    total_snippets: int,
    skip_masks: bool,
) -> None:
    """Write a README.txt in the snippets directory."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    readme = os.path.join(snippets_dir, "README.txt")
    with open(readme, "w") as f:
        f.write(f"Casualty snippet export\n\n")
        f.write(f"Generated at: {ts}\n")
        f.write(f"Generated by: process_annotation.py\n")
        f.write(f"Project ID: {project_id}\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Annotation ID: {annotation_id}\n")
        f.write(f"Video URL: {video_url}\n")
        f.write(f"Person ID filter: {person_id or '<all>'}\n")
        f.write(f"Min frames: {min_frames or '<none>'}\n")
        f.write(f"Min seconds: {min_seconds or '<none>'}\n")
        f.write(f"Target FPS: {target_fps}\n")
        f.write(f"Total snippets: {total_snippets}\n")
        f.write(f"SAM3 masks: {'skipped' if skip_masks else 'enabled'}\n")


def _print_casualty_stats(
    summary: Dict[str, Any],
    sam3_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print per-casualty statistics table."""
    casualties = summary.get("casualties", {})
    if not casualties:
        logger.info("No casualties found in annotation.")
        return

    logger.info("")
    logger.info("%-12s  %-8s  %-20s  %s", "Casualty", "Ranges", "Total Frames", "Frame Spans")
    logger.info("-" * 70)
    for cid in sorted(casualties.keys(), key=lambda x: int(x)):
        ranges = casualties[cid].get("ranges", [])
        total_frames = sum(r["end_frame"] - r["start_frame"] + 1 for r in ranges)
        spans = ", ".join(f"{r['start_frame']}-{r['end_frame']}" for r in ranges)
        logger.info("%-12s  %-8d  %-20d  %s", cid, len(ranges), total_frames, spans)

    if sam3_results:
        logger.info("")
        logger.info("SAM3 Results:")
        for r in sam3_results:
            if "error" in r:
                logger.info("  FAIL  %s: %s", r["name"], r["error"])
            else:
                logger.info(
                    "  OK    %s: %d frames, avg_score=%.3f, %.1fs",
                    r["name"], r.get("total_frames", 0),
                    r.get("avg_score", 0), r.get("elapsed_sec", 0),
                )


if __name__ == "__main__":
    main()
