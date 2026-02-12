"""
Shared helpers for SAM3 video seeding CLI tools.

Provides:
- Label Studio I/O (client, fetch, upload, video path resolution)
- SAM3 image embeddings for keyframe detection (replaces SAM2)
- Text-based object detection via Sam3VideoModel (replaces Grounding DINO)
- Coordinate conversion utilities
- Change-detection math (cosine distance, smoothing, keyframe selection)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
import requests
import torch
from joblib import Memory
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.client import LabelStudio
from PIL import Image

logger = logging.getLogger(__name__)


class InitialSeedingError(Exception):
    pass


@dataclass
class KeyframeDetection:
    frame_idx: int
    xyxy: np.ndarray
    score: float
    label: str
    track_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/sam3')
DTYPE = torch.bfloat16 if DEVICE == 'cuda' else torch.float32


# ---------------------------------------------------------------------------
# Lazy-loaded SAM3 model singletons
# ---------------------------------------------------------------------------

_sam3_image_model = None
_sam3_image_processor = None
_sam3_video_model = None
_sam3_video_processor = None
_sam3_tracker_model = None
_sam3_tracker_processor = None


def _get_sam3_image_model():
    """Lazy-load Sam3Model for image embeddings (keyframe detection, teacher confirmation)."""
    global _sam3_image_model, _sam3_image_processor
    if _sam3_image_model is None:
        from transformers import Sam3Model, Sam3Processor
        logger.info("Loading Sam3Model (image) from %s ...", MODEL_NAME)
        _sam3_image_model = Sam3Model.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
        _sam3_image_processor = Sam3Processor.from_pretrained(MODEL_NAME)
        logger.info("Sam3Model loaded on %s", DEVICE)
    return _sam3_image_model, _sam3_image_processor


def _get_sam3_video_model():
    """Lazy-load Sam3VideoModel for text-based detection (replaces Grounding DINO)."""
    global _sam3_video_model, _sam3_video_processor
    if _sam3_video_model is None:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        logger.info("Loading Sam3VideoModel (PCS) from %s ...", MODEL_NAME)
        _sam3_video_model = Sam3VideoModel.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
        _sam3_video_processor = Sam3VideoProcessor.from_pretrained(MODEL_NAME)
        logger.info("Sam3VideoModel loaded on %s", DEVICE)
    return _sam3_video_model, _sam3_video_processor


def _get_sam3_tracker_model():
    """Lazy-load Sam3TrackerVideoModel for box-prompted tracking."""
    global _sam3_tracker_model, _sam3_tracker_processor
    if _sam3_tracker_model is None:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        logger.info("Loading Sam3TrackerVideoModel from %s ...", MODEL_NAME)
        _sam3_tracker_model = Sam3TrackerVideoModel.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
        _sam3_tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(MODEL_NAME)
        logger.info("Sam3TrackerVideoModel loaded on %s", DEVICE)
    return _sam3_tracker_model, _sam3_tracker_processor


# ---------------------------------------------------------------------------
# Text+Box Refinement (Hybrid approach for imperfect boxes)
# ---------------------------------------------------------------------------

def refine_box_with_text_prompt(
    image: Image.Image,
    box_xyxy: np.ndarray,
    text_label: str,
    search_scale: float = 1.3,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Refine a bounding box using Sam3Model with combined text+box prompts.

    This hybrid approach handles boxes that are too large OR too small:
    - The expanded box (search region) ensures the target is likely contained
    - The text prompt tells SAM3 WHAT to segment (e.g., "person")

    Args:
        image: PIL Image containing the frame
        box_xyxy: Original bounding box in xyxy pixel coordinates
        text_label: Text prompt describing the object (e.g., "person", "Player")
        search_scale: Factor to expand the search region (default 1.3 = 30% larger)
        threshold: Confidence threshold for accepting refinement (default 0.5)

    Returns:
        Tuple of (refined_box_xyxy, confidence_score)
        If refinement fails, returns (original_box, 0.0)
    """
    sam3_model, sam3_processor = _get_sam3_image_model()

    w, h = image.size
    x0, y0, x1, y1 = box_xyxy

    # Compute expanded search region centered on original box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = x1 - x0
    bh = y1 - y0

    half_w = 0.5 * search_scale * bw
    half_h = 0.5 * search_scale * bh

    # Clamp to image bounds
    x0_search = max(0.0, cx - half_w)
    y0_search = max(0.0, cy - half_h)
    x1_search = min(float(w), cx + half_w)
    y1_search = min(float(h), cy + half_h)

    search_box = [int(round(x0_search)), int(round(y0_search)),
                  int(round(x1_search)), int(round(y1_search))]

    if search_box[2] <= search_box[0] or search_box[3] <= search_box[1]:
        logger.debug("Search box invalid after clamping, returning original")
        return box_xyxy.copy(), 0.0

    try:
        # Run SAM3 with combined text + positive box prompt
        inputs = sam3_processor(
            images=image,
            text=text_label,
            input_boxes=[[search_box]],
            input_boxes_labels=[[1]],  # 1 = positive box
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
            outputs = sam3_model(**inputs)

        # Post-process using instance segmentation
        results = sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])
        boxes = results.get("boxes", [])

        if len(masks) == 0:
            logger.debug("No masks returned for text='%s', returning original box", text_label)
            return box_xyxy.copy(), 0.0

        # Select best mask by confidence score
        if len(scores) > 0:
            score_vals = [s.item() if hasattr(s, 'item') else float(s) for s in scores]
            best_idx = int(np.argmax(score_vals))
            best_score = score_vals[best_idx]
        else:
            best_idx = 0
            best_score = 0.5

        # Extract refined box
        if len(boxes) > best_idx:
            best_box = boxes[best_idx]
            if hasattr(best_box, 'tolist'):
                best_box = best_box.tolist()
            elif hasattr(best_box, 'cpu'):
                best_box = best_box.cpu().numpy().tolist()
            refined_xyxy = np.array(best_box, dtype=np.float32)
        else:
            # Fall back to extracting bbox from mask
            mask = masks[best_idx]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()

            ys, xs = np.where(mask > 0)
            if xs.size == 0 or ys.size == 0:
                return box_xyxy.copy(), 0.0

            refined_xyxy = np.array([
                int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1
            ], dtype=np.float32)

        # Clamp to image bounds
        refined_xyxy[0] = max(0, min(w - 1, refined_xyxy[0]))
        refined_xyxy[1] = max(0, min(h - 1, refined_xyxy[1]))
        refined_xyxy[2] = max(0, min(w, refined_xyxy[2]))
        refined_xyxy[3] = max(0, min(h, refined_xyxy[3]))

        if refined_xyxy[2] <= refined_xyxy[0] or refined_xyxy[3] <= refined_xyxy[1]:
            return box_xyxy.copy(), 0.0

        logger.debug(
            "Refined box: [%.1f,%.1f,%.1f,%.1f] -> [%.1f,%.1f,%.1f,%.1f] (score=%.3f)",
            x0, y0, x1, y1,
            refined_xyxy[0], refined_xyxy[1], refined_xyxy[2], refined_xyxy[3],
            best_score,
        )
        return refined_xyxy, best_score

    except Exception as exc:
        logger.warning("Text+box refinement failed: %s", exc)
        return box_xyxy.copy(), 0.0


def refine_boxes_batch(
    image: Image.Image,
    boxes_xyxy: List[np.ndarray],
    text_labels: List[str],
    search_scale: float = 1.3,
    threshold: float = 0.5,
) -> List[Tuple[np.ndarray, float]]:
    """Refine multiple boxes on the same frame using text+box prompts.

    Processes boxes one at a time (batching with different text prompts per box
    is complex with the current API).

    Args:
        image: PIL Image containing the frame
        boxes_xyxy: List of bounding boxes in xyxy pixel coordinates
        text_labels: List of text prompts (one per box)
        search_scale: Factor to expand search regions
        threshold: Confidence threshold

    Returns:
        List of (refined_box_xyxy, confidence_score) tuples
    """
    results = []
    for box, label in zip(boxes_xyxy, text_labels):
        refined, score = refine_box_with_text_prompt(
            image, box, label, search_scale, threshold
        )
        results.append((refined, score))
    return results


# ---------------------------------------------------------------------------
# LS I/O helpers (preserved from SAM2 seeding_common)
# ---------------------------------------------------------------------------

def _ensure_meta_text_placeholder(result: Dict[str, Any]) -> None:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        result["meta"] = meta
    raw_text = meta.get("text")
    texts: List[str] = []
    if isinstance(raw_text, str):
        texts = [raw_text]
    elif isinstance(raw_text, list):
        texts = [t for t in raw_text if isinstance(t, str)]
    if not texts or all(not t.strip() for t in texts):
        meta["text"] = "id:"


def _build_ls_client(ls_url: str, ls_api_key: str):
    if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
        raise InitialSeedingError(
            "LABEL_STUDIO_API_KEY is required. "
            "Provide it via --ls-api-key or the LABEL_STUDIO_API_KEY env var."
        )

    os.environ.setdefault("LABEL_STUDIO_URL", ls_url)
    os.environ.setdefault("LABEL_STUDIO_API_KEY", ls_api_key)

    logger.info("Connecting to Label Studio at %s", ls_url)
    client = LabelStudio(base_url=ls_url, api_key=ls_api_key, timeout=600)
    logger.info("Connected to Label Studio")
    return client


def _fetch_task(ls, project_id: int, task_id: int) -> Dict[str, Any]:
    logger.info("Fetching task %s from project %s", task_id, project_id)
    task_obj = ls.tasks.get(task_id)
    if not task_obj:
        raise InitialSeedingError(f"Task {task_id} not found")

    task_project = getattr(task_obj, "project", None)
    if task_project is not None and task_project != project_id:
        logger.warning(
            "Task %s belongs to project %s (not %s)",
            getattr(task_obj, "id", task_id),
            task_project,
            project_id,
        )

    task = {"id": task_obj.id, "data": getattr(task_obj, "data", {})}
    logger.info("Task fetched: %s", task.get("id"))
    return task


def _fetch_annotation(ls, annotation_id: int) -> Any:
    logger.info("Fetching annotation %s", annotation_id)
    ann = ls.annotations.get(id=annotation_id)
    if not ann:
        raise InitialSeedingError(f"Annotation {annotation_id} not found")

    result = getattr(ann, "result", None)
    if result is None:
        raise InitialSeedingError(f"Annotation {annotation_id} has no regions")

    logger.info(
        "Annotation fetched: id=%s with %d regions", getattr(ann, "id", annotation_id), len(result or [])
    )
    return ann


def _detect_video_key(task_data: Dict[str, Any]) -> Tuple[str, str]:
    preferred_keys = ["video", "video_url", "video_path"]
    for key in preferred_keys:
        if key in task_data and isinstance(task_data[key], str):
            return key, task_data[key]

    for key, value in task_data.items():
        if not isinstance(value, str):
            continue
        lower = value.lower()
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            return key, value

    raise InitialSeedingError(
        "Could not detect video field in task data. "
        "Ensure your task has a field like 'video' with a video URL/path."
    )


def _manual_download_video(url: str, dest_path: str) -> None:
    """Manually download video with Authorization header if needed."""
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"

    logger.info("Starting manual download from %s to %s", url, dest_path)
    try:
        with requests.get(url, headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Manual download completed")
    except Exception as e:
        logger.error("Manual download failed: %s", e)
        # Clean up partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise


def _get_video_path(task: Dict[str, Any]) -> Tuple[str, str]:
    data = task.get("data") or {}
    key, video_url = _detect_video_key(data)
    logger.info("Using video field '%s' with URL %s", key, video_url)

    if not video_url.startswith("http") and video_url.startswith("/"):
        host = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL")
        if host:
            from urllib.parse import urljoin

            video_url = urljoin(host.rstrip("/"), video_url)
            logger.info("Resolved relative video URL to %s", video_url)

    logger.info("Downloading/caching video via get_local_path...")
    local_path = get_local_path(video_url, task_id=task["id"])

    # Check for empty or missing file
    if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
        logger.warning("Cached video file is empty (0 bytes). Removing and attempting manual download...")
        try:
            os.remove(local_path)
        except OSError:
            pass

        try:
            _manual_download_video(video_url, local_path)
        except Exception:
            pass

    if not os.path.exists(local_path):
        raise InitialSeedingError(f"Video file not found after download: {local_path}")

    size_mb = os.path.getsize(local_path) / 1024**2
    logger.info("Video cached at: %s (%.2f MB)", local_path, size_mb)

    if size_mb == 0:
        raise InitialSeedingError(f"Video file is empty (0 bytes) after download attempts: {local_path}")

    return local_path, key


# ---------------------------------------------------------------------------
# PyAV video utilities (replaces cv2.VideoCapture)
# ---------------------------------------------------------------------------

def _get_video_info_pyav(video_path: str) -> Tuple[int, int, int, float]:
    """Get (width, height, frames_count, fps) via PyAV."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frames_count = stream.frames
        if not frames_count and stream.duration and stream.time_base:
            frames_count = int(float(stream.duration * stream.time_base) * fps)
        return width, height, max(frames_count, 0), fps
    finally:
        container.close()


def _read_frame_pyav(video_path: str, frame_idx: int) -> Optional[Image.Image]:
    """Read a single frame by index via PyAV, return as PIL RGB Image.

    Uses seek + PTS comparison for O(GOP) decode instead of linear scan.
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        tb = float(stream.time_base) if stream.time_base else None

        if frame_idx > 0 and tb:
            target_ts = int(frame_idx / fps / tb)
            container.seek(target_ts, stream=stream)

        # After seek, compare decoded frame PTS against target PTS.
        # This avoids the old bug where current_idx started at 0 after seek,
        # causing ~frame_idx extra decodes.
        target_pts = (frame_idx / fps / tb) if tb else None
        decoded = 0
        for frame in container.decode(video=0):
            decoded += 1
            if target_pts is not None:
                if frame.pts is not None and frame.pts >= target_pts:
                    return frame.to_image()
            else:
                # No time_base — fall back to counting from 0 (no seek)
                if decoded - 1 >= frame_idx:
                    return frame.to_image()
            # Safety: don't decode more than 500 frames past seek
            if decoded > 500:
                logger.warning("_read_frame_pyav: exceeded 500 frames for idx %d", frame_idx)
                break

        return None
    finally:
        container.close()


def _decode_frames_pyav(
    video_path: str,
    start_frame: int,
    end_frame: int,
    stride: int = 1,
) -> List[Tuple[int, Image.Image]]:
    """Decode a range of frames to [(frame_idx, PIL.Image)] via PyAV."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]

        if start_frame > 0 and stream.average_rate and stream.time_base:
            avg_fps = float(stream.average_rate)
            target_ts = int(start_frame / avg_fps / stream.time_base)
            container.seek(target_ts, stream=stream)

        results = []
        frame_idx = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_idx < start_frame:
                    frame_idx += 1
                    continue
                if frame_idx >= end_frame:
                    return results
                if stride > 1 and (frame_idx - start_frame) % stride != 0:
                    frame_idx += 1
                    continue
                results.append((frame_idx, frame.to_image()))
                frame_idx += 1

        return results
    finally:
        container.close()


# ---------------------------------------------------------------------------
# SAM3 image embeddings (replaces SAM2 embedding pipeline)
# ---------------------------------------------------------------------------

def _global_pool_embed(embed: torch.Tensor) -> torch.Tensor:
    if embed.ndim == 4:
        return embed.mean(dim=[2, 3])
    if embed.ndim == 3:
        return embed.mean(dim=[1, 2])
    return embed


def _extract_sam3_image_embedding(
    sam3_model,
    sam3_processor,
    pil_image: Image.Image,
) -> torch.Tensor:
    """Extract image embedding from Sam3Model's vision encoder.

    Sam3Model exposes ``get_vision_features(pixel_values=...)`` which returns
    a ``Sam3VisionEncoderOutput`` with FPN feature maps.  We take the
    highest-resolution FPN map and global-average-pool it to a flat vector.
    """
    inputs = sam3_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        vision_output = sam3_model.get_vision_features(
            pixel_values=inputs.pixel_values
        )
    # fpn_hidden_states[0] is highest-resolution FPN feature: (B, C, H, W)
    feat = vision_output.fpn_hidden_states[0]
    return _global_pool_embed(feat)


def _embed_batch_sam3(
    sam3_model,
    sam3_processor,
    frames: List[Image.Image],
) -> np.ndarray:
    """Embed a batch of PIL frames using Sam3Model with true GPU batching.

    Preprocesses all frames in `frames` into a single tensor and runs one
    forward pass through the vision encoder, then global-average-pools each
    FPN feature map to produce (B, C) embeddings.

    Falls back to half-batch on OOM, recursively, until batch_size=1 at
    which point the single-image path is used.
    """
    if len(frames) == 0:
        return np.zeros((0, 1), dtype=np.float32)

    if len(frames) == 1:
        embed = _extract_sam3_image_embedding(sam3_model, sam3_processor, frames[0])
        return embed.detach().cpu().float().numpy()

    try:
        inputs = sam3_processor(images=frames, return_tensors="pt").to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
            vision_output = sam3_model.get_vision_features(
                pixel_values=inputs.pixel_values
            )
        feat = vision_output.fpn_hidden_states[0]  # (B, C, H, W)
        pooled = _global_pool_embed(feat)           # (B, C)
        return pooled.detach().cpu().float().numpy()
    except torch.cuda.OutOfMemoryError:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        half = len(frames) // 2
        logger.warning(
            "OOM with batch_size=%d, retrying with two halves of %d and %d",
            len(frames), half, len(frames) - half,
        )
        a = _embed_batch_sam3(sam3_model, sam3_processor, frames[:half])
        b = _embed_batch_sam3(sam3_model, sam3_processor, frames[half:])
        return np.concatenate([a, b], axis=0)


class _SuppressBelowWarning(logging.Filter):
    """Filter that suppresses log records below WARNING.

    Added temporarily to handlers during bulk embedding to reduce noise
    without mutating global logger levels (which is not thread-safe).
    """
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


def _do_embed_all_frames(
    video_path: str,
    batch_size: int,
    progress_callback: Optional[Any] = None,
    target_fps: Optional[float] = None,
    pause_event: Optional[Any] = None,
    change_callback: Optional[Any] = None,
) -> "tuple[np.ndarray, list[int]]":
    """Embed video frames using GPU-batched SAM3 vision encoder.

    Uses a prefetch thread to decode frames via PyAV concurrently with GPU
    inference, keeping the GPU fed and avoiding decode-wait bottlenecks.

    The decode thread owns the container lifecycle — it opens, decodes, and
    closes the container.  The main thread never touches the container,
    eliminating the race condition where ``container.close()`` could be
    called while the decode thread is still reading.

    Args:
        video_path:        Path to the video file.
        batch_size:        Number of frames per GPU forward pass.
        progress_callback: Optional callable(current: int, total: int) for
                           progress reporting (e.g. from Interview UI).
        target_fps:        If set, subsample to at most this many frames per
                           second.  E.g. target_fps=10 on a 30fps video
                           processes every 3rd frame.
        pause_event:       Optional threading.Event — if provided, the main
                           loop calls ``pause_event.wait()`` between batches,
                           blocking while paused.
        change_callback:   Optional callable(change_keyframes: List[int]) —
                           called incrementally after each batch with the
                           change-detected keyframe indices computed so far.

    Returns:
        Tuple of:
        - (N, C) float16 numpy array of per-frame embeddings.
        - List of original 0-based frame indices that were sampled
          (length N; identity map [0..total-1] when target_fps is None).
    """
    import queue
    import threading

    sam3_model, sam3_processor = _get_sam3_image_model()

    # Probe total_frames and fps from a quick container open (closed immediately).
    probe = av.open(video_path)
    probe_stream = probe.streams.video[0]
    total_frames = probe_stream.frames
    video_fps = float(probe_stream.average_rate) if probe_stream.average_rate else 30.0
    if not total_frames and probe_stream.duration and probe_stream.time_base:
        total_frames = int(float(probe_stream.duration * probe_stream.time_base) * video_fps)
    probe.close()

    # Compute skip interval for FPS subsampling
    skip = 1
    if target_fps is not None and target_fps > 0 and video_fps > target_fps:
        skip = max(1, round(video_fps / target_fps))

    # Estimate sampled count for progress reporting
    sampled_total = (total_frames + skip - 1) // skip if total_frames else 0

    # -- Prefetch thread: owns its own container, decodes into a bounded queue --
    # Queue items are (original_frame_idx, PIL Image) or _SENTINEL
    frame_queue: queue.Queue = queue.Queue(maxsize=batch_size * 2)
    _SENTINEL = None  # signals end-of-video

    def _decode_worker():
        container = av.open(video_path)
        try:
            frame_idx = 0
            for av_frame in container.decode(video=0):
                if frame_idx % skip == 0:
                    frame_queue.put((frame_idx, av_frame.to_image()))
                frame_idx += 1
        except Exception as exc:
            logger.error("Prefetch decode error: %s", exc)
        finally:
            container.close()
            frame_queue.put(_SENTINEL)

    decode_thread = threading.Thread(target=_decode_worker, daemon=True, name="embed-prefetch")
    decode_thread.start()

    # Suppress noisy logging during bulk embedding using a filter (thread-safe)
    _suppress_filter = _SuppressBelowWarning()
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        h.addFilter(_suppress_filter)

    try:
        embeds: List[np.ndarray] = []
        sampled_indices: List[int] = []
        frames_batch: List[Image.Image] = []
        indices_batch: List[int] = []
        frames_done = 0

        while True:
            item = frame_queue.get()
            if item is _SENTINEL:
                break
            orig_idx, pil_img = item
            frames_batch.append(pil_img)
            indices_batch.append(orig_idx)

            if len(frames_batch) >= batch_size:
                # Pause check — blocks if embedding is paused
                if pause_event is not None:
                    pause_event.wait()

                embeds.append(_embed_batch_sam3(sam3_model, sam3_processor, frames_batch))
                sampled_indices.extend(indices_batch)
                frames_done += len(frames_batch)
                frames_batch = []
                indices_batch = []
                if progress_callback is not None:
                    progress_callback(frames_done, sampled_total or frames_done)

                # Incremental change detection callback (every 10 batches to avoid O(N^2))
                if change_callback is not None and len(embeds) >= 2 and len(embeds) % 10 == 0:
                    _invoke_change_callback(embeds, sampled_indices, change_callback)

        if frames_batch:
            if pause_event is not None:
                pause_event.wait()
            embeds.append(_embed_batch_sam3(sam3_model, sam3_processor, frames_batch))
            sampled_indices.extend(indices_batch)
            frames_done += len(frames_batch)
            if progress_callback is not None:
                progress_callback(frames_done, sampled_total or frames_done)
            if change_callback is not None and len(embeds) >= 2:
                _invoke_change_callback(embeds, sampled_indices, change_callback)

        if not embeds:
            raise InitialSeedingError("No frames read from video for embedding computation")

        stacked = np.concatenate(embeds, axis=0).astype("float16")
        logger.info(
            "Computed SAM3 embeddings for %d frames (skip=%d, shape=%s)",
            stacked.shape[0], skip, stacked.shape,
        )
        return stacked, sampled_indices
    finally:
        decode_thread.join(timeout=30)
        if decode_thread.is_alive():
            logger.warning("Prefetch decode thread did not finish in 30s")
        for h in root_logger.handlers:
            h.removeFilter(_suppress_filter)


def _invoke_change_callback(
    embeds: List[np.ndarray],
    sampled_indices: List[int],
    change_callback: Any,
) -> None:
    """Compute change scores on accumulated embeddings and invoke callback."""
    try:
        stacked = np.concatenate(embeds, axis=0).astype("float16")
        if stacked.shape[0] < 2:
            return
        diff = compute_change_scores(stacked)
        smooth = smooth_change_scores(diff, kernel_size=5)
        keyframes_sub = select_keyframes(
            stacked.shape[0], 0.04, smooth, min_spacing=30,
        )
        # Map subsampled indices back to original frame indices
        change_originals = [sampled_indices[k] for k in keyframes_sub if k < len(sampled_indices)]
        change_callback(change_originals)
    except Exception as exc:
        logger.warning("Incremental change callback error: %s", exc)


def _compute_sam3_frame_embeddings(
    video_id: str,
    video_path: str,
    batch_size: int,
    cache_dir: str,
    progress_callback: Optional[Any] = None,
) -> np.ndarray:
    """Compute per-frame SAM3 image embeddings with joblib caching.

    Delegates to :func:`_do_embed_all_frames` which uses a prefetch thread
    for PyAV decoding and GPU-batched inference.

    Args:
        video_id:          Stable cache key for this video.
        video_path:        Filesystem path to video.
        batch_size:        Frames per GPU batch.
        cache_dir:         Joblib cache directory.
        progress_callback: Optional callable(current, total) for UI progress.
    """
    memory = Memory(cache_dir, verbose=0)

    @memory.cache(ignore=["video_path_arg", "batch_size_arg", "progress_cb"])
    def _cached_compute(
        video_id_key: str,
        video_path_arg: str,
        batch_size_arg: int,
        progress_cb: Optional[Any] = None,
    ) -> np.ndarray:
        result = _do_embed_all_frames(video_path_arg, batch_size_arg, progress_cb)
        # _do_embed_all_frames now returns (embeds, sampled_indices) tuple.
        # Legacy callers through this cache function only need the embeddings.
        if isinstance(result, tuple):
            return result[0]
        return result

    return _cached_compute(video_id, video_path, batch_size, progress_callback)


# ---------------------------------------------------------------------------
# Change detection & keyframe selection (preserved, pure numpy)
# ---------------------------------------------------------------------------

def compute_change_scores(embeds: np.ndarray) -> np.ndarray:
    if embeds.ndim != 2:
        raise InitialSeedingError(f"Expected embeddings with shape [T, D], got {embeds.shape}")
    norm = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
    norm_embeds = embeds / norm
    T_len = norm_embeds.shape[0]
    diff = np.zeros(T_len, dtype=np.float32)
    diff[1:] = np.linalg.norm(norm_embeds[1:] - norm_embeds[:-1], axis=1)
    return diff


# ---------------------------------------------------------------------------
# Lightweight change detection (CPU-only, no SAM3)
# ---------------------------------------------------------------------------

def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to HSV float array.

    Input: (H, W, 3) uint8. Output: (H, W, 3) float32
    with H in [0, 360), S in [0, 1], V in [0, 1].
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    with np.errstate(invalid="ignore"):
        s = np.where(cmax > 0, delta / cmax, 0.0)

    return np.stack([h, s, cmax], axis=-1)


def _histogram_distance(frame_a: np.ndarray, frame_b: np.ndarray,
                         bins: int = 8) -> float:
    """Chi-squared distance between HSV histograms of two frames.

    Both inputs should be RGB uint8 arrays (H, W, 3). Returns a scalar
    distance in [0, +inf), where 0 means identical histograms.
    """
    hsv_a = _rgb_to_hsv_numpy(frame_a)
    hsv_b = _rgb_to_hsv_numpy(frame_b)

    ranges = [(0, 360), (0, 1), (0, 1)]
    hist_a, _ = np.histogramdd(
        hsv_a.reshape(-1, 3),
        bins=[bins, bins, bins],
        range=ranges,
    )
    hist_b, _ = np.histogramdd(
        hsv_b.reshape(-1, 3),
        bins=[bins, bins, bins],
        range=ranges,
    )

    hist_a = hist_a / (hist_a.sum() + 1e-8)
    hist_b = hist_b / (hist_b.sum() + 1e-8)

    denom = hist_a + hist_b + 1e-8
    return float(0.5 * np.sum((hist_a - hist_b) ** 2 / denom))


_CHANGE_THUMB_SIZE = (128, 128)


def compute_lightweight_change_scores_from_frames(
    frames: List[Image.Image],
) -> np.ndarray:
    """Compute per-frame change scores using pixel diff + histogram distance.

    Two complementary signals combined via max():
      - Pixel L1: mean absolute difference on 128x128 thumbnails
      - Histogram chi-sq: HSV histogram distance (lighting, color shifts)

    Both signals are normalized to [0, 1] using their respective max values,
    then combined: score[i] = max(norm_pixel[i], norm_hist[i]).

    Returns:
        (N,) float32 array of change scores. First element is always 0.
    """
    n = len(frames)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    pixel_scores = np.zeros(n, dtype=np.float32)
    hist_scores = np.zeros(n, dtype=np.float32)

    prev_thumb = np.array(frames[0].resize(_CHANGE_THUMB_SIZE, Image.BILINEAR))

    for i in range(1, n):
        thumb = np.array(frames[i].resize(_CHANGE_THUMB_SIZE, Image.BILINEAR))

        pixel_scores[i] = np.mean(np.abs(
            thumb.astype(np.float32) - prev_thumb.astype(np.float32)
        )) / 255.0

        hist_scores[i] = _histogram_distance(prev_thumb, thumb)

        prev_thumb = thumb

    px_max = pixel_scores.max()
    if px_max > 0:
        pixel_scores /= px_max

    hs_max = hist_scores.max()
    if hs_max > 0:
        hist_scores /= hs_max

    return np.maximum(pixel_scores, hist_scores)


def _decode_frames_for_change(
    video_path: str,
    target_fps: Optional[float] = None,
    pause_event: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
) -> Tuple[List[Image.Image], List[int]]:
    """Decode video frames (subsampled to target_fps) for change detection.

    Returns (frames_list, sampled_indices) where sampled_indices maps
    position in frames_list to the original 0-based frame index.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate or 30)
    total_frames = stream.frames or 0

    skip = max(1, int(round(src_fps / target_fps))) if target_fps else 1

    frames: List[Image.Image] = []
    indices: List[int] = []
    frame_idx = 0

    for av_frame in container.decode(video=0):
        if frame_idx % skip == 0:
            frames.append(av_frame.to_image())
            indices.append(frame_idx)

            if progress_callback and len(frames) % 500 == 0:
                est_total = total_frames // skip if total_frames else 0
                progress_callback(len(frames), est_total)

            if pause_event is not None:
                pause_event.wait()

        frame_idx += 1

    container.close()
    return frames, indices


def _compute_change_score_moving_avg(
    thumb: np.ndarray,
    thumb_window: "collections.deque",
) -> Tuple[float, float]:
    """Compute pixel L1 and histogram distance against a moving average baseline.

    Compares the current thumbnail against the mean of the thumbnails in
    ``thumb_window`` (up to 10 recent frames).  This is more robust than
    pairwise comparison because it smooths over camera jitter and minor
    per-frame noise.

    Args:
        thumb:        Current frame thumbnail (128×128×3 uint8).
        thumb_window: Deque of recent thumbnails (max 10).

    Returns:
        (pixel_score, hist_score) — unnormalized change scores.
    """
    if not thumb_window:
        return 0.0, 0.0

    # Compute pixel-wise mean of the window as the temporal baseline
    baseline = np.mean(
        np.stack(list(thumb_window)).astype(np.float32), axis=0,
    )

    pixel_score = float(np.mean(np.abs(
        thumb.astype(np.float32) - baseline
    )) / 255.0)

    # For histogram distance, compare to the most recent frame in the window
    # (histogram of an average image is meaningless)
    hist_score = float(_histogram_distance(thumb_window[-1], thumb))

    return pixel_score, hist_score


def compute_lightweight_change_from_video(
    video_path: str,
    target_fps: Optional[float] = 10.0,
    pause_event: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
    cache_key: Optional[str] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Compute lightweight change scores directly from a video file.

    Uses a 10-frame moving average for robust change detection: each frame
    is compared against the mean of the previous 10 thumbnails (pixel L1)
    and the most recent frame (histogram distance).

    Two code paths depending on disk frame cache state:

    **Path A — no disk cache (first run):**
    Stream-decode from video, write each frame as JPEG to the disk frame
    cache (if ``cache_key`` is provided), compute change scores on-the-fly.
    Memory: ~0.5 MB constant (10 thumbnails × 128×128×3).

    **Path B — disk cache exists:**
    Read JPEGs from disk, compute change scores.  No video decode needed.

    Args:
        video_path:        Path to the video file.
        target_fps:        Subsample rate (default 10 FPS).
        pause_event:       Optional threading.Event for pause/resume.
        progress_callback: Optional callable(current, total).
        cache_key:         Session cache key for disk frame cache.
                           If None, no disk cache is read or written.

    Returns:
        (scores, sampled_indices) — same contract as _do_embed_all_frames
        except scores is (N,) instead of (N, C) embeddings.
    """
    import collections

    # --- Path B: read from existing disk frame cache ---
    if cache_key:
        try:
            from interview.disk_frame_cache import (
                frame_cache_exists, iter_cached_frames, get_frame_cache_meta,
            )
        except ImportError:
            frame_cache_exists = None  # running outside interview package

        if frame_cache_exists and frame_cache_exists(cache_key):
            logger.info("Disk frame cache found for %s — reading from cache", cache_key)
            meta = get_frame_cache_meta(cache_key)
            sampled_indices = meta.get("sampled_indices", []) if meta else []
            est_total = len(sampled_indices)

            pixel_scores_list: List[float] = []
            hist_scores_list: List[float] = []
            indices: List[int] = []
            thumb_window: collections.deque = collections.deque(maxlen=10)

            for n_done, (fidx, pil_img) in enumerate(iter_cached_frames(cache_key), 1):
                thumb = np.array(pil_img.resize(_CHANGE_THUMB_SIZE, Image.BILINEAR))
                del pil_img

                px, hs = _compute_change_score_moving_avg(thumb, thumb_window)
                pixel_scores_list.append(px)
                hist_scores_list.append(hs)
                thumb_window.append(thumb)
                indices.append(fidx)

                if progress_callback and n_done % 500 == 0:
                    progress_callback(n_done, est_total)
                if pause_event is not None:
                    pause_event.wait()

            return _normalize_change_scores(pixel_scores_list, hist_scores_list), indices

    # --- Path A: stream decode from video, optionally write disk cache ---
    disk_cache_writer = None
    if cache_key:
        try:
            from interview.disk_frame_cache import (
                init_frame_cache, write_frame, finalize_frame_cache,
            )
            disk_cache_writer = True
        except ImportError:
            disk_cache_writer = None

    container = av.open(video_path)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate or 30)
    total_frames = stream.frames or 0

    skip = max(1, int(round(src_fps / target_fps))) if target_fps else 1
    est_total = total_frames // skip if total_frames else 0

    # Initialize disk cache
    if disk_cache_writer:
        probe_stream = container.streams.video[0]
        w = probe_stream.codec_context.width
        h = probe_stream.codec_context.height
        import time as _time
        init_frame_cache(cache_key, {
            "target_fps": target_fps,
            "src_fps": src_fps,
            "resolution": [w, h],
            "created_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    pixel_scores_list = []
    hist_scores_list = []
    indices: List[int] = []
    thumb_window: collections.deque = collections.deque(maxlen=10)
    frame_idx = 0
    cache_bytes = 0

    for av_frame in container.decode(video=0):
        if frame_idx % skip == 0:
            pil_img = av_frame.to_image()

            # Write to disk cache before discarding
            if disk_cache_writer:
                write_frame(cache_key, frame_idx, pil_img)
                # Estimate size (JPEG ~300 KB per 1080p frame)
                cache_bytes += 300_000

            thumb = np.array(pil_img.resize(_CHANGE_THUMB_SIZE, Image.BILINEAR))
            del pil_img  # free full-res image immediately

            px, hs = _compute_change_score_moving_avg(thumb, thumb_window)
            pixel_scores_list.append(px)
            hist_scores_list.append(hs)
            thumb_window.append(thumb)
            indices.append(frame_idx)

            n_done = len(indices)
            if progress_callback and n_done % 500 == 0:
                progress_callback(n_done, est_total)

            if pause_event is not None:
                pause_event.wait()

        frame_idx += 1

    container.close()

    # Finalize disk cache
    if disk_cache_writer and indices:
        finalize_frame_cache(cache_key, indices, cache_bytes)
        logger.info(
            "Disk frame cache written: %d frames, ~%.1f GB",
            len(indices), cache_bytes / 1e9,
        )

    return _normalize_change_scores(pixel_scores_list, hist_scores_list), indices


def _normalize_change_scores(
    pixel_scores_list: List[float],
    hist_scores_list: List[float],
) -> np.ndarray:
    """Normalize pixel and histogram scores to [0, 1] and combine via max."""
    pixel_scores = np.array(pixel_scores_list, dtype=np.float32)
    hist_scores = np.array(hist_scores_list, dtype=np.float32)

    px_max = pixel_scores.max() if len(pixel_scores) else 0.0
    if px_max > 0:
        pixel_scores /= px_max

    hs_max = hist_scores.max() if len(hist_scores) else 0.0
    if hs_max > 0:
        hist_scores /= hs_max

    if len(pixel_scores):
        return np.maximum(pixel_scores, hist_scores)
    return np.empty(0, dtype=np.float32)


def _median_filter_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 1:
        raise InitialSeedingError(f"kernel_size must be positive, got {kernel_size}")
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    try:
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
        return np.median(windows, axis=-1).astype(values.dtype, copy=False)
    except AttributeError:
        filtered = np.empty_like(values)
        for i in range(len(values)):
            start = i
            end = i + kernel_size
            filtered[i] = np.median(padded[start:end])
        return filtered


def smooth_change_scores(diff: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return _median_filter_1d(diff, kernel_size=kernel_size)


def uniform_indices(T_len: int, K: int) -> List[int]:
    return sorted({int(round(i * T_len / K)) for i in range(max(K, 1))})


def top_change_indices(smooth_diff: np.ndarray, max_candidates: int, min_spacing: int) -> List[int]:
    idx_sorted = np.argsort(-smooth_diff)
    chosen: List[int] = []
    for idx in idx_sorted:
        if len(chosen) >= max_candidates:
            break
        if all(abs(idx - c) >= min_spacing for c in chosen):
            chosen.append(int(idx))
    return sorted(chosen)


def select_keyframes(T_len: int, frac: float, smooth_diff: np.ndarray, min_spacing: int = 30) -> List[int]:
    K = max(1, int(frac * T_len))
    base = set(uniform_indices(T_len, K))
    changed = set(top_change_indices(smooth_diff, max_candidates=3 * K, min_spacing=min_spacing))
    merged = sorted(base.union(changed))
    if len(merged) > K:
        idx = np.linspace(0, len(merged) - 1, num=K, dtype=int)
        merged = [merged[i] for i in idx]
    return sorted(set(merged))


def _detect_keyframes(
    video_path: str,
    cache_dir: str,
    cache_key: str,
    embedding_batch: int,
    keyframe_frac: float,
    min_spacing: int,
) -> Tuple[List[int], int, int, int, float]:
    width, height, frames_count, fps = _get_video_info_pyav(video_path)

    embeds = _compute_sam3_frame_embeddings(cache_key, video_path, embedding_batch, cache_dir)
    if embeds.shape[0] != frames_count:
        logger.warning(
            "Embedding frame count (%d) does not match video frames (%d); proceeding with min length",
            embeds.shape[0],
            frames_count,
        )
        frames_count = min(frames_count, embeds.shape[0])
        embeds = embeds[:frames_count]

    diff = compute_change_scores(embeds)
    smooth = smooth_change_scores(diff, kernel_size=5)
    keyframes = select_keyframes(frames_count, keyframe_frac, smooth, min_spacing=min_spacing)
    logger.info("Selected %d keyframes out of %d total frames", len(keyframes), frames_count)
    return keyframes, width, height, frames_count, fps


# ---------------------------------------------------------------------------
# Sam3TextDetector (replaces GroundingDINOHelper)
# ---------------------------------------------------------------------------

class Sam3TextDetector:
    """Text-prompted object detection using Sam3VideoModel.

    Replaces GroundingDINOHelper. Uses SAM3's built-in text-based detection
    to find objects in individual frames.
    """

    def __init__(self):
        self.model, self.processor = _get_sam3_video_model()
        self.prompt = self._resolve_prompt()

    @staticmethod
    def _resolve_prompt() -> str:
        """Resolve text prompt from env vars."""
        prompt = os.getenv('PROMPT_TEXT', '')
        if not prompt:
            # Backward compat: fall back to GDINO env vars during transition
            prompt = os.getenv('GROUNDING_DINO_LABELS', 'person')
        return prompt.strip()

    def infer_frame(
        self,
        frame: Image.Image,
        *,
        prompt: Optional[str] = None,
    ) -> List[KeyframeDetection]:
        """Detect objects in a single frame using text prompt.

        Returns list of KeyframeDetection with xyxy pixel coords and scores.
        """
        prompt_text = prompt or self.prompt

        # Single-frame inference via Sam3VideoModel
        session = self.processor.init_video_session(
            inference_device=DEVICE, dtype=DTYPE
        )
        session = self.processor.add_text_prompt(
            inference_session=session, text=prompt_text
        )

        inputs = self.processor(images=frame, device=DEVICE, return_tensors="pt")
        with torch.no_grad():
            output = self.model(
                inference_session=session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )

        processed = self.processor.postprocess_outputs(
            session, output, original_sizes=inputs.original_sizes
        )

        detections: List[KeyframeDetection] = []
        obj_ids = processed.get('object_ids', [])
        scores = processed.get('scores', [])
        boxes = processed.get('boxes', [])  # xyxy pixel coords

        for i in range(len(obj_ids)):
            box = boxes[i] if i < len(boxes) else None
            score = float(scores[i]) if i < len(scores) else 0.0
            if box is None:
                continue
            if hasattr(box, 'tolist'):
                box = box.tolist()
            xyxy = np.array(box, dtype=np.float32)
            detections.append(
                KeyframeDetection(
                    frame_idx=-1,  # set by caller
                    xyxy=xyxy,
                    score=score,
                    label=prompt_text,
                )
            )

        return detections


def _run_text_detection_on_keyframes(
    video_path: str,
    keyframes: List[int],
    prompt: Optional[str],
) -> Dict[int, List[KeyframeDetection]]:
    """Run Sam3VideoModel text detection on selected keyframes (replaces GDINO)."""
    from tqdm import tqdm

    detector = Sam3TextDetector()
    detections: Dict[int, List[KeyframeDetection]] = {}

    for frame_idx in tqdm(keyframes, desc="Text detection on keyframes", unit="kf"):
        pil_frame = _read_frame_pyav(video_path, frame_idx)
        if pil_frame is None:
            logger.warning("Failed to read keyframe %d", frame_idx)
            continue
        dets = detector.infer_frame(pil_frame, prompt=prompt)
        for d in dets:
            d.frame_idx = frame_idx
        detections[frame_idx] = dets

    return detections


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def xyxy_to_percent(xyxy: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    # Guard against division by zero
    if width <= 0 or height <= 0:
        logger.warning("Invalid image dimensions (width=%s, height=%s), returning zeros", width, height)
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0, x1, y1 = xyxy
    x0 = max(0.0, min(float(width - 1), float(x0)))
    y0 = max(0.0, min(float(height - 1), float(y0)))
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    return (x0 / width) * 100.0, (y0 / height) * 100.0, (w / width) * 100.0, (h / height) * 100.0


def _percent_xywh_to_xyxy_px(
    x_pct: float, y_pct: float, w_pct: float, h_pct: float,
    img_w: int, img_h: int,
) -> np.ndarray:
    """Convert LS percent coords to pixel xyxy. Canonical pattern."""
    x1 = (x_pct / 100.0) * img_w
    y1 = (y_pct / 100.0) * img_h
    x2 = x1 + (w_pct / 100.0) * img_w
    y2 = y1 + (h_pct / 100.0) * img_h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Prediction building & upload
# ---------------------------------------------------------------------------

def _build_prediction(
    tracks: List[Dict[str, Any]],
    width: int,
    height: int,
    frames_count: int,
    fps: float,
) -> Dict[str, Any]:
    duration = frames_count / fps if fps > 0 else 0.0
    results: List[Dict[str, Any]] = []
    for tr in tracks:
        seq_items = []
        for item in tr["sequence"]:
            x_pct, y_pct, w_pct, h_pct = xyxy_to_percent(item["xyxy"], width, height)
            frame_num = int(item["frame"])
            seq_items.append(
                {
                    "frame": frame_num,
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "enabled": item.get("enabled", True),
                    "rotation": 0,
                    "time": (frame_num - 1) / fps if fps > 0 else 0.0,
                }
            )

        if not seq_items:
            continue

        results.append(
            {
                "id": f"auto-track-{tr['track_id']}",
                "type": "videorectangle",
                "from_name": "box",
                "to_name": "video",
                "score": 1.0,
                "origin": "manual",
                "value": {
                    "sequence": seq_items,
                    "framesCount": frames_count,
                    "duration": duration,
                    "labels": [tr.get("label") or "object"],
                },
                "meta": {"text": "id:"},
            }
        )
        _ensure_meta_text_placeholder(results[-1])

    prediction = {"result": results, "score": 1.0, "model_version": "sam3-init-seed"}
    return prediction


def _upload_prediction(ls, task_id: int, prediction: Dict[str, Any]):
    try:
        result = ls.predictions.create(
            task=task_id,
            score=prediction.get("score", 0.0),
            model_version=prediction.get("model_version", "sam3-init-seed"),
            result=prediction.get("result", []),
        )
        pred_id = getattr(result, "id", None)
        if pred_id is not None:
            logger.info("Upload complete, prediction id=%s", pred_id)
        else:
            logger.info("Upload request completed (no prediction id in response)")
    except Exception as exc:  # pragma: no cover
        msg = str(exc)
        err_no = getattr(exc, "errno", None)
        if "504" in msg:
            logger.warning("Received 504 from LS during prediction upload; assuming it succeeded.")
        else:
            if err_no is not None:
                logger.error("Failed to upload prediction (errno=%s): %s", err_no, msg)
            else:
                logger.error("Failed to upload prediction: %s", msg)
