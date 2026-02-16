"""Detection pipeline for Interview UI.

Implements text-based object detection using Sam3Model (image mode) with
Promptable Concept Segmentation (PCS). Detections are produced per sampled
keyframe, filtered through NMS, padded, and stored as CropData objects on
the InterviewSession.

Key components:
- Sam3TextBasedDetector: wraps Sam3Model for text-prompted instance
  segmentation.  Unlike the Sam3TextDetector in seeding_common.py (which
  uses Sam3VideoModel), this class works with the image-only model and
  supports multi-prompt reuse via cached pixel_values.
- nms_numpy: greedy non-maximum suppression (pure numpy, no cv2/torchvision).
- pad_boxes: expand detections by a configurable fraction, clamped to frame
  bounds.
- run_detection_pipeline: end-to-end entry point called from routes.py.
- run_recall_strategy: additional detection passes to close recall gaps.
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Parent-directory imports (seeding_common lives one level up)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av  # noqa: E402

from seeding_common import (  # noqa: E402
    _get_sam3_image_model,
    _read_frame_pyav,
    _get_video_info_pyav,
    _compute_sam3_frame_embeddings,
    _do_embed_all_frames,
    compute_change_scores,
    compute_lightweight_change_from_video,
    smooth_change_scores,
    select_keyframes,
    DEVICE,
    DTYPE,
)

from .state import CropData, CropLabel, CropSource, InterviewSession, Phase  # noqa: E402
from .cache_manager import save_session  # noqa: E402

logger = logging.getLogger(__name__)


def _read_frame_cached_or_pyav(
    video_path: str,
    frame_idx: int,
    cache_key: Optional[str] = None,
) -> Optional[Image.Image]:
    """Read a frame via 3-tier cache (LRU → disk → PyAV seek).

    Thin wrapper so callers don't need a local import each time.
    """
    from .frame_cache import read_frame_cached
    return read_frame_cached(video_path, frame_idx, cache_key=cache_key)


def _log_rss(tag: str) -> int:
    """Log current RSS (resident set size) in MB and return value."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is in KB; on macOS it's in bytes
    if sys.platform == "linux":
        rss_mb = rss_kb / 1024
    else:
        rss_mb = rss_kb / (1024 * 1024)
    logger.info("[MEM] %s: RSS=%.0f MB", tag, rss_mb)
    return int(rss_mb)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Keyframe sampling
DEFAULT_KEYFRAME_FRAC = float(os.getenv("INTERVIEW_KEYFRAME_FRAC", "0.04"))
DEFAULT_MIN_SPACING = int(os.getenv("INTERVIEW_MIN_SPACING", "30"))
DEFAULT_EMBEDDING_BATCH = int(os.getenv("INTERVIEW_EMBEDDING_BATCH", "64"))
EMBEDDING_CACHE_DIR = os.getenv("INTERVIEW_EMBEDDING_CACHE", "/tmp/interview_embed_cache")
INITIAL_KEYFRAME_COUNT = int(os.getenv("INTERVIEW_INITIAL_KEYFRAMES", "40"))

# Batch detection
DEFAULT_DETECT_BATCH = int(os.getenv("INTERVIEW_DETECT_BATCH", "8"))

# Detection
DEFAULT_DETECTION_THRESHOLD = float(os.getenv("INTERVIEW_DETECT_THRESHOLD", "0.3"))
DEFAULT_MASK_THRESHOLD = float(os.getenv("INTERVIEW_MASK_THRESHOLD", "0.5"))
DEFAULT_NMS_IOU_THRESHOLD = float(os.getenv("INTERVIEW_NMS_IOU", "0.5"))
DEFAULT_PAD_FRAC = float(os.getenv("INTERVIEW_PAD_FRAC", "0.1"))

# Deduplication
DEFAULT_DEDUP_IOU_THRESHOLD = float(os.getenv("INTERVIEW_DEDUP_IOU", "0.5"))

# Minimum box area in pixels to keep a detection
MIN_BOX_AREA_PX = int(os.getenv("INTERVIEW_MIN_BOX_AREA", "100"))

# Embedding FPS cap — subsample to this FPS during background embedding
EMBEDDING_TARGET_FPS = float(os.getenv("INTERVIEW_EMBEDDING_FPS", "10"))

# Embedding mode: "lightweight" (CPU pixel+histogram) or "sam3" (GPU embeddings)
EMBEDDING_MODE = os.getenv("INTERVIEW_EMBEDDING_MODE", "lightweight")


def _get_embedding_mode() -> str:
    """Return the current embedding mode for testability."""
    return EMBEDDING_MODE


# ===========================================================================
# NMS (pure numpy)
# ===========================================================================

def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes_a: (N, 4) array of [x1, y1, x2, y2].
        boxes_b: (M, 4) array of [x1, y1, x2, y2].

    Returns:
        (N, M) IoU matrix.
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)  # (N, M)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter_area
    iou = np.where(union > 0, inter_area / union, 0.0)
    return iou


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Greedy non-maximum suppression (pure numpy, no cv2/torchvision).

    Sorts detections by score in descending order.  Keeps each box only if
    its IoU with every previously kept box is below ``iou_threshold``.

    Args:
        boxes:  (N, 4) float array of [x1, y1, x2, y2] coordinates.
        scores: (N,) float array of confidence scores.
        iou_threshold: Suppress boxes with IoU >= this value against a
            higher-scoring kept box.

    Returns:
        (K,) int array of indices into the original arrays that survive NMS.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep: List[int] = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    suppressed = np.zeros(len(boxes), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))

        # Compute IoU of this box against all remaining unsuppressed boxes
        xx1 = np.maximum(x1[idx], x1)
        yy1 = np.maximum(y1[idx], y1)
        xx2 = np.minimum(x2[idx], x2)
        yy2 = np.minimum(y2[idx], y2)

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union = areas[idx] + areas - inter_area
        iou = np.where(union > 0, inter_area / union, 0.0)

        suppressed |= iou >= iou_threshold

    return np.array(keep, dtype=np.int64)


# ===========================================================================
# Box padding
# ===========================================================================

def pad_boxes(
    boxes: np.ndarray,
    width: int,
    height: int,
    pad_frac: float = 0.1,
) -> np.ndarray:
    """Expand each box by ``pad_frac`` on all sides, clamped to frame bounds.

    For a 10 % pad (the default), a 100 px-wide box gains 10 px on the left
    and 10 px on the right (total 120 px).

    Args:
        boxes:    (N, 4) float array of [x1, y1, x2, y2] in pixel coords.
        width:    Frame width in pixels.
        height:   Frame height in pixels.
        pad_frac: Fraction of box width/height to add on each side.

    Returns:
        (N, 4) padded boxes, clamped to [0, width] x [0, height].
    """
    if len(boxes) == 0:
        return boxes.copy()

    padded = boxes.copy().astype(np.float32)
    bw = padded[:, 2] - padded[:, 0]
    bh = padded[:, 3] - padded[:, 1]

    dx = bw * pad_frac
    dy = bh * pad_frac

    padded[:, 0] -= dx
    padded[:, 1] -= dy
    padded[:, 2] += dx
    padded[:, 3] += dy

    # Clamp to frame bounds
    padded[:, 0] = np.clip(padded[:, 0], 0, width)
    padded[:, 1] = np.clip(padded[:, 1], 0, height)
    padded[:, 2] = np.clip(padded[:, 2], 0, width)
    padded[:, 3] = np.clip(padded[:, 3], 0, height)

    return padded


# ===========================================================================
# Batch decode + detect helpers
# ===========================================================================

def uniform_indices(total: int, k: int) -> List[int]:
    """Return *k* uniformly-spaced frame indices in [0, total).

    Always includes the first and last frame when k >= 2.
    """
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    if k == 1:
        return [total // 2]
    return [int(round(i * (total - 1) / (k - 1))) for i in range(k)]


FRAMES_PER_ROUND = int(os.getenv("INTERVIEW_FRAMES_PER_ROUND", "40"))
VALIDATION_FRAMES_COUNT = int(os.getenv("INTERVIEW_VALIDATION_FRAMES", "20"))


def select_round_frames(
    session: "InterviewSession",
    round_num: int,
    frames_per_round: int = FRAMES_PER_ROUND,
) -> List[int]:
    """Select frames for a round.

    Round 1: uniform temporal stratification with change-keyframe preference.
    Round 2+: sample primarily from change-detected keyframes (if available),
    falling back to uniform stratification with prior-round exclusion.
    """
    total = session.frames_count
    if total <= 0:
        return []

    # Exclude frames from prior rounds
    used: set = set()
    for rn, rf in session.round_frames.items():
        if rn < round_num:
            used.update(rf)

    if round_num >= 2:
        # Round 2+: draw from change-detected keyframes
        change = session.change_keyframes
        if change:
            available_change = [f for f in change if f not in used]
            if not available_change:
                # All change frames used — fall back to full range
                available_change = [f for f in range(total) if f not in used]
                if not available_change:
                    available_change = list(range(total))

            k = min(frames_per_round, len(available_change))
            if k <= 0:
                return []
            # Uniformly spaced among the available change frames
            indices = [available_change[i * len(available_change) // k] for i in range(k)]
            return sorted(set(indices))

    # Round 1 (or round 2+ without change data): uniform temporal bins
    available = [i for i in range(total) if i not in used]
    if not available:
        available = list(range(total))

    k = min(frames_per_round, len(available))
    if k <= 0:
        return []

    bin_width = total / k
    change_set = set(session.change_keyframes) if session.change_keyframes else set()

    selected: List[int] = []
    for i in range(k):
        bin_start = int(i * bin_width)
        bin_end = int((i + 1) * bin_width)
        candidates = [f for f in available if bin_start <= f < bin_end]

        if not candidates:
            mid = (bin_start + bin_end) // 2
            nearest = min(available, key=lambda f: abs(f - mid))
            if nearest not in selected:
                selected.append(nearest)
            continue

        change_in_bin = [f for f in candidates if f in change_set]
        mid = (bin_start + bin_end) // 2
        if change_in_bin:
            selected.append(min(change_in_bin, key=lambda f: abs(f - mid)))
        else:
            selected.append(min(candidates, key=lambda f: abs(f - mid)))

    return sorted(set(selected))


def select_validation_frames(
    session: "InterviewSession",
    count: int = VALIDATION_FRAMES_COUNT,
) -> List[int]:
    """Select held-out validation frames for Round 1.

    Uses the same temporal-bin strategy as ``select_round_frames`` but
    excludes Round 1 detection frames.  These frames are never used for
    MLP training — only for evaluation.

    Args:
        session: InterviewSession (must have frames_count set).
        count:   Number of validation frames to select.

    Returns:
        Sorted list of 0-based frame indices.
    """
    total = session.frames_count
    if total <= 0 or count <= 0:
        return []

    # Exclude Round 1 detection frames
    excluded: set = set()
    if 1 in session.round_frames:
        excluded.update(session.round_frames[1])

    available = [i for i in range(total) if i not in excluded]
    if not available:
        return []

    k = min(count, len(available))
    if k <= 0:
        return []

    bin_width = total / k
    change_set = set(session.change_keyframes) if session.change_keyframes else set()

    selected: List[int] = []
    for i in range(k):
        bin_start = int(i * bin_width)
        bin_end = int((i + 1) * bin_width)
        candidates = [f for f in available if bin_start <= f < bin_end]

        if not candidates:
            mid = (bin_start + bin_end) // 2
            nearest = min(available, key=lambda f: abs(f - mid))
            if nearest not in selected:
                selected.append(nearest)
            continue

        change_in_bin = [f for f in candidates if f in change_set]
        mid = (bin_start + bin_end) // 2
        if change_in_bin:
            selected.append(min(change_in_bin, key=lambda f: abs(f - mid)))
        else:
            selected.append(min(candidates, key=lambda f: abs(f - mid)))

    return sorted(set(selected))


_MAX_DECODE_AFTER_SEEK = int(os.getenv("INTERVIEW_MAX_DECODE_AFTER_SEEK", "500"))


def _decode_frames_sequential(
    video_path: str,
    frame_indices: List[int],
    max_decode_after_seek: int = _MAX_DECODE_AFTER_SEEK,
    cache_key: Optional[str] = None,
) -> Dict[int, Image.Image]:
    """Decode specific frames using keyframe-seeking for widely-spaced targets.

    For each target frame, seeks to the nearest prior keyframe and then
    decodes forward to the exact frame.  This avoids decoding every frame
    in the video (which for 30K frames takes ~10 minutes) when only ~40
    uniformly-spaced frames are needed.

    When *cache_key* is provided, the disk frame cache is checked first
    and only frames not found on disk fall through to PyAV decode.

    A safety limit (``max_decode_after_seek``) caps how many frames we
    decode after each seek.  If the target isn't found within that window
    the frame is skipped and a warning is logged.

    Args:
        video_path:             Path to video file.
        frame_indices:          List of 0-based frame indices to decode (need not be sorted).
        max_decode_after_seek:  Maximum frames to decode after each seek before giving up.
        cache_key:              Optional session cache key for disk frame cache lookup.

    Returns:
        Dict mapping frame_idx -> PIL Image.
    """
    if not frame_indices:
        return {}

    result: Dict[int, Image.Image] = {}
    sorted_targets = sorted(set(frame_indices))

    # Check disk frame cache first (avoids PyAV seeks for cached frames)
    if cache_key:
        from .frame_cache import read_frame_cached
        still_needed = []
        for fidx in sorted_targets:
            img = read_frame_cached(video_path, fidx, cache_key=cache_key)
            if img is not None:
                result[fidx] = img
            else:
                still_needed.append(fidx)
        if result:
            logger.info(
                "Disk cache hit for %d / %d frames", len(result), len(sorted_targets),
            )
        if not still_needed:
            return result
        sorted_targets = still_needed

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        time_base = stream.time_base

        for target_idx in sorted_targets:
            # Seek to just before the target frame.  PyAV seeks to the
            # nearest prior keyframe, then we decode forward.
            target_pts = int(target_idx / fps / time_base)
            container.seek(target_pts, stream=stream)

            frame_count_after_seek = None
            decoded_count = 0
            found = False
            for av_frame in container.decode(video=0):
                decoded_count += 1
                # After seek, the first decoded frame is the keyframe at or
                # before our target.  We figure out which frame index it is
                # from its pts, then count forward.
                if frame_count_after_seek is None:
                    if av_frame.pts is not None and time_base:
                        frame_count_after_seek = int(
                            round(float(av_frame.pts * time_base) * fps)
                        )
                    else:
                        # Fallback: assume seek landed on target (best effort)
                        frame_count_after_seek = target_idx

                if frame_count_after_seek == target_idx:
                    result[target_idx] = av_frame.to_image()
                    found = True
                    break
                elif frame_count_after_seek > target_idx:
                    # Overshot (rare, can happen with some codecs) — take it
                    result[target_idx] = av_frame.to_image()
                    found = True
                    break

                frame_count_after_seek += 1

                if decoded_count >= max_decode_after_seek:
                    break

            if not found:
                logger.warning(
                    "Could not find frame %d after decoding %d frames post-seek; skipping",
                    target_idx, decoded_count,
                )

        logger.info(
            "Decoded %d / %d target frames via seek from %s",
            len(result), len(sorted_targets), video_path,
        )
    finally:
        container.close()

    return result


def precompute_text_tokens(
    detector: "Sam3TextBasedDetector",
    prompt: str,
) -> Optional[Dict[str, torch.Tensor]]:
    """Pre-tokenize a text prompt for reuse across batches.

    Returns a dict with ``input_ids`` and ``attention_mask`` tensors
    (shape [1, seq_len]) on the target device, or None if the processor
    has no tokenizer attribute.
    """
    tokenizer = getattr(detector.processor, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        tokens = tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True,
        )
        return {k: v.to(DEVICE) for k, v in tokens.items()}
    except Exception:
        return None


def _detect_batch(
    detector: "Sam3TextBasedDetector",
    frames: Dict[int, Image.Image],
    prompt: str,
    width: int,
    height: int,
    batch_size: int = DEFAULT_DETECT_BATCH,
    nms_iou: float = DEFAULT_NMS_IOU_THRESHOLD,
    pad_frac: float = DEFAULT_PAD_FRAC,
    precomputed_text: Optional[Dict[str, torch.Tensor]] = None,
) -> List[CropData]:
    """Run batched detection on pre-decoded frames.

    Groups frames into batches of ``batch_size`` and runs the Sam3 model
    in a single forward pass per batch.  The processor requires ``text``
    to be a **list of prompts** (one per image) for batched inference —
    we replicate the same prompt for every image in the batch.

    Falls back to per-frame inference on OOM.

    Args:
        detector:       Sam3TextBasedDetector instance.
        frames:         Dict of frame_idx -> PIL Image (from _decode_frames_sequential).
        prompt:         Text prompt.
        width:          Video width in pixels.
        height:         Video height in pixels.
        batch_size:     Frames per GPU forward pass.
        nms_iou:        NMS IoU threshold.
        pad_frac:       Box padding fraction.
        precomputed_text: Optional pre-tokenized text (from precompute_text_tokens).
            When provided, skips re-tokenization and expands tokens to batch size.

    Returns:
        List of CropData across all frames.
    """
    all_crops: List[CropData] = []
    sorted_indices = sorted(frames.keys())

    # Post-process at reduced resolution to avoid massive CPU mask allocations.
    # Full-res masks (e.g. 32 × 100 queries × 1920 × 1080 × 4 bytes ≈ 26 GB)
    # are never used — we only need bounding boxes.  Post-process at 256×256,
    # then scale boxes back to original resolution.
    _PP_SIZE = 256  # post-process mask resolution (small — we only want boxes)

    _log_rss("_detect_batch start")

    for batch_start in range(0, len(sorted_indices), batch_size):
        batch_indices = sorted_indices[batch_start:batch_start + batch_size]
        batch_images = [frames[idx] for idx in batch_indices]
        n_batch = len(batch_images)

        # Remember original sizes for box rescaling
        orig_sizes = [(img.height, img.width) for img in batch_images]

        try:
            if precomputed_text is not None:
                # Image-only preprocessing + expand pre-tokenized text
                inputs = detector.processor(
                    images=batch_images, return_tensors="pt",
                ).to(DEVICE)
                for key, val in precomputed_text.items():
                    inputs[key] = val.expand(n_batch, -1)
            else:
                text_prompts = [prompt] * n_batch
                inputs = detector.processor(
                    images=batch_images, text=text_prompts, return_tensors="pt",
                ).to(DEVICE)

            with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
                outputs = detector.model(**inputs)

            # Use small target_sizes to avoid huge CPU mask allocations
            small_targets = [[_PP_SIZE, _PP_SIZE]] * n_batch

            batch_results = detector.processor.post_process_instance_segmentation(
                outputs,
                threshold=detector.threshold,
                mask_threshold=detector.mask_threshold,
                target_sizes=small_targets,
            )

            # Free model outputs immediately (large GPU tensors)
            del outputs, inputs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            logger.warning(
                "OOM during batch detection (batch_size=%d), falling back to per-frame",
                len(batch_images),
            )
            for idx in batch_indices:
                crops = _detect_single_frame(
                    detector, frames[idx], prompt, idx, width, height,
                    nms_iou=nms_iou, pad_frac=pad_frac,
                )
                all_crops.extend(crops)
            continue

        # Post-process each image in the batch
        for i, frame_idx in enumerate(batch_indices):
            results_i = batch_results[i]
            dets = Sam3TextBasedDetector._parse_results(results_i, prompt)

            if not dets:
                continue

            boxes = np.array([d["xyxy"] for d in dets], dtype=np.float32)
            scores_arr = np.array([d["score"] for d in dets], dtype=np.float32)

            # Scale boxes from _PP_SIZE back to original resolution
            oh, ow = orig_sizes[i]
            boxes[:, 0] *= ow / _PP_SIZE  # x1
            boxes[:, 1] *= oh / _PP_SIZE  # y1
            boxes[:, 2] *= ow / _PP_SIZE  # x2
            boxes[:, 3] *= oh / _PP_SIZE  # y2

            keep_idx = nms_numpy(boxes, scores_arr, iou_threshold=nms_iou)
            if len(keep_idx) == 0:
                continue

            boxes = boxes[keep_idx]
            scores_arr = scores_arr[keep_idx]
            boxes = pad_boxes(boxes, width, height, pad_frac=pad_frac)

            for j in range(len(boxes)):
                crop = CropData(
                    crop_id=str(uuid.uuid4())[:12],
                    frame_idx=frame_idx,
                    xyxy=boxes[j],
                    score=float(scores_arr[j]),
                    label=CropLabel.PENDING,
                    source=CropSource.TEXT_DETECT,
                    prompt=prompt,
                )
                all_crops.append(crop)

        # Free batch results (contains masks) and collect
        del batch_results
        gc.collect()

    _log_rss("_detect_batch end")
    return all_crops


# ===========================================================================
# Sam3TextBasedDetector (image-mode PCS)
# ===========================================================================

class Sam3TextBasedDetector:
    """Text-prompted instance segmentation via Sam3Model (image mode).

    Unlike ``Sam3TextDetector`` in seeding_common.py which wraps
    ``Sam3VideoModel``, this class uses the lighter-weight ``Sam3Model``
    (image-only) and supports caching of preprocessed pixel values so that
    multiple text prompts can be run against the same frame cheaply.

    Typical usage::

        detector = Sam3TextBasedDetector(threshold=0.3)
        detector.set_frame(pil_image)
        dets = detector.detect("person")
        more = detector.detect("bicycle")  # reuses cached frame
    """

    def __init__(
        self,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    ):
        self.model, self.processor = _get_sam3_image_model()
        self.threshold = threshold
        self.mask_threshold = mask_threshold

        # Cached per-frame data
        self._cached_pil: Optional[Image.Image] = None
        self._cached_pixel_values: Optional[torch.Tensor] = None
        self._cached_original_sizes: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    # Frame caching
    # ------------------------------------------------------------------

    def set_frame(self, pil_image: Image.Image) -> None:
        """Pre-process and cache pixel values for a frame.

        Calling :meth:`detect` afterwards will reuse these tensors, saving
        the image encoding cost when running multiple prompts on the same
        frame.
        """
        if pil_image is self._cached_pil:
            return  # already cached

        inputs = self.processor(images=pil_image, return_tensors="pt").to(DEVICE)
        self._cached_pil = pil_image
        self._cached_pixel_values = inputs.pixel_values
        # original_sizes may be present depending on processor version
        self._cached_original_sizes = (
            inputs.get("original_sizes").tolist()
            if inputs.get("original_sizes") is not None
            else [[pil_image.height, pil_image.width]]
        )

    def clear_cache(self) -> None:
        """Release cached frame tensors to free GPU memory."""
        self._cached_pil = None
        self._cached_pixel_values = None
        self._cached_original_sizes = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        prompt: str,
        pil_image: Optional[Image.Image] = None,
    ) -> List[Dict[str, Any]]:
        """Run text-based detection (PCS) on a frame.

        If ``pil_image`` is provided it replaces any cached frame.  If the
        frame was previously set via :meth:`set_frame` and ``pil_image`` is
        ``None``, the cached tensors are reused.

        Args:
            prompt:    Text describing the target class (e.g. "person").
            pil_image: Optional PIL Image.  If omitted, uses cached frame.

        Returns:
            List of dicts, each with keys:
                - ``xyxy``: np.ndarray of [x1, y1, x2, y2] in pixel coords.
                - ``score``: float confidence.
                - ``label``: str (the text prompt used).
        """
        if pil_image is not None:
            self.set_frame(pil_image)

        if self._cached_pil is None:
            raise RuntimeError(
                "No frame cached. Call set_frame() or pass pil_image."
            )

        # Full forward pass with text prompt.
        # NOTE: We re-run the full model here rather than passing pre-
        # computed vision features because the Sam3Processor text+image
        # encoding is tightly coupled.  A future optimisation can split
        # the vision backbone call from the prompt encoder call.
        inputs = self.processor(
            images=self._cached_pil,
            text=prompt,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
            outputs = self.model(**inputs)

        # Remember original size for box rescaling
        orig_h = self._cached_pil.height
        orig_w = self._cached_pil.width

        # Post-process at small resolution to avoid huge CPU mask allocations.
        # We only need boxes (not masks), so 256×256 is sufficient for
        # deriving bounding boxes from the mask fallback path.
        _PP_SIZE = 256
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[[_PP_SIZE, _PP_SIZE]],
        )[0]

        del outputs, inputs

        dets = self._parse_results(results, prompt)

        # Scale boxes from _PP_SIZE back to original resolution
        for d in dets:
            d["xyxy"][0] *= orig_w / _PP_SIZE
            d["xyxy"][1] *= orig_h / _PP_SIZE
            d["xyxy"][2] *= orig_w / _PP_SIZE
            d["xyxy"][3] *= orig_h / _PP_SIZE

        return dets

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(
        results: Dict[str, Any],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Convert processor output to a list of detection dicts.

        Handles both tensor and plain-list formats returned by various
        versions of the Sam3Processor post-processing API.
        """
        masks = results.get("masks", [])
        scores_raw = results.get("scores", [])
        boxes_raw = results.get("boxes", [])
        labels_raw = results.get("labels", [])

        detections: List[Dict[str, Any]] = []

        n_detections = max(len(scores_raw), len(boxes_raw), len(masks))
        if n_detections == 0:
            return detections

        for i in range(n_detections):
            # --- score ---
            if i < len(scores_raw):
                s = scores_raw[i]
                score = float(s.item() if hasattr(s, "item") else s)
            else:
                score = 0.0

            # --- box ---
            box = None
            if i < len(boxes_raw):
                b = boxes_raw[i]
                if hasattr(b, "tolist"):
                    b = b.tolist()
                elif hasattr(b, "cpu"):
                    b = b.cpu().numpy().tolist()
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    box = np.array(b[:4], dtype=np.float32)

            # Fall back: derive box from mask
            if box is None and i < len(masks):
                mask = masks[i]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                elif not isinstance(mask, np.ndarray):
                    mask = np.asarray(mask)
                ys, xs = np.where(mask > 0)
                if xs.size > 0 and ys.size > 0:
                    box = np.array(
                        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                        dtype=np.float32,
                    )

            if box is None:
                continue

            # Sanity: skip degenerate boxes
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            if bw <= 0 or bh <= 0 or (bw * bh) < MIN_BOX_AREA_PX:
                continue

            # --- label ---
            if i < len(labels_raw):
                lbl = labels_raw[i]
                label = str(lbl.item() if hasattr(lbl, "item") else lbl)
            else:
                label = prompt

            detections.append({
                "xyxy": box,
                "score": score,
                "label": label,
            })

        return detections


# ===========================================================================
# Keyframe sampling helpers
# ===========================================================================

def _sample_keyframes(
    session: InterviewSession,
    progress: Any,
    keyframe_frac: float = DEFAULT_KEYFRAME_FRAC,
    min_spacing: int = DEFAULT_MIN_SPACING,
    embedding_batch: int = DEFAULT_EMBEDDING_BATCH,
) -> List[int]:
    """Compute or reuse sampled keyframes for a session.

    If the session already has ``sampled_frames`` populated (e.g. from a
    previous detection pass or a cached session), those are returned
    directly.  Otherwise, SAM3 frame embeddings are computed and
    change-detection-based keyframe selection is performed.

    Args:
        session:         The interview session (must have video_path set).
        progress:        JobProgress object to update UI.
        keyframe_frac:   Fraction of total frames to sample.
        min_spacing:     Minimum frame gap between keyframes.
        embedding_batch: Batch size for SAM3 embedding computation.

    Returns:
        Sorted list of 0-based frame indices.
    """
    if session.sampled_frames:
        logger.info(
            "Reusing %d previously sampled keyframes", len(session.sampled_frames)
        )
        return session.sampled_frames

    video_path = session.video_path
    if not video_path:
        raise RuntimeError("Session has no video_path set. Call video_info first.")

    progress.step = "Computing frame embeddings for keyframe selection..."
    progress.current = 0

    cache_key = session.cache_key
    embeds = _compute_sam3_frame_embeddings(
        cache_key, video_path, embedding_batch, EMBEDDING_CACHE_DIR
    )

    # Use actual embedding count as ground truth for frame count
    frames_count = embeds.shape[0]
    if session.frames_count and session.frames_count != frames_count:
        logger.warning(
            "Session frames_count=%d but embeddings have %d frames; using min",
            session.frames_count,
            frames_count,
        )
        frames_count = min(session.frames_count, frames_count)
        embeds = embeds[:frames_count]

    progress.step = "Selecting keyframes via change detection..."
    diff = compute_change_scores(embeds)
    smooth = smooth_change_scores(diff, kernel_size=5)
    keyframes = select_keyframes(
        frames_count, keyframe_frac, smooth, min_spacing=min_spacing
    )

    logger.info(
        "Selected %d keyframes from %d total frames (frac=%.3f)",
        len(keyframes), frames_count, keyframe_frac,
    )

    # Persist on session
    with session._lock:
        session.sampled_frames = keyframes
        session.touch()

    return keyframes


# ===========================================================================
# Deduplication helpers
# ===========================================================================

def _deduplicate_against_existing(
    new_boxes: np.ndarray,
    new_scores: np.ndarray,
    existing_boxes: np.ndarray,
    iou_threshold: float = DEFAULT_DEDUP_IOU_THRESHOLD,
) -> np.ndarray:
    """Return indices of ``new_boxes`` that do NOT overlap with existing.

    Args:
        new_boxes:      (N, 4) candidate detections.
        new_scores:     (N,) confidence scores (unused but kept for API symmetry).
        existing_boxes: (M, 4) already-accepted detections.
        iou_threshold:  Max IoU a new box may have with any existing box.

    Returns:
        (K,) int array of surviving indices into ``new_boxes``.
    """
    if len(new_boxes) == 0 or len(existing_boxes) == 0:
        return np.arange(len(new_boxes), dtype=np.int64)

    iou = _compute_iou_matrix(new_boxes, existing_boxes)  # (N, M)
    max_iou_per_new = iou.max(axis=1)  # (N,)
    keep_mask = max_iou_per_new < iou_threshold
    return np.where(keep_mask)[0].astype(np.int64)


# ===========================================================================
# Detection on a single frame
# ===========================================================================

def _detect_single_frame(
    detector: Sam3TextBasedDetector,
    pil_image: Image.Image,
    prompt: str,
    frame_idx: int,
    width: int,
    height: int,
    nms_iou: float = DEFAULT_NMS_IOU_THRESHOLD,
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> List[CropData]:
    """Run detection + NMS + padding on one frame and return CropData list.

    Args:
        detector:   Sam3TextBasedDetector instance.
        pil_image:  PIL Image of the frame.
        prompt:     Text prompt for detection.
        frame_idx:  0-based frame index.
        width:      Video frame width.
        height:     Video frame height.
        nms_iou:    IoU threshold for NMS.
        pad_frac:   Fraction for box padding.

    Returns:
        List of CropData objects for this frame.
    """
    raw_dets = detector.detect(prompt, pil_image)

    if not raw_dets:
        return []

    # Stack into arrays for vectorised NMS / padding
    boxes = np.array([d["xyxy"] for d in raw_dets], dtype=np.float32)
    scores = np.array([d["score"] for d in raw_dets], dtype=np.float32)

    # NMS
    keep_idx = nms_numpy(boxes, scores, iou_threshold=nms_iou)
    if len(keep_idx) == 0:
        return []

    boxes = boxes[keep_idx]
    scores = scores[keep_idx]

    # Padding
    boxes = pad_boxes(boxes, width, height, pad_frac=pad_frac)

    # Build CropData
    crops: List[CropData] = []
    for i in range(len(boxes)):
        crop = CropData(
            crop_id=str(uuid.uuid4())[:12],
            frame_idx=frame_idx,
            xyxy=boxes[i],
            score=float(scores[i]),
            label=CropLabel.PENDING,
            source=CropSource.TEXT_DETECT,
            prompt=prompt,
        )
        crops.append(crop)

    return crops


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run_detection_pipeline(
    session: InterviewSession,
    prompt: str,
    progress: Any,
) -> Dict[str, Any]:
    """Run the full detection pipeline and populate session crops.

    Called from ``routes.detect_start`` via the background job executor.

    Steps:
        1. Sample keyframes (or reuse existing).
        2. For each keyframe, read frame, detect with Sam3Model, NMS, pad.
        3. Store CropData objects on the session.
        4. Advance session phase to DETECTION.
        5. Persist session to disk.

    Args:
        session:  InterviewSession with video_path already set.
        prompt:   Text prompt for detection (e.g. "person").
        progress: JobProgress object (updated for UI polling).

    Returns:
        Summary dict with detection statistics.
    """
    t0 = time.time()

    # Record prompt
    if prompt not in session.prompts:
        session.prompts.append(prompt)

    # Step 1: sample keyframes
    progress.step = "Sampling keyframes..."
    progress.total = 3  # rough phases: sample, detect, finalise
    progress.current = 0

    keyframes = _sample_keyframes(session, progress)
    progress.current = 1

    # Step 2: detect on each keyframe
    progress.step = "Running text-based detection on keyframes..."
    progress.total = len(keyframes) + 2  # +2 for sample + finalise
    progress.current = 1

    detector = Sam3TextBasedDetector()
    width = session.width
    height = session.height
    total_crops = 0

    for i, frame_idx in enumerate(keyframes):
        progress.step = f"Detecting on frame {frame_idx} ({i + 1}/{len(keyframes)})..."
        progress.current = i + 2  # offset by 1 for sampling phase

        pil_image = _read_frame_cached_or_pyav(
            session.video_path, frame_idx, cache_key=session.cache_key,
        )
        if pil_image is None:
            logger.warning("Failed to read frame %d, skipping", frame_idx)
            continue

        crops = _detect_single_frame(
            detector, pil_image, prompt, frame_idx, width, height,
        )

        with session._lock:
            for crop in crops:
                session.add_crop(crop)
                total_crops += 1

    # Release GPU memory
    detector.clear_cache()

    # Step 3: finalise
    progress.step = "Saving session..."
    progress.current = progress.total - 1

    with session._lock:
        session.advance_to(Phase.DETECTION)

    save_session(session)

    elapsed = time.time() - t0
    progress.step = "Detection complete."
    progress.current = progress.total

    summary = {
        "keyframes": len(keyframes),
        "total_crops": total_crops,
        "prompt": prompt,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(
        "Detection pipeline finished: %d crops on %d keyframes in %.1fs",
        total_crops, len(keyframes), elapsed,
    )
    return summary


# ===========================================================================
# Round-based active learning detection
# ===========================================================================

def run_round_detection(
    session: InterviewSession,
    prompt: str,
    progress: Any,
    round_num: int = 1,
) -> Dict[str, Any]:
    """Run detection for one active learning round.

    1. Select frames via temporal stratification (excludes prior rounds)
    2. Batch-decode and batch-detect
    3. Store crops on session (NO auto-scoring — MLP trains at round boundary)
    4. Record round state

    Args:
        session: The interview session.
        prompt: Text prompt for SAM3 detection.
        progress: Progress reporting object with step/current/total.
        round_num: Which round (1-indexed).

    Returns:
        Summary dict with round, keyframes, total_crops, prompt, elapsed.
    """
    import time as _time
    t0 = _time.time()

    _log_rss("run_round_detection START")

    if prompt not in session.prompts:
        session.prompts.append(prompt)

    # Step 1: Select frames
    progress.step = f"Round {round_num}: Selecting frames..."
    progress.total = 3
    progress.current = 0

    frame_indices = select_round_frames(session, round_num)
    if not frame_indices:
        raise RuntimeError(f"No frames available for round {round_num}")
    progress.current = 1

    # Step 2: Decode frames (single sequential PyAV pass)
    progress.step = f"Round {round_num}: Decoding {len(frame_indices)} frames..."
    frame_images = _decode_frames_sequential(
        session.video_path, frame_indices, cache_key=session.cache_key,
    )
    if not frame_images:
        raise RuntimeError(f"Failed to decode frames for round {round_num}")
    progress.current = 2
    _log_rss("after frame decode")

    # Step 3: Batch detect
    progress.step = f"Round {round_num}: Detecting on {len(frame_images)} frames..."
    detector = Sam3TextBasedDetector()
    _log_rss("after Sam3TextBasedDetector init")
    crops = _detect_batch(
        detector, frame_images, prompt,
        session.width, session.height,
        batch_size=DEFAULT_DETECT_BATCH,
    )
    detector.clear_cache()
    _log_rss("after detection + clear_cache")
    progress.current = 3

    # Step 4: Store crops (no auto-scoring)
    total_crops = 0
    with session._lock:
        session.round_frames[round_num] = sorted(frame_images.keys())
        session.sampled_frames = sorted(
            set(session.sampled_frames) | set(frame_images.keys())
        )
        session.current_round = round_num

        for crop in crops:
            session.add_crop(crop)
            total_crops += 1

        if round_num == 1:
            session.advance_to(Phase.DETECTION)

    # Round 1: also detect on held-out validation frames
    val_crops_count = 0
    if round_num == 1:
        val_frame_indices = select_validation_frames(session)
        if val_frame_indices:
            progress.step = f"Round 1: Detecting on {len(val_frame_indices)} validation frames..."
            val_frame_images = _decode_frames_sequential(
                session.video_path, val_frame_indices, cache_key=session.cache_key,
            )
            if val_frame_images:
                val_detector = Sam3TextBasedDetector()
                val_crops = _detect_batch(
                    val_detector, val_frame_images, prompt,
                    session.width, session.height,
                    batch_size=DEFAULT_DETECT_BATCH,
                )
                val_detector.clear_cache()
                del val_frame_images

                with session._lock:
                    session.validation_frames = sorted(val_frame_indices)
                    session.sampled_frames = sorted(
                        set(session.sampled_frames) | set(val_frame_indices)
                    )
                    for crop in val_crops:
                        session.add_crop(crop)
                        val_crops_count += 1
                        total_crops += 1

            logger.info("Round 1 validation: %d crops on %d frames",
                        val_crops_count, len(val_frame_indices))

    # Free GPU memory after detection
    del frame_images
    gc.collect()
    torch.cuda.empty_cache()
    _log_rss("run_round_detection END (after cleanup)")

    elapsed = _time.time() - t0
    progress.step = f"Round {round_num} complete."

    round_info = {
        "round": round_num,
        "frames": len(frame_indices),
        "new_crops": total_crops,
        "elapsed_seconds": round(elapsed, 1),
    }

    with session._lock:
        session.round_history.append(round_info)

    save_session(session)

    summary = {
        "round": round_num,
        "keyframes": len(frame_indices),
        "total_crops": total_crops,
        "prompt": prompt,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info("Round %d: %d crops on %d frames in %.1fs",
                round_num, total_crops, len(frame_indices), elapsed)
    return summary


# ===========================================================================
# Background embedding + change detection
# ===========================================================================

def run_embedding_background(
    session: InterviewSession,
    progress: Any,
) -> Dict[str, Any]:
    """Compute change-detected keyframes via lightweight or SAM3 pipeline.

    Mode is controlled by ``INTERVIEW_EMBEDDING_MODE`` env var:
    - ``"lightweight"`` (default): CPU-based pixel diff + HSV histogram.
      Completes in ~5-15s for a typical video.  No GPU needed.
    - ``"sam3"``: GPU-batched SAM3 image embeddings (original pipeline).
      Falls back for cases where neural features are needed.

    Both modes produce the same output contract: ``session.change_keyframes``
    is a list of original (0-based) frame indices where scene changes occur.

    Args:
        session:  InterviewSession with video_path already set.
        progress: JobProgress object (with _pause_event for pause/resume).

    Returns:
        Summary dict with embedding stats.
    """
    t0 = time.time()

    video_path = session.video_path
    if not video_path:
        raise RuntimeError("Session has no video_path set.")

    pause_event = getattr(progress, '_pause_event', None)

    def _progress_cb(current: int, total: int):
        progress.step = f"Analyzing frames {current:,} / {total:,}..."
        progress.current = current
        progress.total = total

    if EMBEDDING_MODE == "lightweight":
        progress.step = "Analyzing video for scene changes..."
        scores, sampled_indices = compute_lightweight_change_from_video(
            video_path,
            target_fps=EMBEDDING_TARGET_FPS,
            pause_event=pause_event,
            progress_callback=_progress_cb,
            cache_key=session.cache_key,
        )
        frames_count = len(scores)
        smooth = smooth_change_scores(scores, kernel_size=5)
    else:
        def _change_cb(change_keyframes: list):
            with session._lock:
                session.change_keyframes = change_keyframes
                session.touch()

        progress.step = "Computing frame embeddings..."
        embeds, sampled_indices = _do_embed_all_frames(
            video_path, DEFAULT_EMBEDDING_BATCH,
            progress_callback=_progress_cb,
            target_fps=EMBEDDING_TARGET_FPS,
            pause_event=pause_event,
            change_callback=_change_cb,
        )
        frames_count = embeds.shape[0]
        diff = compute_change_scores(embeds)
        smooth = smooth_change_scores(diff, kernel_size=5)

    # Common: keyframe selection
    progress.step = "Selecting keyframes..."
    change_keyframes_sub = select_keyframes(
        frames_count, DEFAULT_KEYFRAME_FRAC, smooth,
        min_spacing=DEFAULT_MIN_SPACING,
    )
    change_keyframes = [
        sampled_indices[k] for k in change_keyframes_sub
        if k < len(sampled_indices)
    ]

    # Store on session
    with session._lock:
        session.embedding_complete = True
        session.change_keyframes = change_keyframes
        session.embedding_sampled_indices = sampled_indices
        session.touch()
        save_session(session)

    elapsed = time.time() - t0
    progress.step = "Analysis complete."
    progress.current = progress.total

    summary = {
        "frames_embedded": int(frames_count),
        "frames_total_in_video": sampled_indices[-1] + 1 if sampled_indices else 0,
        "change_keyframes": len(change_keyframes),
        "elapsed_seconds": round(elapsed, 1),
        "mode": EMBEDDING_MODE,
    }
    logger.info(
        "Background analysis (%s): %d frames, %d change keyframes in %.1fs",
        EMBEDDING_MODE, frames_count, len(change_keyframes), elapsed,
    )
    return summary


# ===========================================================================
# Recall strategy: additional detection passes
# ===========================================================================

def _run_multi_prompt_strategy(
    session: InterviewSession,
    extra_prompts: List[str],
    progress: Any,
) -> Dict[str, Any]:
    """Run detector with additional text prompts on already-sampled frames.

    Deduplicates new detections against existing crops on each frame using
    IoU to avoid redundant proposals.

    Args:
        session:       InterviewSession (must already be in DETECTION phase).
        extra_prompts: Additional text prompts to try.
        progress:      JobProgress object.

    Returns:
        Summary dict.
    """
    keyframes = session.sampled_frames
    if not keyframes:
        raise RuntimeError("No sampled frames available. Run detection first.")

    width = session.width
    height = session.height
    detector = Sam3TextBasedDetector()

    total_new = 0
    total_deduped = 0

    progress.total = len(keyframes) * len(extra_prompts)
    progress.current = 0
    step_counter = 0

    for prompt in extra_prompts:
        if prompt not in session.prompts:
            session.prompts.append(prompt)

        for frame_idx in keyframes:
            step_counter += 1
            progress.step = (
                f"Multi-prompt '{prompt}' on frame {frame_idx} "
                f"({step_counter}/{progress.total})..."
            )
            progress.current = step_counter

            pil_image = _read_frame_cached_or_pyav(
                session.video_path, frame_idx, cache_key=session.cache_key,
            )
            if pil_image is None:
                continue

            new_crops = _detect_single_frame(
                detector, pil_image, prompt, frame_idx, width, height,
            )

            if not new_crops:
                continue

            # Deduplicate against existing crops on this frame
            existing_on_frame = session.get_crops_by_frame(frame_idx)
            if existing_on_frame:
                existing_boxes = np.array(
                    [c.xyxy for c in existing_on_frame], dtype=np.float32
                )
                new_boxes = np.array(
                    [c.xyxy for c in new_crops], dtype=np.float32
                )
                new_scores = np.array(
                    [c.score for c in new_crops], dtype=np.float32
                )
                keep_idx = _deduplicate_against_existing(
                    new_boxes, new_scores, existing_boxes,
                    iou_threshold=DEFAULT_DEDUP_IOU_THRESHOLD,
                )
                deduped = len(new_crops) - len(keep_idx)
                total_deduped += deduped
                new_crops = [new_crops[i] for i in keep_idx]

            # Tag source as multi-prompt
            for crop in new_crops:
                crop.source = CropSource.MULTI_PROMPT

            with session._lock:
                for crop in new_crops:
                    session.add_crop(crop)
                    total_new += 1

    detector.clear_cache()
    save_session(session)

    summary = {
        "strategy": "multi_prompt",
        "extra_prompts": extra_prompts,
        "new_crops": total_new,
        "deduplicated": total_deduped,
    }
    logger.info(
        "Multi-prompt recall: %d new crops (%d deduplicated)",
        total_new, total_deduped,
    )
    return summary


def run_recall_strategy(
    session: InterviewSession,
    strategy: str,
    extra_prompts: List[str],
    progress: Any,
) -> Dict[str, Any]:
    """Dispatch to the requested recall-gap strategy.

    Called from ``routes.detect_recall_strategy`` via background executor.

    Supported strategies:
        - ``"multi_prompt"``: Run detector with additional text prompts on
          already-sampled frames; deduplicate against existing crops.

    Args:
        session:       InterviewSession.
        strategy:      ``"multi_prompt"``.
        extra_prompts: Additional text prompts (used by multi_prompt).
        progress:      JobProgress object.

    Returns:
        Summary dict from the selected strategy.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    t0 = time.time()

    if strategy == "multi_prompt":
        if not extra_prompts:
            raise ValueError(
                "multi_prompt strategy requires at least one prompt in 'prompts'."
            )
        result = _run_multi_prompt_strategy(session, extra_prompts, progress)
    else:
        raise ValueError(
            f"Unknown recall strategy: {strategy!r}. "
            f"Supported: 'multi_prompt'."
        )

    elapsed = time.time() - t0
    result["elapsed_seconds"] = round(elapsed, 1)
    progress.step = f"Recall strategy '{strategy}' complete."

    logger.info(
        "Recall strategy '%s' finished in %.1fs: %s",
        strategy, elapsed, result,
    )
    return result
