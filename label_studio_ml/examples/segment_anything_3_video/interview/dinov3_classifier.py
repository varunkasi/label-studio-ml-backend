"""DINOv3 feature extractor for the Interview UI.

Provides lazy-loaded DINOv3 ViT-L backbone, CLS-token feature extraction
(crop + context patch), spatial metadata computation, and feature caching.
The quality gate classifier lives in ``knn_classifier.py``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .background import JobProgress
from .mask_utils import compute_mask_quality  # noqa: F401 – re-export
from .state import CropData, CropLabel, InterviewSession

logger = logging.getLogger(__name__)

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# DINOv3 singleton
# ---------------------------------------------------------------------------

_dinov3_model = None
_dinov3_processor = None


def _get_dinov3():
    """Lazy-load DINOv3 ViT-L backbone (frozen, bfloat16).

    Configurable via ``DINOV3_MODEL`` env var.  Defaults to
    ``facebook/dinov3-vitl16-pretrain-lvd1689m`` (1024-dim CLS tokens).
    """
    global _dinov3_model, _dinov3_processor
    if _dinov3_model is None:
        from transformers import AutoImageProcessor, AutoModel
        model_name = os.getenv("DINOV3_MODEL", "facebook/dinov3-vitl16-pretrain-lvd1689m")
        logger.info("Loading DINOv3 backbone from %s ...", model_name)
        _dinov3_model = (
            AutoModel.from_pretrained(model_name)
            .to(DEVICE, dtype=DTYPE)
            .eval()
        )
        for p in _dinov3_model.parameters():
            p.requires_grad = False
        _dinov3_processor = AutoImageProcessor.from_pretrained(model_name)
        logger.info("DINOv3 loaded on %s (%s)", DEVICE, DTYPE)
    return _dinov3_model, _dinov3_processor


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(crops: List[Image.Image], batch_size: int = 64) -> np.ndarray:
    """Extract DINOv3 CLS-token features from crop images.

    Returns: (N, 1024) float32 array, L2-normalized.
    Uses batch_size=64 and bfloat16 autocast for GPU utilization.
    Falls back to half batch on OOM.
    """
    if not crops:
        return np.empty((0, 1024), dtype=np.float32)

    model, processor = _get_dinov3()
    all_features: List[np.ndarray] = []

    for start in range(0, len(crops), batch_size):
        batch_imgs = crops[start : start + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE, dtype=DTYPE) for k, v in inputs.items()}

        try:
            with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
                outputs = model(**inputs)
        except torch.cuda.OutOfMemoryError:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            # Retry with half batch size
            half = max(1, len(batch_imgs) // 2)
            logger.warning("DINOv3 OOM at batch_size=%d, retrying with %d", len(batch_imgs), half)
            for sub_start in range(0, len(batch_imgs), half):
                sub_imgs = batch_imgs[sub_start : sub_start + half]
                sub_inputs = processor(images=sub_imgs, return_tensors="pt")
                sub_inputs = {k: v.to(DEVICE, dtype=DTYPE) for k, v in sub_inputs.items()}
                with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
                    sub_out = model(**sub_inputs)
                cls_tokens = sub_out.last_hidden_state[:, 0, :].float().cpu().numpy()
                norms = np.maximum(np.linalg.norm(cls_tokens, axis=1, keepdims=True), 1e-8)
                all_features.append(cls_tokens / norms)
            continue

        cls_tokens = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        norms = np.maximum(np.linalg.norm(cls_tokens, axis=1, keepdims=True), 1e-8)
        all_features.append(cls_tokens / norms)

    return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# Context patch feature extraction
# ---------------------------------------------------------------------------

def extract_context_features(
    crops: List[CropData],
    frames: Dict[int, Image.Image],
    batch_size: int = 64,
) -> np.ndarray:
    """Extract DINOv3 CLS features from 50%-expanded crop regions.

    For each crop, expands the bounding box by 50% of max(w, h) in each
    direction, clamps to frame boundaries, extracts the region, and runs
    it through DINOv3.

    Args:
        crops: List of CropData with xyxy pixel coords and frame_idx.
        frames: Mapping of frame_idx -> PIL Image.
        batch_size: DINOv3 inference batch size.

    Returns:
        (N, 1024) L2-normalized float32 array. Rows corresponding to
        crops whose frames are missing are zero-vectors.
    """
    context_imgs: List[Image.Image] = []
    valid_indices: List[int] = []

    for i, crop in enumerate(crops):
        pil_frame = frames.get(crop.frame_idx)
        if pil_frame is None:
            continue

        x1, y1, x2, y2 = crop.xyxy.astype(float)
        bw, bh = x2 - x1, y2 - y1
        expand = 0.5 * max(bw, bh)

        cx1 = max(0, int(x1 - expand))
        cy1 = max(0, int(y1 - expand))
        cx2 = min(pil_frame.width, int(x2 + expand))
        cy2 = min(pil_frame.height, int(y2 + expand))

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        context_imgs.append(pil_frame.crop((cx1, cy1, cx2, cy2)))
        valid_indices.append(i)

    result = np.zeros((len(crops), 1024), dtype=np.float32)
    if context_imgs:
        feats = extract_features(context_imgs, batch_size=batch_size)
        for j, idx in enumerate(valid_indices):
            result[idx] = feats[j]

    return result


# ---------------------------------------------------------------------------
# Crop metadata
# ---------------------------------------------------------------------------

def compute_crop_metadata(
    xyxy: np.ndarray, frame_width: int, frame_height: int
) -> np.ndarray:
    """Compute normalized crop metadata: [cx, cy, scale, aspect_ratio].

    - cx, cy: center position normalized to [0, 1]
    - scale: sqrt(area) / sqrt(frame_area)
    - aspect_ratio: width / height
    """
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0 / frame_width
    cy = (y1 + y2) / 2.0 / frame_height
    scale = math.sqrt(bw * bh) / math.sqrt(frame_width * frame_height)
    return np.array([cx, cy, scale, bw / bh], dtype=np.float32)




# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_crop_features(
    session: InterviewSession,
    crop_ids: List[str],
    progress: Optional[JobProgress] = None,
) -> None:
    """Extract and cache DINOv3 features for crops that don't have them yet.

    Uses the shared LRU frame cache first, then batch-decodes remaining
    frames via ``_decode_frames_sequential`` (single PyAV container open).
    This replaces the old ``_read_single_frame`` which opened/closed a
    container per frame and had a broken seek that returned None for high
    frame indices.
    """
    from collections import defaultdict
    from .frame_cache import read_frame_cached

    _ck = getattr(session, "cache_key", None)

    # Helper: decode frames from LRU cache + batch fallback
    def _decode_needed_frames(frame_idxs: List[int]) -> Dict[int, Image.Image]:
        images: Dict[int, Image.Image] = {}
        uncached: List[int] = []
        for fidx in sorted(set(frame_idxs)):
            cached = read_frame_cached(session.video_path, fidx, cache_key=_ck)
            if cached is not None:
                images[fidx] = cached
            else:
                uncached.append(fidx)
        if uncached:
            from .detection import _decode_frames_sequential
            decoded = _decode_frames_sequential(session.video_path, uncached, cache_key=_ck)
            images.update(decoded)
            from .frame_cache import put_cached_frame
            for fidx, img in decoded.items():
                put_cached_frame(session.video_path, fidx, img)
        return images

    # --- Phase 1: Extract DINOv3 crop features for crops that lack them ---
    missing = [cid for cid in crop_ids if session.crops[cid].features is None]
    if missing:
        if progress:
            progress.step = f"Extracting DINOv3 features for {len(missing)} crops"

        frame_to_cids: Dict[int, List[str]] = defaultdict(list)
        for cid in missing:
            frame_to_cids[session.crops[cid].frame_idx].append(cid)

        frame_images = _decode_needed_frames(list(frame_to_cids.keys()))

        processed = 0
        for frame_idx in sorted(frame_to_cids.keys()):
            cids = frame_to_cids[frame_idx]
            pil_frame = frame_images.get(frame_idx)
            if pil_frame is None:
                logger.warning("Could not decode frame %d", frame_idx)
                continue

            crop_imgs: List[Image.Image] = []
            crop_id_order: List[str] = []
            for cid in cids:
                crop = session.crops[cid]
                x1, y1, x2, y2 = crop.xyxy.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_frame.width, x2), min(pil_frame.height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_imgs.append(pil_frame.crop((x1, y1, x2, y2)))
                crop_id_order.append(cid)

            if not crop_imgs:
                continue
            feats = extract_features(crop_imgs)
            for i, cid in enumerate(crop_id_order):
                c = session.crops[cid]
                c.features = feats[i]
                c.metadata = compute_crop_metadata(c.xyxy, session.width, session.height)
            processed += len(crop_id_order)
            if progress:
                progress.current = processed
                progress.total = len(missing)

    # --- Phase 2: Extract context features for crops that have DINOv3 but no context ---
    ctx_missing = [
        session.crops[cid] for cid in crop_ids
        if session.crops[cid].features is not None
        and session.crops[cid].context_features is None
    ]
    if ctx_missing:
        if progress:
            progress.step = f"Extracting context features for {len(ctx_missing)} crops"
        ctx_frame_idxs = [c.frame_idx for c in ctx_missing]
        ctx_frames = _decode_needed_frames(ctx_frame_idxs)
        if ctx_frames:
            ctx_feats = extract_context_features(ctx_missing, ctx_frames)
            for i, crop in enumerate(ctx_missing):
                if np.any(ctx_feats[i] != 0):
                    crop.context_features = ctx_feats[i]




