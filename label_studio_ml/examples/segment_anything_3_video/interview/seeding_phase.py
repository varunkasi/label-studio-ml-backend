"""Dense seed generation and Label Studio upload for the Interview UI.

Implements Phase 3 of the interview workflow: after the user has labeled
crops in the detection phase and resolved identity clusters (ReID phase),
this module scans every Nth frame of the video, detects candidate bounding
boxes, scores them with a distance-weighted k-NN quality gate, assigns
identities via nearest ReID cluster centroid, and uploads the results to
Label Studio as videorectangle regions with ``enabled=false`` keyframes.

Functions:
    generate_seeds  -- run the full detection+k-NN scoring+identity pipeline
    upload_seeds    -- push seed regions to Label Studio as a prediction
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeding_common import (
    _build_ls_client, _build_prediction, _upload_prediction,
    _read_frame_pyav, xyxy_to_percent, DEVICE, DTYPE,
)

from .state import CropData, CropLabel, InterviewSession, Phase
from .cache_manager import save_session
from .background import JobProgress
from .dinov3_classifier import extract_features
from .knn_classifier import build_support_set, score_crops

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_cluster_centroids(
    session: InterviewSession,
) -> Dict[int, np.ndarray]:
    """Compute mean DINOv3 feature vector per ReID cluster.

    Iterates over ``session.reid_clusters`` (mapping of cluster_id to list of
    crop_ids), collects the DINOv3 feature vector stored on each
    :class:`CropData`, and averages them to produce a single centroid per
    identity cluster.

    Args:
        session: The current interview session with populated ``reid_clusters``
            and ``crops`` (each crop should have a ``.features`` array).

    Returns:
        Dictionary mapping cluster_id (int) to the L2-normalised centroid
        vector of shape ``(feat_dim,)`` (typically 1024 for DINOv3).
    """
    centroids: Dict[int, np.ndarray] = {}

    for cluster_id, crop_ids in session.reid_clusters.items():
        feature_vectors: List[np.ndarray] = []
        for cid in crop_ids:
            crop = session.get_crop(cid)
            if crop is not None and crop.features is not None:
                feature_vectors.append(crop.features.astype(np.float32))

        if not feature_vectors:
            logger.warning(
                "Cluster %d has no crops with features; skipping centroid",
                cluster_id,
            )
            continue

        centroid = np.mean(feature_vectors, axis=0)
        # L2-normalise so cosine similarity reduces to a dot product
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cluster_id] = centroid

    logger.info(
        "Computed centroids for %d / %d clusters",
        len(centroids),
        len(session.reid_clusters),
    )
    return centroids


def _assign_identity(
    features: np.ndarray,
    centroids: Dict[int, np.ndarray],
) -> Tuple[int, float]:
    """Assign a detection to the nearest ReID cluster centroid.

    Computes cosine similarity between the candidate feature vector and every
    cluster centroid, returning the cluster with the highest similarity.

    Args:
        features: DINOv3 CLS-token vector for the candidate crop, shape
            ``(feat_dim,)``.
        centroids: Mapping of cluster_id to L2-normalised centroid vector (as
            returned by :func:`_compute_cluster_centroids`).

    Returns:
        Tuple of ``(cluster_id, similarity)`` for the best-matching cluster.
        If *centroids* is empty, returns ``(-1, 0.0)``.
    """
    if not centroids:
        return -1, 0.0

    # Normalise the query vector for cosine similarity
    feat = features.astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm

    best_id = -1
    best_sim = -1.0

    for cluster_id, centroid in centroids.items():
        sim = float(np.dot(feat, centroid))
        if sim > best_sim:
            best_sim = sim
            best_id = cluster_id

    return best_id, best_sim


# ---------------------------------------------------------------------------
# Dual-proposer configuration (Path B refinement)
# ---------------------------------------------------------------------------

_REFINE_THRESHOLD = float(os.getenv("INTERVIEW_REFINE_THRESHOLD", "0.3"))
_ENABLE_REFINEMENT = os.getenv("INTERVIEW_ENABLE_REFINEMENT", "true").lower() == "true"
_SEED_CHUNK_SIZE = int(os.getenv("INTERVIEW_SEED_CHUNK_SIZE", "100"))
_SEED_DETECT_BATCH = int(os.getenv("INTERVIEW_SEED_DETECT_BATCH", "8"))


def _infer_reject_reasons(
    query_features: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    support_reasons: List[Optional[str]],
) -> List[Optional[str]]:
    """Infer the dominant reject reason per query from reject supports.

    Uses inverse-square cosine-distance voting over reject-labeled support
    examples and returns the reason with the largest total vote.
    """
    if query_features.shape[0] == 0:
        return []

    reject_idx = np.where(support_labels == 0.0)[0]
    if reject_idx.size == 0:
        return [None] * query_features.shape[0]

    support_reject = support_features[reject_idx]
    reasons_reject = [support_reasons[i] for i in reject_idx]

    eps = 1e-6
    q_norm = query_features / np.maximum(
        np.linalg.norm(query_features, axis=1, keepdims=True), eps,
    )
    s_norm = support_reject / np.maximum(
        np.linalg.norm(support_reject, axis=1, keepdims=True), eps,
    )

    sim = q_norm @ s_norm.T
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    weights = 1.0 / np.maximum(dist ** 2, eps)

    inferred: List[Optional[str]] = []
    for qi in range(weights.shape[0]):
        reason_votes: Dict[Optional[str], float] = {}
        for si, reason in enumerate(reasons_reject):
            reason_votes[reason] = reason_votes.get(reason, 0.0) + float(weights[qi, si])
        inferred.append(max(reason_votes.items(), key=lambda kv: kv[1])[0] if reason_votes else None)
    return inferred


def _get_sam3_image_model():
    """Import and return the Sam3Model singleton from seeding_common."""
    from seeding_common import _get_sam3_image_model as _get
    return _get()


def _refine_candidates_sam3(
    frames: Dict[int, Any],
    candidates: List[Tuple[int, np.ndarray, float]],
    prompt: str = "person",
    expand_frac: float = 0.2,
    tags: Optional[List[str]] = None,
) -> List[Tuple]:
    """Refine candidate boxes using Sam3Model with text+box prompts.

    For each candidate, expands the box by *expand_frac* on each side,
    runs Sam3Model with combined text + box prompt, and extracts the tight
    bounding box from the best mask.

    Args:
        frames:      Mapping of frame_idx -> decoded PIL Image.
        candidates:  List of (frame_idx, box_xyxy, det_score).
        prompt:      Text prompt for Sam3Model (e.g., "person").
        expand_frac: Fraction to expand each side of the box.
        tags:        Optional per-candidate source tags.  When provided,
                     output tuples include a 4th element with the tag.

    Returns:
        List of (frame_idx, refined_box_xyxy, det_score[, tag]) for successful
        refinements.
    """
    import torch
    model, processor = _get_sam3_image_model()

    refined: List[Tuple] = []

    for ci, (frame_idx, box, det_score) in enumerate(candidates):
        pil_frame = frames.get(frame_idx)
        if pil_frame is None:
            continue

        w, h = pil_frame.size
        x1, y1, x2, y2 = box

        # Expand box
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * expand_frac, bh * expand_frac
        ex1 = max(0, int(x1 - dx))
        ey1 = max(0, int(y1 - dy))
        ex2 = min(w, int(x2 + dx))
        ey2 = min(h, int(y2 + dy))

        if ex2 <= ex1 or ey2 <= ey1:
            continue

        try:
            inputs = processor(
                images=pil_frame,
                text=prompt,
                input_boxes=[[[ex1, ey1, ex2, ey2]]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(DEVICE)

            with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
                outputs = model(**inputs)

            target_sizes = inputs.get("original_sizes")
            if target_sizes is not None:
                if hasattr(target_sizes, "tolist"):
                    target_sizes = target_sizes.tolist()
                # else already a plain list
            else:
                target_sizes = [[h, w]]

            # Try the high-level post-processor first; fall back to raw
            # outputs if it hits the "Boolean value of Tensor" bug in
            # some SAM3 processor versions.
            tight = None
            try:
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=target_sizes,
                )[0]

                masks = results.get("masks", [])
                scores = results.get("scores", [])
                boxes_out = results.get("boxes", [])

                n_masks = len(masks) if hasattr(masks, "__len__") else 0
                n_boxes = len(boxes_out) if hasattr(boxes_out, "__len__") else 0
                n_scores = len(scores) if hasattr(scores, "__len__") else 0

                n_results = max(n_masks, n_boxes)
                if n_results > 0:
                    best_idx = int(np.argmax([
                        s.item() if hasattr(s, "item") else float(s) for s in scores
                    ])) if n_scores > 0 else 0

                    if best_idx < n_boxes:
                        b = boxes_out[best_idx]
                        tight = np.array(b.tolist() if hasattr(b, "tolist") else list(b), dtype=np.float32)
                    elif best_idx < n_masks:
                        mask = masks[best_idx]
                        if hasattr(mask, "cpu"):
                            mask = mask.cpu().numpy()
                        ys, xs = np.where(mask > 0)
                        if xs.size > 0:
                            tight = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

            except (RuntimeError, ValueError):
                # Fallback: process raw pred_masks + iou_scores directly
                pred_masks = getattr(outputs, "pred_masks", None)
                iou_scores = getattr(outputs, "iou_scores", None)
                if pred_masks is not None:
                    # pred_masks: (batch, n_masks, H_pred, W_pred) — logits
                    pm = pred_masks[0]  # first (only) batch item
                    if iou_scores is not None:
                        best_idx = int(iou_scores[0].argmax().item())
                    else:
                        best_idx = 0
                    if best_idx < pm.shape[0]:
                        mask_logits = pm[best_idx]  # (H_pred, W_pred)
                        # Resize to original and threshold
                        mask_bool = (mask_logits > 0).float()
                        # Resize to original image dimensions
                        th, tw = target_sizes[0]
                        mask_resized = torch.nn.functional.interpolate(
                            mask_bool.unsqueeze(0).unsqueeze(0),
                            size=(th, tw),
                            mode="nearest",
                        )[0, 0]
                        mask_np = mask_resized.cpu().numpy().astype(bool)
                        ys, xs = np.where(mask_np)
                        if xs.size > 0:
                            tight = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
                            logger.debug("Fallback raw mask extraction succeeded for frame %d", frame_idx)

            if tight is not None:
                entry = (frame_idx, tight, det_score)
                if tags is not None and ci < len(tags):
                    entry = (frame_idx, tight, det_score, tags[ci])
                refined.append(entry)

        except Exception as exc:
            logger.warning("Refinement failed for frame %d: %s", frame_idx, exc)
            continue

    return refined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _score_and_accept_seed(
    box: np.ndarray,
    pil_frame,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    support_reasons: list,
    centroids: Dict[int, np.ndarray],
    threshold: float,
    source: str,
    frame_idx: int,
    mask: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Crop, extract features, score with k-NN, and return seed dict if accepted.

    Shared helper for Path A and refinement. Uses 2056-dim input:
    [DINOv3 crop CLS (1024) + context CLS (1024) + spatial (4) + mask quality (4)].

    Returns None if the crop fails to pass the k-NN confidence threshold.
    """
    from .dinov3_classifier import compute_crop_metadata, compute_mask_quality

    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(pil_frame.width, x2), min(pil_frame.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop_img = pil_frame.crop((x1, y1, x2, y2))
    feat = extract_features([crop_img])  # (1, 1024)
    meta = compute_crop_metadata(box, pil_frame.width, pil_frame.height)

    if mask is not None:
        mq = compute_mask_quality(mask, box, pil_frame.width, pil_frame.height)
    else:
        mq = np.zeros(4, dtype=np.float32)

    # Context features: expand box by 50% of max(w, h), extract DINOv3
    bw, bh = float(x2 - x1), float(y2 - y1)
    expand = 0.5 * max(bw, bh)
    cx1 = max(0, int(x1 - expand))
    cy1 = max(0, int(y1 - expand))
    cx2 = min(pil_frame.width, int(x2 + expand))
    cy2 = min(pil_frame.height, int(y2 + expand))
    if cx2 > cx1 and cy2 > cy1:
        ctx_img = pil_frame.crop((cx1, cy1, cx2, cy2))
        ctx_feat = extract_features([ctx_img])  # (1, 1024)
    else:
        ctx_feat = np.zeros((1, 1024), dtype=np.float32)

    # Build 2056-dim query: [crop CLS | context CLS | spatial | mask quality]
    query = np.concatenate([feat, ctx_feat, meta.reshape(1, -1), mq.reshape(1, -1)], axis=1)

    confidences, _ = score_crops(
        query, support_features, support_labels, support_reasons,
    )
    conf = float(confidences[0])

    if conf < threshold:
        return None

    identity, identity_sim = _assign_identity(feat[0], centroids)
    return {
        "frame_idx": int(frame_idx),
        "xyxy": box.tolist(),
        "confidence": round(conf, 4),
        "identity": int(identity),
        "identity_similarity": round(float(identity_sim), 4),
        "source": source,
    }


def generate_seeds(
    session: InterviewSession,
    progress: JobProgress,
) -> Dict[str, Any]:
    """Generate dense seeds: multi-prompt SAM3 detection + k-NN quality gate.

    For every Nth frame across the entire video:
      1. Run SAM3 text detection with ALL prompts accumulated during rounds 1-4
      2. NMS + pad boxes across all prompts (cross-prompt dedup)
      3. For each candidate: extract DINOv3 crop + context features + metadata
      4. k-NN quality gate (2056-dim: crop CLS + context CLS + spatial + mask quality)
      5. Assign identity via nearest ReID centroid
      6. Accept if k-NN confidence >= threshold

    This is fully automatic — no human interaction needed for 30K+ frames.

    Results are stored in ``session.seeds`` as a list of dicts::

        {
            "frame_idx": int,
            "xyxy": [x1, y1, x2, y2],
            "confidence": float,
            "identity": int,
            "identity_similarity": float,
            "source": str,   # "multi_prompt_knn", "refined_medium", or "refined_oversized"
        }

    Args:
        session: The interview session (must be at REID phase or later,
            with labeled crops for k-NN support set).
        progress: :class:`JobProgress` handle for reporting status to the
            frontend polling loop.

    Returns:
        Summary dict with ``total_seeds``, ``frames_scanned``,
        ``identities``, and ``prompts_used``.

    Raises:
        RuntimeError: If no labeled crops exist or no ReID clusters are defined.
    """
    from .detection import (
        Sam3TextBasedDetector, nms_numpy, pad_boxes,
        _decode_frames_sequential, _detect_batch,
        precompute_text_tokens,
    )
    from .dinov3_classifier import (
        compute_crop_metadata, compute_mask_quality,
    )
    from seeding_common import _get_video_info_pyav

    import torch

    # ---- Validate prerequisites ----
    progress.step = "Validating session state..."
    progress.current = 0

    # Build k-NN support set from labeled crops
    support_feats, support_labels, _, support_reasons = build_support_set(session)
    if support_feats.shape[0] == 0:
        raise RuntimeError(
            "No labeled crops with features found. Complete the detection phase first."
        )
    if not session.reid_clusters:
        raise RuntimeError(
            "No ReID clusters found. Complete the ReID phase first."
        )

    # ---- Load models ----
    progress.step = "Loading models..."
    detector = Sam3TextBasedDetector()

    centroids = _compute_cluster_centroids(session)
    prompts = session.prompts if session.prompts else ["person"]

    # ---- Determine target frames ----
    change = set(session.change_keyframes) if session.embedding_complete else set()

    # Determine candidate frame pool from disk cache
    _disk_cache_meta = None
    try:
        from .disk_frame_cache import frame_cache_exists as _fce, get_frame_cache_meta as _gfcm
        if _fce(session.cache_key):
            _disk_cache_meta = _gfcm(session.cache_key)
    except ImportError:
        pass

    if _disk_cache_meta and "sampled_indices" in _disk_cache_meta:
        cached_indices = _disk_cache_meta["sampled_indices"]
    else:
        # Fallback: generate uniform indices from video frame count (~10fps from 30fps)
        cached_indices = list(range(0, session.frames_count, 3))

    pct = max(1, min(100, session.seed_config.frame_pct))

    if pct >= 100:
        sampled = set(cached_indices)
    else:
        # Uniformly subsample cached frames to desired %
        n_target = max(1, int(len(cached_indices) * pct / 100))
        step = max(1, len(cached_indices) // n_target)
        sampled = set(cached_indices[::step])

    # Always include change keyframes (they're guaranteed in the cache)
    all_targets = sorted(sampled | change)
    total_frames = len(all_targets)

    progress.step = "Generating seeds..."
    progress.total = total_frames
    progress.current = 0

    threshold = session.seed_config.confidence_threshold
    seeds: List[Dict[str, Any]] = []

    # Get video dimensions for _detect_batch
    vid_width, vid_height, _, _ = _get_video_info_pyav(session.video_path)

    # Pre-compute text tokens for each prompt (avoids re-tokenization per batch)
    precomputed_tokens: Dict[str, Any] = {}
    for p in prompts:
        tokens = precompute_text_tokens(detector, p)
        if tokens is not None:
            precomputed_tokens[p] = tokens
    if precomputed_tokens:
        logger.info("Pre-computed text tokens for %d prompts", len(precomputed_tokens))

    logger.info(
        "Seed generation: scanning %d frames (pct=%d%%, threshold=%.2f, "
        "prompts=%s, refinement=%s, batch_size=%d, support_set=%d)",
        total_frames, pct, threshold,
        prompts, _ENABLE_REFINEMENT, _SEED_DETECT_BATCH,
        support_feats.shape[0],
    )

    # Check if disk frame cache is available for fast reads
    _disk_cache_available = False
    try:
        from .disk_frame_cache import frame_cache_exists, read_cached_frame as _read_disk_frame
        if frame_cache_exists(session.cache_key):
            _disk_cache_available = True
            logger.info("Seeding: disk frame cache available — using cached frames")
    except ImportError:
        pass

    # ---- Process in chunks (batched detection) ----
    for chunk_start in range(0, total_frames, _SEED_CHUNK_SIZE):
        chunk_indices = all_targets[chunk_start:chunk_start + _SEED_CHUNK_SIZE]
        progress.step = (
            f"Decoding frames {chunk_start + 1}"
            f"-{chunk_start + len(chunk_indices)} / {total_frames}..."
        )

        # Try disk frame cache first, fall back to video decode
        if _disk_cache_available:
            frames = {}
            missing = []
            for fidx in chunk_indices:
                img = _read_disk_frame(session.cache_key, fidx)
                if img is not None:
                    frames[fidx] = img
                else:
                    missing.append(fidx)
            if missing:
                frames.update(_decode_frames_sequential(
                    session.video_path, missing, cache_key=session.cache_key,
                ))
        else:
            frames = _decode_frames_sequential(
                session.video_path, chunk_indices, cache_key=session.cache_key,
            )

        # --- Batched detection: run all prompts via _detect_batch ---
        # Collect detections grouped by frame_idx across all prompts
        from collections import defaultdict
        frame_detections: Dict[int, List[Tuple[np.ndarray, float]]] = defaultdict(list)

        for prompt_text in prompts:
            progress.step = (
                f"Detecting '{prompt_text}' on frames "
                f"{chunk_start + 1}-{chunk_start + len(chunk_indices)}..."
            )
            crops = _detect_batch(
                detector, frames, prompt_text,
                vid_width, vid_height,
                batch_size=_SEED_DETECT_BATCH,
                nms_iou=0.5,
                pad_frac=0.1,
                precomputed_text=precomputed_tokens.get(prompt_text),
            )
            for crop in crops:
                frame_detections[crop.frame_idx].append(
                    (crop.xyxy, crop.score)
                )

        progress.current = chunk_start + len(chunk_indices)

        # --- Per-frame: cross-prompt NMS + features + k-NN gate ---
        medium_candidates: List[Tuple[int, np.ndarray, float]] = []
        oversized_candidates: List[Tuple[int, np.ndarray, float]] = []

        for frame_idx in chunk_indices:
            dets = frame_detections.get(frame_idx)
            if not dets:
                continue

            pil_frame = frames.get(frame_idx)
            if pil_frame is None:
                continue

            boxes = np.array([d[0] for d in dets], dtype=np.float32)
            det_scores = np.array([d[1] for d in dets], dtype=np.float32)

            # Cross-prompt NMS (boxes are already padded by _detect_batch)
            if len(prompts) > 1:
                keep = nms_numpy(boxes, det_scores, iou_threshold=0.5)
                boxes = boxes[keep]
                det_scores = det_scores[keep]

            if len(boxes) == 0:
                continue

            # Crop images + context images for DINOv3
            crop_images = []
            context_images = []
            valid_indices = []
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_frame.width, x2), min(pil_frame.height, y2)
                if x2 > x1 and y2 > y1:
                    crop_images.append(pil_frame.crop((x1, y1, x2, y2)))
                    # Context: expand by 50% of max(w, h)
                    bw, bh = float(x2 - x1), float(y2 - y1)
                    expand = 0.5 * max(bw, bh)
                    cx1 = max(0, int(x1 - expand))
                    cy1 = max(0, int(y1 - expand))
                    cx2 = min(pil_frame.width, int(x2 + expand))
                    cy2 = min(pil_frame.height, int(y2 + expand))
                    if cx2 > cx1 and cy2 > cy1:
                        context_images.append(pil_frame.crop((cx1, cy1, cx2, cy2)))
                    else:
                        context_images.append(crop_images[-1])
                    valid_indices.append(idx)

            if not crop_images:
                continue

            boxes = boxes[valid_indices]
            det_scores = det_scores[valid_indices]

            # Extract DINOv3 features for crops and contexts
            crop_features = extract_features(crop_images)
            ctx_features = extract_features(context_images)

            metadata = np.array([
                compute_crop_metadata(b, pil_frame.width, pil_frame.height)
                for b in boxes
            ], dtype=np.float32)

            # Mask quality: not available from _detect_batch (no masks returned)
            mask_quality_arr = np.array([
                [0.0, float(det_scores[idx]), 0.0, 0.0]
                for idx in range(len(boxes))
            ], dtype=np.float32)

            # Build 2056-dim query: [crop CLS | context CLS | spatial | mask quality]
            query_features = np.concatenate(
                [crop_features, ctx_features, metadata, mask_quality_arr], axis=1,
            )

            # k-NN scoring against labeled support set
            confidences, _ = score_crops(
                query_features, support_feats, support_labels, support_reasons,
            )
            inferred_reasons = _infer_reject_reasons(
                query_features, support_feats, support_labels, support_reasons,
            )

            for i in range(len(boxes)):
                conf = float(confidences[i])
                if conf >= threshold:
                    identity, identity_sim = _assign_identity(
                        crop_features[i], centroids,
                    )
                    seeds.append({
                        "frame_idx": int(frame_idx),
                        "xyxy": boxes[i].tolist(),
                        "confidence": round(conf, 4),
                        "identity": int(identity),
                        "identity_similarity": round(float(identity_sim), 4),
                        "source": "multi_prompt_knn",
                    })
                elif _ENABLE_REFINEMENT and conf >= _REFINE_THRESHOLD:
                    # Preserve original behavior: refine all medium-confidence rejects.
                    medium_candidates.append((frame_idx, boxes[i], det_scores[i], "refined_medium"))
                elif (
                    _ENABLE_REFINEMENT
                    and i < len(inferred_reasons)
                    and inferred_reasons[i] == "oversized_box"
                ):
                    # Additional fallback: if inferred reject subtype is oversized_box,
                    # attempt SAM3 refinement even for low-confidence rejects.
                    oversized_candidates.append((frame_idx, boxes[i], det_scores[i], "refined_oversized"))

        # Refine medium-confidence candidates and oversized fallbacks.
        refine_candidates = medium_candidates + oversized_candidates
        if refine_candidates and _ENABLE_REFINEMENT:
            progress.step = f"Refining {len(refine_candidates)} candidates..."
            refine_inputs = [(fi, bx, sc) for fi, bx, sc, _tag in refine_candidates]
            refine_tags = [tag for _fi, _bx, _sc, tag in refine_candidates]
            refined = _refine_candidates_sam3(
                frames, refine_inputs, prompt=prompts[0],
                tags=refine_tags,
            )
            for result in refined:
                frame_idx, box, _det_score = result[0], result[1], result[2]
                tag = result[3] if len(result) > 3 else "refined"
                pil_frame = frames.get(frame_idx)
                if pil_frame is None:
                    continue
                seed = _score_and_accept_seed(
                    box, pil_frame,
                    support_feats, support_labels, support_reasons,
                    centroids, threshold, tag, frame_idx,
                )
                if seed is not None:
                    seeds.append(seed)

        # Free chunk memory before next iteration
        del frames, frame_detections
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Finalise ----
    with session._lock:
        session.seeds = seeds
        session.advance_to(Phase.SEEDING)

    save_session(session)

    identity_counts: Dict[int, int] = {}
    for seed in seeds:
        ident = seed["identity"]
        identity_counts[ident] = identity_counts.get(ident, 0) + 1

    summary = {
        "total_seeds": len(seeds),
        "frames_scanned": total_frames,
        "identities": identity_counts,
        "prompts_used": prompts,
    }

    logger.info(
        "Seed generation complete: %d seeds across %d frames, %d identities",
        len(seeds), total_frames, len(identity_counts),
    )
    progress.step = f"Done - {len(seeds)} seeds generated"
    return summary


def upload_seeds(
    session: InterviewSession,
    progress: JobProgress,
) -> Dict[str, Any]:
    """Upload seed regions to Label Studio with ``enabled=false`` keyframes.

    Each ReID identity becomes one Label Studio *videorectangle* region.
    Every seed detection for that identity becomes a keyframe entry in the
    region's ``sequence`` array, with ``enabled: false`` to prevent Label
    Studio from auto-interpolating between keyframes.

    The tracks structure expected by :func:`_build_prediction`::

        tracks = [
            {
                "track_id": identity_id,
                "label": "person",
                "sequence": [
                    {
                        "frame": frame_num,    # 1-based for LS
                        "xyxy": np.array([x1, y1, x2, y2]),
                        "enabled": False,
                    },
                    ...
                ],
            },
            ...
        ]

    Args:
        session: The interview session containing ``session.seeds`` (populated
            by :func:`generate_seeds`).
        progress: :class:`JobProgress` handle for status reporting.

    Returns:
        Upload result info including the number of tracks and keyframes
        pushed to Label Studio.

    Raises:
        RuntimeError: If no seeds have been generated yet.
        seeding_common.InitialSeedingError: If Label Studio connection fails.
    """
    progress.step = "Preparing upload..."
    progress.current = 0
    progress.total = 4

    if not session.seeds:
        raise RuntimeError(
            "No seeds to upload. Run seed generation first."
        )

    # ---- Group seeds by identity ----
    progress.step = "Grouping seeds by identity..."
    progress.current = 1

    identity_seeds: Dict[int, List[Dict[str, Any]]] = {}
    for seed in session.seeds:
        ident = seed["identity"]
        identity_seeds.setdefault(ident, []).append(seed)

    # Determine the label text for tracks
    label_text = session.prompts[0] if session.prompts else "person"

    # ---- Build track structures ----
    progress.step = "Building track structures..."
    progress.current = 2

    tracks: List[Dict[str, Any]] = []
    for identity_id, id_seeds in sorted(identity_seeds.items()):
        # Sort seeds by frame for a coherent sequence
        id_seeds_sorted = sorted(id_seeds, key=lambda s: s["frame_idx"])

        sequence: List[Dict[str, Any]] = []
        for seed in id_seeds_sorted:
            sequence.append({
                "frame": seed["frame_idx"] + 1,  # convert 0-based to 1-based
                "xyxy": np.array(seed["xyxy"], dtype=np.float32),
                "enabled": False,
            })

        tracks.append({
            "track_id": identity_id,
            "label": label_text,
            "sequence": sequence,
        })

    total_keyframes = sum(len(t["sequence"]) for t in tracks)
    logger.info(
        "Upload: %d tracks, %d total keyframes",
        len(tracks),
        total_keyframes,
    )

    # ---- Build LS prediction payload ----
    progress.step = "Building prediction payload..."
    progress.current = 3

    prediction = _build_prediction(
        tracks=tracks,
        width=session.width,
        height=session.height,
        frames_count=session.frames_count,
        fps=session.fps,
    )

    # ---- Connect and upload ----
    progress.step = "Uploading to Label Studio..."
    progress.current = 4

    ls_url = (
        os.getenv("LABEL_STUDIO_HOST")
        or os.getenv("LABEL_STUDIO_URL", "")
    )
    ls_api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    ls = _build_ls_client(ls_url, ls_api_key)

    _upload_prediction(ls, session.task_id, prediction)

    # ---- Finalise ----
    result = {
        "tracks_uploaded": len(tracks),
        "total_keyframes": total_keyframes,
        "identities": list(identity_seeds.keys()),
        "model_version": prediction.get("model_version", "sam3-init-seed"),
    }

    with session._lock:
        session.upload_result = result
        session.advance_to(Phase.COMPLETE)

    save_session(session)

    logger.info(
        "Upload complete: %d tracks (%d keyframes) for task %d",
        len(tracks),
        total_keyframes,
        session.task_id,
    )
    progress.step = f"Done - uploaded {len(tracks)} tracks"
    return result
