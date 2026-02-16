"""Visual ReID pipeline — multi-cue crop enrichment and similarity functions.

Tier 1 (T1): Crop appearance descriptors — modality detection, body
proportions, and basic enrichment of CropData into EnrichedCrop.

Later tiers add temporal co-occurrence, context patches, constraint graphs,
hierarchical clustering, merge proposal scoring, and information-gain
ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from complete_reid import _compute_hist

from .state import (
    CropData, CropLabel, CueWeights, EnrichedCrop,
    InterviewSession, MergeProposal, RunGroup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T1.1  Modality Detection
# ---------------------------------------------------------------------------


def detect_modality(pixels: np.ndarray) -> str:
    """Detect whether a crop is RGB, grayscale, or infrared.

    Args:
        pixels: (H, W, 3) uint8 array in RGB order.

    Returns:
        "rgb", "grayscale", or "ir".

    Logic:
        1. If any channel differs from the others -> "rgb".
        2. If all channels identical and dynamic range < 60 -> "ir".
        3. If all channels identical and dynamic range >= 60 -> "grayscale".
    """
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

    # Check if all three channels are identical
    all_same = np.array_equal(r, g) and np.array_equal(g, b)

    if not all_same:
        return "rgb"

    # All channels identical — distinguish grayscale from IR by dynamic range
    dynamic_range = int(r.max()) - int(r.min())
    if dynamic_range < 60:
        return "ir"
    return "grayscale"


# ---------------------------------------------------------------------------
# T1.2  Body Proportions
# ---------------------------------------------------------------------------


def compute_body_props(
    xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
    crop_pixels: np.ndarray,
) -> np.ndarray:
    """Compute body proportion features from a bounding box and its crop pixels.

    Args:
        xyxy: (4,) array [x1, y1, x2, y2] in pixel coordinates.
        frame_w: Full frame width in pixels.
        frame_h: Full frame height in pixels.
        crop_pixels: (crop_h, crop_w, 3) uint8 array of the cropped region.

    Returns:
        (4,) float32 array: [aspect_ratio, bbox_area_norm, torso_ratio_est, limb_spread].

        - aspect_ratio: height / width of the bounding box.
        - bbox_area_norm: bbox area / frame area (0-1).
        - torso_ratio_est: mean row index of foreground pixels / crop height.
          For a standing person this tends toward 0.4-0.5 (center of mass).
        - limb_spread: fraction of foreground pixels in the top and bottom
          thirds of the crop. Higher values indicate extended limbs.
    """
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    bbox_w = max(x2 - x1, 1.0)
    bbox_h = max(y2 - y1, 1.0)

    aspect_ratio = bbox_h / bbox_w
    bbox_area_norm = (bbox_w * bbox_h) / max(frame_w * frame_h, 1)

    crop_h, crop_w = crop_pixels.shape[:2]

    # Foreground mask: any pixel not pure black (simple heuristic)
    # For uniform-filled crops this will be all True
    fg_mask = crop_pixels.sum(axis=2) > 0  # (crop_h, crop_w)
    n_fg = int(fg_mask.sum())

    if n_fg == 0:
        # No foreground — return neutral values
        return np.array([aspect_ratio, bbox_area_norm, 0.5, 0.0], dtype=np.float32)

    # torso_ratio_est: mean row index of foreground pixels normalized by height
    row_indices = np.arange(crop_h).reshape(crop_h, 1)  # (crop_h, 1)
    fg_row_sum = float((row_indices * fg_mask).sum())
    torso_ratio_est = fg_row_sum / (n_fg * max(crop_h, 1))

    # limb_spread: fraction of foreground in top third + bottom third
    third_h = max(crop_h // 3, 1)
    top_third_fg = int(fg_mask[:third_h, :].sum())
    bottom_third_fg = int(fg_mask[crop_h - third_h:, :].sum())
    limb_spread = (top_third_fg + bottom_third_fg) / n_fg

    return np.array(
        [aspect_ratio, bbox_area_norm, torso_ratio_est, limb_spread],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# T1.3  Crop Enrichment (Tier 1)
# ---------------------------------------------------------------------------


def enrich_crops_t1(
    crops: List[CropData],
    frame_w: int,
    frame_h: int,
    read_frame: Callable,
    video_path: str,
    cache_key: Optional[str] = None,
) -> List[EnrichedCrop]:
    """Build Tier-1 EnrichedCrop descriptors from CropData objects.

    For each CropData that has DINOv3 features, reads the corresponding
    video frame, extracts the crop region, detects modality, computes body
    proportions, and assembles an EnrichedCrop.

    Features (dinov3_cls) and histogram (color_hist) are reused from the
    CropData if available, avoiding redundant computation.

    Args:
        crops: List of CropData objects (accepted crops with features).
        frame_w: Video frame width in pixels.
        frame_h: Video frame height in pixels.
        read_frame: Callable(video_path, frame_idx, cache_key=None) -> PIL.Image.
            Returns a PIL Image for the given frame index.
        video_path: Path to the video file.
        cache_key: Optional cache key for frame reads.

    Returns:
        List of EnrichedCrop objects (one per CropData that has features).
        Crops without features are skipped.
    """
    enriched: List[EnrichedCrop] = []

    # Group crops by frame to minimize video seeks
    frame_to_crops: Dict[int, List[CropData]] = {}
    for crop in crops:
        if crop.features is None:
            continue  # skip crops without DINOv3 features
        frame_to_crops.setdefault(crop.frame_idx, []).append(crop)

    for frame_idx in sorted(frame_to_crops.keys()):
        pil_frame = read_frame(video_path, frame_idx, cache_key=cache_key)
        if pil_frame is None:
            logger.warning("Could not read frame %d for enrichment", frame_idx)
            continue

        frame_rgb = np.array(pil_frame.convert("RGB"))
        h, w = frame_rgb.shape[:2]

        for crop in frame_to_crops[frame_idx]:
            x1, y1, x2, y2 = crop.xyxy.astype(int)
            # Clamp to frame bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            crop_pixels = frame_rgb[y1:y2, x1:x2]
            if crop_pixels.size == 0:
                logger.debug("Empty crop %s at frame %d", crop.crop_id, frame_idx)
                continue

            modality = detect_modality(crop_pixels)
            body_props = compute_body_props(crop.xyxy, frame_w, frame_h, crop_pixels)

            # Scene position: normalized center of bbox
            cx = (float(crop.xyxy[0]) + float(crop.xyxy[2])) / 2.0 / max(frame_w, 1)
            cy = (float(crop.xyxy[1]) + float(crop.xyxy[3])) / 2.0 / max(frame_h, 1)
            scene_pos = np.array([cx, cy], dtype=np.float32)

            # Quadrant (3x3 grid, 0-8)
            qx = min(int(cx * 3), 2)
            qy = min(int(cy * 3), 2)
            quadrant = qy * 3 + qx

            # Compute color histogram from crop pixels if not already cached
            color_hist = None
            if modality == "rgb":
                if crop.histogram is not None:
                    color_hist = crop.histogram
                else:
                    color_hist = _compute_hist(crop_pixels, (8, 8, 8))
                    crop.histogram = color_hist  # cache for future use

            ec = EnrichedCrop(
                crop_id=crop.crop_id,
                frame_idx=crop.frame_idx,
                modality=modality,
                dinov3_cls=crop.features,
                body_props=body_props,
                scene_pos=scene_pos,
                quadrant=quadrant,
                color_hist=color_hist,
            )
            enriched.append(ec)

    logger.info(
        "Enriched %d crops (T1) from %d total input crops",
        len(enriched), len(crops),
    )
    return enriched


# ---------------------------------------------------------------------------
# T3  Temporal Co-occurrence Map
# ---------------------------------------------------------------------------


def compute_cooccurrence(crops: List[EnrichedCrop]) -> np.ndarray:
    """Compute NxN boolean co-occurrence matrix.

    Entry (i, j) is True if crops[i] and crops[j] share the same frame_idx.

    Args:
        crops: List of EnrichedCrop objects.

    Returns:
        (N, N) bool ndarray.
    """
    n = len(crops)
    frames = np.array([c.frame_idx for c in crops])
    # Broadcasting: frames[:, None] == frames[None, :] gives (N, N) bool
    cooc = frames[:, None] == frames[None, :]
    # Diagonal is trivially True (self-cooccurrence) — clear it so the
    # matrix only encodes "distinct detections share a frame".
    np.fill_diagonal(cooc, False)
    return cooc


def compute_temporal_adjacency(
    crops: List[EnrichedCrop],
    tau: float = 50.0,
) -> np.ndarray:
    """Compute NxN temporal adjacency matrix.

    Entry (i, j) = exp(-|frame_i - frame_j| / tau).

    Args:
        crops: List of EnrichedCrop objects.
        tau: Decay constant in frames. Larger tau = slower decay.

    Returns:
        (N, N) float64 ndarray with values in [0, 1].
    """
    n = len(crops)
    frames = np.array([c.frame_idx for c in crops], dtype=np.float64)
    diffs = np.abs(frames[:, None] - frames[None, :])
    return np.exp(-diffs / tau)


def compute_spatial_temporal(
    crops: List[EnrichedCrop],
    tau: float = 50.0,
) -> np.ndarray:
    """Compute NxN spatial-temporal affinity matrix.

    Entry (i, j) = temporal_adj(i, j) * max(0, 1 - euclidean(pos_i, pos_j)).

    The spatial term is clamped to [0, 1] so distant crops get zero
    spatial affinity regardless of temporal proximity.

    Args:
        crops: List of EnrichedCrop objects (must have scene_pos set).
        tau: Temporal decay constant.

    Returns:
        (N, N) float64 ndarray with values in [0, 1].
    """
    temporal = compute_temporal_adjacency(crops, tau=tau)

    n = len(crops)
    positions = np.zeros((n, 2), dtype=np.float64)
    for i, c in enumerate(crops):
        if c.scene_pos is not None:
            positions[i] = c.scene_pos

    # Pairwise euclidean distances
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    dists = np.sqrt((diff ** 2).sum(axis=2))  # (N, N)

    spatial = np.clip(1.0 - dists, 0.0, 1.0)
    return temporal * spatial


def segment_runs(
    crops: List[EnrichedCrop],
    spatial_temporal: np.ndarray,
    threshold: float = 0.5,
) -> List[List[str]]:
    """Greedy chaining of crops into temporal runs.

    Crops are sorted by frame_idx. A new crop extends the current run
    if its spatial_temporal score with the PREVIOUS crop in the run
    is >= threshold. Otherwise a new run begins.

    Args:
        crops: List of EnrichedCrop objects.
        spatial_temporal: (N, N) affinity matrix (from compute_spatial_temporal).
        threshold: Minimum affinity to extend a run.

    Returns:
        List of runs, each run is a list of crop_id strings in frame order.
    """
    if not crops:
        return []

    # Sort crops by frame_idx, preserving original indices for matrix lookup
    indexed = sorted(enumerate(crops), key=lambda t: t[1].frame_idx)

    runs: List[List[str]] = []
    current_run: List[str] = [indexed[0][1].crop_id]
    prev_idx = indexed[0][0]  # matrix index of previous crop

    for orig_idx, crop in indexed[1:]:
        score = spatial_temporal[prev_idx, orig_idx]
        if score >= threshold:
            current_run.append(crop.crop_id)
        else:
            runs.append(current_run)
            current_run = [crop.crop_id]
        prev_idx = orig_idx

    runs.append(current_run)
    return runs


def extract_cannot_links(
    cooccurrence: np.ndarray,
    crops: List[EnrichedCrop],
) -> Set[Tuple[str, str]]:
    """Extract cannot-link pairs from co-occurring crops.

    Two crops that appear in the same frame cannot be the same identity.
    Returns pairs (a, b) where a < b lexicographically.

    Args:
        cooccurrence: (N, N) bool matrix from compute_cooccurrence.
        crops: List of EnrichedCrop objects (same order as matrix).

    Returns:
        Set of (crop_id_a, crop_id_b) tuples with a < b.
    """
    n = len(crops)
    links: Set[Tuple[str, str]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if cooccurrence[i, j]:
                a, b = crops[i].crop_id, crops[j].crop_id
                if a > b:
                    a, b = b, a
                links.add((a, b))
    return links


# ---------------------------------------------------------------------------
# T5  Constraint Graph Builder
# ---------------------------------------------------------------------------


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance: 1 - cosine_similarity.

    Returns 1.0 if either vector has near-zero norm (undefined similarity).

    Args:
        a: (D,) float array.
        b: (D,) float array.

    Returns:
        Float in [0, 2]. 0 = identical direction, 2 = opposite.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    cosine_sim = float(np.dot(a, b)) / (norm_a * norm_b)
    # Clamp to [-1, 1] for numerical safety
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    return 1.0 - cosine_sim


def validate_and_split_runs(
    raw_runs: List[List[str]],
    crop_map: Dict[str, EnrichedCrop],
    sigma_threshold: float = 2.0,
) -> List[List[str]]:
    """Validate runs and split at feature discontinuities.

    For each run with > 2 crops, computes consecutive pairwise cosine
    distances. If any distance exceeds mean + sigma_threshold * std,
    the run is split at that point. Runs with <= 2 crops pass through.

    Args:
        raw_runs: List of runs (each a list of crop_id strings).
        crop_map: Mapping from crop_id to EnrichedCrop.
        sigma_threshold: Number of standard deviations above mean to trigger split.

    Returns:
        Possibly expanded list of runs after splitting.
    """
    result: List[List[str]] = []

    for run in raw_runs:
        if len(run) <= 2:
            result.append(run)
            continue

        # Compute consecutive pairwise cosine distances
        distances: List[float] = []
        for k in range(len(run) - 1):
            feat_a = crop_map[run[k]].dinov3_cls
            feat_b = crop_map[run[k + 1]].dinov3_cls
            distances.append(_cosine_distance(feat_a, feat_b))

        dist_arr = np.array(distances)
        mean_d = float(dist_arr.mean())
        std_d = float(dist_arr.std())
        threshold = mean_d + sigma_threshold * std_d

        # Find split points (where distance exceeds threshold)
        split_indices: List[int] = []
        for k, d in enumerate(distances):
            if d > threshold:
                split_indices.append(k + 1)  # split AFTER position k

        if not split_indices:
            result.append(run)
        else:
            # Build sub-runs from split points
            prev = 0
            for si in split_indices:
                result.append(run[prev:si])
                prev = si
            result.append(run[prev:])

    return result


def collapse_runs(
    raw_runs: List[List[str]],
    crop_map: Dict[str, EnrichedCrop],
) -> List[RunGroup]:
    """Collapse runs into RunGroup objects by averaging per-crop features.

    Each run becomes one RunGroup with averaged features, scene_pos,
    body_props, and computed frame_range.

    Args:
        raw_runs: List of runs (each a list of crop_id strings).
        crop_map: Mapping from crop_id to EnrichedCrop.

    Returns:
        List of RunGroup objects, one per input run.
    """
    groups: List[RunGroup] = []

    for run_id, run in enumerate(raw_runs):
        if not run:
            continue

        crops_in_run = [crop_map[cid] for cid in run]

        # Mean DINOv3 features
        features_stack = np.stack([c.dinov3_cls for c in crops_in_run])
        mean_features = features_stack.mean(axis=0).astype(np.float32)

        # Mean scene position
        positions = []
        for c in crops_in_run:
            if c.scene_pos is not None:
                positions.append(c.scene_pos)
        if positions:
            mean_scene_pos = np.stack(positions).mean(axis=0).astype(np.float32)
        else:
            mean_scene_pos = np.array([0.5, 0.5], dtype=np.float32)

        # Frame range
        frames = [c.frame_idx for c in crops_in_run]
        frame_range = (min(frames), max(frames))

        # Mean body props
        body_stack = np.stack([c.body_props for c in crops_in_run])
        mean_body_props = body_stack.mean(axis=0).astype(np.float32)

        # Mean color hist (only if all crops have it)
        hists = [c.color_hist for c in crops_in_run if c.color_hist is not None]
        mean_color_hist = None
        if len(hists) == len(crops_in_run) and hists:
            mean_color_hist = np.stack(hists).mean(axis=0).astype(np.float32)

        # Mean context CLS (only if all crops have it)
        ctx = [c.context_cls for c in crops_in_run if c.context_cls is not None]
        mean_context_cls = None
        if len(ctx) == len(crops_in_run) and ctx:
            mean_context_cls = np.stack(ctx).mean(axis=0).astype(np.float32)

        rg = RunGroup(
            run_id=run_id,
            crop_ids=list(run),
            mean_features=mean_features,
            mean_scene_pos=mean_scene_pos,
            frame_range=frame_range,
            mean_body_props=mean_body_props,
            mean_color_hist=mean_color_hist,
            mean_context_cls=mean_context_cls,
        )
        groups.append(rg)

    return groups


def propagate_cannot_links_to_runs(
    runs: List[RunGroup],
    crop_cannot_links: Set[Tuple[str, str]],
) -> Set[Tuple[int, int]]:
    """Propagate crop-level cannot-links to run-level cannot-links.

    If any crop in run A has a cannot-link with any crop in run B,
    then (min(A.run_id, B.run_id), max(A.run_id, B.run_id)) is added.

    Cannot-links between crops in the SAME run are ignored (they don't
    produce a run-level constraint).

    Args:
        runs: List of RunGroup objects.
        crop_cannot_links: Set of (crop_id_a, crop_id_b) pairs (a < b).

    Returns:
        Set of (run_id_a, run_id_b) tuples with a < b.
    """
    # Build crop_id -> run_id mapping
    crop_to_run: Dict[str, int] = {}
    for rg in runs:
        for cid in rg.crop_ids:
            crop_to_run[cid] = rg.run_id

    run_links: Set[Tuple[int, int]] = set()

    for cid_a, cid_b in crop_cannot_links:
        run_a = crop_to_run.get(cid_a)
        run_b = crop_to_run.get(cid_b)
        if run_a is None or run_b is None:
            continue  # crop not in any run
        if run_a == run_b:
            continue  # same run — skip
        lo, hi = min(run_a, run_b), max(run_a, run_b)
        run_links.add((lo, hi))

    return run_links


# ---------------------------------------------------------------------------
# T6  Over-Cluster (COP-HAC)
# ---------------------------------------------------------------------------


def _histogram_intersection(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Compute histogram intersection (similarity in [0, 1]).

    Returns 0.0 if either histogram is None.  Normalizes by the smaller
    histogram mass so the result is always in [0, 1].
    """
    if a is None or b is None:
        return 0.0
    denom = min(float(a.sum()), float(b.sum()))
    if denom < 1e-12:
        return 0.0
    return float(np.minimum(a, b).sum()) / denom


def _compute_run_distance(a: RunGroup, b: RunGroup, weights: CueWeights) -> float:
    """Compute weighted multi-cue distance between two RunGroups.

    Cue distances:
        - appearance: cosine_distance of mean_features.
        - spatial: euclidean of mean_scene_pos.
        - body: euclidean of mean_body_props (0.5 if None).
        - color: 1 - histogram_intersection (0.5 if None).
        - context: cosine_distance of mean_context_cls (0.5 if None).
        - temporal: frame overlap ratio (0 if non-overlapping = compatible).

    Returns:
        Weighted sum of cue distances (lower = more similar).
    """
    # Appearance
    d_appearance = _cosine_distance(a.mean_features, b.mean_features)

    # Spatial
    d_spatial = float(np.linalg.norm(a.mean_scene_pos - b.mean_scene_pos))

    # Body
    if a.mean_body_props is not None and b.mean_body_props is not None:
        d_body = float(np.linalg.norm(a.mean_body_props - b.mean_body_props))
    else:
        d_body = 0.5

    # Color
    if a.mean_color_hist is not None and b.mean_color_hist is not None:
        d_color = 1.0 - _histogram_intersection(a.mean_color_hist, b.mean_color_hist)
    else:
        d_color = 0.5

    # Context
    if a.mean_context_cls is not None and b.mean_context_cls is not None:
        d_context = _cosine_distance(a.mean_context_cls, b.mean_context_cls)
    else:
        d_context = 0.5

    # Temporal: frame overlap ratio
    a_start, a_end = a.frame_range
    b_start, b_end = b.frame_range
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    overlap_len = max(0, overlap_end - overlap_start)
    union_len = max(a_end, b_end) - min(a_start, b_start)
    d_temporal = overlap_len / max(union_len, 1)  # 0 if non-overlapping

    return (
        weights.appearance * d_appearance
        + weights.spatial * d_spatial
        + weights.body * d_body
        + weights.color * d_color
        + weights.context * d_context
        + weights.temporal * d_temporal
    )


def over_cluster(
    runs: List[RunGroup],
    cannot_links: Set[Tuple[int, int]],
    weights: CueWeights,
    distance_threshold: float = 0.35,
    min_k: int = 1,
) -> Dict[int, List[int]]:
    """COP-HAC: agglomerative clustering with cannot-link constraints.

    Starts with each run as its own cluster. Repeatedly merges the closest
    pair (average linkage) unless blocked by a cannot-link. Stops when the
    minimum distance exceeds *distance_threshold* or the number of clusters
    reaches *min_k*.

    Args:
        runs: List of RunGroup objects.
        cannot_links: Set of (run_id_a, run_id_b) pairs (a < b) that must
            not be in the same cluster.
        weights: CueWeights for distance computation.
        distance_threshold: Maximum distance to allow a merge.
        min_k: Minimum number of clusters (stop merging at this count).

    Returns:
        {cluster_id: [run_ids]} mapping.
    """
    if not runs:
        return {}

    # Build run_id -> RunGroup mapping
    run_map: Dict[int, RunGroup] = {r.run_id: r for r in runs}

    # Initialize: each run in its own cluster
    # cluster_id -> set of run_ids
    clusters: Dict[int, List[int]] = {r.run_id: [r.run_id] for r in runs}

    def _are_blocked(cid_a: int, cid_b: int) -> bool:
        """Check if merging two clusters violates any cannot-link."""
        for rid_a in clusters[cid_a]:
            for rid_b in clusters[cid_b]:
                lo, hi = min(rid_a, rid_b), max(rid_a, rid_b)
                if (lo, hi) in cannot_links:
                    return True
        return False

    def _average_linkage(cid_a: int, cid_b: int) -> float:
        """Compute average linkage distance between two clusters."""
        total = 0.0
        count = 0
        for rid_a in clusters[cid_a]:
            for rid_b in clusters[cid_b]:
                total += _compute_run_distance(run_map[rid_a], run_map[rid_b], weights)
                count += 1
        return total / max(count, 1)

    while len(clusters) > min_k:
        # Find closest pair of clusters that are not blocked
        best_dist = float("inf")
        best_pair: Optional[Tuple[int, int]] = None

        cluster_ids = sorted(clusters.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cid_a, cid_b = cluster_ids[i], cluster_ids[j]
                if _are_blocked(cid_a, cid_b):
                    continue
                dist = _average_linkage(cid_a, cid_b)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (cid_a, cid_b)

        # Stop if no valid merge or distance exceeds threshold
        if best_pair is None or best_dist > distance_threshold:
            break

        # Merge: add all runs from cid_b into cid_a
        cid_a, cid_b = best_pair
        clusters[cid_a].extend(clusters[cid_b])
        del clusters[cid_b]

    return clusters


# ---------------------------------------------------------------------------
# T7  Merge Proposal Scorer
# ---------------------------------------------------------------------------


def _compute_temporal_overlap(
    frame_ranges_a: List[Tuple[int, int]],
    frame_ranges_b: List[Tuple[int, int]],
) -> float:
    """Compute frame overlap ratio between two sets of frame ranges.

    The overlap is the total overlapping frames divided by the union span.

    Returns:
        Float in [0, 1]. 0 = no overlap, 1 = complete overlap.
    """
    if not frame_ranges_a or not frame_ranges_b:
        return 0.0

    all_start_a = min(s for s, _ in frame_ranges_a)
    all_end_a = max(e for _, e in frame_ranges_a)
    all_start_b = min(s for s, _ in frame_ranges_b)
    all_end_b = max(e for _, e in frame_ranges_b)

    overlap_start = max(all_start_a, all_start_b)
    overlap_end = min(all_end_a, all_end_b)
    overlap_len = max(0, overlap_end - overlap_start)

    union_start = min(all_start_a, all_start_b)
    union_end = max(all_end_a, all_end_b)
    union_len = union_end - union_start

    return overlap_len / max(union_len, 1)


def _cluster_centroid(
    run_ids: List[int],
    run_map: Dict[int, RunGroup],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute centroid of a cluster by averaging RunGroup attributes.

    Returns:
        (mean_features, mean_scene_pos, mean_body_props, mean_color_hist, mean_context_cls)
    """
    features = np.stack([run_map[rid].mean_features for rid in run_ids])
    mean_features = features.mean(axis=0)

    positions = np.stack([run_map[rid].mean_scene_pos for rid in run_ids])
    mean_scene_pos = positions.mean(axis=0)

    # Body props
    bp_list = [run_map[rid].mean_body_props for rid in run_ids
               if run_map[rid].mean_body_props is not None]
    mean_body_props = np.stack(bp_list).mean(axis=0) if bp_list else None

    # Color hist
    ch_list = [run_map[rid].mean_color_hist for rid in run_ids
               if run_map[rid].mean_color_hist is not None]
    mean_color_hist = np.stack(ch_list).mean(axis=0) if ch_list else None

    # Context cls
    cc_list = [run_map[rid].mean_context_cls for rid in run_ids
               if run_map[rid].mean_context_cls is not None]
    mean_context_cls = np.stack(cc_list).mean(axis=0) if cc_list else None

    return mean_features, mean_scene_pos, mean_body_props, mean_color_hist, mean_context_cls


def score_merge_proposals(
    clusters: Dict[int, List[int]],
    run_map: Dict[int, RunGroup],
    cannot_links: Set[Tuple[int, int]],
    weights: CueWeights,
) -> List[MergeProposal]:
    """Score all pairwise cluster merge proposals.

    For each pair of clusters, computes per-cue similarity scores and a
    combined merge_score that accounts for temporal overlap and cooccurrence
    conflicts.

    Args:
        clusters: {cluster_id: [run_ids]} mapping.
        run_map: {run_id: RunGroup} mapping.
        cannot_links: Set of (run_id_a, run_id_b) pairs (a < b).
        weights: CueWeights for similarity computation.

    Returns:
        List of MergeProposal objects sorted by merge_score descending.
    """
    cluster_ids = sorted(clusters.keys())
    proposals: List[MergeProposal] = []

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cid_a, cid_b = cluster_ids[i], cluster_ids[j]
            rids_a, rids_b = clusters[cid_a], clusters[cid_b]

            # Compute centroids
            feat_a, pos_a, bp_a, ch_a, cc_a = _cluster_centroid(rids_a, run_map)
            feat_b, pos_b, bp_b, ch_b, cc_b = _cluster_centroid(rids_b, run_map)

            # Per-cue similarities (1 - distance, clamped to [0, 1])
            sim_appearance = 1.0 - _cosine_distance(feat_a, feat_b)
            sim_appearance = max(0.0, min(1.0, sim_appearance))

            sim_spatial = 1.0 - float(np.linalg.norm(pos_a - pos_b))
            sim_spatial = max(0.0, min(1.0, sim_spatial))

            if bp_a is not None and bp_b is not None:
                sim_body = 1.0 - float(np.linalg.norm(bp_a - bp_b))
                sim_body = max(0.0, min(1.0, sim_body))
            else:
                sim_body = 0.5

            if ch_a is not None and ch_b is not None:
                sim_color = _histogram_intersection(ch_a, ch_b)
            else:
                sim_color = 0.5

            if cc_a is not None and cc_b is not None:
                sim_context = 1.0 - _cosine_distance(cc_a, cc_b)
                sim_context = max(0.0, min(1.0, sim_context))
            else:
                sim_context = 0.5

            # Temporal overlap
            ranges_a = [run_map[rid].frame_range for rid in rids_a]
            ranges_b = [run_map[rid].frame_range for rid in rids_b]
            temporal_overlap = _compute_temporal_overlap(ranges_a, ranges_b)

            # Temporal cue: similarity = 1 - overlap (compatible if non-overlapping)
            sim_temporal = 1.0 - temporal_overlap

            per_cue = {
                "appearance": sim_appearance,
                "spatial": sim_spatial,
                "body": sim_body,
                "color": sim_color,
                "context": sim_context,
                "temporal": sim_temporal,
            }

            # Cooccurrence conflict: any run in A has cannot-link with any run in B
            cooccurrence_conflict = False
            for rid_a in rids_a:
                for rid_b in rids_b:
                    lo, hi = min(rid_a, rid_b), max(rid_a, rid_b)
                    if (lo, hi) in cannot_links:
                        cooccurrence_conflict = True
                        break
                if cooccurrence_conflict:
                    break

            # Weighted sum of cue similarities (temporal is already in per_cue)
            raw_score = (
                weights.appearance * sim_appearance
                + weights.spatial * sim_spatial
                + weights.body * sim_body
                + weights.color * sim_color
                + weights.context * sim_context
                + weights.temporal * sim_temporal
            )

            # Apply penalties — co-occurrence conflict vetoes the proposal.
            # Temporal overlap is already captured in sim_temporal (weighted cue),
            # so we do NOT apply a separate multiplicative penalty to avoid
            # double-counting.
            conflict_penalty = 0.0 if cooccurrence_conflict else 1.0
            merge_score = raw_score * conflict_penalty

            proposals.append(MergeProposal(
                cluster_a=cid_a,
                cluster_b=cid_b,
                merge_score=merge_score,
                per_cue=per_cue,
                cooccurrence_conflict=cooccurrence_conflict,
                temporal_overlap=temporal_overlap,
            ))

    # Sort descending by merge_score
    proposals.sort(key=lambda p: -p.merge_score)
    return proposals


# ---------------------------------------------------------------------------
# T8+T10  Information Gain + Weight Update
# ---------------------------------------------------------------------------


def compute_information_gain(proposal: MergeProposal) -> float:
    """Compute information gain for a merge proposal.

    High when cues disagree (high variance) AND the merge_score is
    ambiguous (near 0.5).

    Formula: variance(per_cue values) * (1 - |merge_score - 0.5| * 2)

    Args:
        proposal: MergeProposal with per_cue and merge_score set.

    Returns:
        Non-negative float. Higher = more informative to ask a human about.
    """
    cue_values = list(proposal.per_cue.values())
    if not cue_values:
        return 0.0

    variance = float(np.var(cue_values))
    ambiguity = 1.0 - abs(proposal.merge_score - 0.5) * 2.0
    ambiguity = max(0.0, ambiguity)  # clamp to [0, 1]

    return variance * ambiguity


def select_informative_proposals(
    proposals: List[MergeProposal],
    k: int = 3,
) -> List[MergeProposal]:
    """Select the top-k most informative merge proposals.

    Computes information gain for each proposal, sets the
    ``information_gain`` attribute, and returns the top k sorted by
    information gain descending.

    Args:
        proposals: List of MergeProposal objects.
        k: Number of proposals to return.

    Returns:
        List of up to k MergeProposal objects, sorted by info gain descending.
    """
    for p in proposals:
        p.information_gain = compute_information_gain(p)

    ranked = sorted(proposals, key=lambda p: -p.information_gain)
    return ranked[:k]


def update_weights(
    weights: CueWeights,
    verdicts: List[Tuple[MergeProposal, str]],
    learning_rate: float = 0.1,
    floor: float = 0.05,
) -> CueWeights:
    """Update cue weights based on human verdicts.

    For each verdict (proposal, "same" or "different"):
        target = 1.0 if "same" else 0.0
        error = target - proposal.merge_score
        For each cue: gradient = error * cue_score
        weight += lr * gradient

    After all verdicts, normalize with floor.

    Args:
        weights: Current CueWeights.
        verdicts: List of (MergeProposal, verdict_str) tuples.
        learning_rate: Step size for gradient update.
        floor: Minimum weight value after normalization.

    Returns:
        New CueWeights (original is not mutated).
    """
    # Work with a mutable copy
    new_w = CueWeights(
        appearance=weights.appearance,
        spatial=weights.spatial,
        body=weights.body,
        color=weights.color,
        context=weights.context,
        temporal=weights.temporal,
    )

    cue_names = ["appearance", "spatial", "body", "color", "context", "temporal"]

    for proposal, verdict in verdicts:
        target = 1.0 if verdict == "same" else 0.0
        error = target - proposal.merge_score

        for cue_name in cue_names:
            cue_score = proposal.per_cue.get(cue_name, 0.0)
            gradient = error * cue_score
            current = getattr(new_w, cue_name)
            setattr(new_w, cue_name, current + learning_rate * gradient)

    new_w.normalize(floor=floor)
    return new_w


# ---------------------------------------------------------------------------
# T11+T12  Convergence Check + Final Output
# ---------------------------------------------------------------------------


def check_convergence(
    proposals: List[MergeProposal],
    max_verdicts: int,
    current_verdicts: int,
    score_threshold: float = 0.3,
) -> bool:
    """Check if the interactive ReID loop should stop.

    Converges when any of:
        1. No proposals remain.
        2. Top proposal merge_score < score_threshold.
        3. current_verdicts >= max_verdicts (budget exhausted).

    Args:
        proposals: List of MergeProposal objects (assumed sorted descending).
        max_verdicts: Maximum human verdicts allowed.
        current_verdicts: Number of verdicts collected so far.
        score_threshold: Minimum top score to continue.

    Returns:
        True if the loop should stop.
    """
    if not proposals:
        return True
    if current_verdicts >= max_verdicts:
        return True
    # Proposals are expected sorted descending; check the top one
    top_score = proposals[0].merge_score
    if top_score < score_threshold:
        return True
    return False


def build_final_cluster_map(
    run_clusters: Dict[int, List[int]],
    runs: List[RunGroup],
) -> Tuple[Dict[str, List[str]], Dict[int, str]]:
    """Build identity groups by expanding run_ids to crop_ids.

    Args:
        run_clusters: {cluster_id: [run_ids]} mapping.
        runs: List of RunGroup objects.

    Returns:
        Tuple of:
        - {"identity_0": [crop_ids], "identity_1": [...], ...}
        - {cluster_id: "identity_N", ...} mapping for remapping proposals
    """
    run_map: Dict[int, RunGroup] = {r.run_id: r for r in runs}

    result: Dict[str, List[str]] = {}
    cluster_id_to_identity: Dict[int, str] = {}
    for idx, (cluster_id, run_ids) in enumerate(sorted(run_clusters.items())):
        crop_ids: List[str] = []
        for rid in run_ids:
            rg = run_map.get(rid)
            if rg is not None:
                crop_ids.extend(rg.crop_ids)
        identity_name = f"identity_{idx}"
        result[identity_name] = crop_ids
        cluster_id_to_identity[cluster_id] = identity_name

    return result, cluster_id_to_identity


# ---------------------------------------------------------------------------
# T9  Pipeline Orchestrator
# ---------------------------------------------------------------------------


def run_visual_reid_pipeline(
    session: InterviewSession,
    read_frame: Callable,
    weights: CueWeights,
    distance_threshold: float = 0.35,
    tau: float = 25.0,
    run_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run the full visual ReID pipeline end-to-end.

    Steps:
        1. Collect accepted crops with features.
        2. Enrich crops (T1): modality, body props, scene position.
        3. Compute co-occurrence and spatial-temporal affinity matrices.
        4. Segment temporal runs and extract cannot-links.
        5. Validate/split runs and collapse into RunGroups.
        6. Propagate cannot-links from crop level to run level.
        7. Over-cluster (COP-HAC) with cannot-link constraints.
        8. Score merge proposals between resulting clusters.
        9. Build final cluster map (identity -> crop_ids).

    Args:
        session: InterviewSession with accepted, feature-bearing crops.
        read_frame: Callable(video_path, frame_idx, cache_key=None) -> PIL.Image.
        weights: CueWeights for distance computation.
        distance_threshold: Max distance for COP-HAC merges.
        tau: Temporal decay constant for spatial-temporal affinity.
        run_threshold: Minimum spatial-temporal affinity to extend a run.

    Returns:
        Dict with keys:
            "clusters": {identity_name: [crop_ids]} final assignment.
            "merge_proposals": list of MergeProposal.to_dict().
            "session_stats": {n_crops, n_runs, n_initial_clusters, n_final_clusters}.
            "enriched_crops": list of EnrichedCrop objects (for later use).
    """
    # Step 1: accepted crops with features
    accepted = [
        c for c in session.crops.values()
        if c.label == CropLabel.ACCEPTED and c.features is not None
    ]
    logger.info("Visual ReID pipeline: %d accepted crops with features", len(accepted))

    if not accepted:
        return {
            "clusters": {},
            "merge_proposals": [],
            "session_stats": {
                "n_crops": 0, "n_runs": 0,
                "n_initial_clusters": 0, "n_final_clusters": 0,
            },
            "enriched_crops": [],
        }

    # Step 2: enrich
    enriched = enrich_crops_t1(
        accepted, session.width, session.height,
        read_frame, session.video_path, session.cache_key,
    )

    # Step 3: co-occurrence + spatial-temporal
    co = compute_cooccurrence(enriched)
    st = compute_spatial_temporal(enriched, tau=tau)

    # Step 4: segment runs + crop-level cannot-links
    raw_runs = segment_runs(enriched, st, threshold=run_threshold)
    crop_cannot = extract_cannot_links(co, enriched)

    # Step 5: validate/split runs, collapse into RunGroups
    crop_map: Dict[str, EnrichedCrop] = {c.crop_id: c for c in enriched}
    validated = validate_and_split_runs(raw_runs, crop_map)
    run_groups = collapse_runs(validated, crop_map)

    n_initial_runs = len(run_groups)

    # Step 6: propagate cannot-links to run level
    run_cannot = propagate_cannot_links_to_runs(run_groups, crop_cannot)

    # Step 7: over-cluster (COP-HAC)
    clusters = over_cluster(run_groups, run_cannot, weights, distance_threshold)

    n_initial_clusters = len(clusters)

    # Step 8: score merge proposals
    run_map: Dict[int, RunGroup] = {rg.run_id: rg for rg in run_groups}
    proposals = score_merge_proposals(clusters, run_map, run_cannot, weights)

    # Step 9: build final cluster map + cluster_id → identity_name mapping
    final_clusters, cid_to_identity = build_final_cluster_map(clusters, run_groups)

    # Remap proposal cluster_ids to identity names so frontend can look up crops
    for p in proposals:
        p.cluster_a = cid_to_identity.get(p.cluster_a, p.cluster_a)
        p.cluster_b = cid_to_identity.get(p.cluster_b, p.cluster_b)

    # Step 10: diagnostics (cheap — all cached data, no GPU)
    feature_diag = compute_feature_space_diagnostics(enriched)
    cluster_diag = compute_clustering_diagnostics(clusters, run_map, weights, enriched)
    cue_diag = compute_cue_effectiveness(enriched, run_groups, clusters, run_map, weights)
    advisories = generate_advisories(feature_diag, cluster_diag, cue_diag)

    return {
        "clusters": final_clusters,
        "merge_proposals": [p.to_dict() for p in proposals],
        "session_stats": {
            "n_crops": len(enriched),
            "n_runs": n_initial_runs,
            "n_initial_clusters": n_initial_clusters,
            "n_final_clusters": len(final_clusters),
        },
        "enriched_crops": enriched,
        "diagnostics": {
            "feature_space": feature_diag,
            "clustering_quality": cluster_diag,
            "cue_effectiveness": cue_diag,
            "advisories": advisories,
        },
    }


# ---------------------------------------------------------------------------
# Diagnostics: Feature Space, Clustering Quality, Cue Effectiveness
# ---------------------------------------------------------------------------


def compute_feature_space_diagnostics(
    enriched: List[EnrichedCrop],
) -> Dict[str, Any]:
    """Compute pre-clustering feature space metrics (M1-M5).

    Uses only cached features and histograms — no video reads, no GPU.
    O(N^2) pairwise for N crops, trivially fast for N<300.

    Returns dict with keys: dinov3_similarity, color_distance, modality,
    spatial, temporal.
    """
    n = len(enriched)
    if n == 0:
        return {}

    # M1: DINOv3 pairwise cosine similarity
    features = np.stack([c.dinov3_cls for c in enriched])  # (N, D)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = features / norms
    sim_matrix = normed @ normed.T  # (N, N)
    np.fill_diagonal(sim_matrix, np.nan)  # exclude self-similarity
    upper = sim_matrix[np.triu_indices(n, k=1)]

    hist_bins = np.linspace(0.0, 1.0, 11)  # 10 bins
    hist_counts, _ = np.histogram(upper, bins=hist_bins)

    dinov3_sim = {
        "min": float(np.nanmin(upper)) if len(upper) > 0 else 0.0,
        "max": float(np.nanmax(upper)) if len(upper) > 0 else 0.0,
        "mean": float(np.nanmean(upper)) if len(upper) > 0 else 0.0,
        "std": float(np.nanstd(upper)) if len(upper) > 0 else 0.0,
        "histogram": {
            "bins": [float(b) for b in hist_bins],
            "counts": [int(c) for c in hist_counts],
        },
    }

    # M2: Color histogram distance distribution
    hists_available = [c for c in enriched if c.color_hist is not None]
    n_with_hists = len(hists_available)
    color_dist: Dict[str, Any] = {"n_with_histograms": n_with_hists, "n_total": n}

    if n_with_hists >= 2:
        # Compute pairwise histogram intersection distances
        color_dists_list = []
        for i in range(n_with_hists):
            for j in range(i + 1, n_with_hists):
                inter = _histogram_intersection(
                    hists_available[i].color_hist,
                    hists_available[j].color_hist,
                )
                color_dists_list.append(1.0 - inter)
        cd = np.array(color_dists_list)
        color_dist.update({
            "min": float(cd.min()),
            "max": float(cd.max()),
            "mean": float(cd.mean()),
            "std": float(cd.std()),
        })

    # M3: Modality
    modality_counts: Dict[str, int] = {"rgb": 0, "grayscale": 0, "ir": 0}
    for c in enriched:
        modality_counts[c.modality] = modality_counts.get(c.modality, 0) + 1
    dominant_modality = max(modality_counts, key=modality_counts.get)

    # M4: Spatial diversity
    positions = np.array(
        [c.scene_pos if c.scene_pos is not None else [0.5, 0.5] for c in enriched],
        dtype=np.float64,
    )
    spatial_std_x = float(positions[:, 0].std())
    spatial_std_y = float(positions[:, 1].std())
    quadrant_counts = [0] * 9
    for c in enriched:
        quadrant_counts[c.quadrant] += 1
    n_quadrants_occupied = sum(1 for q in quadrant_counts if q > 0)

    # M5: Temporal coverage
    frame_indices = sorted([c.frame_idx for c in enriched])
    frame_span = [frame_indices[0], frame_indices[-1]] if frame_indices else [0, 0]
    unique_frames = sorted(set(frame_indices))
    n_frames = len(unique_frames)

    # Co-occurrence: frames with >1 crop
    from collections import Counter
    frame_counts = Counter(c.frame_idx for c in enriched)
    cooccurrence_frames = sum(1 for cnt in frame_counts.values() if cnt > 1)
    cooccurrence_pairs = sum(
        cnt * (cnt - 1) // 2 for cnt in frame_counts.values() if cnt > 1
    )

    # Max gap between consecutive unique frames
    max_gap_frames = 0
    if len(unique_frames) > 1:
        gaps = [unique_frames[i + 1] - unique_frames[i] for i in range(len(unique_frames) - 1)]
        max_gap_frames = max(gaps)

    return {
        "n_crops": n,
        "dinov3_similarity": dinov3_sim,
        "color_distance": color_dist,
        "modality": {
            "dominant": dominant_modality,
            "counts": modality_counts,
        },
        "spatial": {
            "std_x": spatial_std_x,
            "std_y": spatial_std_y,
            "n_quadrants_occupied": n_quadrants_occupied,
            "quadrant_counts": quadrant_counts,
        },
        "temporal": {
            "frame_span": frame_span,
            "n_frames_with_crops": n_frames,
            "n_cooccurrence_frames": cooccurrence_frames,
            "n_cooccurrence_pairs": cooccurrence_pairs,
            "max_gap_frames": max_gap_frames,
        },
    }


def compute_clustering_diagnostics(
    clusters: Dict[int, List[int]],
    run_map: Dict[int, RunGroup],
    weights: CueWeights,
    enriched: List[EnrichedCrop],
) -> Dict[str, Any]:
    """Compute post-clustering quality metrics (M6-M9).

    M6: Silhouette-like cohesion score (per-cluster and global).
    M7: Cluster size balance (CV, singletons, giants).
    M8: Intra-cluster coherence (mean pairwise DINOv3 sim per cluster).
    M9: Inter-cluster separation (centroid cosine distances).
    """
    if not clusters or not run_map:
        return {}

    cluster_ids = sorted(clusters.keys())
    n_clusters = len(cluster_ids)

    # Build crop_id -> feature lookup for coherence
    crop_features: Dict[str, np.ndarray] = {c.crop_id: c.dinov3_cls for c in enriched}

    # M7: Cluster size balance
    sizes = []
    for cid in cluster_ids:
        # Count total crops in cluster
        total_crops = sum(len(run_map[rid].crop_ids) for rid in clusters[cid] if rid in run_map)
        sizes.append(total_crops)

    sizes_arr = np.array(sizes, dtype=np.float64)
    mean_size = float(sizes_arr.mean()) if len(sizes_arr) > 0 else 0
    cv = float(sizes_arr.std() / mean_size) if mean_size > 0 else 0.0
    n_singletons = int(sum(1 for s in sizes if s == 1))
    n_giants = int(sum(1 for s in sizes if mean_size > 0 and s > 2 * mean_size))

    size_balance = {
        "sizes": [int(s) for s in sorted(sizes, reverse=True)],
        "cv": cv,
        "n_singletons": n_singletons,
        "n_giants": n_giants,
        "mean_size": mean_size,
    }

    # M8: Intra-cluster coherence (DINOv3 cosine similarity)
    intra_coherence: Dict[str, Dict[str, Any]] = {}
    for idx, cid in enumerate(cluster_ids):
        crop_ids = []
        for rid in clusters[cid]:
            rg = run_map.get(rid)
            if rg:
                crop_ids.extend(rg.crop_ids)

        if len(crop_ids) < 2:
            intra_coherence[f"identity_{idx}"] = {
                "coherence": 1.0,
                "min_pair_sim": 1.0,
                "n_members": len(crop_ids),
            }
            continue

        feats = [crop_features[cid_c] for cid_c in crop_ids if cid_c in crop_features]
        if len(feats) < 2:
            intra_coherence[f"identity_{idx}"] = {
                "coherence": 1.0,
                "min_pair_sim": 1.0,
                "n_members": len(crop_ids),
            }
            continue

        feat_stack = np.stack(feats)
        norms = np.linalg.norm(feat_stack, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = feat_stack / norms
        sim = normed @ normed.T
        np.fill_diagonal(sim, np.nan)
        upper_tri = sim[np.triu_indices(len(feats), k=1)]

        intra_coherence[f"identity_{idx}"] = {
            "coherence": float(np.nanmean(upper_tri)),
            "min_pair_sim": float(np.nanmin(upper_tri)),
            "n_members": len(crop_ids),
        }

    # M9: Inter-cluster separation (centroid cosine distances)
    centroids = {}
    for idx, cid in enumerate(cluster_ids):
        feat, _, _, _, _ = _cluster_centroid(clusters[cid], run_map)
        centroids[f"identity_{idx}"] = feat

    inter_separation: Dict[str, Any] = {}
    if n_clusters >= 2:
        centroid_names = sorted(centroids.keys())
        pairwise_dists: Dict[str, float] = {}
        min_dist = float("inf")
        closest_pair = None
        dist_values = []

        for i in range(len(centroid_names)):
            for j in range(i + 1, len(centroid_names)):
                name_a, name_b = centroid_names[i], centroid_names[j]
                d = _cosine_distance(centroids[name_a], centroids[name_b])
                key = f"{name_a}_vs_{name_b}"
                pairwise_dists[key] = d
                dist_values.append(d)
                if d < min_dist:
                    min_dist = d
                    closest_pair = [name_a, name_b]

        inter_separation = {
            "min_distance": float(min_dist),
            "mean_distance": float(np.mean(dist_values)) if dist_values else 0.0,
            "closest_pair": closest_pair,
            "pairwise": pairwise_dists,
        }

    # M6: Simplified silhouette score
    # For each cluster: ratio of mean inter-cluster distance to mean intra-cluster distance
    silhouette_per_cluster: Dict[str, float] = {}
    if n_clusters >= 2:
        for idx, cid in enumerate(cluster_ids):
            name = f"identity_{idx}"
            # Intra: mean distance to same-cluster centroid
            intra_d = 1.0 - intra_coherence.get(name, {}).get("coherence", 1.0)
            # Inter: min distance to nearest other centroid
            min_inter = float("inf")
            for other_idx, other_cid in enumerate(cluster_ids):
                if other_cid == cid:
                    continue
                other_name = f"identity_{other_idx}"
                d = _cosine_distance(centroids.get(name, np.zeros(1)), centroids.get(other_name, np.zeros(1)))
                if d < min_inter:
                    min_inter = d

            if min_inter == float("inf"):
                min_inter = 0.0

            # Silhouette-like: (inter - intra) / max(inter, intra)
            denom = max(min_inter, intra_d)
            sil = (min_inter - intra_d) / denom if denom > 1e-12 else 0.0
            silhouette_per_cluster[name] = float(sil)

    avg_silhouette = (
        float(np.mean(list(silhouette_per_cluster.values())))
        if silhouette_per_cluster else 0.0
    )

    return {
        "k": n_clusters,
        "silhouette_avg": avg_silhouette,
        "silhouette_per_cluster": silhouette_per_cluster,
        "size_balance": size_balance,
        "intra_cluster_coherence": intra_coherence,
        "inter_cluster_separation": inter_separation,
    }


def compute_cue_effectiveness(
    enriched: List[EnrichedCrop],
    runs: List[RunGroup],
    clusters: Dict[int, List[int]],
    run_map: Dict[int, RunGroup],
    weights: CueWeights,
) -> Dict[str, Any]:
    """Compute per-cue discriminative power (M10) and recommended weights (M12).

    Fisher discriminant ratio per cue: measures how well each cue separates
    same-cluster pairs from different-cluster pairs.

    Returns dict with fisher_ratios, recommended_weights, current_weights, max_delta.
    """
    if len(clusters) < 2 or not runs:
        return {}

    # Build run_id -> cluster_id mapping
    run_to_cluster: Dict[int, int] = {}
    for cid, rids in clusters.items():
        for rid in rids:
            run_to_cluster[rid] = cid

    cue_names = ["appearance", "spatial", "body", "color", "context", "temporal"]

    # Compute per-cue distances for all run pairs, separated by same/different cluster
    intra_dists: Dict[str, List[float]] = {c: [] for c in cue_names}
    inter_dists: Dict[str, List[float]] = {c: [] for c in cue_names}

    run_ids = [r.run_id for r in runs]
    for i in range(len(run_ids)):
        for j in range(i + 1, len(run_ids)):
            rid_a, rid_b = run_ids[i], run_ids[j]
            rg_a, rg_b = run_map.get(rid_a), run_map.get(rid_b)
            if rg_a is None or rg_b is None:
                continue

            cluster_a = run_to_cluster.get(rid_a)
            cluster_b = run_to_cluster.get(rid_b)
            if cluster_a is None or cluster_b is None:
                continue

            same = cluster_a == cluster_b
            target = intra_dists if same else inter_dists

            # Per-cue distances (mirrors _compute_run_distance decomposition)
            target["appearance"].append(_cosine_distance(rg_a.mean_features, rg_b.mean_features))
            target["spatial"].append(float(np.linalg.norm(rg_a.mean_scene_pos - rg_b.mean_scene_pos)))

            if rg_a.mean_body_props is not None and rg_b.mean_body_props is not None:
                target["body"].append(float(np.linalg.norm(rg_a.mean_body_props - rg_b.mean_body_props)))
            else:
                target["body"].append(0.5)

            if rg_a.mean_color_hist is not None and rg_b.mean_color_hist is not None:
                target["color"].append(1.0 - _histogram_intersection(rg_a.mean_color_hist, rg_b.mean_color_hist))
            else:
                target["color"].append(0.5)

            if rg_a.mean_context_cls is not None and rg_b.mean_context_cls is not None:
                target["context"].append(_cosine_distance(rg_a.mean_context_cls, rg_b.mean_context_cls))
            else:
                target["context"].append(0.5)

            # Temporal
            a_s, a_e = rg_a.frame_range
            b_s, b_e = rg_b.frame_range
            ol = max(0, min(a_e, b_e) - max(a_s, b_s))
            un = max(a_e, b_e) - min(a_s, b_s)
            target["temporal"].append(ol / max(un, 1))

    # M10: Fisher discriminant ratio per cue
    fisher_ratios: Dict[str, float] = {}
    for cue in cue_names:
        intra = np.array(intra_dists[cue]) if intra_dists[cue] else np.array([0.5])
        inter = np.array(inter_dists[cue]) if inter_dists[cue] else np.array([0.5])

        mean_diff_sq = (float(inter.mean()) - float(intra.mean())) ** 2
        var_sum = float(intra.var()) + float(inter.var())

        fisher_ratios[cue] = mean_diff_sq / var_sum if var_sum > 1e-12 else 0.0

    # M12: Recommended weights (normalized Fisher ratios)
    total_fisher = sum(fisher_ratios.values())
    recommended: Dict[str, float] = {}
    if total_fisher > 1e-12:
        for cue in cue_names:
            recommended[cue] = fisher_ratios[cue] / total_fisher
    else:
        # All cues are equally uninformative — use uniform
        for cue in cue_names:
            recommended[cue] = 1.0 / len(cue_names)

    current = weights.to_dict()
    max_delta = max(abs(recommended.get(c, 0) - current.get(c, 0)) for c in cue_names)

    return {
        "fisher_ratios": {c: round(v, 4) for c, v in fisher_ratios.items()},
        "recommended_weights": {c: round(v, 4) for c, v in recommended.items()},
        "current_weights": {c: round(v, 4) for c, v in current.items()},
        "max_weight_delta": round(max_delta, 4),
    }


def generate_advisories(
    feature_space: Dict[str, Any],
    clustering: Dict[str, Any],
    cue_eff: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Synthesize actionable advisory messages from diagnostic metrics.

    Each advisory: {level: "success"|"info"|"warning"|"error", message: str, action: str|None}
    """
    advisories: List[Dict[str, Any]] = []

    # Feature space advisories
    if feature_space:
        n = feature_space.get("n_crops", 0)
        advisories.append({
            "level": "info",
            "message": f"{n} accepted crops",
            "action": None,
        })

        # Modality
        mod = feature_space.get("modality", {})
        dominant = mod.get("dominant", "rgb")
        if dominant in ("ir", "grayscale"):
            advisories.append({
                "level": "warning",
                "message": f"{dominant.upper()} video detected. Color cue is unreliable.",
                "action": "Set color weight to 0, increase appearance weight.",
            })

        # DINOv3 similarity spread
        dino = feature_space.get("dinov3_similarity", {})
        std = dino.get("std", 0)
        mean = dino.get("mean", 0)
        if std < 0.08:
            advisories.append({
                "level": "warning",
                "message": f"Low appearance diversity (std={std:.3f}). People look similar to DINOv3.",
                "action": "Consider increasing color and spatial weights.",
            })
        elif mean > 0.85 and std < 0.12:
            advisories.append({
                "level": "warning",
                "message": f"High mean similarity ({mean:.2f}). Crops may be near-identical.",
                "action": "Check for duplicate crops or very similar appearances.",
            })

        # Color
        color = feature_space.get("color_distance", {})
        n_hist = color.get("n_with_histograms", 0)
        n_total = color.get("n_total", 0)
        if n_total > 0 and n_hist < n_total * 0.5:
            advisories.append({
                "level": "warning",
                "message": f"Only {n_hist}/{n_total} crops have color histograms. Color cue is partial.",
                "action": "Color weight may be ineffective with missing data.",
            })

        # Temporal
        temporal = feature_space.get("temporal", {})
        cooc = temporal.get("n_cooccurrence_frames", 0)
        if cooc == 0:
            advisories.append({
                "level": "info",
                "message": "No frames with multiple people. Cannot-link constraints unavailable.",
                "action": "Clustering relies solely on feature similarity.",
            })
        elif cooc > 5:
            advisories.append({
                "level": "success",
                "message": f"{cooc} frames with multiple people — strong co-occurrence signal.",
                "action": None,
            })

    # Clustering advisories
    if clustering:
        sil = clustering.get("silhouette_avg", 0)
        if sil > 0.5:
            advisories.append({
                "level": "success",
                "message": f"Good cluster separation (silhouette={sil:.2f}).",
                "action": None,
            })
        elif sil > 0.25:
            advisories.append({
                "level": "info",
                "message": f"Moderate cluster separation (silhouette={sil:.2f}).",
                "action": "Some identities may overlap. Review merge proposals.",
            })
        else:
            advisories.append({
                "level": "warning",
                "message": f"Weak cluster separation (silhouette={sil:.2f}).",
                "action": "Consider adjusting K or cue weights.",
            })

        balance = clustering.get("size_balance", {})
        singletons = balance.get("n_singletons", 0)
        k = clustering.get("k", 0)
        if k > 0 and singletons > k * 0.3:
            advisories.append({
                "level": "warning",
                "message": f"{singletons} singleton clusters. May indicate over-splitting.",
                "action": "Try lower K or increase distance threshold.",
            })

        giants = balance.get("n_giants", 0)
        if giants > 0:
            advisories.append({
                "level": "warning",
                "message": f"{giants} oversized cluster(s). Possible under-splitting.",
                "action": "Try higher K or decrease distance threshold.",
            })

        # Closest pair
        sep = clustering.get("inter_cluster_separation", {})
        min_d = sep.get("min_distance")
        closest = sep.get("closest_pair")
        if min_d is not None and min_d < 0.15 and closest:
            advisories.append({
                "level": "warning",
                "message": f"Close clusters: {closest[0]} and {closest[1]} (dist={min_d:.3f}).",
                "action": "Review the top merge proposal.",
            })

    # Cue effectiveness advisories
    if cue_eff:
        fisher = cue_eff.get("fisher_ratios", {})
        for cue, ratio in fisher.items():
            if ratio < 0.1 and cue_eff.get("current_weights", {}).get(cue, 0) > 0.1:
                advisories.append({
                    "level": "warning",
                    "message": f"{cue.capitalize()} cue ineffective (Fisher={ratio:.2f}) but weight is high.",
                    "action": f"Reduce {cue} weight or click 'Apply Recommended'.",
                })

        max_delta = cue_eff.get("max_weight_delta", 0)
        if max_delta > 0.15:
            advisories.append({
                "level": "info",
                "message": f"Weights differ from data-recommended by up to {max_delta:.2f}.",
                "action": "Consider applying recommended weights.",
            })

    return advisories


def apply_verdicts_and_recluster(
    session: InterviewSession,
    verdicts: List[Dict],
    weights: CueWeights,
    read_frame: Callable,
    proposals: Optional[List[MergeProposal]] = None,
    cumulative_verdicts: int = 0,
) -> Dict[str, Any]:
    """Apply human merge verdicts, update cue weights, and re-run the pipeline.

    Each verdict maps a merge proposal to "same" (merge the two clusters)
    or "different" (keep them apart). The per-cue evidence from the proposal
    is used to adjust cue weights via gradient descent, making the pipeline
    learn which cues are reliable for this video.

    Args:
        session: InterviewSession with accepted crops.
        verdicts: List of dicts with "proposal_idx" (int) and "verdict"
            ("same" or "different") keys.
        weights: Current CueWeights — updated in-place by this function.
        read_frame: Frame reader callable.
        proposals: List of MergeProposal objects from the previous pipeline
            run. Required when verdicts are non-empty.
        cumulative_verdicts: Total verdicts submitted across all rounds
            (before this call). Used for convergence budget tracking.

    Returns:
        Same dict structure as run_visual_reid_pipeline(), plus
        "converged" (bool) and "weights" (dict) keys.
    """
    # Convert verdicts to the format expected by update_weights()
    verdict_tuples: List[Tuple[MergeProposal, str]] = []
    if verdicts and proposals:
        for v in verdicts:
            idx = v.get("proposal_idx")
            label = v.get("verdict")
            if idx is not None and 0 <= idx < len(proposals) and label in ("same", "different"):
                verdict_tuples.append((proposals[idx], label))

    # Update cue weights based on verdicts
    if verdict_tuples:
        weights = update_weights(weights, verdict_tuples)

    # Check convergence using cumulative verdicts across all rounds
    converged = check_convergence(
        proposals or [],
        max_verdicts=10,
        current_verdicts=cumulative_verdicts + len(verdict_tuples),
    )

    # Re-run pipeline with updated weights
    result = run_visual_reid_pipeline(session, read_frame, weights)
    result["converged"] = converged
    result["weights"] = weights.to_dict()
    result["valid_verdicts_count"] = len(verdict_tuples)
    return result
