"""UFM-based ReID clustering and assignment.

Replaces the DINOv3 + COP-HAC pipeline with:
  1. UFM covisibility as the sole similarity signal
  2. Average-linkage HAC for initial clustering (user specifies k)
  3. Temporal co-occurrence warnings (UI-only, not a hard constraint)
  4. Multi-select crop assignment for human corrections

No multi-cue fusion, no constraint propagation, no auto-estimation of k.
The human specifies k and directly assigns crops to clusters.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)
SAME_FRAME_NMS_IOU = float(os.getenv("INTERVIEW_REID_SAME_FRAME_NMS_IOU", "0.7"))
SAME_FRAME_CANNOT_LINK_IOU = float(os.getenv("INTERVIEW_REID_SAME_FRAME_CANNOT_LINK_IOU", "0.7"))


# ---------------------------------------------------------------------------
# 1. HAC clustering
# ---------------------------------------------------------------------------

def cluster_hac(
    sim_matrix: np.ndarray,
    k: int,
    method: str = "average",
) -> np.ndarray:
    """Cluster using agglomerative HAC on a similarity matrix.

    Converts similarity to distance (1 - sim), runs scipy linkage,
    and cuts at k clusters.

    Args:
        sim_matrix: (N, N) symmetric similarity matrix with 1.0 on diagonal.
        k: Number of clusters to produce.
        method: Linkage method (default "average").

    Returns:
        (N,) integer array of cluster assignments in [0, k).
    """
    n = sim_matrix.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.intp)

    dist = np.clip(1.0 - sim_matrix, 0, None)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed
    return labels.astype(np.intp)


def build_ufm_pair_plan(
    crop_ids: List[str],
    crops: Dict[str, Any],
    same_frame_nms_iou: float = SAME_FRAME_NMS_IOU,
    same_frame_cannot_link_iou: float = SAME_FRAME_CANNOT_LINK_IOU,
) -> Dict[str, Any]:
    """Build representative crops and candidate UFM pairs.

    Strategy:
      1. Per-frame NMS on accepted crops to suppress near-duplicate boxes.
      2. Pre-prune same-frame pairs with IoU below threshold (cannot-link),
         so those pairs skip UFM inference entirely.

    Returns:
        Dict with keys:
          rep_indices: indices into crop_ids used as UFM representatives.
          rep_of_full: full crop index -> representative index mapping.
          pair_indices: representative-index pairs to run through UFM.
          stats: counters for logging/telemetry.
    """
    from .detection import _compute_iou_matrix, nms_numpy

    n = len(crop_ids)
    if n == 0:
        return {
            "rep_indices": [],
            "rep_of_full": [],
            "pair_indices": [],
            "stats": {
                "n_all_crops": 0,
                "n_representatives": 0,
                "n_nms_suppressed": 0,
                "n_all_pairs": 0,
                "n_representative_pairs": 0,
                "n_pruned_same_frame_pairs": 0,
                "n_ufm_pairs": 0,
            },
        }

    # Start with identity mapping: each crop is its own representative.
    rep_global_of_full = list(range(n))

    # Group by frame index for per-frame NMS.
    frame_to_full: Dict[int, List[int]] = {}
    for full_idx, cid in enumerate(crop_ids):
        crop = crops.get(cid)
        frame_idx = int(crop.frame_idx) if crop is not None else -1
        frame_to_full.setdefault(frame_idx, []).append(full_idx)

    for full_indices in frame_to_full.values():
        if len(full_indices) < 2:
            continue

        boxes = np.array(
            [crops[crop_ids[idx]].xyxy for idx in full_indices], dtype=np.float32,
        )
        # Tiny tie-break to preserve deterministic early-index preference.
        base_scores = np.array(
            [float(getattr(crops[crop_ids[idx]], "score", 0.0)) for idx in full_indices],
            dtype=np.float32,
        )
        tie_break = (len(full_indices) - 1 - np.arange(len(full_indices), dtype=np.float32)) * 1e-6
        keep_local = nms_numpy(
            boxes,
            base_scores + tie_break,
            iou_threshold=same_frame_nms_iou,
        )
        keep_local_set = {int(k) for k in keep_local.tolist()}

        if len(keep_local_set) == len(full_indices):
            continue

        iou = _compute_iou_matrix(boxes, boxes)
        for local_idx, full_idx in enumerate(full_indices):
            if local_idx in keep_local_set:
                rep_global_of_full[full_idx] = full_idx
                continue

            # Map suppressed crop to the kept crop with max IoU.
            best_keep = max(
                keep_local_set,
                key=lambda k: float(iou[local_idx, k]),
            )
            if float(iou[local_idx, best_keep]) >= same_frame_nms_iou:
                rep_global_of_full[full_idx] = full_indices[best_keep]
            else:
                # Safety fallback: leave as its own representative.
                rep_global_of_full[full_idx] = full_idx

    # Build compact representative index space.
    rep_indices: List[int] = []
    global_rep_to_local: Dict[int, int] = {}
    for rep_global in rep_global_of_full:
        if rep_global not in global_rep_to_local:
            global_rep_to_local[rep_global] = len(rep_indices)
            rep_indices.append(rep_global)

    rep_of_full = [global_rep_to_local[rep_global_of_full[i]] for i in range(n)]

    # Pre-prune impossible same-frame pairs among representatives.
    rep_frame_to_local: Dict[int, List[int]] = {}
    for rep_local, full_idx in enumerate(rep_indices):
        cid = crop_ids[full_idx]
        crop = crops.get(cid)
        frame_idx = int(crop.frame_idx) if crop is not None else -1
        rep_frame_to_local.setdefault(frame_idx, []).append(rep_local)

    pruned_same_frame_pairs: set[Tuple[int, int]] = set()
    for rep_local_indices in rep_frame_to_local.values():
        if len(rep_local_indices) < 2:
            continue

        boxes = np.array(
            [crops[crop_ids[rep_indices[i]]].xyxy for i in rep_local_indices],
            dtype=np.float32,
        )
        iou = _compute_iou_matrix(boxes, boxes)
        for a in range(len(rep_local_indices)):
            for b in range(a + 1, len(rep_local_indices)):
                if float(iou[a, b]) < same_frame_cannot_link_iou:
                    i = rep_local_indices[a]
                    j = rep_local_indices[b]
                    pruned_same_frame_pairs.add((i, j) if i < j else (j, i))

    pair_indices: List[Tuple[int, int]] = []
    m = len(rep_indices)
    for i in range(m):
        for j in range(i + 1, m):
            if (i, j) in pruned_same_frame_pairs:
                continue
            pair_indices.append((i, j))

    n_nms_suppressed = sum(
        1 for full_idx, rep_global in enumerate(rep_global_of_full) if full_idx != rep_global
    )
    n_all_pairs = n * (n - 1) // 2
    n_rep_pairs = m * (m - 1) // 2
    stats = {
        "n_all_crops": n,
        "n_representatives": m,
        "n_nms_suppressed": n_nms_suppressed,
        "n_all_pairs": n_all_pairs,
        "n_representative_pairs": n_rep_pairs,
        "n_pruned_same_frame_pairs": len(pruned_same_frame_pairs),
        "n_ufm_pairs": len(pair_indices),
    }
    return {
        "rep_indices": rep_indices,
        "rep_of_full": rep_of_full,
        "pair_indices": pair_indices,
        "stats": stats,
    }


def apply_same_frame_constraints(
    sim_matrix: np.ndarray,
    crop_ids: List[str],
    crops: Dict[str, Any],
    iou_threshold: float = 0.3,
) -> int:
    """Zero similarity for same-frame non-overlapping accepted crops.

    Two accepted crops at different spatial locations on the same frame
    are physically different people.  Zeroing their similarity forces
    HAC to treat them as maximally distant, preventing same-cluster
    assignment for any reasonable k.

    Corrected crops (BOX_CORRECTED) are excluded — they spatially
    overlap with their rejected parent and don't represent independent
    spatial assertions.

    Modifies *sim_matrix* **in place**.

    Args:
        sim_matrix: (N, N) symmetric similarity matrix.
        crop_ids: ordered crop IDs matching matrix rows/cols.
        crops: crop_id -> CropData mapping.
        iou_threshold: pairs with IoU >= this are NOT constrained
            (they may be near-duplicates rather than distinct people).

    Returns:
        Number of pairs zeroed.
    """
    from .state import CropSource
    from .detection import _compute_iou_matrix

    # Build crop_id -> matrix index, skipping corrected crops
    id_to_idx: Dict[str, int] = {}
    for idx, cid in enumerate(crop_ids):
        crop = crops.get(cid)
        if crop is None:
            continue
        if getattr(crop, "source", None) == CropSource.BOX_CORRECTED:
            continue
        id_to_idx[cid] = idx

    # Group by frame
    frame_map: Dict[int, List[str]] = {}
    for cid, idx in id_to_idx.items():
        crop = crops[cid]
        frame_map.setdefault(crop.frame_idx, []).append(cid)

    n_zeroed = 0
    for frame_idx, cids in frame_map.items():
        if len(cids) < 2:
            continue

        # Build box array for IoU computation
        boxes = np.array([crops[cid].xyxy for cid in cids], dtype=np.float32)
        iou = _compute_iou_matrix(boxes, boxes)

        # Zero pairs with IoU below threshold
        for a in range(len(cids)):
            for b in range(a + 1, len(cids)):
                if iou[a, b] < iou_threshold:
                    i, j = id_to_idx[cids[a]], id_to_idx[cids[b]]
                    sim_matrix[i, j] = 0.0
                    sim_matrix[j, i] = 0.0
                    n_zeroed += 1

    return n_zeroed


def silhouette_score(sim_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score from a similarity matrix and labels.

    Args:
        sim_matrix: (N, N) similarity matrix.
        labels: (N,) cluster assignments.

    Returns:
        Average silhouette score in [-1, 1]. Higher is better.
    """
    n = len(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    dist = np.clip(1.0 - sim_matrix, 0, None)
    np.fill_diagonal(dist, 0)

    sils = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a_i = dist[i, same].mean()
        b_i = min(
            dist[i, labels == lbl].mean()
            for lbl in unique if lbl != labels[i]
        )
        sils[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)
    return float(sils.mean())


# ---------------------------------------------------------------------------
# 2. Temporal co-occurrence analysis
# ---------------------------------------------------------------------------

def compute_co_occurrence_warnings(
    clusters: Dict[int, List[str]],
    crops: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect crops from the same frame assigned to the same cluster.

    Two distinct people cannot occupy the exact same bounding box in the
    same frame, so same-frame crops in one cluster likely indicate a
    clustering error. This returns warnings for the UI — NOT hard
    constraints.

    Args:
        clusters: cluster_id -> [crop_ids].
        crops: crop_id -> CropData (must have .frame_idx).

    Returns:
        List of warning dicts: {cluster_id, frame_idx, crop_ids}.
    """
    warnings = []
    for cluster_id, crop_ids in clusters.items():
        # Group by frame
        frame_map: Dict[int, List[str]] = {}
        for cid in crop_ids:
            crop = crops.get(cid)
            if crop is None:
                continue
            frame_map.setdefault(crop.frame_idx, []).append(cid)

        for frame_idx, cids_in_frame in frame_map.items():
            if len(cids_in_frame) > 1:
                warnings.append({
                    "cluster_id": cluster_id,
                    "frame_idx": frame_idx,
                    "crop_ids": cids_in_frame,
                })

    return warnings


# ---------------------------------------------------------------------------
# 3. Cluster assignment helpers
# ---------------------------------------------------------------------------

def assign_crops_to_cluster(
    session,
    crop_ids: List[str],
    target_cluster_id: int,
) -> Dict[str, Any]:
    """Move selected crops into a target cluster.

    Removes each crop from its current cluster (if any) and adds it to
    the target cluster. Creates the target cluster if it doesn't exist.

    Args:
        session: InterviewSession with reid_clusters.
        crop_ids: Crops to move.
        target_cluster_id: Destination cluster.

    Returns:
        Summary dict with updated cluster info.
    """
    # Ensure target cluster exists
    if target_cluster_id not in session.reid_clusters:
        session.reid_clusters[target_cluster_id] = []

    for crop_id in crop_ids:
        # Remove from any existing cluster
        for cid, members in session.reid_clusters.items():
            if crop_id in members:
                members.remove(crop_id)

        # Add to target
        if crop_id not in session.reid_clusters[target_cluster_id]:
            session.reid_clusters[target_cluster_id].append(crop_id)

        # Update crop's reid_cluster_id
        crop = session.get_crop(crop_id)
        if crop is not None:
            crop.reid_cluster_id = target_cluster_id

    # Clean up empty clusters
    empty = [k for k, v in session.reid_clusters.items() if not v]
    for k in empty:
        del session.reid_clusters[k]

    session.n_identities = len(session.reid_clusters)
    session.touch()

    return _build_cluster_summary(session)


def create_new_cluster(
    session,
    crop_ids: List[str],
) -> Dict[str, Any]:
    """Create a new cluster and move the selected crops into it.

    Args:
        session: InterviewSession with reid_clusters.
        crop_ids: Crops to assign to the new cluster.

    Returns:
        Summary dict including the new cluster_id.
    """
    existing_keys = list(session.reid_clusters.keys())
    new_id = (max(existing_keys) + 1) if existing_keys else 0
    session.reid_clusters[new_id] = []

    for crop_id in crop_ids:
        # Remove from old cluster
        for cid, members in session.reid_clusters.items():
            if cid != new_id and crop_id in members:
                members.remove(crop_id)

        session.reid_clusters[new_id].append(crop_id)
        crop = session.get_crop(crop_id)
        if crop is not None:
            crop.reid_cluster_id = new_id

    # Clean up empty clusters
    empty = [k for k, v in session.reid_clusters.items() if not v]
    for k in empty:
        del session.reid_clusters[k]

    session.n_identities = len(session.reid_clusters)
    session.touch()

    result = _build_cluster_summary(session)
    result["new_cluster_id"] = new_id
    return result


# ---------------------------------------------------------------------------
# 4. Full pipeline
# ---------------------------------------------------------------------------

def run_ufm_reid_pipeline(
    session,
    n_clusters: int,
    progress,
    read_frame_fn,
) -> Dict[str, Any]:
    """Full UFM-based ReID pipeline.

    Steps:
        1. Extract crop images from accepted crops
        2. Compute UFM pairwise similarity matrix
        3. Run HAC clustering with user-specified k
        4. Compute co-occurrence warnings
        5. Save to session

    Args:
        session: InterviewSession in REID phase.
        n_clusters: User-specified number of identities.
        progress: JobProgress for status reporting.
        read_frame_fn: Frame reader function.

    Returns:
        Summary dict.
    """
    from .ufm_model import compute_pairwise_similarity, extract_crop_images_from_session
    from .cache_manager import save_session

    progress.step = "Extracting crop images"
    progress.total = 4
    progress.current = 0
    progress.eta_seconds = None
    progress.items_per_second = None

    # Step 1: Extract crop images
    crop_images, crop_ids = extract_crop_images_from_session(
        session, read_frame_fn,
    )
    n = len(crop_images)
    if n < 2:
        raise ValueError(
            f"Need at least 2 accepted crops for ReID, got {n}"
        )
    progress.current = 1
    logger.info("UFM ReID: %d accepted crops", n)

    # Step 2: Build pair plan (same-frame NMS + cannot-link pruning),
    # then run UFM only on representative candidate pairs.
    pair_plan = build_ufm_pair_plan(
        crop_ids,
        session.crops,
        same_frame_nms_iou=SAME_FRAME_NMS_IOU,
        same_frame_cannot_link_iou=SAME_FRAME_CANNOT_LINK_IOU,
    )
    rep_indices = pair_plan["rep_indices"]
    rep_of_full = np.asarray(pair_plan["rep_of_full"], dtype=np.intp)
    pair_indices = pair_plan["pair_indices"]
    pair_stats = pair_plan["stats"]
    rep_images = [crop_images[idx] for idx in rep_indices]
    naive_pairs = int(pair_stats["n_all_pairs"])
    n_pairs = int(pair_stats["n_ufm_pairs"])
    avoided_pairs = max(naive_pairs - n_pairs, 0)
    reduction_pct = (100.0 * avoided_pairs / naive_pairs) if naive_pairs > 0 else 0.0
    reduction_label = f"avoided {avoided_pairs}/{naive_pairs} ({reduction_pct:.1f}% cut)"

    logger.info(
        "UFM ReID pair plan: crops=%d reps=%d nms_suppressed=%d pairs=%d->%d "
        "(pruned_same_frame=%d)",
        pair_stats["n_all_crops"],
        pair_stats["n_representatives"],
        pair_stats["n_nms_suppressed"],
        pair_stats["n_all_pairs"],
        pair_stats["n_ufm_pairs"],
        pair_stats["n_pruned_same_frame_pairs"],
    )
    logger.info("UFM ReID pair reduction summary: %s", reduction_label)

    progress.total = n_pairs
    progress.current = 0
    progress.step = f"UFM pairs: 0/{n_pairs} [{reduction_label}]"
    progress.eta_seconds = None
    progress.items_per_second = None
    pair_t0 = time.time()

    def _progress_cb(done, total):
        progress.step = f"UFM pairs: {done}/{total} [{reduction_label}]"
        progress.current = done
        elapsed = max(time.time() - pair_t0, 1e-6)
        rate = done / elapsed if done > 0 else 0.0
        progress.items_per_second = rate if rate > 0 else None
        if rate > 0:
            progress.eta_seconds = max((total - done) / rate, 0.0)
        else:
            progress.eta_seconds = None

    rep_sim_matrix = compute_pairwise_similarity(
        rep_images,
        pair_indices=pair_indices,
        progress_callback=_progress_cb,
    )
    # Broadcast representative similarities back to the full accepted set.
    sim_matrix = np.array(
        rep_sim_matrix[np.ix_(rep_of_full, rep_of_full)],
        dtype=np.float32,
        copy=True,
    )
    np.fill_diagonal(sim_matrix, 1.0)

    # Restore to step-level for remaining steps
    progress.total = 4
    progress.current = 2
    progress.eta_seconds = None
    progress.items_per_second = None

    # Step 2.5: Enforce same-frame cannot-links before clustering
    n_constrained = apply_same_frame_constraints(
        sim_matrix, crop_ids, session.crops, iou_threshold=SAME_FRAME_CANNOT_LINK_IOU,
    )
    if n_constrained:
        logger.info("UFM ReID: zeroed %d same-frame pairs", n_constrained)

    # Step 3: HAC clustering
    progress.step = "Clustering identities"
    k = max(2, min(n_clusters, n - 1))
    labels = cluster_hac(sim_matrix, k)
    sil = silhouette_score(sim_matrix, labels)
    progress.current = 3

    # Build cluster mapping
    clusters: Dict[int, List[str]] = {}
    for idx, cid in enumerate(crop_ids):
        cluster_id = int(labels[idx])
        clusters.setdefault(cluster_id, []).append(cid)
        crop = session.get_crop(cid)
        if crop is not None:
            crop.reid_cluster_id = cluster_id

    # Step 4: Save to session
    progress.step = "Saving results"
    session.reid_clusters = clusters
    session.n_identities = len(clusters)
    session.ufm_similarity_matrix = sim_matrix
    session.ufm_crop_ids = crop_ids
    session.ufm_complete = True
    session.touch()
    save_session(session)
    progress.current = 4
    progress.eta_seconds = None
    progress.items_per_second = None

    # Compute warnings
    warnings = compute_co_occurrence_warnings(clusters, session.crops)

    cluster_sizes = {cid: len(m) for cid, m in clusters.items()}
    summary = {
        "n_clusters": len(clusters),
        "n_crops": n,
        "cluster_sizes": cluster_sizes,
        "silhouette": round(sil, 4),
        "co_occurrence_warnings": warnings,
        "ufm_pairs_naive": naive_pairs,
        "ufm_pairs_executed": n_pairs,
        "ufm_pairs_avoided": avoided_pairs,
        "ufm_pairs_reduction_pct": round(reduction_pct, 1),
        "ufm_representatives": int(pair_stats["n_representatives"]),
        "ufm_same_frame_nms_suppressed": int(pair_stats["n_nms_suppressed"]),
        "ufm_same_frame_pairs_pruned": int(pair_stats["n_pruned_same_frame_pairs"]),
    }
    logger.info(
        "UFM ReID complete: %d clusters, silhouette=%.3f, sizes=%s",
        summary["n_clusters"], sil, cluster_sizes,
    )
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cluster_summary(session) -> Dict[str, Any]:
    """Build cluster summary for API responses."""
    clusters_info = {}
    for cid, crop_ids in session.reid_clusters.items():
        clusters_info[str(cid)] = {
            "crop_ids": crop_ids,
            "count": len(crop_ids),
        }

    warnings = compute_co_occurrence_warnings(
        session.reid_clusters, session.crops,
    )

    return {
        "n_identities": session.n_identities,
        "clusters": clusters_info,
        "cluster_sizes": {cid: len(m) for cid, m in session.reid_clusters.items()},
        "co_occurrence_warnings": warnings,
    }
