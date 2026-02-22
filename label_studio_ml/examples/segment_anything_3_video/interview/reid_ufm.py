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
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


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

    # Step 2: Compute UFM pairwise similarity — this dominates runtime.
    # Switch to pair-level progress for a responsive bar.
    n_pairs = n * (n - 1) // 2
    progress.total = n_pairs
    progress.current = 0
    progress.step = "Computing UFM similarity (this takes a few minutes)"
    progress.eta_seconds = None
    progress.items_per_second = None
    pair_t0 = time.time()

    def _progress_cb(done, total):
        progress.step = f"UFM pairs: {done}/{total}"
        progress.current = done
        elapsed = max(time.time() - pair_t0, 1e-6)
        rate = done / elapsed if done > 0 else 0.0
        progress.items_per_second = rate if rate > 0 else None
        if rate > 0:
            progress.eta_seconds = max((total - done) / rate, 0.0)
        else:
            progress.eta_seconds = None

    sim_matrix = compute_pairwise_similarity(
        crop_images,
        progress_callback=_progress_cb,
    )

    # Restore to step-level for remaining steps
    progress.total = 4
    progress.current = 2
    progress.eta_seconds = None
    progress.items_per_second = None

    # Step 2.5: Enforce same-frame cannot-links before clustering
    n_constrained = apply_same_frame_constraints(
        sim_matrix, crop_ids, session.crops,
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
