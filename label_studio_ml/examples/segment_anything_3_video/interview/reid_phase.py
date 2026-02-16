"""ReID (Re-Identification) clustering for the Interview UI.

After all crops are classified (accepted/rejected), this module groups
accepted crops into identity clusters using fused DINOv3 + color histogram
features, then generates calibrated pairs for human verification. Pair
resolutions are applied with a burden-of-proof merge/split policy.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import ReID helpers from complete_reid.py in parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from complete_reid import _compute_hist, _rgb_to_hsv, _hist_intersection

from .state import (
    CropData, CropLabel, InterviewSession, Phase, ReIDPair,
)
from .cache_manager import save_session
from .background import JobProgress
from .frame_cache import read_frame_cached

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Feature Fusion
# ---------------------------------------------------------------------------

def compute_fused_similarity(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    dinov3_weight: float = 0.7,
    color_weight: float = 0.3,
) -> float:
    """Compute weighted combination of DINOv3 cosine similarity and color histogram intersection.

    Args:
        feat_a: DINOv3 CLS token embedding for crop A, shape (1024,).
        feat_b: DINOv3 CLS token embedding for crop B, shape (1024,).
        hist_a: Normalized HSV color histogram for crop A.
        hist_b: Normalized HSV color histogram for crop B.
        dinov3_weight: Weight for the cosine similarity component (default 0.7).
        color_weight: Weight for the histogram intersection component (default 0.3).

    Returns:
        Fused similarity score in [0, 1].
    """
    # Cosine similarity for DINOv3 features
    norm_a = float(np.linalg.norm(feat_a))
    norm_b = float(np.linalg.norm(feat_b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(feat_a, feat_b) / (norm_a * norm_b))
    # Map from [-1, 1] to [0, 1]
    cosine_sim = 0.5 * (cosine_sim + 1.0)

    # Histogram intersection for color features
    color_sim = _hist_intersection(hist_a, hist_b)

    # Weighted combination
    fused = dinov3_weight * cosine_sim + color_weight * color_sim
    return max(0.0, min(1.0, fused))


# ---------------------------------------------------------------------------
# 1b. Identity Centroid Averaging
# ---------------------------------------------------------------------------

def _apply_centroid_averaging(
    feature_matrix: np.ndarray,
    hist_matrix: np.ndarray,
    crop_ids: List[str],
    must_links: List[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace individual crop features with group centroids for must-linked crops.

    For each connected component of must-linked crops, computes the mean
    feature vector (L2-normalized) and mean histogram, then replaces every
    member's row with the group centroid.  Singletons and crops not in any
    must-link are left unchanged.

    This is analogous to the 15-frame track averaging in complete_reid.py,
    but uses human-confirmed "same" verdicts instead of temporal adjacency.

    Args:
        feature_matrix: (N, D) DINOv3 feature matrix (not modified in-place).
        hist_matrix: (N, H) histogram matrix (not modified in-place).
        crop_ids: Ordered list of crop IDs matching matrix rows.
        must_links: List of (crop_id_a, crop_id_b) pairs confirmed as same person.

    Returns:
        (new_feature_matrix, new_hist_matrix) with centroid-replaced rows.
    """
    if not must_links:
        return feature_matrix, hist_matrix

    # Build index: crop_id -> row index
    id_to_idx = {cid: i for i, cid in enumerate(crop_ids)}
    id_set = set(crop_ids)

    # Build adjacency list for must-link connected components (simple BFS)
    adj: Dict[str, List[str]] = {}
    for a, b in must_links:
        if a not in id_set or b not in id_set:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    if not adj:
        return feature_matrix, hist_matrix

    # Find connected components via BFS
    visited: set = set()
    groups: List[List[str]] = []
    for node in adj:
        if node in visited:
            continue
        component: List[str] = []
        queue = [node]
        visited.add(node)
        while queue:
            current = queue.pop(0)
            component.append(current)
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(component) >= 2:
            groups.append(component)

    if not groups:
        return feature_matrix, hist_matrix

    # Copy matrices so we don't mutate inputs
    new_feat = feature_matrix.copy()
    new_hist = hist_matrix.copy()

    for group in groups:
        indices = [id_to_idx[cid] for cid in group]

        # Feature centroid: mean then L2-normalize
        centroid_feat = new_feat[indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid_feat))
        if norm > 1e-8:
            centroid_feat /= norm

        # Histogram centroid: plain mean
        centroid_hist = new_hist[indices].mean(axis=0)

        # Replace all group members with centroid
        for idx in indices:
            new_feat[idx] = centroid_feat
            new_hist[idx] = centroid_hist

    return new_feat, new_hist


# ---------------------------------------------------------------------------
# 1c. Phase 1 Transition Check
# ---------------------------------------------------------------------------

def _phase1_complete(session: InterviewSession) -> bool:
    """Check if centroid building (Phase 1) is done.

    Phase 1 is complete when every non-singleton cluster has at least one
    must-link involving its members.  Singleton clusters cannot build
    centroids, so they don't block the transition.

    Returns:
        True if Phase 1 is complete (all non-singleton clusters covered).
    """
    if not session.reid_clusters:
        return True

    must_link_nodes: set = set()
    for a, b in session.reid_must_links:
        must_link_nodes.add(a)
        must_link_nodes.add(b)

    for members in session.reid_clusters.values():
        if len(members) < 2:
            continue  # singleton — can't build centroid, skip
        if not any(m in must_link_nodes for m in members):
            return False  # this cluster has no confirmed "same"

    return True


# ---------------------------------------------------------------------------
# 1d. Identity Centroids + Auto-Assignment (Phase 3)
# ---------------------------------------------------------------------------

DECISIVE_MIN_TOP1 = float(os.getenv("REID_DECISIVE_MIN_TOP1", "0.6"))
DECISIVE_MARGIN = float(os.getenv("REID_DECISIVE_MARGIN", "0.15"))


def _compute_identity_centroids(
    session: InterviewSession,
) -> Dict[int, np.ndarray]:
    """Compute L2-normalized feature centroid for each identity cluster.

    Args:
        session: Session with reid_clusters and crops with features.

    Returns:
        Dict mapping cluster_id to (D,) normalized centroid vector.
    """
    centroids: Dict[int, np.ndarray] = {}
    for cluster_id, members in session.reid_clusters.items():
        feats = []
        for cid in members:
            crop = session.get_crop(cid)
            if crop is not None and crop.features is not None:
                feats.append(crop.features)
        if feats:
            centroid = np.mean(np.stack(feats), axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-8:
                centroid /= norm
            centroids[cluster_id] = centroid
    return centroids


def compute_auto_assignments(
    session: InterviewSession,
    min_top1: float = DECISIVE_MIN_TOP1,
    min_margin: float = DECISIVE_MARGIN,
) -> Dict[str, Any]:
    """Compute assignment confidence for all accepted crops vs identity centroids.

    For each accepted crop, computes cosine similarity to each identity
    centroid.  Crops are classified as:
      - **auto_assigned**: best match exceeds ``min_top1`` AND margin over
        second-best is at least ``min_margin``.  These can be confidently
        assigned (or confirmed if already clustered).
      - **unresolved**: too ambiguous to auto-assign — close to multiple
        centroids.

    Crops already in constraints (must-link/cannot-link) are skipped since
    their placement is already determined by human verdicts.

    Args:
        session: Interview session with reid_clusters and crops.
        min_top1: Minimum cosine similarity for the best centroid.
        min_margin: Minimum gap between best and second-best similarity.

    Returns:
        Dict with keys:
          ``auto_assigned``: {crop_id: {cluster_id, confidence, margin, already_clustered}}
          ``unresolved``: {crop_id: {top_candidates: [(cluster_id, sim)], current_cluster}}
    """
    if not session.reid_clusters:
        return {"auto_assigned": {}, "unresolved": {}}

    centroids = _compute_identity_centroids(session)
    if not centroids:
        return {"auto_assigned": {}, "unresolved": {}}

    # Build set of crops already in constraints (placement decided by human)
    constrained: set = set()
    for a, b in session.reid_must_links:
        constrained.add(a)
        constrained.add(b)
    for a, b in session.reid_cannot_links:
        constrained.add(a)
        constrained.add(b)

    # Map crops to their current cluster
    crop_to_cluster: Dict[str, int] = {}
    for cluster_id, members in session.reid_clusters.items():
        for cid in members:
            crop_to_cluster[cid] = cluster_id

    auto_assigned: Dict[str, Dict[str, Any]] = {}
    unresolved: Dict[str, Dict[str, Any]] = {}

    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    for crop in accepted:
        if crop.features is None:
            continue
        if crop.crop_id in constrained:
            continue  # placement decided by human verdicts

        # Compute cosine similarity to each centroid
        feat = crop.features
        feat_norm = float(np.linalg.norm(feat))
        if feat_norm < 1e-8:
            continue

        sims: List[Tuple[int, float]] = []
        for cid, centroid in centroids.items():
            cos = float(np.dot(feat, centroid) / feat_norm)
            # centroid is already unit-normalized
            sim = 0.5 * (cos + 1.0)  # map [-1, 1] → [0, 1]
            sims.append((cid, sim))

        sims.sort(key=lambda x: -x[1])

        if len(sims) < 1:
            continue

        top_cid, top_sim = sims[0]
        second_sim = sims[1][1] if len(sims) > 1 else 0.0
        margin = top_sim - second_sim
        current = crop_to_cluster.get(crop.crop_id)

        if top_sim >= min_top1 and margin >= min_margin:
            auto_assigned[crop.crop_id] = {
                "cluster_id": top_cid,
                "confidence": round(top_sim, 4),
                "margin": round(margin, 4),
                "already_clustered": current is not None,
                "current_cluster": current,
            }
        else:
            top_candidates = [(cid, round(s, 4)) for cid, s in sims[:3]]
            unresolved[crop.crop_id] = {
                "top_candidates": top_candidates,
                "current_cluster": current,
            }

    return {"auto_assigned": auto_assigned, "unresolved": unresolved}


# ---------------------------------------------------------------------------
# 2. Spherical K-Means
# ---------------------------------------------------------------------------

def spherical_kmeans(features: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    """Spherical K-Means clustering on L2-normalized features.

    Uses K-Means++ style initialization: the first centroid is chosen
    uniformly at random, and each subsequent centroid is sampled with
    probability proportional to (1 - cosine_similarity) to the nearest
    existing centroid.

    Iteration proceeds by assigning each point to the nearest centroid
    (highest cosine similarity), recomputing centroids as the mean of
    assigned points, and L2-normalizing them. Convergence is declared
    when assignments do not change between iterations.

    Args:
        features: (N, D) array of feature vectors (will be L2-normalized internally).
        k: Number of clusters.
        max_iter: Maximum number of iterations.

    Returns:
        Cluster assignments as an (N,) integer array with values in [0, k).
    """
    n, d = features.shape
    if n <= k:
        return np.arange(n, dtype=np.intp)

    # L2-normalize all feature vectors
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X = features / norms

    # K-Means++ initialization
    rng = np.random.RandomState(42)
    centroids = np.empty((k, d), dtype=np.float64)
    first_idx = rng.randint(0, n)
    centroids[0] = X[first_idx]

    for c in range(1, k):
        # Compute cosine similarity to nearest existing centroid
        sims = X @ centroids[:c].T  # (N, c)
        max_sim = sims.max(axis=1)  # (N,)
        # Distance proportional to (1 - max_sim), clipped for stability
        dists = np.maximum(1.0 - max_sim, 0.0)
        dist_sum = dists.sum()
        if dist_sum < 1e-12:
            # Fallback: pick randomly if all points are equidistant
            centroids[c] = X[rng.randint(0, n)]
        else:
            probs = dists / dist_sum
            idx = rng.choice(n, p=probs)
            centroids[c] = X[idx]

    # Iterative assignment and update
    assignments = np.full(n, -1, dtype=np.intp)
    for iteration in range(max_iter):
        # Assign each point to the nearest centroid (highest cosine sim)
        sims = X @ centroids.T  # (N, k)
        new_assignments = sims.argmax(axis=1)

        # Check convergence
        if np.array_equal(assignments, new_assignments):
            logger.debug("Spherical K-Means converged at iteration %d", iteration)
            break
        assignments = new_assignments

        # Update centroids
        for c in range(k):
            mask = assignments == c
            if mask.any():
                centroid = X[mask].mean(axis=0)
                norm = float(np.linalg.norm(centroid))
                if norm > 1e-8:
                    centroid /= norm
                centroids[c] = centroid
            # If no points assigned, keep the old centroid (avoids empty cluster)

    return assignments


# ---------------------------------------------------------------------------
# 3. Silhouette-based K estimation
# ---------------------------------------------------------------------------

REID_OVERCLUSTER_BIAS = float(os.getenv("REID_OVERCLUSTER_BIAS", "0.15"))


def estimate_k(
    features: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    overcluster_bias: float = REID_OVERCLUSTER_BIAS,
) -> int:
    """Estimate optimal number of clusters using silhouette score with overclustering bias.

    Runs spherical K-Means for each candidate K in the given range and
    picks the K with the highest *biased* silhouette score. The bias term
    gives higher K values a linear bonus so that the estimator prefers
    overclustering (more clusters) over underclustering. Humans can merge
    clusters down, but without a split mechanism underclustering is much
    harder to fix.

    The biased score is:
        biased = avg_silhouette + overcluster_bias * (k - min_k) / (max_k - min_k)

    Args:
        features: (N, D) feature matrix.
        k_range: (min_k, max_k) inclusive range of K values to try.
        overcluster_bias: Linear bonus for higher K values (default from
            REID_OVERCLUSTER_BIAS env var, typically 0.15).

    Returns:
        Optimal K value. Returns 1 if the dataset is too small for
        clustering or no valid K is found.
    """
    n = features.shape[0]
    min_k, max_k = k_range

    # Clamp range to data size
    max_k = min(max_k, n - 1)
    if max_k < min_k or n < 3:
        return max(1, min(n, min_k))

    # L2-normalize for cosine distance computation
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X = features / norms

    # Pairwise cosine distance matrix: dist = 1 - cosine_similarity
    sim_matrix = X @ X.T
    dist_matrix = 1.0 - sim_matrix

    best_k = min_k
    best_score = -1.0
    k_span = max(max_k - min_k, 1)

    for k in range(min_k, max_k + 1):
        assignments = spherical_kmeans(features, k)
        n_actual_clusters = len(set(assignments))
        if n_actual_clusters < 2:
            continue

        # Compute average silhouette score
        silhouette_sum = 0.0
        for i in range(n):
            ci = assignments[i]

            # a(i): mean distance to points in same cluster
            same_mask = (assignments == ci)
            same_mask[i] = False
            n_same = same_mask.sum()
            if n_same == 0:
                continue
            a_i = float(dist_matrix[i][same_mask].mean())

            # b(i): min mean distance to points in any other cluster
            b_i = float("inf")
            for cj in range(k):
                if cj == ci:
                    continue
                other_mask = (assignments == cj)
                if not other_mask.any():
                    continue
                mean_dist = float(dist_matrix[i][other_mask].mean())
                if mean_dist < b_i:
                    b_i = mean_dist

            if b_i == float("inf"):
                continue

            denom = max(a_i, b_i)
            if denom > 1e-12:
                silhouette_sum += (b_i - a_i) / denom

        avg_silhouette = silhouette_sum / n

        # Apply overclustering bias: higher K gets a linear bonus
        biased_score = avg_silhouette + overcluster_bias * (k - min_k) / k_span
        logger.debug(
            "K=%d silhouette=%.4f biased=%.4f (bias=%.4f)",
            k, avg_silhouette, biased_score,
            overcluster_bias * (k - min_k) / k_span,
        )

        if biased_score > best_score:
            best_score = biased_score
            best_k = k

    logger.info("Estimated optimal K=%d (biased_score=%.4f)", best_k, best_score)
    return best_k


# ---------------------------------------------------------------------------
# 4. Color histogram extraction for crops
# ---------------------------------------------------------------------------

def extract_crop_histograms(
    session: InterviewSession,
    accepted_crops: List[CropData],
    progress: JobProgress,
) -> Dict[str, np.ndarray]:
    """Extract HSV color histograms for accepted crops.

    Reads video frames via PyAV (_read_frame_pyav from seeding_common),
    crops to the bounding box stored in each CropData, and computes a
    normalized HSV histogram using _compute_hist from complete_reid.

    Args:
        session: The interview session (provides video_path and dimensions).
        accepted_crops: List of CropData with label == ACCEPTED.
        progress: JobProgress object for reporting extraction status.

    Returns:
        Mapping from crop_id to its normalized histogram array.
    """
    from .frame_cache import read_frame_cached

    histograms: Dict[str, np.ndarray] = {}
    if not accepted_crops:
        return histograms

    # Reuse cached histograms from previous runs (survives in-memory across reclusters)
    need_compute: List[CropData] = []
    for crop in accepted_crops:
        if crop.histogram is not None:
            histograms[crop.crop_id] = crop.histogram
        else:
            need_compute.append(crop)

    if not need_compute:
        logger.info(
            "Reused cached histograms for all %d accepted crops", len(accepted_crops),
        )
        return histograms

    logger.info(
        "Histogram cache: %d cached, %d need computation",
        len(histograms), len(need_compute),
    )

    # Group crops by frame to minimize video seeks
    frame_to_crops: Dict[int, List[CropData]] = {}
    for crop in need_compute:
        frame_to_crops.setdefault(crop.frame_idx, []).append(crop)

    total_frames = len(frame_to_crops)
    progress.step = "Extracting color histograms"
    progress.total = total_frames
    progress.current = 0

    bins = (8, 8, 8)
    video_path = session.video_path
    img_w, img_h = session.width, session.height

    cache_key = getattr(session, "cache_key", None)

    for frame_count, (frame_idx, crops) in enumerate(sorted(frame_to_crops.items()), 1):
        pil_frame = read_frame_cached(video_path, frame_idx, cache_key=cache_key)
        if pil_frame is None:
            logger.warning("Could not read frame %d for histogram extraction", frame_idx)
            progress.current = frame_count
            continue

        frame_rgb = np.array(pil_frame.convert("RGB"))
        h, w = frame_rgb.shape[:2]

        for crop in crops:
            x1, y1, x2, y2 = crop.xyxy.astype(int)
            # Clamp to frame bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            crop_rgb = frame_rgb[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                logger.debug("Empty crop %s at frame %d", crop.crop_id, frame_idx)
                continue

            hist = _compute_hist(crop_rgb, bins)
            histograms[crop.crop_id] = hist
            crop.histogram = hist

        progress.current = frame_count

    computed = len(histograms) - (len(accepted_crops) - len(need_compute))
    logger.info(
        "Histograms: %d computed, %d cached, %d total accepted crops",
        computed, len(accepted_crops) - len(need_compute), len(accepted_crops),
    )
    return histograms


def _rebuild_sim_matrix(
    session: InterviewSession,
) -> Tuple[np.ndarray, List[str]]:
    """Rebuild fused similarity matrix from persisted crop features + histograms.

    Used by generate_next_round_pairs() to pick informative pairs without
    re-reading video frames. Crops missing features are skipped; crops
    missing histograms use a zero histogram (DINOv3-only similarity).

    Returns:
        (sim_matrix, crop_ids) where sim_matrix is (N, N) float32 and
        crop_ids[i] is the crop ID for row/column i.
    """
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    featured = [c for c in accepted if c.features is not None]
    if len(featured) < 2:
        crop_ids = [c.crop_id for c in featured]
        n = len(crop_ids)
        return np.eye(n, dtype=np.float32), crop_ids

    crop_ids = [c.crop_id for c in featured]
    n = len(crop_ids)

    feature_matrix = np.stack([c.features for c in featured])

    # Find a reference histogram shape from any crop that has one
    ref_hist = None
    for c in featured:
        if c.histogram is not None:
            ref_hist = c.histogram
            break
    default_hist = np.zeros_like(ref_hist) if ref_hist is not None else np.zeros(512, dtype=np.float32)

    hist_list = [c.histogram if c.histogram is not None else default_hist for c in featured]
    hist_matrix = np.stack(hist_list)

    # Apply centroid averaging for must-linked crops (human-confirmed "same")
    if session.reid_must_links:
        feature_matrix, hist_matrix = _apply_centroid_averaging(
            feature_matrix, hist_matrix, crop_ids, session.reid_must_links,
        )

    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = compute_fused_similarity(
                feature_matrix[i], feature_matrix[j],
                hist_matrix[i], hist_matrix[j],
            )
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix, crop_ids


# ---------------------------------------------------------------------------
# 5. Pair Sampling (calibrated)
# ---------------------------------------------------------------------------

def sample_pairs(
    session: InterviewSession,
    clusters: Dict[int, List[str]],
    similarity_matrix: np.ndarray,
    crop_ids: List[str],
    max_pairs: int = 30,
    round_num: int = 1,
) -> List[ReIDPair]:
    """Sample pairs with phase-aware allocation.

    Phase 1 (centroid building, ``session.reid_phase_stage == 1``):
        - 70% budget: intra-cluster high-similarity pairs (build centroids)
        - 30% budget: cross-cluster merge candidates (catch obvious merges)
        - Breadth-first across clusters, prefer different frames for diversity

    Phase 2+ (ambiguous resolution):
        - 100% cross-cluster pairs with current priority order:
          merge candidates > ambiguous > separation confirmation

    Args:
        session: Current interview session.
        clusters: Mapping from cluster_id to list of crop_ids.
        similarity_matrix: (N, N) symmetric similarity matrix.
        crop_ids: Ordered list mapping matrix indices to crop IDs.
        max_pairs: Maximum pairs per round.
        round_num: Current round (1-based). Calibration only in round 1.

    Returns:
        List of ReIDPair objects, shuffled for unbiased presentation.
    """
    id_to_idx = {cid: i for i, cid in enumerate(crop_ids)}
    cluster_ids_sorted = sorted(clusters.keys())

    # Build set of crop pairs that already have human constraints (skip these)
    constrained_pairs: set = set()
    for a, b in session.reid_must_links:
        constrained_pairs.add((a, b))
        constrained_pairs.add((b, a))
    for a, b in session.reid_cannot_links:
        constrained_pairs.add((a, b))
        constrained_pairs.add((b, a))

    # ======================================================================
    # Phase 1: Centroid Building — front-load intra-cluster pairs
    # ======================================================================
    centroid_building_pairs: List[ReIDPair] = []
    if session.reid_phase_stage == 1:
        intra_budget = int(max_pairs * 0.7)
        # Collect intra-cluster candidates per cluster (breadth-first)
        intra_per_cluster: Dict[int, List[Tuple[str, str, float]]] = {}
        for ci in cluster_ids_sorted:
            members = clusters[ci]
            if len(members) < 2:
                continue
            candidates = []
            for i in range(len(members)):
                a = members[i]
                if a not in id_to_idx:
                    continue
                for j in range(i + 1, len(members)):
                    b = members[j]
                    if b not in id_to_idx:
                        continue
                    if (a, b) in constrained_pairs:
                        continue
                    # Prefer different frames
                    crop_a = session.get_crop(a)
                    crop_b = session.get_crop(b)
                    same_frame = (crop_a is not None and crop_b is not None
                                  and crop_a.frame_idx == crop_b.frame_idx)
                    if same_frame:
                        continue
                    sim = float(similarity_matrix[id_to_idx[a], id_to_idx[b]])
                    if sim >= 0.6:  # only "easy same" pairs
                        candidates.append((a, b, sim))
            candidates.sort(key=lambda x: -x[2])
            intra_per_cluster[ci] = candidates

        # Breadth-first allocation: 1 pair per cluster, then 2nd, etc.
        used_intra: set = set()
        pointer_per_cluster = {ci: 0 for ci in intra_per_cluster}
        while len(centroid_building_pairs) < intra_budget:
            added_any = False
            for ci in cluster_ids_sorted:
                if ci not in intra_per_cluster:
                    continue
                cands = intra_per_cluster[ci]
                ptr = pointer_per_cluster[ci]
                while ptr < len(cands):
                    a, b, sim = cands[ptr]
                    ptr += 1
                    if (a, b) not in used_intra:
                        centroid_building_pairs.append(ReIDPair(
                            pair_id=str(uuid.uuid4())[:12],
                            crop_id_a=a, crop_id_b=b,
                            cluster_a=ci, cluster_b=ci,
                            pool="centroid_building",
                            similarity=sim,
                        ))
                        used_intra.add((a, b))
                        used_intra.add((b, a))
                        added_any = True
                        break
                pointer_per_cluster[ci] = ptr
                if len(centroid_building_pairs) >= intra_budget:
                    break
            if not added_any:
                break

        # Reduce cross-cluster budget to 30% for Phase 1
        original_max_pairs = max_pairs
        max_pairs = max_pairs - len(centroid_building_pairs)
    else:
        original_max_pairs = max_pairs

    # ======================================================================
    # Cross-cluster pairs (used by both Phase 1 remainder and Phase 2+)
    # ======================================================================
    cluster_pair_candidates: Dict[Tuple[int, int], List[Tuple[str, str, float]]] = {}

    for i_idx, ci in enumerate(cluster_ids_sorted):
        for j_idx in range(i_idx + 1, len(cluster_ids_sorted)):
            cj = cluster_ids_sorted[j_idx]
            key = (ci, cj)
            candidates = []
            for a in clusters[ci]:
                if a not in id_to_idx:
                    continue
                for b in clusters[cj]:
                    if b not in id_to_idx:
                        continue
                    if (a, b) in constrained_pairs:
                        continue
                    sim = float(similarity_matrix[id_to_idx[a], id_to_idx[b]])
                    candidates.append((a, b, sim))
            candidates.sort(key=lambda x: -x[2])
            cluster_pair_candidates[key] = candidates

    def _pool_for_sim(sim: float) -> str:
        if sim > 0.7:
            return "merge_candidate"
        elif sim >= 0.3:
            return "ambiguous"
        else:
            return "confident_different"

    # -- First pass: 1 best pair per cluster-pair --
    first_pass: List[ReIDPair] = []
    used_crop_pairs: set = set()

    sorted_cluster_pairs = sorted(
        cluster_pair_candidates.keys(),
        key=lambda k: cluster_pair_candidates[k][0][2] if cluster_pair_candidates[k] else -1,
        reverse=True,
    )

    for key in sorted_cluster_pairs:
        candidates = cluster_pair_candidates[key]
        if not candidates:
            continue
        a, b, sim = candidates[0]
        ci, cj = key
        first_pass.append(ReIDPair(
            pair_id=str(uuid.uuid4())[:12],
            crop_id_a=a,
            crop_id_b=b,
            cluster_a=ci,
            cluster_b=cj,
            pool=_pool_for_sim(sim),
            similarity=sim,
        ))
        used_crop_pairs.add((a, b))

    # -- Second pass: add 2nd pair per cluster-pair (different crops) --
    second_pass: List[ReIDPair] = []
    for key in sorted_cluster_pairs:
        candidates = cluster_pair_candidates[key]
        ci, cj = key
        for a, b, sim in candidates:
            if (a, b) not in used_crop_pairs:
                second_pass.append(ReIDPair(
                    pair_id=str(uuid.uuid4())[:12],
                    crop_id_a=a,
                    crop_id_b=b,
                    cluster_a=ci,
                    cluster_b=cj,
                    pool=_pool_for_sim(sim),
                    similarity=sim,
                ))
                used_crop_pairs.add((a, b))
                break

    # -- Calibration (round 1 only) --
    calibration: List[ReIDPair] = []
    if round_num == 1:
        intra_candidates = []
        for ci in cluster_ids_sorted:
            members = clusters[ci]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if a not in id_to_idx or b not in id_to_idx:
                        continue
                    sim = float(similarity_matrix[id_to_idx[a], id_to_idx[b]])
                    if sim > 0.8:
                        intra_candidates.append((a, b, sim, ci))

        rng = random.Random(42)
        rng.shuffle(intra_candidates)
        for a, b, sim, ci in intra_candidates[:2]:
            calibration.append(ReIDPair(
                pair_id=str(uuid.uuid4())[:12],
                crop_id_a=a,
                crop_id_b=b,
                cluster_a=ci,
                cluster_b=ci,
                pool="confident_same",
                similarity=sim,
            ))

        diff_candidates = []
        for key, candidates in cluster_pair_candidates.items():
            ci, cj = key
            for a, b, sim in candidates:
                if sim < 0.3 and (a, b) not in used_crop_pairs:
                    diff_candidates.append((a, b, sim, ci, cj))
        rng.shuffle(diff_candidates)
        for a, b, sim, ci, cj in diff_candidates[:2]:
            calibration.append(ReIDPair(
                pair_id=str(uuid.uuid4())[:12],
                crop_id_a=a,
                crop_id_b=b,
                cluster_a=ci,
                cluster_b=cj,
                pool="confident_different",
                similarity=sim,
            ))

    # -- Assemble and truncate --
    cross_pairs = first_pass + second_pass + calibration
    if len(cross_pairs) > max_pairs:
        budget_remaining = max_pairs - len(first_pass)
        extras = second_pass + calibration
        rng = random.Random(42)
        rng.shuffle(extras)
        cross_pairs = first_pass + extras[:max(0, budget_remaining)]

    # Combine centroid-building + cross-cluster pairs, enforce total budget
    all_pairs = centroid_building_pairs + cross_pairs
    if len(all_pairs) > original_max_pairs:
        all_pairs = all_pairs[:original_max_pairs]

    rng = random.Random(42)
    rng.shuffle(all_pairs)

    n_intra = len(centroid_building_pairs)
    logger.info(
        "Sampled %d pairs (round %d, stage %d): %d centroid-building, "
        "%d first-pass, %d second-pass, %d calibration "
        "(cluster-pairs: %d total)",
        len(all_pairs), round_num, session.reid_phase_stage,
        n_intra, len(first_pass), len(second_pass),
        len(calibration), len(cluster_pair_candidates),
    )
    return all_pairs


# ---------------------------------------------------------------------------
# 5b. Convergence check + adaptive round generation
# ---------------------------------------------------------------------------

MAX_PAIRS_PER_ROUND = int(os.getenv("REID_MAX_PAIRS_PER_ROUND", "30"))


def compute_cluster_pair_coverage(
    session: InterviewSession,
) -> Dict[str, Any]:
    """Check whether all cluster-pair relationships have sufficient evidence.

    A cluster-pair is "resolved" if any of:
      - same >= 2  (enough human confirmations to trigger a merge)
      - different >= 1  (single veto is decisive)
      - unsure >= 2  (humans genuinely uncertain; treat as separate)

    Args:
        session: Interview session with reid_clusters and reid_pairs populated.

    Returns:
        Dict with keys: needs_more_rounds, uncovered_count, uncovered_pairs,
        resolved_count, total_cluster_pairs.
    """
    # 1. Build crop_to_cluster mapping from live session state
    crop_to_cluster: Dict[str, int] = {}
    for cid_int, members in session.reid_clusters.items():
        for crop_id in members:
            crop_to_cluster[crop_id] = cid_int

    # 2. Accumulate evidence per cluster-pair from resolved pairs
    evidence: Dict[Tuple[int, int], Dict[str, int]] = {}

    for pair in session.reid_pairs.values():
        if pair.resolution is None:
            continue
        ca = crop_to_cluster.get(pair.crop_id_a)
        cb = crop_to_cluster.get(pair.crop_id_b)
        if ca is None or cb is None:
            continue
        if ca == cb:
            continue  # same-cluster pairs don't count
        key = (min(ca, cb), max(ca, cb))
        if key not in evidence:
            evidence[key] = {"same": 0, "different": 0, "unsure": 0}

        if pair.resolution == "same":
            evidence[key]["same"] += 1
        elif pair.resolution == "different":
            evidence[key]["different"] += 1
        else:  # "unsure"
            evidence[key]["unsure"] += 1

    # 4. Enumerate all C(k,2) cluster-pair relationships
    cluster_ids_sorted = sorted(session.reid_clusters.keys())
    all_pairs_set: List[Tuple[int, int]] = []
    for i in range(len(cluster_ids_sorted)):
        for j in range(i + 1, len(cluster_ids_sorted)):
            all_pairs_set.append((cluster_ids_sorted[i], cluster_ids_sorted[j]))

    total_cluster_pairs = len(all_pairs_set)

    # 5. Check resolution status per pair
    uncovered: List[Tuple[int, int]] = []
    resolved_count = 0

    for cp in all_pairs_set:
        ev = evidence.get(cp)
        if ev is None:
            uncovered.append(cp)
            continue
        # Resolved if: same >= 2 OR different >= 1 OR unsure >= 2
        if ev["same"] >= 2 or ev["different"] >= 1 or ev["unsure"] >= 2:
            resolved_count += 1
        else:
            uncovered.append(cp)

    needs_more = len(uncovered) > 0

    return {
        "needs_more_rounds": needs_more,
        "uncovered_count": len(uncovered),
        "uncovered_pairs": uncovered,
        "resolved_count": resolved_count,
        "total_cluster_pairs": total_cluster_pairs,
    }


def generate_next_round_pairs(
    session: InterviewSession,
    max_pairs: int = MAX_PAIRS_PER_ROUND,
) -> Tuple[List[ReIDPair], Dict[str, Any]]:
    """Generate targeted pairs for the next ReID round, focusing on uncovered cluster-pairs.

    Steps:
      1. Check cluster-pair coverage to find gaps.
      2. If converged (no gaps), return empty list.
      3. Rebuild similarity matrix from persisted features.
      4. For each uncovered cluster-pair, pick up to 2 crop pairs (highest sim)
         that haven't been shown before.
      5. Increment reid_round, save, and return new pairs + coverage info.

    Args:
        session: Interview session with reid_clusters and reid_pairs populated.
        max_pairs: Maximum number of new pairs to generate.

    Returns:
        (new_pairs, coverage) where new_pairs is a list of ReIDPair objects
        and coverage is the dict from compute_cluster_pair_coverage.
    """
    # 1. Check coverage
    coverage = compute_cluster_pair_coverage(session)

    # 2. If converged, return empty
    if not coverage["needs_more_rounds"]:
        return ([], coverage)

    # 3. Rebuild similarity matrix
    sim_matrix, sim_crop_ids = _rebuild_sim_matrix(session)
    id_to_idx = {cid: i for i, cid in enumerate(sim_crop_ids)}

    # 4. Build set of existing crop pairs (both directions)
    existing_crop_pairs: set = set()
    for pair in session.reid_pairs.values():
        existing_crop_pairs.add((pair.crop_id_a, pair.crop_id_b))
        existing_crop_pairs.add((pair.crop_id_b, pair.crop_id_a))

    # 5. Increment round
    session.reid_round += 1

    def _pool_for_sim(sim: float) -> str:
        if sim > 0.7:
            return "merge_candidate"
        elif sim >= 0.3:
            return "ambiguous"
        else:
            return "confident_different"

    # 6. For each uncovered cluster-pair, find new crop pairs
    new_pairs: List[ReIDPair] = []

    for (ca, cb) in coverage["uncovered_pairs"]:
        members_a = session.reid_clusters.get(ca, [])
        members_b = session.reid_clusters.get(cb, [])

        # Collect candidate crop pairs with similarity
        candidates: List[Tuple[str, str, float]] = []
        for a in members_a:
            if a not in id_to_idx:
                continue
            for b in members_b:
                if b not in id_to_idx:
                    continue
                if (a, b) in existing_crop_pairs:
                    continue
                sim = float(sim_matrix[id_to_idx[a], id_to_idx[b]])
                candidates.append((a, b, sim))

        # Sort by similarity descending, take up to 2
        candidates.sort(key=lambda x: -x[2])
        for a, b, sim in candidates[:2]:
            pair = ReIDPair(
                pair_id=str(uuid.uuid4())[:12],
                crop_id_a=a,
                crop_id_b=b,
                cluster_a=ca,
                cluster_b=cb,
                pool=_pool_for_sim(sim),
                similarity=sim,
            )
            new_pairs.append(pair)
            existing_crop_pairs.add((a, b))
            existing_crop_pairs.add((b, a))

        if len(new_pairs) >= max_pairs:
            break

    # Truncate to max_pairs
    new_pairs = new_pairs[:max_pairs]

    # 8. Add new pairs to session
    for p in new_pairs:
        session.reid_pairs[p.pair_id] = p

    # 9. Save
    session.touch()
    save_session(session)

    logger.info(
        "Generated %d next-round pairs (round %d): %d uncovered cluster-pairs",
        len(new_pairs), session.reid_round, coverage["uncovered_count"],
    )

    return (new_pairs, coverage)


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def run_reid_pipeline(
    session: InterviewSession,
    n_clusters: Optional[int],
    progress: JobProgress,
    overcluster_bias: Optional[float] = None,
) -> Dict[str, Any]:
    """Full ReID pipeline for the interview workflow.

    Steps:
        1. Collect accepted crops that have DINOv3 features.
        2. Extract HSV color histograms for those crops.
        3. Compute a fused (DINOv3 + color) similarity matrix.
        4. Cluster using spherical K-Means (auto-estimating K if not provided).
        5. Generate calibrated pairs for human verification.
        6. Update session state with clusters and pairs.

    Args:
        session: The current interview session (must be in REID phase).
        n_clusters: User-specified number of identities, or None for auto.
        progress: JobProgress object for status reporting.

    Returns:
        Summary dict with keys: n_clusters, n_pairs, cluster_sizes.
    """
    progress.step = "Collecting accepted crops"
    progress.total = 6

    # Step 1: Collect accepted crops with DINOv3 features
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    featured = [c for c in accepted if c.features is not None]
    if len(featured) < 2:
        raise ValueError(
            f"Need at least 2 accepted crops with features for ReID, got {len(featured)}"
        )
    progress.current = 1
    logger.info("ReID pipeline: %d accepted crops with features", len(featured))

    # Step 2: Extract color histograms
    progress.step = "Extracting color histograms"
    histograms = extract_crop_histograms(session, featured, progress)
    progress.current = 2

    # Build ordered arrays for matrix computation
    crop_ids = [c.crop_id for c in featured]
    n = len(crop_ids)

    feature_matrix = np.stack([c.features for c in featured])  # (N, 1024)

    # Default histogram for crops where extraction failed
    default_hist = np.zeros_like(next(iter(histograms.values()))) if histograms else np.zeros(512)

    hist_list = [histograms.get(cid, default_hist) for cid in crop_ids]
    hist_matrix = np.stack(hist_list)  # (N, H)

    # Step 3: Compute fused similarity matrix
    progress.step = "Computing similarity matrix"

    # Apply centroid averaging: replace individual features with group centroids
    # for must-linked crops (human-confirmed "same" verdicts).  This gives the
    # same noise-reduction effect as complete_reid.py's 15-frame track averaging.
    if session.reid_must_links:
        feature_matrix, hist_matrix = _apply_centroid_averaging(
            feature_matrix, hist_matrix, crop_ids, session.reid_must_links,
        )

    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = compute_fused_similarity(
                feature_matrix[i], feature_matrix[j],
                hist_matrix[i], hist_matrix[j],
            )
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    progress.current = 3

    # Step 4: Cluster
    progress.step = "Clustering identities"
    if n_clusters is None or n_clusters < 2:
        bias = overcluster_bias if overcluster_bias is not None else REID_OVERCLUSTER_BIAS
        k = estimate_k(feature_matrix, k_range=(2, min(15, n - 1)), overcluster_bias=bias)
    else:
        k = min(n_clusters, n - 1)
    k = max(2, k)

    assignments = spherical_kmeans(feature_matrix, k)

    progress.current = 4

    # Build cluster mapping: cluster_id -> [crop_ids]
    clusters: Dict[int, List[str]] = {}
    for idx, cid in enumerate(crop_ids):
        cluster_id = int(assignments[idx])
        clusters.setdefault(cluster_id, []).append(cid)
        # Also update the crop's reid_cluster_id
        crop = session.get_crop(cid)
        if crop is not None:
            crop.reid_cluster_id = cluster_id

    # Step 5: Generate calibrated pairs
    progress.step = "Generating verification pairs"
    session.reid_round = 1
    pairs = sample_pairs(session, clusters, sim_matrix, crop_ids,
                         max_pairs=MAX_PAIRS_PER_ROUND, round_num=1)
    progress.current = 5

    # Step 6: Update session state
    progress.step = "Saving session"
    session.reid_clusters = clusters
    session.reid_pairs.update({p.pair_id: p for p in pairs})
    session.n_identities = len(clusters)
    session.touch()
    save_session(session)
    progress.current = 6

    cluster_sizes = {cid: len(members) for cid, members in clusters.items()}
    summary = {
        "n_clusters": len(clusters),
        "n_pairs": len(pairs),
        "cluster_sizes": cluster_sizes,
        "n_accepted": len(featured),
    }
    logger.info(
        "ReID pipeline complete: %d clusters, %d pairs, sizes=%s",
        summary["n_clusters"], summary["n_pairs"], cluster_sizes,
    )
    return summary


# ---------------------------------------------------------------------------
# 7. Outlier flagging (split mechanism)
# ---------------------------------------------------------------------------

def flag_outlier(session: InterviewSession, crop_id: str) -> Dict[str, Any]:
    """Flag a crop as an outlier and move it to a new singleton cluster.

    This is the split mechanism: when a human spots a crop that doesn't
    belong in its current cluster, flagging it:
      1. Removes it from its current cluster.
      2. Creates a new singleton cluster for it.
      3. Generates ~1 new pair per existing cluster (flagged crop vs.
         a representative from each cluster) for human verification.
      4. Renumbers clusters and saves.

    Args:
        session: Current interview session with reid_clusters populated.
        crop_id: ID of the crop to flag.

    Returns:
        Dict with keys: new_cluster_id, new_pairs_count, cluster_sizes.

    Raises:
        ValueError: If crop_id not found or not in any cluster.
    """
    # Find the crop's current cluster
    old_cluster_id = None
    for cid_int, members in session.reid_clusters.items():
        if crop_id in members:
            old_cluster_id = cid_int
            break

    if old_cluster_id is None:
        raise ValueError(f"Crop {crop_id} not found in any cluster")

    # Remove from old cluster
    session.reid_clusters[old_cluster_id].remove(crop_id)

    # If old cluster is now empty, remove it
    if not session.reid_clusters[old_cluster_id]:
        del session.reid_clusters[old_cluster_id]

    # Create new singleton cluster
    existing_keys = list(session.reid_clusters.keys())
    new_cluster_id = (max(existing_keys) + 1) if existing_keys else 0
    session.reid_clusters[new_cluster_id] = [crop_id]

    # Update crop's cluster assignment
    crop = session.get_crop(crop_id)
    if crop is not None:
        crop.reid_cluster_id = new_cluster_id

    # Generate new pairs: flagged crop vs. 1 representative per cluster
    flagged_crop = session.get_crop(crop_id)
    new_pairs: List[ReIDPair] = []

    if flagged_crop is not None and flagged_crop.features is not None:
        for cid_int, members in session.reid_clusters.items():
            if cid_int == new_cluster_id:
                continue  # Skip the new singleton cluster

            # Pick the representative with highest cosine similarity to the flagged crop
            best_rep = None
            best_sim = -1.0
            for rep_id in members:
                rep_crop = session.get_crop(rep_id)
                if rep_crop is None or rep_crop.features is None:
                    continue
                norm_a = float(np.linalg.norm(flagged_crop.features))
                norm_b = float(np.linalg.norm(rep_crop.features))
                if norm_a < 1e-8 or norm_b < 1e-8:
                    cos_sim = 0.0
                else:
                    cos_sim = float(
                        np.dot(flagged_crop.features, rep_crop.features) / (norm_a * norm_b)
                    )
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    best_rep = rep_id

            if best_rep is not None:
                # Map cosine similarity from [-1, 1] to [0, 1]
                mapped_sim = 0.5 * (best_sim + 1.0)
                pair = ReIDPair(
                    pair_id=str(uuid.uuid4())[:12],
                    crop_id_a=crop_id,
                    crop_id_b=best_rep,
                    cluster_a=new_cluster_id,
                    cluster_b=cid_int,
                    pool="ambiguous",
                    similarity=mapped_sim,
                )
                new_pairs.append(pair)

    # Add new pairs to session
    for p in new_pairs:
        session.reid_pairs[p.pair_id] = p

    # Renumber clusters from 0
    old_clusters = dict(session.reid_clusters)
    new_clusters: Dict[int, List[str]] = {}
    for new_id, (_, members) in enumerate(sorted(old_clusters.items())):
        new_clusters[new_id] = members
        for cid in members:
            c = session.get_crop(cid)
            if c is not None:
                c.reid_cluster_id = new_id

    session.reid_clusters = new_clusters
    session.n_identities = len(new_clusters)
    session.touch()
    save_session(session)

    # Find the new cluster ID of the flagged crop after renumbering
    flagged_new_id = None
    for cid_int, members in new_clusters.items():
        if crop_id in members:
            flagged_new_id = cid_int
            break

    cluster_sizes = {cid: len(m) for cid, m in new_clusters.items()}
    logger.info(
        "Flagged crop %s: moved to cluster %s, generated %d new pairs",
        crop_id, flagged_new_id, len(new_pairs),
    )
    return {
        "new_cluster_id": flagged_new_id,
        "new_pairs_count": len(new_pairs),
        "n_identities": len(new_clusters),
        "cluster_sizes": cluster_sizes,
    }


# ---------------------------------------------------------------------------
# 7b. Centroid-Growing Progressive Association (Phase 2)
# ---------------------------------------------------------------------------


def _confirmed_crop_ids(session: InterviewSession) -> set:
    """Return set of crop IDs that appear in any must-link (human-confirmed)."""
    confirmed: set = set()
    for a, b in session.reid_must_links:
        confirmed.add(a)
        confirmed.add(b)
    return confirmed


def _cannot_link_set(session: InterviewSession) -> set:
    """Return set of (a, b) AND (b, a) for all cannot-links."""
    cl: set = set()
    for a, b in session.reid_cannot_links:
        cl.add((a, b))
        cl.add((b, a))
    return cl


def compute_centroid_assignments(
    session: InterviewSession,
    min_top1: float = DECISIVE_MIN_TOP1,
    min_margin: float = DECISIVE_MARGIN,
    max_representatives: int = 3,
) -> Dict[str, Any]:
    """Compute decisive/indecisive assignment of crops to identity centroids.

    For each non-confirmed crop, computes cosine similarity to every
    identity centroid (mean of confirmed crops in that cluster).

    Returns:
        Dict with keys:
          ``decisive``: {crop_id: {cluster_id, confidence, margin, representatives}}
          ``indecisive``: {crop_id: {candidates: [{cluster_id, similarity, representatives}]}}
          ``centroid_count``: number of centroids with confirmed members
          ``unassigned_count``: crops not in any cluster
    """
    if not session.reid_clusters:
        return {"decisive": {}, "indecisive": {}, "centroid_count": 0, "unassigned_count": 0}

    # Build centroids from confirmed crops only (must-linked members).
    # If a cluster has no confirmed members, use all members as centroid.
    confirmed = _confirmed_crop_ids(session)
    cl_set = _cannot_link_set(session)

    centroids: Dict[int, np.ndarray] = {}
    representative_crops: Dict[int, List[str]] = {}  # cluster -> top representative crop_ids

    for cluster_id, members in session.reid_clusters.items():
        confirmed_members = [cid for cid in members if cid in confirmed]
        if not confirmed_members:
            continue  # skip — no confirmed members, centroid would be unreliable
        source = confirmed_members
        feats = []
        for cid in source:
            crop = session.get_crop(cid)
            if crop is not None and crop.features is not None:
                feats.append(crop.features)
        if feats:
            centroid = np.mean(np.stack(feats), axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-8:
                centroid /= norm
            centroids[cluster_id] = centroid

        # Pick representative crops (up to max_representatives, prefer confirmed)
        reps = []
        for cid in confirmed_members[:max_representatives]:
            reps.append(cid)
        if len(reps) < max_representatives:
            for cid in members:
                if cid not in reps and len(reps) < max_representatives:
                    reps.append(cid)
        representative_crops[cluster_id] = reps

    if not centroids:
        return {"decisive": {}, "indecisive": {}, "centroid_count": 0, "unassigned_count": 0}

    # Evaluate all non-confirmed accepted crops
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    decisive: Dict[str, Dict[str, Any]] = {}
    indecisive: Dict[str, Dict[str, Any]] = {}

    for crop in accepted:
        if crop.features is None:
            continue
        if crop.crop_id in confirmed:
            continue  # already placed by human verdict

        feat = crop.features
        feat_norm = float(np.linalg.norm(feat))
        if feat_norm < 1e-8:
            continue

        sims: List[Tuple[int, float]] = []
        for cid, centroid in centroids.items():
            # Check cannot-link: skip centroids whose confirmed members conflict
            blocked = False
            for member in session.reid_clusters.get(cid, []):
                if (crop.crop_id, member) in cl_set:
                    blocked = True
                    break
            if blocked:
                continue

            cos = float(np.dot(feat, centroid) / feat_norm)
            sim = 0.5 * (cos + 1.0)
            sims.append((cid, sim))

        sims.sort(key=lambda x: -x[1])

        if len(sims) < 1:
            continue

        top_cid, top_sim = sims[0]
        second_sim = sims[1][1] if len(sims) > 1 else 0.0
        margin = top_sim - second_sim

        if top_sim >= min_top1 and margin >= min_margin:
            decisive[crop.crop_id] = {
                "cluster_id": top_cid,
                "confidence": round(top_sim, 4),
                "margin": round(margin, 4),
                "representatives": representative_crops.get(top_cid, []),
            }
        else:
            # Include top 2 candidates with representatives
            candidates = []
            for cid, sim in sims[:2]:
                candidates.append({
                    "cluster_id": cid,
                    "similarity": round(sim, 4),
                    "representatives": representative_crops.get(cid, []),
                })
            indecisive[crop.crop_id] = {"candidates": candidates}

    return {
        "decisive": decisive,
        "indecisive": indecisive,
        "centroid_count": len(centroids),
        "unassigned_count": len(indecisive),
    }


def apply_association_round(
    session: InterviewSession,
    assignments: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    """Apply Phase 2 human decisions: assign crops to identities.

    For each crop, the human picks a cluster_id (assign to that identity)
    or None (leave unassigned / new identity).

    After applying:
      1. Update cluster memberships and centroid.
      2. Re-compute decisive/indecisive for remaining unassigned.
      3. If no new decisive assignments, set ``converged = True``.

    Args:
        session: Interview session with reid_clusters.
        assignments: {crop_id: cluster_id} or {crop_id: None} for "neither".

    Returns:
        Summary dict with: applied_count, new_decisive, new_indecisive,
        converged, n_identities, clusters, reid_phase_stage.
    """
    applied_count = 0

    for crop_id, cluster_id in assignments.items():
        crop = session.get_crop(crop_id)
        if crop is None:
            continue

        if cluster_id is not None and cluster_id in session.reid_clusters:
            # Assign to existing cluster
            if crop_id not in session.reid_clusters[cluster_id]:
                session.reid_clusters[cluster_id].append(crop_id)
            crop.reid_cluster_id = cluster_id

            # Remove from any other cluster
            for cid, members in session.reid_clusters.items():
                if cid != cluster_id and crop_id in members:
                    members.remove(crop_id)

            # Add as must-link with a confirmed member from that cluster
            confirmed = _confirmed_crop_ids(session)
            for member in session.reid_clusters[cluster_id]:
                if member in confirmed and member != crop_id:
                    link = (crop_id, member)
                    rev = (member, crop_id)
                    if link not in session.reid_must_links and rev not in session.reid_must_links:
                        session.reid_must_links.append(link)
                    break

            applied_count += 1
        elif cluster_id is None:
            # "Neither / New Identity" — remove from old cluster, create singleton

            # Remove from any existing cluster (mirrors the assign-to-cluster path)
            for cid, members in list(session.reid_clusters.items()):
                if crop_id in members:
                    members.remove(crop_id)

            existing_keys = list(session.reid_clusters.keys())
            new_id = (max(existing_keys) + 1) if existing_keys else 0
            session.reid_clusters[new_id] = [crop_id]
            crop.reid_cluster_id = new_id

            # Self-must-link marks the crop as "confirmed" so
            # compute_centroid_assignments won't re-evaluate it
            link = (crop_id, crop_id)
            if link not in session.reid_must_links:
                session.reid_must_links.append(link)

            applied_count += 1

    # Clean up empty clusters
    empty = [k for k, v in session.reid_clusters.items() if not v]
    for k in empty:
        del session.reid_clusters[k]

    session.n_identities = len(session.reid_clusters)
    session.touch()
    save_session(session)

    # Re-compute assignments for remaining unplaced crops
    result = compute_centroid_assignments(session)

    new_decisive_count = len(result["decisive"])
    new_indecisive_count = len(result["indecisive"])
    converged = new_decisive_count == 0 and new_indecisive_count == 0

    # Auto-stop: if no new decisive, plateau reached
    plateau = new_decisive_count == 0

    # Phase transition: if Phase 2 and converged, move to Phase 3 (done)
    if session.reid_phase_stage == 2 and converged:
        session.reid_phase_stage = 3
        logger.info("Phase transition: 2 → 3 (centroid plateau reached)")
        save_session(session)

    clusters_info = {
        str(cid): {"crop_ids": members, "count": len(members)}
        for cid, members in session.reid_clusters.items()
    }

    summary = {
        "applied_count": applied_count,
        "new_decisive": result["decisive"],
        "new_indecisive": result["indecisive"],
        "converged": converged,
        "plateau": plateau,
        "n_identities": session.n_identities,
        "clusters": clusters_info,
        "reid_phase_stage": session.reid_phase_stage,
    }

    logger.info(
        "Association round: %d applied, %d new decisive, %d indecisive, converged=%s",
        applied_count, new_decisive_count, new_indecisive_count, converged,
    )
    return summary


# ---------------------------------------------------------------------------
# 8. Apply resolutions (centroid-growing approach)
# ---------------------------------------------------------------------------

def apply_resolutions(
    session: InterviewSession,
    resolutions: Dict[str, str],
) -> Dict[str, Any]:
    """Apply human pair resolutions using centroid-growing approach.

    Converts human verdicts into constraints:
        - "same" → must-link (crops confirmed as same identity)
        - "different" → cannot-link (crops confirmed as different identities)
        - "unsure" → no constraint

    Then re-assigns non-confirmed crops to nearest centroid (formed from
    confirmed "same" groups).  Confirmed groups stay locked to their
    cluster; other crops get re-evaluated by centroid proximity.

    Args:
        session: Current interview session with reid_pairs populated.
        resolutions: Mapping from pair_id to "same", "different", or "unsure".

    Returns:
        Summary dict with cluster info, phase transition, and
        centroid assignment data (when transitioning to Phase 2).
    """
    # 1. Store verdicts as constraints
    for pair_id, resolution in resolutions.items():
        pair = session.reid_pairs.get(pair_id)
        if pair is not None:
            pair.resolution = resolution
            if resolution == "same":
                link = (pair.crop_id_a, pair.crop_id_b)
                if link not in session.reid_must_links and \
                   (pair.crop_id_b, pair.crop_id_a) not in session.reid_must_links:
                    session.reid_must_links.append(link)
            elif resolution == "different":
                link = (pair.crop_id_a, pair.crop_id_b)
                if link not in session.reid_cannot_links and \
                   (pair.crop_id_b, pair.crop_id_a) not in session.reid_cannot_links:
                    session.reid_cannot_links.append(link)

    # Build cannot-link lookup for merge blocking and re-assignment
    cl_set = _cannot_link_set(session)

    # 2. Merge "same" pairs into the same cluster (respecting cannot-links)
    for pair_id, resolution in resolutions.items():
        if resolution != "same":
            continue
        pair = session.reid_pairs.get(pair_id)
        if pair is None:
            continue
        # Find clusters of both crops
        cluster_a = cluster_b = None
        for cid, members in session.reid_clusters.items():
            if pair.crop_id_a in members:
                cluster_a = cid
            if pair.crop_id_b in members:
                cluster_b = cid
        if cluster_a is not None and cluster_b is not None and cluster_a != cluster_b:
            # Check cannot-links: block merge if any member of cluster_a
            # has a cannot-link with any member of cluster_b
            blocked = False
            for m_a in session.reid_clusters[cluster_a]:
                for m_b in session.reid_clusters[cluster_b]:
                    if (m_a, m_b) in cl_set or (m_b, m_a) in cl_set:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                logger.debug(
                    "Merge of clusters %d and %d blocked by cannot-link", cluster_a, cluster_b,
                )
                continue
            # Merge cluster_b into cluster_a
            session.reid_clusters[cluster_a].extend(session.reid_clusters[cluster_b])
            for cid in session.reid_clusters[cluster_b]:
                crop = session.get_crop(cid)
                if crop is not None:
                    crop.reid_cluster_id = cluster_a
            del session.reid_clusters[cluster_b]

    # 2b. Split intra-cluster cannot-links: if two crops in the same cluster
    # have a cannot-link, move the non-confirmed one to a new cluster.
    for a, b in session.reid_cannot_links:
        # Find which cluster each is in
        ca = cb = None
        for cid, members in session.reid_clusters.items():
            if a in members:
                ca = cid
            if b in members:
                cb = cid
        if ca is None or cb is None or ca != cb:
            continue  # not in same cluster, skip

        # Decide which crop to evict: prefer evicting the non-confirmed one
        confirmed = _confirmed_crop_ids(session)
        a_confirmed = a in confirmed
        b_confirmed = b in confirmed
        if a_confirmed and not b_confirmed:
            evict = b
        elif b_confirmed and not a_confirmed:
            evict = a
        else:
            # Both confirmed or both unconfirmed — evict the second one
            evict = b

        # Move evicted crop to new singleton cluster
        existing_keys = list(session.reid_clusters.keys())
        new_id = (max(existing_keys) + 1) if existing_keys else 0
        session.reid_clusters[ca].remove(evict)
        session.reid_clusters[new_id] = [evict]
        crop = session.get_crop(evict)
        if crop is not None:
            crop.reid_cluster_id = new_id

    # 3. Compute centroids from confirmed groups and re-assign non-confirmed crops
    confirmed = _confirmed_crop_ids(session)

    if confirmed and len(session.reid_clusters) >= 2:
        # Build centroids from confirmed members only
        centroids: Dict[int, np.ndarray] = {}
        for cluster_id, members in session.reid_clusters.items():
            confirmed_in_cluster = [c for c in members if c in confirmed]
            if not confirmed_in_cluster:
                continue
            feats = []
            for cid in confirmed_in_cluster:
                crop = session.get_crop(cid)
                if crop is not None and crop.features is not None:
                    feats.append(crop.features)
            if feats:
                centroid = np.mean(np.stack(feats), axis=0)
                norm = float(np.linalg.norm(centroid))
                if norm > 1e-8:
                    centroid /= norm
                centroids[cluster_id] = centroid

        # Re-assign non-confirmed crops to nearest centroid (hybrid approach).
        # Need ≥2 centroids — with only 1, everything would collapse into it.
        if len(centroids) >= 2:
            for cluster_id, members in list(session.reid_clusters.items()):
                for crop_id in list(members):
                    if crop_id in confirmed:
                        continue  # locked
                    crop = session.get_crop(crop_id)
                    if crop is None or crop.features is None:
                        continue
                    feat = crop.features
                    feat_norm = float(np.linalg.norm(feat))
                    if feat_norm < 1e-8:
                        continue
                    # Find nearest centroid (respecting cannot-links)
                    best_cid = None
                    best_sim = -1.0
                    for cid, centroid in centroids.items():
                        blocked = any(
                            (crop_id, m) in cl_set
                            for m in session.reid_clusters.get(cid, [])
                        )
                        if blocked:
                            continue
                        cos = float(np.dot(feat, centroid) / feat_norm)
                        sim = 0.5 * (cos + 1.0)
                        if sim > best_sim:
                            best_sim = sim
                            best_cid = cid
                    # Move to better cluster if different
                    if best_cid is not None and best_cid != cluster_id:
                        members.remove(crop_id)
                        session.reid_clusters[best_cid].append(crop_id)
                        crop.reid_cluster_id = best_cid

    # Clean empty clusters and renumber
    non_empty = {k: v for k, v in session.reid_clusters.items() if v}
    new_clusters: Dict[int, List[str]] = {}
    for new_id, (_, members) in enumerate(sorted(non_empty.items())):
        new_clusters[new_id] = members
        for cid in members:
            crop = session.get_crop(cid)
            if crop is not None:
                crop.reid_cluster_id = new_id

    session.reid_clusters = new_clusters
    session.n_identities = len(new_clusters)
    session.touch()
    save_session(session)

    # Build response
    clusters_info = {
        str(cid): {"crop_ids": members, "count": len(members)}
        for cid, members in new_clusters.items()
    }

    summary: Dict[str, Any] = {
        "n_identities": len(new_clusters),
        "cluster_sizes": {cid: len(m) for cid, m in new_clusters.items()},
        "clusters": clusters_info,
        "reid_round": session.reid_round,
    }

    # Phase transition: Phase 1 → 2 when centroids are established
    if session.reid_phase_stage == 1 and _phase1_complete(session):
        session.reid_phase_stage = 2
        logger.info("Phase transition: 1 → 2 (centroid building complete)")
        save_session(session)
        # Compute initial centroid assignments for Phase 2
        assignments = compute_centroid_assignments(session)
        summary["centroid_assignments"] = assignments

    # In Phase 1, check if more intra-cluster rounds needed
    if session.reid_phase_stage == 1:
        coverage = compute_cluster_pair_coverage(session)
        summary["needs_more_rounds"] = coverage["needs_more_rounds"]
        summary["uncovered_count"] = coverage["uncovered_count"]
    else:
        summary["needs_more_rounds"] = False

    summary["reid_phase_stage"] = session.reid_phase_stage

    logger.info(
        "Applied resolutions (centroid-growing): %d must-links, %d cannot-links, "
        "%d clusters, phase=%d",
        len(session.reid_must_links), len(session.reid_cannot_links),
        len(new_clusters), session.reid_phase_stage,
    )
    return summary


# ---------------------------------------------------------------------------
# 9. Visual ReID: Montage + Cluster Injection
# ---------------------------------------------------------------------------

def generate_montage(
    session: InterviewSession,
    per_row: int = 10,
    crop_size: int = 128,
) -> Optional[tuple]:
    """Generate a labeled grid image of all accepted crops.

    Returns (PIL.Image, list[str]) where the list contains crop IDs in grid
    order (left-to-right, top-to-bottom), or None if no accepted crops.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    if not accepted:
        return None

    # Deterministic order: by frame_idx then crop_id
    accepted.sort(key=lambda c: (c.frame_idx, c.crop_id))

    # Read frames and crop each accepted detection
    cell_images = []
    crop_ids = []
    for crop in accepted:
        frame = read_frame_cached(
            session.video_path, crop.frame_idx, cache_key=session.cache_key,
        )
        if frame is None:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in crop.xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.width, x2)
        y2 = min(frame.height, y2)
        cropped = frame.crop((x1, y1, x2, y2))
        cropped = cropped.resize((crop_size, crop_size), PILImage.LANCZOS)
        cell_images.append(cropped)
        crop_ids.append(crop.crop_id)

    if not cell_images:
        return None

    # Layout: per_row columns, ceil(n / per_row) rows
    n = len(cell_images)
    n_rows = (n + per_row - 1) // per_row
    label_h = 16  # height for text label below each crop
    cell_h = crop_size + label_h

    montage = PILImage.new("RGB", (per_row * crop_size, n_rows * cell_h), (40, 40, 40))
    draw = ImageDraw.Draw(montage)

    for idx, (img, cid) in enumerate(zip(cell_images, crop_ids)):
        row, col = divmod(idx, per_row)
        x = col * crop_size
        y = row * cell_h
        montage.paste(img, (x, y))
        # Label below the crop
        draw.text((x + 2, y + crop_size + 1), cid, fill=(255, 255, 255))

    return montage, crop_ids


def inject_clusters(
    session: InterviewSession,
    cluster_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Inject visual ReID cluster assignments into the session.

    Args:
        session: The interview session to update.
        cluster_map: Mapping of arbitrary label names to lists of crop IDs.
            Example: {"person_A": ["c0", "c1"], "person_B": ["c2"]}

    Returns:
        Dict with summary stats: n_clusters, n_assigned, n_must_links,
        n_cannot_links.

    Raises:
        ValueError: If any crop ID is unknown or duplicated across clusters.
    """
    from itertools import combinations

    # --- Validate ---
    all_ids = []
    for label, cids in cluster_map.items():
        all_ids.extend(cids)

    # Check for unknowns
    known_ids = set(session.crops.keys())
    unknown = set(all_ids) - known_ids
    if unknown:
        raise ValueError(f"Unknown crop IDs: {sorted(unknown)}")

    # Check for duplicates
    if len(all_ids) != len(set(all_ids)):
        from collections import Counter
        dupes = [cid for cid, cnt in Counter(all_ids).items() if cnt > 1]
        raise ValueError(f"Duplicate crop IDs across clusters: {sorted(dupes)}")

    # --- Clear previous ReID state ---
    session.reid_clusters = {}
    session.reid_must_links = []
    session.reid_cannot_links = []

    # Reset all crops' reid_cluster_id
    for crop in session.crops.values():
        crop.reid_cluster_id = None

    if not cluster_map:
        return {
            "n_clusters": 0,
            "n_assigned": 0,
            "n_must_links": 0,
            "n_cannot_links": 0,
        }

    # --- Assign sequential integer cluster IDs ---
    cluster_groups: List[List[str]] = []  # index = cluster_id
    for label in cluster_map:
        cluster_groups.append(cluster_map[label])

    reid_clusters: Dict[int, List[str]] = {}
    for cluster_id, cids in enumerate(cluster_groups):
        reid_clusters[cluster_id] = list(cids)
        for cid in cids:
            crop = session.get_crop(cid)
            if crop is not None:
                crop.reid_cluster_id = cluster_id

    session.reid_clusters = reid_clusters

    # --- Generate must-links (within-cluster pairs) ---
    must_links = []
    for cids in cluster_groups:
        for a, b in combinations(cids, 2):
            must_links.append((a, b))
    session.reid_must_links = must_links

    # --- Generate cannot-links (cross-cluster pairs) ---
    cannot_links = []
    for i, j in combinations(range(len(cluster_groups)), 2):
        for a in cluster_groups[i]:
            for b in cluster_groups[j]:
                cannot_links.append((a, b))
    session.reid_cannot_links = cannot_links

    return {
        "n_clusters": len(cluster_groups),
        "n_assigned": len(all_ids),
        "n_must_links": len(must_links),
        "n_cannot_links": len(cannot_links),
    }
