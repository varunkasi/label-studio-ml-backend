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
    import seeding_common as base

    histograms: Dict[str, np.ndarray] = {}
    if not accepted_crops:
        return histograms

    # Group crops by frame to minimize video seeks
    frame_to_crops: Dict[int, List[CropData]] = {}
    for crop in accepted_crops:
        frame_to_crops.setdefault(crop.frame_idx, []).append(crop)

    total_frames = len(frame_to_crops)
    progress.step = "Extracting color histograms"
    progress.total = total_frames
    progress.current = 0

    bins = (8, 8, 8)
    video_path = session.video_path
    img_w, img_h = session.width, session.height

    for frame_count, (frame_idx, crops) in enumerate(sorted(frame_to_crops.items()), 1):
        pil_frame = base._read_frame_pyav(video_path, frame_idx)
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

    logger.info(
        "Extracted histograms for %d / %d accepted crops",
        len(histograms), len(accepted_crops),
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
    """Sample pairs with coverage-first allocation across cluster-pair relationships.

    Priority order for cross-cluster pairs:
        1. Merge candidates (sim > 0.7): most likely same-person across clusters.
        2. Ambiguous (0.3 <= sim <= 0.7): borderline cases needing human judgment.
        3. Separation confirmation (sim < 0.3): confirm they're different.

    Algorithm:
        - First pass: 1 best pair per cluster-pair (highest sim), all relationships.
        - Second pass: add 2nd pair per cluster-pair (different crops) if budget allows.
        - Round 1 only: append up to 4 calibration pairs (2 intra-same, 2 cross-diff).
        - Truncate to max_pairs, shuffle for presentation.

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

    # -- Collect ALL cross-cluster crop pairs, grouped by cluster-pair --
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
    all_pairs = first_pass + second_pass + calibration
    if len(all_pairs) > max_pairs:
        budget_remaining = max_pairs - len(first_pass)
        extras = second_pass + calibration
        rng = random.Random(42)
        rng.shuffle(extras)
        all_pairs = first_pass + extras[:max(0, budget_remaining)]

    rng = random.Random(42)
    rng.shuffle(all_pairs)

    logger.info(
        "Sampled %d pairs (round %d): %d first-pass, %d second-pass, %d calibration "
        "(cluster-pairs: %d total)",
        len(all_pairs), round_num, len(first_pass), len(second_pass),
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

    # Step 4: Cluster with spherical K-Means
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
    session.reid_pairs = {p.pair_id: p for p in pairs}
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
# 8. Apply resolutions (merge/split logic)
# ---------------------------------------------------------------------------

def apply_resolutions(
    session: InterviewSession,
    resolutions: Dict[str, str],
) -> Dict[str, Any]:
    """Apply human pair resolutions with burden-of-proof merge/split policy.

    Merge rules:
        - Need 2+ confirming "same" (Yes) pairs between the same two clusters
          to trigger a merge.
        - A single "different" (No) pair vetoes the merge entirely for that
          cluster pair, regardless of how many "same" pairs exist.
        - "unsure" pairs are treated as abstentions; at the end, any cluster
          pair that only has "unsure" evidence is left separate.

    The method tracks per-cluster-pair evidence as a dict keyed by
    (cluster_a, cluster_b) tuples, then executes merges that meet the
    threshold. Merged clusters are renumbered starting from 0.

    Args:
        session: Current interview session with reid_pairs populated.
        resolutions: Mapping from pair_id to "same", "different", or "unsure".

    Returns:
        Summary dict with keys: merges_executed, final_clusters, vetoed_pairs.
    """
    # Apply resolution labels to stored pairs
    for pair_id, resolution in resolutions.items():
        pair = session.reid_pairs.get(pair_id)
        if pair is not None:
            pair.resolution = resolution

    # Build crop_id → current cluster_id mapping from live session state.
    # This is critical: pair.cluster_a/cluster_b may be stale after prior
    # merges renumbered clusters. Using crop IDs to look up CURRENT cluster
    # membership ensures evidence is accumulated correctly.
    crop_to_cluster: Dict[str, int] = {}
    for cid_int, members in session.reid_clusters.items():
        for crop_id in members:
            crop_to_cluster[crop_id] = cid_int

    # Accumulate evidence per cluster pair
    # Use sorted tuple keys so (a, b) and (b, a) are treated identically
    evidence: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for pair in session.reid_pairs.values():
        if pair.resolution is None:
            continue
        # Look up CURRENT cluster for each crop (not stale pair fields)
        ca = crop_to_cluster.get(pair.crop_id_a)
        cb = crop_to_cluster.get(pair.crop_id_b)
        if ca is None or cb is None:
            continue
        if ca == cb:
            # Already in the same cluster (possibly from a prior merge); skip
            continue
        key = (min(ca, cb), max(ca, cb))
        if key not in evidence:
            evidence[key] = {"yes_count": 0, "no_count": 0, "unsure_count": 0, "max_sim": 0.0}

        if pair.resolution == "same":
            evidence[key]["yes_count"] += 1
            evidence[key]["max_sim"] = max(evidence[key]["max_sim"], pair.similarity)
        elif pair.resolution == "different":
            evidence[key]["no_count"] += 1
        else:  # "unsure"
            evidence[key]["unsure_count"] += 1

    # Decide which cluster pairs to merge
    merges_to_execute: List[Tuple[int, int]] = []
    vetoed: List[Tuple[int, int]] = []

    for (ca, cb), ev in evidence.items():
        if ev["no_count"] > 0:
            # Single "No" vetoes the merge
            vetoed.append((ca, cb))
            continue
        if ev["yes_count"] >= 2:
            merges_to_execute.append((ca, cb))
        # Otherwise: insufficient evidence, leave separate

    # Execute merges using union-find for transitive closure
    # (if A merges with B and B merges with C, then A, B, C are one cluster)
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Initialize all existing cluster IDs
    for cid in session.reid_clusters:
        parent[cid] = cid

    for ca, cb in merges_to_execute:
        union(ca, cb)

    # Rebuild clusters from union-find roots
    new_clusters_raw: Dict[int, List[str]] = {}
    for old_cid, members in session.reid_clusters.items():
        root = find(old_cid)
        if root not in new_clusters_raw:
            new_clusters_raw[root] = []
        new_clusters_raw[root].extend(members)

    # Renumber clusters starting from 0
    new_clusters: Dict[int, List[str]] = {}
    root_to_new_id: Dict[int, int] = {}
    for new_id, (root, members) in enumerate(sorted(new_clusters_raw.items())):
        root_to_new_id[root] = new_id
        new_clusters[new_id] = members

    # Update crop reid_cluster_id to reflect new numbering
    for new_id, members in new_clusters.items():
        for cid in members:
            crop = session.get_crop(cid)
            if crop is not None:
                crop.reid_cluster_id = new_id

    # Update session
    session.reid_clusters = new_clusters
    session.n_identities = len(new_clusters)
    session.touch()
    save_session(session)

    # Build clusters response (crop_ids per cluster) for frontend display
    clusters_info = {
        str(cid): {"crop_ids": members, "count": len(members)}
        for cid, members in new_clusters.items()
    }

    summary = {
        "merges_executed": len(merges_to_execute),
        "vetoed_pairs": len(vetoed),
        "final_clusters": len(new_clusters),
        "n_identities": len(new_clusters),
        "cluster_sizes": {cid: len(m) for cid, m in new_clusters.items()},
        "clusters": clusters_info,
    }

    # Add convergence information
    coverage = compute_cluster_pair_coverage(session)
    summary["needs_more_rounds"] = coverage["needs_more_rounds"]
    summary["uncovered_count"] = coverage["uncovered_count"]
    summary["reid_round"] = session.reid_round

    logger.info(
        "Applied resolutions: %d merges, %d vetoed, %d final clusters",
        summary["merges_executed"], summary["vetoed_pairs"], summary["final_clusters"],
    )
    return summary
