"""Distance-weighted k-NN quality gate for the Interview UI.

Replaces the MLP classifier with a non-parametric approach.  Every labeled
crop becomes a support vector — no training step, no learned parameters.

Feature representation (2056-dim):
    [DINOv3 crop CLS (1024) | DINOv3 context CLS (1024) | spatial (4) | mask quality (4)]

Scoring:
    weight_i = 1 / max(cosine_dist(query, support_i)^2, epsilon)
    accept_score = sum(weight_i for accepted neighbors)
    reject_score = sum(weight_i * subcat_weight_i for rejected neighbors)
    confidence = accept_score / (accept_score + reject_score)
    uncertainty = 1 - |confidence - 0.5| / 0.5
                = 1 - |2 * confidence - 1|

Subcategory weights downweight soft reject reasons:
    not_person:    1.0  (hard reject — clearly not a person)
    partial_box:   0.8  (box doesn't cover the full visible person)
    oversized_box: 0.5  (box is too large / includes background)
    None:          1.0  (untagged reject, treated as full weight)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state import CropData, CropLabel, InterviewSession

logger = logging.getLogger(__name__)

# Default reject subcategory weights
DEFAULT_SUBCAT_WEIGHTS: Dict[Optional[str], float] = {
    "not_person": 1.0,
    "partial_box": 0.8,
    "oversized_box": 0.5,
    None: 1.0,
}

# Confidence threshold for accept decisions (env-configurable)
KNN_THRESHOLD = float(os.getenv("INTERVIEW_KNN_THRESHOLD", "0.6"))

# Epsilon to avoid division by zero in weight computation
_EPS = 1e-6

# Pair-aware vote weights for corrected/rejected boundary examples
_PAIR_ACCEPT_MULT = 1.5
_PAIR_REJECT_MULT = 1.25
_MAX_SAMPLE_MULT = 2.0


def _build_full_feature_vector(crop: CropData) -> Optional[np.ndarray]:
    """Build 2056-dim feature vector for a single crop.

    Returns None if required features are missing.
    """
    if crop.features is None or crop.metadata is None:
        return None
    ctx = crop.context_features if crop.context_features is not None else np.zeros(1024, dtype=np.float32)
    mq = crop.mask_quality if crop.mask_quality is not None else np.zeros(4, dtype=np.float32)
    return np.concatenate([crop.features, ctx, crop.metadata, mq])


def build_support_set(
    session: InterviewSession,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Optional[str]]]:
    """Collect all labeled crops with complete features.

    Returns:
        features: (N, 2056) float32 array
        labels: (N,) float32 array (1.0 = accept, 0.0 = reject)
        crop_ids: list of N crop IDs
        reject_reasons: list of N reject reasons (None for accepts)
    """
    features_list: List[np.ndarray] = []
    labels_list: List[float] = []
    ids_list: List[str] = []
    reasons_list: List[Optional[str]] = []

    for crop in session.crops.values():
        if crop.label not in (CropLabel.ACCEPTED, CropLabel.REJECTED):
            continue
        vec = _build_full_feature_vector(crop)
        if vec is None:
            continue
        features_list.append(vec)
        labels_list.append(1.0 if crop.label == CropLabel.ACCEPTED else 0.0)
        ids_list.append(crop.crop_id)
        reasons_list.append(crop.reject_reason if crop.label == CropLabel.REJECTED else None)

    if not features_list:
        return (
            np.empty((0, 2056), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
            [],
        )

    return (
        np.stack(features_list).astype(np.float32),
        np.array(labels_list, dtype=np.float32),
        ids_list,
        reasons_list,
    )


def score_crops(
    query_features: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    support_reject_reasons: List[Optional[str]],
    subcategory_weights: Optional[Dict[Optional[str], float]] = None,
    support_crop_ids: Optional[List[str]] = None,
    support_corrected_from: Optional[List[Optional[str]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance-weighted k-NN scoring.

    Args:
        query_features: (M, 2056) features for crops to score
        support_features: (N, 2056) features for labeled support set
        support_labels: (N,) labels (1.0 = accept, 0.0 = reject)
        support_reject_reasons: (N,) reject reason strings (None for accepts)
        subcategory_weights: per-reason weight multiplier for reject votes

    Returns:
        confidences: (M,) float32 in [0, 1] — probability of being "accept"
        uncertainties: (M,) float32 in [0, 1] — 0 = certain, 1 = uncertain
    """
    if subcategory_weights is None:
        subcategory_weights = DEFAULT_SUBCAT_WEIGHTS

    M = query_features.shape[0]

    if support_features.shape[0] == 0:
        # No support set: return uniform uncertainty
        return (
            np.full(M, 0.5, dtype=np.float32),
            np.ones(M, dtype=np.float32),
        )

    # Normalize for cosine distance
    q_norm = query_features / np.maximum(
        np.linalg.norm(query_features, axis=1, keepdims=True), _EPS
    )
    s_norm = support_features / np.maximum(
        np.linalg.norm(support_features, axis=1, keepdims=True), _EPS
    )

    # Cosine similarity: (M, N)
    sim_matrix = q_norm @ s_norm.T
    # Cosine distance: 1 - sim, clamped to [0, 2]
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

    # Inverse-square distance weights: (M, N)
    weights = 1.0 / np.maximum(dist_matrix ** 2, _EPS)

    N = support_features.shape[0]
    # Pair-aware sample scaling (applied to both accept/reject sides)
    pair_scale = np.ones(N, dtype=np.float32)
    if (
        support_crop_ids is not None
        and support_corrected_from is not None
        and len(support_crop_ids) == N
        and len(support_corrected_from) == N
    ):
        corrected_from_ids = {
            corrected_from
            for corrected_from in support_corrected_from
            if corrected_from
        }
        for i in range(N):
            cid = support_crop_ids[i]
            corrected_from = support_corrected_from[i]
            lbl = support_labels[i]
            if lbl == 1.0 and corrected_from:
                pair_scale[i] *= _PAIR_ACCEPT_MULT
            elif lbl == 0.0 and cid in corrected_from_ids:
                pair_scale[i] *= _PAIR_REJECT_MULT
        pair_scale = np.minimum(pair_scale, _MAX_SAMPLE_MULT)

    # Build reject scaling vector: subcategory weight * pair weight
    reject_scale = np.array([
        (subcategory_weights.get(r, 1.0) if lbl == 0.0 else 1.0)
        for lbl, r in zip(support_labels, support_reject_reasons)
    ], dtype=np.float32)
    reject_scale = np.minimum(reject_scale * pair_scale, _MAX_SAMPLE_MULT)

    # Accept score: sum of weights where label == 1.0
    accept_mask = (support_labels == 1.0).astype(np.float32) * pair_scale  # (N,)
    accept_scores = weights @ accept_mask  # (M,)

    # Reject score: sum of weights * subcat_scale where label == 0.0
    reject_mask = (support_labels == 0.0).astype(np.float32) * reject_scale  # (N,)
    reject_scores = weights @ reject_mask  # (M,)

    # Confidence = accept / (accept + reject)
    total = accept_scores + reject_scores
    confidences = np.where(total > _EPS, accept_scores / total, 0.5)
    confidences = confidences.astype(np.float32)

    # Uncertainty: 0 at confidence extremes, 1 at confidence = 0.5
    uncertainties = (1.0 - np.abs(2.0 * confidences - 1.0)).astype(np.float32)

    return confidences, uncertainties


def compute_uncertainties(session: InterviewSession) -> int:
    """Re-score all PENDING crops using k-NN and update uncertainty fields.

    Returns the number of crops scored.
    """
    support_feats, support_labels, support_ids, support_reasons = build_support_set(session)

    pending = [c for c in session.crops.values() if c.label == CropLabel.PENDING]
    if not pending:
        return 0

    query_vecs: List[np.ndarray] = []
    scorable: List[CropData] = []
    for crop in pending:
        vec = _build_full_feature_vector(crop)
        if vec is not None:
            query_vecs.append(vec)
            scorable.append(crop)

    if not scorable:
        return 0

    query_features = np.stack(query_vecs).astype(np.float32)
    support_corrected_from: List[Optional[str]] = []
    for cid in support_ids:
        crop = session.crops.get(cid)
        support_corrected_from.append(
            crop.corrected_from if crop is not None else None
        )

    confidences, uncertainties = score_crops(
        query_features,
        support_feats,
        support_labels,
        support_reasons,
        support_crop_ids=support_ids,
        support_corrected_from=support_corrected_from,
    )

    for i, crop in enumerate(scorable):
        crop.uncertainty = float(uncertainties[i])

    logger.info(
        "k-NN scored %d pending crops (support set: %d accept, %d reject)",
        len(scorable),
        int(support_labels.sum()),
        len(support_labels) - int(support_labels.sum()),
    )
    return len(scorable)
