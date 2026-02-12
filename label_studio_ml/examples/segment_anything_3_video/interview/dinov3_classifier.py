"""DINOv3 feature extractor and MLP classifier for the Interview UI.

Provides lazy-loaded DINOv3 ViT-L backbone, CLS-token feature extraction,
2-layer MLP binary classifier (1032-dim: DINOv3 + spatial + mask quality)
with feature-level augmentation, and training loop with class weighting,
LR decay per round, and uncertainty sampling.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .background import JobProgress
from .mask_utils import compute_lr, compute_mask_quality  # noqa: F401 – re-export
from .cache_manager import load_model, save_model, save_session
from .state import CropData, CropLabel, CropSource, InterviewSession, Phase

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
# MLP Classifier
# ---------------------------------------------------------------------------

class CropClassifier(nn.Module):
    """2-layer MLP for binary classification of crops.

    Input: 1032-dim (1024 DINOv3 + 4 spatial metadata + 4 mask quality)
    Architecture: Linear(1032, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)
    Output: logit (use BCEWithLogitsLoss for training, sigmoid for inference)
    """

    def __init__(self, input_dim: int = 1032, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (B, 1032) -> (B, 1) logit."""
        return self.fc2(self.drop(self.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Feature-level augmentation
# ---------------------------------------------------------------------------

def augment_features(
    features: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply feature-level augmentation: MixUp, CutMix, RandomErasing.

    Augmented samples are appended to the original batch (originals preserved).
    Returns (augmented_features, augmented_labels).
    """
    n, d = features.shape
    if n < 2:
        return features, labels

    aug_feats: List[torch.Tensor] = []
    aug_labels: List[torch.Tensor] = []

    # MixUp: blend random pairs with lambda ~ Beta(0.2, 0.2)
    perm = torch.randperm(n)
    lam = torch.distributions.Beta(0.2, 0.2).sample((n,)).to(features.device).unsqueeze(1)
    aug_feats.append(lam * features + (1.0 - lam) * features[perm])
    aug_labels.append(lam.squeeze(1) * labels + (1.0 - lam.squeeze(1)) * labels[perm])

    # CutMix: swap random contiguous band of 20-40% of dimensions
    perm2 = torch.randperm(n)
    band_len = max(1, int(d * (0.2 + 0.2 * torch.rand(1).item())))
    start = torch.randint(0, max(1, d - band_len), (1,)).item()
    cutmix = features.clone()
    cutmix[:, start : start + band_len] = features[perm2, start : start + band_len]
    frac = band_len / d
    aug_feats.append(cutmix)
    aug_labels.append((1.0 - frac) * labels + frac * labels[perm2])

    # RandomErasing: zero out random 10-20% of feature dimensions
    n_erase = max(1, int(d * (0.1 + 0.1 * torch.rand(1).item())))
    mask = torch.ones_like(features)
    for i in range(n):
        mask[i, torch.randperm(d)[:n_erase]] = 0.0
    aug_feats.append(features * mask)
    aug_labels.append(labels.clone())

    return (
        torch.cat([features] + aug_feats, dim=0),
        torch.cat([labels] + aug_labels, dim=0),
    )


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
    missing = [cid for cid in crop_ids if session.crops[cid].features is None]
    if not missing:
        return
    if progress:
        progress.step = f"Extracting DINOv3 features for {len(missing)} crops"

    from collections import defaultdict
    from .frame_cache import read_frame_cached

    frame_to_cids: Dict[int, List[str]] = defaultdict(list)
    for cid in missing:
        frame_to_cids[session.crops[cid].frame_idx].append(cid)

    # Try LRU cache first, batch-decode the rest
    frame_images: Dict[int, Image.Image] = {}
    uncached: List[int] = []
    for fidx in sorted(frame_to_cids.keys()):
        cached = read_frame_cached(session.video_path, fidx)
        if cached is not None:
            frame_images[fidx] = cached
        else:
            uncached.append(fidx)

    if uncached:
        from .detection import _decode_frames_sequential
        decoded = _decode_frames_sequential(session.video_path, uncached)
        frame_images.update(decoded)
        # Populate the LRU cache with newly decoded frames
        from .frame_cache import put_cached_frame
        for fidx, img in decoded.items():
            put_cached_frame(session.video_path, fidx, img)

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


def _build_feature_matrix(session: InterviewSession, crop_ids: List[str]) -> torch.Tensor:
    """Build (N, 1032) feature matrix: DINOv3 (1024) + metadata (4) + mask_quality (4)."""
    rows = []
    for cid in crop_ids:
        crop = session.crops.get(cid)
        if crop is None or crop.features is None or crop.metadata is None:
            continue
        mq = crop.mask_quality if crop.mask_quality is not None else np.zeros(4, dtype=np.float32)
        rows.append(np.concatenate([crop.features, crop.metadata, mq]))
    if not rows:
        return torch.empty(0, 1032)
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def _overlaps_any(box: np.ndarray, existing: List[np.ndarray], threshold: float) -> bool:
    """True if box overlaps any existing box above the IoU threshold."""
    return any(_compute_iou(box, eb) >= threshold for eb in existing)


# ---------------------------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------------------------

def _evaluate_validation(
    session: InterviewSession,
    model: CropClassifier,
    device: torch.device,
    round_num: int,
    train_accuracy: float,
) -> Dict[str, Any]:
    """Evaluate the trained model on held-out validation crops.

    Collects labeled crops on validation frames, builds feature matrix,
    runs model predictions, computes accuracy, and appends to
    ``session.validation_history``.

    Returns:
        Dict with val_accuracy, val_n_pos, val_n_neg (or empty if no
        validation data available).
    """
    val_crop_ids = session.get_validation_crop_ids()
    if not val_crop_ids:
        return {}

    # Collect only labeled validation crops
    val_pos_ids = []
    val_neg_ids = []
    for cid in val_crop_ids:
        crop = session.crops.get(cid)
        if crop is None:
            continue
        if crop.label == CropLabel.ACCEPTED:
            val_pos_ids.append(cid)
        elif crop.label == CropLabel.REJECTED:
            val_neg_ids.append(cid)

    val_ids = val_pos_ids + val_neg_ids
    if not val_ids:
        return {}

    y_val_np = np.array(
        [1.0] * len(val_pos_ids) + [0.0] * len(val_neg_ids), dtype=np.float32
    )

    X_val = _build_feature_matrix(session, val_ids)
    if X_val.shape[0] == 0:
        return {}

    # Filter to crops that actually have features
    valid_mask = np.array([
        session.crops[cid].features is not None and session.crops[cid].metadata is not None
        for cid in val_ids
    ])
    y_val_np = y_val_np[valid_mask]
    if len(y_val_np) == 0:
        return {}

    y_val = torch.tensor(y_val_np, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_val.to(device)).squeeze(-1)) >= 0.5).float()
        val_acc = (preds == y_val).float().mean().item()

    n_val_pos = int(y_val_np.sum())
    n_val_neg = len(y_val_np) - n_val_pos

    entry = {
        "round": round_num,
        "val_accuracy": round(val_acc, 4),
        "val_n_pos": n_val_pos,
        "val_n_neg": n_val_neg,
        "train_accuracy": round(train_accuracy, 4),
    }
    session.validation_history.append(entry)

    logger.info("Validation round %d: acc=%.2f%% (%d pos, %d neg), train_acc=%.2f%%",
                round_num, val_acc * 100, n_val_pos, n_val_neg, train_accuracy * 100)

    return {"val_accuracy": val_acc, "val_n_pos": n_val_pos, "val_n_neg": n_val_neg}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_classifier(
    session: InterviewSession, progress: JobProgress,
    round_num: int = 1,
) -> Dict[str, Any]:
    """Train the MLP classifier on labeled crops.

    1. Collect accepted (positive) and rejected (negative) crops
    2. Build feature matrix (1032-dim: DINOv3 + metadata + mask_quality)
    3. Apply class weights for imbalance
    4. Train with AdamW, LR decaying per round (0.7^(round-1))
    5. Score all unlabeled crops -> uncertainty sampling
    6. uncertainty = 1.0 - abs(2 * sigmoid(logit) - 1.0)
    7. Save model to cache, update session stats

    Target: <2s per training cycle on RTX 6000 Ada.
    """
    progress.step = "Collecting labelled crops"
    val_frame_set = set(session.validation_frames)
    accepted = [c for c in session.get_crops_by_label(CropLabel.ACCEPTED)
                if c.frame_idx not in val_frame_set]
    rejected = [c for c in session.get_crops_by_label(CropLabel.REJECTED)
                if c.frame_idx not in val_frame_set]
    n_pos, n_neg = len(accepted), len(rejected)

    if val_frame_set:
        logger.info("Training: excluded %d validation frames, %d pos / %d neg training crops",
                     len(val_frame_set), n_pos, n_neg)

    if n_pos == 0 or n_neg == 0:
        logger.warning("Need >= 1 pos and >= 1 neg (got %d / %d)", n_pos, n_neg)
        return {"accuracy": 0.0, "n_pos": n_pos, "n_neg": n_neg,
                "epochs": 0, "pending_scored": 0, "mean_uncertainty": 0.5}

    # Ensure features extracted for all crops
    _ensure_crop_features(session, list(session.crops.keys()), progress)

    # Build training data (1032-dim: DINOv3 + spatial + mask_quality)
    progress.step = "Building training data"
    train_ids = [c.crop_id for c in accepted] + [c.crop_id for c in rejected]
    y_all = np.array([1.0] * n_pos + [0.0] * n_neg, dtype=np.float32)

    X_train = _build_feature_matrix(session, train_ids)

    # Filter labels to match features — some crops may lack features
    # if their frames failed to decode
    valid_mask = np.array([
        session.crops[cid].features is not None and session.crops[cid].metadata is not None
        for cid in train_ids
    ])
    y_np = y_all[valid_mask]
    if len(y_np) == 0:
        logger.warning("No crops with features after filtering, cannot train")
        return {"accuracy": 0.0, "n_pos": n_pos, "n_neg": n_neg,
                "epochs": 0, "pending_scored": 0, "mean_uncertainty": 0.5}

    n_pos_actual = int(y_np.sum())
    n_neg_actual = len(y_np) - n_pos_actual
    if n_pos_actual == 0 or n_neg_actual == 0:
        logger.warning("After filtering: %d pos, %d neg — need both", n_pos_actual, n_neg_actual)
        return {"accuracy": 0.0, "n_pos": n_pos_actual, "n_neg": n_neg_actual,
                "epochs": 0, "pending_scored": 0, "mean_uncertainty": 0.5}

    y_train = torch.tensor(y_np, dtype=torch.float32)

    # Inverse-frequency class weights
    w_pos = len(y_np) / (2.0 * max(n_pos_actual, 1))
    w_neg = len(y_np) / (2.0 * max(n_neg_actual, 1))
    sample_weights = torch.where(y_train == 1.0, torch.tensor(w_pos), torch.tensor(w_neg))

    # Init model (or load cached)
    model = CropClassifier(input_dim=1032)
    cached_sd = load_model(session.cache_key)
    if cached_sd is not None:
        try:
            model.load_state_dict(cached_sd)
        except Exception as e:
            logger.warning("Could not load cached model: %s", e)
    model.train()

    device = torch.device(DEVICE)
    model, X_train, y_train, sample_weights = (
        model.to(device), X_train.to(device), y_train.to(device), sample_weights.to(device)
    )

    lr = compute_lr(1e-3, round_num)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_epochs = 20
    progress.step = "Training classifier"
    progress.total = n_epochs
    best_acc = 0.0

    for epoch in range(n_epochs):
        X_aug, y_aug = augment_features(X_train, y_train)
        n_aug_extra = X_aug.shape[0] - X_train.shape[0]
        aug_w = torch.cat([sample_weights, torch.ones(n_aug_extra, device=device)])

        optimizer.zero_grad()
        logits = model(X_aug).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(logits, y_aug, reduction="none") * aug_w).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(model(X_train).squeeze(-1)) >= 0.5).float()
            best_acc = max(best_acc, (preds == y_train).float().mean().item())
        progress.current = epoch + 1

    # Score pending crops
    progress.step = "Scoring pending crops"
    model.eval()
    pending = session.get_crops_by_label(CropLabel.PENDING)
    pending_scored = 0
    mean_unc = 0.5

    if pending:
        pids = [c.crop_id for c in pending]
        with torch.no_grad():
            probs = torch.sigmoid(model(_build_feature_matrix(session, pids).to(device)).squeeze(-1)).cpu().numpy()
        uncertainties = 1.0 - np.abs(2.0 * probs - 1.0)
        for i, cid in enumerate(pids):
            session.crops[cid].uncertainty = float(uncertainties[i])
        pending_scored = len(pids)
        mean_unc = float(np.mean(uncertainties))

    # Evaluate on held-out validation set
    progress.step = "Evaluating on validation set"
    val_result = _evaluate_validation(session, model, device, round_num, best_acc)

    # Persist
    save_model(session.cache_key, model.cpu().state_dict())
    session.model_trained = True
    session.training_epochs += n_epochs
    session.training_accuracy = best_acc
    session.touch()
    save_session(session)

    result = {"accuracy": best_acc, "n_pos": n_pos, "n_neg": n_neg,
              "epochs": n_epochs, "pending_scored": pending_scored,
              "mean_uncertainty": mean_unc, "round_num": round_num,
              "lr": lr}
    if val_result:
        result["val_accuracy"] = val_result["val_accuracy"]
        result["val_n"] = val_result["val_n_pos"] + val_result["val_n_neg"]
    logger.info("Training complete: %s", result)
    return result


