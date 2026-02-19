"""UFM (UniFlowMatch) model singleton and pairwise similarity computation.

Provides a lazy-loaded UFM-Base model for computing dense correspondence-based
similarity scores between crop image pairs. Used by the ReID pipeline as a
replacement for DINOv3 cosine similarity.

UFM-Base (428M params) produces covisibility masks via predict_correspondences_batched.
The mean covisibility score serves as the pairwise similarity metric.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals — singleton model
# ---------------------------------------------------------------------------

_ufm_model = None
_ufm_lock = None  # Lazy-init to avoid import-time threading issues

DEVICE = os.getenv("DEVICE", "cuda")
UFM_MODEL_NAME = os.getenv("UFM_MODEL", "infinity1096/UFM-Base")
UFM_BATCH_SIZE = int(os.getenv("UFM_BATCH_SIZE", "16"))


def _get_lock():
    """Lazy-init threading lock."""
    global _ufm_lock
    if _ufm_lock is None:
        import threading
        _ufm_lock = threading.Lock()
    return _ufm_lock


def _add_device_fix_hooks(model):
    """Fix UFM's image_scaler sending tensors to CPU.

    UFM's internal image_scaler runs on CPU, producing CPU tensors that
    then hit CUDA layers.  We add forward_pre_hooks to move inputs to
    the correct device before they reach:
      1. The encoder's patch_embed.proj (all models)
      2. The UNet feature branch (Refine model only)
    """
    # Hook 1: Encoder patch embedding
    proj = model.encoder.model.patch_embed.proj

    def enc_hook(module, args):
        x = args[0]
        target_device = module.weight.device
        if x.device != target_device:
            return (x.to(target_device),)

    proj.register_forward_pre_hook(enc_hook)

    # Hook 2: UNet branch (UFM-Refine only, but harmless to check)
    if hasattr(model, 'unet_feature') and getattr(model, 'use_unet_feature', False):
        def unet_hook(module, args):
            x = args[0]
            target_device = next(module.parameters()).device
            if x.device != target_device:
                return (x.to(target_device),)

        model.unet_feature.register_forward_pre_hook(unet_hook)
        logger.info("Added UNet device-fix hook (CPU->CUDA)")


def get_ufm_model():
    """Lazy-load and return the UFM model singleton.

    Thread-safe.  First call downloads weights and loads to GPU.
    Subsequent calls return the cached model.

    Returns:
        The UFM model (UniFlowMatchConfidence or UniFlowMatchClassificationRefinement).
    """
    global _ufm_model
    lock = _get_lock()

    with lock:
        if _ufm_model is not None:
            return _ufm_model

        import torch

        model_name = UFM_MODEL_NAME
        logger.info("Loading UFM model: %s ...", model_name)
        t0 = time.time()

        # Auto-detect model class from config
        from huggingface_hub import hf_hub_download
        import json as _json
        cfg_path = hf_hub_download(model_name, "config.json")
        with open(cfg_path) as f:
            cfg = _json.load(f)

        if "classification_head_kwargs" in cfg:
            from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
            cls = UniFlowMatchClassificationRefinement
            logger.info("Detected UniFlowMatchClassificationRefinement (Refine model)")
        else:
            from uniflowmatch.models.ufm import UniFlowMatchConfidence
            cls = UniFlowMatchConfidence
            logger.info("Detected UniFlowMatchConfidence (Base model)")

        model = cls.from_pretrained(model_name)
        model.float().eval().to(DEVICE)
        _add_device_fix_hooks(model)

        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        elapsed = time.time() - t0
        logger.info("UFM loaded: %.1fM params in %.1fs", n_params, elapsed)

        _ufm_model = model
        return _ufm_model


def release_ufm_model():
    """Release the UFM model to free GPU memory."""
    global _ufm_model
    lock = _get_lock()
    with lock:
        if _ufm_model is not None:
            import torch
            del _ufm_model
            _ufm_model = None
            torch.cuda.empty_cache()
            logger.info("Released UFM model")


# ---------------------------------------------------------------------------
# Pairwise similarity computation
# ---------------------------------------------------------------------------

def compute_pairwise_similarity(
    crop_images: List[Any],
    batch_size: int = UFM_BATCH_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Compute NxN pairwise similarity matrix using UFM covisibility.

    For each pair (i, j) where i < j, runs UFM's predict_correspondences_batched
    and takes the mean covisibility mask value as the similarity score.

    Args:
        crop_images: List of PIL Images (will be resized to 224x224 internally).
        batch_size: Number of pairs per forward pass.
        progress_callback: Optional fn(pairs_done, total_pairs) for progress.

    Returns:
        (N, N) float32 symmetric similarity matrix with 1.0 on the diagonal.
    """
    import torch
    from PIL import Image

    model = get_ufm_model()
    N = len(crop_images)

    # Prepare crop tensors: UFM expects uint8 (B, H, W, 3)
    target_size = (224, 224)
    crop_tensors = []
    for img in crop_images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        resized = img.convert("RGB").resize(target_size, Image.BILINEAR)
        crop_tensors.append(torch.from_numpy(np.array(resized)))

    # Build pair indices
    pair_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]
    total_pairs = len(pair_indices)

    sim_matrix = np.zeros((N, N), dtype=np.float32)
    np.fill_diagonal(sim_matrix, 1.0)

    if total_pairs == 0:
        return sim_matrix

    logger.info(
        "Computing UFM pairwise similarity: %d crops, %d pairs, bs=%d",
        N, total_pairs, batch_size,
    )
    t0 = time.time()
    pairs_done = 0

    for batch_start in range(0, total_pairs, batch_size):
        batch_pairs = pair_indices[batch_start:batch_start + batch_size]
        source_batch = torch.stack([crop_tensors[i] for i, j in batch_pairs])
        target_batch = torch.stack([crop_tensors[j] for i, j in batch_pairs])

        # UFM handles autocast internally — do NOT wrap with external autocast
        with torch.no_grad():
            result = model.predict_correspondences_batched(
                source_image=source_batch,
                target_image=target_batch,
            )

        # Extract similarity from covisibility mask or flow fallback
        if result.covisibility is not None:
            covis_masks = result.covisibility.mask
            for k, (i, j) in enumerate(batch_pairs):
                mean_covis = covis_masks[k].mean().item()
                sim_matrix[i, j] = mean_covis
                sim_matrix[j, i] = mean_covis
        else:
            # Fallback: inverse flow magnitude as similarity proxy
            flows = result.flow.flow_output
            for k, (i, j) in enumerate(batch_pairs):
                flow_mag = flows[k].norm(dim=0).mean().item()
                sim = 1.0 / (1.0 + flow_mag)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        pairs_done += len(batch_pairs)

        if progress_callback:
            progress_callback(pairs_done, total_pairs)

        if pairs_done % 200 == 0 or pairs_done == total_pairs:
            elapsed = time.time() - t0
            rate = pairs_done / max(elapsed, 0.001)
            eta = (total_pairs - pairs_done) / max(rate, 0.001)
            logger.info(
                "  UFM %d/%d (%.0f%%) [%.1f pairs/s, ETA %.0fs]",
                pairs_done, total_pairs,
                100 * pairs_done / total_pairs, rate, eta,
            )

    elapsed = time.time() - t0
    logger.info(
        "UFM pairwise done: %d pairs in %.1fs (%.1f pairs/s)",
        total_pairs, elapsed, total_pairs / max(elapsed, 0.001),
    )
    return sim_matrix


def extract_crop_images_from_session(
    session,
    read_frame_fn: Callable,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[List[Any], List[str]]:
    """Extract crop images for all accepted crops with features.

    Args:
        session: InterviewSession with crops.
        read_frame_fn: Function(video_path, frame_idx, cache_key=...) -> PIL.Image.
        target_size: Resize target for crop images.

    Returns:
        (crop_images, crop_ids) — parallel lists of PIL Images and crop IDs.
    """
    from PIL import Image
    from .state import CropLabel

    # ReID should cluster crops from the current task timeline only.
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED, include_imported=False)
    # Sort by frame then crop_id for deterministic ordering
    accepted.sort(key=lambda c: (c.frame_idx, c.crop_id))

    crop_images = []
    crop_ids = []

    # Group by frame to minimize video seeks
    frame_to_crops: Dict[int, list] = {}
    for crop in accepted:
        frame_to_crops.setdefault(crop.frame_idx, []).append(crop)

    for frame_idx in sorted(frame_to_crops.keys()):
        frame = read_frame_fn(
            session.video_path, frame_idx, cache_key=session.cache_key,
        )
        if frame is None:
            logger.warning("Could not read frame %d for UFM crop extraction", frame_idx)
            continue

        for crop in frame_to_crops[frame_idx]:
            x1, y1, x2, y2 = [int(round(v)) for v in crop.xyxy]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.width, x2)
            y2 = min(frame.height, y2)

            cropped = frame.crop((x1, y1, x2, y2))
            cropped = cropped.resize(target_size, Image.BILINEAR)
            crop_images.append(cropped)
            crop_ids.append(crop.crop_id)

    logger.info("Extracted %d crop images for UFM from %d frames",
                len(crop_images), len(frame_to_crops))
    return crop_images, crop_ids
