"""Mask-quality feature computation and training utilities.

Pure numpy/scipy — no torch, PIL, or model dependencies.
"""

from __future__ import annotations


import numpy as np


def compute_mask_quality(
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Compute mask-quality features for classifier input.

    Args:
        mask:         Binary mask (H, W) from SAM3 instance segmentation.
        box_xyxy:     Bounding box [x1, y1, x2, y2] in pixel coords.
        frame_width:  Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        (4,) float32 array: [mask_fill_ratio, detection_score_placeholder,
                              edge_contact, mask_compactness]
        detection_score is set to 0.0 here — caller should overwrite with
        the actual SAM3 detection score.
    """
    x1, y1, x2, y2 = box_xyxy
    box_area = max(1.0, float((x2 - x1) * (y2 - y1)))

    # Mask fill ratio: how much of the box is filled by the mask
    mask_region = mask[int(max(0, y1)):int(min(frame_height, y2)),
                       int(max(0, x1)):int(min(frame_width, x2))]
    mask_area = float(np.count_nonzero(mask_region))
    fill_ratio = mask_area / box_area

    # Edge contact: fraction of box edges that touch frame boundary
    edge_margin = 3  # pixels
    contacts = 0
    if x1 <= edge_margin:
        contacts += 1
    if y1 <= edge_margin:
        contacts += 1
    if x2 >= frame_width - edge_margin:
        contacts += 1
    if y2 >= frame_height - edge_margin:
        contacts += 1
    edge_contact = contacts / 4.0

    # Mask compactness: 4pi * area / perimeter^2 (circle = 1.0)
    if np.count_nonzero(mask) < 4:
        compactness = 0.0
    else:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        perimeter = float(np.count_nonzero(mask) - np.count_nonzero(eroded))
        perimeter = max(1.0, perimeter)
        total_mask_area = float(np.count_nonzero(mask))
        compactness = min(1.0, (4.0 * np.pi * total_mask_area) / (perimeter * perimeter))

    return np.array([fill_ratio, 0.0, edge_contact, compactness], dtype=np.float32)


# ---------------------------------------------------------------------------
# LR decay for round-based training
# ---------------------------------------------------------------------------

