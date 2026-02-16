# Extract Instance Segmentation Masks from Snippet Videos — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Python CLI tool that takes a snippet `.mp4` and its per-frame bounding-box JSON (same format as `overlay_snippet_bboxes.sh`), runs each frame through `Sam3Model` with the bounding box as a positive prompt, and writes per-frame binary instance-segmentation masks as PNG images at original video resolution.

**Architecture:** The script (`extract_snippet_masks.py`) follows the same CLI pattern as `overlay_snippet_bboxes.sh` but is Python (because it requires `Sam3Model` GPU inference). For each frame with a bounding box, it decodes the frame via PyAV, converts the percent-coordinate bbox to pixel xyxy, runs `Sam3Model` + `Sam3Processor` with a combined text + positive-box prompt, and saves the resulting binary mask as a PNG. Frames without a bbox get a blank (all-zero) mask. An optional `--mask-video` flag encodes all masks into a grayscale MP4, and `--overlay-video` composites masks onto the original frames.

**Tech Stack:** Python 3.12, PyTorch, HuggingFace Transformers (`Sam3Model`, `Sam3Processor`), PyAV, PIL, NumPy, `seeding_common.py` (model singletons)

---

## Reference: Input JSON Format

From `overlay_snippet_bboxes.sh`, the bbox JSON is a flat array:

```json
[
  {"x": 25.0, "y": 10.0, "width": 15.0, "height": 40.0, "snippet_frame": 1},
  {"x": 25.5, "y": 10.2, "width": 14.8, "height": 39.5, "snippet_frame": 2},
  ...
]
```

- `x`, `y`, `width`, `height` — **percent coordinates [0, 100]**, top-left origin (same as Label Studio)
- `snippet_frame` — **1-based** frame number within the snippet video

## Reference: SAM3 API (from HuggingFace Transformers docs)

```python
from transformers import Sam3Model, Sam3Processor

processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)

# Combined text + box prompt
inputs = processor(
    images=pil_frame,           # PIL.Image.Image
    text="person",              # text prompt: WHAT to segment
    input_boxes=[[box_xyxy]],   # [[x1, y1, x2, y2]] pixel coords
    input_boxes_labels=[[1]],   # 1 = positive box
    return_tensors="pt",
).to(device)

with torch.inference_mode():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist(),  # resize masks to original
)[0]

# results["masks"]  — list of 2D boolean tensors at original (H, W)
# results["boxes"]  — list of [x1, y1, x2, y2] pixel coords
# results["scores"] — list of float confidence scores
```

## Reference: Coordinate Conversion (concrete example)

Video: 1920x1080. JSON entry: `{"x": 25.0, "y": 10.0, "width": 15.0, "height": 40.0, "snippet_frame": 1}`

```
snippet_frame=1 → frame_idx=0 (1-based to 0-based)
x1 = (25.0 / 100) * 1920 = 480
y1 = (10.0 / 100) * 1080 = 108
x2 = 480 + (15.0 / 100) * 1920 = 480 + 288 = 768
y2 = 108 + (40.0 / 100) * 1080 = 108 + 432 = 540
box_xyxy = [480, 108, 768, 540]
```

## Output Structure

```
output_dir/
├── mask_000001.png   # Frame 1: (1920x1080) binary, 0=background, 255=person
├── mask_000002.png   # Frame 2
├── ...
├── mask_000300.png   # Last frame
├── masks.mp4         # (optional --mask-video) white-on-black mask video
└── overlay.mp4       # (optional --overlay-video) original + translucent mask
```

---

## Task 1: Core helpers — JSON loading and coordinate conversion

**Files:**
- Create: `extract_snippet_masks.py`
- Test: `test_extract_snippet_masks.py`

### Step 1: Write failing test for JSON loading

```python
# test_extract_snippet_masks.py
"""Tests for extract_snippet_masks.py — SAM3 instance mask extraction from snippets."""

import json
import os
import tempfile

import numpy as np
import pytest


def test_load_bbox_json_basic():
    """Load a valid bbox JSON and verify parsed entries."""
    from extract_snippet_masks import load_bbox_json

    data = [
        {"x": 25.0, "y": 10.0, "width": 15.0, "height": 40.0, "snippet_frame": 1},
        {"x": 26.0, "y": 11.0, "width": 14.0, "height": 39.0, "snippet_frame": 2},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        result = load_bbox_json(path)
        assert len(result) == 2
        assert result[0]["snippet_frame"] == 1
        assert result[1]["x"] == 26.0
    finally:
        os.unlink(path)


def test_load_bbox_json_empty():
    """Empty JSON array raises ValueError."""
    from extract_snippet_masks import load_bbox_json

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([], f)
        path = f.name
    try:
        with pytest.raises(ValueError, match="empty"):
            load_bbox_json(path)
    finally:
        os.unlink(path)


def test_percent_xywh_to_xyxy():
    """Convert percent bbox to pixel xyxy."""
    from extract_snippet_masks import percent_xywh_to_xyxy

    # Video: 1920x1080, bbox: x=25%, y=10%, w=15%, h=40%
    box = percent_xywh_to_xyxy(25.0, 10.0, 15.0, 40.0, 1920, 1080)
    assert box == [480, 108, 768, 540]


def test_percent_xywh_to_xyxy_clamping():
    """Boxes extending past image edges are clamped."""
    from extract_snippet_masks import percent_xywh_to_xyxy

    # x=90%, w=20% → x2 would be 110% → clamped to width
    box = percent_xywh_to_xyxy(90.0, 0.0, 20.0, 50.0, 100, 100)
    assert box[2] == 100  # clamped to width
    assert box[0] == 90
    assert box[3] == 50


def test_build_frame_bbox_map():
    """Build a {0-based frame_idx: xyxy} dict from JSON entries."""
    from extract_snippet_masks import build_frame_bbox_map

    entries = [
        {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "snippet_frame": 1},
        {"x": 15.0, "y": 25.0, "width": 35.0, "height": 45.0, "snippet_frame": 5},
    ]
    frame_map = build_frame_bbox_map(entries, img_w=200, img_h=100)
    # snippet_frame=1 → frame_idx=0
    assert 0 in frame_map
    assert frame_map[0] == [20, 20, 80, 60]  # (10/100)*200=20, (20/100)*100=20, ...
    # snippet_frame=5 → frame_idx=4
    assert 4 in frame_map
```

### Step 2: Run test to verify it fails

Run: `cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video && python -m pytest test_extract_snippet_masks.py -v -x`
Expected: FAIL — `ModuleNotFoundError: No module named 'extract_snippet_masks'`

### Step 3: Implement JSON loading and coordinate conversion

```python
# extract_snippet_masks.py (initial skeleton)
"""
Extract per-frame instance segmentation masks from snippet videos using SAM3.

Takes an MP4 video and its per-frame bounding-box JSON (same format as
overlay_snippet_bboxes.sh) and produces binary PNG masks at original resolution.

Uses Sam3Model with combined text + positive-box prompts for robust segmentation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_bbox_json(path: str) -> List[Dict[str, Any]]:
    """Load and validate the per-frame bounding box JSON.

    Expected format: array of objects with keys:
      x, y, width, height (percent [0,100]) and snippet_frame (1-based).
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"BBox JSON is empty or not an array: {path}")
    return data


def percent_xywh_to_xyxy(
    x_pct: float, y_pct: float, w_pct: float, h_pct: float,
    img_w: int, img_h: int,
) -> List[int]:
    """Convert percent [0,100] xywh bbox to pixel [x1, y1, x2, y2], clamped to image bounds."""
    x1 = int(round((x_pct / 100.0) * img_w))
    y1 = int(round((y_pct / 100.0) * img_h))
    x2 = int(round(((x_pct + w_pct) / 100.0) * img_w))
    y2 = int(round(((y_pct + h_pct) / 100.0) * img_h))
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    return [x1, y1, x2, y2]


def build_frame_bbox_map(
    entries: List[Dict[str, Any]], img_w: int, img_h: int,
) -> Dict[int, List[int]]:
    """Build a mapping from 0-based frame index to pixel xyxy bbox.

    If multiple entries share a snippet_frame, the last one wins.
    """
    frame_map: Dict[int, List[int]] = {}
    for entry in entries:
        sf = int(entry["snippet_frame"])
        frame_idx = sf - 1  # 1-based to 0-based
        box = percent_xywh_to_xyxy(
            float(entry["x"]), float(entry["y"]),
            float(entry["width"]), float(entry["height"]),
            img_w, img_h,
        )
        frame_map[frame_idx] = box
    return frame_map
```

### Step 4: Run test to verify it passes

Run: `cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video && python -m pytest test_extract_snippet_masks.py -v -x`
Expected: PASS (all 5 tests)

### Step 5: Commit

```bash
git add extract_snippet_masks.py test_extract_snippet_masks.py
git commit -m "feat(extract-masks): add JSON loading and coordinate conversion helpers"
```

---

## Task 2: SAM3 mask extraction core — mock-based test and implementation

**Files:**
- Modify: `extract_snippet_masks.py`
- Modify: `test_extract_snippet_masks.py`

### Step 1: Write failing test for single-frame mask extraction

This test mocks `Sam3Model` and `Sam3Processor` to verify the inference pipeline without GPU.

```python
# Add to test_extract_snippet_masks.py
from unittest.mock import MagicMock, patch
from PIL import Image


def _make_mock_sam3():
    """Create mock Sam3Model + Sam3Processor that return a known binary mask."""
    mock_model = MagicMock(name="Sam3Model")
    mock_processor = MagicMock(name="Sam3Processor")

    # Processor __call__ returns a dict-like object with original_sizes
    proc_output = MagicMock()
    proc_output.get.return_value = [[200, 300]]  # original_sizes: [[H, W]]
    proc_output.to.return_value = proc_output
    mock_processor.return_value = proc_output

    # Model __call__ returns outputs
    mock_outputs = MagicMock(name="model_outputs")
    mock_model.return_value = mock_outputs

    # post_process_instance_segmentation returns list of result dicts
    # Mask: 200x300, person region in center
    import torch
    mask = torch.zeros(200, 300, dtype=torch.bool)
    mask[50:150, 80:220] = True  # person region
    result = {
        "masks": [mask],
        "boxes": [torch.tensor([80.0, 50.0, 220.0, 150.0])],
        "scores": [torch.tensor(0.92)],
    }
    mock_processor.post_process_instance_segmentation.return_value = [result]

    return mock_model, mock_processor


def test_extract_mask_single_frame():
    """Extract mask for a single frame with a box prompt."""
    from extract_snippet_masks import extract_mask_for_frame
    import numpy as np

    mock_model, mock_processor = _make_mock_sam3()
    frame = Image.new("RGB", (300, 200), color=(128, 128, 128))
    box_xyxy = [80, 50, 220, 150]

    mask, score = extract_mask_for_frame(
        frame, box_xyxy,
        model=mock_model, processor=mock_processor,
        text_prompt="person",
    )

    assert mask.shape == (200, 300)  # same as frame dimensions
    assert mask.dtype == np.uint8
    assert mask.max() == 255  # binary: 0 or 255
    assert mask[100, 150] == 255  # center of person region
    assert mask[0, 0] == 0  # background
    assert score > 0.5


def test_extract_mask_no_detection():
    """When SAM3 finds no masks, return blank mask with score 0."""
    from extract_snippet_masks import extract_mask_for_frame
    import numpy as np

    mock_model, mock_processor = _make_mock_sam3()
    # Override post_process to return empty
    mock_processor.post_process_instance_segmentation.return_value = [
        {"masks": [], "boxes": [], "scores": []}
    ]

    frame = Image.new("RGB", (300, 200))
    box_xyxy = [80, 50, 220, 150]

    mask, score = extract_mask_for_frame(
        frame, box_xyxy,
        model=mock_model, processor=mock_processor,
        text_prompt="person",
    )

    assert mask.shape == (200, 300)
    assert mask.max() == 0  # blank mask
    assert score == 0.0
```

### Step 2: Run test to verify it fails

Run: `python -m pytest test_extract_snippet_masks.py::test_extract_mask_single_frame -v -x`
Expected: FAIL — `ImportError: cannot import name 'extract_mask_for_frame'`

### Step 3: Implement `extract_mask_for_frame`

Add to `extract_snippet_masks.py`:

```python
import numpy as np
import torch
from PIL import Image


def extract_mask_for_frame(
    frame: Image.Image,
    box_xyxy: List[int],
    *,
    model,
    processor,
    text_prompt: str = "person",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cpu",
    dtype=None,
) -> Tuple[np.ndarray, float]:
    """Run Sam3Model on a single frame with a box prompt and return binary mask.

    Args:
        frame: PIL RGB image.
        box_xyxy: Bounding box [x1, y1, x2, y2] in pixel coordinates.
        model: Sam3Model instance.
        processor: Sam3Processor instance.
        text_prompt: Text describing the object (e.g. "person").
        threshold: Confidence threshold for post-processing.
        mask_threshold: Mask binarization threshold.
        device: Torch device string.
        dtype: Torch dtype for autocast (e.g. torch.bfloat16).

    Returns:
        (mask, score) where mask is uint8 (H, W) with 0=bg, 255=fg,
        and score is float confidence. Returns blank mask with 0.0 on failure.
    """
    w, h = frame.size
    blank = np.zeros((h, w), dtype=np.uint8)

    try:
        inputs = processor(
            images=frame,
            text=text_prompt,
            input_boxes=[[box_xyxy]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            if dtype is not None and device != "cpu":
                with torch.autocast(device_type=device, dtype=dtype):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])

        if len(masks) == 0:
            return blank, 0.0

        # Select best mask by confidence
        if len(scores) > 0:
            score_vals = [
                s.item() if hasattr(s, "item") else float(s)
                for s in scores
            ]
            best_idx = int(np.argmax(score_vals))
            best_score = score_vals[best_idx]
        else:
            best_idx = 0
            best_score = 0.5

        best_mask = masks[best_idx]
        if hasattr(best_mask, "cpu"):
            best_mask = best_mask.cpu().numpy()
        elif hasattr(best_mask, "numpy"):
            best_mask = best_mask.numpy()

        # Convert to uint8 binary: 0 or 255
        mask_uint8 = (best_mask.astype(bool).astype(np.uint8)) * 255

        # Ensure correct shape (H, W)
        if mask_uint8.ndim > 2:
            mask_uint8 = mask_uint8.squeeze()
        if mask_uint8.shape != (h, w):
            # Resize if shape doesn't match (shouldn't happen with target_sizes)
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask_uint8, mode="L")
            mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
            mask_uint8 = np.array(mask_pil)

        return mask_uint8, best_score

    except Exception as exc:
        logger.warning("SAM3 mask extraction failed: %s", exc)
        return blank, 0.0
```

### Step 4: Run test to verify it passes

Run: `python -m pytest test_extract_snippet_masks.py::test_extract_mask_single_frame test_extract_snippet_masks.py::test_extract_mask_no_detection -v -x`
Expected: PASS

### Step 5: Commit

```bash
git add extract_snippet_masks.py test_extract_snippet_masks.py
git commit -m "feat(extract-masks): add SAM3 single-frame mask extraction"
```

---

## Task 3: Full pipeline — decode video, process all frames, write mask PNGs

**Files:**
- Modify: `extract_snippet_masks.py`
- Modify: `test_extract_snippet_masks.py`

### Step 1: Write failing test for the full pipeline

```python
# Add to test_extract_snippet_masks.py
import struct
import tempfile


def _make_minimal_mp4(width=160, height=120, num_frames=5, fps=25):
    """Create a minimal MP4 with solid-color frames via PyAV."""
    import av

    path = os.path.join(tempfile.mkdtemp(), "test_snippet.mp4")
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        # Solid color frames: R channel varies per frame
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = int(255 * i / max(num_frames - 1, 1))
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def test_process_snippet_full_pipeline(tmp_path):
    """Full pipeline: video + bbox JSON → mask PNGs."""
    from extract_snippet_masks import process_snippet

    video_path = _make_minimal_mp4(width=160, height=120, num_frames=5)

    # Create bbox JSON: bboxes for frames 1, 3, 5 (1-based)
    bbox_data = [
        {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "snippet_frame": 1},
        {"x": 12.0, "y": 22.0, "width": 28.0, "height": 38.0, "snippet_frame": 3},
        {"x": 14.0, "y": 24.0, "width": 26.0, "height": 36.0, "snippet_frame": 5},
    ]
    bbox_path = str(tmp_path / "bboxes.json")
    with open(bbox_path, "w") as f:
        json.dump(bbox_data, f)

    output_dir = str(tmp_path / "masks")

    # Mock the SAM3 model to return a mask with person in the box region
    mock_model, mock_processor = _make_mock_sam3()

    # Override mock to return masks sized to whatever original_sizes says
    def fake_post_process(outputs, threshold, mask_threshold, target_sizes):
        import torch
        h, w = target_sizes[0]
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[30:80, 20:60] = True  # person region
        return [{"masks": [mask], "boxes": [torch.tensor([20, 30, 60, 80])], "scores": [torch.tensor(0.9)]}]

    mock_processor.post_process_instance_segmentation.side_effect = fake_post_process

    stats = process_snippet(
        video_path=video_path,
        bbox_json_path=bbox_path,
        output_dir=output_dir,
        model=mock_model,
        processor=mock_processor,
        text_prompt="person",
    )

    # Check output files exist
    assert os.path.isdir(output_dir)
    # Should have mask PNGs for frames with bboxes
    for sf in [1, 3, 5]:
        mask_path = os.path.join(output_dir, f"mask_{sf:06d}.png")
        assert os.path.isfile(mask_path), f"Missing {mask_path}"
        img = Image.open(mask_path)
        assert img.size == (160, 120)  # W x H matches video
        assert img.mode == "L"  # grayscale

    # Frames without bboxes (2, 4) should have blank masks
    for sf in [2, 4]:
        mask_path = os.path.join(output_dir, f"mask_{sf:06d}.png")
        assert os.path.isfile(mask_path), f"Missing blank mask {mask_path}"
        arr = np.array(Image.open(mask_path))
        assert arr.max() == 0  # all black

    assert stats["frames_processed"] == 3
    assert stats["frames_blank"] == 2
    assert stats["total_frames"] == 5


def test_process_snippet_writes_scores_json(tmp_path):
    """Pipeline writes a scores.json alongside masks."""
    from extract_snippet_masks import process_snippet

    video_path = _make_minimal_mp4(width=80, height=60, num_frames=2)

    bbox_data = [
        {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "snippet_frame": 1},
        {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "snippet_frame": 2},
    ]
    bbox_path = str(tmp_path / "bboxes.json")
    with open(bbox_path, "w") as f:
        json.dump(bbox_data, f)

    output_dir = str(tmp_path / "masks")
    mock_model, mock_processor = _make_mock_sam3()

    def fake_post_process(outputs, threshold, mask_threshold, target_sizes):
        import torch
        h, w = target_sizes[0]
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[10:40, 10:50] = True
        return [{"masks": [mask], "boxes": [], "scores": [torch.tensor(0.85)]}]

    mock_processor.post_process_instance_segmentation.side_effect = fake_post_process

    process_snippet(
        video_path=video_path,
        bbox_json_path=bbox_path,
        output_dir=output_dir,
        model=mock_model,
        processor=mock_processor,
    )

    scores_path = os.path.join(output_dir, "scores.json")
    assert os.path.isfile(scores_path)
    with open(scores_path) as f:
        scores = json.load(f)
    # Keys are snippet_frame strings, values are float scores
    assert "1" in scores
    assert "2" in scores
    assert scores["1"] > 0.5
```

### Step 2: Run test to verify it fails

Run: `python -m pytest test_extract_snippet_masks.py::test_process_snippet_full_pipeline -v -x`
Expected: FAIL — `ImportError: cannot import name 'process_snippet'`

### Step 3: Implement `process_snippet`

Add to `extract_snippet_masks.py`:

```python
def process_snippet(
    video_path: str,
    bbox_json_path: str,
    output_dir: str,
    *,
    model=None,
    processor=None,
    text_prompt: str = "person",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cpu",
    dtype=None,
) -> Dict[str, Any]:
    """Process a snippet video: extract SAM3 instance masks for each frame.

    For each frame that has a bounding box in the JSON, runs Sam3Model with the
    box as a positive prompt and saves the binary mask as a PNG. Frames without
    a bbox get a blank (all-zero) mask.

    Args:
        video_path: Path to the snippet .mp4 file.
        bbox_json_path: Path to the per-frame bounding box JSON.
        output_dir: Directory to write mask PNGs and scores.json.
        model: Sam3Model instance (loaded lazily if None).
        processor: Sam3Processor instance (loaded lazily if None).
        text_prompt: Text prompt for SAM3 (default "person").
        threshold: Post-processing confidence threshold.
        mask_threshold: Mask binarization threshold.
        device: Torch device.
        dtype: Torch dtype for autocast.

    Returns:
        Dict with stats: frames_processed, frames_blank, total_frames, avg_score.
    """
    import av
    from PIL import Image as PILImage

    # Load model if not provided
    if model is None or processor is None:
        import seeding_common as base
        model, processor = base._get_sam3_image_model()
        device = base.DEVICE
        dtype = base.DTYPE

    # Load bbox JSON
    entries = load_bbox_json(bbox_json_path)

    # Get video dimensions and frame count
    container = av.open(video_path)
    stream = container.streams.video[0]
    img_w = stream.codec_context.width
    img_h = stream.codec_context.height
    total_frames = stream.frames
    if not total_frames and stream.duration and stream.time_base:
        fps_est = float(stream.average_rate) if stream.average_rate else 30.0
        total_frames = int(float(stream.duration * stream.time_base) * fps_est)
    container.close()

    # Build frame → bbox map
    frame_map = build_frame_bbox_map(entries, img_w, img_h)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Decode all frames sequentially and process
    container = av.open(video_path)
    scores_map: Dict[str, float] = {}
    frames_processed = 0
    frames_blank = 0
    frame_idx = 0

    try:
        for av_frame in container.decode(video=0):
            if frame_idx >= total_frames and total_frames > 0:
                break

            snippet_frame = frame_idx + 1  # 0-based to 1-based
            mask_filename = f"mask_{snippet_frame:06d}.png"
            mask_path = os.path.join(output_dir, mask_filename)

            if frame_idx in frame_map:
                pil_frame = av_frame.to_image()
                box_xyxy = frame_map[frame_idx]

                mask, score = extract_mask_for_frame(
                    pil_frame, box_xyxy,
                    model=model, processor=processor,
                    text_prompt=text_prompt,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    device=device,
                    dtype=dtype,
                )
                frames_processed += 1
                scores_map[str(snippet_frame)] = round(float(score), 4)
            else:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                frames_blank += 1
                scores_map[str(snippet_frame)] = 0.0

            # Save mask as grayscale PNG
            PILImage.fromarray(mask, mode="L").save(mask_path)

            frame_idx += 1

            # Progress logging
            if frame_idx % 50 == 0 or frame_idx == total_frames:
                logger.info(
                    "Progress: %d/%d frames (processed=%d, blank=%d)",
                    frame_idx, total_frames, frames_processed, frames_blank,
                )
    finally:
        container.close()

    # Write scores JSON
    scores_path = os.path.join(output_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores_map, f, indent=2)

    avg_score = (
        sum(v for v in scores_map.values() if v > 0) / max(frames_processed, 1)
    )

    stats = {
        "frames_processed": frames_processed,
        "frames_blank": frames_blank,
        "total_frames": frame_idx,  # actual decoded count
        "avg_score": round(avg_score, 4),
    }
    logger.info("Done: %s", stats)
    return stats
```

### Step 4: Run test to verify it passes

Run: `python -m pytest test_extract_snippet_masks.py::test_process_snippet_full_pipeline test_extract_snippet_masks.py::test_process_snippet_writes_scores_json -v -x`
Expected: PASS

### Step 5: Commit

```bash
git add extract_snippet_masks.py test_extract_snippet_masks.py
git commit -m "feat(extract-masks): add full pipeline — decode, segment, write mask PNGs"
```

---

## Task 4: Optional mask video and overlay video output

**Files:**
- Modify: `extract_snippet_masks.py`
- Modify: `test_extract_snippet_masks.py`

### Step 1: Write failing test for mask video encoding

```python
# Add to test_extract_snippet_masks.py
def test_encode_mask_video(tmp_path):
    """Encode mask PNGs into a grayscale MP4."""
    from extract_snippet_masks import encode_mask_video

    output_dir = str(tmp_path / "masks")
    os.makedirs(output_dir)

    # Create 3 dummy mask PNGs
    for i in range(1, 4):
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[30:80, 40:100] = 255
        Image.fromarray(mask, mode="L").save(
            os.path.join(output_dir, f"mask_{i:06d}.png")
        )

    video_path = str(tmp_path / "masks.mp4")
    encode_mask_video(output_dir, video_path, fps=25.0)

    assert os.path.isfile(video_path)
    assert os.path.getsize(video_path) > 0

    # Verify it's a valid video with correct dimensions
    import av
    c = av.open(video_path)
    s = c.streams.video[0]
    assert s.codec_context.width == 160
    assert s.codec_context.height == 120
    c.close()


def test_encode_overlay_video(tmp_path):
    """Composite masks onto original video frames."""
    from extract_snippet_masks import encode_overlay_video

    # Create a source video
    video_path = _make_minimal_mp4(width=160, height=120, num_frames=3)

    # Create mask PNGs
    mask_dir = str(tmp_path / "masks")
    os.makedirs(mask_dir)
    for i in range(1, 4):
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[30:80, 40:100] = 255
        Image.fromarray(mask, mode="L").save(
            os.path.join(mask_dir, f"mask_{i:06d}.png")
        )

    overlay_path = str(tmp_path / "overlay.mp4")
    encode_overlay_video(video_path, mask_dir, overlay_path, alpha=0.4)

    assert os.path.isfile(overlay_path)
    assert os.path.getsize(overlay_path) > 0
```

### Step 2: Run test to verify it fails

Run: `python -m pytest test_extract_snippet_masks.py::test_encode_mask_video -v -x`
Expected: FAIL — `ImportError`

### Step 3: Implement mask video and overlay encoding

Add to `extract_snippet_masks.py`:

```python
def encode_mask_video(
    mask_dir: str, output_path: str, fps: float = 30.0,
) -> None:
    """Encode mask PNGs in a directory into a grayscale MP4.

    Reads mask_000001.png, mask_000002.png, ... in order and encodes as
    a video where white pixels = person, black = background.
    """
    import av
    import glob

    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))
    if not mask_paths:
        raise ValueError(f"No mask PNGs found in {mask_dir}")

    # Read first mask to get dimensions
    first_mask = np.array(Image.open(mask_paths[0]))
    h, w = first_mask.shape[:2]

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for mask_path in mask_paths:
        mask_gray = np.array(Image.open(mask_path).convert("L"))
        # Convert to RGB for encoding (white on black)
        rgb = np.stack([mask_gray, mask_gray, mask_gray], axis=-1)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    logger.info("Wrote mask video: %s (%d frames)", output_path, len(mask_paths))


def encode_overlay_video(
    source_video: str,
    mask_dir: str,
    output_path: str,
    alpha: float = 0.4,
    fps: Optional[float] = None,
) -> None:
    """Composite mask PNGs onto original video frames as translucent overlay.

    The mask region is tinted green (configurable) with the given alpha.
    """
    import av
    import glob

    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))
    if not mask_paths:
        raise ValueError(f"No mask PNGs found in {mask_dir}")

    # Get source video info
    src = av.open(source_video)
    src_stream = src.streams.video[0]
    w = src_stream.codec_context.width
    h = src_stream.codec_context.height
    if fps is None:
        fps = float(src_stream.average_rate) if src_stream.average_rate else 30.0
    src.close()

    # Open source for decoding
    src = av.open(source_video)
    out = av.open(output_path, mode="w")
    out_stream = out.add_stream("libx264", rate=int(fps))
    out_stream.width = w
    out_stream.height = h
    out_stream.pix_fmt = "yuv420p"

    # Green overlay color
    overlay_color = np.array([0, 255, 100], dtype=np.float32)

    frame_idx = 0
    for av_frame in src.decode(video=0):
        rgb = av_frame.to_ndarray(format="rgb24").astype(np.float32)

        if frame_idx < len(mask_paths):
            mask = np.array(
                Image.open(mask_paths[frame_idx]).convert("L")
            ).astype(np.float32) / 255.0

            # Blend: where mask > 0, mix original with overlay color
            mask_3d = mask[:, :, np.newaxis]
            blended = rgb * (1.0 - alpha * mask_3d) + overlay_color * alpha * mask_3d
            rgb = blended

        composite = np.clip(rgb, 0, 255).astype(np.uint8)
        out_frame = av.VideoFrame.from_ndarray(composite, format="rgb24")
        for packet in out_stream.encode(out_frame):
            out.mux(packet)

        frame_idx += 1

    for packet in out_stream.encode():
        out.mux(packet)

    src.close()
    out.close()
    logger.info("Wrote overlay video: %s (%d frames)", output_path, frame_idx)
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest test_extract_snippet_masks.py::test_encode_mask_video test_extract_snippet_masks.py::test_encode_overlay_video -v -x`
Expected: PASS

### Step 5: Commit

```bash
git add extract_snippet_masks.py test_extract_snippet_masks.py
git commit -m "feat(extract-masks): add mask video and overlay video encoding"
```

---

## Task 5: CLI entry point with argparse

**Files:**
- Modify: `extract_snippet_masks.py`

### Step 1: Write failing test for CLI

```python
# Add to test_extract_snippet_masks.py
import subprocess


def test_cli_help():
    """CLI --help exits 0 and shows usage."""
    result = subprocess.run(
        [sys.executable, "extract_snippet_masks.py", "--help"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert result.returncode == 0
    assert "--snippet" in result.stdout
    assert "--bbox-json" in result.stdout


def test_cli_missing_args():
    """CLI with missing required args exits non-zero."""
    result = subprocess.run(
        [sys.executable, "extract_snippet_masks.py"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert result.returncode != 0
```

### Step 2: Run test to verify it fails

Run: `python -m pytest test_extract_snippet_masks.py::test_cli_help -v -x`
Expected: FAIL — no `main()` or argparse defined

### Step 3: Implement CLI

Add to `extract_snippet_masks.py`:

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-frame instance segmentation masks from a snippet video "
            "using SAM3 with bounding-box prompts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n\n"
            "  python extract_snippet_masks.py \\\n"
            "    --snippet snippets_proj.../snippet_001.mp4 \\\n"
            "    --bbox-json snippets_proj.../snippet_001_bboxes.json \\\n"
            "    --output-dir snippets_proj.../masks/ \\\n"
            "    --text-prompt person\n"
        ),
    )
    parser.add_argument(
        "--snippet", required=True,
        help="Path to the snippet MP4 video file.",
    )
    parser.add_argument(
        "--bbox-json", required=True,
        help="Path to the per-frame bounding box JSON file.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Directory to write mask PNGs. Defaults to "
            "<snippet_basename>_masks/ next to the snippet."
        ),
    )
    parser.add_argument(
        "--text-prompt", default="person",
        help="Text prompt for SAM3 (default: 'person').",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Post-processing confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--mask-video", action="store_true",
        help="Also encode masks as a grayscale MP4.",
    )
    parser.add_argument(
        "--overlay-video", action="store_true",
        help="Also encode an overlay composite video.",
    )
    parser.add_argument(
        "--overlay-alpha", type=float, default=0.4,
        help="Overlay transparency (default: 0.4).",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not os.path.isfile(args.snippet):
        logger.error("Snippet not found: %s", args.snippet)
        sys.exit(1)
    if not os.path.isfile(args.bbox_json):
        logger.error("BBox JSON not found: %s", args.bbox_json)
        sys.exit(1)

    # Default output dir: <snippet_base>_masks/
    output_dir = args.output_dir
    if output_dir is None:
        base = os.path.splitext(args.snippet)[0]
        output_dir = f"{base}_masks"

    logger.info("=" * 70)
    logger.info("EXTRACT SNIPPET MASKS — SAM3 Instance Segmentation")
    logger.info("=" * 70)
    logger.info("  Snippet:     %s", args.snippet)
    logger.info("  BBox JSON:   %s", args.bbox_json)
    logger.info("  Output dir:  %s", output_dir)
    logger.info("  Text prompt: %s", args.text_prompt)
    logger.info("  Threshold:   %.2f", args.threshold)
    logger.info("=" * 70)

    stats = process_snippet(
        video_path=args.snippet,
        bbox_json_path=args.bbox_json,
        output_dir=output_dir,
        text_prompt=args.text_prompt,
        threshold=args.threshold,
    )

    if args.mask_video:
        import av as _av  # verify import
        mask_video_path = os.path.join(output_dir, "masks.mp4")
        # Infer FPS from source video
        c = _av.open(args.snippet)
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else 30.0
        c.close()
        encode_mask_video(output_dir, mask_video_path, fps=fps)

    if args.overlay_video:
        overlay_path = os.path.join(output_dir, "overlay.mp4")
        encode_overlay_video(
            args.snippet, output_dir, overlay_path, alpha=args.overlay_alpha,
        )

    logger.info("=" * 70)
    logger.info("COMPLETE — %d frames segmented, %d blank, avg score %.3f",
                stats["frames_processed"], stats["frames_blank"], stats["avg_score"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
```

### Step 4: Run CLI tests

Run: `python -m pytest test_extract_snippet_masks.py::test_cli_help test_extract_snippet_masks.py::test_cli_missing_args -v -x`
Expected: PASS

### Step 5: Commit

```bash
git add extract_snippet_masks.py test_extract_snippet_masks.py
git commit -m "feat(extract-masks): add CLI entry point with argparse"
```

---

## Task 6: Run all tests, final verification

### Step 1: Run the full test suite

Run: `cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video && python -m pytest test_extract_snippet_masks.py -v --tb=short`
Expected: All tests PASS (approximately 9 tests)

### Step 2: Verify CLI usage string

Run: `python extract_snippet_masks.py --help`
Expected: Clean usage output showing all flags

### Step 3: Run existing test suite to verify no regressions

Run: `python -m pytest test_tracking_fixes.py -v --tb=short`
Expected: All 25 existing tests PASS

### Step 4: Final commit

```bash
git add -A
git commit -m "feat: add extract_snippet_masks.py — SAM3 instance segmentation from snippet videos

Parallel to overlay_snippet_bboxes.sh: takes an MP4 + per-frame bbox JSON and
produces binary mask PNGs at original video resolution using Sam3Model with
combined text + box prompts. Optional mask video and overlay video output."
```

---

## Summary of Deliverables

| File | Purpose |
|------|---------|
| `extract_snippet_masks.py` | CLI tool: MP4 + bbox JSON → per-frame mask PNGs via SAM3 |
| `test_extract_snippet_masks.py` | ~9 tests covering JSON loading, coord conversion, mask extraction, full pipeline, video encoding, CLI |

## Architecture Diagram

```
                        ┌──────────────────────┐
                        │  snippet_001.mp4     │
                        │  (e.g. 1920x1080)    │
                        └──────────┬───────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
  ┌──────────┐            ┌──────────────┐           ┌──────────────┐
  │ Frame 1  │            │   Frame 2    │           │   Frame N    │
  │ (PIL)    │            │   (PIL)      │           │   (PIL)      │
  └────┬─────┘            └──────┬───────┘           └──────┬───────┘
       │                         │                          │
       ▼                         ▼                          ▼
  ┌──────────┐            ┌──────────────┐           ┌──────────────┐
  │ bbox.json│            │  bbox.json   │           │  bbox.json   │
  │ entry    │            │  entry       │           │  entry       │
  │ pct→xyxy │            │  pct→xyxy    │           │  pct→xyxy    │
  └────┬─────┘            └──────┬───────┘           └──────┬───────┘
       │                         │                          │
       ▼                         ▼                          ▼
  ┌────────────────────────────────────────────────────────────────┐
  │              Sam3Model + Sam3Processor                        │
  │  processor(image, text="person", box=xyxy, label=1)           │
  │  model(**inputs)                                               │
  │  post_process_instance_segmentation(target_sizes=original)     │
  └──────────────────────────┬─────────────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
          mask_000001   mask_000002  mask_0000N
            .png          .png        .png
          (H×W, L)      (H×W, L)   (H×W, L)
          0=bg          0=bg        0=bg
          255=person    255=person  255=person
```
