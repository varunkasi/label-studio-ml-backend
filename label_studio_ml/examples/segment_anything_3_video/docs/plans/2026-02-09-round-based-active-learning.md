# Round-Based Active Learning Redesign (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the Interview UI detection/classification pipeline from a flat one-shot flow into a multi-round active learning loop where each round brings new temporally-stratified frames, MLP training happens only at round boundaries, and human effort decreases progressively across ~4 rounds. Seeding uses multi-prompt SAM3 detection on all target frames, gated by an MLP trained on mask-quality-augmented features.

**Architecture:** Rounds are the atomic unit of interaction. Each round: (1) selects new frames via temporal stratification + change-detection weighting, (2) runs SAM3 detection, (3) presents crops for human labeling (accept/reject/skip), (4) allows multi-prompt recall + manual drawing. Clicking "Next Round" trains the MLP on ALL accumulated labels (with LR decay), selects new frames, and repeats. After ~4 rounds, user advances to ReID, then Seeding runs automatically with multi-prompt SAM3 on every Nth frame + MLP quality gate.

**Tech Stack:** Python 3.12, Flask (routes), vanilla JS SPA (app.js/components.js), PyTorch (MLP), SAM3 via HuggingFace transformers, DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m), PyAV for video decoding.

---

## Key Design Decisions (from brainstorming)

### 1. No auto-scoring during active rounds
MLP trains ONLY at round boundaries ("Next Round" click). During a round, new crops from detection or recall get `uncertainty=0.5` (default). Auto-scoring with a stale model adds latency for dubious value. The MLP's role is to sort crops by uncertainty AFTER training, not during labeling.

### 2. MLP input augmented with mask-quality features
Current: `[DINOv3(1024) + spatial(4)]` = 1028 dims.
New: `[DINOv3(1024) + spatial(4) + mask_quality(4)]` = 1032 dims.

Mask-quality features (computed from SAM3 instance segmentation output):
- `mask_fill_ratio`: mask_area / box_area (tight-fit indicator)
- `detection_score`: SAM3's instance confidence
- `edge_contact`: fraction of box edges touching frame boundary (zoom/partial-in-frame indicator)
- `mask_compactness`: 4π × area / perimeter² (shape regularity)

These are modality-agnostic (work on RGB, grayscale IR, thermal).

### 3. Labeling guidance: reject vs skip
- **REJECT**: Box partially covers a fully-visible person (bad box quality). The MLP learns from this via mask-quality features (no edge contact + partial coverage pattern).
- **SKIP**: Genuinely ambiguous — can't tell if coverage is right. Excluded from training.
- **ACCEPT**: Tight full-body box, or tight box around visible portion of partially-in-frame person.

### 4. Seeding: multi-prompt SAM3 + MLP gate (no tracking, no grid search)
For each target frame (every Nth frame across 30K+ video):
1. Run SAM3 text detection with ALL accumulated prompts from rounds 1-4
2. NMS + pad → crop → DINOv3 features + mask-quality features → MLP quality gate
3. Assign identity via nearest ReID centroid
4. Accept if MLP confidence >= threshold

No grid search (removed). No tracking-primary (human seeds too few for dense coverage). SAM3 with multiple prompts is the candidate generator; MLP is the quality judge.

### 5. Grid search fully removed
- `run_feature_search()` from dinov3_classifier.py — deleted
- `_run_feature_search_strategy()` from detection.py — deleted
- `_grid_search_fallback()` from seeding_phase.py — deleted
- Path C logic from `generate_seeds()` — deleted
- `CropSource.FEATURE_SEARCH` enum — deleted
- All UI references — deleted
- All config constants (`_ENABLE_GRID_SEARCH`, etc.) — deleted

---

## Overview of Changes

| Area | What Changes |
|------|-------------|
| `state.py` | Add round tracking fields; add `mask_quality` to CropData; remove `CropSource.FEATURE_SEARCH` |
| `detection.py` | Add `select_round_frames()`, `run_round_detection()`; enhance `_detect_batch` to return mask-quality metrics; remove `_run_feature_search_strategy`; remove `feature_search` from `run_recall_strategy` |
| `dinov3_classifier.py` | Widen `CropClassifier` to 1032-dim input; add `_compute_mask_quality()`; catastrophic forgetting safeguards; remove `run_feature_search()` |
| `seeding_phase.py` | Rewrite `generate_seeds()` to multi-prompt SAM3 + MLP gate; remove `_grid_search_fallback`, Path C, grid config constants |
| `routes.py` | Add `/api/detect/next_round` endpoint; wire `detect_start` to `run_round_detection` round 1; remove `feature_search` strategy |
| `cache_manager.py` | Persist round fields + mask_quality in CropData |
| `app.js` | Round counter, "Next Round" button, remove `feature_search` from toolbar |
| `components.js` | Update toolbar for round-based UI; remove `feature_search` CSS conditional |
| `test_interview_detection.py` | Tests for all new functionality |

---

## Task 1: Add Round State + Mask Quality to State/Cache

**Files:**
- Modify: `interview/state.py:50-95` (CropData), `interview/state.py:119-163` (InterviewSession), `interview/state.py:40-48` (CropSource enum)
- Modify: `interview/cache_manager.py:125-153` (save_session), `interview/cache_manager.py:196-226` (load_session)

**Step 1: Write the failing test**

Add to `test_interview_detection.py`:

```python
def test_session_round_state():
    """Round tracking fields exist and default correctly."""
    from interview.state import InterviewSession, create_session
    session = create_session(project_id=1, task_id=1)
    assert session.current_round == 0
    assert session.round_history == []
    assert session.round_frames == {}


def test_crop_data_mask_quality():
    """CropData has mask_quality field (4-element array)."""
    from interview.state import CropData, CropSource
    import numpy as np
    mq = np.array([0.85, 0.92, 0.0, 0.78], dtype=np.float32)
    crop = CropData(
        crop_id="test", frame_idx=0,
        xyxy=np.array([0, 0, 100, 200], dtype=np.float32),
        score=0.9,
        mask_quality=mq,
    )
    assert crop.mask_quality is not None
    assert crop.mask_quality.shape == (4,)
    # Verify FEATURE_SEARCH is removed
    assert not hasattr(CropSource, "FEATURE_SEARCH")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_interview_detection.py::test_session_round_state test_interview_detection.py::test_crop_data_mask_quality -xvs`
Expected: FAIL with `AttributeError`

**Step 3: Implement changes**

In `interview/state.py`:

1. Remove `FEATURE_SEARCH` from `CropSource` enum (line 45).

2. Add `mask_quality` to `CropData` (after line 64, `metadata`):
```python
    mask_quality: Optional[np.ndarray] = None  # [fill_ratio, det_score, edge_contact, compactness] (4,)
```

3. Add round fields to `InterviewSession` (after line 148, `change_keyframes`):
```python
    # Round-based active learning
    current_round: int = 0
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    round_frames: Dict[int, List[int]] = field(default_factory=dict)
```

4. Update `stats()` to include:
```python
    "current_round": self.current_round,
    "rounds_completed": len(self.round_history),
```

In `interview/cache_manager.py`:

1. In `save_session` config dict, add after `change_keyframes`:
```python
    "current_round": session.current_round,
    "round_history": session.round_history,
    "round_frames": {str(k): v for k, v in session.round_frames.items()},
```

2. In `save_session` crops metadata serialization, add `mask_quality`:
```python
    "mask_quality": crop.mask_quality.tolist() if crop.mask_quality is not None else None,
```

3. In `load_session`, restore round fields:
```python
    current_round=config.get("current_round", 0),
    round_history=config.get("round_history", []),
```
And after constructor:
```python
    raw_rf = config.get("round_frames", {})
    session.round_frames = {int(k): v for k, v in raw_rf.items()}
```

4. In `load_session` crop deserialization, restore `mask_quality`:
```python
    mq = crop_meta.get("mask_quality")
    if mq is not None:
        crop.mask_quality = np.array(mq, dtype=np.float32)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_interview_detection.py::test_session_round_state test_interview_detection.py::test_crop_data_mask_quality -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add interview/state.py interview/cache_manager.py test_interview_detection.py
git commit -m "feat: add round tracking, mask_quality to CropData, remove FEATURE_SEARCH"
```

---

## Task 2: Implement Temporally-Stratified Frame Selector

**Files:**
- Modify: `interview/detection.py` (add `select_round_frames` after `uniform_indices`)

Divides video into N equal-width temporal bins. One frame per bin. Prefers change-detected frames within each bin. Excludes frames from all previous rounds.

**Step 1: Write the failing test**

```python
def test_select_round_frames_uniform_no_change():
    """Without change keyframes, frames are uniformly spaced."""
    from interview.state import create_session
    session = create_session(project_id=1, task_id=1)
    session.frames_count = 1000
    session.embedding_complete = False
    session.change_keyframes = []
    session.round_frames = {}

    from interview.detection import select_round_frames
    frames = select_round_frames(session, round_num=1, frames_per_round=40)
    assert len(frames) == 40
    assert frames == sorted(frames)
    assert frames[0] < 50
    assert frames[-1] > 950


def test_select_round_frames_excludes_previous():
    """Frames from previous rounds are excluded."""
    from interview.state import create_session
    session = create_session(project_id=1, task_id=1)
    session.frames_count = 100
    session.embedding_complete = False
    session.round_frames = {1: list(range(0, 100, 10))}

    from interview.detection import select_round_frames
    frames = select_round_frames(session, round_num=2, frames_per_round=10)
    assert len(frames) == 10
    used = set(session.round_frames[1])
    assert not (set(frames) & used)


def test_select_round_frames_prefers_change_keyframes():
    """When embeddings are ready, change keyframes are preferred within bins."""
    from interview.state import create_session
    session = create_session(project_id=1, task_id=1)
    session.frames_count = 1000
    session.embedding_complete = True
    session.change_keyframes = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
    session.round_frames = {}

    from interview.detection import select_round_frames
    frames = select_round_frames(session, round_num=1, frames_per_round=10)
    assert len(frames) == 10
    overlap = set(frames) & set(session.change_keyframes)
    assert len(overlap) >= 8
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_interview_detection.py -k "select_round_frames" -xvs`
Expected: FAIL with `ImportError: cannot import name 'select_round_frames'`

**Step 3: Implement select_round_frames**

Add to `interview/detection.py` after `uniform_indices` (after line 234):

```python
FRAMES_PER_ROUND = int(os.getenv("INTERVIEW_FRAMES_PER_ROUND", "40"))


def select_round_frames(
    session: InterviewSession,
    round_num: int,
    frames_per_round: int = FRAMES_PER_ROUND,
) -> List[int]:
    """Select frames for a round using temporal stratification + change weighting.

    Divides the video into ``frames_per_round`` equal-width temporal bins.
    Within each bin, selects the frame closest to a change-detected keyframe
    (if embeddings are ready) or the bin midpoint. Excludes frames used in
    all previous rounds.
    """
    total = session.frames_count
    if total <= 0:
        return []

    used: set = set()
    for rn, rf in session.round_frames.items():
        if rn < round_num:
            used.update(rf)

    available = [i for i in range(total) if i not in used]
    if not available:
        available = list(range(total))

    k = min(frames_per_round, len(available))
    if k <= 0:
        return []

    bin_width = total / k
    change_set = set(session.change_keyframes) if session.embedding_complete else set()

    selected: List[int] = []
    for i in range(k):
        bin_start = int(i * bin_width)
        bin_end = int((i + 1) * bin_width)
        candidates = [f for f in available if bin_start <= f < bin_end]

        if not candidates:
            mid = (bin_start + bin_end) // 2
            nearest = min(available, key=lambda f: abs(f - mid))
            if nearest not in selected:
                selected.append(nearest)
            continue

        change_in_bin = [f for f in candidates if f in change_set]
        mid = (bin_start + bin_end) // 2
        if change_in_bin:
            selected.append(min(change_in_bin, key=lambda f: abs(f - mid)))
        else:
            selected.append(min(candidates, key=lambda f: abs(f - mid)))

    return sorted(set(selected))
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_interview_detection.py -k "select_round_frames" -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add interview/detection.py test_interview_detection.py
git commit -m "feat: add temporally-stratified frame selector for round-based learning"
```

---

## Task 3: Add Mask-Quality Computation to Detection Pipeline

**Files:**
- Modify: `interview/detection.py:575-651` (Sam3TextBasedDetector._parse_results)
- Modify: `interview/detection.py:330-438` (_detect_batch)
- Add: `interview/dinov3_classifier.py` (add `compute_mask_quality` helper)

Currently `_parse_results` extracts boxes from masks but discards the mask data. We need to preserve mask metrics for the MLP input.

**Step 1: Write the failing test**

```python
def test_compute_mask_quality():
    """Mask quality features are computed from mask + box."""
    from interview.dinov3_classifier import compute_mask_quality
    import numpy as np

    # Create a 100x200 mask that fills most of the box
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:300, 50:150] = True  # person-shaped mask
    box = np.array([50, 100, 150, 300], dtype=np.float32)  # tight box
    frame_h, frame_w = 480, 640

    mq = compute_mask_quality(mask, box, frame_w, frame_h)
    assert mq.shape == (4,)
    # mask_fill_ratio: mask fills the box well
    assert mq[0] > 0.8
    # detection_score: passed as 0.9
    # edge_contact: box doesn't touch frame edges
    assert mq[2] < 0.1
    # mask_compactness: rectangular shape, moderate compactness
    assert 0.0 < mq[3] <= 1.0


def test_compute_mask_quality_edge_contact():
    """Edge contact is high when box touches frame boundary."""
    from interview.dinov3_classifier import compute_mask_quality
    import numpy as np

    mask = np.zeros((480, 640), dtype=bool)
    mask[380:480, 200:400] = True  # person at bottom edge
    box = np.array([200, 380, 400, 480], dtype=np.float32)

    mq = compute_mask_quality(mask, box, 640, 480)
    # Bottom edge contact should be detected
    assert mq[2] > 0.2
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_interview_detection.py -k "compute_mask_quality" -xvs`
Expected: FAIL with `ImportError`

**Step 3: Implement compute_mask_quality**

Add to `interview/dinov3_classifier.py` after `compute_crop_metadata` (after line ~111):

```python
def compute_mask_quality(
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Compute mask-quality features for MLP input.

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
    box_area = max(1.0, (x2 - x1) * (y2 - y1))

    # Mask fill ratio: how much of the box is filled by the mask
    mask_region = mask[int(max(0, y1)):int(min(frame_height, y2)),
                       int(max(0, x1)):int(min(frame_width, x2))]
    mask_area = float(np.count_nonzero(mask_region))
    fill_ratio = mask_area / box_area

    # Edge contact: fraction of box edges that touch frame boundary
    # (0 = fully interior, 1 = all edges on boundary)
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

    # Mask compactness: 4π × area / perimeter² (circle = 1.0)
    ys, xs = np.where(mask > 0)
    if xs.size < 4:
        compactness = 0.0
    else:
        # Approximate perimeter from mask boundary
        # Use erosion-based approach: perimeter ≈ mask_pixels - eroded_mask_pixels
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        perimeter = float(np.count_nonzero(mask) - np.count_nonzero(eroded))
        perimeter = max(1.0, perimeter)
        total_mask_area = float(np.count_nonzero(mask))
        compactness = min(1.0, (4.0 * np.pi * total_mask_area) / (perimeter * perimeter))

    return np.array([fill_ratio, 0.0, edge_contact, compactness], dtype=np.float32)
```

**Step 4: Modify _detect_batch to store mask_quality on CropData**

In `interview/detection.py`, modify `_parse_results` to also return masks:

```python
    # In _parse_results, add mask to each detection dict:
    if i < len(masks):
        m = masks[i]
        if hasattr(m, "cpu"):
            m = m.cpu().numpy()
        elif not isinstance(m, np.ndarray):
            m = np.asarray(m)
        det_dict["mask"] = m  # (H, W) binary
```

In `_detect_batch`, after creating CropData (line ~427-436), compute and store mask_quality:

```python
    from .dinov3_classifier import compute_mask_quality
    # ... inside the per-detection loop:
    mask = dets[keep_idx[j]].get("mask")
    if mask is not None:
        mq = compute_mask_quality(mask, boxes[j], width, height)
        mq[1] = float(scores[j])  # overwrite detection_score placeholder
        crop.mask_quality = mq
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest test_interview_detection.py -k "compute_mask_quality" -xvs`
Expected: PASS

**Step 6: Commit**

```bash
git add interview/detection.py interview/dinov3_classifier.py test_interview_detection.py
git commit -m "feat: add mask-quality features to detection pipeline for MLP input"
```

---

## Task 4: Widen MLP to 1032-dim Input + Catastrophic Forgetting Safeguards

**Files:**
- Modify: `interview/dinov3_classifier.py:118-135` (CropClassifier)
- Modify: `interview/dinov3_classifier.py:260-268` (_build_feature_matrix)
- Modify: `interview/dinov3_classifier.py:293-400` (train_classifier)

**Step 1: Write the failing test**

```python
def test_crop_classifier_1032_input():
    """CropClassifier accepts 1032-dim input (DINOv3 + spatial + mask_quality)."""
    import torch
    from interview.dinov3_classifier import CropClassifier
    model = CropClassifier(input_dim=1032)
    x = torch.randn(4, 1032)
    out = model(x)
    assert out.shape == (4, 1)


def test_build_feature_matrix_includes_mask_quality():
    """Feature matrix includes mask_quality when available."""
    from interview.dinov3_classifier import _build_feature_matrix
    from interview.state import create_session, CropData
    import numpy as np

    session = create_session(1, 1)
    crop = CropData(
        crop_id="test", frame_idx=0,
        xyxy=np.array([0, 0, 100, 200], dtype=np.float32),
        score=0.9,
        features=np.random.randn(1024).astype(np.float32),
        metadata=np.array([0.5, 0.5, 0.1, 0.5], dtype=np.float32),
        mask_quality=np.array([0.85, 0.9, 0.0, 0.78], dtype=np.float32),
    )
    session.add_crop(crop)

    X = _build_feature_matrix(session, ["test"])
    assert X.shape == (1, 1032)  # 1024 + 4 + 4


def test_train_classifier_lr_decay():
    """Learning rate decreases with round number."""
    from interview.dinov3_classifier import _compute_lr
    assert _compute_lr(base_lr=1e-3, round_num=1) == 1e-3
    assert abs(_compute_lr(base_lr=1e-3, round_num=2) - 7e-4) < 1e-6
    assert abs(_compute_lr(base_lr=1e-3, round_num=4) - 1e-3 * 0.7**3) < 1e-6
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_interview_detection.py -k "crop_classifier_1032 or build_feature_matrix_includes or lr_decay" -xvs`
Expected: FAIL

**Step 3: Implement changes**

1. Change `CropClassifier` default `input_dim` from 1028 to 1032:
```python
    def __init__(self, input_dim: int = 1032, hidden_dim: int = 256, dropout: float = 0.3):
```

2. Update `_build_feature_matrix` to concatenate mask_quality:
```python
def _build_feature_matrix(session: InterviewSession, crop_ids: List[str]) -> torch.Tensor:
    rows = []
    for cid in crop_ids:
        crop = session.get_crop(cid)
        if crop is None or crop.features is None or crop.metadata is None:
            continue
        mq = crop.mask_quality if crop.mask_quality is not None else np.zeros(4, dtype=np.float32)
        row = np.concatenate([crop.features, crop.metadata, mq])
        rows.append(row)
    if not rows:
        return torch.empty(0, 1032)
    return torch.from_numpy(np.stack(rows)).float()
```

3. Add LR decay + validation holdout to `train_classifier`:
```python
LR_DECAY_FACTOR = float(os.getenv("INTERVIEW_LR_DECAY", "0.7"))
VALIDATION_FRAC = float(os.getenv("INTERVIEW_VAL_FRAC", "0.15"))
EARLY_STOP_PATIENCE = int(os.getenv("INTERVIEW_EARLY_STOP_PATIENCE", "5"))


def _compute_lr(base_lr: float, round_num: int) -> float:
    """Compute learning rate with exponential decay per round."""
    return base_lr * (LR_DECAY_FACTOR ** max(0, round_num - 1))
```

4. Modify `train_classifier` signature to accept `round_num`:
```python
def train_classifier(
    session: InterviewSession, progress: JobProgress,
    round_num: int = 1,
) -> Dict[str, Any]:
```

Key changes inside:
- `lr=_compute_lr(1e-3, round_num)` instead of `lr=1e-3`
- 15% validation holdout with early stopping (patience=5)
- `CropClassifier(input_dim=1032)` instead of `CropClassifier()`
- Include `round_num` in result dict

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_interview_detection.py -k "crop_classifier_1032 or build_feature_matrix_includes or lr_decay" -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add interview/dinov3_classifier.py test_interview_detection.py
git commit -m "feat: widen MLP to 1032-dim (mask-quality), add LR decay + validation holdout"
```

---

## Task 5: Remove Grid Search Everywhere

**Files:**
- Modify: `interview/detection.py:1187-1268` (remove `_run_feature_search_strategy` + strategy dispatch)
- Modify: `interview/dinov3_classifier.py:407-506` (remove `run_feature_search`)
- Modify: `interview/dinov3_classifier.py:1-6` (update module docstring)
- Modify: `interview/seeding_phase.py:132-142` (remove grid config constants)
- Modify: `interview/seeding_phase.py:262-323` (remove `_grid_search_fallback`)
- Modify: `interview/static/app.js:663` (remove `feature_search` from recallStrategies)
- Modify: `interview/static/app.js:917-970` (remove feature_search handling from `_onRecall`)
- Modify: `interview/static/components.js:520` (remove `feature_search` CSS conditional)
- Modify: `interview/static/components.js:808` (remove `feature_search` from default strategies)
- Modify: `CLAUDE.md:41` (remove "3-scale grid search" reference)

**Step 1: Write the failing test**

```python
def test_recall_strategy_rejects_feature_search():
    """feature_search strategy is no longer supported."""
    from interview.detection import run_recall_strategy
    from interview.state import create_session
    session = create_session(1, 1)
    session.sampled_frames = [0, 100]

    mock_progress = type("P", (), {"step": "", "current": 0, "total": 0})()

    with pytest.raises(ValueError, match="Unknown recall strategy"):
        run_recall_strategy(session, "feature_search", [], mock_progress)
```

**Step 2: Run test to verify it fails (currently feature_search IS supported)**

Run: `python -m pytest test_interview_detection.py::test_recall_strategy_rejects_feature_search -xvs`
Expected: FAIL (no ValueError raised)

**Step 3: Remove grid search from all files**

In `interview/detection.py`:
- Delete `_run_feature_search_strategy` function (lines ~1187-1214)
- In `run_recall_strategy`, remove `elif strategy == "feature_search"` branch
- Update error message and docstring to only mention `multi_prompt`

In `interview/dinov3_classifier.py`:
- Delete `run_feature_search` function (lines ~407-506)
- Update module docstring (line 1-6): remove "dense spatial grid feature search for Strategy B discovery"

In `interview/seeding_phase.py`:
- Delete `_ENABLE_GRID_SEARCH`, `_GRID_SCALE`, `_GRID_SIM_THRESHOLD`, `_GRID_TOP_K` constants (lines 138-141)
- Delete `_grid_search_fallback` function (lines 265-323)
- Delete the `# DINOv3 grid search fallback (Path C)` section header (lines 261-263)
- Note: Path C logic in `generate_seeds()` will be removed in Task 8 when we rewrite that function entirely

In `interview/static/app.js`:
- Line 663: change `recallStrategies: ['multi_prompt', 'feature_search']` to `recallStrategies: ['multi_prompt']`
- In `_onRecall`: remove the `else if (strategy === 'feature_search')` branch

In `interview/static/components.js`:
- Line 520: remove `if (crop.source === 'feature_search') card.classList.add('feature');`
- Line 808: change default `['multi_prompt', 'feature_search']` to `['multi_prompt']`

In `CLAUDE.md`:
- Line 41: change "3-scale grid search" to "MLP quality-gate classifier"

**Step 4: Run tests**

Run: `python -m pytest test_interview_detection.py -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add interview/detection.py interview/dinov3_classifier.py interview/seeding_phase.py interview/static/app.js interview/static/components.js CLAUDE.md test_interview_detection.py
git commit -m "feat: remove grid search (feature_search / Strategy B / Path C) everywhere"
```

---

## Task 6: Implement run_round_detection (No Auto-Scoring)

**Files:**
- Modify: `interview/detection.py` (add `run_round_detection` function)

This function runs detection on a new set of frames for a given round. NO auto-scoring — MLP trains only at round boundaries.

**Step 1: Write the failing test**

```python
def test_run_round_detection_round1():
    """Round 1 detection produces crops and records round state."""
    from interview.state import create_session, Phase
    session = create_session(project_id=1, task_id=1)
    session.frames_count = 1000
    session.width = 640
    session.height = 480
    session.video_path = "/fake/video.mp4"
    session.round_frames = {}
    session.current_round = 0

    from interview.detection import run_round_detection

    mock_progress = type("P", (), {"step": "", "current": 0, "total": 0})()

    with (
        mock.patch("interview.detection._decode_frames_sequential") as mock_decode,
        mock.patch("interview.detection.Sam3TextBasedDetector") as MockDetector,
        mock.patch("interview.detection.select_round_frames") as mock_select,
    ):
        mock_select.return_value = [0, 250, 500, 750]
        mock_decode.return_value = {
            0: _mock_pil_image.new("RGB", (640, 480)),
            250: _mock_pil_image.new("RGB", (640, 480)),
            500: _mock_pil_image.new("RGB", (640, 480)),
            750: _mock_pil_image.new("RGB", (640, 480)),
        }
        detector_instance = MockDetector.return_value
        detector_instance.detect.return_value = [
            {"xyxy": np.array([10, 20, 100, 200]), "score": 0.9, "label": "person"}
        ]
        detector_instance.processor = mock.MagicMock()
        detector_instance.model = mock.MagicMock()
        detector_instance.threshold = 0.3
        detector_instance.mask_threshold = 0.5
        detector_instance.clear_cache = mock.MagicMock()

        result = run_round_detection(session, "person", mock_progress, round_num=1)

    assert result["round"] == 1
    assert result["total_crops"] > 0
    assert session.current_round == 1
    assert 1 in session.round_frames
    assert len(session.round_frames[1]) == 4
    # No auto_scored key — MLP trains only at round boundaries
    assert "auto_scored" not in result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_interview_detection.py::test_run_round_detection_round1 -xvs`
Expected: FAIL with `ImportError`

**Step 3: Implement run_round_detection**

Add to `interview/detection.py` after `run_detection_stage1`:

```python
def run_round_detection(
    session: InterviewSession,
    prompt: str,
    progress: Any,
    round_num: int = 1,
) -> Dict[str, Any]:
    """Run detection for one active learning round.

    1. Select frames via temporal stratification (excludes prior rounds)
    2. Batch-decode and batch-detect
    3. Store crops on session (NO auto-scoring — MLP trains at round boundary)
    4. Record round state
    """
    import time as _time
    t0 = _time.time()

    if prompt not in session.prompts:
        session.prompts.append(prompt)

    # Step 1: Select frames
    progress.step = f"Round {round_num}: Selecting frames..."
    progress.total = 3
    progress.current = 0

    frame_indices = select_round_frames(session, round_num)
    if not frame_indices:
        raise RuntimeError(f"No frames available for round {round_num}")
    progress.current = 1

    # Step 2: Decode frames (single sequential PyAV pass)
    progress.step = f"Round {round_num}: Decoding {len(frame_indices)} frames..."
    frame_images = _decode_frames_sequential(session.video_path, frame_indices)
    if not frame_images:
        raise RuntimeError(f"Failed to decode frames for round {round_num}")
    progress.current = 2

    # Step 3: Batch detect
    progress.step = f"Round {round_num}: Detecting on {len(frame_images)} frames..."
    detector = Sam3TextBasedDetector()
    crops = _detect_batch(
        detector, frame_images, prompt,
        session.width, session.height,
        batch_size=DEFAULT_DETECT_BATCH,
    )
    detector.clear_cache()
    progress.current = 3

    # Step 4: Store crops (no auto-scoring)
    total_crops = 0
    with session._lock:
        session.round_frames[round_num] = sorted(frame_images.keys())
        session.sampled_frames = sorted(
            set(session.sampled_frames) | set(frame_images.keys())
        )
        session.current_round = round_num

        for crop in crops:
            session.add_crop(crop)
            total_crops += 1

        if round_num == 1:
            session.advance_to(Phase.DETECTION)

    save_session(session)

    elapsed = _time.time() - t0
    progress.step = f"Round {round_num} complete."

    round_info = {
        "round": round_num,
        "frames": len(frame_indices),
        "new_crops": total_crops,
        "elapsed_seconds": round(elapsed, 1),
    }
    session.round_history.append(round_info)
    save_session(session)

    summary = {
        "round": round_num,
        "keyframes": len(frame_indices),
        "total_crops": total_crops,
        "prompt": prompt,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info("Round %d: %d crops on %d frames in %.1fs",
                round_num, total_crops, len(frame_indices), elapsed)
    return summary
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_interview_detection.py::test_run_round_detection_round1 -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add interview/detection.py test_interview_detection.py
git commit -m "feat: add run_round_detection for round-based active learning"
```

---

## Task 7: Add /api/detect/next_round Endpoint

**Files:**
- Modify: `interview/routes.py` (add endpoint, modify `detect_start` to use round 1)

**Step 1: Implement the endpoints**

Modify `detect_start` to dispatch `run_round_detection` with `round_num=1`:

```python
@interview_bp.route("/api/detect/start", methods=["POST"])
def detect_start():
    """Start Round 1: detection on stratified frames + background embedding."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    prompt = data.get("prompt", "person")

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _detect_round1(progress):
        from .detection import run_round_detection
        return run_round_detection(session, prompt, progress, round_num=1)

    detect_job_id = submit_job(_detect_round1, name="round_1_detection")

    def _embed_bg(progress):
        from .detection import run_embedding_background
        return run_embedding_background(session, progress)

    embed_job_id = submit_job(_embed_bg, name="embedding_background")

    with session._lock:
        session.embedding_job_id = embed_job_id
        session.touch()

    return jsonify({
        "job_id": detect_job_id,
        "embedding_job_id": embed_job_id,
        "round": 1,
    }), 202
```

Add the new endpoint:

```python
@interview_bp.route("/api/detect/next_round", methods=["POST"])
def detect_next_round():
    """Train MLP on all labels, then start next round of detection.

    Two-phase job:
      1. Train MLP on ALL accumulated labels (with LR decay)
      2. Select new frames, detect on them
    """
    data = request.get_json(force=True)
    session_id = data["session_id"]

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    next_round = session.current_round + 1
    prompt = session.prompts[0] if session.prompts else "person"

    def _next_round(progress):
        from .dinov3_classifier import train_classifier
        from .detection import run_round_detection

        progress.step = f"Training MLP before round {next_round}..."
        train_result = train_classifier(session, progress, round_num=next_round)

        progress.step = f"Starting round {next_round} detection..."
        detect_result = run_round_detection(
            session, prompt, progress, round_num=next_round,
        )

        return {
            "training": train_result,
            "detection": detect_result,
            "round": next_round,
        }

    job_id = submit_job(_next_round, name=f"round_{next_round}")
    return jsonify({"job_id": job_id, "round": next_round}), 202
```

**Step 2: Run all tests**

Run: `python -m pytest test_interview_detection.py -x -q`
Expected: All PASS

**Step 3: Commit**

```bash
git add interview/routes.py
git commit -m "feat: add /api/detect/next_round endpoint for round-based active learning"
```

---

## Task 8: Rewrite Seeding Phase — Multi-Prompt SAM3 + MLP Gate

**Files:**
- Modify: `interview/seeding_phase.py` (rewrite `generate_seeds`, remove Path C remnants)

**Goal:** Replace dual-proposer pipeline with: for every Nth frame, run SAM3 with ALL accumulated prompts from rounds 1-4, apply MLP quality gate (1032-dim input including mask-quality features), assign identity via nearest centroid.

**Step 1: Write the failing test**

```python
def test_generate_seeds_multi_prompt():
    """Seeding uses all accumulated prompts for SAM3 detection."""
    from interview.state import create_session, CropData, CropLabel, Phase
    session = create_session(1, 1)
    session.frames_count = 100
    session.width = 640
    session.height = 480
    session.video_path = "/fake.mp4"
    session.prompts = ["person", "human figure", "pedestrian"]
    session.phase = Phase.REID
    session.seed_config.frame_interval = 5
    session.model_trained = True

    # Add accepted crops with features
    for i in range(3):
        crop = CropData(
            crop_id=f"crop_{i}",
            frame_idx=i * 20,
            xyxy=np.array([100, 100, 200, 300]),
            score=0.95,
            label=CropLabel.ACCEPTED,
            features=np.random.randn(1024).astype(np.float32),
        )
        crop.reid_cluster_id = 0
        session.add_crop(crop)

    session.reid_clusters = {0: ["crop_0", "crop_1", "crop_2"]}
    mock_progress = type("P", (), {"step": "", "current": 0, "total": 0})()

    with (
        mock.patch("interview.seeding_phase.load_model") as mock_load,
        mock.patch("interview.seeding_phase._decode_frames_sequential") as mock_decode,
        mock.patch("interview.seeding_phase.Sam3TextBasedDetector") as MockDet,
        mock.patch("interview.seeding_phase.extract_features") as mock_feat,
        mock.patch("interview.seeding_phase.compute_mask_quality") as mock_mq,
    ):
        mock_load.return_value = {"fc1.weight": None}  # pretend model exists
        mock_decode.return_value = {i: _mock_pil_image.new("RGB", (640, 480)) for i in range(0, 100, 5)}

        det_instance = MockDet.return_value
        det_instance.detect.return_value = [
            {"xyxy": np.array([100, 100, 200, 300]), "score": 0.9, "label": "person", "mask": np.ones((480, 640), dtype=bool)}
        ]
        det_instance.set_frame = mock.MagicMock()
        det_instance.clear_cache = mock.MagicMock()

        mock_feat.return_value = np.random.randn(1, 1024).astype(np.float32)
        mock_mq.return_value = np.array([0.9, 0.9, 0.0, 0.8], dtype=np.float32)

        # Mock the classifier forward pass
        with mock.patch("interview.seeding_phase.CropClassifier") as MockCls:
            import torch
            cls_instance = MockCls.return_value
            cls_instance.eval = mock.MagicMock()
            cls_instance.load_state_dict = mock.MagicMock()
            cls_instance.return_value = torch.tensor([[2.0]])  # high confidence
            cls_instance.to = mock.MagicMock(return_value=cls_instance)

            from interview.seeding_phase import generate_seeds
            result = generate_seeds(session, mock_progress)

    assert result["total_seeds"] > 0
    # All 3 prompts should have been tried
    all_detect_calls = det_instance.detect.call_args_list
    prompts_used = {call[0][0] for call in all_detect_calls}
    assert "person" in prompts_used
    assert "human figure" in prompts_used
    assert "pedestrian" in prompts_used
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_interview_detection.py::test_generate_seeds_multi_prompt -xvs`
Expected: FAIL

**Step 3: Rewrite generate_seeds**

Replace the entire `generate_seeds` function in `interview/seeding_phase.py`:

```python
def generate_seeds(
    session: InterviewSession,
    progress: JobProgress,
) -> Dict[str, Any]:
    """Generate dense seeds: multi-prompt SAM3 detection + MLP quality gate.

    For every Nth frame across the entire video:
      1. Run SAM3 text detection with ALL prompts accumulated during rounds 1-4
      2. NMS + pad boxes
      3. For each candidate: extract DINOv3 features + compute mask quality
      4. MLP quality gate (1032-dim: DINOv3 + spatial + mask_quality)
      5. Assign identity via nearest ReID centroid
      6. Accept if MLP confidence >= threshold

    This is fully automatic — no human interaction needed for 30K+ frames.
    """
    from .detection import (
        Sam3TextBasedDetector, nms_numpy, pad_boxes,
        _decode_frames_sequential,
    )
    from .dinov3_classifier import (
        CropClassifier, compute_crop_metadata, compute_mask_quality,
    )

    import torch

    # ---- Validate prerequisites ----
    progress.step = "Validating session state..."
    progress.current = 0

    state_dict = load_model(session.cache_key)
    if state_dict is None:
        raise RuntimeError(
            "No trained MLP model found. Complete the classification phase first."
        )
    if not session.reid_clusters:
        raise RuntimeError(
            "No ReID clusters found. Complete the ReID phase first."
        )

    # ---- Load models ----
    progress.step = "Loading models..."
    detector = Sam3TextBasedDetector()

    classifier = CropClassifier(input_dim=1032)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    device = torch.device(DEVICE)
    classifier = classifier.to(device)

    centroids = _compute_cluster_centroids(session)
    prompts = session.prompts if session.prompts else ["person"]

    # ---- Determine target frames ----
    interval = max(1, session.seed_config.frame_interval)
    uniform = set(range(0, session.frames_count, interval))
    change = set(session.change_keyframes) if session.embedding_complete else set()
    all_targets = sorted(uniform | change)
    total_frames = len(all_targets)

    progress.step = "Generating seeds..."
    progress.total = total_frames
    progress.current = 0

    threshold = session.seed_config.confidence_threshold
    seeds: List[Dict[str, Any]] = []

    logger.info(
        "Seed generation: scanning %d frames (interval=%d, threshold=%.2f, "
        "prompts=%s, refinement=%s)",
        total_frames, interval, threshold,
        prompts, _ENABLE_REFINEMENT,
    )

    # ---- Process in chunks ----
    for chunk_start in range(0, total_frames, _SEED_CHUNK_SIZE):
        chunk_indices = all_targets[chunk_start:chunk_start + _SEED_CHUNK_SIZE]
        progress.step = (
            f"Decoding frames {chunk_start + 1}"
            f"-{chunk_start + len(chunk_indices)} / {total_frames}..."
        )

        frames = _decode_frames_sequential(session.video_path, chunk_indices)
        medium_candidates: List[Tuple[int, np.ndarray, float]] = []

        for fi, frame_idx in enumerate(chunk_indices):
            progress.current = chunk_start + fi + 1
            pil_frame = frames.get(frame_idx)
            if pil_frame is None:
                continue

            # Try ALL prompts for maximum recall
            all_detections = []
            detector.set_frame(pil_frame)
            for prompt_text in prompts:
                dets = detector.detect(prompt_text)
                all_detections.extend(dets)

            if not all_detections:
                continue

            boxes = np.array([d["xyxy"] for d in all_detections], dtype=np.float32)
            det_scores = np.array([d["score"] for d in all_detections], dtype=np.float32)
            masks = [d.get("mask") for d in all_detections]
            boxes = pad_boxes(boxes, pil_frame.width, pil_frame.height)
            keep = nms_numpy(boxes, det_scores, iou_threshold=0.5)
            boxes = boxes[keep]
            det_scores = det_scores[keep]
            masks = [masks[k] for k in keep]

            if len(boxes) == 0:
                continue

            # Crop + DINOv3 features + mask quality + MLP
            crop_images = []
            valid_indices = []
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_frame.width, x2), min(pil_frame.height, y2)
                if x2 > x1 and y2 > y1:
                    crop_images.append(pil_frame.crop((x1, y1, x2, y2)))
                    valid_indices.append(idx)

            if not crop_images:
                continue

            boxes = boxes[valid_indices]
            det_scores = det_scores[valid_indices]
            masks = [masks[vi] for vi in valid_indices]

            crop_features = extract_features(crop_images)
            metadata = np.array([
                compute_crop_metadata(b, pil_frame.width, pil_frame.height)
                for b in boxes
            ], dtype=np.float32)

            # Compute mask quality features
            mask_qualities = []
            for idx, box in enumerate(boxes):
                m = masks[idx]
                if m is not None:
                    mq = compute_mask_quality(m, box, pil_frame.width, pil_frame.height)
                    mq[1] = float(det_scores[idx])  # fill in detection score
                else:
                    mq = np.array([0.0, float(det_scores[idx]), 0.0, 0.0], dtype=np.float32)
                mask_qualities.append(mq)
            mask_quality_arr = np.array(mask_qualities, dtype=np.float32)

            # Build 1032-dim MLP input
            mlp_input = np.concatenate([crop_features, metadata, mask_quality_arr], axis=1)

            with torch.inference_mode():
                probs = torch.sigmoid(
                    classifier(torch.from_numpy(mlp_input).float().to(device))
                ).squeeze(-1).cpu().numpy()

            for i in range(len(boxes)):
                conf = float(probs[i]) if probs.ndim > 0 else float(probs)
                if conf >= threshold:
                    identity, identity_sim = _assign_identity(
                        crop_features[i], centroids,
                    )
                    seeds.append({
                        "frame_idx": int(frame_idx),
                        "xyxy": boxes[i].tolist(),
                        "confidence": round(conf, 4),
                        "identity": int(identity),
                        "identity_similarity": round(float(identity_sim), 4),
                        "source": "multi_prompt_mlp",
                    })
                elif _ENABLE_REFINEMENT and conf >= _REFINE_THRESHOLD:
                    medium_candidates.append((frame_idx, boxes[i], det_scores[i]))

        # Path B: Refine medium-confidence candidates (unchanged)
        if medium_candidates and _ENABLE_REFINEMENT:
            progress.step = f"Refining {len(medium_candidates)} candidates..."
            refined = _refine_candidates_sam3(
                frames, medium_candidates, prompt=prompts[0],
            )
            for frame_idx, box, _det_score in refined:
                pil_frame = frames.get(frame_idx)
                if pil_frame is None:
                    continue
                seed = _score_and_accept_seed(
                    box, pil_frame, classifier, centroids,
                    threshold, "refined", frame_idx,
                )
                if seed is not None:
                    seeds.append(seed)

    # ---- Finalise ----
    with session._lock:
        session.seeds = seeds
        session.advance_to(Phase.SEEDING)

    save_session(session)

    identity_counts: Dict[int, int] = {}
    for seed in seeds:
        ident = seed["identity"]
        identity_counts[ident] = identity_counts.get(ident, 0) + 1

    summary = {
        "total_seeds": len(seeds),
        "frames_scanned": total_frames,
        "identities": identity_counts,
        "prompts_used": prompts,
    }

    logger.info(
        "Seed generation complete: %d seeds across %d frames, %d identities",
        len(seeds), total_frames, len(identity_counts),
    )
    return summary
```

Note: `_score_and_accept_seed` needs to be updated to use 1032-dim input (mask_quality). Since the refined candidates come from `_refine_candidates_sam3` which returns refined boxes but not masks, we pass zero mask_quality for refined seeds (the refinement itself is the quality signal).

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_interview_detection.py::test_generate_seeds_multi_prompt -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add interview/seeding_phase.py test_interview_detection.py
git commit -m "feat: rewrite seeding to multi-prompt SAM3 + MLP quality gate (1032-dim)"
```

---

## Task 9: Update Frontend for Round-Based Workflow

**Files:**
- Modify: `interview/static/app.js`
- Modify: `interview/static/components.js`

**Changes:**

1. Add `_onNextRound` handler — calls `/api/detect/next_round`, polls for completion, reloads crops

2. Add round counter to toolbar — "Round N"

3. "Next Round" is primary button, "Advance to ReID" appears after round 2+

4. Remove standalone "Train Classifier" button (training happens automatically at round boundaries via "Next Round")

In `app.js`, add handler:
```javascript
async function _onNextRound() {
    const progress = AppState._components.progressOverlay;
    try {
        progress.show('Training MLP and preparing next round...', -1);
        const job = await API.post('/detect/next_round', {
            session_id: AppState.sessionId,
        });
        pollJob(
            job.job_id,
            (p) => progress.show(p.step || 'Processing...', p.percent || -1),
            async (p) => {
                progress.hide();
                if (p.status === 'completed') {
                    showToast(`Round ${job.round} ready`, 'success');
                    await _refreshCrops();
                } else {
                    showToast(`Round failed: ${p.error}`, 'error');
                }
            }
        );
    } catch (err) {
        progress.hide();
    }
}
```

In `_renderToolbar`, pass round info:
```javascript
onNextRound: _onNextRound,
currentRound: AppState.stats.current_round || 1,
roundsCompleted: AppState.stats.rounds_completed || 0,
```

In `components.js` Toolbar, add round badge and "Next Round" button:
```javascript
const roundBadge = document.createElement('span');
roundBadge.className = 'round-badge';
roundBadge.textContent = `Round ${opts.currentRound || 1}`;
toolbarRow.appendChild(roundBadge);

const nextRoundBtn = document.createElement('button');
nextRoundBtn.className = 'btn btn-primary';
nextRoundBtn.textContent = 'Next Round';
nextRoundBtn.title = 'Train MLP on all labels, then detect on new frames';
nextRoundBtn.addEventListener('click', () => opts.onNextRound?.());
toolbarRow.appendChild(nextRoundBtn);
```

**Commit**

```bash
git add interview/static/app.js interview/static/components.js
git commit -m "feat: update UI for round-based active learning workflow"
```

---

## Task 10: Full Test Suite + Documentation

**Files:**
- Modify: `test_interview_detection.py` — run full suite
- Modify: `CLAUDE.md` — update architecture docs

**Step 1: Run full test suite**

```bash
python -m pytest test_interview_detection.py test_tracking_fixes.py -x -q
```

Expected: All tests pass, no mock leakage between test files.

**Step 2: Update CLAUDE.md**

Add to the Interview UI section:

```markdown
### Round-Based Active Learning
- Rounds driven by new frames arriving (user clicks "Next Round")
- Frame selection: temporally-stratified bins + change-detection-weighted
- MLP trains ONLY at round boundaries with LR decay (0.7^round)
- No auto-scoring during active rounds (MLP trains only at boundary)
- MLP input: [DINOv3(1024) + spatial(4) + mask_quality(4)] = 1032 dims
- Mask quality features: mask_fill_ratio, detection_score, edge_contact, mask_compactness
- Grid search (Strategy B) fully removed
- ~4 rounds cover full video with decreasing human effort
- Labeling: REJECT partial-coverage boxes, SKIP ambiguous, ACCEPT tight-fit
- Env vars: INTERVIEW_FRAMES_PER_ROUND (40), INTERVIEW_LR_DECAY (0.7),
  INTERVIEW_VAL_FRAC (0.15), INTERVIEW_EARLY_STOP_PATIENCE (5)

### Seeding Phase
- Fully automatic: multi-prompt SAM3 detection on every Nth frame
- Uses ALL prompts accumulated from rounds 1-4 for maximum recall
- MLP quality gate (1032-dim) filters bad boxes using mask-quality features
- Path B refinement (Sam3Model box+text) for medium-confidence detections
- Identity assignment via nearest ReID centroid
- Modality-agnostic: mask-quality features work on RGB, IR, thermal
```

**Step 3: Commit**

```bash
git add CLAUDE.md test_interview_detection.py
git commit -m "docs: update CLAUDE.md with round-based active learning + seeding architecture"
```

---

## Verification

1. `python -m pytest test_interview_detection.py test_tracking_fixes.py -x -q`
2. Docker build + deploy
3. Open Interview UI → setup → detect → verify crops appear (Round 1)
4. Label ~20 crops (accept/reject/skip) → click "Next Round" → verify:
   - MLP trains (toast shows accuracy)
   - New frames selected (different from round 1)
   - New crops appear
5. Repeat for rounds 3-4 → verify decreasing human effort
6. "Advance to ReID" → ReID → Seeding → verify:
   - All 3+ prompts used during seeding
   - Seeds generated across full video
   - MLP quality gate filtering works

## Performance Characteristics

| Metric | Round 1 | Round 2+ |
|--------|---------|----------|
| Frames selected | 40 uniform | 40 stratified + change-weighted |
| Detection time | ~30-60s (batched) | ~30-60s (batched) |
| MLP training | N/A | ~2s (warm-start, LR decay) |
| Human effort | ~30-40 labels | Decreasing (MLP pre-sorts after training) |
| Total rounds | 1 | 3-4 typical |

| Metric | Seeding |
|--------|---------|
| Frames scanned | All target frames (30K / interval) |
| Prompts per frame | All accumulated (typically 3-5) |
| Detection + MLP | ~2-3s/frame (multi-prompt × DINOv3 bottleneck) |
| Expected coverage | High (multi-prompt recall > single-prompt) |
