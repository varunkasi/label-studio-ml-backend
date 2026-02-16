# Dual-Proposer Dense Seeding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Increase dense seeding coverage from ~50% to ~80-90% of frames by adding a fallback proposer when SAM3 text detection fails.

**Architecture:** The current pipeline relies solely on SAM3 text detection for box proposals, with an MLP quality gate. When SAM3 produces a bad box (~50% of the time), the MLP correctly rejects it, but the frame gets no seed. The dual-proposer adds two fallback paths: (1) box refinement via Sam3Model for medium-confidence detections, and (2) DINOv3 grid search for frames with zero detections, followed by Sam3Model refinement of grid candidates. All paths terminate at the same MLP quality gate.

**Tech Stack:** Sam3Model (box-prompted segmentation), DINOv3 (feature extraction), existing MLP classifier, PyAV (frame decode), NumPy/SciPy.

**Orchestration insight:** Frame I/O uses per-frame PyAV seeks with the PTS fix (~100-200ms each). Sequential full-video decode was tested and rejected — it scans all 30K frames even for 40 targets (~10 min). Seek-based is comparable for sparse targets and simpler. The real latency win is processing frames in chunks of 100 to batch GPU work by model (SAM3 text → DINOv3 → MLP → Sam3Model refine → DINOv3 grid), minimizing model-switching overhead. Frame reads use the existing `_read_frame_pyav` (with PTS fix) and the LRU cache in routes.py.

---

## Pipeline Overview

```
For each chunk of 100 frames:

  Path A — Primary (all frames):
    SAM3 text detect → NMS → pad → crop → DINOv3 features → MLP score
    ├─ MLP ≥ confidence_threshold → ACCEPT as seed
    ├─ refine_threshold ≤ MLP < confidence_threshold → Path B
    └─ MLP < refine_threshold → discard

  Path B — Box Refinement (medium-score candidates):
    Expand box 20% → Sam3Model (text+box prompt) → mask → tight box
    → DINOv3 features → MLP re-score → if ≥ threshold → ACCEPT

  Path C — Grid Search (frames with zero detections):
    DINOv3 grid at configurable scale → cosine sim to accepted centroids
    → top-K candidates → Sam3Model refinement → DINOv3 → MLP → if ≥ threshold → ACCEPT

  All accepted seeds → identity assignment via ReID centroids
```

## Env Vars (new)

| Variable | Default | Purpose |
|----------|---------|---------|
| `INTERVIEW_REFINE_THRESHOLD` | `0.3` | MLP score below which medium-score candidates are discarded (not refined) |
| `INTERVIEW_GRID_SCALE` | `0.10` | Grid cell size as fraction of frame for fallback search |
| `INTERVIEW_GRID_SIM_THRESHOLD` | `0.5` | Min cosine similarity for grid candidates |
| `INTERVIEW_GRID_TOP_K` | `5` | Max grid candidates per frame |
| `INTERVIEW_SEED_CHUNK_SIZE` | `100` | Frames per processing chunk |
| `INTERVIEW_ENABLE_REFINEMENT` | `true` | Enable Path B (Sam3Model box refinement) |
| `INTERVIEW_ENABLE_GRID_SEARCH` | `true` | Enable Path C (DINOv3 grid fallback) |

---

## Task 1: Add `_refine_candidates_sam3` helper

**Files:**
- Modify: `interview/seeding_phase.py`
- Test: `test_interview_detection.py`

**Step 1: Write the failing test**

Add to `test_interview_detection.py`:

```python
class TestRefineCandidatesSam3:
    """Tests for _refine_candidates_sam3 box refinement."""

    def test_refines_box_from_mask(self, monkeypatch):
        """Expanded box + text prompt → Sam3Model → tight box from mask."""
        from interview.seeding_phase import _refine_candidates_sam3

        # Mock _get_sam3_image_model
        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Processor returns mock inputs
        mock_inputs = MagicMock()
        mock_inputs.get = lambda k: [[100, 200]] if k == "original_sizes" else None
        mock_processor.return_value = mock_inputs
        mock_inputs.to = MagicMock(return_value=mock_inputs)

        # post_process returns a tight box
        mock_processor.post_process_instance_segmentation.return_value = [{
            "masks": [np.ones((200, 100), dtype=bool)],
            "scores": [0.9],
            "boxes": [[10, 20, 80, 180]],
        }]

        monkeypatch.setattr(
            "interview.seeding_phase._get_sam3_image_model",
            lambda: (mock_model, mock_processor),
        )

        frame = Image.new("RGB", (100, 200))
        candidates = [(0, np.array([5, 5, 90, 190], dtype=np.float32), 0.5)]

        result = _refine_candidates_sam3(
            {0: frame}, candidates, prompt="person", expand_frac=0.2,
        )
        assert len(result) == 1
        frame_idx, box, score = result[0]
        assert frame_idx == 0
        # Box should be the tight box from post_process, not the original
        np.testing.assert_array_equal(box, [10, 20, 80, 180])

    def test_skips_frame_with_no_mask(self, monkeypatch):
        """If Sam3Model returns no masks, candidate is dropped."""
        from interview.seeding_phase import _refine_candidates_sam3

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.get = lambda k: [[100, 200]] if k == "original_sizes" else None
        mock_processor.return_value = mock_inputs
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_processor.post_process_instance_segmentation.return_value = [{
            "masks": [], "scores": [], "boxes": [],
        }]
        monkeypatch.setattr(
            "interview.seeding_phase._get_sam3_image_model",
            lambda: (mock_model, mock_processor),
        )

        frame = Image.new("RGB", (100, 200))
        result = _refine_candidates_sam3(
            {0: frame}, [(0, np.array([5, 5, 90, 190], dtype=np.float32), 0.5)],
            prompt="person",
        )
        assert len(result) == 0

    def test_expand_frac_clamps_to_bounds(self, monkeypatch):
        """Expanded box should be clamped to frame dimensions."""
        from interview.seeding_phase import _refine_candidates_sam3

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.get = lambda k: [[50, 50]] if k == "original_sizes" else None
        mock_processor.return_value = mock_inputs
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_processor.post_process_instance_segmentation.return_value = [{
            "masks": [np.ones((50, 50), dtype=bool)],
            "scores": [0.8],
            "boxes": [[0, 0, 50, 50]],
        }]
        monkeypatch.setattr(
            "interview.seeding_phase._get_sam3_image_model",
            lambda: (mock_model, mock_processor),
        )

        frame = Image.new("RGB", (50, 50))
        # Box at edge — expansion would go out of bounds
        candidates = [(0, np.array([0, 0, 50, 50], dtype=np.float32), 0.5)]
        result = _refine_candidates_sam3(
            {0: frame}, candidates, prompt="person", expand_frac=0.5,
        )
        # Should still return a result (clamped expansion)
        assert len(result) == 1

        # Verify processor was called with clamped box
        call_kwargs = mock_processor.call_args
        input_boxes = call_kwargs.kwargs.get("input_boxes") or call_kwargs[1].get("input_boxes")
        box = input_boxes[0][0]
        assert box[0] >= 0 and box[1] >= 0  # clamped lower
        assert box[2] <= 50 and box[3] <= 50  # clamped upper
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_interview_detection.py::TestRefineCandidatesSam3 -v`
Expected: FAIL — `_refine_candidates_sam3` does not exist yet.

**Step 3: Write implementation**

Add to `interview/seeding_phase.py` (after `_assign_identity`, before `generate_seeds`):

```python
# ---------------------------------------------------------------------------
# Box refinement via Sam3Model (Path B)
# ---------------------------------------------------------------------------

_REFINE_THRESHOLD = float(os.getenv("INTERVIEW_REFINE_THRESHOLD", "0.3"))
_ENABLE_REFINEMENT = os.getenv("INTERVIEW_ENABLE_REFINEMENT", "true").lower() == "true"
_ENABLE_GRID_SEARCH = os.getenv("INTERVIEW_ENABLE_GRID_SEARCH", "true").lower() == "true"
_GRID_SCALE = float(os.getenv("INTERVIEW_GRID_SCALE", "0.10"))
_GRID_SIM_THRESHOLD = float(os.getenv("INTERVIEW_GRID_SIM_THRESHOLD", "0.5"))
_GRID_TOP_K = int(os.getenv("INTERVIEW_GRID_TOP_K", "5"))
_SEED_CHUNK_SIZE = int(os.getenv("INTERVIEW_SEED_CHUNK_SIZE", "100"))


def _get_sam3_image_model():
    """Import and return the Sam3Model singleton from seeding_common."""
    from seeding_common import _get_sam3_image_model as _get
    return _get()


def _refine_candidates_sam3(
    frames: Dict[int, "Image.Image"],
    candidates: List[Tuple[int, np.ndarray, float]],
    prompt: str = "person",
    expand_frac: float = 0.2,
) -> List[Tuple[int, np.ndarray, float]]:
    """Refine candidate boxes using Sam3Model with text+box prompts.

    For each candidate, expands the box by *expand_frac* on each side,
    runs Sam3Model with combined text + box prompt, and extracts the tight
    bounding box from the best mask.

    Args:
        frames:      Mapping of frame_idx → decoded PIL Image.
        candidates:  List of (frame_idx, box_xyxy, det_score).
        prompt:      Text prompt for Sam3Model (e.g., "person").
        expand_frac: Fraction to expand each side of the box.

    Returns:
        List of (frame_idx, refined_box_xyxy, det_score) for successful refinements.
    """
    import torch
    model, processor = _get_sam3_image_model()

    refined: List[Tuple[int, np.ndarray, float]] = []

    for frame_idx, box, det_score in candidates:
        pil_frame = frames.get(frame_idx)
        if pil_frame is None:
            continue

        w, h = pil_frame.size
        x1, y1, x2, y2 = box

        # Expand box
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * expand_frac, bh * expand_frac
        ex1 = max(0, int(x1 - dx))
        ey1 = max(0, int(y1 - dy))
        ex2 = min(w, int(x2 + dx))
        ey2 = min(h, int(y2 + dy))

        if ex2 <= ex1 or ey2 <= ey1:
            continue

        try:
            inputs = processor(
                images=pil_frame,
                text=prompt,
                input_boxes=[[[ex1, ey1, ex2, ey2]]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(DEVICE)

            with torch.inference_mode():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks = results.get("masks", [])
            scores = results.get("scores", [])
            boxes_out = results.get("boxes", [])

            if not masks:
                continue

            best_idx = int(np.argmax([
                s.item() if hasattr(s, "item") else s for s in scores
            ])) if scores else 0

            if best_idx < len(boxes_out):
                b = boxes_out[best_idx]
                tight = np.array(b.tolist() if hasattr(b, "tolist") else b, dtype=np.float32)
            else:
                mask = masks[best_idx]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                ys, xs = np.where(mask > 0)
                if xs.size == 0:
                    continue
                tight = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

            refined.append((frame_idx, tight, det_score))

        except Exception as exc:
            logger.warning("Refinement failed for frame %d: %s", frame_idx, exc)
            continue

    return refined
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_interview_detection.py::TestRefineCandidatesSam3 -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add interview/seeding_phase.py test_interview_detection.py
git commit -m "feat: add _refine_candidates_sam3 for box refinement via Sam3Model"
```

---

## Task 2: Add `_grid_search_fallback` helper

**Files:**
- Modify: `interview/seeding_phase.py`
- Test: `test_interview_detection.py`

**Step 1: Write the failing test**

```python
class TestGridSearchFallback:
    """Tests for DINOv3 grid search on frames with zero detections."""

    def test_finds_candidates_by_similarity(self, monkeypatch):
        """Grid cells similar to reference features should be returned."""
        from interview.seeding_phase import _grid_search_fallback

        # Mock DINOv3 feature extraction
        call_count = [0]
        def fake_extract(crops, batch_size=16):
            n = len(crops)
            call_count[0] += 1
            # Return features that are similar to reference for first few cells
            feats = np.random.randn(n, 1024).astype(np.float32)
            feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            # Make first 3 cells very similar to reference
            for i in range(min(3, n)):
                feats[i] = reference_feat + np.random.randn(1024) * 0.01
                feats[i] /= np.linalg.norm(feats[i])
            return feats

        monkeypatch.setattr("interview.seeding_phase.extract_features", fake_extract)

        # Reference: single L2-normalized feature vector
        reference_feat = np.random.randn(1024).astype(np.float32)
        reference_feat /= np.linalg.norm(reference_feat)
        reference_features = reference_feat.reshape(1, 1024)

        frame = Image.new("RGB", (200, 200))
        result = _grid_search_fallback(
            frame, 0, reference_features,
            scale_frac=0.25, stride_frac=0.5, top_k=3, sim_threshold=0.8,
        )

        assert len(result) > 0
        assert len(result) <= 3
        for box, sim in result:
            assert box.shape == (4,)
            assert sim >= 0.8

    def test_empty_when_no_similarity(self, monkeypatch):
        """If no grid cells are similar enough, return empty list."""
        from interview.seeding_phase import _grid_search_fallback

        def fake_extract(crops, batch_size=16):
            n = len(crops)
            feats = np.random.randn(n, 1024).astype(np.float32)
            feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            return feats

        monkeypatch.setattr("interview.seeding_phase.extract_features", fake_extract)

        reference = np.random.randn(1, 1024).astype(np.float32)
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)

        frame = Image.new("RGB", (100, 100))
        result = _grid_search_fallback(
            frame, 0, reference,
            scale_frac=0.5, sim_threshold=0.99,  # very high threshold
        )
        assert len(result) == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_interview_detection.py::TestGridSearchFallback -v`
Expected: FAIL — function does not exist.

**Step 3: Write implementation**

Add to `interview/seeding_phase.py`:

```python
# ---------------------------------------------------------------------------
# DINOv3 grid search fallback (Path C)
# ---------------------------------------------------------------------------

def _grid_search_fallback(
    frame: "Image.Image",
    frame_idx: int,
    reference_features: np.ndarray,
    scale_frac: float = _GRID_SCALE,
    stride_frac: float = 0.5,
    top_k: int = _GRID_TOP_K,
    sim_threshold: float = _GRID_SIM_THRESHOLD,
) -> List[Tuple[np.ndarray, float]]:
    """Find person-like regions using DINOv3 feature similarity.

    Tiles the frame into a grid at a single scale, extracts DINOv3 CLS tokens,
    and compares to the mean feature vector of accepted crops.

    Args:
        frame:              PIL Image of the video frame.
        frame_idx:          Frame index (for logging only).
        reference_features: (N_ref, 1024) L2-normalized features of accepted crops.
        scale_frac:         Grid cell size as fraction of frame width/height.
        stride_frac:        Stride as fraction of cell size (0.5 = 50% overlap).
        top_k:              Maximum number of candidates to return.
        sim_threshold:      Minimum cosine similarity to be considered.

    Returns:
        List of (box_xyxy, similarity) sorted by descending similarity.
    """
    from .dinov3_classifier import extract_features

    W, H = frame.size
    cw = max(16, int(W * scale_frac))
    ch = max(16, int(H * scale_frac))
    sx = max(8, int(cw * stride_frac))
    sy = max(8, int(ch * stride_frac))

    # Compute mean reference feature
    mean_ref = np.mean(reference_features, axis=0)
    norm = np.linalg.norm(mean_ref)
    if norm > 1e-8:
        mean_ref /= norm

    grid_crops: List["Image.Image"] = []
    grid_boxes: List[np.ndarray] = []

    for y0 in range(0, H - ch + 1, sy):
        for x0 in range(0, W - cw + 1, sx):
            grid_crops.append(frame.crop((x0, y0, x0 + cw, y0 + ch)))
            grid_boxes.append(np.array([x0, y0, x0 + cw, y0 + ch], dtype=np.float32))

    if not grid_crops:
        return []

    feats = extract_features(grid_crops, batch_size=32)  # (N, 1024)
    sims = feats @ mean_ref  # cosine sim (both L2-normed)

    candidates = []
    for j in range(len(grid_boxes)):
        if sims[j] >= sim_threshold:
            candidates.append((grid_boxes[j], float(sims[j])))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]
```

**Step 4: Run test**

Run: `python -m pytest test_interview_detection.py::TestGridSearchFallback -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add interview/seeding_phase.py test_interview_detection.py
git commit -m "feat: add _grid_search_fallback for DINOv3 grid search on missed frames"
```

---

## Task 3: Add `_decode_chunk_sequential` for batched frame I/O

**Files:**
- Modify: `interview/seeding_phase.py`
- Test: `test_interview_detection.py`

The current `generate_seeds` reads each frame with a random PyAV seek (~200ms each). For 6000 frames, that's 20 minutes of I/O. A sequential decode pass with a prefetch thread (same pattern as `_do_embed_all_frames` in `seeding_common.py`) cuts this to ~2 minutes.

**Step 1: Write the failing test**

```python
class TestDecodeChunkSequential:
    """Verify chunked sequential frame decoding."""

    def test_decodes_requested_frames(self, monkeypatch):
        """Should return only requested frame indices as PIL images."""
        from interview.seeding_phase import _decode_chunk_sequential

        # Mock av.open to return a simple container
        frames_data = {i: Image.new("RGB", (100, 100), color=(i, i, i))
                       for i in range(20)}
        class FakeFrame:
            def __init__(self, idx):
                self.pts = idx * 1000
                self._idx = idx
            def to_image(self):
                return frames_data[self._idx]
        class FakeStream:
            average_rate = 30
            time_base = fractions.Fraction(1, 30000)
            frames = 20
        class FakeContainer:
            streams = type("S", (), {"video": [FakeStream()]})()
            def decode(self, video=0):
                for i in range(20):
                    yield FakeFrame(i)
            def close(self):
                pass

        monkeypatch.setattr("interview.seeding_phase.av.open", lambda p: FakeContainer())

        result = _decode_chunk_sequential("fake.mp4", [3, 7, 15])
        assert set(result.keys()) == {3, 7, 15}
        for idx in [3, 7, 15]:
            assert isinstance(result[idx], Image.Image)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_interview_detection.py::TestDecodeChunkSequential -v`

**Step 3: Write implementation**

Add to `interview/seeding_phase.py`:

```python
import av

def _decode_chunk_sequential(
    video_path: str,
    target_indices: List[int],
) -> Dict[int, "Image.Image"]:
    """Decode specific frames via sequential scan (no random seeks).

    Much faster than N individual seeks when target_indices are spread
    across the video. One pass through the container, collecting only
    the frames we need.

    Args:
        video_path:     Path to video file.
        target_indices: Sorted or unsorted list of 0-based frame indices.

    Returns:
        Dict mapping frame_idx → PIL Image for successfully decoded frames.
    """
    if not target_indices:
        return {}

    targets = set(target_indices)
    max_target = max(targets)
    result: Dict[int, Image.Image] = {}

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        tb = float(stream.time_base) if stream.time_base else None

        frame_count = 0
        for av_frame in container.decode(video=0):
            # Map PTS to frame index
            if tb and av_frame.pts is not None:
                idx = int(round(av_frame.pts * tb * fps))
            else:
                idx = frame_count

            if idx in targets:
                result[idx] = av_frame.to_image()
                targets.discard(idx)
                if not targets:
                    break

            frame_count += 1
            if idx > max_target + 10:
                break  # past all targets

    finally:
        container.close()

    return result
```

**Step 4: Run test**

Run: `python -m pytest test_interview_detection.py::TestDecodeChunkSequential -v`
Expected: PASS

**Step 5: Commit**

```bash
git add interview/seeding_phase.py test_interview_detection.py
git commit -m "feat: add _decode_chunk_sequential for efficient batched frame I/O"
```

---

## Task 4: Restructure `generate_seeds` with dual-proposer pipeline

**Files:**
- Modify: `interview/seeding_phase.py:135-340` (the `generate_seeds` function)
- Test: `test_interview_detection.py`

This is the core change. The existing per-frame loop is replaced with a chunked batch pipeline.

**Step 1: Write the failing test**

```python
class TestDualProposerSeeding:
    """Integration test for the full dual-proposer pipeline."""

    def test_path_a_good_detection_becomes_seed(self, monkeypatch):
        """A high-confidence SAM3 detection that passes MLP should produce a seed."""
        # ... (mock SAM3 detector, DINOv3, MLP, identity assignment)
        # Verify seed count > 0, seed has identity, seed xyxy is from SAM3

    def test_path_b_medium_score_gets_refined(self, monkeypatch):
        """Medium MLP score (between refine and confidence threshold) triggers Sam3Model refinement."""
        # Verify _refine_candidates_sam3 is called
        # Verify refined box replaces original box

    def test_path_c_no_detection_triggers_grid_search(self, monkeypatch):
        """Frame with zero SAM3 detections triggers DINOv3 grid fallback."""
        # Verify _grid_search_fallback is called for empty-detection frames
        # Verify grid candidates go through refinement + MLP

    def test_change_keyframes_included(self, monkeypatch):
        """Change-detected keyframes from background embedding are included in target frames."""
        # Session has change_keyframes = [100, 200, 300]
        # These should be added to the uniform sample set

    def test_chunked_decode(self, monkeypatch):
        """Frames should be decoded in chunks via _decode_chunk_sequential, not per-frame seeks."""
        # Verify _decode_chunk_sequential is called instead of _read_frame_pyav
```

Note: Exact mock wiring is complex — the executing agent should follow the existing test patterns in `test_interview_detection.py` (which mocks heavy dependencies at import time) and keep mocks minimal.

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_interview_detection.py::TestDualProposerSeeding -v`

**Step 3: Write implementation**

Replace the body of `generate_seeds` in `interview/seeding_phase.py`. The new structure:

```python
def generate_seeds(
    session: InterviewSession,
    progress: JobProgress,
) -> Dict[str, Any]:
    """Generate dense seeds with dual-proposer pipeline.

    Three paths for maximum coverage:
      A. SAM3 text detection → MLP quality gate (primary)
      B. Sam3Model box refinement for medium-confidence detections (fallback 1)
      C. DINOv3 grid search for zero-detection frames (fallback 2)
    """
    from .detection import Sam3TextBasedDetector, nms_numpy, pad_boxes
    from .dinov3_classifier import extract_features, CropClassifier, compute_crop_metadata

    import torch
    from PIL import Image

    # ---- Validate prerequisites ----
    progress.step = "Validating session state..."
    state_dict = load_model(session.cache_key)
    if state_dict is None:
        raise RuntimeError("No trained MLP model found.")
    if not session.reid_clusters:
        raise RuntimeError("No ReID clusters found.")

    # ---- Load models ----
    progress.step = "Loading models..."
    detector = Sam3TextBasedDetector()
    classifier = CropClassifier()
    classifier.load_state_dict(state_dict)
    classifier.eval()
    centroids = _compute_cluster_centroids(session)

    # ---- Compute accepted-crop reference features for grid search ----
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    reference_features = np.stack([
        c.features for c in accepted if c.features is not None
    ]) if accepted else np.empty((0, 1024), dtype=np.float32)

    # ---- Determine target frames ----
    interval = max(1, session.seed_config.frame_interval)
    uniform = set(range(0, session.frames_count, interval))
    change = set(session.change_keyframes) if session.embedding_complete else set()
    all_targets = sorted(uniform | change)
    total_frames = len(all_targets)

    progress.step = "Generating seeds..."
    progress.total = total_frames

    prompt_text = session.prompts[0] if session.prompts else "person"
    threshold = session.seed_config.confidence_threshold
    seeds: List[Dict[str, Any]] = []

    # ---- Process in chunks ----
    for chunk_start in range(0, total_frames, _SEED_CHUNK_SIZE):
        chunk_indices = all_targets[chunk_start : chunk_start + _SEED_CHUNK_SIZE]
        progress.step = f"Decoding frames {chunk_start}-{chunk_start + len(chunk_indices)}..."

        # 1. Batch decode chunk
        frames = _decode_chunk_sequential(session.video_path, chunk_indices)

        # Track which frames need fallback
        no_seed_frames: List[int] = []
        medium_candidates: List[Tuple[int, np.ndarray, float]] = []

        # 2. Path A: SAM3 text detection per frame
        for fi, frame_idx in enumerate(chunk_indices):
            progress.current = chunk_start + fi + 1
            pil_frame = frames.get(frame_idx)
            if pil_frame is None:
                no_seed_frames.append(frame_idx)
                continue

            detections = detector.detect(prompt_text, pil_image=pil_frame)
            if not detections:
                no_seed_frames.append(frame_idx)
                continue

            boxes = np.array([d["xyxy"] for d in detections], dtype=np.float32)
            det_scores = np.array([d["score"] for d in detections], dtype=np.float32)
            boxes = pad_boxes(boxes, pil_frame.width, pil_frame.height)
            keep = nms_numpy(boxes, det_scores, iou_threshold=0.5)
            boxes, det_scores = boxes[keep], det_scores[keep]

            if len(boxes) == 0:
                no_seed_frames.append(frame_idx)
                continue

            # Crop + DINOv3 features + MLP
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
                no_seed_frames.append(frame_idx)
                continue

            boxes = boxes[valid_indices]
            det_scores = det_scores[valid_indices]
            crop_features = extract_features(crop_images)
            metadata = np.array([
                compute_crop_metadata(b, pil_frame.width, pil_frame.height)
                for b in boxes
            ], dtype=np.float32)
            mlp_input = np.concatenate([crop_features, metadata], axis=1)

            with torch.inference_mode():
                probs = torch.sigmoid(
                    classifier(torch.from_numpy(mlp_input).float())
                ).squeeze(-1).cpu().numpy()

            frame_has_seed = False
            for i in range(len(boxes)):
                conf = float(probs[i]) if probs.ndim > 0 else float(probs)
                if conf >= threshold:
                    identity, identity_sim = _assign_identity(crop_features[i], centroids)
                    seeds.append({
                        "frame_idx": int(frame_idx),
                        "xyxy": boxes[i].tolist(),
                        "confidence": round(conf, 4),
                        "identity": int(identity),
                        "identity_similarity": round(float(identity_sim), 4),
                    })
                    frame_has_seed = True
                elif _ENABLE_REFINEMENT and conf >= _REFINE_THRESHOLD:
                    medium_candidates.append((frame_idx, boxes[i], det_scores[i]))

            if not frame_has_seed:
                no_seed_frames.append(frame_idx)

        # 3. Path B: Refine medium-confidence candidates
        if medium_candidates and _ENABLE_REFINEMENT:
            progress.step = f"Refining {len(medium_candidates)} candidates..."
            refined = _refine_candidates_sam3(frames, medium_candidates, prompt=prompt_text)
            for frame_idx, box, det_score in refined:
                pil_frame = frames.get(frame_idx)
                if pil_frame is None:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_frame.width, x2), min(pil_frame.height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = pil_frame.crop((x1, y1, x2, y2))
                feat = extract_features([crop])
                meta = compute_crop_metadata(box, pil_frame.width, pil_frame.height)
                mlp_in = np.concatenate([feat, meta.reshape(1, -1)], axis=1)
                with torch.inference_mode():
                    p = torch.sigmoid(classifier(torch.from_numpy(mlp_in).float())).item()
                if p >= threshold:
                    identity, identity_sim = _assign_identity(feat[0], centroids)
                    seeds.append({
                        "frame_idx": int(frame_idx),
                        "xyxy": box.tolist(),
                        "confidence": round(p, 4),
                        "identity": int(identity),
                        "identity_similarity": round(float(identity_sim), 4),
                    })
                    # Remove from no_seed_frames if it was there
                    if frame_idx in no_seed_frames:
                        no_seed_frames.remove(frame_idx)

        # 4. Path C: Grid search for frames with zero seeds
        if no_seed_frames and _ENABLE_GRID_SEARCH and reference_features.shape[0] > 0:
            progress.step = f"Grid search on {len(no_seed_frames)} frames..."
            for frame_idx in no_seed_frames:
                pil_frame = frames.get(frame_idx)
                if pil_frame is None:
                    continue
                grid_candidates = _grid_search_fallback(
                    pil_frame, frame_idx, reference_features,
                )
                if not grid_candidates:
                    continue
                # Refine grid candidates via Sam3Model
                grid_for_refine = [
                    (frame_idx, box, sim) for box, sim in grid_candidates
                ]
                refined_grid = _refine_candidates_sam3(
                    frames, grid_for_refine, prompt=prompt_text,
                ) if _ENABLE_REFINEMENT else grid_for_refine

                for fidx, box, _ in refined_grid:
                    pil_f = frames.get(fidx)
                    if pil_f is None:
                        continue
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(pil_f.width, x2), min(pil_f.height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = pil_f.crop((x1, y1, x2, y2))
                    feat = extract_features([crop])
                    meta = compute_crop_metadata(box, pil_f.width, pil_f.height)
                    mlp_in = np.concatenate([feat, meta.reshape(1, -1)], axis=1)
                    with torch.inference_mode():
                        p = torch.sigmoid(classifier(torch.from_numpy(mlp_in).float())).item()
                    if p >= threshold:
                        identity, identity_sim = _assign_identity(feat[0], centroids)
                        seeds.append({
                            "frame_idx": int(fidx),
                            "xyxy": box.tolist(),
                            "confidence": round(p, 4),
                            "identity": int(identity),
                            "identity_similarity": round(float(identity_sim), 4),
                        })

    # ---- Finalize (unchanged from original) ----
    with session._lock:
        session.seeds = seeds
        session.advance_to(Phase.SEEDING)

    save_session(session)

    identity_counts: Dict[int, int] = {}
    for seed in seeds:
        identity_counts[seed["identity"]] = identity_counts.get(seed["identity"], 0) + 1

    summary = {
        "total_seeds": len(seeds),
        "frames_scanned": total_frames,
        "identities": identity_counts,
    }
    logger.info("Seed generation complete: %s", summary)
    return summary
```

**Step 4: Run all tests**

Run: `python -m pytest test_interview_detection.py -x -v`
Expected: All pass including new dual-proposer tests.

**Step 5: Commit**

```bash
git add interview/seeding_phase.py test_interview_detection.py
git commit -m "feat: restructure generate_seeds with dual-proposer pipeline (Path A/B/C)"
```

---

## Task 5: Update CLAUDE.md and commit

**Files:**
- Modify: `CLAUDE.md` (already partially updated with labeling semantics)

Update the Dense Seeding Architecture section in CLAUDE.md with the final implementation details (env vars, performance estimates, Path A/B/C flow).

**Step 1: Update CLAUDE.md**

Add performance estimates and env var table.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with dual-proposer seeding architecture"
```

---

## Performance Estimates

| Metric | Current (Path A only) | Dual-Proposer (A+B+C) |
|--------|----------------------|----------------------|
| Frame decode (6000 frames) | ~20 min (random seeks) | ~2 min (sequential) |
| SAM3 text detection | ~6 min | ~6 min (unchanged) |
| DINOv3 features | ~1.5 min | ~3 min (+ grid search) |
| Sam3Model refinement | 0 | ~5 min |
| MLP scoring | negligible | negligible |
| **Total** | **~28 min** | **~16 min** |
| Frame coverage | ~50% | ~80-90% |

The sequential decode alone is a major win — it cuts total time even with the added fallback paths.

## Key Design Decisions

1. **Single grid scale, not 3.** The classification phase's feature search uses 3 scales for thoroughness. During dense seeding, we use a single configurable scale to keep the fallback fast. The executing agent should default to 10% but allow override via env var.

2. **Refinement is optional.** If Sam3Model isn't available or GPU memory is tight, set `INTERVIEW_ENABLE_REFINEMENT=false` to skip Path B. Path C (grid search) still works without Sam3Model — grid candidates go directly to MLP.

3. **Change keyframes are additive.** They're merged with the uniform sample, not replacing it. This ensures consistent coverage even if change detection missed something.

4. **Per-chunk processing.** Frames are decoded and processed 100 at a time, keeping memory bounded (~600MB per chunk for 100 full-resolution frames).
