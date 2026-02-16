# Lightweight Change Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace SAM3 GPU-based frame embeddings with a lightweight CPU-based change detection pipeline (pixel diff + histogram distance), reducing the background embedding phase from 5-10 minutes to ~5 seconds.

**Architecture:** Single-pass video scan computes two change signals per consecutive frame pair: downscaled pixel L1 difference and HSV histogram chi-squared distance. Scores are combined via max(), then fed into the existing smoothing + peak detection pipeline. The downstream contract (`session.change_keyframes`) is unchanged.

**Tech Stack:** numpy, PIL (already in requirements). No new dependencies.

---

### Task 1: Implement `compute_lightweight_change_scores()`

**Files:**
- Modify: `seeding_common.py`
- Test: `test_lightweight_change.py` (new)

**Step 1: Write the failing test**

Create `test_lightweight_change.py`:

```python
"""Tests for lightweight change detection (no GPU, no SAM3)."""
import numpy as np
import pytest
from PIL import Image


def _make_frame(r, g, b, w=128, h=128):
    """Create a solid-color PIL Image."""
    arr = np.full((h, w, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr)


class TestLightweightChangeScores:
    """Unit tests for compute_lightweight_change_scores_from_frames()."""

    def test_identical_frames_zero_scores(self):
        from seeding_common import compute_lightweight_change_scores_from_frames
        frames = [_make_frame(128, 128, 128)] * 5
        scores = compute_lightweight_change_scores_from_frames(frames)
        assert scores.shape == (5,)
        assert scores[0] == 0.0  # first frame always 0
        np.testing.assert_allclose(scores[1:], 0.0, atol=1e-6)

    def test_scene_change_spike(self):
        from seeding_common import compute_lightweight_change_scores_from_frames
        frames = (
            [_make_frame(0, 0, 0)] * 3 +
            [_make_frame(255, 255, 255)] +
            [_make_frame(0, 0, 0)] * 3
        )
        scores = compute_lightweight_change_scores_from_frames(frames)
        assert scores.shape == (7,)
        # Spike at index 3 (black -> white) and 4 (white -> black)
        assert scores[3] > 0.5
        assert scores[4] > 0.5
        # Stable regions should be near 0
        assert scores[1] < 0.1
        assert scores[2] < 0.1

    def test_gradual_brightness_detected_by_histogram(self):
        from seeding_common import compute_lightweight_change_scores_from_frames
        # Gradual brightness ramp: 0, 25, 50, ..., 250
        frames = [_make_frame(i * 25, i * 25, i * 25) for i in range(11)]
        scores = compute_lightweight_change_scores_from_frames(frames)
        assert scores.shape == (11,)
        # All transitions should have non-zero score
        assert all(s > 0 for s in scores[1:])

    def test_single_frame_returns_zero(self):
        from seeding_common import compute_lightweight_change_scores_from_frames
        frames = [_make_frame(100, 100, 100)]
        scores = compute_lightweight_change_scores_from_frames(frames)
        assert scores.shape == (1,)
        assert scores[0] == 0.0

    def test_empty_frames_returns_empty(self):
        from seeding_common import compute_lightweight_change_scores_from_frames
        scores = compute_lightweight_change_scores_from_frames([])
        assert scores.shape == (0,)


class TestLightweightFromVideo:
    """Integration test: lightweight scores -> smooth -> select_keyframes."""

    def test_scores_through_existing_pipeline(self):
        from seeding_common import (
            compute_lightweight_change_scores_from_frames,
            smooth_change_scores, select_keyframes,
        )
        # 20 frames: stable -> change -> stable -> change -> stable
        frames = (
            [_make_frame(50, 50, 50)] * 5 +
            [_make_frame(200, 50, 50)] * 5 +
            [_make_frame(50, 200, 50)] * 5 +
            [_make_frame(50, 50, 200)] * 5
        )
        scores = compute_lightweight_change_scores_from_frames(frames)
        smooth = smooth_change_scores(scores, kernel_size=3)
        keyframes = select_keyframes(len(frames), 0.2, smooth, min_spacing=2)
        # Should detect at least 2 of the 3 transitions
        assert len(keyframes) >= 2
        # Keyframes should be near the transition points (5, 10, 15)
        transitions = {5, 10, 15}
        near_transition = sum(1 for kf in keyframes if any(abs(kf - t) <= 2 for t in transitions))
        assert near_transition >= 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_lightweight_change.py -x -v`
Expected: FAIL with ImportError (function doesn't exist yet)

**Step 3: Implement `compute_lightweight_change_scores_from_frames()`**

Add to `seeding_common.py` after the existing `compute_change_scores()` function (~line 828):

```python
def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to HSV float array.

    Uses the existing numpy-based RGB->HSV conversion pattern
    (no cv2 dependency). Input: (H, W, 3) uint8. Output: (H, W, 3) float32
    with H in [0, 360), S in [0, 1], V in [0, 1].
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    s = np.where(cmax > 0, delta / cmax, 0.0)

    return np.stack([h, s, cmax], axis=-1)


def _histogram_distance(frame_a: np.ndarray, frame_b: np.ndarray,
                         bins: int = 8) -> float:
    """Chi-squared distance between HSV histograms of two frames.

    Both inputs should be RGB uint8 arrays (H, W, 3). Returns a scalar
    distance in [0, +inf), where 0 means identical histograms.
    """
    hsv_a = _rgb_to_hsv_numpy(frame_a)
    hsv_b = _rgb_to_hsv_numpy(frame_b)

    # Quantize to bins
    ranges = [(0, 360), (0, 1), (0, 1)]
    hist_a, _ = np.histogramdd(
        hsv_a.reshape(-1, 3),
        bins=[bins, bins, bins],
        range=ranges,
    )
    hist_b, _ = np.histogramdd(
        hsv_b.reshape(-1, 3),
        bins=[bins, bins, bins],
        range=ranges,
    )

    # Normalize
    hist_a = hist_a / (hist_a.sum() + 1e-8)
    hist_b = hist_b / (hist_b.sum() + 1e-8)

    # Chi-squared distance
    denom = hist_a + hist_b + 1e-8
    return float(0.5 * np.sum((hist_a - hist_b) ** 2 / denom))


_CHANGE_THUMB_SIZE = (128, 128)


def compute_lightweight_change_scores_from_frames(
    frames: List[Image.Image],
) -> np.ndarray:
    """Compute per-frame change scores using pixel diff + histogram distance.

    Two complementary signals combined via max():
      - Pixel L1: mean absolute difference on 128x128 thumbnails (fast, gross changes)
      - Histogram chi-sq: HSV histogram distance (lighting, color shifts)

    Both signals are normalized to [0, 1] using their respective max values,
    then combined: score[i] = max(norm_pixel[i], norm_hist[i]).

    Args:
        frames: List of PIL Images in frame order.

    Returns:
        (N,) float32 array of change scores. First element is always 0.
    """
    n = len(frames)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    pixel_scores = np.zeros(n, dtype=np.float32)
    hist_scores = np.zeros(n, dtype=np.float32)

    prev_thumb = np.array(frames[0].resize(_CHANGE_THUMB_SIZE, Image.BILINEAR))

    for i in range(1, n):
        thumb = np.array(frames[i].resize(_CHANGE_THUMB_SIZE, Image.BILINEAR))

        # Signal 1: pixel L1 difference (normalized to [0, 1] by /255)
        pixel_scores[i] = np.mean(np.abs(
            thumb.astype(np.float32) - prev_thumb.astype(np.float32)
        )) / 255.0

        # Signal 2: histogram chi-squared distance
        hist_scores[i] = _histogram_distance(prev_thumb, thumb)

        prev_thumb = thumb

    # Normalize each to [0, 1]
    px_max = pixel_scores.max()
    if px_max > 0:
        pixel_scores /= px_max

    hs_max = hist_scores.max()
    if hs_max > 0:
        hist_scores /= hs_max

    # Combine: max of both signals
    return np.maximum(pixel_scores, hist_scores)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_lightweight_change.py -x -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add seeding_common.py test_lightweight_change.py
git commit -m "feat: add lightweight change detection (pixel diff + histogram)"
```

---

### Task 2: Add video-level wrapper with PyAV decoding

**Files:**
- Modify: `seeding_common.py`
- Modify: `test_lightweight_change.py`

**Step 1: Write the failing test**

Add to `test_lightweight_change.py`:

```python
class TestComputeLightweightFromVideo:
    """Test the video-level wrapper that decodes + computes scores."""

    def test_returns_scores_and_indices(self, tmp_path):
        """Verify the wrapper returns (scores_array, sampled_indices)."""
        from seeding_common import compute_lightweight_change_from_video
        # Create a minimal test video (or mock PyAV)
        # For unit testing, we mock _decode_frames_for_change
        import unittest.mock as mock

        frames = [_make_frame(100, 100, 100)] * 10
        with mock.patch(
            'seeding_common._decode_frames_for_change',
            return_value=(frames, list(range(10))),
        ):
            scores, indices = compute_lightweight_change_from_video(
                "dummy.mp4", target_fps=10,
            )
            assert scores.shape == (10,)
            assert len(indices) == 10
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_lightweight_change.py::TestComputeLightweightFromVideo -x -v`
Expected: FAIL (function doesn't exist yet)

**Step 3: Implement `compute_lightweight_change_from_video()`**

Add to `seeding_common.py`:

```python
def _decode_frames_for_change(
    video_path: str,
    target_fps: Optional[float] = None,
    pause_event: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
) -> Tuple[List[Image.Image], List[int]]:
    """Decode video frames (subsampled to target_fps) for change detection.

    Returns (frames_list, sampled_indices) where sampled_indices maps
    position in frames_list to the original 0-based frame index.
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate or 30)
    total_frames = stream.frames or 0

    skip = max(1, int(round(src_fps / target_fps))) if target_fps else 1

    frames: List[Image.Image] = []
    indices: List[int] = []
    frame_idx = 0

    for av_frame in container.decode(video=0):
        if frame_idx % skip == 0:
            frames.append(av_frame.to_image())
            indices.append(frame_idx)

            if progress_callback and len(frames) % 500 == 0:
                est_total = total_frames // skip if total_frames else 0
                progress_callback(len(frames), est_total)

            if pause_event is not None:
                pause_event.wait()

        frame_idx += 1

    container.close()
    return frames, indices


def compute_lightweight_change_from_video(
    video_path: str,
    target_fps: Optional[float] = 10.0,
    pause_event: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Compute lightweight change scores directly from a video file.

    Decodes at target_fps, computes pixel + histogram change scores.

    Returns:
        (scores, sampled_indices) — same contract as _do_embed_all_frames
        except scores is (N,) instead of (N, C) embeddings.
    """
    frames, indices = _decode_frames_for_change(
        video_path, target_fps, pause_event, progress_callback,
    )
    scores = compute_lightweight_change_scores_from_frames(frames)
    return scores, indices
```

**Step 4: Run tests**

Run: `python -m pytest test_lightweight_change.py -x -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add seeding_common.py test_lightweight_change.py
git commit -m "feat: add video-level lightweight change detection wrapper"
```

---

### Task 3: Wire into `run_embedding_background()`

**Files:**
- Modify: `interview/detection.py` (function `run_embedding_background`, lines 1203-1292)

**Step 1: Write the failing test**

Add to `test_lightweight_change.py`:

```python
class TestEmbeddingModeSwitch:
    """Test that INTERVIEW_EMBEDDING_MODE controls which pipeline runs."""

    def test_lightweight_mode_skips_sam3(self):
        """Verify lightweight mode doesn't import or call SAM3."""
        import os
        import unittest.mock as mock

        os.environ["INTERVIEW_EMBEDDING_MODE"] = "lightweight"

        scores = np.random.rand(100).astype(np.float32)
        indices = list(range(100))

        with mock.patch(
            'seeding_common.compute_lightweight_change_from_video',
            return_value=(scores, indices),
        ) as mock_lw:
            # Import after setting env var
            from interview.detection import _get_embedding_mode
            assert _get_embedding_mode() == "lightweight"
            mock_lw.assert_not_called()  # Just checking function exists

        os.environ.pop("INTERVIEW_EMBEDDING_MODE", None)
```

**Step 2: Implement the mode switch in `run_embedding_background()`**

At the top of `interview/detection.py`, add:

```python
EMBEDDING_MODE = os.getenv("INTERVIEW_EMBEDDING_MODE", "lightweight")

def _get_embedding_mode() -> str:
    return EMBEDDING_MODE
```

Replace the body of `run_embedding_background()` (~line 1226-1292):

```python
def run_embedding_background(
    session: InterviewSession,
    progress: Any,
) -> Dict[str, Any]:
    """Compute change-detected keyframes via lightweight or SAM3 pipeline."""
    t0 = time.time()

    video_path = session.video_path
    if not video_path:
        raise RuntimeError("Session has no video_path set.")

    pause_event = getattr(progress, '_pause_event', None)

    def _progress_cb(current: int, total: int):
        progress.step = f"Analyzing frames {current:,} / {total:,}..."
        progress.current = current
        progress.total = total

    if EMBEDDING_MODE == "lightweight":
        # Lightweight CPU-based change detection (pixel diff + histogram)
        progress.step = "Analyzing video for scene changes..."
        from seeding_common import compute_lightweight_change_from_video
        scores, sampled_indices = compute_lightweight_change_from_video(
            video_path,
            target_fps=EMBEDDING_TARGET_FPS,
            pause_event=pause_event,
            progress_callback=_progress_cb,
        )
        frames_count = len(scores)
        smooth = smooth_change_scores(scores, kernel_size=5)
    else:
        # Original SAM3 GPU embedding pipeline
        progress.step = "Computing frame embeddings..."
        embeds, sampled_indices = _do_embed_all_frames(
            video_path, DEFAULT_EMBEDDING_BATCH,
            progress_callback=_progress_cb,
            target_fps=EMBEDDING_TARGET_FPS,
            pause_event=pause_event,
        )
        frames_count = embeds.shape[0]
        diff = compute_change_scores(embeds)
        smooth = smooth_change_scores(diff, kernel_size=5)

    # Common: keyframe selection (unchanged)
    progress.step = "Selecting keyframes..."
    change_keyframes_sub = select_keyframes(
        frames_count, DEFAULT_KEYFRAME_FRAC, smooth,
        min_spacing=DEFAULT_MIN_SPACING,
    )
    change_keyframes = [
        sampled_indices[k] for k in change_keyframes_sub
        if k < len(sampled_indices)
    ]

    # Store on session (unchanged)
    with session._lock:
        session.embedding_complete = True
        session.change_keyframes = change_keyframes
        session.embedding_sampled_indices = sampled_indices
        session.touch()
        save_session(session)

    elapsed = time.time() - t0
    progress.step = "Analysis complete."
    progress.current = progress.total

    summary = {
        "frames_embedded": int(frames_count),
        "frames_total_in_video": sampled_indices[-1] + 1 if sampled_indices else 0,
        "change_keyframes": len(change_keyframes),
        "elapsed_seconds": round(elapsed, 1),
        "mode": EMBEDDING_MODE,
    }
    logger.info(
        "Background analysis (%s): %d frames, %d change keyframes in %.1fs",
        EMBEDDING_MODE, frames_count, len(change_keyframes), elapsed,
    )
    return summary
```

**Step 3: Run all tests**

Run: `python -m pytest test_lightweight_change.py test_interview_reid.py -x -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add interview/detection.py
git commit -m "feat: wire lightweight change detection into embedding pipeline"
```

---

### Task 4: Deploy and verify

**Step 1: Deploy to remote**

```bash
scp -P 31157 seeding_common.py interview/detection.py \
  dtc@airlab-dtc-main.lan.cmu.edu:/home/dtc/label-studio-ml-backend/label_studio_ml/examples/segment_anything_3_video/
```

Note: `seeding_common.py` goes to the root of `segment_anything_3_video/`, `detection.py` goes to `interview/`.

**Step 2: Restart container**

```bash
ssh bluemachine 'cd /home/dtc/label-studio-ml-backend/label_studio_ml/examples/segment_anything_3_video && docker compose restart segment_anything_3_video'
```

**Step 3: Verify in logs**

Start a new interview session. Watch `docker compose logs -f`:
- Should see `"Background analysis (lightweight): N frames, M change keyframes in X.Xs"`
- X should be < 30 seconds (not 5-10 minutes)
- No SAM3 model loading messages for image embeddings

**Step 4: Verify keyframe quality**

Check that detection round frames include reasonable change-detected keyframes by inspecting the sampled frames in the detection UI.

---

## Files Modified

| File | Changes |
|------|---------|
| `seeding_common.py` | `_rgb_to_hsv_numpy()`, `_histogram_distance()`, `compute_lightweight_change_scores_from_frames()`, `_decode_frames_for_change()`, `compute_lightweight_change_from_video()` |
| `interview/detection.py` | `EMBEDDING_MODE` constant, `_get_embedding_mode()`, `run_embedding_background()` mode switch |
| `test_lightweight_change.py` (new) | 7+ tests for lightweight change detection |

## Dependency Order

```
Task 1 (core functions + tests) ← no deps
Task 2 (video wrapper) ← depends on Task 1
Task 3 (wire into pipeline) ← depends on Task 2
Task 4 (deploy + verify) ← depends on Task 3
```

## Rollback

Set `INTERVIEW_EMBEDDING_MODE=sam3` in docker-compose.yml env to revert to GPU embeddings without any code changes.

## Verification

1. Unit tests: `python -m pytest test_lightweight_change.py -x -v`
2. Integration tests: `python -m pytest test_interview_reid.py -x -v` (no regressions)
3. Deploy, start new session, verify embedding completes in <30s
4. Verify detection rounds sample reasonable keyframes
5. Run full pipeline through seeding to confirm change keyframes are used
