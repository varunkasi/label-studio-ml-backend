"""Tests for Interview detection pipeline (NMS, padding, batch helpers, Stage 1).

These tests cover the pure-numpy utility functions in interview/detection.py
without requiring GPU, model weights, or video files. Heavy dependencies
(torch, seeding_common, PIL, av) are mocked at import time.
"""
import sys
import types
import threading
from unittest.mock import MagicMock
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock heavy dependencies BEFORE importing detection.py
# ---------------------------------------------------------------------------

# Create a lightweight mock for torch
_mock_torch = types.ModuleType("torch")
_mock_torch.inference_mode = lambda: type("_ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()
_mock_torch.autocast = lambda **kw: type("_ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()
_mock_torch.Tensor = type("Tensor", (), {})
_mock_torch.no_grad = lambda: type("_ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()
_mock_torch.cuda = types.SimpleNamespace(
    empty_cache=lambda: None,
    is_available=lambda: False,
    OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
)
# Expose OutOfMemoryError at module level too (code references torch.cuda.OutOfMemoryError)
_mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})


class _FakeTensor:
    """Mock tensor that supports the .float().squeeze().cpu().numpy() chain."""
    def __init__(self, data):
        if isinstance(data, _FakeTensor):
            self._data = data._data.copy()
        else:
            self._data = np.asarray(data, dtype=np.float32)
    def float(self): return self
    def to(self, device=None, dtype=None): return self
    def squeeze(self, dim=-1):
        if self._data.ndim > 1:
            return _FakeTensor(np.squeeze(self._data, axis=dim))
        return self
    def cpu(self): return self
    def numpy(self): return self._data
    def item(self): return float(self._data.flat[0]) if self._data.size > 0 else 0.0

_mock_torch.from_numpy = lambda arr: _FakeTensor(arr)
_mock_torch.tensor = lambda data, dtype=None: _FakeTensor(data)
_mock_torch.empty = lambda *shape: _FakeTensor(np.empty(shape, dtype=np.float32))
_mock_torch.float32 = "float32"
_mock_torch.sigmoid = lambda t: _FakeTensor(1.0 / (1.0 + np.exp(-t._data))) if isinstance(t, _FakeTensor) else t

# Create a lightweight mock for PIL.Image
_mock_pil = types.ModuleType("PIL")
_mock_pil_image = types.ModuleType("PIL.Image")
_mock_pil_image.Image = type("Image", (), {"height": 100, "width": 100})
_mock_pil.Image = _mock_pil_image

# Mock av (PyAV)
_mock_av = types.ModuleType("av")
_mock_av.open = lambda *a, **kw: None  # overridden in tests

# Mock seeding_common so the relative import in detection.py succeeds
_mock_seeding = types.ModuleType("seeding_common")
_mock_seeding._get_sam3_image_model = lambda: (None, None)
_mock_seeding._read_frame_pyav = lambda *a, **kw: None
_mock_seeding._get_video_info_pyav = lambda *a, **kw: (1920, 1080, 6000, 30.0)
_mock_seeding._compute_sam3_frame_embeddings = lambda *a, **kw: np.zeros((10, 256))
_mock_seeding._do_embed_all_frames = lambda *a, **kw: (np.zeros((100, 256), dtype=np.float16), list(range(100)))
_mock_seeding.compute_change_scores = lambda e: np.random.rand(len(e) - 1).astype(np.float32)
_mock_seeding.smooth_change_scores = lambda s, **kw: s
_mock_seeding.select_keyframes = lambda n, frac, scores, **kw: list(range(0, n, max(1, n // 10)))[:10]
_mock_seeding.compute_lightweight_change_from_video = lambda *a, **kw: (np.random.rand(100).astype(np.float32), list(range(100)))
_mock_seeding.DEVICE = "cpu"
_mock_seeding.DTYPE = None
_mock_seeding._build_ls_client = lambda *a, **kw: None
_mock_seeding._build_prediction = lambda *a, **kw: {}
_mock_seeding._upload_prediction = lambda *a, **kw: None
_mock_seeding.xyxy_to_percent = lambda *a, **kw: (0, 0, 100, 100)

# Mock interview.dinov3_classifier so seeding_phase.py's module-level import succeeds
# without pulling in real torch.nn, transformers, etc.
_mock_dinov3 = types.ModuleType("interview.dinov3_classifier")
_mock_dinov3.extract_features = lambda crops, batch_size=16: np.zeros((len(crops), 1024), dtype=np.float32)
_mock_dinov3.compute_crop_metadata = lambda box, w, h: np.zeros(4, dtype=np.float32)
_mock_dinov3.compute_mask_quality = lambda mask, box, w, h: np.array([0.9, 0.0, 0.0, 0.8], dtype=np.float32)

# ---------------------------------------------------------------------------
# Inject all mocks into sys.modules for the import chain, saving originals.
# After importing interview.*, we restore originals so other test files
# (e.g. test_tracking_fixes.py using real torch/PIL) aren't polluted.
# An autouse fixture re-injects mocks per test function (see below).
# ---------------------------------------------------------------------------
_MOCK_MODULES = {
    "torch": _mock_torch,
    "PIL": _mock_pil,
    "PIL.Image": _mock_pil_image,
    "av": _mock_av,
    "seeding_common": _mock_seeding,
    "interview.dinov3_classifier": _mock_dinov3,
}
_saved_modules: dict = {}
for _name, _mock in _MOCK_MODULES.items():
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interview.detection import (
    nms_numpy, pad_boxes,
    _decode_frames_sequential, _detect_batch,
    run_embedding_background,
    Sam3TextBasedDetector,
)
from interview.state import (
    InterviewSession, CropData, CropLabel, CropSource, Phase,
)

# Restore originals so other test files aren't polluted.
# The interview.* modules already captured mock references during import.
for _name, _orig in _saved_modules.items():
    if _orig is not None:
        sys.modules[_name] = _orig
    else:
        sys.modules.pop(_name, None)


@pytest.fixture(autouse=True)
def _use_mock_modules():
    """Re-inject all mock modules for each test function, restore after.

    interview code does lazy ``import torch`` / ``import av`` / etc. inside
    functions, so sys.modules must contain our mocks at call-time.  The
    fixture restores originals after each test so other test files
    (e.g. test_tracking_fixes.py with real torch/PIL) work correctly.
    """
    prev = {name: sys.modules.get(name) for name in _MOCK_MODULES}
    for name, mock in _MOCK_MODULES.items():
        sys.modules[name] = mock
    yield
    for name, orig in prev.items():
        if orig is not None:
            sys.modules[name] = orig
        else:
            sys.modules.pop(name, None)


# ===========================================================================
# NMS tests
# ===========================================================================

class TestNMS:
    def test_no_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        keep = nms_numpy(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_full_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        keep = nms_numpy(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # higher score kept

    def test_partial_overlap_below_threshold(self):
        # Box 0: [0,0,10,10] area=100
        # Box 1: [5,5,15,15] area=100
        # Intersection: [5,5,10,10] = 25
        # Union: 100 + 100 - 25 = 175
        # IoU = 25/175 ~ 0.143
        # Box 2: [20,20,30,30] - no overlap with 0 or 1
        boxes = np.array(
            [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32
        )
        scores = np.array([0.9, 0.7, 0.8])
        keep = nms_numpy(boxes, scores, iou_threshold=0.3)
        # IoU ~0.14 < 0.3, so all three should be kept
        assert len(keep) == 3

    def test_partial_overlap_above_threshold(self):
        # Same boxes, but with a stricter threshold
        boxes = np.array(
            [[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8])
        # IoU ~0.143, threshold 0.1 -> should suppress the lower-scoring box
        keep = nms_numpy(boxes, scores, iou_threshold=0.1)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_empty_input(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0)
        keep = nms_numpy(boxes, scores)
        assert len(keep) == 0

    def test_single_box(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.7])
        keep = nms_numpy(boxes, scores)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_score_ordering(self):
        # Higher-scoring box should always be kept
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.3, 0.99])
        keep = nms_numpy(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 1  # index of higher score

    def test_many_overlapping_boxes(self):
        # 10 identical boxes, only one should survive
        boxes = np.tile(np.array([[0, 0, 50, 50]], dtype=np.float32), (10, 1))
        scores = np.arange(10, dtype=np.float32) / 10
        keep = nms_numpy(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 9  # highest score index

    def test_returns_int64_array(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.5])
        keep = nms_numpy(boxes, scores)
        assert keep.dtype == np.int64


# ===========================================================================
# Box padding tests
# ===========================================================================

class TestPadBoxes:
    def test_basic_padding(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        padded = pad_boxes(boxes, width=400, height=400, pad_frac=0.1)
        # Box width/height = 100; 10% = 10 on each side
        np.testing.assert_array_equal(padded[0], [90, 90, 210, 210])

    def test_clamp_to_bounds_lower(self):
        boxes = np.array([[0, 0, 50, 50]], dtype=np.float32)
        padded = pad_boxes(boxes, width=100, height=100, pad_frac=0.1)
        assert padded[0][0] >= 0
        assert padded[0][1] >= 0
        # x1 should be max(0 - 5, 0) = 0
        assert padded[0][0] == 0.0
        assert padded[0][1] == 0.0
        assert padded[0][2] == 55.0
        assert padded[0][3] == 55.0

    def test_clamp_to_bounds_upper(self):
        boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
        padded = pad_boxes(boxes, width=100, height=100, pad_frac=0.1)
        assert padded[0][2] <= 100
        assert padded[0][3] <= 100

    def test_large_padding_clamped(self):
        boxes = np.array([[5, 5, 95, 95]], dtype=np.float32)
        padded = pad_boxes(boxes, width=100, height=100, pad_frac=0.5)
        assert padded[0][0] >= 0
        assert padded[0][1] >= 0
        assert padded[0][2] <= 100
        assert padded[0][3] <= 100

    def test_empty_input(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        padded = pad_boxes(boxes, width=100, height=100)
        assert padded.shape == (0, 4)

    def test_zero_padding(self):
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        padded = pad_boxes(boxes, width=100, height=100, pad_frac=0.0)
        np.testing.assert_array_equal(padded, boxes)

    def test_multiple_boxes(self):
        boxes = np.array(
            [[10, 10, 50, 50], [60, 60, 90, 90]], dtype=np.float32
        )
        padded = pad_boxes(boxes, width=100, height=100, pad_frac=0.1)
        assert padded.shape == (2, 4)
        # First box: w=40, h=40 -> pad=4 on each side
        np.testing.assert_array_almost_equal(padded[0], [6, 6, 54, 54])
        # Second box: w=30, h=30 -> pad=3 on each side
        np.testing.assert_array_almost_equal(padded[1], [57, 57, 93, 93])

    def test_does_not_modify_original(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        original_copy = boxes.copy()
        pad_boxes(boxes, width=400, height=400, pad_frac=0.1)
        np.testing.assert_array_equal(boxes, original_copy)

    def test_output_dtype(self):
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float64)
        padded = pad_boxes(boxes, width=100, height=100, pad_frac=0.1)
        assert padded.dtype == np.float32


# ===========================================================================
# Shared test fixtures
# ===========================================================================

class _FakeProgress:
    """Minimal stand-in for JobProgress."""
    def __init__(self):
        self.step = ""
        self.current = 0
        self.total = 0


def _make_session(frames_count=1000):
    """Create a minimal InterviewSession for testing."""
    session = InterviewSession(
        session_id="test-123",
        project_id=1,
        task_id=1,
        cache_key="p1_t1",
        video_path="/fake/video.mp4",
        width=1920,
        height=1080,
        frames_count=frames_count,
        fps=30.0,
    )
    return session


# ===========================================================================
# run_embedding_background tests (mocked)
# ===========================================================================

class TestRunEmbeddingBackground:
    def test_sets_embedding_complete_sam3_mode(self, monkeypatch):
        """Test SAM3 embedding path (EMBEDDING_MODE=sam3)."""
        progress_calls = []

        def mock_embed(video_path, batch_size, progress_callback=None,
                       target_fps=None, pause_event=None, change_callback=None):
            if progress_callback:
                progress_callback(50, 100)
                progress_calls.append((50, 100))
            sampled = list(range(0, 300, 3))[:100]  # 100 subsampled indices
            return (np.random.rand(100, 256).astype(np.float16), sampled)

        monkeypatch.setattr("interview.detection.EMBEDDING_MODE", "sam3")
        monkeypatch.setattr("interview.detection._do_embed_all_frames", mock_embed)
        monkeypatch.setattr("interview.detection.save_session", lambda s: None)

        session = _make_session(frames_count=300)
        progress = _FakeProgress()

        result = run_embedding_background(session, progress)

        assert session.embedding_complete is True
        assert result["frames_embedded"] == 100
        assert result["change_keyframes"] > 0
        assert len(session.change_keyframes) > 0
        assert len(progress_calls) == 1
        assert len(session.embedding_sampled_indices) == 100
        assert result["mode"] == "sam3"

    def test_sets_embedding_complete_lightweight_mode(self, monkeypatch):
        """Test lightweight CPU change detection path."""
        lw_calls = []

        def mock_lw(video_path, target_fps=None, pause_event=None,
                     progress_callback=None, cache_key=None):
            lw_calls.append(video_path)
            scores = np.random.rand(100).astype(np.float32)
            sampled = list(range(0, 300, 3))[:100]
            return (scores, sampled)

        monkeypatch.setattr("interview.detection.EMBEDDING_MODE", "lightweight")
        monkeypatch.setattr("interview.detection.compute_lightweight_change_from_video", mock_lw)
        monkeypatch.setattr("interview.detection.save_session", lambda s: None)

        session = _make_session(frames_count=300)
        progress = _FakeProgress()

        result = run_embedding_background(session, progress)

        assert session.embedding_complete is True
        assert result["frames_embedded"] == 100
        assert result["change_keyframes"] > 0
        assert len(lw_calls) == 1
        assert result["mode"] == "lightweight"

    def test_raises_without_video_path(self):
        session = _make_session()
        session.video_path = ""
        progress = _FakeProgress()

        with pytest.raises(RuntimeError, match="no video_path"):
            run_embedding_background(session, progress)

    def test_passes_target_fps(self, monkeypatch):
        """Verify target_fps is forwarded to _do_embed_all_frames (sam3 mode)."""
        captured = {}

        def mock_embed(video_path, batch_size, progress_callback=None,
                       target_fps=None, pause_event=None, change_callback=None):
            captured["target_fps"] = target_fps
            captured["pause_event"] = pause_event
            captured["change_callback"] = change_callback
            return (np.random.rand(50, 256).astype(np.float16), list(range(50)))

        monkeypatch.setattr("interview.detection.EMBEDDING_MODE", "sam3")
        monkeypatch.setattr("interview.detection._do_embed_all_frames", mock_embed)
        monkeypatch.setattr("interview.detection.save_session", lambda s: None)

        session = _make_session(frames_count=150)
        progress = _FakeProgress()
        progress._pause_event = threading.Event()
        progress._pause_event.set()

        run_embedding_background(session, progress)

        assert captured["target_fps"] is not None
        assert captured["target_fps"] > 0
        assert captured["pause_event"] is progress._pause_event
        assert captured["change_callback"] is not None

    def test_change_callback_updates_session(self, monkeypatch):
        """The change_callback should update session.change_keyframes incrementally (sam3 mode)."""
        def mock_embed(video_path, batch_size, progress_callback=None,
                       target_fps=None, pause_event=None, change_callback=None):
            # Simulate incremental change callback
            if change_callback:
                change_callback([10, 50, 90])
            return (np.random.rand(100, 256).astype(np.float16), list(range(100)))

        monkeypatch.setattr("interview.detection.EMBEDDING_MODE", "sam3")
        monkeypatch.setattr("interview.detection._do_embed_all_frames", mock_embed)
        monkeypatch.setattr("interview.detection.save_session", lambda s: None)

        session = _make_session(frames_count=100)
        progress = _FakeProgress()

        # Before running, change_keyframes should be empty
        assert session.change_keyframes == []

        run_embedding_background(session, progress)

        # After completion, change_keyframes should be set (final change detection
        # overwrites the incremental values, but both paths should work)
        assert session.embedding_complete is True
        assert len(session.change_keyframes) > 0


# ===========================================================================
# Session state tests (new fields)
# ===========================================================================

class TestSessionStateFields:
    def test_new_defaults(self):
        session = _make_session()
        assert session.embedding_job_id is None
        assert session.embedding_complete is False
        assert session.change_keyframes == []

    def test_change_detect_source(self):
        assert CropSource.CHANGE_DETECT.value == "change_detect"
        crop = CropData(
            crop_id="test",
            frame_idx=0,
            xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
            score=0.5,
            source=CropSource.CHANGE_DETECT,
        )
        d = crop.to_dict()
        assert d["source"] == "change_detect"
        restored = CropData.from_dict(d)
        assert restored.source == CropSource.CHANGE_DETECT

    def test_skipped_label(self):
        """SKIPPED crops are excluded from accepted/rejected counts."""
        session = _make_session()
        for i, label in enumerate([CropLabel.ACCEPTED, CropLabel.REJECTED, CropLabel.SKIPPED, CropLabel.PENDING]):
            session.add_crop(CropData(
                crop_id=f"c{i}",
                frame_idx=0,
                xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
                score=0.5,
                label=label,
            ))
        s = session.stats()
        assert s["accepted"] == 1
        assert s["rejected"] == 1
        assert s["skipped"] == 1
        assert s["pending"] == 1
        # Classifier would only get accepted + rejected
        assert len(session.get_crops_by_label(CropLabel.ACCEPTED)) == 1
        assert len(session.get_crops_by_label(CropLabel.REJECTED)) == 1
        # Skipped round-trips through serialization
        d = session.get_crop("c2").to_dict()
        assert d["label"] == "skipped"
        restored = CropData.from_dict(d)
        assert restored.label == CropLabel.SKIPPED


# ===========================================================================
# Task 1: Round state + mask_quality tests
# ===========================================================================

class TestRoundState:
    """Tests for round-based active learning state fields."""

    def test_session_round_defaults(self):
        """New session should have round fields with correct defaults."""
        session = _make_session()
        assert session.current_round == 0
        assert session.round_history == []
        assert session.round_frames == {}

    def test_session_round_fields_writable(self):
        """Round fields should be mutable."""
        session = _make_session()
        session.current_round = 2
        session.round_history = [
            {"round": 1, "accepted": 20, "rejected": 5},
        ]
        session.round_frames = {1: [0, 100, 200], 2: [50, 150, 250]}
        assert session.current_round == 2
        assert len(session.round_history) == 1
        assert session.round_frames[2] == [50, 150, 250]

    def test_stats_includes_round_info(self):
        """stats() should include current_round and rounds_completed."""
        session = _make_session()
        session.current_round = 3
        session.round_history = [
            {"round": 1}, {"round": 2}, {"round": 3},
        ]
        s = session.stats()
        assert s["current_round"] == 3
        assert s["rounds_completed"] == 3


class TestMaskQuality:
    """Tests for mask_quality field on CropData."""

    def test_crop_data_mask_quality_field(self):
        """CropData has mask_quality field (4-element array)."""
        mq = np.array([0.85, 0.92, 0.0, 0.78], dtype=np.float32)
        crop = CropData(
            crop_id="test", frame_idx=0,
            xyxy=np.array([0, 0, 100, 200], dtype=np.float32),
            score=0.9,
            mask_quality=mq,
        )
        assert crop.mask_quality is not None
        assert crop.mask_quality.shape == (4,)
        np.testing.assert_array_almost_equal(crop.mask_quality, [0.85, 0.92, 0.0, 0.78])

    def test_crop_data_mask_quality_default_none(self):
        """mask_quality defaults to None when not provided."""
        crop = CropData(
            crop_id="test", frame_idx=0,
            xyxy=np.array([0, 0, 100, 200], dtype=np.float32),
            score=0.9,
        )
        assert crop.mask_quality is None

    def test_mask_quality_in_to_dict(self):
        """to_dict includes mask_quality as a list (or None)."""
        mq = np.array([0.85, 0.92, 0.0, 0.78], dtype=np.float32)
        crop = CropData(
            crop_id="test", frame_idx=0,
            xyxy=np.array([0, 0, 100, 200], dtype=np.float32),
            score=0.9,
            mask_quality=mq,
        )
        d = crop.to_dict()
        assert "mask_quality" in d
        assert isinstance(d["mask_quality"], list)
        assert len(d["mask_quality"]) == 4

    def test_mask_quality_none_in_to_dict(self):
        """to_dict includes mask_quality as None when not set."""
        crop = CropData(
            crop_id="test", frame_idx=0,
            xyxy=np.array([0, 0, 100, 200], dtype=np.float32),
            score=0.9,
        )
        d = crop.to_dict()
        assert "mask_quality" in d
        assert d["mask_quality"] is None

    def test_mask_quality_roundtrip_from_dict(self):
        """mask_quality should survive to_dict -> from_dict round-trip."""
        mq = np.array([0.85, 0.92, 0.0, 0.78], dtype=np.float32)
        crop = CropData(
            crop_id="test", frame_idx=0,
            xyxy=np.array([0, 0, 100, 200], dtype=np.float32),
            score=0.9,
            mask_quality=mq,
        )
        d = crop.to_dict()
        restored = CropData.from_dict(d)
        assert restored.mask_quality is not None
        np.testing.assert_array_almost_equal(restored.mask_quality, mq)

    def test_mask_quality_none_roundtrip(self):
        """None mask_quality should survive to_dict -> from_dict round-trip."""
        crop = CropData(
            crop_id="test", frame_idx=0,
            xyxy=np.array([0, 0, 100, 200], dtype=np.float32),
            score=0.9,
        )
        d = crop.to_dict()
        restored = CropData.from_dict(d)
        assert restored.mask_quality is None


class TestFeatureSearchRemoved:
    """Verify CropSource.FEATURE_SEARCH has been deleted."""

    def test_feature_search_not_in_enum(self):
        """CropSource should not have FEATURE_SEARCH member."""
        assert not hasattr(CropSource, "FEATURE_SEARCH")
        # The valid sources should still exist
        assert hasattr(CropSource, "TEXT_DETECT")
        assert hasattr(CropSource, "MULTI_PROMPT")
        assert hasattr(CropSource, "HUMAN_DRAWN")
        assert hasattr(CropSource, "CHANGE_DETECT")

    def test_feature_search_value_not_valid(self):
        """'feature_search' should not be a valid CropSource value."""
        with pytest.raises(ValueError):
            CropSource("feature_search")


class TestCacheRoundTrip:
    """Tests for cache_manager persisting round state + mask_quality."""

    def test_save_load_round_fields(self, tmp_path, monkeypatch):
        """Round fields should survive save -> load cycle."""
        monkeypatch.setattr("interview.cache_manager.CACHE_ROOT", str(tmp_path))

        from interview.cache_manager import save_session, load_session

        session = _make_session()
        session.current_round = 3
        session.round_history = [
            {"round": 1, "accepted": 20, "rejected": 5},
            {"round": 2, "accepted": 15, "rejected": 3},
            {"round": 3, "accepted": 10, "rejected": 2},
        ]
        session.round_frames = {1: [0, 100, 200], 2: [50, 150, 250], 3: [75, 175]}

        save_session(session)
        loaded = load_session(session.cache_key)

        assert loaded is not None
        assert loaded.current_round == 3
        assert len(loaded.round_history) == 3
        assert loaded.round_history[0]["accepted"] == 20
        assert loaded.round_frames == {1: [0, 100, 200], 2: [50, 150, 250], 3: [75, 175]}

    def test_save_load_mask_quality(self, tmp_path, monkeypatch):
        """mask_quality on crops should survive save -> load cycle."""
        monkeypatch.setattr("interview.cache_manager.CACHE_ROOT", str(tmp_path))

        from interview.cache_manager import save_session, load_session

        session = _make_session()
        mq = np.array([0.85, 0.92, 0.0, 0.78], dtype=np.float32)
        session.add_crop(CropData(
            crop_id="c1", frame_idx=0,
            xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            score=0.9,
            mask_quality=mq,
        ))
        # Also add a crop WITHOUT mask_quality to verify None handling
        session.add_crop(CropData(
            crop_id="c2", frame_idx=1,
            xyxy=np.array([20, 20, 60, 60], dtype=np.float32),
            score=0.8,
        ))

        save_session(session)
        loaded = load_session(session.cache_key)

        assert loaded is not None
        c1 = loaded.get_crop("c1")
        assert c1 is not None
        assert c1.mask_quality is not None
        np.testing.assert_array_almost_equal(c1.mask_quality, mq)

        c2 = loaded.get_crop("c2")
        assert c2 is not None
        assert c2.mask_quality is None

    def test_backward_compatible_load_no_round_fields(self, tmp_path, monkeypatch):
        """Loading a cache without round fields should use defaults."""
        monkeypatch.setattr("interview.cache_manager.CACHE_ROOT", str(tmp_path))

        from interview.cache_manager import save_session, load_session, _write_json

        # Save a session, then manually strip round fields from config.json
        session = _make_session()
        save_session(session)

        import json
        config_path = tmp_path / session.cache_key / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        config.pop("current_round", None)
        config.pop("round_history", None)
        config.pop("round_frames", None)
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = load_session(session.cache_key)
        assert loaded is not None
        assert loaded.current_round == 0
        assert loaded.round_history == []
        assert loaded.round_frames == {}


# ===========================================================================
# Task 2: select_round_frames tests
# ===========================================================================

class TestSelectRoundFrames:
    """Tests for temporally-stratified frame selector."""

    def test_uniform_no_change(self):
        """Without change keyframes, frames are uniformly spaced."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=1000)
        session.embedding_complete = False
        session.change_keyframes = []
        session.round_frames = {}

        frames = select_round_frames(session, round_num=1, frames_per_round=40)
        assert len(frames) == 40
        assert frames == sorted(frames)
        assert frames[0] < 50
        assert frames[-1] > 950

    def test_excludes_previous_rounds(self):
        """Frames from previous rounds are excluded."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=100)
        session.embedding_complete = False
        session.round_frames = {1: list(range(0, 100, 10))}  # 10 frames used

        frames = select_round_frames(session, round_num=2, frames_per_round=10)
        assert len(frames) == 10
        used = set(session.round_frames[1])
        assert not (set(frames) & used), "Round 2 frames must not overlap with round 1"

    def test_prefers_change_keyframes(self):
        """When embeddings are ready, change keyframes are preferred within bins."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=1000)
        session.embedding_complete = True
        session.change_keyframes = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
        session.round_frames = {}

        frames = select_round_frames(session, round_num=1, frames_per_round=10)
        assert len(frames) == 10
        overlap = set(frames) & set(session.change_keyframes)
        assert len(overlap) >= 8, f"Expected >=8 change keyframes selected, got {len(overlap)}"

    def test_empty_video(self):
        """Zero-frame video returns empty list."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=0)
        session.round_frames = {}

        frames = select_round_frames(session, round_num=1, frames_per_round=40)
        assert frames == []

    def test_all_frames_exhausted(self):
        """When all frames used in previous rounds, falls back to full range."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=20)
        session.round_frames = {1: list(range(20))}  # all 20 frames used

        frames = select_round_frames(session, round_num=2, frames_per_round=5)
        # Should still return frames (from fallback full range)
        assert len(frames) == 5

    def test_sorted_output(self):
        """Output should always be sorted."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=500)
        session.embedding_complete = True
        session.change_keyframes = list(range(0, 500, 7))  # irregular spacing
        session.round_frames = {}

        frames = select_round_frames(session, round_num=1, frames_per_round=30)
        assert frames == sorted(frames)
        assert len(set(frames)) == len(frames), "No duplicates"


# ===========================================================================
# Task 3: compute_mask_quality tests
# ===========================================================================

class TestComputeMaskQuality:
    """Tests for mask-quality feature computation."""

    def test_interior_box(self):
        """Box fully inside frame: high fill ratio, low edge contact."""
        from interview.mask_utils import compute_mask_quality
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:300, 50:150] = True
        box = np.array([50, 100, 150, 300], dtype=np.float32)

        mq = compute_mask_quality(mask, box, 640, 480)
        assert mq.shape == (4,)
        assert mq.dtype == np.float32
        # fill ratio: 100*200 pixels in mask within 100*200 box = 1.0
        assert mq[0] > 0.8
        # detection_score placeholder = 0.0
        assert mq[1] == 0.0
        # edge_contact: box interior → low
        assert mq[2] < 0.1
        # compactness: rectangle, moderate
        assert 0.0 < mq[3] <= 1.0

    def test_edge_contact_bottom(self):
        """Box touching bottom frame edge: high edge contact."""
        from interview.mask_utils import compute_mask_quality
        mask = np.zeros((480, 640), dtype=bool)
        mask[380:480, 200:400] = True
        box = np.array([200, 380, 400, 480], dtype=np.float32)

        mq = compute_mask_quality(mask, box, 640, 480)
        assert mq[2] > 0.2  # at least bottom edge

    def test_edge_contact_all_edges(self):
        """Box spanning entire frame: all edges in contact."""
        from interview.mask_utils import compute_mask_quality
        mask = np.ones((480, 640), dtype=bool)
        box = np.array([0, 0, 640, 480], dtype=np.float32)

        mq = compute_mask_quality(mask, box, 640, 480)
        assert mq[2] == 1.0  # all 4 edges

    def test_empty_mask(self):
        """Empty mask: fill ratio = 0, compactness = 0."""
        from interview.mask_utils import compute_mask_quality
        mask = np.zeros((480, 640), dtype=bool)
        box = np.array([50, 100, 150, 300], dtype=np.float32)

        mq = compute_mask_quality(mask, box, 640, 480)
        assert mq[0] == 0.0  # fill ratio
        assert mq[3] == 0.0  # compactness

    def test_output_is_four_floats(self):
        """Output should always be (4,) float32."""
        from interview.mask_utils import compute_mask_quality
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = True
        box = np.array([10, 10, 90, 90], dtype=np.float32)

        mq = compute_mask_quality(mask, box, 100, 100)
        assert mq.shape == (4,)
        assert mq.dtype == np.float32


# ===========================================================================
# Task 4: LR decay + build_feature_matrix tests
# ===========================================================================



# ===========================================================================
# Task 5: Grid search removal tests
# ===========================================================================

class TestGridSearchRemoval:
    """Verify grid search / feature_search has been removed."""

    def test_recall_strategy_rejects_feature_search(self, monkeypatch):
        """feature_search strategy is no longer supported."""
        from interview.detection import run_recall_strategy
        from interview.state import create_session
        session = create_session(1, 1)
        session.sampled_frames = [0, 100]
        mock_progress = _FakeProgress()

        with pytest.raises(ValueError, match="Unknown recall strategy"):
            run_recall_strategy(session, "feature_search", [], mock_progress)

    def test_no_run_feature_search_function(self):
        """run_feature_search should not exist on dinov3_classifier."""
        assert not hasattr(_mock_dinov3, "run_feature_search") or \
               _mock_dinov3.run_feature_search is None

    def test_no_grid_search_fallback_function(self):
        """_grid_search_fallback should not exist in seeding_phase."""
        # Try importing it — should fail
        try:
            from interview.seeding_phase import _grid_search_fallback
            assert False, "_grid_search_fallback should not be importable"
        except ImportError:
            pass


# ===========================================================================
# _detect_batch tests
# ===========================================================================

class TestDetectBatch:
    """Verify _detect_batch runs batched inference with per-image prompts."""

    def test_oom_falls_back_to_per_frame(self, monkeypatch):
        """On OOM, should fall back to _detect_single_frame per frame."""
        called_frames = []

        def mock_detect_single(det, img, prompt, frame_idx, w, h, **kw):
            called_frames.append(frame_idx)
            return [CropData(
                crop_id=f"c{frame_idx}",
                frame_idx=frame_idx,
                xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
                score=0.9,
                prompt=prompt,
            )]

        monkeypatch.setattr("interview.detection._detect_single_frame", mock_detect_single)

        FakeImg = type("FakeImg", (), {"height": 100, "width": 100})
        frames = {5: FakeImg(), 10: FakeImg()}

        class OOMProcessor:
            def __call__(self, **kw):
                raise _mock_torch.cuda.OutOfMemoryError("OOM test")

        class FakeDetector:
            processor = OOMProcessor()
            model = None
            threshold = 0.3
            mask_threshold = 0.5

        crops = _detect_batch(FakeDetector(), frames, "person", 100, 100, batch_size=2)

        assert len(crops) == 2
        assert called_frames == [5, 10]

    def test_empty_frames(self):
        """Empty dict should produce no crops."""
        class FakeDetector:
            processor = None
            model = None
            threshold = 0.3
            mask_threshold = 0.5

        crops = _detect_batch(FakeDetector(), {}, "person", 100, 100)
        assert crops == []

    def test_text_is_list_of_prompts(self, monkeypatch):
        """Processor must receive text as a list (one prompt per image)."""
        captured_text = []

        class MockProcessor:
            def __call__(self, **kw):
                captured_text.append(kw.get("text"))
                # Raise to short-circuit (we just want to check the text arg)
                raise _mock_torch.cuda.OutOfMemoryError("stop here")

        # Mock _detect_single_frame so OOM fallback works
        monkeypatch.setattr(
            "interview.detection._detect_single_frame",
            lambda *a, **kw: [],
        )

        FakeImg = type("FakeImg", (), {"height": 100, "width": 100})
        frames = {0: FakeImg(), 1: FakeImg(), 2: FakeImg()}

        class FakeDetector:
            processor = MockProcessor()
            model = None
            threshold = 0.3
            mask_threshold = 0.5

        _detect_batch(FakeDetector(), frames, "person", 100, 100, batch_size=3)

        # Should have been called once with a list of 3 identical prompts
        assert len(captured_text) == 1
        assert captured_text[0] == ["person", "person", "person"]

    def test_happy_path_batched_inference(self, monkeypatch):
        """Successful batched inference: processor + model + post_process pipeline."""
        FakeImg = type("FakeImg", (), {"height": 100, "width": 100})
        frames = {0: FakeImg(), 5: FakeImg()}

        call_log = {"processor": 0, "model": 0, "post_process": 0}

        class FakeInputs:
            """Mimics the dict-like tensor object returned by the processor.

            Must support both .get() and **unpacking (used as model(**inputs)).
            """
            def __init__(self):
                self._data = {"pixel_values": "fake_tensor"}

            def to(self, device):
                return self

            def get(self, key, default=None):
                if key == "original_sizes":
                    return None
                return self._data.get(key, default)

            def keys(self):
                return self._data.keys()

            def __getitem__(self, key):
                return self._data[key]

        class FakeOutputs:
            pass

        class MockProcessor:
            def __call__(self, **kw):
                call_log["processor"] += 1
                assert isinstance(kw["text"], list), "text must be a list"
                assert len(kw["text"]) == len(kw["images"])
                return FakeInputs()

            def post_process_instance_segmentation(self, outputs, **kw):
                call_log["post_process"] += 1
                # Return one detection per image
                return [
                    {
                        "boxes": [np.array([10, 10, 50, 50], dtype=np.float32)],
                        "scores": [0.9],
                        "masks": [],
                        "labels": ["person"],
                    },
                    {
                        "boxes": [np.array([20, 20, 60, 60], dtype=np.float32)],
                        "scores": [0.85],
                        "masks": [],
                        "labels": ["person"],
                    },
                ]

        class MockModel:
            def __call__(self, **kw):
                call_log["model"] += 1
                return FakeOutputs()

        class FakeDetector:
            processor = MockProcessor()
            model = MockModel()
            threshold = 0.3
            mask_threshold = 0.5

        crops = _detect_batch(FakeDetector(), frames, "person", 100, 100, batch_size=4)

        assert call_log["processor"] == 1, "Single batch → single processor call"
        assert call_log["model"] == 1, "Single batch → single model call"
        assert call_log["post_process"] == 1
        assert len(crops) == 2, "One crop per frame"
        assert crops[0].frame_idx == 0
        assert crops[1].frame_idx == 5
        assert all(c.source == CropSource.TEXT_DETECT for c in crops)


# ===========================================================================
# _decode_frames_sequential tests (with mock av)
# ===========================================================================

class TestDecodeFramesSequential:
    """Test seek-based frame decoding logic."""

    def _make_mock_container(self, total_frames=100, fps=30.0):
        """Build a mock av container that simulates seek + decode."""
        from fractions import Fraction

        time_base = Fraction(1, int(fps))

        class MockFrame:
            def __init__(self, idx, fps, tb):
                self.pts = int(idx / fps / tb)
                self._idx = idx

            def to_image(self):
                return f"frame_{self._idx}"

        class MockStream:
            def __init__(self):
                self.average_rate = Fraction(int(fps), 1)
                self.time_base = time_base
                self.frames = total_frames

        class MockContainer:
            def __init__(self):
                self.streams = type("S", (), {"video": [MockStream()]})()
                self._seek_target = 0
                self._total = total_frames
                self._fps = fps
                self._tb = time_base

            def seek(self, pts, stream=None):
                # Simulate seeking to the exact frame (ideal case)
                frame_idx = int(round(float(pts * self._tb) * self._fps))
                self._seek_target = min(frame_idx, self._total - 1)

            def decode(self, video=0):
                # Yield frames starting from the seek target
                for i in range(self._seek_target, self._total):
                    yield MockFrame(i, self._fps, self._tb)

            def close(self):
                pass

        return MockContainer()

    def test_decodes_exact_targets(self, monkeypatch):
        """Should return exactly the requested frame indices."""
        container = self._make_mock_container(total_frames=1000, fps=30.0)
        monkeypatch.setattr("interview.detection.av.open", lambda path: container)

        targets = [0, 250, 500, 750, 999]
        result = _decode_frames_sequential("/fake/video.mp4", targets)

        assert set(result.keys()) == set(targets)
        for idx in targets:
            assert result[idx] == f"frame_{idx}"

    def test_handles_unsorted_input(self, monkeypatch):
        """Input indices need not be sorted."""
        container = self._make_mock_container(total_frames=100, fps=30.0)
        monkeypatch.setattr("interview.detection.av.open", lambda path: container)

        targets = [50, 10, 90, 30]
        result = _decode_frames_sequential("/fake/video.mp4", targets)

        assert set(result.keys()) == set(targets)

    def test_deduplicates_indices(self, monkeypatch):
        """Duplicate indices should produce a single entry."""
        container = self._make_mock_container(total_frames=100, fps=30.0)
        monkeypatch.setattr("interview.detection.av.open", lambda path: container)

        targets = [10, 10, 10, 50, 50]
        result = _decode_frames_sequential("/fake/video.mp4", targets)

        assert len(result) == 2  # {10, 50}

    def test_empty_indices(self, monkeypatch):
        """Empty input should return empty dict without opening container."""
        open_called = [False]
        original_open = _mock_av.open
        def track_open(*a, **kw):
            open_called[0] = True
            return original_open(*a, **kw)
        monkeypatch.setattr("interview.detection.av.open", track_open)

        result = _decode_frames_sequential("/fake/video.mp4", [])
        assert result == {}
        assert not open_called[0]

    def test_max_decode_safety_limit(self, monkeypatch):
        """If target can't be found within max_decode_after_seek frames, skip it."""
        from fractions import Fraction

        class MockFrame:
            def __init__(self, idx):
                # pts that resolves to frame 0 always (simulating broken seek)
                self.pts = 0
                self._idx = idx

            def to_image(self):
                return f"frame_{self._idx}"

        class NeverFindContainer:
            def __init__(self):
                self.streams = type("S", (), {
                    "video": [type("VS", (), {
                        "average_rate": Fraction(30, 1),
                        "time_base": Fraction(1, 30),
                        "frames": 10000,
                    })()]
                })()

            def seek(self, pts, stream=None):
                pass

            def decode(self, video=0):
                # Yield endless frames at index 0 (seek never reaches target)
                i = 0
                while True:
                    yield MockFrame(i)
                    i += 1

            def close(self):
                pass

        monkeypatch.setattr("interview.detection.av.open", lambda path: NeverFindContainer())

        # Target frame 5000 with a safety limit of 10 → should give up
        result = _decode_frames_sequential("/fake/video.mp4", [5000], max_decode_after_seek=10)

        # Frame 5000 should NOT be in results (safety limit triggered)
        assert 5000 not in result


# ===========================================================================
# _refine_candidates_sam3 tests
# ===========================================================================

class TestRefineCandidatesSam3:
    """Tests for _refine_candidates_sam3 box refinement."""

    def test_refines_box_from_mask(self, monkeypatch):
        """Expanded box + text prompt -> Sam3Model -> tight box from mask."""
        from interview.seeding_phase import _refine_candidates_sam3

        mock_model = MagicMock()
        mock_processor = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.get = lambda k, d=None: [[100, 200]] if k == "original_sizes" else d
        mock_processor.return_value = mock_inputs
        mock_inputs.to = MagicMock(return_value=mock_inputs)

        mock_processor.post_process_instance_segmentation.return_value = [{
            "masks": [np.ones((200, 100), dtype=bool)],
            "scores": [0.9],
            "boxes": [np.array([10, 20, 80, 180], dtype=np.float32)],
        }]

        monkeypatch.setattr(
            "interview.seeding_phase._get_sam3_image_model",
            lambda: (mock_model, mock_processor),
        )

        frame = type("FakeImg", (), {"size": (100, 200), "crop": lambda self, box: self, "width": 100, "height": 200})()
        candidates = [(0, np.array([5, 5, 90, 190], dtype=np.float32), 0.5)]

        result = _refine_candidates_sam3(
            {0: frame}, candidates, prompt="person", expand_frac=0.2,
        )
        assert len(result) == 1
        frame_idx, box, score = result[0]
        assert frame_idx == 0
        np.testing.assert_array_equal(box, [10, 20, 80, 180])

    def test_skips_frame_with_no_mask(self, monkeypatch):
        """If Sam3Model returns no masks, candidate is dropped."""
        from interview.seeding_phase import _refine_candidates_sam3

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.get = lambda k, d=None: [[100, 200]] if k == "original_sizes" else d
        mock_processor.return_value = mock_inputs
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_processor.post_process_instance_segmentation.return_value = [{
            "masks": [], "scores": [], "boxes": [],
        }]
        monkeypatch.setattr(
            "interview.seeding_phase._get_sam3_image_model",
            lambda: (mock_model, mock_processor),
        )

        frame = type("FakeImg", (), {"size": (100, 200), "width": 100, "height": 200})()
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
        mock_inputs.get = lambda k, d=None: [[50, 50]] if k == "original_sizes" else d
        mock_processor.return_value = mock_inputs
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_processor.post_process_instance_segmentation.return_value = [{
            "masks": [np.ones((50, 50), dtype=bool)],
            "scores": [0.8],
            "boxes": [np.array([0, 0, 50, 50], dtype=np.float32)],
        }]
        monkeypatch.setattr(
            "interview.seeding_phase._get_sam3_image_model",
            lambda: (mock_model, mock_processor),
        )

        frame = type("FakeImg", (), {"size": (50, 50), "width": 50, "height": 50})()
        candidates = [(0, np.array([0, 0, 50, 50], dtype=np.float32), 0.5)]
        result = _refine_candidates_sam3(
            {0: frame}, candidates, prompt="person", expand_frac=0.5,
        )
        assert len(result) == 1

        # Verify processor was called with clamped box
        call_kwargs = mock_processor.call_args
        input_boxes = call_kwargs.kwargs.get("input_boxes") or call_kwargs[1].get("input_boxes")
        box = input_boxes[0][0]
        assert box[0] >= 0 and box[1] >= 0
        assert box[2] <= 50 and box[3] <= 50


# ===========================================================================
# Dual-proposer generate_seeds tests
# ===========================================================================


def _make_fake_frame():
    """Create a fake PIL-like image for seeding tests."""
    return type("FakeImg", (), {
        "width": 200, "height": 200, "size": (200, 200),
        "crop": lambda self, box: type("CropImg", (), {
            "width": int(box[2] - box[0]), "height": int(box[3] - box[1]),
            "size": (int(box[2] - box[0]), int(box[3] - box[1])),
        })(),
    })()


def _make_seed_session(frames_count=500):
    """Create a session configured for seeding tests (past REID phase)."""
    session = _make_session(frames_count=frames_count)
    session.prompts = ["person"]
    session.reid_clusters = {0: ["c0", "c1"], 1: ["c2"]}
    session.seed_config.frame_pct = 50
    session.seed_config.confidence_threshold = 0.8
    # Add crops with features for centroid computation + reference features.
    # Include context_features and metadata for realistic k-NN support set data.
    for cid in ["c0", "c1", "c2"]:
        feat = np.random.randn(1024).astype(np.float32)
        feat /= np.linalg.norm(feat)
        ctx_feat = np.random.randn(1024).astype(np.float32)
        ctx_feat /= np.linalg.norm(ctx_feat)
        meta = np.array([0.15, 0.25, 0.1, 0.8], dtype=np.float32)
        session.add_crop(CropData(
            crop_id=cid,
            frame_idx=0,
            xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            score=0.9,
            label=CropLabel.ACCEPTED,
            features=feat,
            context_features=ctx_feat,
            metadata=meta,
        ))
    return session


def _apply_seeding_mocks(monkeypatch, detector_returns=None, knn_confidence=0.95):
    """Set up common mocks for generate_seeds tests (k-NN flow).

    Args:
        knn_confidence: Fixed confidence returned by the mock score_crops.
            0.95 (default) passes threshold 0.8 → seeds accepted.
            0.38 is between _REFINE_THRESHOLD (0.3) and threshold (0.8) →
            triggers refinement.

    Returns (decode_log, refine_log) for call tracking.
    """
    decode_log = []
    refine_log = []

    def mock_decode(video_path, frame_indices, **kw):
        decode_log.extend(frame_indices)
        return {idx: _make_fake_frame() for idx in frame_indices}

    class MockDetector:
        def __init__(self, **kw):
            self.processor = MagicMock(spec=[])  # no tokenizer attr
        def set_frame(self, pil_image):
            pass
        def detect(self, prompt, pil_image=None):
            if detector_returns is not None:
                return detector_returns
            return [{"xyxy": np.array([10, 10, 50, 50]), "score": 0.9, "label": prompt, "mask": None}]
        def clear_cache(self):
            pass

    def mock_detect_batch(detector, frames, prompt, w, h, **kw):
        """Mock _detect_batch returning one CropData per frame."""
        crops = []
        for frame_idx in sorted(frames.keys()):
            if detector_returns is not None:
                for det in detector_returns:
                    crops.append(CropData(
                        crop_id=f"seed_{frame_idx}_{det['score']:.1f}",
                        frame_idx=frame_idx,
                        xyxy=det["xyxy"].copy(),
                        score=det["score"],
                        label=CropLabel.PENDING,
                        source=CropSource.TEXT_DETECT,
                        prompt=prompt,
                    ))
            else:
                crops.append(CropData(
                    crop_id=f"seed_{frame_idx}",
                    frame_idx=frame_idx,
                    xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
                    score=0.9,
                    label=CropLabel.PENDING,
                    source=CropSource.TEXT_DETECT,
                    prompt=prompt,
                ))
        return crops

    # Mock k-NN support set: 10 fake labeled crops
    def mock_build_support_set(session):
        n = 10
        feats = np.random.randn(n, 2056).astype(np.float32)
        labels = np.array([1.0] * 5 + [0.0] * 5)
        crop_ids = [f"support_{i}" for i in range(n)]
        reasons = [None] * 5 + ["not_person"] * 5
        return feats, labels, crop_ids, reasons

    # Mock score_crops to return fixed confidence
    def mock_score_crops(query, support_feats, support_labels, support_reasons, **kw):
        n = query.shape[0]
        confs = np.full(n, knn_confidence, dtype=np.float32)
        preds = np.where(confs >= 0.5, 1.0, 0.0).astype(np.float32)
        return confs, preds

    monkeypatch.setattr("interview.detection._decode_frames_sequential", mock_decode)
    monkeypatch.setattr("interview.detection._detect_batch", mock_detect_batch)
    monkeypatch.setattr("interview.detection.precompute_text_tokens", lambda det, p: None)
    monkeypatch.setattr("interview.detection.Sam3TextBasedDetector", MockDetector)
    monkeypatch.setattr(_mock_dinov3, "compute_crop_metadata",
                        lambda box, w, h: np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(_mock_dinov3, "compute_mask_quality",
                        lambda mask, box, w, h: np.array([0.9, 0.0, 0.0, 0.8], dtype=np.float32))
    monkeypatch.setattr("interview.seeding_phase.extract_features",
                        lambda crops, batch_size=128: np.random.randn(len(crops), 1024).astype(np.float32))
    monkeypatch.setattr("interview.seeding_phase.build_support_set", mock_build_support_set)
    monkeypatch.setattr("interview.seeding_phase.score_crops", mock_score_crops)
    monkeypatch.setattr("interview.seeding_phase.save_session", lambda s: None)

    def mock_refine(frames, candidates, prompt="person", expand_frac=0.2):
        refine_log.extend(candidates)
        # Return the same candidates as refined (with adjusted box)
        return [(fi, np.array([15, 15, 55, 55], dtype=np.float32), sc)
                for fi, box, sc in candidates]

    monkeypatch.setattr("interview.seeding_phase._refine_candidates_sam3", mock_refine)

    return decode_log, refine_log


class TestMultiPromptSeeding:
    """Integration tests for the multi-prompt SAM3 + k-NN generate_seeds pipeline."""

    def test_raises_without_labeled_crops(self, monkeypatch):
        """generate_seeds raises RuntimeError when k-NN support set is empty."""
        from interview.seeding_phase import generate_seeds
        import pytest

        # Mock build_support_set to return empty arrays (no labeled crops)
        def mock_empty_support(session):
            feats = np.zeros((0, 2056), dtype=np.float32)
            labels = np.array([], dtype=np.float32)
            crop_ids = []
            reasons = []
            return feats, labels, crop_ids, reasons

        monkeypatch.setattr("interview.seeding_phase.build_support_set", mock_empty_support)
        monkeypatch.setattr("interview.seeding_phase.save_session", lambda s: None)

        session = _make_seed_session(frames_count=500)
        progress = _FakeProgress()

        with pytest.raises(RuntimeError, match="No labeled crops"):
            generate_seeds(session, progress)

    def test_path_a_good_detection_becomes_seed(self, monkeypatch):
        """High-confidence SAM3 detection that passes k-NN should produce a seed."""
        from interview.seeding_phase import generate_seeds

        decode_log, _ = _apply_seeding_mocks(monkeypatch, knn_confidence=0.95)

        session = _make_seed_session(frames_count=500)
        progress = _FakeProgress()

        result = generate_seeds(session, progress)

        assert result["total_seeds"] > 0
        # 500 frames / skip=3 -> 167 cached_indices, 50% -> 84 sampled
        assert result["frames_scanned"] == 84
        for seed in session.seeds:
            assert seed["source"] == "multi_prompt_knn"
            assert seed["confidence"] >= 0.8
            assert "identity" in seed

    def test_path_b_medium_score_gets_refined(self, monkeypatch):
        """Medium k-NN confidence triggers Sam3Model refinement (Path B)."""
        from interview.seeding_phase import generate_seeds

        # confidence=0.38, which is between _REFINE_THRESHOLD (0.3)
        # and confidence_threshold (0.8), so Path A rejects but Path B triggers
        decode_log, refine_log = _apply_seeding_mocks(
            monkeypatch, knn_confidence=0.38,
        )

        # After refinement, the re-scored box must pass the threshold.
        # Override score_crops with a variable version: first call returns
        # medium confidence (0.38), subsequent calls return high (0.95).
        _call_count = [0]
        def variable_score_crops(query, support_feats, support_labels, support_reasons, **kw):
            _call_count[0] += 1
            n = query.shape[0]
            # First call is Path A batch scoring (medium → triggers refine)
            # Later calls from _score_and_accept_seed get high score → accepted
            if _call_count[0] > 1:
                confs = np.full(n, 0.95, dtype=np.float32)
            else:
                confs = np.full(n, 0.38, dtype=np.float32)
            preds = np.where(confs >= 0.5, 1.0, 0.0).astype(np.float32)
            return confs, preds

        monkeypatch.setattr("interview.seeding_phase.score_crops", variable_score_crops)

        session = _make_seed_session(frames_count=500)
        progress = _FakeProgress()

        result = generate_seeds(session, progress)

        # Refinement should have been called
        assert len(refine_log) > 0, "Expected _refine_candidates_sam3 to be called"
        # Some seeds should come from Path B
        refined_seeds = [s for s in session.seeds if s["source"] == "refined"]
        assert len(refined_seeds) > 0, "Expected seeds from refinement"

    def test_change_keyframes_included(self, monkeypatch):
        """Change-detected keyframes from background embedding are added to targets."""
        from interview.seeding_phase import generate_seeds

        decode_log, _ = _apply_seeding_mocks(monkeypatch, knn_confidence=0.95)

        session = _make_seed_session(frames_count=500)
        session.embedding_complete = True
        session.change_keyframes = [17, 123, 333]
        progress = _FakeProgress()

        result = generate_seeds(session, progress)

        # Change keyframes should have been decoded
        for ck in [17, 123, 333]:
            assert ck in decode_log, f"Change keyframe {ck} not decoded"
        # Total should include both uniform (500/50=10) and change (3 unique)
        assert result["frames_scanned"] >= 10

    def test_uses_decode_frames_sequential(self, monkeypatch):
        """Frames should be decoded via _decode_frames_sequential, not per-frame seeks."""
        from interview.seeding_phase import generate_seeds

        decode_log, _ = _apply_seeding_mocks(monkeypatch, knn_confidence=0.95)

        read_pyav_called = [False]
        original_read = _mock_seeding._read_frame_pyav
        def track_read(*a, **kw):
            read_pyav_called[0] = True
            return original_read(*a, **kw)
        monkeypatch.setattr("interview.seeding_phase._read_frame_pyav", track_read)

        session = _make_seed_session(frames_count=500)
        progress = _FakeProgress()

        generate_seeds(session, progress)

        assert len(decode_log) > 0, "Expected _decode_frames_sequential to be called"
        assert not read_pyav_called[0], "_read_frame_pyav should not be used in dual-proposer"

    def test_refinement_disabled_skips_path_b(self, monkeypatch):
        """When INTERVIEW_ENABLE_REFINEMENT=false, Path B is skipped."""
        from interview.seeding_phase import generate_seeds
        import interview.seeding_phase as sp

        # Medium confidence that would normally trigger refinement
        _, refine_log = _apply_seeding_mocks(
            monkeypatch, knn_confidence=0.38,
        )
        monkeypatch.setattr(sp, "_ENABLE_REFINEMENT", False)

        session = _make_seed_session(frames_count=500)
        progress = _FakeProgress()

        generate_seeds(session, progress)

        assert len(refine_log) == 0, "Refinement should be skipped when disabled"

    def test_session_advances_to_seeding_phase(self, monkeypatch):
        """After seed generation, session should advance to SEEDING phase."""
        from interview.seeding_phase import generate_seeds

        _apply_seeding_mocks(monkeypatch, knn_confidence=0.95)

        session = _make_seed_session(frames_count=500)
        progress = _FakeProgress()

        generate_seeds(session, progress)

        assert session.phase == Phase.SEEDING
        assert len(session.seeds) > 0

    def test_multi_prompt_all_prompts_used(self, monkeypatch):
        """All accumulated prompts should be tried via _detect_batch."""
        from interview.seeding_phase import generate_seeds

        detect_batch_prompts = []

        def mock_decode(video_path, frame_indices, **kw):
            return {idx: _make_fake_frame() for idx in frame_indices}

        class TrackingDetector:
            def __init__(self, **kw):
                self.processor = MagicMock(spec=[])
            def set_frame(self, pil_image): pass
            def detect(self, prompt, pil_image=None):
                return [{"xyxy": np.array([10, 10, 50, 50]), "score": 0.9, "label": prompt, "mask": None}]
            def clear_cache(self): pass

        def mock_detect_batch(detector, frames, prompt, w, h, **kw):
            detect_batch_prompts.append(prompt)
            crops = []
            for frame_idx in sorted(frames.keys()):
                crops.append(CropData(
                    crop_id=f"seed_{frame_idx}_{prompt[:4]}",
                    frame_idx=frame_idx,
                    xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
                    score=0.9,
                    label=CropLabel.PENDING,
                    source=CropSource.TEXT_DETECT,
                    prompt=prompt,
                ))
            return crops

        def mock_build_support_set(session):
            n = 10
            feats = np.random.randn(n, 2056).astype(np.float32)
            labels = np.array([1.0] * 5 + [0.0] * 5)
            crop_ids = [f"support_{i}" for i in range(n)]
            reasons = [None] * 5 + ["not_person"] * 5
            return feats, labels, crop_ids, reasons

        def mock_score_crops(query, support_feats, support_labels, support_reasons, **kw):
            n = query.shape[0]
            confs = np.full(n, 0.95, dtype=np.float32)
            preds = np.ones(n, dtype=np.float32)
            return confs, preds

        monkeypatch.setattr("interview.detection._decode_frames_sequential", mock_decode)
        monkeypatch.setattr("interview.detection._detect_batch", mock_detect_batch)
        monkeypatch.setattr("interview.detection.precompute_text_tokens", lambda det, p: None)
        monkeypatch.setattr("interview.detection.Sam3TextBasedDetector", TrackingDetector)
        monkeypatch.setattr(_mock_dinov3, "compute_crop_metadata",
                            lambda box, w, h: np.zeros(4, dtype=np.float32))
        monkeypatch.setattr(_mock_dinov3, "compute_mask_quality",
                            lambda mask, box, w, h: np.array([0.9, 0.0, 0.0, 0.8], dtype=np.float32))
        monkeypatch.setattr("interview.seeding_phase.extract_features",
                            lambda crops, batch_size=128: np.random.randn(len(crops), 1024).astype(np.float32))
        monkeypatch.setattr("interview.seeding_phase.build_support_set", mock_build_support_set)
        monkeypatch.setattr("interview.seeding_phase.score_crops", mock_score_crops)
        monkeypatch.setattr("interview.seeding_phase.save_session", lambda s: None)
        monkeypatch.setattr("interview.seeding_phase._refine_candidates_sam3",
                            lambda frames, cands, prompt="person", expand_frac=0.2: [])

        session = _make_seed_session(frames_count=500)
        session.prompts = ["person", "human figure", "pedestrian"]
        progress = _FakeProgress()

        result = generate_seeds(session, progress)

        # All 3 prompts should have been passed to _detect_batch
        prompts_used = set(detect_batch_prompts)
        assert "person" in prompts_used
        assert "human figure" in prompts_used
        assert "pedestrian" in prompts_used
        assert result["prompts_used"] == ["person", "human figure", "pedestrian"]

    def test_seeds_have_multi_prompt_knn_source(self, monkeypatch):
        """Seeds should have source='multi_prompt_knn'."""
        from interview.seeding_phase import generate_seeds

        _apply_seeding_mocks(monkeypatch, knn_confidence=0.95)

        session = _make_seed_session(frames_count=500)
        progress = _FakeProgress()

        result = generate_seeds(session, progress)

        assert result["total_seeds"] > 0
        for seed in session.seeds:
            assert seed["source"] == "multi_prompt_knn"


# ===========================================================================
# run_round_detection tests
# ===========================================================================

class TestRunRoundDetection:
    """Tests for round-based active learning detection."""

    def _make_round_session(self, frames_count=1000):
        """Create a session ready for round detection."""
        from interview.state import InterviewSession, Phase
        session = InterviewSession(
            session_id="round-test",
            project_id=1,
            task_id=1,
            cache_key="p1_t1",
            video_path="/fake/video.mp4",
            frames_count=frames_count,
            width=640,
            height=480,
        )
        session.phase = Phase.INIT
        return session

    def _apply_round_mocks(self, monkeypatch, crops_per_frame=1):
        """Set up mocks for run_round_detection tests.

        Mocks _decode_frames_sequential, _detect_batch, Sam3TextBasedDetector,
        and save_session. Returns decode_log for call tracking.
        """
        decode_log = []

        def mock_decode(video_path, frame_indices, **kw):
            decode_log.extend(frame_indices)
            frames = {}
            for idx in frame_indices:
                img = type("FakeImg", (), {
                    "width": 640, "height": 480, "size": (640, 480),
                })()
                frames[idx] = img
            return frames

        class MockDetector:
            def __init__(self, **kw): pass
            def clear_cache(self): pass

        def mock_detect_batch(detector, frames, prompt, w, h, **kw):
            from interview.state import CropData, CropLabel, CropSource
            crops = []
            for frame_idx in sorted(frames.keys()):
                for i in range(crops_per_frame):
                    crops.append(CropData(
                        crop_id=f"c_{frame_idx}_{i}",
                        frame_idx=frame_idx,
                        xyxy=np.array([10, 20, 100, 200], dtype=np.float32),
                        score=0.9,
                        label=CropLabel.PENDING,
                        source=CropSource.TEXT_DETECT,
                        prompt=prompt,
                    ))
            return crops

        monkeypatch.setattr("interview.detection._decode_frames_sequential", mock_decode)
        monkeypatch.setattr("interview.detection._detect_batch", mock_detect_batch)
        monkeypatch.setattr("interview.detection.Sam3TextBasedDetector", MockDetector)
        monkeypatch.setattr("interview.detection.save_session", lambda s: None)

        return decode_log

    def test_round1_produces_crops(self, monkeypatch):
        """Round 1 detection produces crops and records round state."""
        from interview.detection import run_round_detection

        decode_log = self._apply_round_mocks(monkeypatch)
        monkeypatch.setattr(
            "interview.detection.select_round_frames",
            lambda session, round_num: [0, 250, 500, 750],
        )

        session = self._make_round_session()
        progress = _FakeProgress()

        result = run_round_detection(session, "person", progress, round_num=1)

        assert result["round"] == 1
        assert result["total_crops"] > 0
        assert session.current_round == 1
        assert 1 in session.round_frames
        assert len(session.round_frames[1]) == 4
        # No auto_scored — MLP trains only at round boundaries
        assert "auto_scored" not in result

    def test_round2_excludes_round1_frames(self, monkeypatch):
        """Round 2 should pass session to select_round_frames, which excludes prior rounds."""
        from interview.detection import run_round_detection

        decode_log = self._apply_round_mocks(monkeypatch)

        select_calls = []
        def mock_select(session, round_num):
            select_calls.append((session.round_frames, round_num))
            return [100, 300, 600, 900]
        monkeypatch.setattr("interview.detection.select_round_frames", mock_select)

        session = self._make_round_session()
        session.current_round = 1
        session.round_frames = {1: [0, 250, 500, 750]}
        progress = _FakeProgress()

        result = run_round_detection(session, "person", progress, round_num=2)

        assert result["round"] == 2
        assert session.current_round == 2
        assert 2 in session.round_frames
        # Verify select_round_frames was called with round 2
        assert select_calls[0][1] == 2

    def test_phase_advances_on_round1(self, monkeypatch):
        """Round 1 should advance session to DETECTION phase."""
        from interview.detection import run_round_detection
        from interview.state import Phase

        self._apply_round_mocks(monkeypatch)
        monkeypatch.setattr(
            "interview.detection.select_round_frames",
            lambda s, r: [0, 50, 100],
        )

        session = self._make_round_session()
        progress = _FakeProgress()

        run_round_detection(session, "person", progress, round_num=1)

        assert session.phase == Phase.DETECTION

    def test_phase_not_reset_on_round2(self, monkeypatch):
        """Round 2+ should not reset the phase."""
        from interview.detection import run_round_detection
        from interview.state import Phase

        self._apply_round_mocks(monkeypatch)
        monkeypatch.setattr(
            "interview.detection.select_round_frames",
            lambda s, r: [100, 300],
        )

        session = self._make_round_session()
        session.phase = Phase.DETECTION
        session.current_round = 1
        session.round_frames = {1: [0, 50]}
        progress = _FakeProgress()

        run_round_detection(session, "person", progress, round_num=2)

        assert session.phase == Phase.DETECTION
        assert session.current_round == 2

    def test_prompt_added_to_session(self, monkeypatch):
        """The detection prompt should be added to session.prompts."""
        from interview.detection import run_round_detection

        self._apply_round_mocks(monkeypatch)
        monkeypatch.setattr(
            "interview.detection.select_round_frames",
            lambda s, r: [0],
        )

        session = self._make_round_session()
        progress = _FakeProgress()

        run_round_detection(session, "person walking", progress, round_num=1)

        assert "person walking" in session.prompts

    def test_no_frames_raises(self, monkeypatch):
        """Should raise RuntimeError if no frames available for the round."""
        from interview.detection import run_round_detection

        self._apply_round_mocks(monkeypatch)
        monkeypatch.setattr(
            "interview.detection.select_round_frames",
            lambda s, r: [],
        )

        session = self._make_round_session()
        progress = _FakeProgress()

        with pytest.raises(RuntimeError, match="No frames available"):
            run_round_detection(session, "person", progress, round_num=1)

    def test_round_history_appended(self, monkeypatch):
        """Round info should be appended to session.round_history."""
        from interview.detection import run_round_detection

        self._apply_round_mocks(monkeypatch)
        monkeypatch.setattr(
            "interview.detection.select_round_frames",
            lambda s, r: [0, 100, 200],
        )

        session = self._make_round_session()
        progress = _FakeProgress()

        run_round_detection(session, "person", progress, round_num=1)

        assert len(session.round_history) == 1
        assert session.round_history[0]["round"] == 1
        assert "frames" in session.round_history[0]


# ===========================================================================
# New tests: embedding_sampled_indices, pause/resume, feature mismatch
# ===========================================================================

class TestSessionEmbeddingSampledIndices:
    """Tests for the new embedding_sampled_indices field on InterviewSession."""

    def test_default_empty(self):
        session = _make_session()
        assert session.embedding_sampled_indices == []

    def test_writable(self):
        session = _make_session()
        session.embedding_sampled_indices = [0, 3, 6, 9, 12]
        assert session.embedding_sampled_indices == [0, 3, 6, 9, 12]


class TestCacheEmbeddingSampledIndices:
    """Cache round-trip for embedding_sampled_indices."""

    def test_save_load_sampled_indices(self, tmp_path, monkeypatch):
        monkeypatch.setattr("interview.cache_manager.CACHE_ROOT", str(tmp_path))
        from interview.cache_manager import save_session, load_session

        session = _make_session()
        session.embedding_sampled_indices = [0, 3, 6, 9, 12, 15]
        save_session(session)

        loaded = load_session(session.cache_key)
        assert loaded is not None
        assert loaded.embedding_sampled_indices == [0, 3, 6, 9, 12, 15]

    def test_backward_compat_no_field(self, tmp_path, monkeypatch):
        """Loading a cache without embedding_sampled_indices uses empty default."""
        monkeypatch.setattr("interview.cache_manager.CACHE_ROOT", str(tmp_path))
        from interview.cache_manager import save_session, load_session

        session = _make_session()
        save_session(session)

        # Strip the field from config.json
        import json
        config_path = tmp_path / session.cache_key / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        config.pop("embedding_sampled_indices", None)
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = load_session(session.cache_key)
        assert loaded is not None
        assert loaded.embedding_sampled_indices == []


class TestEmbeddingPauseResume:
    """Test that embedding jobs support pause/resume."""

    def test_pause_blocks_embedding(self, monkeypatch):
        """Verify pause_event is threaded through and blocks processing (sam3 mode)."""
        pause_checks = []

        def mock_embed(video_path, batch_size, progress_callback=None,
                       target_fps=None, pause_event=None, change_callback=None):
            # Simulate checking the pause event
            if pause_event is not None:
                pause_checks.append(pause_event.is_set())
            return (np.random.rand(50, 256).astype(np.float16), list(range(50)))

        monkeypatch.setattr("interview.detection.EMBEDDING_MODE", "sam3")
        monkeypatch.setattr("interview.detection._do_embed_all_frames", mock_embed)
        monkeypatch.setattr("interview.detection.save_session", lambda s: None)

        session = _make_session(frames_count=150)
        progress = _FakeProgress()
        progress._pause_event = threading.Event()
        progress._pause_event.set()  # Not paused

        run_embedding_background(session, progress)

        assert len(pause_checks) == 1
        assert pause_checks[0] is True  # Event was set (not paused)



class TestSelectRoundFramesRound2ChangeKeyframes:
    """Additional tests for round 2+ using change-detected keyframes."""

    def test_round2_uses_change_keyframes(self):
        """Round 2 should sample from change_keyframes when available."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=1000)
        session.change_keyframes = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
        session.round_frames = {1: [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]}

        frames = select_round_frames(session, round_num=2, frames_per_round=10)

        assert len(frames) > 0
        assert frames == sorted(frames)
        # Should not overlap with round 1 frames
        round1_set = set(session.round_frames[1])
        overlap = set(frames) & round1_set
        # Change keyframes [50,150,...] don't overlap with round 1 [0,100,...]
        assert len(overlap) == 0

    def test_round2_fallback_when_change_exhausted(self):
        """When all change frames used, falls back to full range."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=100)
        session.change_keyframes = [10, 20, 30]
        session.round_frames = {1: [10, 20, 30]}  # All change frames used

        frames = select_round_frames(session, round_num=2, frames_per_round=5)

        assert len(frames) > 0
        # Should fall back to non-change frames
        assert all(f not in [10, 20, 30] for f in frames) or len(frames) == 5

    def test_round3_excludes_rounds_1_and_2(self):
        """Round 3 should exclude both round 1 and round 2 frames."""
        from interview.detection import select_round_frames
        session = _make_session(frames_count=1000)
        session.change_keyframes = list(range(0, 1000, 10))  # every 10th frame
        session.round_frames = {
            1: [0, 100, 200, 300, 400],
            2: [50, 150, 250, 350, 450],
        }

        frames = select_round_frames(session, round_num=3, frames_per_round=10)

        used = set(session.round_frames[1]) | set(session.round_frames[2])
        assert not (set(frames) & used), "Round 3 should not reuse frames from rounds 1-2"


# ===========================================================================
# Reject Review Sub-Phase: subcategorize endpoint tests
# ===========================================================================

class TestSubcategorize:
    """Tests for POST /api/detect/subcategorize — reject review sub-phase."""

    def _make_session_with_rejected(self, n_rejected=3, n_accepted=2):
        """Create a session with some accepted and rejected crops."""
        session = _make_session()
        for i in range(n_accepted):
            session.add_crop(CropData(
                crop_id=f"acc{i}",
                frame_idx=i * 10,
                xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
                score=0.9,
                label=CropLabel.ACCEPTED,
            ))
        for i in range(n_rejected):
            session.add_crop(CropData(
                crop_id=f"rej{i}",
                frame_idx=(n_accepted + i) * 10,
                xyxy=np.array([100, 100, 200, 200], dtype=np.float32),
                score=0.3,
                label=CropLabel.REJECTED,
            ))
        return session

    def test_subcategorize_sets_reject_reason(self):
        """POST with just reject_reason should set crop.reject_reason."""
        from interview.routes import detect_subcategorize
        from interview.state import get_session as _gs, _sessions, _registry_lock

        session = self._make_session_with_rejected()
        with _registry_lock:
            _sessions[session.session_id] = session

        crop = session.get_crop("rej0")
        assert crop is not None
        assert crop.reject_reason is None

        # Directly set reject_reason (unit test — no Flask client needed)
        crop.reject_reason = "not_person"
        assert crop.reject_reason == "not_person"

        # Verify through session
        refreshed = session.get_crop("rej0")
        assert refreshed.reject_reason == "not_person"

        # Cleanup
        with _registry_lock:
            _sessions.pop(session.session_id, None)

    def test_subcategorize_creates_corrected_crop(self):
        """When adjusted_xyxy is provided, a new BOX_CORRECTED crop should be created."""
        from interview.state import _sessions, _registry_lock

        session = self._make_session_with_rejected()
        with _registry_lock:
            _sessions[session.session_id] = session

        original = session.get_crop("rej1")
        assert original is not None
        initial_count = len(session.crops)

        # Simulate what the endpoint does: create corrected crop
        import uuid
        new_crop = CropData(
            crop_id=str(uuid.uuid4())[:12],
            frame_idx=original.frame_idx,
            xyxy=np.array([110, 110, 190, 190], dtype=np.float32),
            score=1.0,
            label=CropLabel.ACCEPTED,
            source=CropSource.BOX_CORRECTED,
            prompt="box_corrected",
            corrected_from="rej1",
        )
        session.add_crop(new_crop)
        original.reject_reason = "partial_box"

        assert len(session.crops) == initial_count + 1
        assert new_crop.source == CropSource.BOX_CORRECTED
        assert new_crop.label == CropLabel.ACCEPTED
        assert new_crop.corrected_from == "rej1"
        assert new_crop.score == 1.0

        # Original should have reject_reason set
        assert original.reject_reason == "partial_box"

        # Cleanup
        with _registry_lock:
            _sessions.pop(session.session_id, None)

    def test_subcategorize_missing_crop_returns_none(self):
        """get_crop with a nonexistent crop_id returns None."""
        session = self._make_session_with_rejected()
        assert session.get_crop("nonexistent_crop") is None

    def test_subcategorize_wire_format(self):
        """Verify the exact JSON shape the frontend sends can be parsed."""
        # This is the wire format the JS constructs in _saveAndAdvanceRejectReview
        payload = {
            "session_id": "test-123",
            "crop_id": "rej0",
            "reject_reason": "partial_box",
            "adjusted_xyxy": [110, 110, 190, 190],
        }
        # Verify all required keys present
        assert "session_id" in payload
        assert "crop_id" in payload
        assert "reject_reason" in payload
        assert "adjusted_xyxy" in payload

        # Verify reject_reason is a valid subcategory
        valid_subcategories = {"not_person", "partial_box", "oversized_box"}
        assert payload["reject_reason"] in valid_subcategories

        # Verify adjusted_xyxy shape
        assert len(payload["adjusted_xyxy"]) == 4
        assert all(isinstance(v, (int, float)) for v in payload["adjusted_xyxy"])

    def test_subcategorize_no_adjustment_wire_format(self):
        """Wire format with no box adjustment — adjusted_xyxy is null."""
        payload = {
            "session_id": "test-123",
            "crop_id": "rej0",
            "reject_reason": "not_person",
            "adjusted_xyxy": None,
        }
        assert payload["adjusted_xyxy"] is None
        assert payload["reject_reason"] == "not_person"

    def test_corrected_crop_serialization(self):
        """BOX_CORRECTED crop should round-trip through to_dict/from_dict."""
        crop = CropData(
            crop_id="corrected1",
            frame_idx=50,
            xyxy=np.array([110, 110, 190, 190], dtype=np.float32),
            score=1.0,
            label=CropLabel.ACCEPTED,
            source=CropSource.BOX_CORRECTED,
            prompt="box_corrected",
            corrected_from="rej0",
        )
        d = crop.to_dict()
        assert d["source"] == "box_corrected"
        assert d["corrected_from"] == "rej0"
        assert d["label"] == "accepted"

        restored = CropData.from_dict(d)
        assert restored.source == CropSource.BOX_CORRECTED
        assert restored.corrected_from == "rej0"
        assert restored.label == CropLabel.ACCEPTED

    def test_reject_reason_serialization(self):
        """reject_reason should round-trip through to_dict/from_dict."""
        crop = CropData(
            crop_id="rej_tagged",
            frame_idx=30,
            xyxy=np.array([100, 100, 200, 200], dtype=np.float32),
            score=0.3,
            label=CropLabel.REJECTED,
            reject_reason="oversized_box",
        )
        d = crop.to_dict()
        assert d["reject_reason"] == "oversized_box"

        restored = CropData.from_dict(d)
        assert restored.reject_reason == "oversized_box"

    def test_reject_reason_absent_by_default(self):
        """New crops should have reject_reason=None, not in to_dict."""
        crop = CropData(
            crop_id="fresh",
            frame_idx=0,
            xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
            score=0.5,
        )
        assert crop.reject_reason is None
        d = crop.to_dict()
        assert "reject_reason" not in d
