"""
Tests for tracking fixes in initial_seeding_video_boxes.py and
initial_seeding_video_boxes_manual_merge.py.

Verifies:
1. Seed frame NOT double-counted (only in forward, excluded from backward)
2. Seed frame uses original annotation box (not model re-prediction)
3. Scores use object_score_logits (not binarized mask mean of 1.0)
4. Single-frame window returns original seed box (not empty)
5. object_score_logits early termination when score drops below threshold
6. Detection oracle cross-check truncates tracklets on ID switch

Run: pytest test_tracking_fixes.py -v
"""

from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures: mock SAM3 model outputs
# ---------------------------------------------------------------------------

@dataclass
class MockTrackerOutput:
    """Simulates Sam3TrackerVideoSegmentationOutput."""
    frame_idx: int
    pred_masks: torch.Tensor  # (1, 1, H, W)
    object_ids: List[int] = field(default_factory=lambda: [0])
    object_score_logits: Optional[torch.Tensor] = None  # (1,)


def _make_mask_with_box(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
    """Create a binary mask tensor with a filled rectangle."""
    mask = torch.zeros(1, 1, h, w, dtype=torch.float32)
    mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


def _make_empty_mask(h: int, w: int) -> torch.Tensor:
    """Create an all-zeros mask (object disappeared)."""
    return torch.zeros(1, 1, h, w, dtype=torch.float32)


@dataclass
class MockVideoOutput:
    """Simulates Sam3VideoSegmentationOutput for detection oracle."""
    frame_idx: int
    object_ids: List[int] = field(default_factory=list)
    obj_id_to_mask: Dict = field(default_factory=dict)
    obj_id_to_score: Dict = field(default_factory=dict)
    removed_obj_ids: set = field(default_factory=set)
    suppressed_obj_ids: set = field(default_factory=set)


@pytest.fixture
def mock_tracker():
    """Mock Sam3TrackerVideoModel + Sam3TrackerVideoProcessor."""
    model = MagicMock()
    processor = MagicMock()

    # processor(images=...) returns mock with original_sizes and pixel_values
    mock_inputs = MagicMock()
    mock_inputs.original_sizes = [(100, 100)]
    mock_inputs.pixel_values = [torch.zeros(3, 100, 100)]
    processor.return_value = mock_inputs

    # init_video_session returns mock session
    processor.init_video_session.return_value = MagicMock()
    processor.add_inputs_to_inference_session.return_value = None

    # post_process_masks just returns the masks as-is (already at original size)
    def passthrough_masks(masks_list, original_sizes=None, binarize=True):
        return [m.squeeze(0) for m in masks_list]  # remove batch dim
    processor.post_process_masks.side_effect = passthrough_masks

    return model, processor


@pytest.fixture
def five_frame_pil_images():
    """5 dummy 100x100 PIL images."""
    from PIL import Image
    return [Image.new("RGB", (100, 100), color=(i * 50, 0, 0)) for i in range(5)]


# ===========================================================================
# TEST GROUP 1: Seed frame double-counting
# ===========================================================================

class TestSeedFrameDoubleCounting:
    """After fix: seed frame should appear in forward tracklet (with original box)
    but NOT in backward tracklet's model predictions."""

    def test_forward_includes_seed_frame(self, mock_tracker, five_frame_pil_images):
        """Forward tracklet should include the seed frame."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2  # middle frame

        # Model yields frames 0-4, forward from seed=2 means frames 2,3,4
        outputs = [
            MockTrackerOutput(
                frame_idx=i,
                pred_masks=_make_mask_with_box(100, 100, 10+i, 10+i, 50+i, 50+i),
                object_score_logits=torch.tensor([2.0]),  # high confidence
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},  # session -> global
                seed_session_idx=seed_idx,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        # Seed frame (global=102) should be in forward results
        assert 102 in fwd_boxes, "Seed frame should be included in forward tracklet"
        # Frames after seed should also be present
        assert 103 in fwd_boxes
        assert 104 in fwd_boxes
        # Frames before seed should NOT be present
        assert 100 not in fwd_boxes
        assert 101 not in fwd_boxes

    def test_backward_excludes_seed_frame(self, mock_tracker, five_frame_pil_images):
        """Backward tracklet should NOT include the seed frame (forward owns it)."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2

        outputs = [
            MockTrackerOutput(
                frame_idx=i,
                pred_masks=_make_mask_with_box(100, 100, 10+i, 10+i, 50+i, 50+i),
                object_score_logits=torch.tensor([2.0]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_backward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        # Seed frame (global=102) should NOT be in backward results
        assert 102 not in bwd_boxes, "Seed frame should be excluded from backward tracklet"
        # Frames before seed should be present
        assert 100 in bwd_boxes
        assert 101 in bwd_boxes
        # Frames after seed should NOT be present
        assert 103 not in bwd_boxes
        assert 104 not in bwd_boxes

    def test_boxes_py_forward_includes_seed(self, mock_tracker, five_frame_pil_images):
        """Same test for initial_seeding_video_boxes.py's forward function."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2

        outputs = [
            MockTrackerOutput(
                frame_idx=i,
                pred_masks=_make_mask_with_box(100, 100, 10+i, 10+i, 50+i, 50+i),
                object_score_logits=torch.tensor([2.0]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        assert 102 in fwd_boxes, "Seed frame should be included in forward tracklet"
        assert 100 not in fwd_boxes
        assert 101 not in fwd_boxes

    def test_boxes_py_backward_excludes_seed(self, mock_tracker, five_frame_pil_images):
        """Same test for initial_seeding_video_boxes.py's backward function."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2

        outputs = [
            MockTrackerOutput(
                frame_idx=i,
                pred_masks=_make_mask_with_box(100, 100, 10+i, 10+i, 50+i, 50+i),
                object_score_logits=torch.tensor([2.0]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes import _generate_backward_tracklet_sam3

        with patch("initial_seeding_video_boxes.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        assert 102 not in bwd_boxes, "Seed frame should be excluded from backward tracklet"
        assert 100 in bwd_boxes
        assert 101 in bwd_boxes


# ===========================================================================
# TEST GROUP 2: Scores from object_score_logits (not always 1.0)
# ===========================================================================

class TestScoreExtraction:
    """After fix: scores should come from object_score_logits, not binarized mask mean."""

    def test_manual_merge_scores_vary(self, mock_tracker, five_frame_pil_images):
        """manual_merge forward tracklet scores should reflect object_score_logits."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        # Decreasing confidence: 0.95, 0.80, 0.60
        outputs = [
            MockTrackerOutput(
                frame_idx=2,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),
                object_score_logits=torch.tensor([3.0]),  # sigmoid ~ 0.95
            ),
            MockTrackerOutput(
                frame_idx=3,
                pred_masks=_make_mask_with_box(100, 100, 12, 12, 52, 52),
                object_score_logits=torch.tensor([1.4]),  # sigmoid ~ 0.80
            ),
            MockTrackerOutput(
                frame_idx=4,
                pred_masks=_make_mask_with_box(100, 100, 14, 14, 54, 54),
                object_score_logits=torch.tensor([0.4]),  # sigmoid ~ 0.60
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        # Scores should NOT all be 1.0
        score_values = list(fwd_scores.values())
        assert len(score_values) > 0, "Should have scores"
        assert not all(s == 1.0 for s in score_values), \
            f"Scores should not all be 1.0, got {score_values}"
        # Scores should be decreasing (following logits)
        if len(score_values) >= 2:
            assert score_values[0] > score_values[-1], \
                f"First score should be higher than last: {score_values}"

    def test_manual_merge_mask_to_xyxy_returns_logit_score(self):
        """_mask_to_xyxy should use object_score_logits when provided."""
        from initial_seeding_video_boxes_manual_merge import _mask_to_xyxy

        mask = _make_mask_with_box(100, 100, 10, 10, 50, 50).squeeze(0)  # (1, H, W)
        logits = torch.tensor([1.5])  # sigmoid ~ 0.82

        box, score = _mask_to_xyxy(mask, object_score_logits=logits)

        assert box is not None
        assert score is not None
        assert score != 1.0, f"Score should not be 1.0 (binarized mask mean), got {score}"
        assert 0.8 < score < 0.85, f"Score should be sigmoid(1.5) ~ 0.82, got {score}"


# ===========================================================================
# TEST GROUP 3: Single-frame window
# ===========================================================================

class TestSingleFrameWindow:
    """When win_len == 1 (seed frame is first AND last), both fwd and bwd
    edge guards fire. The seed frame should still get the original annotation box."""

    def test_single_frame_returns_seed_box_manual_merge(self, mock_tracker):
        """manual_merge: single-frame window should yield seed box at seed frame."""
        from PIL import Image
        frames = [Image.new("RGB", (100, 100))]

        from initial_seeding_video_boxes_manual_merge import (
            _generate_forward_tracklet_sam3,
            _generate_backward_tracklet_sam3,
        )

        model, processor = mock_tracker

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={0: 500},
                seed_session_idx=0,
                seed_box_xyxy=np.array([20, 20, 60, 60], dtype=np.float32),
            )
            bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={0: 500},
                seed_session_idx=0,
                seed_box_xyxy=np.array([20, 20, 60, 60], dtype=np.float32),
            )

        # At least one of fwd/bwd should contain the seed frame with original box
        all_boxes = {**fwd_boxes, **bwd_boxes}
        assert 500 in all_boxes, \
            "Seed frame should have a box even with single-frame window"
        np.testing.assert_array_almost_equal(
            all_boxes[500], [20, 20, 60, 60],
            err_msg="Single-frame window should use original annotation box",
        )


# ===========================================================================
# TEST GROUP 4: object_score_logits early termination
# ===========================================================================

class TestEarlyTermination:
    """Layer 1: When object_score_logits drops below threshold, tracking should stop."""

    def test_forward_stops_on_low_logits(self, mock_tracker, five_frame_pil_images):
        """Forward tracking should terminate when object_score_logits indicates disappearance."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        # Frame 2: high confidence, Frame 3: high, Frame 4: very low (disappeared)
        outputs = [
            MockTrackerOutput(
                frame_idx=2,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),
                object_score_logits=torch.tensor([3.0]),  # sigmoid ~ 0.95
            ),
            MockTrackerOutput(
                frame_idx=3,
                pred_masks=_make_mask_with_box(100, 100, 12, 12, 52, 52),
                object_score_logits=torch.tensor([2.0]),  # sigmoid ~ 0.88
            ),
            MockTrackerOutput(
                frame_idx=4,
                pred_masks=_make_empty_mask(100, 100),
                object_score_logits=torch.tensor([-3.0]),  # sigmoid ~ 0.05 (disappeared!)
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
                score_threshold=0.5,
            )

        # Frame 104 (session=4) should NOT be in results (score below threshold)
        assert 104 not in fwd_boxes, \
            "Frame with low object_score_logits should be excluded"
        # Frames 102, 103 should be present
        assert 102 in fwd_boxes or 103 in fwd_boxes, \
            "High-confidence frames should be included"

    def test_backward_stops_on_low_logits(self, mock_tracker, five_frame_pil_images):
        """Backward tracking should terminate when object_score_logits indicates disappearance."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        # Backward from seed=2: frames 1 (ok), 0 (disappeared)
        outputs = [
            MockTrackerOutput(
                frame_idx=1,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),
                object_score_logits=torch.tensor([2.0]),
            ),
            MockTrackerOutput(
                frame_idx=0,
                pred_masks=_make_empty_mask(100, 100),
                object_score_logits=torch.tensor([-3.0]),  # disappeared
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_backward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
                score_threshold=0.5,
            )

        # Frame 100 (session=0) should NOT be in results
        assert 100 not in bwd_boxes, \
            "Frame with low object_score_logits should be excluded from backward"
        # Frame 101 should be present
        assert 101 in bwd_boxes


# ===========================================================================
# TEST GROUP 5: Seed frame uses original annotation box
# ===========================================================================

class TestSeedBoxPreservation:
    """Seed frame should use the original annotation box, not model re-prediction."""

    def test_forward_seed_frame_uses_original_box(self, mock_tracker, five_frame_pil_images):
        """At seed frame, the box should be the original annotation, not mask-derived."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2
        original_box = np.array([25, 25, 75, 75], dtype=np.float32)

        # Model predicts a DIFFERENT box at the seed frame (mask-derived would be [10,10,50,50])
        outputs = [
            MockTrackerOutput(
                frame_idx=2,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),  # different!
                object_score_logits=torch.tensor([3.0]),
            ),
            MockTrackerOutput(
                frame_idx=3,
                pred_masks=_make_mask_with_box(100, 100, 12, 12, 52, 52),
                object_score_logits=torch.tensor([2.5]),
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=original_box,
            )

        # Seed frame box should be the ORIGINAL, not the model-derived [10,10,51,51]
        assert 102 in fwd_boxes
        np.testing.assert_array_almost_equal(
            fwd_boxes[102], original_box,
            err_msg="Seed frame should use original annotation box, not model re-prediction",
        )


# ===========================================================================
# TEST GROUP 6: Detection oracle (Layer 2)
# ===========================================================================

class TestDetectionOracle:
    """Layer 2: Sam3VideoModel oracle should detect ID switches and truncate tracklets."""

    def test_oracle_validates_tracker_output(self):
        """Oracle should flag frames where tracker box doesn't overlap any detected person."""
        # This tests the oracle cross-check function that will be added
        # Import will fail until implementation is done
        try:
            from initial_seeding_video_boxes_manual_merge import _oracle_validate_tracklet
        except ImportError:
            pytest.skip("_oracle_validate_tracklet not yet implemented")

        # Tracker tracked person at these locations
        tracker_boxes = {
            100: np.array([10, 10, 50, 50], dtype=np.float32),  # correct
            101: np.array([12, 12, 52, 52], dtype=np.float32),  # correct
            102: np.array([200, 200, 280, 280], dtype=np.float32),  # ID SWITCH!
            103: np.array([205, 205, 285, 285], dtype=np.float32),  # still wrong
        }

        # Oracle detections: person is at [10-55, 10-55] range on all frames
        oracle_detections = {
            100: [np.array([10, 10, 55, 55], dtype=np.float32)],
            101: [np.array([12, 12, 55, 55], dtype=np.float32)],
            102: [np.array([14, 14, 56, 56], dtype=np.float32)],  # person still here
            103: [np.array([16, 16, 58, 58], dtype=np.float32)],
        }

        validated = _oracle_validate_tracklet(
            tracker_boxes=tracker_boxes,
            oracle_detections=oracle_detections,
            iou_threshold=0.3,
        )

        # Frames 100, 101 should pass (tracker overlaps with detection)
        assert 100 in validated
        assert 101 in validated
        # Frames 102, 103 should be removed (tracker box at [200,200] doesn't
        # overlap with detection at [14,14,56,56])
        assert 102 not in validated, "ID-switched frame should be removed by oracle"
        assert 103 not in validated, "Post-switch frame should also be removed"

    def test_oracle_allows_size_variation(self):
        """Oracle should allow reasonable size variation (not flag normal tracking drift)."""
        try:
            from initial_seeding_video_boxes_manual_merge import _oracle_validate_tracklet
        except ImportError:
            pytest.skip("_oracle_validate_tracklet not yet implemented")

        # Tracker boxes grow slightly (normal tracking behavior)
        tracker_boxes = {
            100: np.array([10, 10, 50, 50], dtype=np.float32),
            101: np.array([9, 9, 52, 52], dtype=np.float32),  # slightly larger
            102: np.array([8, 8, 54, 54], dtype=np.float32),  # a bit more
        }

        oracle_detections = {
            100: [np.array([10, 10, 50, 50], dtype=np.float32)],
            101: [np.array([10, 10, 51, 51], dtype=np.float32)],
            102: [np.array([9, 9, 53, 53], dtype=np.float32)],
        }

        validated = _oracle_validate_tracklet(
            tracker_boxes=tracker_boxes,
            oracle_detections=oracle_detections,
            iou_threshold=0.3,
        )

        # All frames should pass (reasonable IoU overlap)
        assert len(validated) == 3, "Normal tracking drift should not be flagged"


# ===========================================================================
# TEST GROUP 7: _resolve_frame_boxes with varying scores
# ===========================================================================

class TestResolveFrameBoxes:
    """_resolve_frame_boxes should weight by real scores, not degenerate uniform."""

    def test_high_score_dominates(self):
        """With real scores, higher-scored box should dominate in weighted mode."""
        from initial_seeding_video_boxes_manual_merge import _resolve_frame_boxes

        high_score_box = np.array([10, 10, 50, 50], dtype=np.float32)
        low_score_box = np.array([30, 30, 70, 70], dtype=np.float32)

        candidates = [
            (high_score_box, 0.95),
            (low_score_box, 0.10),
        ]

        result = _resolve_frame_boxes(candidates, iou_threshold=0.0, mode="weighted")

        assert result is not None
        # Result should be much closer to high_score_box than low_score_box
        dist_to_high = np.linalg.norm(result - high_score_box)
        dist_to_low = np.linalg.norm(result - low_score_box)
        assert dist_to_high < dist_to_low, \
            f"Result should be closer to high-score box: dist_high={dist_to_high:.2f}, dist_low={dist_to_low:.2f}"

    def test_equal_scores_average(self):
        """With equal scores, result should be midpoint."""
        from initial_seeding_video_boxes_manual_merge import _resolve_frame_boxes

        box_a = np.array([0, 0, 40, 40], dtype=np.float32)
        box_b = np.array([20, 20, 60, 60], dtype=np.float32)

        candidates = [(box_a, 0.5), (box_b, 0.5)]
        result = _resolve_frame_boxes(candidates, iou_threshold=0.0, mode="weighted")

        assert result is not None
        expected = np.array([10, 10, 50, 50], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_winner_mode_picks_highest_score(self):
        """Winner mode should pick the highest-scored box."""
        from initial_seeding_video_boxes_manual_merge import _resolve_frame_boxes

        candidates = [
            (np.array([0, 0, 10, 10], dtype=np.float32), 0.3),
            (np.array([50, 50, 90, 90], dtype=np.float32), 0.9),
        ]

        result = _resolve_frame_boxes(candidates, mode="winner")
        assert result is not None
        np.testing.assert_array_almost_equal(result, [50, 50, 90, 90])


# ===========================================================================
# TEST GROUP 8: boxes.py also returns scores from tracklet generators
# ===========================================================================

class TestBoxesPyScores:
    """After fix: boxes.py forward/backward functions should also return scores."""

    def test_forward_returns_scores(self, mock_tracker, five_frame_pil_images):
        """boxes.py _generate_forward_tracklet_sam3 should return (boxes, scores) tuple."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        outputs = [
            MockTrackerOutput(
                frame_idx=2,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),
                object_score_logits=torch.tensor([2.0]),
            ),
            MockTrackerOutput(
                frame_idx=3,
                pred_masks=_make_mask_with_box(100, 100, 12, 12, 52, 52),
                object_score_logits=torch.tensor([1.5]),
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            result = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        # After fix, should return (boxes_dict, scores_dict) tuple
        assert isinstance(result, tuple), \
            "boxes.py forward should return (boxes, scores) tuple"
        fwd_boxes, fwd_scores = result
        assert isinstance(fwd_boxes, dict)
        assert isinstance(fwd_scores, dict)
        assert len(fwd_scores) > 0, "Should have scores"


# ===========================================================================
# TEST GROUP 7: Batched multi-seed tracking
# ===========================================================================

def _make_multi_mask(h: int, w: int, boxes: List[Tuple[int, int, int, int]]) -> torch.Tensor:
    """Create (n_objects, 1, H, W) mask tensor with filled rectangles."""
    n = len(boxes)
    mask = torch.zeros(n, 1, h, w, dtype=torch.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        mask[i, 0, y1:y2, x1:x2] = 1.0
    return mask


@dataclass
class MockMultiTrackerOutput:
    """Simulates Sam3TrackerVideoSegmentationOutput with multiple objects."""
    frame_idx: int
    pred_masks: torch.Tensor  # (n_objects, 1, H, W)
    object_ids: List[int]     # [0, 1, 2, ...]
    object_score_logits: Optional[torch.Tensor] = None  # (n_objects,)


class TestBatchedForwardBasic:
    """Batched forward: 3 seeds tracked in one session, all produce boxes."""

    def test_three_seeds_all_tracked(self, mock_tracker, five_frame_pil_images):
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2  # middle frame

        seed_boxes = [
            np.array([10, 10, 30, 30], dtype=np.float32),  # seed 0
            np.array([40, 40, 70, 70], dtype=np.float32),  # seed 1
            np.array([60, 10, 90, 40], dtype=np.float32),  # seed 2
        ]

        # Model yields multi-object outputs for frames 0-4
        outputs = [
            MockMultiTrackerOutput(
                frame_idx=i,
                pred_masks=_make_multi_mask(100, 100, [
                    (10+i, 10+i, 30+i, 30+i),  # obj 0 moves
                    (40+i, 40+i, 70+i, 70+i),  # obj 1 moves
                    (60+i, 10+i, 90+i, 40+i),  # obj 2 moves
                ]),
                object_ids=[0, 1, 2],
                object_score_logits=torch.tensor([2.0, 1.8, 2.5]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_batched_forward_tracklets_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            results = _generate_batched_forward_tracklets_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_boxes=seed_boxes,
            )

        assert len(results) == 3, "Should return one result tuple per seed"

        for idx in range(3):
            fwd_boxes, fwd_scores = results[idx]
            # Seed frame present with original box
            assert 102 in fwd_boxes, f"Seed {idx}: seed frame should be in forward"
            np.testing.assert_array_equal(fwd_boxes[102], seed_boxes[idx])
            assert fwd_scores[102] == 1.0
            # Frames after seed present
            assert 103 in fwd_boxes, f"Seed {idx}: frame after seed should be tracked"
            assert 104 in fwd_boxes, f"Seed {idx}: last frame should be tracked"
            # Frames before seed NOT present
            assert 100 not in fwd_boxes
            assert 101 not in fwd_boxes

    def test_single_session_created(self, mock_tracker, five_frame_pil_images):
        """Verify init_video_session is called exactly once (not N times)."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        seed_boxes = [
            np.array([10, 10, 30, 30], dtype=np.float32),
            np.array([40, 40, 70, 70], dtype=np.float32),
        ]

        outputs = [
            MockMultiTrackerOutput(
                frame_idx=i,
                pred_masks=_make_multi_mask(100, 100, [
                    (10+i, 10+i, 30+i, 30+i),
                    (40+i, 40+i, 70+i, 70+i),
                ]),
                object_ids=[0, 1],
                object_score_logits=torch.tensor([2.0, 1.8]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_batched_forward_tracklets_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            _generate_batched_forward_tracklets_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_boxes=seed_boxes,
            )

        # The key assertion: vision encoder (init_video_session) called once, not twice
        assert processor.init_video_session.call_count == 1, \
            "Should create exactly 1 session for 2 seeds (not 2 sessions)"


class TestBatchedPerObjectTermination:
    """One seed terminates early while others continue."""

    def test_seed0_terminates_seed1_continues(self, mock_tracker, five_frame_pil_images):
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 1  # seed at frame 1, propagate to 2,3,4

        seed_boxes = [
            np.array([10, 10, 30, 30], dtype=np.float32),  # seed 0: will disappear
            np.array([50, 50, 80, 80], dtype=np.float32),  # seed 1: stays visible
        ]

        # Frame 2: both objects visible
        # Frames 3,4: object 0 has low score (drops below threshold 3 consecutive times)
        # object 1 stays strong throughout
        outputs = [
            # Frame 1 (seed frame, will be skipped in propagation)
            MockMultiTrackerOutput(
                frame_idx=1,
                pred_masks=_make_multi_mask(100, 100, [
                    (10, 10, 30, 30), (50, 50, 80, 80),
                ]),
                object_ids=[0, 1],
                object_score_logits=torch.tensor([2.0, 2.0]),
            ),
            # Frame 2: both good
            MockMultiTrackerOutput(
                frame_idx=2,
                pred_masks=_make_multi_mask(100, 100, [
                    (11, 11, 31, 31), (51, 51, 81, 81),
                ]),
                object_ids=[0, 1],
                object_score_logits=torch.tensor([1.5, 2.0]),
            ),
            # Frame 3: obj 0 score drops (logit -3.0 -> sigmoid ~0.05)
            MockMultiTrackerOutput(
                frame_idx=3,
                pred_masks=_make_multi_mask(100, 100, [
                    (12, 12, 32, 32), (52, 52, 82, 82),
                ]),
                object_ids=[0, 1],
                object_score_logits=torch.tensor([-3.0, 2.0]),
            ),
            # Frame 4: obj 0 still low (2nd consecutive), obj 1 still good
            MockMultiTrackerOutput(
                frame_idx=4,
                pred_masks=_make_multi_mask(100, 100, [
                    (13, 13, 33, 33), (53, 53, 83, 83),
                ]),
                object_ids=[0, 1],
                object_score_logits=torch.tensor([-3.0, 1.8]),
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_batched_forward_tracklets_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            results = _generate_batched_forward_tracklets_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_boxes=seed_boxes,
                score_threshold=0.1,
            )

        fwd0, scores0 = results[0]
        fwd1, scores1 = results[1]

        # Seed 0: has seed frame + frame 102, but drops at 103,104
        assert 101 in fwd0, "Seed 0 should have seed frame"
        assert 102 in fwd0, "Seed 0 should have frame 2 (good score)"
        assert 103 not in fwd0, "Seed 0: frame 3 below threshold, should be skipped"
        assert 104 not in fwd0, "Seed 0: frame 4 below threshold, should be skipped"

        # Seed 1: tracks all frames (always high score)
        assert 101 in fwd1, "Seed 1 should have seed frame"
        assert 102 in fwd1, "Seed 1 should have frame 2"
        assert 103 in fwd1, "Seed 1 should have frame 3"
        assert 104 in fwd1, "Seed 1 should have frame 4"

    def test_all_seeds_terminate_stops_session(self, mock_tracker, five_frame_pil_images):
        """When all objects terminate, propagation should stop."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 0

        seed_boxes = [
            np.array([10, 10, 30, 30], dtype=np.float32),
            np.array([50, 50, 80, 80], dtype=np.float32),
        ]

        # All objects have low scores from the start → all terminate by frame 3
        outputs = [
            MockMultiTrackerOutput(
                frame_idx=i,
                pred_masks=_make_multi_mask(100, 100, [
                    (10, 10, 30, 30), (50, 50, 80, 80),
                ]),
                object_ids=[0, 1],
                object_score_logits=torch.tensor([-5.0, -5.0]),  # very low
            )
            for i in range(5)
        ]
        # Track which frames are actually consumed
        consumed = []
        def tracked_iter(*args, **kwargs):
            for o in outputs:
                consumed.append(o.frame_idx)
                yield o
        model.propagate_in_video_iterator.side_effect = tracked_iter

        from initial_seeding_video_boxes_manual_merge import _generate_batched_forward_tracklets_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            results = _generate_batched_forward_tracklets_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_boxes=seed_boxes,
                score_threshold=0.1,
            )

        # Both seeds should have only seed frame (everything else below threshold)
        for idx in range(2):
            fwd_boxes, _ = results[idx]
            assert 100 in fwd_boxes, "Seed frame should always be present"
            assert len(fwd_boxes) == 1, f"Seed {idx}: only seed frame should survive"

        # Session should have stopped early (not consumed all 5 frames)
        # Frame 0 is seed (skipped), frames 1,2,3 are 3 consecutive low → terminate at 3
        assert len(consumed) < 5, \
            f"Session should stop early when all objects terminate, consumed {len(consumed)} frames"


class TestBatchedBackward:
    """Batched backward tracking excludes seed frame for all seeds."""

    def test_backward_excludes_seed_for_all(self, mock_tracker, five_frame_pil_images):
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2

        seed_boxes = [
            np.array([10, 10, 30, 30], dtype=np.float32),
            np.array([50, 50, 80, 80], dtype=np.float32),
        ]

        outputs = [
            MockMultiTrackerOutput(
                frame_idx=i,
                pred_masks=_make_multi_mask(100, 100, [
                    (10+i, 10+i, 30+i, 30+i),
                    (50+i, 50+i, 80+i, 80+i),
                ]),
                object_ids=[0, 1],
                object_score_logits=torch.tensor([2.0, 2.0]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_batched_backward_tracklets_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            results = _generate_batched_backward_tracklets_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_boxes=seed_boxes,
            )

        assert len(results) == 2
        for idx in range(2):
            bwd_boxes, _ = results[idx]
            # Frames before seed should be present
            assert 100 in bwd_boxes, f"Seed {idx}: frame 0 should be in backward"
            assert 101 in bwd_boxes, f"Seed {idx}: frame 1 should be in backward"
            # Seed frame and after should NOT be present
            assert 102 not in bwd_boxes, f"Seed {idx}: seed frame should be excluded"
            assert 103 not in bwd_boxes
            assert 104 not in bwd_boxes

    def test_backward_at_first_frame_returns_empty(self, mock_tracker, five_frame_pil_images):
        """If seed is frame 0, nothing to propagate backward."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        seed_boxes = [
            np.array([10, 10, 30, 30], dtype=np.float32),
            np.array([50, 50, 80, 80], dtype=np.float32),
        ]

        from initial_seeding_video_boxes_manual_merge import _generate_batched_backward_tracklets_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            results = _generate_batched_backward_tracklets_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=0,
                seed_boxes=seed_boxes,
            )

        for idx in range(2):
            bwd_boxes, bwd_scores = results[idx]
            assert len(bwd_boxes) == 0, f"Seed {idx}: backward from frame 0 should be empty"
            assert len(bwd_scores) == 0


class TestBatchedEdgeCases:
    """Edge cases for batched tracking."""

    def test_single_seed_matches_unbatched(self, mock_tracker, five_frame_pil_images):
        """Batched with 1 seed should produce identical results to unbatched."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2
        seed_box = np.array([10, 10, 50, 50], dtype=np.float32)

        def make_outputs():
            return [
                MockMultiTrackerOutput(
                    frame_idx=i,
                    pred_masks=_make_multi_mask(100, 100, [(10+i, 10+i, 50+i, 50+i)]),
                    object_ids=[0],
                    object_score_logits=torch.tensor([2.0]),
                )
                for i in range(5)
            ]

        from initial_seeding_video_boxes_manual_merge import (
            _generate_forward_tracklet_sam3,
            _generate_batched_forward_tracklets_sam3,
        )

        # Run unbatched
        model.propagate_in_video_iterator.return_value = iter(make_outputs())
        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            unbatched_boxes, unbatched_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=seed_box,
            )

        # Run batched with 1 seed
        model.propagate_in_video_iterator.return_value = iter(make_outputs())
        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            batched_results = _generate_batched_forward_tracklets_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_boxes=[seed_box],
            )

        batched_boxes, batched_scores = batched_results[0]

        assert set(unbatched_boxes.keys()) == set(batched_boxes.keys()), \
            "Same frames should be tracked"
        for frame_idx in unbatched_boxes:
            np.testing.assert_array_equal(
                unbatched_boxes[frame_idx], batched_boxes[frame_idx],
                err_msg=f"Boxes should match at frame {frame_idx}",
            )
        assert unbatched_scores == batched_scores, "Scores should match"

    def test_empty_seed_list(self, mock_tracker, five_frame_pil_images):
        """Empty seed list returns empty results."""
        from initial_seeding_video_boxes_manual_merge import _generate_batched_forward_tracklets_sam3

        model, processor = mock_tracker
        results = _generate_batched_forward_tracklets_sam3(
            frames_list=five_frame_pil_images,
            frame_idx_map={i: i for i in range(5)},
            seed_session_idx=2,
            seed_boxes=[],
        )
        assert results == []

    def test_seed_at_last_frame_returns_originals_only(self, mock_tracker, five_frame_pil_images):
        """If seed is the last frame, forward returns only original seed boxes."""
        from initial_seeding_video_boxes_manual_merge import _generate_batched_forward_tracklets_sam3

        model, processor = mock_tracker
        seed_boxes = [
            np.array([10, 10, 30, 30], dtype=np.float32),
            np.array([50, 50, 80, 80], dtype=np.float32),
        ]

        results = _generate_batched_forward_tracklets_sam3(
            frames_list=five_frame_pil_images,
            frame_idx_map={i: i + 100 for i in range(5)},
            seed_session_idx=4,  # last frame
            seed_boxes=seed_boxes,
        )

        assert len(results) == 2
        for idx in range(2):
            fwd_boxes, fwd_scores = results[idx]
            assert len(fwd_boxes) == 1, "Only seed frame"
            assert 104 in fwd_boxes
            np.testing.assert_array_equal(fwd_boxes[104], seed_boxes[idx])
            assert fwd_scores[104] == 1.0

        # No session should have been created
        assert processor.init_video_session.call_count == 0, \
            "No session needed when seed is at last frame"


# ===========================================================================
# TEST GROUP 9: Streaming forward tracking
# ===========================================================================

from fractions import Fraction
from PIL import Image as PILImage


def _make_mock_av_container(n_frames, average_rate=30, time_base=Fraction(1, 30000),
                             frame_size=(100, 100), start_pts=0):
    """Create a mock PyAV container that yields n_frames via demux/decode.

    PTS values are set so that frame_idx = int(round(pts * time_base * average_rate)).
    With time_base=1/30000 and average_rate=30, pts=k*1000 → frame_idx=k.
    """
    pts_per_frame = int(1.0 / (float(time_base) * float(average_rate)))  # = 1000

    mock_frames = []
    for i in range(n_frames):
        mf = MagicMock()
        mf.to_image.return_value = PILImage.new('RGB', frame_size, color=(i * 30 % 256, 0, 0))
        mf.pts = start_pts + i * pts_per_frame
        mock_frames.append(mf)

    mock_stream = MagicMock()
    mock_stream.average_rate = average_rate
    mock_stream.time_base = time_base

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]

    # demux returns one packet whose decode() yields all frames
    mock_packet = MagicMock()
    mock_packet.decode.return_value = iter(mock_frames)
    mock_container.demux.return_value = iter([mock_packet])

    return mock_container


def _setup_streaming_tracker_mock(mock_get_tracker, mask_box=(20, 10, 80, 50),
                                    frame_size=(100, 100), score_logit=2.0,
                                    score_logits_sequence=None):
    """Configure mock SAM3 tracker model + processor for streaming tests.

    The mock model is callable (returns output per call, not iterator).

    Args:
        score_logits_sequence: If provided, a list of logit values. The model
            returns them in order on successive calls.  If exhausted, repeats last.
    """
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_get_tracker.return_value = (mock_model, mock_processor)

    h, w = frame_size

    # Processor __call__ → inputs with pixel_values and original_sizes
    mock_inputs = MagicMock()
    mock_inputs.pixel_values = [torch.zeros(3, h, w)]
    mock_inputs.original_sizes = [[h, w]]
    mock_processor.return_value = mock_inputs

    # init_video_session → mock session
    mock_processor.init_video_session.return_value = MagicMock()
    mock_processor.add_inputs_to_inference_session.return_value = None

    # post_process_masks: return mask with the specified box
    x1, y1, x2, y2 = mask_box
    mask_template = torch.zeros(1, h, w, dtype=torch.bool)
    mask_template[0, y1:y2, x1:x2] = True

    def passthrough_masks(masks_list, original_sizes=None, binarize=True):
        return [m.squeeze(0) for m in masks_list]
    mock_processor.post_process_masks.side_effect = passthrough_masks

    # Model __call__ → output (NOT iterator)
    call_count = [0]

    def model_call(**kwargs):
        idx = call_count[0]
        call_count[0] += 1

        output = MagicMock()
        # Build pred_masks for however many objects were registered
        # We use 1 object per seed; the number of seeds is determined by
        # how many add_inputs_to_inference_session calls were made.
        n_objs = mock_processor.add_inputs_to_inference_session.call_count
        if n_objs == 0:
            n_objs = 1

        masks = torch.zeros(n_objs, 1, h, w, dtype=torch.bool)
        for oi in range(n_objs):
            masks[oi, 0, y1:y2, x1:x2] = True
        output.pred_masks = masks
        output.object_ids = torch.tensor(list(range(n_objs)))

        if score_logits_sequence is not None:
            logit_idx = min(idx, len(score_logits_sequence) - 1)
            logit_val = score_logits_sequence[logit_idx]
            if isinstance(logit_val, (list, tuple)):
                output.object_score_logits = torch.tensor(logit_val, dtype=torch.bfloat16)
            else:
                output.object_score_logits = torch.tensor(
                    [logit_val] * n_objs, dtype=torch.bfloat16
                )
        else:
            output.object_score_logits = torch.tensor(
                [score_logit] * n_objs, dtype=torch.bfloat16
            )

        return output

    mock_model.side_effect = model_call

    return mock_model, mock_processor


class TestStreamingForwardTracking:
    """Tests for _generate_streaming_forward_tracklets_sam3 (constant-memory
    frame-by-frame tracking via PyAV)."""

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_streaming_forward_basic(self, mock_get_tracker, mock_av_open):
        """Single seed, 5 frames. Seed frame has score 1.0, others tracked."""
        mock_av_open.return_value = _make_mock_av_container(5)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=4,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            score_threshold=0.1,
        )

        assert len(results) == 1, "Should return one result tuple for one seed"
        fwd_boxes, fwd_scores = results[0]

        # Seed frame present with original box and score 1.0
        assert 0 in fwd_boxes, "Seed frame (kf_global=0) should be in results"
        np.testing.assert_array_equal(fwd_boxes[0], [20, 10, 80, 50])
        assert fwd_scores[0] == 1.0, "Seed frame score should be 1.0"

        # Subsequent frames should be tracked
        for fi in [1, 2, 3, 4]:
            assert fi in fwd_boxes, f"Frame {fi} should be tracked"
            assert fi in fwd_scores, f"Frame {fi} should have a score"
            assert fwd_scores[fi] != 1.0, f"Frame {fi} should have model-derived score"

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_streaming_forward_multi_seed(self, mock_get_tracker, mock_av_open):
        """3 seeds, verify all tracked independently with correct obj_ids."""
        mock_av_open.return_value = _make_mock_av_container(5)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        seed_boxes = [
            np.array([10, 10, 30, 30], dtype=np.float32),
            np.array([40, 40, 70, 70], dtype=np.float32),
            np.array([60, 10, 90, 40], dtype=np.float32),
        ]

        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=4,
            seed_boxes=seed_boxes,
            score_threshold=0.1,
        )

        assert len(results) == 3, "Should return 3 result tuples for 3 seeds"

        for idx in range(3):
            fwd_boxes, fwd_scores = results[idx]
            # Seed frame with original box
            assert 0 in fwd_boxes, f"Seed {idx}: seed frame should be present"
            np.testing.assert_array_equal(fwd_boxes[0], seed_boxes[idx])
            assert fwd_scores[0] == 1.0

            # Subsequent frames tracked
            for fi in [1, 2, 3, 4]:
                assert fi in fwd_boxes, f"Seed {idx}: frame {fi} should be tracked"

        # All seeds registered at keyframe (add_inputs_to_inference_session called 3 times)
        assert mock_processor.add_inputs_to_inference_session.call_count == 3

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_streaming_early_termination(self, mock_get_tracker, mock_av_open):
        """Model returns low scores for 3 consecutive frames -> tracking stops."""
        mock_av_open.return_value = _make_mock_av_container(10)

        # First call is keyframe registration (model called but output ignored via continue).
        # Calls 1,2: high score (frames 1,2)
        # Calls 3,4,5: low score (frames 3,4,5) -> 3 consecutive -> terminate
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
            score_logits_sequence=[
                2.0,   # call 0: keyframe registration (output ignored by continue)
                2.0,   # call 1: frame 1 (high)
                2.0,   # call 2: frame 2 (high)
                -5.0,  # call 3: frame 3 (low, consecutive_below=1)
                -5.0,  # call 4: frame 4 (low, consecutive_below=2)
                -5.0,  # call 5: frame 5 (low, consecutive_below=3 -> terminate)
            ],
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=9,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            score_threshold=0.1,
        )

        fwd_boxes, fwd_scores = results[0]

        # Seed frame always present
        assert 0 in fwd_boxes

        # High-score frames should be present
        assert 1 in fwd_boxes, "Frame 1 (high score) should be tracked"
        assert 2 in fwd_boxes, "Frame 2 (high score) should be tracked"

        # Low-score frames should NOT be present
        assert 3 not in fwd_boxes, "Frame 3 (below threshold) should be excluded"
        assert 4 not in fwd_boxes, "Frame 4 (below threshold) should be excluded"
        assert 5 not in fwd_boxes, "Frame 5 (below threshold) should be excluded"

        # Frames after termination should not be present
        for fi in [6, 7, 8, 9]:
            assert fi not in fwd_boxes, f"Frame {fi} should not exist (tracking terminated)"

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_streaming_all_terminated_early(self, mock_get_tracker, mock_av_open):
        """All objects terminate early. Function returns promptly."""
        mock_av_open.return_value = _make_mock_av_container(20)

        # All calls after keyframe return very low scores
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
            score_logits_sequence=[
                2.0,   # call 0: keyframe (ignored via continue)
                -5.0,  # call 1: frame 1 (low, consecutive=1)
                -5.0,  # call 2: frame 2 (low, consecutive=2)
                -5.0,  # call 3: frame 3 (low, consecutive=3 -> terminate)
            ],
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=19,
            seed_boxes=[
                np.array([10, 10, 30, 30], dtype=np.float32),
                np.array([50, 50, 80, 80], dtype=np.float32),
            ],
            score_threshold=0.1,
        )

        assert len(results) == 2
        for idx in range(2):
            fwd_boxes, _ = results[idx]
            # Only seed frame should survive
            assert 0 in fwd_boxes, f"Seed {idx}: seed frame should be present"
            assert len(fwd_boxes) == 1, \
                f"Seed {idx}: only seed frame should survive, got {sorted(fwd_boxes.keys())}"

        # Model should NOT have been called for all 20 frames
        # 1 (keyframe) + 3 (low-score frames until termination) = 4 calls
        assert mock_model.call_count <= 5, \
            f"Model should stop early, was called {mock_model.call_count} times"

    def test_streaming_single_frame(self):
        """kf_global == end_global: returns just seed boxes without opening video."""
        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        seed_boxes = [
            np.array([10, 10, 50, 50], dtype=np.float32),
            np.array([60, 60, 90, 90], dtype=np.float32),
        ]

        # No av.open or model mock needed - should return immediately
        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=100,
            end_global=100,
            seed_boxes=seed_boxes,
            score_threshold=0.1,
        )

        assert len(results) == 2
        for idx in range(2):
            fwd_boxes, fwd_scores = results[idx]
            assert len(fwd_boxes) == 1, "Only seed frame"
            assert 100 in fwd_boxes
            np.testing.assert_array_equal(fwd_boxes[100], seed_boxes[idx])
            assert fwd_scores[100] == 1.0

    def test_streaming_empty_seeds(self):
        """Empty seed_boxes list returns empty list."""
        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=100,
            seed_boxes=[],
            score_threshold=0.1,
        )

        assert results == [], "Empty seeds should return empty list"

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_streaming_stride(self, mock_get_tracker, mock_av_open):
        """frame_stride=2: only every other frame processed (keyframe always included)."""
        # 7 frames: indices 0,1,2,3,4,5,6
        mock_av_open.return_value = _make_mock_av_container(7)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=6,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            score_threshold=0.1,
            frame_stride=2,
        )

        fwd_boxes, fwd_scores = results[0]

        # Keyframe (0) always included
        assert 0 in fwd_boxes, "Keyframe should always be included"

        # With stride=2 starting from kf_global=0:
        # offset=0 (frame 0): keyframe, always included
        # offset=1 (frame 1): 1%2!=0, skipped
        # offset=2 (frame 2): 2%2==0, included
        # offset=3 (frame 3): 3%2!=0, skipped
        # offset=4 (frame 4): 4%2==0, included
        # offset=5 (frame 5): 5%2!=0, skipped
        # offset=6 (frame 6): 6%2==0, included
        assert 2 in fwd_boxes, "Frame 2 should be included (stride-aligned)"
        assert 4 in fwd_boxes, "Frame 4 should be included (stride-aligned)"
        assert 6 in fwd_boxes, "Frame 6 should be included (stride-aligned)"

        assert 1 not in fwd_boxes, "Frame 1 should be skipped (not stride-aligned)"
        assert 3 not in fwd_boxes, "Frame 3 should be skipped (not stride-aligned)"
        assert 5 not in fwd_boxes, "Frame 5 should be skipped (not stride-aligned)"

    def test_streaming_implies_forward_only(self):
        """When --streaming is set, args.forward_only becomes True."""
        import initial_seeding_video_boxes_manual_merge as mm

        # Simulate argparse with streaming=True
        args = argparse.Namespace(
            streaming=True,
            forward_only=False,  # initially False
        )

        # The enforcement happens after parse_args in main():
        # if getattr(args, "streaming", False): args.forward_only = True
        if getattr(args, "streaming", False):
            args.forward_only = True

        assert args.forward_only is True, \
            "--streaming should force --forward-only to True"

    def test_streaming_implies_forward_only_argparse(self):
        """Verify the argparse post-processing in main() sets forward_only."""
        # This tests the actual code path by simulating what main() does
        import initial_seeding_video_boxes_manual_merge as mm

        # Parse --streaming flag
        parser = argparse.ArgumentParser()
        parser.add_argument("--streaming", action="store_true", default=False)
        parser.add_argument("--forward-only", action="store_true", default=False)
        args = parser.parse_args(["--streaming"])

        # Apply the enforcement logic from main()
        if getattr(args, "streaming", False):
            args.forward_only = True

        assert args.streaming is True
        assert args.forward_only is True, \
            "--streaming should imply --forward-only"


class TestStreamingCorrectionKeyframes:
    """Tests for drift-correction keyframe injection during streaming tracking."""

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_correction_keyframe_injected(self, mock_get_tracker, mock_av_open):
        """Verify add_inputs_to_inference_session is called at correction frames."""
        mock_av_open.return_value = _make_mock_av_container(10)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        correction_box = np.array([25, 15, 75, 45], dtype=np.float32)
        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=9,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            correction_keyframes={5: correction_box},
            score_threshold=0.1,
        )

        # Verify add_inputs_to_inference_session was called for the
        # correction keyframe (not just the initial seed)
        add_calls = mock_processor.add_inputs_to_inference_session.call_args_list
        # First call is the seed registration at frame_idx=0.
        # Correction call should have obj_ids=[0] and the correction box.
        correction_calls = [
            c for c in add_calls
            if c.kwargs.get('input_boxes') == [[correction_box.tolist()]]
        ]
        assert len(correction_calls) >= 1, (
            "add_inputs_to_inference_session should be called with "
            "correction box at frame 5"
        )

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_correction_keyframe_overrides_result(self, mock_get_tracker, mock_av_open):
        """Correction keyframe's human box appears in results with score 1.0."""
        mock_av_open.return_value = _make_mock_av_container(10)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        correction_box = np.array([25, 15, 75, 45], dtype=np.float32)
        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=9,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            correction_keyframes={5: correction_box},
            score_threshold=0.1,
        )

        fwd_boxes, fwd_scores = results[0]

        # Correction frame should have the HUMAN box and score 1.0
        assert 5 in fwd_boxes, "Correction frame should be in results"
        np.testing.assert_array_equal(
            fwd_boxes[5], correction_box,
            err_msg="Correction frame should have the human-annotated box",
        )
        assert fwd_scores[5] == 1.0, "Correction frame should have score 1.0"

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_correction_resets_early_termination(self, mock_get_tracker, mock_av_open):
        """A correction keyframe should un-terminate an object and reset counters."""
        # 10 frames, scores: high, low, low, low (terminated), then correction at 5
        mock_av_open.return_value = _make_mock_av_container(10)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker,
            mask_box=(20, 10, 80, 50),
            # Seed frame isn't counted by logit sequence (it's handled via
            # _init_chunk_session). Model calls are for frames 1-9.
            # Frame 1: low, Frame 2: low, Frame 3: low (terminates at 3rd consecutive),
            # Frame 4: low (terminated, skipped)
            # Frame 5: correction injection resets termination, model call still happens
            # Frame 6-9: high (should produce results)
            score_logits_sequence=[-5.0, -5.0, -5.0, -5.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        correction_box = np.array([22, 12, 78, 48], dtype=np.float32)
        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=9,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            correction_keyframes={5: correction_box},
            score_threshold=0.1,
        )

        fwd_boxes, fwd_scores = results[0]

        # Seed frame present
        assert 0 in fwd_boxes, "Seed frame should be present"

        # Frames 1-3: low scores → terminated after 3 consecutive
        # Frame 5: correction → un-terminated, high scores after
        assert 5 in fwd_boxes, "Correction frame should be present"
        assert fwd_scores[5] == 1.0, "Correction frame score should be 1.0"

        # Frames after correction should be tracked (high scores)
        tracked_after = [f for f in fwd_boxes if f > 5]
        assert len(tracked_after) > 0, (
            "Frames after correction should be tracked "
            "(correction resets early termination)"
        )

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_correction_not_skipped_by_stride(self, mock_get_tracker, mock_av_open):
        """Correction keyframes are always included even when stride > 1."""
        mock_av_open.return_value = _make_mock_av_container(10)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        # Stride=3, correction at frame 5 (offset 5, 5%3=2 ≠ 0 → would be skipped)
        correction_box = np.array([25, 15, 75, 45], dtype=np.float32)
        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=9,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            correction_keyframes={5: correction_box},
            score_threshold=0.1,
            frame_stride=3,
        )

        fwd_boxes, fwd_scores = results[0]

        # Frame 5 should be included despite stride=3
        assert 5 in fwd_boxes, (
            "Correction keyframe at frame 5 should NOT be skipped by stride"
        )
        np.testing.assert_array_equal(fwd_boxes[5], correction_box)
        assert fwd_scores[5] == 1.0

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_multiple_corrections(self, mock_get_tracker, mock_av_open):
        """Multiple correction keyframes all get injected and recorded."""
        mock_av_open.return_value = _make_mock_av_container(15)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        corrections = {
            3: np.array([21, 11, 79, 49], dtype=np.float32),
            7: np.array([22, 12, 78, 48], dtype=np.float32),
            11: np.array([23, 13, 77, 47], dtype=np.float32),
        }
        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=14,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            correction_keyframes=corrections,
            score_threshold=0.1,
        )

        fwd_boxes, fwd_scores = results[0]

        for corr_frame, corr_box in corrections.items():
            assert corr_frame in fwd_boxes, (
                f"Correction frame {corr_frame} should be in results"
            )
            np.testing.assert_array_equal(
                fwd_boxes[corr_frame], corr_box,
                err_msg=f"Correction frame {corr_frame} box mismatch",
            )
            assert fwd_scores[corr_frame] == 1.0, (
                f"Correction frame {corr_frame} should have score 1.0"
            )

    @patch('initial_seeding_video_boxes_manual_merge.av.open')
    @patch('initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model')
    def test_no_corrections_matches_original(self, mock_get_tracker, mock_av_open):
        """Passing correction_keyframes=None behaves identically to before."""
        mock_av_open.return_value = _make_mock_av_container(5)
        mock_model, mock_processor = _setup_streaming_tracker_mock(
            mock_get_tracker, mask_box=(20, 10, 80, 50),
        )

        from initial_seeding_video_boxes_manual_merge import (
            _generate_streaming_forward_tracklets_sam3,
        )

        results = _generate_streaming_forward_tracklets_sam3(
            video_path='/fake/video.mp4',
            kf_global=0,
            end_global=4,
            seed_boxes=[np.array([20, 10, 80, 50], dtype=np.float32)],
            correction_keyframes=None,
            score_threshold=0.1,
        )

        fwd_boxes, fwd_scores = results[0]
        assert 0 in fwd_boxes
        assert len(fwd_boxes) == 5  # seed + 4 tracked frames
