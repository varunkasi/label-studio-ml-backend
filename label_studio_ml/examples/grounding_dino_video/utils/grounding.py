"""Grounding DINO inference helpers for image detection and video tracking.

This module wraps Grounding DINO utilities so the ML backend can expose
YOLO-compatible behaviours while relying on zero-shot text prompted detections.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, predict, get_phrases_from_posmap

try:
    from torch.cuda.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%s, using default %.2f", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        logger.warning("Invalid int for %s=%s, using default %d", name, value, default)
        return default


@dataclass
class AutotuneConfig:
    expected_detections_per_frame: float
    expected_unique_tracks: int
    frame_limit: int
    max_trials: int
    detection_weight: float = 0.6
    track_weight: float = 0.4

    def normalized_weights(self) -> Tuple[float, float]:
        total = self.detection_weight + self.track_weight
        if total <= 0:
            return 0.5, 0.5
        return self.detection_weight / total, self.track_weight / total


def _load_autotune_config() -> Optional[AutotuneConfig]:
    if not _env_flag("TRACKER_AUTOTUNE_ENABLED", False):
        return None

    config = AutotuneConfig(
        expected_detections_per_frame=_env_float("EXPECTED_DETECTIONS_PER_FRAME_1000", 5.0),
        expected_unique_tracks=_env_int("EXPECTED_UNIQUE_TRACKS_1000", 8),
        frame_limit=max(1, _env_int("TRACKER_AUTOTUNE_FRAME_LIMIT", 1000)),
        max_trials=max(1, _env_int("TRACKER_AUTOTUNE_TRIALS", 8)),
        detection_weight=_env_float("TRACKER_AUTOTUNE_DETECTION_WEIGHT", 0.6),
        track_weight=_env_float("TRACKER_AUTOTUNE_TRACK_WEIGHT", 0.4),
    )
    return config


def _score_autotune_trial(
    avg_detections: float,
    unique_tracks: int,
    config: AutotuneConfig,
) -> float:
    det_target = max(config.expected_detections_per_frame, 1e-3)
    recall_ratio = avg_detections / det_target
    recall_score = min(recall_ratio, 1.5)

    if config.expected_unique_tracks > 0:
        track_error = abs(unique_tracks - config.expected_unique_tracks) / config.expected_unique_tracks
        track_score = max(1.0 - track_error, 0.0)
    else:
        track_score = 1.0

    det_weight, track_weight = config.normalized_weights()
    return (recall_score * det_weight) + (track_score * track_weight)


@dataclass
class FrameDetections:
    """Container for detections produced for a single video frame."""

    frame_index: int
    height: int
    width: int
    detections: sv.Detections


@dataclass
class VideoTrackingResult:
    """Aggregated detections over a video sequence."""

    frames_count: int
    duration: float
    fps: float
    frames: List[FrameDetections]


class GroundingDINOInference:
    """Singleton-style helper around Grounding DINO inference primitives."""

    _instance: Optional["GroundingDINOInference"] = None

    def __init__(self) -> None:
        repo_root = Path(os.getenv("GROUNDINGDINO_REPO_PATH", "/GroundingDINO"))
        config_name = os.getenv("GROUNDING_DINO_CONFIG", "GroundingDINO_SwinT_OGC.py")
        weights_name = os.getenv("GROUNDING_DINO_WEIGHTS", "groundingdino_swint_ogc.pth")

        self.prompt = self._resolve_prompt()
        self.labels = self._resolve_labels()
        self.names: Dict[int, str] = {index: label for index, label in enumerate(self.labels)}
        self.allowed_terms = [label.lower() for label in self.labels]

        self.device = os.getenv("GROUNDING_DINO_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        config_path = repo_root / "groundingdino" / "config" / config_name
        weights_path = repo_root / "weights" / weights_name

        if self.device.startswith("cuda") and torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_props = torch.cuda.get_device_properties(current_device)
            logger.info(
                "Using CUDA device index=%d name='%s' total_memory=%.2f GB",
                current_device,
                device_props.name,
                device_props.total_memory / (1024 ** 3),
            )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Grounding DINO config not found at {config_path}. Configure "
                "GROUNDING_DINO_CONFIG or GROUNDINGDINO_REPO_PATH appropriately."
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Grounding DINO weights not found at {weights_path}. Configure "
                "GROUNDING_DINO_WEIGHTS or ensure weights are downloaded."
            )

        logger.info(
            "Loading Grounding DINO from config '%s' and weights '%s' on device '%s'",
            config_path,
            weights_path,
            self.device,
        )
        self.model = load_model(
            model_config_path=str(config_path),
            model_checkpoint_path=str(weights_path),
            device=self.device,
        )
        
        # Enable mixed precision for faster inference if available
        self.use_amp = AMP_AVAILABLE and self.device.startswith("cuda")
        if self.use_amp:
            logger.info("Mixed precision (FP16) enabled for faster inference")

        # Note: torch.compile is disabled for Grounding DINO due to compatibility issues
        # with the Swin Transformer backbone's dynamic slicing operations
        
        # Resolution settings for inference
        # For batched inference, we need fixed sizes (not RandomResize)
        self.max_size = int(os.getenv("GROUNDING_DINO_MAX_SIZE", "1333"))
        self.base_size = int(os.getenv("GROUNDING_DINO_BASE_SIZE", "800"))
        
        # Standard transform for inference
        self.transform = T.Compose(
            [
                T.RandomResize([self.base_size], max_size=self.max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        # Initialize annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)

    @classmethod
    def get_instance(cls) -> "GroundingDINOInference":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _resolve_prompt() -> str:
        prompt = os.getenv("GROUNDING_DINO_PROMPT")
        if prompt:
            prompt = prompt.strip()
        if not prompt:
            labels = os.getenv("GROUNDING_DINO_LABELS", "person")
            label_list = [token.strip() for token in labels.split(",") if token.strip()]
            prompt = ". ".join(label_list) + "."
        if not prompt.endswith("."):
            prompt = prompt + "."
        return prompt

    @staticmethod
    def _resolve_labels() -> List[str]:
        raw = os.getenv("GROUNDING_DINO_LABELS")
        if raw:
            tokens = [token.strip().strip(".") for token in raw.split(",") if token.strip()]
        else:
            prompt = os.getenv("GROUNDING_DINO_PROMPT", "person")
            tokens = [token.strip().strip(".") for token in prompt.split(".") if token.strip()]
        if not tokens:
            tokens = ["person"]
        return tokens

    def detect_image(
        self,
        path: str,
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Tuple[sv.Detections, int, int]:
        image = Image.open(path).convert("RGB")
        np_frame = np.asarray(image)
        detections = self.infer_frame(
            np_frame,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        height, width = np_frame.shape[:2]
        return detections, height, width

    def infer_frame(
        self,
        frame: np.ndarray,
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Perform inference on a single frame.

        Args:
            frame: Input frame as numpy array (RGB format)
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            supervision Detections object
        """
        return self._infer_frame_standard(
            frame,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    def _infer_frame_standard(
        self,
        frame: np.ndarray,
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Standard full-frame Grounding DINO inference on a single frame.

        Args:
            frame: Input frame as numpy array (RGB format)
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            supervision Detections object
        """
        prompt = (prompt or self.prompt).strip()
        if not prompt.endswith("."):
            prompt = prompt + "."

        tensor, _ = self.transform(Image.fromarray(frame).convert("RGB"), None)
        tensor = tensor.to(self.device)

        # logger.info(
        #     "Grounding DINO inference using device=%s (tensor device=%s)",
        #     self.device,
        #     tensor.device,
        # )

        box_threshold = (
            float(box_threshold)
            if box_threshold is not None
            else float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", 0.35))
        )
        text_threshold = (
            float(text_threshold)
            if text_threshold is not None
            else float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", 0.25))
        )

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    boxes, scores, phrases = predict(
                        model=self.model,
                        image=tensor,
                        caption=prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=self.device,
                    )
            else:
                boxes, scores, phrases = predict(
                    model=self.model,
                    image=tensor,
                    caption=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=self.device,
                )

        if boxes.numel() == 0:
            return self._empty_detections()

        height, width = frame.shape[:2]
        xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([width, height, width, height])
        xyxy = xyxy.cpu().numpy().astype(np.float32)
        confidences = scores.cpu().numpy().astype(np.float32)

        matches = self._filter_detections(phrases, xyxy, confidences)
        if not matches:
            return self._empty_detections()

        filtered_xyxy, filtered_scores, class_ids = zip(*matches)
        detections = sv.Detections(
            xyxy=np.stack(filtered_xyxy, axis=0),
            confidence=np.array(filtered_scores, dtype=np.float32),
            class_id=np.array(class_ids, dtype=np.int32),
        )
        return detections

    def track_video(
        self,
        path: str,
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        tracker_kwargs: Optional[Dict] = None,
        max_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
        save_frames: bool = False,
        tracker_scenarios: Optional[List[Dict[str, Any]]] = None,
        _skip_autotune: bool = False,
    ) -> VideoTrackingResult:
        """Track objects in a video running inference on every frame.
        
        Args:
            path: Path to video file
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            tracker_kwargs: Arguments for ByteTrack tracker
            max_frames: Stop after processing this many frames (for quick testing)
            output_dir: Directory to save annotated frames (required if save_frames=True)
            save_frames: Whether to save annotated frames with bounding boxes
        """
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

        # Setup output directory if saving frames
        if save_frames:
            if not output_dir:
                raise ValueError("output_dir must be specified when save_frames=True")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving annotated frames to: %s", output_path)

        # Apply max_frames limit if specified
        effective_frame_count = frame_count
        if max_frames is not None and max_frames > 0:
            effective_frame_count = min(frame_count, max_frames) if frame_count else max_frames
            logger.info("Limiting to %d frames (--max-frames)", max_frames)

        logger.info(
            "Starting video tracking: path=%s, total_frames=%s, fps=%.2f",
            path,
            effective_frame_count or "unknown",
            fps,
        )

        progress_every = os.getenv("GROUNDING_DINO_PROGRESS_EVERY", "25")
        try:
            progress_every_int = max(int(progress_every), 0)
        except ValueError:
            logger.warning(
                "Invalid GROUNDING_DINO_PROGRESS_EVERY value '%s', falling back to 25",
                progress_every,
            )
            progress_every_int = 25

        tracker_kwargs = dict(tracker_kwargs or {})
        tracker_kwargs["frame_rate"] = fps
        tracker_kwargs = self._prepare_tracker_kwargs(tracker_kwargs)

        autotune_summary: Optional[Dict[str, Any]] = None
        if not _skip_autotune:
            tracker_kwargs, autotune_summary = self._maybe_autotune_tracker(
                path=path,
                base_kwargs=tracker_kwargs,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_frames=max_frames,
                tracker_scenarios=tracker_scenarios,
            )

        tracker = sv.ByteTrack(**tracker_kwargs)
        if autotune_summary:
            logger.info(
                "Optuna autotune selected tracker parameters: %s (score=%.3f across %d trials)",
                autotune_summary.get("best_params"),
                autotune_summary.get("best_score"),
                autotune_summary.get("trials"),
            )

        if tracker_scenarios:
            return self._compare_trackers(
                path=path,
                base_kwargs=tracker_kwargs,
                scenarios=tracker_scenarios,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                tracker_kwargs=tracker_kwargs,
                max_frames=max_frames,
                output_dir=output_dir,
                save_frames=save_frames,
            )

        internal_tracker_config = self._extract_tracker_config(tracker, fps)
        frames: List[FrameDetections] = []
        latencies_ms: List[float] = []
        raw_detection_counts: List[int] = []
        device_is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        tracked_ids: Set[int] = set()
        
        frame_index = 0

        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    logger.info("Reached max_frames limit (%d), stopping early", max_frames)
                    break

                ret, frame = capture.read()
                if not ret:
                    break
                # Progress logging
                if progress_every_int and frame_index % progress_every_int == 0:
                    logger.info(
                        "Tracking progress: frame %d/%s",
                        frame_index,
                        frame_count or "?",
                    )

                if device_is_cuda:
                    torch.cuda.synchronize()
                frame_start_time = time.perf_counter()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self.infer_frame(
                    frame_rgb,
                    prompt=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                raw_detection_counts.append(detections.xyxy.shape[0])

                tracked = tracker.update_with_detections(detections)

                if tracked.tracker_id is not None:
                    for tracker_id in tracked.tracker_id:
                        if tracker_id is None:
                            continue
                        try:
                            tracked_ids.add(int(tracker_id))
                        except (TypeError, ValueError):
                            continue

                frames.append(
                    FrameDetections(
                        frame_index=frame_index,
                        height=frame.shape[0],
                        width=frame.shape[1],
                        detections=tracked,
                    )
                )

                if save_frames:
                    annotated_frame = self._annotate_frame(frame, detections, frame_index)
                    frame_filename = output_path / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(frame_filename), annotated_frame)

                if device_is_cuda:
                    torch.cuda.synchronize()
                    gpu_mem_mib = torch.cuda.memory_allocated() / (1024 ** 2)
                    gpu_mem_reserved_mib = torch.cuda.memory_reserved() / (1024 ** 2)
                else:
                    gpu_mem_mib = None
                    gpu_mem_reserved_mib = None

                frame_time_ms = (time.perf_counter() - frame_start_time) * 1000.0
                latencies_ms.append(frame_time_ms)

                logger.debug(
                    "Processed frame %d: detections=%d, latency=%.1f ms%s",
                    frame_index,
                    tracked.xyxy.shape[0],
                    frame_time_ms,
                    f", gpu_mem={gpu_mem_mib:.1f}/{gpu_mem_reserved_mib:.1f} MiB"
                    if gpu_mem_mib is not None else "",
                )

                frame_index += 1

        except Exception as e:
            logger.error("Error during video tracking: %s", e)
            raise
        finally:
            capture.release()

        if frame_count == 0:
            frame_count = len(frames)
        duration = frame_count / fps if fps else float(frame_count)

        total_detections = sum(frame.detections.xyxy.shape[0] for frame in frames)
        total_tracks = len(tracked_ids)
        avg_latency = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
        max_latency = max(latencies_ms) if latencies_ms else 0.0
        
        # Compute detection stats
        det_counts = [frame.detections.xyxy.shape[0] for frame in frames]
        avg_det_per_frame = sum(det_counts) / len(det_counts) if det_counts else 0
        min_det = min(det_counts) if det_counts else 0
        max_det = max(det_counts) if det_counts else 0
        
        # Stats for frames with detections (effective recall)
        nonzero_counts = [c for c in det_counts if c > 0]
        avg_nonzero = sum(nonzero_counts) / len(nonzero_counts) if nonzero_counts else 0
        
        # Track fragmentation ratio: ideal is 1.0 (one track per object)
        # Higher means more fragmentation
        fragmentation_ratio = total_tracks / avg_det_per_frame if avg_det_per_frame > 0 else 0

        logger.info(
            "Completed video tracking: processed=%d frames, detections=%d, tracks=%d, avg_latency=%.1f ms/frame, max_latency=%.1f ms/frame",
            len(frames),
            total_detections,
            total_tracks,
            avg_latency,
            max_latency,
        )
        logger.debug("ByteTrack internal config at completion: %s", internal_tracker_config)
        logger.info(
            "Detection stats: avg=%.1f/frame (nonzero=%.1f), min=%d, max=%d, fragmentation_ratio=%.1f",
            avg_det_per_frame,
            avg_nonzero,
            min_det,
            max_det,
            fragmentation_ratio,
        )

        if raw_detection_counts:
            raw_total = sum(raw_detection_counts)
            raw_avg = raw_total / len(raw_detection_counts)
            raw_nonzero = (
                sum(count for count in raw_detection_counts if count > 0) / len(raw_detection_counts)
                if any(raw_detection_counts) else 0.0
            )
            raw_min = min(raw_detection_counts)
            raw_max = max(raw_detection_counts)
            logger.info(
                "Raw detection stats (pre-tracker): avg=%.1f/frame (nonzero_avg=%.1f), min=%d, max=%d, total=%d",
                raw_avg,
                raw_nonzero,
                raw_min,
                raw_max,
                raw_total,
            )

        return VideoTrackingResult(
            frames_count=frame_count,
            duration=duration,
            fps=fps,
            frames=frames,
        )

    @staticmethod
    def _extract_tracker_config(tracker: sv.ByteTrack, fps: float) -> Dict[str, Any]:
        return {
            "track_activation_threshold": tracker.track_activation_threshold,
            "minimum_matching_threshold": tracker.minimum_matching_threshold,
            "minimum_consecutive_frames": tracker.minimum_consecutive_frames,
            "lost_track_buffer": int(tracker.max_time_lost),
            "frame_rate": fps,
        }

    def _compare_trackers(
        self,
        *,
        path: str,
        base_kwargs: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
        prompt: Optional[str],
        box_threshold: Optional[float],
        text_threshold: Optional[float],
        tracker_kwargs: Dict[str, Any],
        max_frames: Optional[int],
        output_dir: Optional[str],
        save_frames: bool,
    ) -> VideoTrackingResult:
        """Run multiple tracker configs and log comparative stats."""
        results = []
        last_result: Optional[VideoTrackingResult] = None

        for idx, overrides in enumerate(scenarios, start=1):
            scenario_kwargs = {**base_kwargs, **overrides}
            label = overrides.get("label", f"scenario_{idx}")
            logger.info("Running tracker scenario '%s' overrides=%s", label, overrides)

            scenario_result, unique_tracks = self._run_single_tracker(
                path=path,
                tracker_kwargs=scenario_kwargs,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_frames=max_frames,
                output_dir=output_dir,
                save_frames=save_frames,
                skip_autotune=True,
            )
            stats = {
                "label": label,
                "frames": len(scenario_result.frames),
                "total_detections": sum(frame.detections.xyxy.shape[0] for frame in scenario_result.frames),
                "unique_tracks": unique_tracks,
            }
            logger.info("Scenario '%s' summary: %s", label, stats)
            results.append({"scenario": label, "stats": stats, "config": scenario_kwargs})
            last_result = scenario_result

        logger.info("Tracker scenario comparison: %s", json.dumps(results, indent=2))
        if last_result:
            return last_result

        fallback_result, _ = self._run_single_tracker(
            path=path,
            tracker_kwargs=base_kwargs,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_frames=max_frames,
            output_dir=output_dir,
            save_frames=save_frames,
            skip_autotune=True,
        )
        return fallback_result

    def _run_single_tracker(
        self,
        *,
        path: str,
        tracker_kwargs: Dict[str, Any],
        prompt: Optional[str],
        box_threshold: Optional[float],
        text_threshold: Optional[float],
        max_frames: Optional[int],
        output_dir: Optional[str],
        save_frames: bool,
        skip_autotune: bool = False,
    ) -> Tuple[VideoTrackingResult, int]:
        """Run tracking with a specific tracker configuration."""
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

        tracker_kwargs = dict(tracker_kwargs or {})
        tracker_kwargs["frame_rate"] = fps
        tracker_kwargs = self._prepare_tracker_kwargs(tracker_kwargs)
        tracker = sv.ByteTrack(**tracker_kwargs)

        capture.release()
        result = self.track_video(
            path,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            tracker_kwargs=tracker_kwargs,
            max_frames=max_frames,
            output_dir=output_dir,
            save_frames=save_frames,
            _skip_autotune=skip_autotune,
        )
        unique_tracks = {
            track_id
            for frame in result.frames
            if frame.detections.tracker_id is not None
            for track_id in frame.detections.tracker_id
            if track_id is not None
        }
        return result, len(unique_tracks)

    def _maybe_autotune_tracker(
        self,
        *,
        path: str,
        base_kwargs: Dict[str, Any],
        prompt: Optional[str],
        box_threshold: Optional[float],
        text_threshold: Optional[float],
        max_frames: Optional[int],
        tracker_scenarios: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        config = _load_autotune_config()
        if not config:
            return base_kwargs, None

        if tracker_scenarios:
            logger.info(
                "TRACKER_AUTOTUNE_ENABLED is set but tracker scenarios were provided; skipping autotune so scenarios can run.")
            return base_kwargs, None

        if optuna is None:
            logger.warning(
                "TRACKER_AUTOTUNE_ENABLED is set but Optuna is not installed inside the environment."
            )
            return base_kwargs, None

        trial_frame_limit = config.frame_limit
        if max_frames is not None:
            trial_frame_limit = min(trial_frame_limit, max_frames)

        logger.info(
            "Starting Optuna tracker autotune: trials=%d, frame_limit=%d, expected_det=%.2f, expected_tracks=%d",
            config.max_trials,
            trial_frame_limit,
            config.expected_detections_per_frame,
            config.expected_unique_tracks,
        )

        study = optuna.create_study(direction="maximize", study_name="byte_track_autotune")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "track_activation_threshold": trial.suggest_float("track_activation_threshold", 0.20, 0.55),
                "minimum_matching_threshold": trial.suggest_float("minimum_matching_threshold", 0.35, 0.75),
                "minimum_consecutive_frames": trial.suggest_int("minimum_consecutive_frames", 3, 12),
                "lost_track_buffer": trial.suggest_int("lost_track_buffer", 200, 2400, step=50),
            }
            candidate = {**base_kwargs, **params}
            trial_index = trial.number + 1
            logger.info(
                "Optuna trial %d/%d started with params=%s",
                trial_index,
                config.max_trials,
                params,
            )

            result, unique_tracks = self._run_single_tracker(
                path=path,
                tracker_kwargs=candidate,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_frames=trial_frame_limit,
                output_dir=None,
                save_frames=False,
                skip_autotune=True,
            )

            det_counts = [frame.detections.xyxy.shape[0] for frame in result.frames]
            avg_det = (sum(det_counts) / len(det_counts)) if det_counts else 0.0
            score = _score_autotune_trial(avg_det, unique_tracks, config)
            logger.info(
                "Optuna trial %d finished: avg_det=%.2f, unique_tracks=%d, score=%.3f",
                trial_index,
                avg_det,
                unique_tracks,
                score,
            )
            trial.set_user_attr("avg_det", avg_det)
            trial.set_user_attr("unique_tracks", unique_tracks)
            return score

        try:
            study.optimize(objective, n_trials=config.max_trials)
        except Exception as exc:
            logger.exception("Optuna autotune failed; falling back to original tracker parameters: %s", exc)
            return base_kwargs, None

        if not study.best_trials:
            logger.warning("Optuna autotune produced no successful trials; using original tracker parameters.")
            return base_kwargs, None

        best_trial = study.best_trial
        best_params = best_trial.params
        summary = {
            "best_params": best_params,
            "best_score": best_trial.value,
            "trials": len(study.trials),
        }
        logger.info(
            "Optuna autotune complete: best_score=%.3f, params=%s", best_trial.value, best_params
        )
        tuned_kwargs = {**base_kwargs, **best_params}
        return tuned_kwargs, summary

    def _prepare_tracker_kwargs(self, tracker_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tracker kwargs so they match the current supervision.ByteTrack signature."""

        signature = inspect.signature(sv.ByteTrack.__init__)
        accepted_params = {name for name in signature.parameters.keys() if name != "self"}

        normalized: Dict[str, Any] = {}
        alias_map = {
            "track_activation_threshold": ("track_activation_threshold", "track_thresh"),
            "lost_track_buffer": ("lost_track_buffer", "track_buffer"),
            "minimum_matching_threshold": ("minimum_matching_threshold", "match_thresh"),
            "minimum_consecutive_frames": ("minimum_consecutive_frames",),
        }

        for logical_key, aliases in alias_map.items():
            if logical_key not in tracker_kwargs:
                continue
            value = tracker_kwargs[logical_key]
            for alias in aliases:
                if alias in accepted_params:
                    normalized[alias] = value
                    break
            else:
                logger.debug(
                    "Dropping tracker parameter '%s'; unsupported by current supervision.ByteTrack",
                    logical_key,
                )

        for key, value in tracker_kwargs.items():
            if key in alias_map:
                continue
            if key in accepted_params:
                normalized[key] = value
            else:
                logger.debug(
                    "Dropping tracker parameter '%s'; unsupported by current supervision.ByteTrack",
                    key,
                )

        return normalized

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        frame_index: int,
    ) -> np.ndarray:
        """Annotate a frame with bounding boxes, labels, and tracking IDs.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            detections: Detection results with bounding boxes, labels, and tracker IDs
            frame_index: Current frame number
            
        Returns:
            Annotated frame with visualizations
        """
        annotated = frame.copy()
        
        if len(detections.xyxy) > 0:
            # Create labels with class names, confidence scores, and tracker IDs
            labels = []
            for idx in range(len(detections.xyxy)):
                class_id = int(detections.class_id[idx]) if detections.class_id is not None else 0
                confidence = float(detections.confidence[idx]) if detections.confidence is not None else 0.0
                tracker_id = int(detections.tracker_id[idx]) if detections.tracker_id is not None else None
                
                class_name = self.names.get(class_id, f"class_{class_id}")
                
                if tracker_id is not None:
                    label = f"#{tracker_id} {class_name} {confidence:.2f}"
                else:
                    label = f"{class_name} {confidence:.2f}"
                labels.append(label)
            
            # Annotate with boxes and labels
            annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
            annotated = self.label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
        
        # Add frame number in top-left corner
        cv2.putText(
            annotated,
            f"Frame: {frame_index}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        
        return annotated

    def _filter_detections(
        self,
        phrases: Iterable[str],
        xyxy: np.ndarray,
        scores: np.ndarray,
    ) -> List[Tuple[np.ndarray, float, int]]:
        matches: List[Tuple[np.ndarray, float, int]] = []

        for phrase, bbox, score in zip(phrases, xyxy, scores):
            class_id = self._match_class(phrase.lower())
            if class_id is None:
                continue
            matches.append((bbox, float(score), class_id))

        return matches
        
    def _match_class(self, phrase: str) -> Optional[int]:
        for index, term in enumerate(self.allowed_terms):
            if term in phrase:
                return index
        return None

    @staticmethod
    def _empty_detections() -> sv.Detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
        )


__all__ = ["FrameDetections", "GroundingDINOInference", "VideoTrackingResult"]
