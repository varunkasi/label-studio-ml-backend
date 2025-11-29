"""Grounding DINO inference helpers for image detection and video tracking.

This module wraps Grounding DINO utilities so the ML backend can expose
YOLO-compatible behaviours while relying on zero-shot text prompted detections.
"""

from __future__ import annotations

import inspect
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

import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, predict, get_phrases_from_posmap

try:
    from torch.cuda.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        logger.info("Prepared tracker kwargs (passed to ByteTrack): %s", tracker_kwargs)
        tracker = sv.ByteTrack(**tracker_kwargs)

        internal_tracker_config = {
            "track_activation_threshold": tracker.track_activation_threshold,
            "minimum_matching_threshold": tracker.minimum_matching_threshold,
            "minimum_consecutive_frames": tracker.minimum_consecutive_frames,
            "lost_track_buffer": int(tracker.max_time_lost),
            "frame_rate": fps,
        }
        logger.info("ByteTrack internal config: %s", internal_tracker_config)
        frames: List[FrameDetections] = []
        latencies_ms: List[float] = []
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
        logger.info("ByteTrack internal config at completion: %s", internal_tracker_config)
        logger.info(
            "Detection stats: avg=%.1f/frame (nonzero=%.1f), min=%d, max=%d, fragmentation_ratio=%.1f",
            avg_det_per_frame,
            avg_nonzero,
            min_det,
            max_det,
            fragmentation_ratio,
        )

        return VideoTrackingResult(
            frames_count=frame_count,
            duration=duration,
            fps=fps,
            frames=frames,
        )

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
