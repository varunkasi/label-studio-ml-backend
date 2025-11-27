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
        
        # Standard transform with RandomResize (for single-frame inference)
        self.transform = T.Compose(
            [
                T.RandomResize([self.base_size], max_size=self.max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        # Fixed-size transform for batched inference (all frames must be same size)
        # We use ResizeDebug which takes a fixed (h, w) tuple
        self.batch_transform = T.Compose(
            [
                T.ResizeDebug((self.base_size, self.max_size)),  # Fixed size for batching
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

    def infer_batch(
        self,
        frames: List[np.ndarray],
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[sv.Detections]:
        """Perform TRUE batched inference on multiple frames for better GPU utilization.

        This method stacks all frames into a single batch tensor and runs a single
        forward pass through the model, significantly improving GPU utilization
        compared to sequential frame-by-frame inference.

        Args:
            frames: List of frames (numpy arrays in RGB/BGR format from OpenCV)
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            List of detections, one per frame
        """
        if not frames:
            return []

        # For single frame, use standard inference (no batching overhead)
        if len(frames) == 1:
            return [self.infer_frame(
                frames[0],
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )]

        resolved_prompt = (prompt or self.prompt).strip()
        if not resolved_prompt.endswith("."):
            resolved_prompt = resolved_prompt + "."

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

        # Store original frame dimensions for coordinate scaling
        frame_sizes: List[Tuple[int, int]] = []  # (height, width) for each frame

        # Transform all frames to fixed-size tensors and stack into batch
        tensors: List[torch.Tensor] = []
        for frame in frames:
            h, w = frame.shape[:2]
            frame_sizes.append((h, w))
            
            # Convert BGR (OpenCV) to RGB PIL Image
            pil_image = Image.fromarray(frame).convert("RGB")
            tensor, _ = self.batch_transform(pil_image, None)
            tensors.append(tensor)

        # Stack into batch tensor: (batch_size, 3, H, W)
        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        batch_size = batch_tensor.shape[0]

        # Run batched inference
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        batch_tensor,
                        captions=[resolved_prompt] * batch_size,
                    )
            else:
                outputs = self.model(
                    batch_tensor,
                    captions=[resolved_prompt] * batch_size,
                )

        # outputs["pred_logits"]: (batch_size, num_queries, 256)
        # outputs["pred_boxes"]: (batch_size, num_queries, 4) in cxcywh normalized format
        pred_logits = outputs["pred_logits"].cpu().sigmoid()
        pred_boxes = outputs["pred_boxes"].cpu()

        # Process each frame's predictions
        detections_list: List[sv.Detections] = []
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(resolved_prompt)

        for i in range(batch_size):
            frame_logits = pred_logits[i]  # (num_queries, 256)
            frame_boxes = pred_boxes[i]    # (num_queries, 4)
            orig_h, orig_w = frame_sizes[i]

            # Filter by box threshold
            max_logits = frame_logits.max(dim=1)[0]
            mask = max_logits > box_threshold

            if not mask.any():
                detections_list.append(self._empty_detections())
                continue

            filtered_logits = frame_logits[mask]
            filtered_boxes = frame_boxes[mask]
            filtered_scores = max_logits[mask]

            # Convert boxes from cxcywh normalized to xyxy pixel coordinates
            # The boxes are normalized to the resized image, need to scale to original
            xyxy = box_ops.box_cxcywh_to_xyxy(filtered_boxes)
            xyxy = xyxy * torch.tensor([orig_w, orig_h, orig_w, orig_h])
            xyxy = xyxy.numpy().astype(np.float32)
            confidences = filtered_scores.numpy().astype(np.float32)

            # Extract phrases for each detection
            phrases = []
            for logit in filtered_logits:
                phrase = get_phrases_from_posmap(
                    logit > text_threshold, tokenized, tokenizer
                ).replace(".", "")
                phrases.append(phrase)

            # Filter by allowed labels
            matches = self._filter_detections(phrases, xyxy, confidences)
            if not matches:
                detections_list.append(self._empty_detections())
                continue

            filtered_xyxy, filtered_scores_final, class_ids = zip(*matches)
            detections = sv.Detections(
                xyxy=np.stack(filtered_xyxy, axis=0),
                confidence=np.array(filtered_scores_final, dtype=np.float32),
                class_id=np.array(class_ids, dtype=np.int32),
            )
            detections_list.append(detections)

        return detections_list

    def track_video(
        self,
        path: str,
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        tracker_kwargs: Optional[Dict] = None,
        batch_size: Optional[int] = None,
        output_dir: Optional[str] = None,
        save_frames: bool = False,
    ) -> VideoTrackingResult:
        """Track objects in a video with optional batch processing and frame saving.
        
        Args:
            path: Path to video file
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            tracker_kwargs: Arguments for ByteTrack tracker
            batch_size: Number of frames to process in parallel (default from env or 8)
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

        logger.info(
            "Starting video tracking: path=%s, total_frames=%s, fps=%.2f",
            path,
            frame_count or "unknown",
            fps,
        )

        # Get batch size from parameter or environment.
        # Default to 8 for true batched inference (significant GPU utilization improvement)
        if batch_size is None:
            env_batch = os.getenv("GROUNDING_DINO_BATCH_SIZE")
            if env_batch is None:
                batch_size = 8  # Default to batched inference for better GPU utilization
            else:
                try:
                    batch_size = int(env_batch)
                except ValueError:
                    logger.warning(
                        "Invalid GROUNDING_DINO_BATCH_SIZE value '%s', falling back to 8",
                        env_batch,
                    )
                    batch_size = 8
        if batch_size <= 0:
            logger.warning("Received non-positive batch_size=%d, defaulting to 8", batch_size)
            batch_size = 8

        use_batch_inference = batch_size > 1
        logger.info(
            "Effective batch size: %d (%s)",
            batch_size,
            "true batched inference" if use_batch_inference else "single-frame inference",
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
        print("Prepared tracker kwargs:", tracker_kwargs)
        tracker = sv.ByteTrack(**tracker_kwargs)
        print("track_activation_threshold:", tracker.track_activation_threshold)
        print("lost_track_buffer:", tracker.max_time_lost * 30.0 / fps)
        print("minimum_matching_threshold:", tracker.minimum_matching_threshold)
        print("minimum_consecutive_frames:", tracker.minimum_consecutive_frames)
        frames: List[FrameDetections] = []
        latencies_ms: List[float] = []
        device_is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        tracked_ids: Set[int] = set()
        
        # Process frames in streaming batches to avoid memory issues
        frame_index = 0
        batch_frames: List[np.ndarray] = []
        batch_indices: List[int] = []
        
        try:
            while True:
                while len(batch_frames) < batch_size:
                    ret, frame = capture.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                    batch_indices.append(frame_index)
                    frame_index += 1

                if len(batch_frames) == 0:
                    break

                if progress_every_int and (frame_index - len(batch_frames)) % (
                    progress_every_int * batch_size
                ) < batch_size:
                    logger.info(
                        "Tracking progress: frames %d/%s",
                        frame_index,
                        frame_count or "?",
                    )

                if device_is_cuda:
                    torch.cuda.synchronize()
                batch_start_time = time.perf_counter()

                batch_detections_count = 0
                print(
                    "Batch thresholds:",
                    "box_threshold=", box_threshold,
                    "text_threshold=", text_threshold,
                )
                process_as_batch = use_batch_inference and len(batch_frames) > 1
                detections_payload: List[Tuple[np.ndarray, int, sv.Detections]] = []

                if process_as_batch:
                    try:
                        batch_detections_list = self.infer_batch(
                            batch_frames,
                            prompt=prompt,
                            box_threshold=box_threshold,
                            text_threshold=text_threshold,
                        )
                        detections_payload = list(
                            zip(batch_frames, batch_indices, batch_detections_list)
                        )
                    except Exception as batch_error:
                        logger.error(
                            "Error processing batch starting at frame %d: %s",
                            batch_indices[0] if batch_indices else 0,
                            batch_error,
                        )
                        process_as_batch = False

                if not process_as_batch:
                    detections_payload = []
                    for frame, frame_idx in zip(batch_frames, batch_indices):
                        try:
                            detections = self.infer_frame(
                                frame,
                                prompt=prompt,
                                box_threshold=box_threshold,
                                text_threshold=text_threshold,
                            )
                        except Exception as frame_error:
                            logger.error(
                                "Error processing frame %d: %s", frame_idx, frame_error
                            )
                            continue
                        detections_payload.append((frame, frame_idx, detections))

                for frame, frame_idx, detections in detections_payload:
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
                            frame_index=frame_idx,
                            height=frame.shape[0],
                            width=frame.shape[1],
                            detections=tracked,
                        )
                    )

                    batch_detections_count += tracked.xyxy.shape[0]

                    if save_frames:
                        annotated_frame = self._annotate_frame(frame, tracked, frame_idx)
                        frame_filename = output_path / f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(frame_filename), annotated_frame)

                if device_is_cuda:
                    torch.cuda.synchronize()
                    gpu_mem_mib = torch.cuda.memory_allocated() / (1024 ** 2)
                    gpu_mem_reserved_mib = torch.cuda.memory_reserved() / (1024 ** 2)
                else:
                    gpu_mem_mib = None
                    gpu_mem_reserved_mib = None

                batch_time_ms = (time.perf_counter() - batch_start_time) * 1000.0
                per_frame_ms = (
                    batch_time_ms / len(batch_frames) if len(batch_frames) > 0 else 0.0
                )
                latencies_ms.append(per_frame_ms)

                if batch_indices and batch_indices[0] == batch_indices[-1]:
                    frame_msg = "Batch frames %d: detections=%d, latency=%.1f ms/frame%s"
                    frame_args = (
                        batch_indices[0],
                        batch_detections_count,
                        per_frame_ms,
                        (
                            f", gpu_mem={gpu_mem_mib:.1f}/{gpu_mem_reserved_mib:.1f} MiB"
                            if gpu_mem_mib is not None
                            else ""
                        ),
                    )
                else:
                    frame_msg = "Batch frames %d-%d: detections=%d, latency=%.1f ms/frame%s"
                    frame_args = (
                        batch_indices[0] if batch_indices else 0,
                        batch_indices[-1] if batch_indices else 0,
                        batch_detections_count,
                        per_frame_ms,
                        (
                            f", gpu_mem={gpu_mem_mib:.1f}/{gpu_mem_reserved_mib:.1f} MiB"
                            if gpu_mem_mib is not None
                            else ""
                        ),
                    )

                logger.info(frame_msg, *frame_args)

                batch_frames = []
                batch_indices = []

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

        logger.info(
            "Completed video tracking: processed=%d frames, detections=%d, tracks=%d, avg_latency=%.1f ms/frame, max_latency=%.1f ms/frame",
            len(frames),
            total_detections,
            total_tracks,
            avg_latency,
            max_latency,
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
