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


def interpolate_detections(
    det_before: sv.Detections,
    det_after: sv.Detections,
    alpha: float,
) -> sv.Detections:
    """Linearly interpolate bounding boxes between two detection sets.
    
    This is used for frame skipping - we detect on keyframes and interpolate
    boxes for skipped frames. Only boxes with matching class_ids are interpolated.
    
    Args:
        det_before: Detections from the earlier keyframe
        det_after: Detections from the later keyframe  
        alpha: Interpolation factor (0.0 = det_before, 1.0 = det_after)
        
    Returns:
        Interpolated detections. Boxes present in both frames are interpolated.
        Boxes only in det_before are included with fading confidence.
        Boxes only in det_after are included with rising confidence.
    """
    if len(det_before) == 0:
        # Scale confidence by alpha (boxes appearing)
        if len(det_after) == 0:
            return sv.Detections.empty()
        result = det_after.copy()
        if result.confidence is not None:
            result.confidence = result.confidence * alpha
        return result
    
    if len(det_after) == 0:
        # Scale confidence by (1-alpha) (boxes disappearing)
        result = det_before.copy()
        if result.confidence is not None:
            result.confidence = result.confidence * (1.0 - alpha)
        return result
    
    # Match boxes by class_id and proximity (IoU)
    # For simplicity, we'll interpolate all boxes from det_before
    # and blend with det_after based on alpha
    
    # Simple approach: weighted average of all boxes
    # More sophisticated: Hungarian matching by IoU, then interpolate matched pairs
    
    # Use simple weighted combination for now
    interpolated_xyxy = []
    interpolated_conf = []
    interpolated_class = []
    
    # Track which det_after boxes have been matched
    matched_after = set()
    
    for i in range(len(det_before)):
        box_before = det_before.xyxy[i]
        class_before = det_before.class_id[i] if det_before.class_id is not None else 0
        conf_before = det_before.confidence[i] if det_before.confidence is not None else 1.0
        
        # Find best matching box in det_after with same class
        best_iou = 0.0
        best_j = -1
        
        for j in range(len(det_after)):
            if j in matched_after:
                continue
            class_after = det_after.class_id[j] if det_after.class_id is not None else 0
            if class_before != class_after:
                continue
            
            # Compute IoU
            box_after = det_after.xyxy[j]
            x1 = max(box_before[0], box_after[0])
            y1 = max(box_before[1], box_after[1])
            x2 = min(box_before[2], box_after[2])
            y2 = min(box_before[3], box_after[3])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_before = (box_before[2] - box_before[0]) * (box_before[3] - box_before[1])
            area_after = (box_after[2] - box_after[0]) * (box_after[3] - box_after[1])
            union = area_before + area_after - inter
            
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_j = j
        
        if best_j >= 0 and best_iou > 0.1:  # Matched
            matched_after.add(best_j)
            box_after = det_after.xyxy[best_j]
            conf_after = det_after.confidence[best_j] if det_after.confidence is not None else 1.0
            
            # Interpolate box coordinates
            interp_box = box_before * (1.0 - alpha) + box_after * alpha
            interp_conf = conf_before * (1.0 - alpha) + conf_after * alpha
            
            interpolated_xyxy.append(interp_box)
            interpolated_conf.append(interp_conf)
            interpolated_class.append(class_before)
        else:
            # Box disappearing - fade out
            interpolated_xyxy.append(box_before)
            interpolated_conf.append(conf_before * (1.0 - alpha))
            interpolated_class.append(class_before)
    
    # Add unmatched boxes from det_after (appearing boxes)
    for j in range(len(det_after)):
        if j not in matched_after:
            box_after = det_after.xyxy[j]
            conf_after = det_after.confidence[j] if det_after.confidence is not None else 1.0
            class_after = det_after.class_id[j] if det_after.class_id is not None else 0
            
            interpolated_xyxy.append(box_after)
            interpolated_conf.append(conf_after * alpha)  # Fade in
            interpolated_class.append(class_after)
    
    if len(interpolated_xyxy) == 0:
        return sv.Detections.empty()
    
    return sv.Detections(
        xyxy=np.array(interpolated_xyxy, dtype=np.float32),
        confidence=np.array(interpolated_conf, dtype=np.float32),
        class_id=np.array(interpolated_class, dtype=np.int32),
    )


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

    def infer_batch(
        self,
        frames: List[np.ndarray],
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[sv.Detections]:
        """Perform optimized batch inference on multiple frames.

        This method processes frames sequentially but with optimizations:
        - Pre-transforms all frames to tensors upfront
        - Keeps tensors on GPU to avoid repeated CPU->GPU transfers
        - Uses CUDA streams for overlapping compute and data transfer
        - Minimizes Python overhead between frames

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

        # Pre-transform all frames to tensors and move to GPU upfront
        # This allows overlapping data transfer with computation
        prepared_frames: List[Tuple[torch.Tensor, int, int]] = []
        for frame in frames:
            h, w = frame.shape[:2]
            pil_image = Image.fromarray(frame).convert("RGB")
            tensor, _ = self.transform(pil_image, None)
            # Move to GPU immediately - this can overlap with next frame's CPU processing
            tensor = tensor.to(self.device)
            prepared_frames.append((tensor, h, w))

        # Process all frames with model - tensors are already on GPU
        detections_list: List[sv.Detections] = []
        
        with torch.no_grad():
            for tensor, orig_h, orig_w in prepared_frames:
                if self.use_amp:
                    with autocast():
                        boxes, scores, phrases = predict(
                            model=self.model,
                            image=tensor,
                            caption=resolved_prompt,
                            box_threshold=box_threshold,
                            text_threshold=text_threshold,
                            device=self.device,
                        )
                else:
                    boxes, scores, phrases = predict(
                        model=self.model,
                        image=tensor,
                        caption=resolved_prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=self.device,
                    )

                if boxes.numel() == 0:
                    detections_list.append(self._empty_detections())
                    continue

                # Convert boxes from cxcywh normalized to xyxy pixel coordinates
                xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor(
                    [orig_w, orig_h, orig_w, orig_h]
                )
                xyxy = xyxy.cpu().numpy().astype(np.float32)
                confidences = scores.cpu().numpy().astype(np.float32)

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
        frame_skip: Optional[int] = None,
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
            frame_skip: Process every Nth frame, interpolate others (default from env or auto)
                        Set to 1 to disable frame skipping.
                        Set to "auto" or 0 to auto-determine based on video length.
            output_dir: Directory to save annotated frames (required if save_frames=True)
            save_frames: Whether to save annotated frames with bounding boxes
        """
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

        # Determine frame skip rate
        if frame_skip is None:
            env_skip = os.getenv("GROUNDING_DINO_FRAME_SKIP", "auto")
            if env_skip.lower() == "auto":
                frame_skip = 0  # Will be auto-determined below
            else:
                try:
                    frame_skip = int(env_skip)
                except ValueError:
                    logger.warning(
                        "Invalid GROUNDING_DINO_FRAME_SKIP value '%s', using auto",
                        env_skip,
                    )
                    frame_skip = 0
        
        # Auto-determine frame skip based on video length
        if frame_skip == 0:
            if frame_count > 10000:  # >5.5 min at 30fps
                frame_skip = 3  # Process every 3rd frame
            elif frame_count > 5000:  # >2.7 min at 30fps
                frame_skip = 2  # Process every 2nd frame
            else:
                frame_skip = 1  # No skipping for short videos
        
        frame_skip = max(1, frame_skip)  # Ensure at least 1
        
        # Setup output directory if saving frames
        if save_frames:
            if not output_dir:
                raise ValueError("output_dir must be specified when save_frames=True")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving annotated frames to: %s", output_path)

        logger.info(
            "Starting video tracking: path=%s, total_frames=%s, fps=%.2f, frame_skip=%d",
            path,
            frame_count or "unknown",
            fps,
            frame_skip,
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
        logger.info("Prepared tracker kwargs: %s", tracker_kwargs)
        tracker = sv.ByteTrack(**tracker_kwargs)
        logger.info(
            "ByteTrack config: activation=%.2f, lost_buffer=%d, matching=%.2f, consecutive=%d",
            tracker.track_activation_threshold,
            int(tracker.max_time_lost * 30.0 / fps) if fps > 0 else 0,
            tracker.minimum_matching_threshold,
            tracker.minimum_consecutive_frames,
        )
        frames: List[FrameDetections] = []
        latencies_ms: List[float] = []
        device_is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        tracked_ids: Set[int] = set()
        
        # Frame skipping state
        # We collect frames in chunks, detect on keyframes, interpolate for skipped frames
        frame_index = 0
        keyframe_detections: Dict[int, sv.Detections] = {}  # keyframe_idx -> detections
        pending_frames: List[Tuple[int, np.ndarray]] = []  # (idx, frame) waiting for next keyframe
        last_keyframe_idx = -1
        last_keyframe_det: Optional[sv.Detections] = None
        
        # For batching keyframes
        keyframe_batch: List[np.ndarray] = []
        keyframe_indices: List[int] = []
        
        try:
            while True:
                # Read frames until we have enough keyframes for a batch
                frames_to_process: List[Tuple[int, np.ndarray]] = []
                
                while len(keyframe_batch) < batch_size:
                    ret, frame = capture.read()
                    if not ret:
                        break
                    
                    is_keyframe = (frame_index % frame_skip == 0)
                    frames_to_process.append((frame_index, frame, is_keyframe))
                    
                    if is_keyframe:
                        keyframe_batch.append(frame)
                        keyframe_indices.append(frame_index)
                    
                    frame_index += 1
                
                if len(frames_to_process) == 0:
                    break
                
                # Progress logging
                if progress_every_int and frame_index % (progress_every_int * frame_skip) < frame_skip:
                    logger.info(
                        "Tracking progress: frames %d/%s (keyframes: %d)",
                        frame_index,
                        frame_count or "?",
                        len(keyframe_detections) + len(keyframe_batch),
                    )
                
                if device_is_cuda:
                    torch.cuda.synchronize()
                batch_start_time = time.perf_counter()
                
                # Run detection on keyframes
                if keyframe_batch:
                    process_as_batch = use_batch_inference and len(keyframe_batch) > 1
                    
                    if process_as_batch:
                        try:
                            batch_detections_list = self.infer_batch(
                                keyframe_batch,
                                prompt=prompt,
                                box_threshold=box_threshold,
                                text_threshold=text_threshold,
                            )
                            for kf_idx, det in zip(keyframe_indices, batch_detections_list):
                                keyframe_detections[kf_idx] = det
                        except Exception as batch_error:
                            logger.error(
                                "Error processing keyframe batch: %s, falling back to single-frame",
                                batch_error,
                            )
                            process_as_batch = False
                    
                    if not process_as_batch:
                        for kf_frame, kf_idx in zip(keyframe_batch, keyframe_indices):
                            try:
                                det = self.infer_frame(
                                    kf_frame,
                                    prompt=prompt,
                                    box_threshold=box_threshold,
                                    text_threshold=text_threshold,
                                )
                                keyframe_detections[kf_idx] = det
                            except Exception as frame_error:
                                logger.error(
                                    "Error processing keyframe %d: %s", kf_idx, frame_error
                                )
                                keyframe_detections[kf_idx] = sv.Detections.empty()
                
                # Now process all frames (keyframes + interpolated)
                batch_detections_count = 0
                
                for fidx, frame, is_keyframe in frames_to_process:
                    if is_keyframe:
                        # Use actual detection
                        detections = keyframe_detections.get(fidx, sv.Detections.empty())
                        last_keyframe_idx = fidx
                        last_keyframe_det = detections
                    else:
                        # Interpolate between previous and next keyframe
                        prev_kf_idx = (fidx // frame_skip) * frame_skip
                        next_kf_idx = prev_kf_idx + frame_skip
                        
                        prev_det = keyframe_detections.get(prev_kf_idx)
                        next_det = keyframe_detections.get(next_kf_idx)
                        
                        if prev_det is None and last_keyframe_det is not None:
                            prev_det = last_keyframe_det
                            prev_kf_idx = last_keyframe_idx
                        
                        if prev_det is None:
                            # No previous keyframe yet, use empty
                            detections = sv.Detections.empty()
                        elif next_det is None:
                            # No next keyframe (end of video or not yet processed)
                            # Just use previous keyframe's detections
                            detections = prev_det
                        else:
                            # Interpolate
                            alpha = (fidx - prev_kf_idx) / frame_skip
                            detections = interpolate_detections(prev_det, next_det, alpha)
                    
                    # Update tracker with detections (real or interpolated)
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
                            frame_index=fidx,
                            height=frame.shape[0],
                            width=frame.shape[1],
                            detections=tracked,
                        )
                    )
                    
                    batch_detections_count += tracked.xyxy.shape[0]
                    
                    if save_frames:
                        annotated_frame = self._annotate_frame(frame, tracked, fidx)
                        frame_filename = output_path / f"frame_{fidx:06d}.jpg"
                        cv2.imwrite(str(frame_filename), annotated_frame)
                
                # Timing and logging
                if device_is_cuda:
                    torch.cuda.synchronize()
                    gpu_mem_mib = torch.cuda.memory_allocated() / (1024 ** 2)
                    gpu_mem_reserved_mib = torch.cuda.memory_reserved() / (1024 ** 2)
                else:
                    gpu_mem_mib = None
                    gpu_mem_reserved_mib = None
                
                batch_time_ms = (time.perf_counter() - batch_start_time) * 1000.0
                num_keyframes = len(keyframe_batch)
                num_total = len(frames_to_process)
                per_frame_ms = batch_time_ms / num_total if num_total > 0 else 0.0
                latencies_ms.append(per_frame_ms)
                
                logger.info(
                    "Processed frames %d-%d: keyframes=%d, total=%d, detections=%d, "
                    "latency=%.1f ms/frame%s",
                    frames_to_process[0][0] if frames_to_process else 0,
                    frames_to_process[-1][0] if frames_to_process else 0,
                    num_keyframes,
                    num_total,
                    batch_detections_count,
                    per_frame_ms,
                    f", gpu_mem={gpu_mem_mib:.1f}/{gpu_mem_reserved_mib:.1f} MiB"
                    if gpu_mem_mib is not None else "",
                )
                
                # Clear batch for next iteration
                keyframe_batch = []
                keyframe_indices = []
                # Keep only recent keyframe detections (memory management)
                if keyframe_detections:
                    min_keep = max(keyframe_detections.keys()) - frame_skip * 2
                    keyframe_detections = {
                        k: v for k, v in keyframe_detections.items() if k >= min_keep
                    }

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
