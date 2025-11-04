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
from groundingdino.util.inference import load_model, predict

try:
    from torch.cuda.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# SAHI imports with graceful fallback
SAHI_AVAILABLE = False
SAHI_IMPORT_ERROR = None
try:
    from sahi.prediction import ObjectPrediction, PredictionResult
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError as e:
    SAHI_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)

# Log SAHI import status after logger is initialized
if not SAHI_AVAILABLE and SAHI_IMPORT_ERROR:
    logger.warning(
        "SAHI library not available. Sliced inference will be disabled. "
        "Install with: pip install sahi>=0.11.14 (Error: %s)", SAHI_IMPORT_ERROR
    )


class GroundingDINOSAHIAdapter:
    """Adapter to make GroundingDINO inference compatible with SAHI's detection model interface.

    SAHI expects a detection model with:
    - perform_inference(image) method returning raw predictions
    - convert_original_predictions(predictions) method converting to ObjectPrediction list
    
    This adapter wraps the GroundingDINOInference instance to provide that interface.
    """

    def __init__(
        self,
        inference_instance: "GroundingDINOInference",
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ):
        """Initialize the SAHI adapter.

        Args:
            inference_instance: Parent GroundingDINOInference instance
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
        """
        self.inference = inference_instance
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.category_mapping = {i: name for i, name in inference_instance.names.items()}
        self._original_predictions = None
        self.object_prediction_list: List[ObjectPrediction] = []

    def perform_inference(self, image: np.ndarray):
        """Perform inference on a single image slice and stash raw predictions.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            None. Predictions are retrieved via convert_original_predictions.
        """
        # Get raw detections from standard inference (avoids SAHI recursion)
        detections = self.inference._infer_frame_standard(
            image,
            prompt=self.prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # Store raw predictions for SAHI to convert later
        self._original_predictions = detections
        return None

    def convert_original_predictions(self, original_predictions=None, shift_amount=None, **_):
        """Convert raw predictions to SAHI ObjectPrediction list.

        SAHI calls this method to convert the output of perform_inference into
        its standardized ObjectPrediction format for postprocessing and merging.

        Args:
            original_predictions: Either sv.Detections object or list of ObjectPrediction instances
            shift_amount: Optional tuple/list with (x, y) offsets applied when converting slice predictions

        Returns:
            List of ObjectPrediction instances

        Raises:
            TypeError: If original_predictions is neither sv.Detections nor a list
        """
        if original_predictions is None:
            if self._original_predictions is None:
                raise ValueError("No original predictions available for conversion.")
            original_predictions = self._original_predictions
        shift_x, shift_y = self._normalize_shift_amount(shift_amount)

        # Handle already-converted ObjectPrediction lists (for compatibility)
        if isinstance(original_predictions, list):
            # Verify it's actually a list of ObjectPrediction instances
            if original_predictions and not isinstance(original_predictions[0], ObjectPrediction):
                raise TypeError(
                    f"Expected list of ObjectPrediction instances, got list of {type(original_predictions[0])}"
                )
            converted = self._shift_object_predictions(original_predictions, shift_x, shift_y)
        elif isinstance(original_predictions, sv.Detections):
            converted = self._detections_to_object_predictions(
                original_predictions, shift_x=shift_x, shift_y=shift_y
            )
        else:
            # Unexpected type
            raise TypeError(
                f"Expected sv.Detections or list of ObjectPrediction, got {type(original_predictions)}"
            )

        self.object_prediction_list = converted
        return converted

    @staticmethod
    def _normalize_shift_amount(shift_amount: Optional[Iterable[float]]) -> Tuple[float, float]:
        if shift_amount is None:
            return 0.0, 0.0

        try:
            shift_x, shift_y = shift_amount
        except (TypeError, ValueError):
            raise TypeError(
                f"shift_amount must be an iterable of length 2, got {shift_amount!r}"
            ) from None

        return float(shift_x), float(shift_y)

    def _shift_object_predictions(
        self, predictions: List[ObjectPrediction], shift_x: float, shift_y: float
    ) -> List[ObjectPrediction]:
        if not (shift_x or shift_y):
            return [prediction.copy() for prediction in predictions]

        shifted_predictions = []
        for prediction in predictions:
            if hasattr(prediction, "get_shifted_object_prediction"):
                shifted_predictions.append(
                    prediction.get_shifted_object_prediction(
                        shift_amount=(shift_x, shift_y)
                    )
                )
                continue

            # Fallback: manually rebuild prediction with shifted bbox
            x1, y1, x2, y2 = prediction.bbox.to_xyxy()
            xywh = [x1 + shift_x, y1 + shift_y, x2 - x1, y2 - y1]
            shifted_predictions.append(
                ObjectPrediction(
                    bbox=xywh,
                    category_id=prediction.category_id,
                    category_name=prediction.category_name,
                    score=prediction.score.value,
                )
            )

        return shifted_predictions

    def _detections_to_object_predictions(
        self, detections: sv.Detections, *, shift_x: float = 0.0, shift_y: float = 0.0
    ) -> List[ObjectPrediction]:
        """Convert supervision Detections to SAHI ObjectPrediction list.

        Args:
            detections: supervision Detections object

        Returns:
            List of ObjectPrediction instances
        """
        object_predictions = []

        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            score = (
                float(detections.confidence[i]) if detections.confidence is not None else 0.0
            )
            class_id = int(detections.class_id[i]) if detections.class_id is not None else 0

            # Convert xyxy to xywh format (SAHI expects [x, y, width, height])
            x1, y1, x2, y2 = bbox
            xywh = [
                float(x1 + shift_x),
                float(y1 + shift_y),
                float(x2 - x1),
                float(y2 - y1),
            ]

            category_name = self.category_mapping.get(class_id, f"class_{class_id}")

            object_prediction = ObjectPrediction(
                bbox=xywh,
                category_id=class_id,
                category_name=category_name,
                score=float(score),
            )
            object_predictions.append(object_prediction)

        return object_predictions


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

        # SAHI sliced inference configuration
        self.sahi_enabled = False
        self.sahi_config = {}
        if SAHI_AVAILABLE:
            self._initialize_sahi_config()
        elif os.getenv("SAHI_ENABLED", "false").lower() in ["true", "1", "yes"]:
            logger.warning(
                "SAHI_ENABLED=true but SAHI library is not available. "
                "Sliced inference will be disabled."
            )
        
        # Note: torch.compile is disabled for Grounding DINO due to compatibility issues
        # with the Swin Transformer backbone's dynamic slicing operations
        
        # Use larger resolution for better GPU utilization
        max_size = int(os.getenv("GROUNDING_DINO_MAX_SIZE", "1333"))
        base_size = int(os.getenv("GROUNDING_DINO_BASE_SIZE", "800"))
        self.transform = T.Compose(
            [
                T.RandomResize([base_size], max_size=max_size),
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

    def _initialize_sahi_config(self) -> None:
        """Initialize SAHI configuration from environment variables."""
        sahi_enabled_str = os.getenv("SAHI_ENABLED", "false").lower()
        self.sahi_enabled = sahi_enabled_str in ["true", "1", "yes"]

        if not self.sahi_enabled:
            return

        # Parse SAHI slice dimensions
        try:
            slice_height = int(os.getenv("SAHI_SLICE_HEIGHT", "640"))
            slice_width = int(os.getenv("SAHI_SLICE_WIDTH", "640"))
            if slice_height <= 0 or slice_width <= 0:
                raise ValueError("Slice dimensions must be positive")
        except (ValueError, TypeError) as e:
            logger.warning(
                "Invalid SAHI slice dimensions, falling back to 640x640: %s", e
            )
            slice_height = 640
            slice_width = 640

        # Parse SAHI overlap ratios
        try:
            overlap_height_ratio = float(os.getenv("SAHI_OVERLAP_HEIGHT_RATIO", "0.2"))
            overlap_width_ratio = float(os.getenv("SAHI_OVERLAP_WIDTH_RATIO", "0.2"))
            if not (0.0 <= overlap_height_ratio <= 1.0 and 0.0 <= overlap_width_ratio <= 1.0):
                raise ValueError("Overlap ratios must be between 0.0 and 1.0")
        except (ValueError, TypeError) as e:
            logger.warning(
                "Invalid SAHI overlap ratios, falling back to 0.2: %s", e
            )
            overlap_height_ratio = 0.2
            overlap_width_ratio = 0.2

        # Parse SAHI postprocessing configuration
        postprocess_type = os.getenv("SAHI_POSTPROCESS_TYPE", "NMS").upper()
        if postprocess_type not in ["NMS", "NMM", "GREEDYNMM"]:
            logger.warning(
                "Invalid SAHI_POSTPROCESS_TYPE '%s', falling back to 'NMS'",
                postprocess_type,
            )
            postprocess_type = "NMS"

        postprocess_match_metric = os.getenv("SAHI_POSTPROCESS_MATCH_METRIC", "IOS").upper()
        if postprocess_match_metric not in ["IOU", "IOS"]:
            logger.warning(
                "Invalid SAHI_POSTPROCESS_MATCH_METRIC '%s', falling back to 'IOS'",
                postprocess_match_metric,
            )
            postprocess_match_metric = "IOS"

        try:
            postprocess_match_threshold = float(
                os.getenv("SAHI_POSTPROCESS_MATCH_THRESHOLD", "0.5")
            )
            if not (0.0 <= postprocess_match_threshold <= 1.0):
                raise ValueError("Match threshold must be between 0.0 and 1.0")
        except (ValueError, TypeError) as e:
            logger.warning(
                "Invalid SAHI_POSTPROCESS_MATCH_THRESHOLD, falling back to 0.5: %s", e
            )
            postprocess_match_threshold = 0.5

        postprocess_class_agnostic_str = os.getenv(
            "SAHI_POSTPROCESS_CLASS_AGNOSTIC", "true"
        ).lower()
        postprocess_class_agnostic = postprocess_class_agnostic_str in ["true", "1", "yes"]

        # Parse minimum area ratio for filtering small detections
        try:
            min_area_ratio = float(os.getenv("SAHI_MIN_AREA_RATIO", "0.0"))
            if not (0.0 <= min_area_ratio <= 1.0):
                raise ValueError("Min area ratio must be between 0.0 and 1.0")
        except (ValueError, TypeError) as e:
            logger.warning(
                "Invalid SAHI_MIN_AREA_RATIO, falling back to 0.0: %s", e
            )
            min_area_ratio = 0.0

        self.sahi_config = {
            "slice_height": slice_height,
            "slice_width": slice_width,
            "overlap_height_ratio": overlap_height_ratio,
            "overlap_width_ratio": overlap_width_ratio,
            "postprocess_type": postprocess_type,
            "postprocess_match_metric": postprocess_match_metric,
            "postprocess_match_threshold": postprocess_match_threshold,
            "postprocess_class_agnostic": postprocess_class_agnostic,
            "min_area_ratio": min_area_ratio,
        }

        try:
            score_threshold_env = os.getenv("SAHI_SCORE_THRESHOLD")
            score_threshold = float(score_threshold_env) if score_threshold_env else float("nan")
        except (ValueError, TypeError):
            logger.warning(
                "Invalid SAHI_SCORE_THRESHOLD '%s', inheriting box threshold",
                os.getenv("SAHI_SCORE_THRESHOLD"),
            )
            score_threshold = float("nan")

        self.sahi_config["score_threshold"] = score_threshold

        if np.isnan(score_threshold):
            effective_score = "inherit(box_threshold)"
        else:
            effective_score = f"{score_threshold:.3f}"

        logger.info(
            "SAHI score threshold: %s (values below are filtered before merging)",
            effective_score,
        )

        logger.info(
            "SAHI sliced inference enabled: slice_size=%dx%d, overlap=(%.2f, %.2f), "
            "postprocess=%s, match_metric=%s, match_threshold=%.2f, class_agnostic=%s, min_area_ratio=%.3f",
            slice_height,
            slice_width,
            overlap_height_ratio,
            overlap_width_ratio,
            postprocess_type,
            postprocess_match_metric,
            postprocess_match_threshold,
            postprocess_class_agnostic,
            min_area_ratio,
        )

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
        """Perform inference on a single frame with optional SAHI slicing.

        Args:
            frame: Input frame as numpy array (RGB format)
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            supervision Detections object
        """
        # Route to SAHI path if enabled, otherwise use standard inference
        if self.sahi_enabled and SAHI_AVAILABLE:
            try:
                return self._infer_frame_with_sahi(
                    frame,
                    prompt=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            except Exception as e:
                logger.error(
                    "SAHI inference failed, falling back to standard inference: %s", e
                )
                # Fall through to standard inference

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
        """Standard (non-SAHI) inference on a single frame.

        This is the original infer_frame implementation, extracted to avoid
        recursion when SAHI calls back into inference.

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

    def _infer_frame_with_sahi(
        self,
        frame: np.ndarray,
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Perform sliced inference using SAHI on a single frame.

        Args:
            frame: Input frame as numpy array (RGB format)
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            supervision Detections object with merged results from all slices
        """
        resolved_prompt = (prompt or self.prompt).strip()
        if not resolved_prompt.endswith("."):
            resolved_prompt = resolved_prompt + "."

        resolved_box_threshold = (
            float(box_threshold)
            if box_threshold is not None
            else float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", 0.35))
        )
        resolved_text_threshold = (
            float(text_threshold)
            if text_threshold is not None
            else float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", 0.25))
        )

        # Create SAHI adapter for this inference session
        adapter = GroundingDINOSAHIAdapter(
            inference_instance=self,
            prompt=resolved_prompt,
            box_threshold=resolved_box_threshold,
            text_threshold=resolved_text_threshold,
        )

        # Check if frame is smaller than slice size (no benefit from slicing)
        height, width = frame.shape[:2]
        slice_height = self.sahi_config["slice_height"]
        slice_width = self.sahi_config["slice_width"]

        if height <= slice_height and width <= slice_width:
            logger.debug(
                "Frame size (%dx%d) smaller than slice size (%dx%d), using standard inference",
                height, width, slice_height, slice_width,
            )
            return self._infer_frame_standard(
                frame,
                prompt=resolved_prompt,
                box_threshold=resolved_box_threshold,
                text_threshold=resolved_text_threshold,
            )

        # Perform sliced prediction
        sahi_score_threshold = float(
            os.getenv("SAHI_SCORE_THRESHOLD", str(resolved_box_threshold))
        )

        sahi_result: PredictionResult = get_sliced_prediction(
            image=frame,
            detection_model=adapter,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=self.sahi_config["overlap_height_ratio"],
            overlap_width_ratio=self.sahi_config["overlap_width_ratio"],
            postprocess_type=self.sahi_config["postprocess_type"],
            postprocess_match_metric=self.sahi_config["postprocess_match_metric"],
            postprocess_match_threshold=self.sahi_config["postprocess_match_threshold"],
            postprocess_class_agnostic=self.sahi_config["postprocess_class_agnostic"],
            score_threshold=sahi_score_threshold,
        )

        # Convert SAHI result back to supervision Detections
        detections = self._sahi_result_to_detections(sahi_result)

        # Log slice statistics if available
        slice_groups = getattr(sahi_result, "object_prediction_list_per_image", None)
        if slice_groups is not None:
            num_slices = len(slice_groups)
        else:
            slice_predictions = getattr(sahi_result, "slice_prediction_list", None)
            num_slices = len(slice_predictions) if slice_predictions is not None else None

        if num_slices is not None:
            logger.debug(
                "SAHI sliced inference: frame_size=%dx%d, slices=%d, detections=%d",
                width, height, num_slices, len(detections.xyxy),
            )
        else:
            logger.debug(
                "SAHI sliced inference: frame_size=%dx%d, detections=%d (slice count unavailable)",
                width, height, len(detections.xyxy),
            )

        return detections

    def _sahi_result_to_detections(self, sahi_result: "PredictionResult") -> sv.Detections:
        """Convert SAHI PredictionResult to supervision Detections.

        Args:
            sahi_result: SAHI prediction result containing merged object predictions

        Returns:
            supervision Detections object
        """
        object_predictions = sahi_result.object_prediction_list

        if not object_predictions:
            return self._empty_detections()

        xyxy_list = []
        confidence_list = []
        class_id_list = []

        frame_area = None
        min_area_ratio = self.sahi_config.get("min_area_ratio", 0.0)

        for pred in object_predictions:
            # Convert xywh to xyxy format
            x, y, w, h = pred.bbox.to_xywh()
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Filter by minimum area ratio if configured
            if min_area_ratio > 0.0:
                if frame_area is None:
                    # Infer frame area from first prediction's image dimensions
                    # SAHI stores full image dimensions in the prediction
                    frame_area = float(sahi_result.image_height * sahi_result.image_width)

                box_area = w * h
                if box_area / frame_area < min_area_ratio:
                    continue

            xyxy_list.append([x1, y1, x2, y2])
            confidence_list.append(pred.score.value)
            class_id_list.append(pred.category.id)

        if not xyxy_list:
            return self._empty_detections()

        detections = sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            confidence=np.array(confidence_list, dtype=np.float32),
            class_id=np.array(class_id_list, dtype=np.int32),
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
        """Perform batched inference on multiple frames for better GPU utilization.

        Args:
            frames: List of frames (numpy arrays in RGB format)
            prompt: Text prompt for detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            List of detections, one per frame
        """
        if not frames:
            return []

        # When SAHI is enabled, process frames individually through sliced inference
        if self.sahi_enabled and SAHI_AVAILABLE:
            if len(frames) > 4:
                logger.warning(
                    "SAHI enabled with batch_size=%d. Note: SAHI processes slices sequentially, "
                    "which may reduce effective throughput compared to batched full-frame inference.",
                    len(frames),
                )

            detections_list: List[sv.Detections] = []
            for frame in frames:
                try:
                    detections = self.infer_frame(
                        frame,
                        prompt=prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                    )
                    detections_list.append(detections)
                except Exception as e:
                    logger.error("Error in SAHI inference for frame: %s", e)
                    # Return empty detections for failed frame
                    detections_list.append(self._empty_detections())

            return detections_list

        # Standard batched inference (original implementation)
        resolved_prompt = (prompt or self.prompt).strip()
        detections_list: List[sv.Detections] = []

        for frame in frames:
            detections = self.infer_frame(
                frame,
                prompt=resolved_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
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

        # Get batch size from parameter or environment. Default to single-frame inference.
        if batch_size is None:
            env_batch = os.getenv("GROUNDING_DINO_BATCH_SIZE")
            if env_batch is None:
                batch_size = 1
            else:
                try:
                    batch_size = int(env_batch)
                except ValueError:
                    logger.warning(
                        "Invalid GROUNDING_DINO_BATCH_SIZE value '%s', falling back to 1",
                        env_batch,
                    )
                    batch_size = 1
        if batch_size <= 0:
            logger.warning("Received non-positive batch_size=%d, defaulting to 1", batch_size)
            batch_size = 1

        use_batch_inference = batch_size > 1
        logger.info("Effective batch size: %d", batch_size)
        if not use_batch_inference:
            logger.info("Batch size <= 1; using single-frame inference path")

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
