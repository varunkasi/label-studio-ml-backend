import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

from control_models.base import ControlModel
from utils.grounding import VideoTrackingResult
from label_studio_sdk.label_interface.control_tags import ControlTag


logger = logging.getLogger(__name__)

ALLOWED_LABELS = {
    label.strip().lower()
    for label in os.getenv("GROUNDING_DINO_ALLOWED_LABELS", "person").split(",")
    if label.strip()
}

SPARSIFY_KF_FOR_SAM = os.getenv("SPARSIFY_KF_FOR_SAM", "false").lower() in [
    "1",
    "true",
    "yes",
]

TRACKER_ENV_MAP = {
    "lost_track_buffer": "TRACKER_LOST_BUFFER",
    "minimum_matching_threshold": "TRACKER_MATCH_THRESHOLD",
    "track_activation_threshold": "TRACKER_ACTIVATION_THRESHOLD",
    "minimum_consecutive_frames": "TRACKER_MIN_CONSECUTIVE_FRAMES",
}

MODEL_THRESHOLD_ENV_MAP = {
    "model_box_threshold": "GROUNDING_DINO_BOX_THRESHOLD",
    "model_text_threshold": "GROUNDING_DINO_TEXT_THRESHOLD",
}

TRACKER_PARAM_CASTERS = {
    "lost_track_buffer": int,
    "minimum_consecutive_frames": int,
    "minimum_matching_threshold": float,
    "track_activation_threshold": float,
}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(
            f"Required environment variable '{name}' is not set or empty."
        )
    return value


class VideoRectangleModel(ControlModel):
    """Video rectangle control using Grounding DINO detections with ByteTrack."""

    type = "VideoRectangle"
    last_tracking_result: Optional[VideoTrackingResult] = None

    @classmethod
    def is_control_matched(cls, control: ControlTag) -> bool:
        return control.objects[0].tag == "Video" and control.tag == cls.type

    @staticmethod
    def get_from_name_for_label_map(label_interface, target_name) -> str:
        target: ControlTag = label_interface.get_control(target_name)
        if not target:
            raise ValueError(f'Control tag with name "{target_name}" not found')

        for connected in label_interface.controls:
            if connected.tag == "Labels" and connected.to_name == target.to_name:
                return connected.name

        raise ValueError("VideoRectangle detected, but no connected 'Labels' tag found")

    def predict_regions(self, path, output_dir=None, save_frames=False, batch_size=None) -> List[Dict]:
        tracker_kwargs = self._build_tracker_kwargs()
        box_threshold = self._get_float_attr("model_box_threshold")
        text_threshold = self._get_float_attr("model_text_threshold")

        # Reset cached tracking result before processing the current task
        self.last_tracking_result = None

        tracking = self.inference.track_video(
            path,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            tracker_kwargs=tracker_kwargs,
            output_dir=output_dir,
            save_frames=save_frames,
            batch_size=batch_size,
        )
        self.last_tracking_result = tracking
        return self.create_video_rectangles(tracking)

    def create_video_rectangles(self, tracking_result) -> List[Dict]:
        frames_count = tracking_result.frames_count or len(tracking_result.frames)
        fps = tracking_result.fps if tracking_result.fps else 0.0
        frame_interval = (1.0 / fps) if fps else (
            tracking_result.duration / frames_count if frames_count else 0.0
        )

        tracks = defaultdict(list)
        track_labels: Dict[int, str] = {}

        for frame_info in tracking_result.frames:
            detections = frame_info.detections
            if detections.tracker_id is None:
                continue

            for bbox, score, class_id, track_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
                detections.tracker_id,
            ):
                if track_id is None:
                    continue
                if score < self.model_score_threshold:
                    continue

                model_label = self.inference.names[int(class_id)]
                if model_label not in self.label_map:
                    continue
                output_label = self.label_map[model_label]
                if output_label.strip().lower() not in ALLOWED_LABELS:
                    continue

                track_labels[int(track_id)] = output_label

                x1, y1, x2, y2 = bbox.tolist()
                width = frame_info.width
                height = frame_info.height
                frame_number = frame_info.frame_index + 1

                box_entry = {
                    "frame": frame_number,
                    "enabled": True,
                    "rotation": 0,
                    "x": x1 / width * 100,
                    "y": y1 / height * 100,
                    "width": (x2 - x1) / width * 100,
                    "height": (y2 - y1) / height * 100,
                    "time": frame_info.frame_index * frame_interval,
                    "score": float(score),
                }
                tracks[int(track_id)].append(box_entry)

        regions: List[Dict] = []
        for track_id, sequence in tracks.items():
            sequence.sort(key=lambda item: item["frame"])
            if SPARSIFY_KF_FOR_SAM:
                sequence = self._sparsify_sequence_for_sam(sequence)
            sequence = self.process_lifespans_enabled(sequence)

            label = track_labels.get(track_id)
            if not label:
                continue

            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "videorectangle",
                "value": {
                    "framesCount": frames_count,
                    "duration": tracking_result.duration,
                    "sequence": sequence,
                    "labels": [label],
                },
                "score": max(frame_info["score"] for frame_info in sequence),
                "origin": "manual",
            }
            regions.append(region)

        return regions

    @staticmethod
    def process_lifespans_enabled(sequence: List[Dict]) -> List[Dict]:
        prev = None
        for i, box in enumerate(sequence):
            if prev is None:
                prev = sequence[i]
                continue
            if box["frame"] - prev["frame"] > 1:
                sequence[i - 1]["enabled"] = False
            prev = sequence[i]

        if sequence:
            sequence[-1]["enabled"] = False
        return sequence

    @staticmethod
    def _sparsify_sequence_for_sam(sequence: List[Dict]) -> List[Dict]:
        if not sequence:
            return sequence

        chunk_size = 2000
        max_kf_per_chunk = 5

        by_chunk: Dict[int, List[Dict]] = {}
        for box in sequence:
            frame = box.get("frame")
            if frame is None:
                continue
            chunk_index = (frame - 1) // chunk_size
            if chunk_index not in by_chunk:
                by_chunk[chunk_index] = []
            by_chunk[chunk_index].append(box)

        result: List[Dict] = []
        for chunk_index in sorted(by_chunk.keys()):
            chunk_boxes = by_chunk[chunk_index]
            count = len(chunk_boxes)
            if count <= max_kf_per_chunk:
                result.extend(chunk_boxes)
                continue

            desired = max_kf_per_chunk
            if desired <= 0:
                continue

            indices = set()
            if desired == 1:
                indices.add(count // 2)
            else:
                for i in range(desired):
                    pos = int(round(i * (count - 1) / (desired - 1)))
                    indices.add(pos)

            for idx, box in enumerate(chunk_boxes):
                if idx in indices:
                    result.append(box)

        result.sort(key=lambda item: item["frame"])
        return result

    def _build_tracker_kwargs(self) -> Dict:
        kwargs: Dict = {}

        for param, env_name in TRACKER_ENV_MAP.items():
            value = _require_env(env_name)
            try:
                caster = TRACKER_PARAM_CASTERS.get(param, float)
                if caster is int:
                    kwargs[param] = int(float(value))
                else:
                    kwargs[param] = caster(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value for environment variable '{env_name}': {value}"
                ) from exc

        control_threshold = self.control.attr.get("tracker_match_threshold")
        if control_threshold:
            try:
                kwargs.setdefault("minimum_matching_threshold", float(control_threshold))
            except ValueError:
                logger.warning(
                    "Invalid tracker_match_threshold value '%s'", control_threshold
                )

        return kwargs

    def _get_float_attr(self, key: str) -> Optional[float]:
        env_name = MODEL_THRESHOLD_ENV_MAP.get(key)
        if env_name:
            env_value = _require_env(env_name)
            try:
                return float(env_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid value for environment variable '{env_name}': {env_value}"
                ) from exc

        value = self.control.attr.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            logger.warning("Invalid attribute %s='%s'", key, value)
            return None
