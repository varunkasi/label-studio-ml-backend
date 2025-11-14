import os
import cv2
import logging
import yaml
import hashlib

from collections import defaultdict
from control_models.base import ControlModel, MODEL_ROOT
from label_studio_sdk.label_interface.control_tags import ControlTag
from typing import List, Dict, Union
from pathlib import Path


logger = logging.getLogger(__name__)

ALLOWED_LABELS = {
    label.strip().lower()
    for label in os.getenv("YOLO_ALLOWED_LABELS", "person").split(",")
    if label.strip()
}


class VideoRectangleModelYoloBotSort(ControlModel):
    """
    Class representing a RectangleLabels (bounding boxes) control tag for YOLO model.
    """

    type = "VideoRectangle"
    model_path = "yolo11m.pt"

    @classmethod
    def is_control_matched(cls, control: ControlTag) -> bool:
        # check object tag type
        if control.objects[0].tag != "Video":
            return False
        # check control type VideoRectangle
        return control.tag == cls.type

    @staticmethod
    def get_from_name_for_label_map(label_interface, target_name) -> str:
        """VideoRectangle doesn't have labels inside, and we should find a connected Labels tag
        and return its name as a source for the label map.
        """
        target: ControlTag = label_interface.get_control(target_name)
        if not target:
            raise ValueError(f'Control tag with name "{target_name}" not found')

        for connected in label_interface.controls:
            if connected.tag == "Labels" and connected.to_name == target.to_name:
                return connected.name

        logger.error("VideoRectangle detected, but no connected 'Labels' tag found")

    @staticmethod
    def get_video_duration(path):
        if not os.path.exists(path):
            raise ValueError(f"Video file not found: {path}")
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        logger.info(
            f"Video duration: {duration} seconds, {frame_count} frames, {fps} fps"
        )
        return frame_count, duration
    
    def fit(self, data_yaml: str, epochs: int = 50, imgsz: int = 640, batch_size: int = 4, **kwargs) -> Dict:
        """Train YOLO model on a dataset.
        
        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size for training
            **kwargs: Additional arguments including model_path and model_version
            
        Returns:
            Dictionary with training results
        """
        # Load augmentation config based on model version
        aug_config = kwargs.get('aug_config', {})  # default empty dict if not provided
        output_dir = kwargs.get('output_dir', 'runs/train')  # default output

        try:
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=0,
                project=output_dir,
                **aug_config  # Unpack augmentation parameters from YAML
            )

            logger.info(f"Training completed successfully. Output saved to {output_dir}")

            return {
                "status": "success",
                "results": results,
            }
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def predict_regions(self, path, keyframe_interval = 5) -> List[Dict]:
        # bounding box parameters
        # https://docs.ultralytics.com/modes/track/?h=track#tracking-arguments
        conf = float(self.control.attr.get("model_conf", 0.25))
        iou = float(self.control.attr.get("model_iou", 0.70))

        # tracking parameters
        # https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers
        tracker_name = self.control.attr.get(
            "model_tracker", "botsort"
        )  # or 'bytetrack'
        original = f"{MODEL_ROOT}/{tracker_name}.yaml"
        tmp_yaml = self.update_tracker_params(original, prefix=tracker_name + "_")
        tracker = tmp_yaml if tmp_yaml else original

        # run model track
        track_kwargs = {}
        imgsz_env = os.getenv("YOLO_IMGSZ")
        if imgsz_env:
            try:
                track_kwargs["imgsz"] = int(imgsz_env)
            except ValueError:
                logger.warning(
                    "Invalid YOLO_IMGSZ value '%s'; falling back to model default",
                    imgsz_env,
                )

        try:
            results = self.model.track(
                path, 
                conf=conf, 
                iou=iou, 
                tracker=tracker, 
                stream=True, 
                **track_kwargs
            )
            # convert model results to label studio regions while tracker config exists
            return self.create_video_rectangles_in_steps(results, path, keyframe_interval=keyframe_interval)
        finally:
            # clean temporary file after inference completes
            if tmp_yaml and os.path.exists(tmp_yaml):
                os.remove(tmp_yaml)

    def create_video_rectangles(self, results, path):
        """Create regions of video rectangles from the yolo tracker results"""
        frames_count, duration = self.get_video_duration(path)
        model_names = self.model.names
        logger.debug(
            f"create_video_rectangles: {self.from_name}, {frames_count} frames"
        )

        tracks = defaultdict(list)
        track_labels = dict()
        frame = -1
        for result in results:
            frame += 1
            data = result.boxes
            if not data.is_track:
                continue

            for i, track_id in enumerate(data.id.tolist()):
                score = float(data.conf[i])
                x, y, w, h = data.xywhn[i].tolist()
                # get label
                model_label = model_names[int(data.cls[i])]
                if model_label not in self.label_map:
                    continue
                output_label = self.label_map[model_label]
                if output_label.strip().lower() not in ALLOWED_LABELS:
                    continue
                track_labels[track_id] = output_label

                box = {
                    "frame": frame + 1,
                    "enabled": True,
                    "rotation": 0,
                    "x": (x - w / 2) * 100,
                    "y": (y - h / 2) * 100,
                    "width": w * 100,
                    "height": h * 100,
                    "time": (frame + 1) * (duration / frames_count),
                    "score": score,
                }
                tracks[track_id].append(box)

        regions = []
        for track_id in tracks:
            sequence = tracks[track_id]
            sequence = self.process_lifespans_enabled(sequence)

            label = track_labels[track_id]
            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "videorectangle",
                "value": {
                    "framesCount": frames_count,
                    "duration": duration,
                    "sequence": sequence,
                    "labels": [label],
                },
                "score": max([frame_info["score"] for frame_info in sequence]),
                "origin": "manual",
            }
            regions.append(region)

        return regions

    def create_video_rectangles_in_steps(self, results, path, keyframe_interval=5):
        """Create regions of video rectangles from YOLO tracker results 
        with keyframe sampling per track"""
        frames_count, duration = self.get_video_duration(path)
        model_names = self.model.names
        logger.debug(
            f"create_video_rectangles: {self.from_name}, {frames_count} frames"
        )

        tracks = defaultdict(list)
        track_labels = dict()
        frame = -1

        # Collect all frames
        for result in results:
            frame += 1
            data = result.boxes
            if not data.is_track:
                continue

            for i, track_id in enumerate(data.id.tolist()):
                score = float(data.conf[i])
                x, y, w, h = data.xywhn[i].tolist()
                model_label = model_names[int(data.cls[i])]
                if model_label not in self.label_map:
                    continue
                output_label = self.label_map[model_label]
                if output_label.strip().lower() not in ALLOWED_LABELS:
                    continue
                track_labels[track_id] = output_label

                box = {
                    "frame": frame + 1,
                    "enabled": True,
                    "rotation": 0,
                    "x": (x - w / 2) * 100,
                    "y": (y - h / 2) * 100,
                    "width": w * 100,
                    "height": h * 100,
                    "time": (frame + 1) * (duration / frames_count),
                    "score": score,
                }
                tracks[track_id].append(box)

        regions = []
        for track_id, sequence in tracks.items():
            sequence = self.process_lifespans_enabled(sequence)

            # Sample keyframes for the track
            sampled_sequence = []
            for idx, box in enumerate(sequence):
                if idx % keyframe_interval == 0:
                    sampled_sequence.append(box)
            # Always include last frame of the track
            if sequence[-1]["frame"] != sampled_sequence[-1]["frame"]:
                sampled_sequence.append(sequence[-1])
            sequence = sampled_sequence

            label = track_labels[track_id]
            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "videorectangle",
                "value": {
                    "framesCount": frames_count,
                    "duration": duration,
                    "sequence": sequence,
                    "labels": [label],
                },
                "score": max([frame_info["score"] for frame_info in sequence]),
                "origin": "manual",
            }
            regions.append(region)

        return regions



    @staticmethod
    def process_lifespans_enabled(sequence: List[Dict]) -> List[Dict]:
        """This function detects gaps in the sequence of bboxes
        and disables lifespan line for the gaps assigning "enabled": False
        to the last bboxes in the whole span sequence.
        """
        prev = None
        for i, box in enumerate(sequence):
            if prev is None:
                prev = sequence[i]
                continue
            if box["frame"] - prev["frame"] > 1:
                sequence[i - 1]["enabled"] = False
            prev = sequence[i]

        # the last frame enabled is false to turn off lifespan line
        sequence[-1]["enabled"] = False
        return sequence

    @staticmethod
    def generate_hash_filename(extension=".yaml"):
        """Store yaml configs as temporary files just for one model.track() run"""
        hash_name = hashlib.sha256(os.urandom(16)).hexdigest()
        os.makedirs(f"{MODEL_ROOT}/tmp/", exist_ok=True)
        return f"{MODEL_ROOT}/tmp/{hash_name}{extension}"

    def update_tracker_params(self, yaml_path: str, prefix: str) -> Union[str, None]:
        """Update tracker parameters in the yaml file with the attributes from the ControlTag,
        e.g. <VideoRectangle model_tracker="bytetrack" bytetrack_max_age="10" bytetrack_min_hits="3" />
        or <VideoRectangle model_tracker="botsort" botsort_max_age="10" botsort_min_hits="3" />
        Args:
            yaml_path: Path to the original yaml file.
            prefix: Prefix for attributes of control tag to extract
        Returns:
            The file path for new yaml with updated parameters
        """
        # check if there are any custom parameters in the labeling config
        for attr_name, attr_value in self.control.attr.items():
            if attr_name.startswith(prefix):
                break
        else:
            # no custom parameters, exit
            return None

        # Load the original yaml file
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        # Extract parameters with prefix from ControlTag
        for attr_name, attr_value in self.control.attr.items():
            if attr_name.startswith(prefix):
                # Remove prefix and update the corresponding yaml key
                key = attr_name[len(prefix) :]

                # Convert value to the appropriate type (bool, int, float, etc.)
                if isinstance(config[key], bool):
                    attr_value = attr_value.lower() == "true"
                elif isinstance(config[key], int):
                    attr_value = int(attr_value)
                elif isinstance(config[key], float):
                    attr_value = float(attr_value)

                config[key] = attr_value

        # Generate a new filename with a random hash
        new_yaml_filename = self.generate_hash_filename()

        # Save the updated config to a new yaml file
        with open(new_yaml_filename, "w") as file:
            yaml.dump(config, file)

        # Return the new filename
        return new_yaml_filename


# pre-load and cache default model at startup
VideoRectangleModelYoloBotSort.get_cached_model(VideoRectangleModelYoloBotSort.model_path)
