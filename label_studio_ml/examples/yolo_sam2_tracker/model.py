import os
import logging

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from control_models.base import ControlModel
from control_models.choices import ChoicesModel
from control_models.rectangle_labels import RectangleLabelsModel
from control_models.rectangle_labels_obb import RectangleLabelsObbModel
from control_models.polygon_labels import PolygonLabelsModel
from control_models.keypoint_labels import KeypointLabelsModel
from control_models.video_rectangle import VideoRectangleModel
from control_models.timeline_labels import TimelineLabelsModel
from control_models.video_rectangle_yoloBotSort import VideoRectangleModelYoloBotSort
from typing import List, Dict, Optional

from pathlib import Path
import yaml
from workspace.utils.convert_ls2yolo import convert_labelstudio_to_yolo
from workspace.utils.YOLO_helper import get_augmentation_config, get_next_run_number, generate_unique_dataset_dirs, combine_yolo_datasets, delete_folder
from datetime import datetime

logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# Register available model classes
available_model_classes = [
    ChoicesModel,
    RectangleLabelsModel,
    RectangleLabelsObbModel,
    PolygonLabelsModel,
    KeypointLabelsModel,
    VideoRectangleModel,
    TimelineLabelsModel,
    VideoRectangleModelYoloBotSort
]


class YOLO(LabelStudioMLBase):
    """Label Studio ML Backend based on Ultralytics YOLO"""

    def setup_yoloBotsort(self, args):
        """Custom setup with CLI arguments"""
        self.args = args
        self.useYoloBotSort = True
        print("useYoloBotSort set")

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "yolo")

    def detect_control_models(self, custom_model_path = None) -> List[ControlModel]:
        """Detect control models based on the labeling config.
        Control models are used to predict regions for different control tags in the labeling config.
        """
        control_models = []

        # print(self.label_interface.controls)

        # # Joshua
        # if self.useYoloBotSort:
        #     logger.info("YOLO BotSort is enabled for video tracking")
        #     control_models.append(VideoRectangleModelYoloBotSort.create(self, 
        #                                                                 self.label_interface.controls[0]))
        #     return control_models   # Skip other control models if YoloBotSort is used

        for control in self.label_interface.controls:
            # skipping tags without toName
            if not control.to_name:
                logger.warning(
                    f'{control.tag} {control.name} has no "toName" attribute, skipping it'
                )
                continue

            # match control tag with available control models
            for model_class in available_model_classes:
                # Joshua
                if model_class != VideoRectangleModelYoloBotSort:
                    continue
                
                if model_class.is_control_matched(control):
                    # print("VideoRectangleModelYoloBotSort")
                    
                    instance = model_class.create(self, control, custom_model_path = custom_model_path)
                    if not instance:
                        logger.debug(
                            f"No instance created for {control.tag} {control.name}"
                        )
                        continue
                    if not instance.label_map:
                        logger.error(
                            f"No label map built for the '{control.tag}' control tag '{instance.from_name}'.\n"
                            f"This indicates that your Label Studio config labels do not match the model's labels.\n"
                            f"To fix this, ensure that the 'value' or 'predicted_values' attribute "
                            f"in your Label Studio config matches one or more of these model labels.\n"
                            f"If you don't want to use this control tag for predictions, "
                            f'add `model_skip="true"` to it.\n'
                            f"Examples:\n"
                            f'  <Label value="Car"/>\n'
                            f'  <Label value="YourLabel" predicted_values="label1,label2"/>\n'
                            f"Labels provided in your labeling config:\n"
                            f"  {str(control.labels_attrs)}\n"
                            f"Available '{instance.model_path}' model labels:\n"
                            f"  {list(instance.model.names.values())}"
                        )
                        continue

                    control_models.append(instance)
                    logger.debug(f"Control tag with model detected: {instance}")
                    break

        if not control_models:
            control_tags = ", ".join([c.type for c in available_model_classes])
            raise ValueError(
                f"No suitable control tags (e.g. {control_tags} connected to Image or Video object tags) "
                f"detected in the label config:\n{self.label_config}"
            )

        return control_models

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Run YOLO predictions on the tasks
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions [Predictions array in JSON format]
            (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(
            f"Run prediction on {len(tasks)} tasks, project ID = {self.project_id}"
        )
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:

            regions = []
            for model in control_models:
                path = model.get_path(task) # returns path of media
                print(f"[YOLO] {model.__class__.__name__} using media at {path}")
                regions += model.predict_regions(path)

            # calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
            }
            predictions.append(prediction)

        return ModelResponse(predictions=predictions)
    
    @staticmethod
    def get_augmentation_config(model_version: str) -> Dict:
        """Load augmentation configuration from YAML file based on model version.
        
        Args:
            model_version: Model version identifier (e.g., 'UAV_RGB', 'UGV_IR')
            
        Returns:
            Dictionary containing augmentation parameters for the model
        """
        # Get the directory where this file is located
        config_path = "workspace/autotrain/augmentations" + f"{model_version}.yaml"
        
        config_path = Path(f"workspace/autotrain/augmentations/{model_version}.yaml")

        if not config_path.exists():
            logger.warning(
                f"!!!Augmentation config for '{model_version}' not found at {config_path}. !!!"
                f"Using default UAV_RGB config."
            )
            config_path = Path("workspace/autotrain/augmentations/UAV_RGB.yaml")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded augmentation config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load augmentation config: {str(e)}")
            return {}

    @staticmethod
    def get_next_run_number(base_path: str) -> int:
        """Get the next run number for a given model version.
        
        Args:
            base_path: Base path like 'train_weights/UAV_RGB'
            
        Returns:
            Next available run number (1, 2, 3, etc.)
        """
        import glob
        
        parent_dir = os.path.dirname(base_path)
        model_name = os.path.basename(base_path)
        
        # Find all directories matching the pattern: model_name_1, model_name_2, etc.
        pattern = os.path.join(parent_dir, f"{model_name}_*")
        existing = glob.glob(pattern)
        
        if not existing:
            return 1
        
        # Extract numbers and find the max
        numbers = []
        for path in existing:
            try:
                num = int(os.path.basename(path).split("_")[-1])
                numbers.append(num)
            except (ValueError, IndexError):
                continue
        
        return max(numbers) + 1 if numbers else 1

    def fit(self, event, task, **kwargs):
        """
        This method is called each time an annotation is created or updated.
        Or it's called when "Start training" clicked on the model in the project settings.

        """

        model_path = kwargs.get("model_path", "yolo11m.pt")
        model_version = kwargs.get("model_version", "UAV_RGB")
        classes = kwargs.get("classes", None)
        annotations_ls = kwargs.get("annotations_ls", None)
        
        results = {}
        control_models = self.detect_control_models(custom_model_path=model_path)


        for model in control_models:
            video_path = model.get_path(task) # returns path of media by downloading if needed

            # Prepare dataset in YOLO format 
            # temp_folder = generate_unique_dataset_dirs()

            # output_labels_dir, output_frames_dir = convert_labelstudio_to_yolo(
            #                                         labelstudio_json = annotations_ls,
            #                                         output_labels_dir = temp_folder + "/labels",
            #                                         output_frames_dir = temp_folder + "/images",
            #                                         video_path = video_path,
            #                                         jpeg_quality = 95,
            #                                         class_names=None,
            #                                         save_empty_labels = False,
            #                                         reencode_video = False,
            #                                         reencode_fps = None
            #                                     )

            # # Split the dataset into train/val/test and prepare yaml file 
            # yaml_file_path = combine_yolo_datasets(
            #     source_dirs=[temp_folder],
            #     output_dir=temp_folder + '/split',
            #     train_ratio=0.8,
            #     val_ratio=0.2,
            #     test_ratio=0.0,
            #     skip_empty_labels=True,
            #     class_names=classes
            # )


            # ----- Temporary debugging -----
            temp_folder = "workspace/autotrain/temp/6b59f269"
            yaml_file_path = "workspace/autotrain/temp/6b59f269/split/dataset.yaml"
            # --------

            logger.info(f"Training split created")

            # Load augmentation config based on model version
            aug_config = self.get_augmentation_config(model_version)

            # Debugging
            # print(yaml_file_path)
            # print(aug_config)

            # Setup training output directory: train_weights/model_version_x
            base_output_dir = "workspace/autotrain/train_weights"
            os.makedirs(base_output_dir, exist_ok=True)
            
            base_path = os.path.join(base_output_dir, model_version)
            run_number = self.get_next_run_number(base_path)
            project_name = f'{model_version}_{datetime.now().strftime("%Y%m%d")}_{run_number}'
            final_output_dir = os.path.join(base_output_dir, project_name)
            os.makedirs(final_output_dir, exist_ok=True)

            logger.info(f"Training {model_version} - Run {run_number}")
            logger.info(f"Output directory: {final_output_dir}")

            kwargs['aug_config'] = aug_config
            kwargs['output_dir'] = final_output_dir

            print(f"[YOLO] {model.__class__.__name__} using training data splits at {temp_folder}")

            # Run training
            training_result = model.fit(data_yaml = yaml_file_path, **kwargs)

            # TODO: Cleanup files (downloaded video, dataset split, etc.)
            # delete_folder(temp_folder)

        return results
