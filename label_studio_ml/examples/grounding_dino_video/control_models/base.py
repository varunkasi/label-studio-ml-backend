import base64
import logging
import os
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

from pydantic import BaseModel
from typing import Optional, List, Dict, ClassVar

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.label_interface.control_tags import ControlTag
from label_studio_sdk.label_interface import LabelInterface

from utils.grounding import GroundingDINOInference


DEBUG_PLOT = os.getenv("DEBUG_PLOT", "false").lower() in ["1", "true"]
MODEL_SCORE_THRESHOLD_ENV = os.getenv("MODEL_SCORE_THRESHOLD")
MODEL_SCORE_THRESHOLD_DEFAULT = 0.5
MODEL_SCORE_THRESHOLD = (
    float(MODEL_SCORE_THRESHOLD_ENV)
    if MODEL_SCORE_THRESHOLD_ENV not in (None, "")
    else MODEL_SCORE_THRESHOLD_DEFAULT
)

logger = logging.getLogger(__name__)


def get_bool(attr, attr_name, default="false"):
    return attr.get(attr_name, default).lower() in ["1", "true", "yes"]


class ControlModel(BaseModel):
    """
    Represents a control tag in Label Studio, which is associated with a specific type of labeling task
    and is used to generate predictions using Grounding DINO inference.

    Attributes:
        type (str): Type of the control, e.g., RectangleLabels, Choices, etc.
        control (ControlTag): The actual control element from the Label Studio configuration.
        from_name (str): The name of the control tag, used to link the control to the data.
        to_name (str): The name of the data field that this control is associated with.
        value (str): The value name from the object that this control operates on, e.g., an image or text field.
        inference (GroundingDINOInference): Shared Grounding DINO inference helper.
        model_score_threshold (float): Threshold for prediction scores; predictions below this value will be ignored.
        label_map (Optional[Dict[str, str]]): A mapping of model labels to Label Studio labels.
    """

    type: ClassVar[str]
    control: ControlTag
    from_name: str
    to_name: str
    value: str
    inference: GroundingDINOInference
    model_score_threshold: float = 0.5
    label_map: Optional[Dict[str, str]] = {}
    label_studio_ml_backend: LabelStudioMLBase
    project_id: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def is_control_matched(cls, control) -> bool:
        """Check if the control tag matches the model type.
        Args:
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    @staticmethod
    def get_from_name_for_label_map(
        label_interface: LabelInterface, target_name: str
    ) -> str:
        """Get the 'from_name' attribute for the label map building."""
        return target_name

    @classmethod
    def create(cls, mlbackend: LabelStudioMLBase, control: ControlTag):
        """Factory method to create an instance of a specific control model class.
        Args:
            mlbackend (LabelStudioMLBase): The ML backend instance.
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        from_name = control.name
        to_name = control.to_name[0]
        value = control.objects[0].value_name

        # if skip is true, don't process this control
        if get_bool(control.attr, "model_skip", "false"):
            logger.info(
                f"Skipping control tag '{control.tag}' with name '{from_name}', model_skip=true found"
            )
            return None
        # read threshold attribute from the control tag, e.g.: <RectangleLabels model_score_threshold="0.5">
        control_threshold = (
            control.attr.get("model_score_threshold")
            or control.attr.get(
                "score_threshold"
            )  # not recommended option, use `model_score_threshold`
        )

        if MODEL_SCORE_THRESHOLD_ENV not in (None, ""):
            model_score_threshold = MODEL_SCORE_THRESHOLD
        elif control_threshold is not None:
            model_score_threshold = float(control_threshold)
        else:
            model_score_threshold = MODEL_SCORE_THRESHOLD_DEFAULT
        inference = GroundingDINOInference.get_instance()
        model_names = inference.names.values()
        # from_name for label mapping can be differed from control.name (e.g. VideoRectangle)
        label_map_from_name = cls.get_from_name_for_label_map(
            mlbackend.label_interface, from_name
        )
        label_map = mlbackend.build_label_map(label_map_from_name, model_names)

        return cls(
            control=control,
            from_name=from_name,
            to_name=to_name,
            value=value,
            inference=inference,
            model_score_threshold=model_score_threshold,
            label_map=label_map,
            label_studio_ml_backend=mlbackend,
            project_id=mlbackend.project_id,
        )

    def debug_plot(self, image):
        if not DEBUG_PLOT:
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(image[..., ::-1])
        plt.axis("off")
        plt.title(self.type)
        plt.show()

    def predict_regions(self, path, output_dir=None, save_frames=False, batch_size=None) -> List[Dict]:
        """Predict regions in the image/video using the model.
        Args:
            path (str): Path to the file with media
            output_dir (str, optional): Directory to save annotated frames
            save_frames (bool): Whether to save annotated frames
            batch_size (int, optional): Batch size for processing
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    def fit(self, event, data, **kwargs):
        """Fit the model."""
        logger.warning("The fit method is not implemented for this control model")
        return False

    def get_path(self, task):
        task_path = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )
        if task_path is None:
            raise ValueError(
                f"Can't load path using key '{self.value}' from task {task}"
            )
        if not isinstance(task_path, str):
            raise ValueError(f"Path should be a string, but got {task_path}")

        if not task_path.startswith("http") and task_path.startswith("/"):
            host = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL")
            if host:
                task_path = urljoin(host.rstrip("/"), task_path)
            else:
                logger.debug(
                    "Relative task path %s found but LABEL_STUDIO_HOST/LABEL_STUDIO_URL is not set",
                    task_path,
                )

        download_source = task_path

        # try path as local file or try to load it from Label Studio instance/download via http
        path = (
            task_path
            if os.path.exists(task_path)
            else get_local_path(task_path, task_id=task.get("id"))
        )

        if os.path.exists(path):
            suffix = Path(path).suffix
            parsed = urlparse(download_source)
            if not suffix:
                parsed_suffix = Path(parsed.path).suffix
                if not parsed_suffix:
                    query = parse_qs(parsed.query)
                    fileuri = (query.get("fileuri") or [None])[0]
                    if fileuri:
                        padding = (-len(fileuri)) % 4
                        fileuri_padded = fileuri + ("=" * padding)
                        try:
                            decoded = base64.urlsafe_b64decode(fileuri_padded).decode("utf-8")
                        except (ValueError, UnicodeDecodeError):
                            decoded = ""
                        parsed_suffix = Path(decoded).suffix

                if parsed_suffix:
                    candidate_path = f"{path}{parsed_suffix}"
                    if not os.path.exists(candidate_path):
                        try:
                            os.symlink(path, candidate_path)
                        except OSError:
                            shutil.copyfile(path, candidate_path)
                    path = candidate_path

        logger.debug(f"load_image: {task_path} => {path}")
        return path

    def __str__(self):
        """Return a string with full representation of the control tag."""
        return (
            f"{self.type} from_name={self.from_name}, "
            f"label_map={self.label_map}, model_score_threshold={self.model_score_threshold}"
        )

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ("__.*__", "_.*")  # Excludes 'model_'
