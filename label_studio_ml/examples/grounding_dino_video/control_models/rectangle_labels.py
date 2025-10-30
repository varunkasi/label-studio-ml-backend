import logging
from typing import Dict, List

from control_models.base import ControlModel, get_bool
from label_studio_sdk.label_interface.control_tags import ControlTag


logger = logging.getLogger(__name__)


def is_obb(control: ControlTag) -> bool:
    """Check if the model should use oriented bounding boxes (OBB)."""
    return get_bool(control.attr, "model_obb", "false")


class RectangleLabelsModel(ControlModel):
    """Rectangle label control for Grounding DINO detections."""

    type = "RectangleLabels"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        if control.objects[0].tag != "Image":
            return False
        if is_obb(control):
            return False
        return control.tag == cls.type

    def predict_regions(self, path) -> List[Dict]:
        detections, height, width = self.inference.detect_image(path)
        if len(detections) == 0:
            return []

        regions: List[Dict] = []
        for bbox, score, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id
        ):
            if score < self.model_score_threshold:
                continue

            model_label = self.inference.names[int(class_id)]
            if model_label not in self.label_map:
                continue
            output_label = self.label_map[model_label]

            x1, y1, x2, y2 = bbox.tolist()
            w = x2 - x1
            h = y2 - y1

            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"bbox > {x1, y1, x2, y2}\n"
                f"model label > {model_label}\n"
                f"score > {score}\n"
            )

            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [output_label],
                    "x": x1 / width * 100,
                    "y": y1 / height * 100,
                    "width": w / width * 100,
                    "height": h / height * 100,
                },
                "score": float(score),
            }
            regions.append(region)
        return regions
