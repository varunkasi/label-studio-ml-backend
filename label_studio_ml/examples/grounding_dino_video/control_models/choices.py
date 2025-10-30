import logging
from control_models.base import ControlModel
from typing import List, Dict


logger = logging.getLogger(__name__)


class ChoicesModel(ControlModel):
    """Placeholder for image classification controls (not supported with Grounding DINO)."""

    type = "Choices"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        return False

    def predict_regions(self, path) -> List[Dict]:
        logger.warning(
            "ChoicesModel is disabled: Grounding DINO backend currently supports detection tasks only."
        )
        return []
