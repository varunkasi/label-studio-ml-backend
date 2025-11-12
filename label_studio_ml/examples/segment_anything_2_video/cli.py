import os
import logging
import json
import cv2

from tqdm import tqdm
from argparse import ArgumentParser
from model import NewModel
from label_studio_sdk.client import LabelStudio
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_ml.response import ModelResponse

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "your_api_key")
PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID", "1")

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def arg_parser():
    parser = ArgumentParser(description="SAM2 Video Tracker CLI for Label Studio")

    parser.add_argument(
        "--ls-url", type=str, default=LABEL_STUDIO_URL, help="Label Studio URL"
    )
    parser.add_argument(
        "--ls-api-key",
        type=str,
        default=LABEL_STUDIO_API_KEY,
        help="Label Studio API Key",
    )
    parser.add_argument(
        "--project", type=str, default="1", help="Label Studio Project ID"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task ID to process",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        required=True,
        help="Annotation ID to use as tracking prompts",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to track (default: track to end of video)",
    )
    return parser.parse_args()


class LabelStudioSAM2Predictor:
    def __init__(self, ls_url, ls_api_key):
        # Validate API key
        if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
            raise ValueError(
                "LABEL_STUDIO_API_KEY is required. Please set it via environment variable or --ls-api-key argument."
            )

        # Set environment variables for SDK internal functions (like get_local_path)
        os.environ.setdefault("LABEL_STUDIO_URL", ls_url)
        os.environ.setdefault("LABEL_STUDIO_API_KEY", ls_api_key)

        self.ls = LabelStudio(base_url=ls_url, api_key=ls_api_key)
        logger.info(f"Successfully connected to Label Studio: {ls_url}")

    def run(self, project, task_id, annotation_id, max_frames=None):
        # Initialize Label Studio SDK client
        ls = self.ls
        project = ls.projects.get(id=project)
        logger.info(f"Project retrieved: {project.id}")

        # Fetch task
        task = ls.tasks.get(task_id)
        task_dict = {"id": task.id, "data": task.data}
        logger.info(f"Task {task_id} retrieved")

        # Fetch annotation
        logger.info(f"Fetching annotation {annotation_id}...")
        annotation = ls.annotations.get(id=annotation_id)
        logger.info(f"Annotation {annotation_id} retrieved with {len(annotation.result)} regions")

        # Validate annotation
        if not annotation.result:
            logger.error(f"Annotation {annotation_id} has no regions/keyframes")
            return

        # Convert annotation to context format
        context = self._annotation_to_context(annotation, task_dict)

        # Validate/extract FPS
        fps = self._ensure_fps(task_dict, context)
        if fps:
            logger.info(f"Using FPS: {fps}")

        # Load SAM2 model
        model = NewModel(
            project_id=str(project.id),
            label_config=project.label_config
        )
        logger.info("SAM2 ML backend initialized")

        # Override max_frames if specified
        if max_frames:
            os.environ['MAX_FRAMES_TO_TRACK'] = str(max_frames)
            logger.info(f"Set MAX_FRAMES_TO_TRACK to {max_frames}")

        # Run prediction
        logger.info(f"Running SAM2 tracking on task {task_id}...")
        response = model.predict([task_dict], context=context)
        predictions = self.postprocess_response(model, response, task_dict)

        if not predictions:
            logger.error("No predictions generated")
            return

        # Send predictions to Label Studio
        for prediction in predictions:
            score = prediction.get("score", 0)
            logger.info(
                f"Submitting prediction for task {task_id} with score={score:.4f}, "
                f"{len(prediction['result'])} regions"
            )
            ls.predictions.create(
                task=task_dict["id"],
                score=score,
                model_version=prediction.get("model_version", "sam2-video"),
                result=prediction["result"],
            )

        logger.info("SAM2 tracking predictions complete!")

    def _annotation_to_context(self, annotation, task):
        """Convert Label Studio annotation to context format expected by model."""
        # The annotation.result is already in the correct format
        # Just need to wrap it
        context = {
            'result': annotation.result
        }

        # Log keyframe information
        for region in annotation.result:
            if region.get('type') == 'videorectangle':
                sequence = region.get('value', {}).get('sequence', [])
                logger.info(
                    f"Region {region.get('id')}: {len(sequence)} keyframes, "
                    f"frames: {[s['frame'] for s in sequence[:5]]}"
                    f"{'...' if len(sequence) > 5 else ''}"
                )

        return context

    def _ensure_fps(self, task, context):
        """Ensure FPS is available in task data or extract from video."""
        fps = task['data'].get('fps')

        if fps:
            logger.info(f"Using FPS from task data: {fps}")
            return fps

        # Try to extract FPS from video
        try:
            # Get video data key from context
            result = context['result'][0]
            to_name = result.get('to_name', 'video')
            video_url = task['data'].get(to_name)

            if not video_url:
                logger.warning("Could not find video URL in task data")
                return None

            # Download/cache video
            video_path = get_local_path(video_url, task_id=task['id'])

            # Extract FPS using OpenCV
            capture = cv2.VideoCapture(video_path)
            if capture.isOpened():
                fps = capture.get(cv2.CAP_PROP_FPS)
                capture.release()

                if fps > 0:
                    logger.info(f"Extracted FPS from video: {fps}")
                    # Update task data
                    task['data']['fps'] = fps
                    try:
                        self.ls.tasks.update(task['id'], data=task['data'])
                        logger.info(f"Updated task {task['id']} with fps={fps}")
                    except Exception as e:
                        logger.warning(f"Failed to update task with FPS: {e}")
                    return fps
        except Exception as e:
            logger.warning(f"Failed to extract FPS from video: {e}")

        return None

    @staticmethod
    def postprocess_response(model, response, task):
        """Process model response into predictions format."""
        if response is None:
            logger.warning(f"No predictions for task: {task}")
            return None

        # Model returned ModelResponse
        if isinstance(response, ModelResponse):
            # Check model version
            if not response.has_model_version():
                if model.model_version:
                    response.set_version(str(model.model_version))
            else:
                response.update_predictions_version()
            response = response.model_dump()
            predictions = response.get("predictions")
        # Model returned list of dicts with predictions (old format)
        elif isinstance(response, list):
            predictions = response
        else:
            logger.error("No predictions generated by model")
            return None

        return predictions


if __name__ == "__main__":
    args = arg_parser()
    predictor = LabelStudioSAM2Predictor(args.ls_url, args.ls_api_key)
    predictor.run(
        args.project,
        args.task,
        args.annotation,
        max_frames=args.max_frames,
    )
