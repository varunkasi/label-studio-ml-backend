#!/usr/bin/env python
"""
CLI for SAM2 Video Tracking - Process Label Studio tasks with existing annotations

This CLI allows batch processing of video tracking tasks by:
1. Fetching task and annotation data from Label Studio
2. Running SAM2 tracking on keyframes
3. Uploading predictions back to Label Studio

Usage:
    python cli.py --ls-url https://app.heartex.com --ls-api-key YOUR_KEY \
                  --project 123 --task 456 --annotation 789
"""

import os
import sys
import argparse
import logging
import signal
import time
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


class CLIError(Exception):
    """Custom CLI error for graceful failure"""
    pass


def setup_signal_handlers():
    """Setup graceful shutdown handlers"""
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.warning(f'\n⚠️  Received signal {signal_name}, shutting down gracefully...')
        sys.exit(130)  # Standard exit code for SIGINT
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_environment():
    """Validate required environment variables and system requirements"""
    logger.info('🔍 Validating environment...')
    
    # Check CUDA availability
    try:
        import torch
        if os.getenv('DEVICE', 'cuda') == 'cuda':
            if not torch.cuda.is_available():
                logger.error('❌ CUDA not available but DEVICE=cuda')
                raise CLIError('GPU required but not available')
            logger.info(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    except ImportError:
        logger.error('❌ PyTorch not installed')
        raise CLIError('PyTorch is required')
    
    # Check SAM2 installation
    try:
        from sam2.build_sam import build_sam2_video_predictor
        logger.info('✅ SAM2 installed')
    except ImportError:
        logger.error('❌ SAM2 not installed')
        raise CLIError('SAM2 is required. Run: pip install -e /sam2')
    
    logger.info('✅ Environment validation complete')


def fetch_task_data(ls, project_id: int, task_id: int, annotation_id: int):
    """Fetch task and annotation data from Label Studio with timeout"""
    logger.info(f'📥 Fetching task {task_id} from project {project_id}...')
    
    start_time = time.time()
    timeout = 60  # 60 second timeout for API calls
    
    try:
        # Fetch task
        task_obj = ls.tasks.get(task_id)
        task = {"id": task_obj.id, "data": task_obj.data}
        if not task:
            raise CLIError(f'Task {task_id} not found')
        logger.info(f'✅ Task fetched: {task.get("id")}')
        
        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError(f'Task fetch exceeded {timeout}s timeout')
        
        # Fetch annotation
        logger.info(f'📥 Fetching annotation {annotation_id}...')
        annotations = ls.annotations.list(task=task_id)
        annotation = next((a for a in annotations if a.id == annotation_id), None)
        
        if not annotation:
            raise CLIError(f'Annotation {annotation_id} not found in task {task_id}')
        
        # Convert annotation to dict format
        annotation_dict = {
            "id": annotation.id,
            "result": annotation.result
        }
        
        logger.info(f'✅ Annotation fetched: {annotation_dict.get("id")} with {len(annotation_dict.get("result", []))} regions')
        
        # Validate annotation has regions
        if not annotation_dict.get('result'):
            raise CLIError(f'Annotation {annotation_id} has no keyframe regions')
        
        return task, annotation_dict
        
    except TimeoutError:
        logger.error(f'❌ API request timed out after {timeout}s')
        raise
    except Exception as e:
        logger.error(f'❌ Failed to fetch data: {e}')
        raise CLIError(f'API error: {e}')


def upload_prediction(ls, task_id: int, prediction_data: dict):
    """Upload prediction to Label Studio"""
    logger.info(f'📤 Uploading prediction for task {task_id}...')
    
    try:
        result = ls.predictions.create(
            task=task_id,
            score=prediction_data.get('score', 0),
            model_version=prediction_data.get('model_version', 'none'),
            result=prediction_data.get('result', [])
        )
        logger.info(f'✅ Prediction uploaded successfully: ID={result.get("id")}')
        return result
    except Exception as e:
        logger.error(f'❌ Failed to upload prediction: {e}')
        raise CLIError(f'Upload error: {e}')


def main():
    """Main CLI execution"""
    parser = argparse.ArgumentParser(
        description='SAM2 Video Tracking CLI - Process Label Studio tasks with keyframe annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Track a video with keyframes
  python cli.py --ls-url https://app.heartex.com --ls-api-key YOUR_KEY \\
                --project 123 --task 456 --annotation 789

  # Limit tracking to 100 frames
  python cli.py --ls-url https://app.heartex.com --ls-api-key YOUR_KEY \\
                --project 123 --task 456 --annotation 789 --max-frames 100
        """
    )
    
    parser.add_argument('--ls-url', required=True, help='Label Studio URL (e.g., https://app.heartex.com)')
    parser.add_argument('--ls-api-key', required=True, help='Label Studio API key')
    parser.add_argument('--project', type=int, required=True, help='Project ID')
    parser.add_argument('--task', type=int, required=True, help='Task ID to process')
    parser.add_argument('--annotation', type=int, required=True, help='Annotation ID with keyframes')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames to track (0 = unlimited)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Setup signal handlers
    setup_signal_handlers()
    
    logger.info('='*80)
    logger.info('🚀 SAM2 VIDEO CLI STARTED')
    logger.info('='*80)
    logger.info(f'📋 Parameters:')
    logger.info(f'   • Label Studio URL: {args.ls_url}')
    logger.info(f'   • Project ID: {args.project}')
    logger.info(f'   • Task ID: {args.task}')
    logger.info(f'   • Annotation ID: {args.annotation}')
    logger.info(f'   • Max frames: {args.max_frames if args.max_frames > 0 else "unlimited"}')
    logger.info('='*80)
    
    exit_code = 0
    
    try:
        # Validate environment
        validate_environment()
        
        # Initialize Label Studio client
        logger.info('🔗 Connecting to Label Studio...')
        from label_studio_sdk.client import LabelStudio
        
        ls = LabelStudio(base_url=args.ls_url, api_key=args.ls_api_key)
        logger.info('✅ Connected to Label Studio')
        
        # Fetch task and annotation data
        task, annotation = fetch_task_data(ls, args.project, args.task, args.annotation)
        
        # Get label config from project
        logger.info(f'📥 Fetching project configuration...')
        project = ls.projects.get(id=args.project)
        label_config = project.label_config or project.parsed_label_config
        
        if not label_config:
            raise CLIError('Could not fetch label config from project')
        
        logger.info('✅ Label config fetched')
        
        # Initialize model
        logger.info('🧠 Initializing SAM2 model...')
        from model import NewModel
        
        # Override MAX_FRAMES_TO_TRACK if specified
        if args.max_frames > 0:
            os.environ['MAX_FRAMES_TO_TRACK'] = str(args.max_frames)
            logger.info(f'⚙️  Set MAX_FRAMES_TO_TRACK={args.max_frames}')
        
        model = NewModel(label_config=label_config)
        logger.info('✅ Model initialized')
        
        # Prepare context from annotation
        context = {
            'result': annotation['result']
        }
        
        # Run prediction
        logger.info('🎬 Starting SAM2 tracking...')
        start_time = time.time()
        
        response = model.predict(tasks=[task], context=context)
        
        elapsed = time.time() - start_time
        logger.info(f'✅ Tracking complete in {elapsed:.2f}s')
        
        # Extract prediction data
        if not response or not response.predictions:
            raise CLIError('Model returned no predictions')
        
        prediction = response.predictions[0]
        prediction_data = prediction.model_dump() if hasattr(prediction, 'model_dump') else prediction
        
        # Upload prediction
        upload_prediction(ls, args.task, prediction_data)
        
        logger.info('='*80)
        logger.info('✅ CLI EXECUTION SUCCESSFUL')
        logger.info('='*80)
        
    except KeyboardInterrupt:
        logger.warning('\n⚠️  Interrupted by user')
        exit_code = 130
    except CLIError as e:
        logger.error(f'❌ CLI Error: {e}')
        exit_code = 1
    except TimeoutError as e:
        logger.error(f'❌ Timeout Error: {e}')
        exit_code = 124  # Standard timeout exit code
    except Exception as e:
        logger.error(f'❌ Unexpected error: {e}', exc_info=True)
        exit_code = 1
    finally:
        if exit_code != 0:
            logger.info('='*80)
            logger.info(f'❌ CLI EXECUTION FAILED (exit code: {exit_code})')
            logger.info('='*80)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
