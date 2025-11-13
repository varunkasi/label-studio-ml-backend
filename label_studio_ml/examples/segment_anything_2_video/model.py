import torch
import numpy as np
import os
import pathlib
import cv2
import tempfile
import logging
from urllib.parse import urljoin

from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.label_interface.objects import PredictionValue
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from tqdm import tqdm

logger = logging.getLogger(__name__)


DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'sam2_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2_hiera_large.pt')
MAX_FRAMES_TO_TRACK = int(os.getenv('MAX_FRAMES_TO_TRACK', 10))

if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
sam2_checkpoint = str(pathlib.Path(__file__).parent / "/sam2" / "checkpoints" / MODEL_CHECKPOINT)
predictor = build_sam2_video_predictor(MODEL_CONFIG, sam2_checkpoint)


# manage cache for inference state
# TODO: make it process-safe and implement cache invalidation
_predictor_state_key = ''
_inference_state = None

def get_inference_state(video_dir):
    global _predictor_state_key, _inference_state
    if _predictor_state_key != video_dir:
        _predictor_state_key = video_dir
        _inference_state = predictor.init_state(video_path=video_dir)
    return _inference_state


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def split_frames(self, video_path, temp_dir, start_frame=0, end_frame=100):
        # Open the video file
        logger.info(f'📹 Opening video file: {video_path}')
        video = cv2.VideoCapture(video_path)

        # check if loaded correctly
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames_in_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_extract = end_frame - start_frame
        logger.info(f'📊 Video has {total_frames_in_video} total frames')
        logger.info(f'🎬 Extracting frames {start_frame} to {end_frame} ({frames_to_extract} frames)')

        frame_count = 0
        extracted_count = 0
        while True:
            # Read a frame from the video
            success, frame = video.read()
            if frame_count < start_frame:
                frame_count += 1
                continue
            if frame_count >= end_frame:
                break

            # If frame is read correctly, success is True
            if not success:
                logger.error(f'❌ Failed to read frame {frame_count}')
                break

            # Generate a filename for the frame using the pattern with frame number: '%05d.jpg'
            frame_filename = os.path.join(temp_dir, f'{frame_count:05d}.jpg')
            if os.path.exists(frame_filename):
                logger.debug(f'Frame {frame_count}: {frame_filename} already exists')
                yield frame_filename, frame
            else:
                # Save the frame as an image file
                cv2.imwrite(frame_filename, frame)
                logger.debug(f'Frame {frame_count}: {frame_filename}')
                yield frame_filename, frame

            extracted_count += 1
            # Log progress every 10 frames
            if extracted_count % 10 == 0:
                logger.info(f'⏳ Extracted {extracted_count}/{frames_to_extract} frames...')

            frame_count += 1

        # Release the video object
        video.release()
        logger.info(f'✅ Frame extraction complete: {extracted_count} frames extracted')

    def get_prompts(self, context) -> List[Dict]:
        logger.debug(f'Extracting keypoints from context: {context}')
        prompts = []
        for ctx in context['result']:
            # Process each video tracking object separately
            obj_id = ctx['id']
            for obj in ctx['value']['sequence']:
                x = obj['x'] / 100
                y = obj['y'] / 100
                box_width = obj['width'] / 100
                box_height = obj['height'] / 100
                frame_idx = obj['frame'] - 1

                # SAM2 video works with keypoints - convert the rectangle to the set of keypoints within the rectangle

                # bbox (x, y) is top-left corner
                kps = [
                    # center of the bbox
                    [x + box_width / 2, y + box_height / 2],
                    # half of the bbox width to the left
                    [x + box_width / 4, y + box_height / 2],
                    # half of the bbox width to the right
                    [x + 3 * box_width / 4, y + box_height / 2],
                    # half of the bbox height to the top
                    [x + box_width / 2, y + box_height / 4],
                    # half of the bbox height to the bottom
                    [x + box_width / 2, y + 3 * box_height / 4]
                ]

                points = np.array(kps, dtype=np.float32)
                labels = np.array([1] * len(kps), dtype=np.int32)
                prompts.append({
                    'points': points,
                    'labels': labels,
                    'frame_idx': frame_idx,
                    'obj_id': obj_id
                })

        return prompts

    def _get_fps(self, context):
        # get the fps from the context
        frames_count = context['result'][0]['value']['framesCount']
        duration = context['result'][0]['value']['duration']
        return frames_count, duration

    # def convert_mask_to_bbox(self, mask):
    #     # convert mask to bbox
    #     h, w = mask.shape[-2:]
    #     mask_int = mask.reshape(h, w, 1).astype(np.uint8)
    #     contours, _ = cv2.findContours(mask_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 0:
    #         return None
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     return {
    #         'x': x,
    #         'y': y,
    #         'width': w,
    #         'height': h
    #     }

    def convert_mask_to_bbox(self, mask):
        # squeeze
        mask = mask.squeeze()

        y_indices, x_indices = np.where(mask == 1)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        # Find the min and max indices
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        # Get mask dimensions
        height, width = mask.shape

        # Calculate bounding box dimensions
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1

        # Normalize and scale to percentage
        x_pct = (xmin / width) * 100
        y_pct = (ymin / height) * 100
        width_pct = (box_width / width) * 100
        height_pct = (box_height / height) * 100

        return {
            "x": round(x_pct, 2),
            "y": round(y_pct, 2),
            "width": round(width_pct, 2),
            "height": round(height_pct, 2)
        }


    def dump_image_with_mask(self, frame, mask, output_file, obj_id=None, random_color=False):
        from matplotlib import pyplot as plt
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # create an image file to display image overlayed with mask
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2BGR)
        mask_image = cv2.addWeighted(frame, 1.0, mask_image, 0.8, 0)
        logger.debug(f'Shapes: frame={frame.shape}, mask={mask.shape}, mask_image={mask_image.shape}')
        # save in file
        logger.debug(f'Saving image with mask to {output_file}')
        cv2.imwrite(output_file, mask_image)


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        logger.info('='*80)
        logger.info('🎬 SAM2 VIDEO TRACKING STARTED')
        logger.info('='*80)

        from_name, to_name, value = self.get_first_tag_occurence('VideoRectangle', 'Video')

        task = tasks[0]
        task_id = task['id']
        logger.info(f'📋 Processing task ID: {task_id}')

        # Get the video URL from the task
        video_url = task['data'][value]
        logger.info(f'🔗 Video URL: {video_url}')

        # Resolve relative URL if needed
        if not video_url.startswith("http") and video_url.startswith("/"):
            host = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL")
            if host:
                video_url = urljoin(host.rstrip("/"), video_url)
            else:
                logger.debug(
                    "Relative video URL %s found but LABEL_STUDIO_HOST/LABEL_STUDIO_URL is not set",
                    video_url,
                )

        # cache the video locally
        logger.info(f'⬇️  Downloading/caching video...')
        video_path = get_local_path(video_url, task_id=task_id)
        logger.info(f'💾 Video cached at: {video_path}')

        # get prompts from context
        logger.info(f'🔍 Extracting prompts from annotation context...')
        prompts = self.get_prompts(context)
        all_obj_ids = set(p['obj_id'] for p in prompts)
        # create a map from obj_id to integer
        obj_ids = {obj_id: i for i, obj_id in enumerate(all_obj_ids)}
        # find the last frame index
        first_frame_idx = min(p['frame_idx'] for p in prompts) if prompts else 0
        last_frame_idx = max(p['frame_idx'] for p in prompts) if prompts else 0
        frames_count, duration = self._get_fps(context)
        fps = frames_count / duration

        logger.info(
            f'📍 Found {len(prompts)} prompt(s) for {len(obj_ids)} object(s), '
            f'keyframes range: [{first_frame_idx}, {last_frame_idx}]')
        logger.debug(f'Object ID mapping: {obj_ids}')

        # Calculate frames to track: from first keyframe to end of video
        # If MAX_FRAMES_TO_TRACK is set (not None), use it as a limit
        if MAX_FRAMES_TO_TRACK > 0:
            frames_to_track = min(MAX_FRAMES_TO_TRACK, frames_count - first_frame_idx)
            logger.info(f'Tracking limited to {frames_to_track} frames (MAX_FRAMES_TO_TRACK={MAX_FRAMES_TO_TRACK})')
        else:
            frames_to_track = frames_count - first_frame_idx
            logger.info(f'Tracking full video: {frames_to_track} frames from frame {first_frame_idx} to {frames_count}')

        # Split the video into frames
        with tempfile.TemporaryDirectory() as temp_dir:

            # # use persisted dir for debug
            # temp_dir = '/tmp/frames'
            # os.makedirs(temp_dir, exist_ok=True)

            # get all frames from first keyframe to end of tracking range
            end_frame = first_frame_idx + frames_to_track
            frames = list(self.split_frames(
                video_path, temp_dir,
                start_frame=first_frame_idx,
                end_frame=end_frame
            ))
            height, width, _ = frames[0][1].shape
            logger.info(f'📐 Video dimensions: {width}x{height}')

            # get inference state
            logger.info(f'🧠 Initializing SAM2 inference state...')
            inference_state = get_inference_state(temp_dir)
            predictor.reset_state(inference_state)
            logger.info(f'✅ Inference state initialized')

            logger.info(f'📌 Adding {len(prompts)} tracking prompt(s) to SAM2...')
            for idx, prompt in enumerate(prompts, 1):
                # multiply points by the frame size
                prompt['points'][:, 0] *= width
                prompt['points'][:, 1] *= height

                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=prompt['frame_idx'],
                    obj_id=obj_ids[prompt['obj_id']],
                    points=prompt['points'],
                    labels=prompt['labels']
                )
                logger.info(f'  ✓ Prompt {idx}/{len(prompts)}: frame={prompt["frame_idx"]}, obj_id={prompt["obj_id"]}, points={len(prompt["points"])}')

            logger.info(f'✅ All prompts added successfully')

            # Dictionary to store sequences per object (for multi-person tracking)
            from collections import defaultdict
            sequences_by_obj = defaultdict(list)

            debug_dir = './debug-frames'
            os.makedirs(debug_dir, exist_ok=True)

            logger.info(f'🚀 Starting SAM2 video propagation from frame {first_frame_idx} to {end_frame}')
            logger.info(f'🎯 Tracking {len(obj_ids)} object(s) across {frames_to_track} frames')

            # Create progress bar for tracking
            pbar = tqdm(
                total=frames_to_track,
                desc="🎥 Tracking frames",
                unit="frame",
                ncols=100
            )

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=first_frame_idx,
                max_frame_num_to_track=frames_to_track
            ):
                real_frame_idx = out_frame_idx + first_frame_idx
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                    # to debug, save the mask as an image
                    # self.dump_image_with_mask(frames[out_frame_idx][1], mask, f'{debug_dir}/{out_frame_idx:05d}_{out_obj_id}.jpg', obj_id=out_obj_id, random_color=True)

                    bbox = self.convert_mask_to_bbox(mask)
                    if bbox:
                        # Append to the specific object's sequence
                        sequences_by_obj[out_obj_id].append({
                            'frame': real_frame_idx + 1,
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'enabled': True,
                            'rotation': 0,
                            'time': real_frame_idx / fps
                        })

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'frame': real_frame_idx + 1,
                    'objects': len(out_obj_ids)
                })

            pbar.close()
            logger.info(f'✅ Video propagation complete!')

            # Create a map from obj_id (SAM2 internal ID) to original annotation ID
            # obj_ids maps original annotation ID -> SAM2 internal ID
            # We need the reverse: SAM2 internal ID -> original annotation ID
            reverse_obj_ids = {v: k for k, v in obj_ids.items()}

            # Get keyframes from context (one result per person)
            context_results = {r['id']: r for r in context['result']}

            # Build separate regions for each tracked person (multi-person tracking)
            regions = []
            for sam_obj_id, predicted_sequence in sequences_by_obj.items():
                # Get the original annotation ID
                original_obj_id = reverse_obj_ids.get(sam_obj_id)

                if original_obj_id not in context_results:
                    logger.warning(f'Could not find context result for obj_id {original_obj_id}')
                    continue

                # Get the original keyframes from context
                context_result = context_results[original_obj_id]
                original_keyframes = context_result['value'].get('sequence', [])

                # Get labels from context
                labels = context_result['value'].get('labels', ['Person'])

                # Merge original keyframes with predicted sequence
                # Sort by frame number to maintain temporal order
                merged_sequence = original_keyframes + predicted_sequence
                merged_sequence = sorted(merged_sequence, key=lambda x: x['frame'])

                # Calculate score (use 1.0 as default for SAM2 tracking)
                avg_score = 1.0

                region = {
                    'value': {
                        'framesCount': frames_count,
                        'duration': duration,
                        'sequence': merged_sequence,
                        'labels': labels,
                    },
                    'from_name': 'box',
                    'to_name': 'video',
                    'type': 'videorectangle',
                    'origin': 'manual',
                    'id': original_obj_id,
                    'score': avg_score,
                }
                regions.append(region)
                logger.info(
                    f'Created region for person {original_obj_id}: '
                    f'{len(original_keyframes)} keyframes + {len(predicted_sequence)} tracked frames = '
                    f'{len(merged_sequence)} total frames'
                )

            prediction = PredictionValue(result=regions)
            logger.debug(f'Prediction with {len(regions)} regions: {prediction.model_dump()}')

            logger.info('='*80)
            logger.info(f'✅ SAM2 TRACKING COMPLETE!')
            logger.info(f'📊 Summary:')
            logger.info(f'   • Objects tracked: {len(regions)}')
            logger.info(f'   • Total frames processed: {frames_to_track}')
            for region in regions:
                obj_id = region['id']
                total_frames = len(region['value']['sequence'])
                logger.info(f'   • Object {obj_id}: {total_frames} frames')
            logger.info('='*80)

            return ModelResponse(predictions=[prediction])
