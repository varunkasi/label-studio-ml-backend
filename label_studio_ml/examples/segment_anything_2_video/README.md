<!--
---
title: SAM2 with Videos
type: guide
tier: all
order: 15
hide_menu: true
hide_frontmatter_title: true
meta_title: Using SAM2 with Label Studio for Video Annotation
categories:
    - Computer Vision
    - Video Annotation
    - Object Detection
    - Segment Anything Model
image: "/tutorials/sam2-video.png"
---
-->

# Using SAM2 with Label Studio for Video Annotation

This guide describes the simplest way to start using **SegmentAnything 2** with Label Studio.

This repository is specifically for working with object tracking in videos. For working with images, 
see the [segment_anything_2_image repository](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image)

![sam2](./Sam2Video.gif)

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart). 

This tutorial uses the [`segment_anything_2_video` example](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_video). 

## Running from source

1. To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
cd label_studio_ml/examples/segment_anything_2_video
pip install -r requirements.txt
```

2. Download [`segment-anything-2` repo](https://github.com/facebookresearch/segment-anything-2) into the root directory. Install SegmentAnything model and download checkpoints using [the official Meta documentation](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#installation). Make sure that you complete the steps for downloadingn the checkpoint files! 

3. Export the following environment variables (fill them in with your credentials!):
- LABEL_STUDIO_URL: the http:// or https:// link to your label studio instance (include the prefix!) 
- LABEL_STUDIO_API_KEY: your api key for label studio, available in your profile. 

4. Then you can start the ML backend on the default port `9090`:

```bash
cd ../
label-studio-ml start ./segment_anything_2_video
```
Note that if you're running in a cloud server, you'll need to run on an exposed port. To change the port, add `-p <port number>` to the end of the start command above.
5. Connect running ML backend server to Label Studio: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL. Read more in the official [Label Studio documentation](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio).
 Again, if you're running in the cloud, you'll need to replace this localhost location with whatever the external ip address is of your container, along with the exposed port.

# Labeling Config
For your project, you can use any labeling config with video properties. Here's a basic one to get you started!

```xml     
<View>
    <Labels name="videoLabels" toName="video" allowEmpty="true">
        <Label value="Player" background="#11A39E"/>
        <Label value="Ball" background="#D4380D"/>
    </Labels>

    <!-- Please specify FPS carefully, it will be used for all project videos -->
    <Video name="video" value="$video" framerate="25.0"/>
    <VideoRectangle name="box" toName="video" smart="true"/>
</View>
```

## CLI Usage for Batch Processing

You can use the CLI to run SAM2 tracking on tasks with existing annotations (keyframes):

1. Start the Docker container:
```bash
docker compose up -d
```

2. Draw bounding boxes (keyframes) in Label Studio for the people/objects you want to track

3. Run tracking via CLI:
```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST=https://app.heartex.com
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/cli.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 198563 \
  --task 227350954 \
  --annotation 12345'
```

### CLI Parameters:
- `--ls-url`: Label Studio URL (e.g., https://app.heartex.com)
- `--ls-api-key`: Your Label Studio API key
- `--project`: Project ID
- `--task`: Task ID to process
- `--annotation`: Annotation ID containing keyframes to track
- `--max-frames`: (Optional) Limit tracking to N frames (default: tracks full video)

### Tracking Strategy:
- **Multiple keyframes (recommended for long videos)**: Draw boxes at key moments (start, turns, occlusions). SAM2 uses them as guidance points for better tracking accuracy.
- **Single keyframe**: Draw one box per person at video start. SAM2 tracks forward from there.
- The model tracks from the first keyframe to the end of the video (or `--max-frames` limit).
- Supports multi-person tracking: annotate multiple people and track them all simultaneously.

## Configuration

### Environment Variables:
- `MAX_FRAMES_TO_TRACK`: Set to `0` for no limit (tracks full video), or a specific number to limit frames. Default: `0`
- `MODEL_CONFIG`: SAM2 model config (default: `configs/sam2.1/sam2.1_hiera_t.yaml`)
- `MODEL_CHECKPOINT`: SAM2 checkpoint (default: `sam2.1_hiera_tiny.pt`)
- `DEVICE`: Computing device (default: `cuda`)

## Known limitations
- As of 8/11/2024, SAM2 only runs on GPU servers.
- Currently, we do not support video segmentation (only bounding boxes).
- For very long videos (40,000+ frames), tracking may take significant time. Consider using `--max-frames` to process in chunks.

If you want to contribute to this repository to help with some of these limitations, you can submit a PR.

## Customization

The ML backend can be customized by adding your own models and logic inside the `./segment_anything_2_video` directory. 
