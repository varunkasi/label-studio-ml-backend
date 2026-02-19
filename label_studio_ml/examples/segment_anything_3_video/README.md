<!--
---
title: SAM3 with Videos
type: guide
tier: all
order: 15
hide_menu: true
hide_frontmatter_title: true
meta_title: Using SAM3 with Label Studio for Video Annotation
categories:
    - Computer Vision
    - Video Annotation
    - Object Detection
    - Segment Anything Model
image: "/tutorials/sam3-video.png"
---
-->

# Using SAM3 with Label Studio for Video Annotation

This guide describes how to use **Segment Anything 3** (SAM3) with Label Studio for video object tracking.

SAM3 is loaded via HuggingFace Transformers (`from_pretrained`), replacing the previous SAM2 standalone repo + custom CUDA ops + Grounding DINO stack. Video decoding uses PyAV (no disk-based frame extraction).

This repository is specifically for working with object tracking in videos. For working with images,
see the [segment_anything_2_image repository](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image).

## What changed from SAM2

| Area | SAM2 | SAM3 |
|------|------|------|
| Model source | Meta repo clone + custom CUDA build | `transformers.from_pretrained()` |
| Object detection | Grounding DINO (separate repo + weights) | `Sam3VideoModel` with text prompts (`HINTS=true`) |
| Instance tracking | `build_sam2_video_predictor` + keypoints hack | `Sam3TrackerVideoModel` with native box prompts |
| Video decoding | OpenCV `cv2.VideoCapture` + JPEG extraction to disk | PyAV in-memory streaming |
| Dockerfile | 104 lines, `devel` base, CUDA compilation | ~49 lines, parameterized `devel` base, multi-arch build args |
| Prompt format | 5 synthetic keypoints from bbox | Native xyxy bounding boxes |
| Processing modes | Single (all frames to disk) | `streaming` (constant memory) or `chunked_batch` (bidirectional context) |

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

This tutorial uses the `segment_anything_3_video` example.

## Running with Docker (recommended)

```bash
cd label_studio_ml/examples/segment_anything_3_video

# Set your credentials
export LABEL_STUDIO_API_KEY="your-api-key"
export HF_TOKEN="your-huggingface-token"  # if model is gated

# Build and start (amd64 / CUDA 12.6 defaults)
docker compose up --build
```

The model weights are downloaded automatically on first startup via `from_pretrained()`.

### Multi-architecture support

The Dockerfile is parameterized for different GPU architectures via build args:

| Build Arg | Default (amd64 / RTX 6000) | DGX Spark (arm64) | Purpose |
|-----------|---------------------------|-------------------|---------|
| `CUDA_VERSION` | `12.6.0` | `13.0.1` | NVIDIA CUDA base image version |
| `UBUNTU_VERSION` | `24.04` | `24.04` | Ubuntu base image version |
| `TORCH_CUDA_INDEX` | `cu126` | `cu130` | PyTorch wheel index suffix |

For DGX Spark (arm64 / CUDA 13.0):

```bash
cp .env.dgx-spark .env
docker compose up --build -d
```

Or inline: `CUDA_VERSION=13.0.1 TORCH_CUDA_INDEX=cu130 docker compose up --build -d`

## Running from source

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
cd label_studio_ml/examples/segment_anything_3_video

# Install PyTorch with the correct CUDA index for your GPU:
#   cu126 for CUDA 12.6 (e.g., RTX 6000 Ada)
#   cu130 for CUDA 13.0 (e.g., DGX Spark)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies (torch/torchvision are NOT in requirements.txt)
pip install -r requirements.txt
```

No separate model repo or checkpoint download is required. Weights are fetched automatically by HuggingFace Transformers on first import.

2. Export environment variables:

```bash
export LABEL_STUDIO_URL="https://your-label-studio-instance.com"
export LABEL_STUDIO_API_KEY="your-api-key"
export MODEL_NAME="facebook/sam3"
export HINTS=false           # or true for text-based detection
export PROCESSING_MODE=streaming  # or chunked_batch
```

3. Start the ML backend:

```bash
cd ../
label-studio-ml start ./segment_anything_3_video
```

4. Connect the running ML backend to Label Studio: go to your project **Settings > Machine Learning > Add Model** and specify `http://localhost:9090` as the URL.

## Labeling Config

For your project, you can use any labeling config with video properties. Here's a basic one to get you started:

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

## Model Variants

### Tracker mode (`HINTS=false`, default)

Uses `Sam3TrackerVideoModel` + `Sam3TrackerVideoProcessor`. Requires user-drawn bounding boxes in Label Studio as tracking prompts. Best for interactive annotation where you draw a box on the first frame and the model tracks it forward.

### PCS / hints mode (`HINTS=true`)

Uses `Sam3VideoModel` + `Sam3VideoProcessor`. Replaces Grounding DINO with SAM3's built-in text-based detection. Set `PROMPT_TEXT` to specify what to detect (e.g., `person`, `car`). No drawn bounding boxes required.

## Processing Modes

### Streaming (`PROCESSING_MODE=streaming`, default)

Decodes frames one at a time via PyAV. Constant memory usage regardless of video length. Best for long videos and production use.

### Chunked batch (`PROCESSING_MODE=chunked_batch`)

Decodes all frames in `[start_frame, end_frame]` into memory at once. Provides bidirectional temporal context for better tracking quality. Use for shorter clips or when you have ample GPU memory.

## Interview UI (Active Learning)

The Interview UI is a browser-based active learning workflow for generating seed annotations. It runs as a Flask Blueprint at `/interview/` alongside the ML backend.

**Access:** After `docker compose up`, open `http://<host>:9090/interview/`

### Workflow Phases

| Phase | What happens | User action |
|-------|-------------|-------------|
| **Init** | Create session with project/task IDs | Enter LS project + task ID |
| **Detection** | Stage 1: SAM3 text detection on ~40 uniform keyframes (~30-60s). Stage 2: FPS-capped background embeddings with incremental change detection. Round 2+ samples from change-detected keyframes | Label crops: Accept (good box) / Reject (bad box or not target) / Skip (ambiguous). For rejects, run subcategory review (`not_person`, `partial_box`, `oversized_box`) and optionally add corrected boxes |
| **Quality Scoring (k-NN)** | The "Next Round" action re-scores pending crops using DINOv3 + distance-weighted k-NN over all labeled support crops, then runs detection on newly selected frames. Click "Finish Labeling → ReID" after round 1+ to proceed | Continue active learning rounds until coverage/quality are sufficient |
| **ReID** | Three-phase centroid-growing pipeline: **Phase 1** (centroid building) — human judges crop pairs to accumulate must-links; confirmed "same" crops are averaged into strong cluster centroids. **Phase 2** (ambiguous resolution) — unassigned crops are compared against Phase 1 centroids; decisive matches auto-assign, ambiguous cases go to human. **Phase 3** (done) — summary view with expandable cluster thumbnails. Constraint-based clustering (COP-KMeans) respects must-links and cannot-links across re-clusters | Judge pairs: same person / different / unsure |
| **Seeding** | Three-path dual-proposer pipeline generates dense seeds across all frames, filtered by the same k-NN quality gate used during detection rounds. Guards against missing ReID clusters. Upload to Label Studio | Configure frame interval + confidence threshold, review + upload |

### Session modes

`session/init` returns cache options and `session/resume` applies one of these modes:

- **Resume:** Load the exact saved session state and phase for the same task.
- **Build on:** Re-open cached state for the same task and continue in Detection.
- **Build from (`use_from_<task_id>`):** Start a new task in Detection Round 1 (`current_round=0`, empty round history) while transferring labeled support crops/features from another task cache.
- **Fresh:** Start from an empty session for the task.

After upload, seeds are created with `enabled=false` keyframes in Label Studio. Run the tracking CLI to fill gaps:

```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes_manual_merge.py \
  --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <ID> --task <ID> --annotation <ID> --max-frames-to-track 300
```

### Architecture

- **Backend:** Flask Blueprint (`interview/routes.py`) with background job executor for long-running operations
- **Frontend:** Vanilla JS SPA with hash-based routing and clickable phase navigation, no framework dependencies
- **Models and scoring used:** SAM3 (text-based detection + box-prompted refinement), DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m` for feature extraction + 3-scale grid search), distance-weighted k-NN quality gate (full-support scoring over labeled crops)
- **State:** In-memory sessions with optional disk persistence to `/data/adapters/`

### Detection Pipeline

Detection runs in two decoupled stages so the user can start labeling immediately:

1. **Stage 1 (fast, ~30-60s):** Selects ~40 uniformly-spaced keyframes, batch-decodes them via PyAV, runs SAM3 text detection in GPU batches, stores crops on the session. User sees crops and can start labeling.
2. **Stage 2 (background, ~3-5 min):** GPU-batched SAM3 frame embeddings with prefetch threading. Computes temporal change scores and selects change-detected keyframes for active learning rounds. A progress banner shows completion in the UI. Decoded frames are saved to the disk frame cache for reuse by later phases.

### 3-Tier Frame Cache

Frame access uses a three-tier cache hierarchy to avoid redundant video decoding:

| Tier | Location | Speed | Persistence |
|------|----------|-------|-------------|
| **1. LRU memory** (`frame_cache.py`) | OrderedDict, ~64 entries | Instant | Per-process (lost on restart) |
| **2. Disk JPEG** (`disk_frame_cache.py`) | `/data/adapters/{cache_key}/frames/` | ~ms seek | Survives restarts (volume-mounted) |
| **3. PyAV seek** (`seeding_common._read_frame_pyav`) | Video container seek | ~10-50ms | N/A (source video) |

During background embedding (Stage 2), frames are decoded once and written to tier 2. All subsequent reads (change detection, seeding, UI crop display) go through `read_frame_cached()` which checks tiers 1→2→3 in order. Disk frame cache is ~6-10 GB per video at 10 FPS.

### Seeding Pipeline (Dual-Proposer)

Dense seeding uses three paths for maximum coverage (~80-90% of frames):

| Path | Source | Flow |
|------|--------|------|
| **A** (primary) | SAM3 text detection | Detect on frame → NMS → crop → DINOv3 features → k-NN quality gate → seed |
| **B** (refinement) | Sam3Model box+text | Medium-confidence Path A boxes → expand 20% → Sam3Model with text+box prompts → tight box from mask → k-NN gate → seed |
| **C** (grid search) | DINOv3 similarity | Zero-detection frames → tile into grid cells → DINOv3 features per cell → cosine similarity to accepted crops → top-K cells → Sam3Model refinement → k-NN gate → seed |

Frames are processed in chunks of 100 (configurable) to bound memory. Each seed includes a `"source"` field (`"path_a"`, `"path_b"`, `"path_c"`) for provenance tracking.

### Quality gate labels: Accept / Reject / Skip

The k-NN scorer acts as a **quality gate** during active learning and dense seeding. It does NOT propose boxes — SAM3 does. k-NN only scores SAM3 proposals using labeled support crops (accepted/rejected/corrected). Labels encode **box quality**, not just "is there a person?"

| Label | Meaning | k-NN effect |
|-------|---------|-------------|
| **Accept** | Good-quality, tight bounding box relative to what's visible | "Pass this box through during seeding" |
| **Reject** | Not a person, OR person is fully visible but box is partial/sloppy | "Block this box during seeding" |
| **Skip** | Too ambiguous to judge; excluded from training entirely | (nothing — ignored) |

Rejects support subcategories for boundary-aware weighting:

- **`not_person`**: hard reject
- **`partial_box`**: softer reject
- **`oversized_box`**: softer reject

When a reject is tagged `partial_box`/`oversized_box`, reviewers may optionally draw a corrected box. Corrected boxes are added immediately as accepted support examples and tracked separately in stats (`corrected_total`). At most one corrected crop exists per rejected crop (upsert semantics). Re-visiting a previously corrected crop loads the corrected box into the adjuster. Navigation auto-saves adjusted boxes and skips the API call entirely when nothing changed (dirty-tracking).

**Critical distinction:** A person partially visible in the frame (walking out of frame edge) with a tight box around whatever IS visible is **Accept**. A person fully visible but only partially boxed is **Reject**. The judgment is always relative to what's visible.

### ReID: Centroid-Growing Pipeline

Re-identification uses a three-phase elicitation strategy designed to build strong identity centroids with minimal human effort:

| Phase | Goal | Elicitation strategy | Outcome |
|-------|------|---------------------|---------|
| **1. Centroid building** | Accumulate must-links per cluster | Pairs chosen to maximize "same" confirmations across diverse clusters early | Confirmed crops averaged into reliable centroids |
| **2. Ambiguous resolution** | Assign remaining crops to centroids | Compare unassigned crops against Phase 1 centroids; decisive matches auto-assign, ambiguous cases shown to human | All crops either assigned or flagged as new identity |
| **3. Done** | Summary view | Expandable cluster thumbnails with "+N more" | Final identity clusters ready for seeding |

**Constraints**: Human "same" verdicts create must-links, "different" create cannot-links. These persist across re-clusters (stored in `session.reid_must_links` / `reid_cannot_links`). COP-KMeans respects constraints when re-clustering. Centroid computation only uses confirmed (must-linked) members — clusters with no confirmed members are skipped during Phase 2 assignment.

**Keyboard shortcuts**: `1` = same, `2` = different, `3` = unsure (configurable).

### Backend Modules

| Module | Purpose |
|--------|---------|
| `interview/routes.py` | REST endpoints, SPA serving, background-job orchestration for scoring + detection rounds |
| `interview/state.py` | Session state management (phases, crops, clusters, embedding indices) |
| `interview/background.py` | Thread-based job executor with progress polling, pause/resume via `threading.Event` |
| `interview/cache_manager.py` | Disk persistence for sessions, features, models, embedding indices |
| `interview/detection.py` | Two-stage detection pipeline: batch SAM3 inference + FPS-capped background embeddings with incremental change detection |
| `interview/dinov3_classifier.py` | DINOv3 feature extraction (crop/context features, metadata, mask-quality support features) |
| `interview/knn_classifier.py` | Distance-weighted full-support k-NN scorer with reject subcategory and corrected/reject pair weighting |
| `interview/mask_utils.py` | Mask-quality feature computation (fill ratio, edge contact, compactness) |
| `interview/frame_cache.py` | Shared LRU frame cache (in-memory, ~64 entries) — tier 1 of 3-tier cache hierarchy |
| `interview/disk_frame_cache.py` | Disk-based JPEG frame cache under `/data/adapters/{cache_key}/frames/` — tier 2 of 3-tier cache; populated during background embedding, reused by change detection, seeding, and UI |
| `interview/reid_ufm.py` | UFM (Universal Feature Model) pairwise similarity computation for ReID |
| `interview/ufm_model.py` | UFM model definition and inference |
| `interview/seeding_phase.py` | Three-path dual-proposer dense seeding + Label Studio upload |

### Interview Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INTERVIEW_INITIAL_KEYFRAMES` | `40` | Number of uniformly-sampled keyframes for Stage 1 detection |
| `INTERVIEW_REFINE_THRESHOLD` | `0.3` | Path A score below which Path B refinement activates |
| `INTERVIEW_ENABLE_REFINEMENT` | `true` | Enable/disable Path B (Sam3Model box refinement) |
| `INTERVIEW_ENABLE_GRID_SEARCH` | `true` | Enable/disable Path C (DINOv3 grid search) |
| `INTERVIEW_GRID_SCALE` | `0.10` | Grid cell size as fraction of frame dimensions |
| `INTERVIEW_GRID_SIM_THRESHOLD` | `0.5` | Cosine similarity threshold for grid cell match |
| `INTERVIEW_GRID_TOP_K` | `5` | Max grid cells to refine per frame |
| `INTERVIEW_SEED_CHUNK_SIZE` | `100` | Frames per processing chunk in seeding |
| `INTERVIEW_EMBEDDING_FPS` | `10` | FPS cap for background embedding (subsamples to this rate) |
| `INTERVIEW_EMBEDDING_BATCH` | `64` | Batch size for SAM3 frame embedding |
| `INTERVIEW_DETECT_BATCH` | `8` | Batch size for SAM3 text detection |
| `INTERVIEW_KNN_THRESHOLD` | `0.6` | Confidence threshold for k-NN quality gating decisions |
| `INTERVIEW_FRAME_CACHE_SIZE` | `64` | LRU frame cache entries (~6 MB each) |
| `INTERVIEW_FRAMES_PER_ROUND` | `40` | Frames selected per active learning round |
| `INTERVIEW_VALIDATION_FRAMES` | `20` | Held-out frames used for round-level validation tracking |
| `INTERVIEW_CACHE_ROOT` | `/data/adapters` | Root directory for disk persistence (sessions, frames, models) |
| `INTERVIEW_EMBEDDING_MODE` | `lightweight` | Embedding strategy: `lightweight` (change-detection) or `full` |

## Configuration

### Core Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Computing device (`cuda` or `cpu`) |
| `MODEL_NAME` | `facebook/sam3` | HuggingFace model identifier |
| `HINTS` | `false` | `false` = Tracker (box prompts), `true` = PCS (text detection) |
| `PROCESSING_MODE` | `streaming` | `streaming` or `chunked_batch` |
| `PROMPT_TEXT` | `person` | Text prompt for PCS mode (`HINTS=true`) |
| `HF_TOKEN` | (empty) | HuggingFace token for gated models |
| `LABEL_STUDIO_HOST` | | URL of your Label Studio instance |
| `LABEL_STUDIO_API_KEY` | | Your Label Studio API key |

### Tracking Performance & Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FRAMES_TO_TRACK` | `0` | Max frames to track from first keyframe. `0` = unlimited |
| `TRACK_FPS` | `0` | Temporal downsampling target FPS. `0` = use original FPS |

- **Example**: Set `TRACK_FPS=2` to track only 2 frames per second, reducing processing time for high-FPS videos.
- **Warning**: In `chunked_batch` mode, high frame counts require significant RAM/VRAM since all frames are held in memory.

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `9090` | Server listen port |
| `WORKERS` | `1` | Gunicorn worker processes |
| `THREADS` | `4` | Threads per worker |
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |

- **Docker `shm_size`**: Set `shm_size: '32g'` or higher in `docker-compose.yml` for GPU operations.
- **Docker `mem_limit`**: Recommended `48g` to avoid system instability.

## CLI Tools

This directory contains several CLI tools for video tracking automation. All commands are run inside the Docker container via `docker compose exec`.

> **Migration status**: All CLI tools have been **migrated to SAM3**. They use the same HuggingFace Transformers-based SAM3 models as the ML backend. Video decoding uses PyAV (no cv2/OpenCV dependency). All commands should be run in the `segment_anything_3_video` container.

### 1. Automatic Object Detection & Tracking (`initial_seeding_video.py`)

**Status**: Migrated to SAM3

**Use case:** You have a raw video and want to automatically find and track all objects of a certain class (e.g., "person", "car").
**Input:** Raw video task. No existing annotations or bounding boxes are required.
**Method:** Uses SAM3 text-based detection (`Sam3VideoModel`) to find objects at keyframes, then `Sam3TrackerVideoModel` to track them, and stitches the results into complete tracks. Seed boxes are refined using text+box prompts before tracking (configurable).

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --prompt "person" --keyframe-frac 0.1
```

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--prompt` | No | `None` | Text prompt for detection (e.g., "person", "car") |
| `--keyframe-frac` | No | `0.1` | Fraction of frames to use as keyframes (0.1 = 10%) |
| `--min-spacing` | No | `30` | Minimum spacing between keyframes |
| `--embedding-batch` | No | `8` | Batch size for SAM3 embedding computation |
| `--cache-dir` | No | `./cache_dir/joblib` | Cache directory for embeddings |
| `--stitch-mode` | No | `teacher` | Stitching mode: `teacher` (embedding similarity) or `hungarian` (IoU + distance) |
| `--merge-threshold` | No | `0.6` | Cosine similarity threshold for teacher stitching (higher = stricter) |
| `--sparse-sequence` | No | `None` | Enable sparse sequence generation |
| `--no-sparse-sequence` | No | — | Disable sparse sequence generation |
| `--no-refine-seeds` | No | — | Disable seed box refinement (enabled by default) |
| `--refine-search-scale` | No | `1.3` | Search scale for refinement (1.3 = 30% expansion) |
| `--dry-run` | No | `False` | Save prediction to JSON file instead of uploading |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 2. Track Existing Manual Keyframes (`initial_seeding_video_boxes.py`)

**Status**: Migrated to SAM3

**Use case:** You have already drawn some bounding boxes (keyframes) in Label Studio and want SAM3 to track them forward and backward to fill the gaps.
**Input:** Video task with at least a few manual bounding boxes (keyframes).
**Method:** Uses your existing boxes as anchors, generates bidirectional tracklets using `Sam3TrackerVideoModel`, and uses the Hungarian algorithm to stitch them robustly. Seed boxes are refined using text+box prompts before tracking (configurable).

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --global-start 0 --global-end 2000 --max-frames-to-track 300
```

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--prompt` | No | `None` | Label override for all tracked objects |
| `--global-start` | No | `0` | Starting frame index (0-based inclusive) |
| `--global-end` | No | last frame | Ending frame index (0-based inclusive) |
| `--max-frames-to-track` | No | `300` | Max frames to track in each direction from keyframe |
| `--frame-stride` | No | `1` | Sample every N frames (1 = no downsampling) |
| `--score-threshold` | No | `0.1` | Minimum `object_score_logits` sigmoid score; 3 consecutive frames below this terminates tracking |
| `--enable-oracle` | No | `False` | Run Sam3VideoModel text detection per window to cross-check tracker output |
| `--oracle-stride` | No | `30` | Check every N-th frame with the oracle (lower = more thorough, slower) |
| `--no-refine-seeds` | No | — | Disable seed box refinement (enabled by default) |
| `--refine-search-scale` | No | `1.3` | Search scale for refinement (1.3 = 30% expansion) |
| `--dry-run` | No | `False` | Print prediction JSON instead of uploading |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 2a. Manual Merge Tracking (`initial_seeding_video_boxes_manual_merge.py`)

**Status**: Migrated to SAM3

**Use case:** You want bidirectional tracking per seed box, but you prefer to merge tracks manually using `mergevideoregions.py` and meta text IDs.
**Input:** Video task with manual keyframes. Optional track-id filtering lets you target specific seed regions per iteration.
**Method:** Builds forward+backward tracks per seed using `Sam3TrackerVideoModel` and keeps them in the same region; no automatic cross-seed merging. Each output region gets `meta.text="id:"` to ease manual ID assignment.

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes_manual_merge.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 225664 --task 245750672 --annotation 85260070 --global-start 1000 --global-end 1800 --max-frames-to-track 300
```

**With track filtering:**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes_manual_merge.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --track-id auto-track-14,auto-track-15 --global-start 1000 --global-end 1800
```

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--prompt` | No | `None` | Label override for all tracked objects |
| `--track-id` | No | `None` | Comma-separated region IDs to use as anchors (omit for all) |
| `--global-start` | No | `None` | Starting frame index (0-based inclusive) |
| `--global-end` | No | `None` | Ending frame index (0-based inclusive) |
| `--max-frames-to-track` | No | `300` | Max frames to track in each direction from keyframe |
| `--frame-stride` | No | `1` | Sample every N frames (1 = no downsampling) |
| `--overlap-mode` | No | `iou-weighted` | Overlap resolution: `iou-weighted`, `weighted`, `winner` |
| `--overlap-iou-threshold` | No | `0.3` | IoU threshold for iou-weighted mode |
| `--score-threshold` | No | `0.1` | Minimum `object_score_logits` sigmoid score; 3 consecutive frames below this terminates tracking |
| `--enable-oracle` | No | `False` | Run Sam3VideoModel text detection per window to cross-check tracker output |
| `--oracle-stride` | No | `30` | Check every N-th frame with the oracle (lower = more thorough, slower) |
| `--forward-only` | No | `False` | Skip backward tracking (forward pass only) |
| `--streaming` | No | `False` | Streaming mode: constant-memory forward tracking (implies `--forward-only`) |
| `--streaming-chunk-size` | No | `2000` | Max frames per SAM3 session in streaming mode before GPU memory reset |
| `--no-refine-seeds` | No | — | Disable seed box refinement (enabled by default) |
| `--refine-search-scale` | No | `1.3` | Search scale for refinement (1.3 = 30% expansion) |
| `--dump-payload` | No | `None` | Path to write submission payload JSON before upload |
| `--no-progress` | No | `False` | Disable progress bars |
| `--dry-run` | No | `False` | Print prediction JSON instead of uploading |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

**Performance:** Seeds at the same keyframe are tracked in a single batched session (multi-object `obj_ids`), so the vision encoder runs once per direction per keyframe window instead of once per seed. This gives ~Nx speedup where N is the average seeds per keyframe.

**Submission behavior:** If `--track-id`, `--global-start`, and `--global-end` are all provided, the script patches the existing annotation; otherwise it creates a new prediction.

#### Streaming mode

Use `--streaming` for long videos (thousands of frames) where loading all frames into memory would OOM. Streaming mode:

- Processes frames one at a time via PyAV (constant memory)
- Groups all keyframes for the same `--track-id` region into a single pass
- First keyframe = seed, all subsequent keyframes = **drift-correction anchors** injected via `add_inputs_to_inference_session` as the tracker reaches each frame
- GPU temporal memory is bounded by `--streaming-chunk-size`: after that many frames, the SAM3 session is destroyed and re-created with the last tracked box

**Drift correction:** When a region has multiple keyframes (e.g., human annotations at frames 100, 500, 1200), streaming mode uses the first as the seed and injects the rest as corrections. At each correction frame, the tracker re-anchors to the human box, preventing accumulated drift over long ranges.

**GPU memory per chunk** (measured with SAM3 `facebook/sam3`):
- Model weights: ~2.6 GB (fixed)
- Per-frame temporal memory: ~8.7 MB (linear growth)
- Formula: `total ≈ 2.6 + (chunk_size × 0.0087)` GB

| GPU | VRAM | Safe `--streaming-chunk-size` |
|-----|------|-------------------------------|
| RTX 6000 Ada | 49 GB | 5000 |
| A100 (40 GB) | 40 GB | 4000 |
| A100 / H100 (80 GB) | 80 GB | 8000 |

**Example (long-range streaming with stride):**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes_manual_merge.py \
  --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 123 --task 456 --annotation 789 \
  --track-id=4NeFy4BYus --global-start 16650 --global-end 23639 \
  --streaming --streaming-chunk-size 5000 --frame-stride 3
```

This tracks every 3rd frame (plus all correction keyframes) with 5000-frame GPU memory chunks.

### 3. Simple Forward Tracking (`cli.py`)

**Status**: Migrated to SAM3

**Use case:** Simple "predict" functionality similar to the UI Submit/Update button. Tracks from start to finish linearly.
**Input:** Video task with at least one manual keyframe (bounding box) to start tracking from.
**Method:** Uses SAM3 via model.py to load the entire video and propagate all keyframes at once. Best for short clips.

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID with keyframes |
| `--device` | No | env `DEVICE` or `cuda` | Compute device (`cuda` or `cpu`) |
| `--hints` | No | env `HINTS` or `false` | `true` = Sam3VideoModel (text detection), `false` = Sam3TrackerVideoModel (box tracking) |
| `--model-name` | No | env `MODEL_NAME` or `facebook/sam3` | HuggingFace model ID |
| `--processing-mode` | No | env `PROCESSING_MODE` or `streaming` | `streaming` (constant memory) or `chunked_batch` (all frames in memory) |
| `--track-fps` | No | env `TRACK_FPS` or `0` | Target FPS for temporal downsampling (`0` = no downsampling) |
| `--prompt-text` | No | env `PROMPT_TEXT` or `person` | Text prompt for detection (only used when `--hints true`) |
| `--max-frames` | No | env `MAX_FRAMES_TO_TRACK` or `0` | Max frames to track (`0` = unlimited) |
| `--log-level` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

> **Override semantics:** Optional flags only override the corresponding environment variable when explicitly provided. If omitted, the value from docker-compose.yml (or the system environment) is preserved. This lets you set defaults in docker-compose.yml and override per-run from the CLI.

**Examples:**

```bash
# Basic tracking (uses env defaults from docker-compose.yml)
docker compose exec segment_anything_3_video python /app/cli.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789

# Limit to 500 frames with debug logging
docker compose exec segment_anything_3_video python /app/cli.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --max-frames 500 --log-level DEBUG

# Text-detection mode (Sam3VideoModel, no manual boxes required)
docker compose exec segment_anything_3_video python /app/cli.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --hints true --prompt-text "person"

# Chunked batch mode with temporal downsampling to 5 FPS
docker compose exec segment_anything_3_video python /app/cli.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --processing-mode chunked_batch --track-fps 5

# Run on CPU (e.g. for testing)
docker compose exec segment_anything_3_video python /app/cli.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --device cpu --max-frames 50
```

### 4. Post-Processing Tools (`video_tools.py`)

**Status**: No model dependency -- works in either container.

**Use case:** Clean up tracking results (sparsify dense frames, swap IDs, smooth jitter).

**Sparsify (Downsample Keyframes)**

Reduce the density of keyframes (e.g., keep 10% of frames) to make manual editing easier.

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py sparsify --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --track-id auto-track-0 --start-frame 1000 --end-frame 2000 --ratio 0.1
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--track-id` | Yes | — | Region track ID (e.g., auto-track-0) |
| `--start-frame` | Yes | — | Start frame (1-based) |
| `--end-frame` | Yes | — | End frame (1-based) |
| `--ratio` | Yes | — | Fraction of frames to keep [0,1]. Use `0` to remove ALL keyframes in range. |

**Swap IDs (Fix Identity Switches)**

Move a segment of tracking history from one object ID to another (e.g., if the tracker swapped "Person A" to "Person B").

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py swap-ids --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --source-track-id auto-track-0 --target-track-id auto-track-1 --start-frame 500 --end-frame 600
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--source-track-id` | Yes | — | Source region track ID |
| `--target-track-id` | Yes | — | Target region track ID |
| `--start-frame` | Yes | — | Start frame (1-based) |
| `--end-frame` | Yes | — | End frame (1-based) |

**Trim Tail (Delete Trailing Frames)**

Delete all keyframes for a specific track after a certain cutoff frame (e.g., when an object leaves the view).

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py trim-tail --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --track-id auto-track-0 --cutoff-frame 1500
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--track-id` | Yes | — | Region track ID |
| `--cutoff-frame` | Yes | — | Delete all frames after this (1-based) |

**Smooth (Stabilize Jitter)**

Apply a moving average filter to smooth out shaky bounding boxes.

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py smooth --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --track-id auto-track-0 --window 5
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--track-id` | Yes | — | Region track ID |
| `--window` | No | `5` | Moving average window size |

**Pad (Expand Bounding Boxes)**

Inflate bounding boxes by a percentage (e.g., 10%) over a specific frame range. Useful if the tracker is consistently too tight.

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py pad --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --track-id auto-track-0 --percent 0.10 --start-frame 0 --end-frame 1000
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--track-id` | Yes | — | Region track ID |
| `--percent` | Yes | — | Expansion percentage (e.g., 0.10 = 10%) |
| `--start-frame` | Yes | — | Start frame (1-based) |
| `--end-frame` | Yes | — | End frame (1-based) |

**Common arguments for all video_tools.py commands:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | No | env `LABEL_STUDIO_URL` | Label Studio URL |
| `--ls-api-key` | No | env `LABEL_STUDIO_API_KEY` | Label Studio API key |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--dry-run` | No | `False` | Write updated JSON to file instead of PATCH |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 5. Prediction Validation (`validate_prediction.py`)

**Status**: No model dependency -- works in either container.

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/validate_prediction.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --prediction-file prediction.json --upload
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--task` | Yes | — | Task ID |
| `--prediction-file` | Yes | — | Path to prediction JSON file |
| `--upload` | No | `False` | Upload prediction to Label Studio |

### 6. Export Utilities (`export_interpolated_annotation.py`)

**Status**: No model dependency -- works in either container.

**Use case:** Download a single annotation JSON with *all* interpolated video frames included (not just keyframes). This is critical for getting the full frame-by-frame tracking data out of Label Studio.

**Example (Python):**
```bash
docker compose exec segment_anything_3_video python /app/export_interpolated_annotation.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --output-dir /app/exports
```

**Example (Bash script):**
```bash
docker compose exec segment_anything_3_video /app/export_interpolated_annotation.sh --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --output /app/exports/annotation.json
```

**Summary output:** The bash script also writes a per-casualty summary JSON next to the exported annotation. If the output is `annotation.json`, the summary is written to `annotation.summary.json`. The summary includes frame/time ranges per `meta.text` ID.

### 6a. Generate Casualty Snippets (integrated into `export_interpolated_annotation.sh`)

**Use case:** Export the interpolated annotation, create summary JSON, and generate per-casualty snippets plus per-snippet bbox JSON outputs in one step.

**Example:**
```bash
docker compose exec segment_anything_3_video /app/export_interpolated_annotation.sh --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --output /app/exports/annotation.json --snippets --person-id 31 --min-seconds 2 --fps 10
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--snippets` | No | `False` | Enable snippet generation |
| `--snippets-dir` | No | auto-generated | Output directory for snippets |
| `--person-id` | No | all | Generate snippets for specific person ID only |
| `--min-frames` | No | — | Skip ranges shorter than N frames (mutually exclusive with `--min-seconds`) |
| `--min-seconds` | No | — | Skip ranges shorter than N seconds (mutually exclusive with `--min-frames`) |
| `--fps` | No | original | Output FPS (omit to use original with stream-copy) |

**Output files:**
- `casualty_<id>_f<start>-<end>_fps<fpsInt>.mp4`
- `casualty_<id>_f<start>-<end>_fps<fpsInt>.json` with frame-level bbox entries
- `README.txt` in the output folder capturing parameters used

### 6b. Unified Annotation Pipeline (`process_annotation.py`)

**Status**: Optional SAM3 dependency (skippable with `--skip-masks`).

**Use case:** One-command pipeline from Label Studio annotation to all artifacts: interpolated export, per-casualty video snippets, bbox JSONs, SAM3 mask PNGs, and overlay videos. Replaces the old `overlay_snippet_bboxes.sh` script and combines the functionality of `export_interpolated_annotation.sh` + `extract_snippet_masks.py` into a single Python entry point.

**Example (full pipeline with SAM3):**
```bash
docker compose exec segment_anything_3_video python /app/process_annotation.py \
  --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 123 --task 456 --annotation 789
```

**Example (no GPU — snippets + bbox only):**
```bash
python process_annotation.py \
  --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 123 --task 456 --annotation 789 --skip-masks
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes* | env `LABEL_STUDIO_URL` / `LABEL_STUDIO_HOST` | Label Studio URL |
| `--ls-api-key` | Yes* | env `LABEL_STUDIO_API_KEY` | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--snippets-dir` | No | auto-generated | Output directory |
| `--person-id` | No | all | Filter to a single casualty ID |
| `--min-frames` | No | — | Skip ranges shorter than N frames |
| `--min-seconds` | No | — | Skip ranges shorter than N seconds |
| `--fps` | No | source FPS | Target FPS (stream-copy when same as source) |
| `--skip-snippets` | No | `False` | Export + summary only, no video cutting |
| `--skip-masks` | No | `False` | Skip SAM3 masks (no GPU required) |
| `--skip-mask-video` | No | `False` | Skip grayscale mask MP4 |
| `--skip-overlay-video` | No | `False` | Skip green overlay MP4 |
| `--skip-bbox-video` | No | `False` | Skip red bbox overlay MP4 |
| `--chunk-size` | No | `1000` | ffmpeg drawbox chunk size |
| `--host-uid` / `--host-gid` | No | `1000` | Docker file ownership fix |
| `--poll-interval` | No | `5` | Export poll interval (sec) |
| `--timeout` | No | `300` | Export poll timeout (sec) |
| `--log-level` | No | `INFO` | Logging level |

**Output directory tree:**
```
snippets_proj{P}_task{T}_ann{A}_{timestamp}/
├── README.txt
├── project{P}_task{T}_ann{A}.json              # interpolated annotation
├── project{P}_task{T}_ann{A}.summary.json      # casualty frame ranges
├── casualty_{id}_f{start}-{end}_fps{N}.mp4     # snippet video
├── casualty_{id}_f{start}-{end}_fps{N}_frame_bbox.json  # per-frame bboxes
├── casualty_{id}_f{start}-{end}_fps{N}_masks/  # SAM3 mask PNGs (if not --skip-masks)
│   ├── mask_000001.png … mask_NNNNNN.png
│   └── scores.json
├── casualty_{id}_…_masks.mp4                   # grayscale mask video
├── casualty_{id}_…_overlaid_masks.mp4          # green overlay video
└── casualty_{id}_…_bbox_overlaid.mp4           # red bbox overlay video
```

### 7. Deletion Utilities (`delete_annotation_or_prediction.py`)

**Status**: No model dependency -- works in either container.

**Delete an annotation:**
```bash
docker compose exec segment_anything_3_video python /app/delete_annotation_or_prediction.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789
```

**Delete a prediction:**
```bash
docker compose exec segment_anything_3_video python /app/delete_annotation_or_prediction.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --prediction 555
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Either | — | Annotation ID to delete (mutually exclusive with `--prediction`) |
| `--prediction` | Either | — | Prediction ID to delete (mutually exclusive with `--annotation`) |

### 8. Merge Video Regions (`mergevideoregions.py`)

**Status**: No model dependency -- works in either container.

**Use case:** Consolidate fragmented tracks that share the same text ID (e.g., `id:31` in `meta.text`) into single continuous track objects. Useful after manual labeling or ReID where multiple regions represent the same object.

**Example (from annotation):**
```bash
docker compose exec segment_anything_3_video python /app/mergevideoregions.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789
```

**Example (from prediction):**
```bash
docker compose exec segment_anything_3_video python /app/mergevideoregions.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --prediction 555
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Either | — | Annotation ID as source (mutually exclusive with `--prediction`) |
| `--prediction` | Either | — | Prediction ID as source (mutually exclusive with `--annotation`) |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 9. Bounding Box Refinement (`adjust_bboxes_sam3.py`)

**Status**: Migrated to SAM3

**Use case:** Tighten or adjust existing bounding boxes that have drifted due to tracker instability (camera movement, zoom, person motion). Works for boxes that are too large OR too small.

**Method:** Uses SAM3's combined text+box prompt capability:
- The **text prompt** (from track label or `--default-label`) tells SAM3 WHAT to segment (e.g., "person")
- The **expanded box prompt** tells SAM3 WHERE to look

This approach handles bidirectional drift - even if the original box doesn't fully contain the target, the expanded search region should, and the text prompt ensures SAM3 finds the right object.

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/adjust_bboxes_sam3.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --search-scale 1.3 --default-label person
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID whose bboxes will be refined |
| `--search-scale` | No | `1.3` | Search region expansion (1.3 = 30% larger). Increase for more drift tolerance |
| `--default-label` | No | `person` | Text prompt when track has no label |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 10. Re-Identification (`complete_reid.py`)

**Status**: Migrated to SAM3

**Use case:** Automatically suggest identity matches for broken tracks. Uses appearance features (color, geometry, or SAM3 embeddings) to find likely matches between "candidate" tracks (no ID) and "reference" tracks (confirmed ID).

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/complete_reid.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --profile uav --feature-backend sam3
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID as source of tracks |
| `--profile` | No | `uav` | Feature weighting preset: `uav`, `ugv` |
| `--feature-backend` | No | `classic` | `classic` (color/geometry via numpy/scipy) or `sam3` (neural embeddings) |
| `--sam3-padding-fraction` | No | `0.1` | Padding around boxes for SAM3 embedding extraction |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

## CLI Migration Status Summary

| Script | Status | Notes |
|--------|--------|-------|
| `model.py` (ML backend) | **Migrated** | SAM3 via HuggingFace Transformers |
| `cli.py` | **Migrated** | Uses SAM3 via model.py |
| `initial_seeding_video.py` | **Migrated** | Sam3VideoModel (text detection) + Sam3TrackerVideoModel |
| `initial_seeding_video_boxes.py` | **Migrated** | Sam3TrackerVideoModel with bidirectional tracking |
| `initial_seeding_video_boxes_manual_merge.py` | **Migrated** | Sam3TrackerVideoModel, no cross-seed merging |
| `seeding_common.py` | **Migrated** | Lazy-loaded SAM3 singletons, PyAV video I/O |
| `adjust_bboxes_sam3.py` | **Migrated** | Sam3Model for box-prompted segmentation |
| `complete_reid.py` | **Migrated** | SAM3 embeddings backend, numpy/scipy for classic features |
| `video_tools.py` | No model dependency | Post-processing utilities |
| `export_interpolated_annotation.py` | No model dependency | Export with interpolation |
| `export_interpolated_annotation.sh` | No model dependency | Bash export + snippets (lightweight, no Python) |
| `process_annotation.py` | Optional SAM3 | Unified pipeline: export → snippets → SAM3 masks → overlay videos |
| `extract_snippet_masks.py` | SAM3 required | SAM3 mask extraction + ffmpeg encoding (used by `process_annotation.py`) |
| `validate_prediction.py` | No model dependency | Prediction validation |
| `delete_annotation_or_prediction.py` | No model dependency | Deletion utility |
| `mergevideoregions.py` | No model dependency | Track merging |
| `update_video_paths.py` | No model dependency | Path updates |

All CLI tools now run in the `segment_anything_3_video` container. OpenCV (cv2) has been completely removed; video decoding uses PyAV and image processing uses PIL/numpy/scipy.

## Testing

Tests run without GPU or model weights using lightweight mocks. All tests are in the `segment_anything_3_video/` directory.

```bash
# Run all tests (476 total)
python -m pytest test_*.py -v

# Interview pipeline tests (core)
python -m pytest test_interview_detection.py test_interview_reid.py test_constraint_reid.py -v

# Tracking + CLI tests
python -m pytest test_tracking_fixes.py test_cli.py -v

# Annotation pipeline tests
python -m pytest test_process_annotation.py test_extract_snippet_masks.py -v
```

| Test file | Tests | Coverage |
|-----------|-------|----------|
| `test_interview_detection.py` | 106 | NMS, batch detection, embedding pipeline, dual-proposer seeding (3 paths), frame cache integration, mock isolation |
| `test_interview_reid.py` | 64 | Fused similarity, spherical K-means, pair sampling, burden-of-proof policy, validation crops |
| `test_constraint_reid.py` | 57 | COP-KMeans, must-link/cannot-link constraints, centroid averaging, phase transitions, apply_resolutions |
| `test_tracking_fixes.py` | 40 | Seed frame handling, score extraction, early termination, oracle validation, batched sessions, streaming mode, correction keyframes |
| `test_process_annotation.py` | 52 | Export API, annotation extraction (4 formats), summary generation, FPS resampling, bbox JSON, snippet cutting, end-to-end pipeline |
| `test_cli.py` | 38 | CLI argument parsing, model invocation, SAM3 integration |
| `test_interview_state.py` | 26 | Session state, crop CRUD, phase transitions |
| `test_interview_cache.py` | 14 | Disk persistence, save/load round-trip, cache invalidation |
| `test_interview_background.py` | 9 | Background job executor, pause/resume, progress polling |
| `test_extract_snippet_masks.py` | 25 | SAM3 mask extraction, ffmpeg encoding |
| `test_seed_frame_pct.py` | 20 | Frame percentage slider, seeding frame selection |
| `test_disk_frame_cache.py` | 16 | 3-tier cache hierarchy, JPEG read/write, cache reuse |
| `test_lightweight_change.py` | 14 | Change-detection keyframe scoring |
| `test_reid_keybindings.py` | 10 | ReID keyboard shortcuts (1/2/3 for same/different/unsure) |

## Known Limitations

- SAM3 is designed to run on GPU servers; CPU execution is not recommended for practical video workloads.
- Currently, we do not support video segmentation (only bounding boxes).
- For very long videos (40,000+ frames), tracking may take significant time. Consider using `MAX_FRAMES_TO_TRACK` to process in chunks.
- The Interview UI persists sessions to disk via `cache_manager.py` under `/data/adapters/`. Sessions survive container restarts when the volume is mounted. Frame cache (decoded JPEGs) is also stored on disk and reused across restarts.

## Customization

The ML backend can be customized by adding your own models and logic inside the `./segment_anything_3_video` directory.
