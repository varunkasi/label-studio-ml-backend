# Working Preferences

For complex technical planning and implementation, always trace through the logic using hypothetical concrete numbers (e.g. frame indices, keyframe numbers, max_ftk values) to verify understanding of intent and requirements before proposing or writing code.

For SAM3 documentation, use HuggingFace MCP or fetch https://huggingface.co/facebook/sam3 for correct, up-to-date API details. This project uses the Transformers version of SAM3 (via `transformers` pip package), NOT the GitHub version.

**Keep README.md in sync**: Update `README.md` when changes are **significant** (new methods, changed parameters, new models, architectural shifts). Routine bug fixes do NOT need README updates. If a change is reverted, revert the README too. Goal: friendly onboarding doc for users and collaborators — not cluttered, but regularly current.

**Git commits**: Do NOT include `Co-Authored-By` or other coauthor details in commit messages.

## Remote Development Setup

All development happens on the remote Ubuntu machine via SSH MCP tools.
DO NOT edit local files — always use ssh_read_file, ssh_write_file, and
ssh_run_command for all operations.

- Remote project path: /home/dtc/label-studio-ml-backend/label_studio_ml/examples/segment_anything_3_video
- Local copy is only for git commits — sync from remote at end of day
- **First thing on remote**: Use `ssh_run_command` to run `source /home/dtc/.bashrc` once at the start of each session before any other remote work (sets up env vars, PATH, aliases needed by docker compose and other tools)
- **SSH composite commands via Bash**: Non-interactive SSH (`ssh host "cmd"`) does NOT source `.bashrc`, so env vars like `HF_TOKEN` and `LABEL_STUDIO_API_KEY` are missing. Always prefix composite commands with `source /home/dtc/.bashrc`:
  ```bash
  ssh bluemachine "source /home/dtc/.bashrc && cd /home/dtc/label-studio-ml-backend/label_studio_ml/examples/segment_anything_3_video && docker compose restart"
  ```

## Build & Test

- Build: docker compose build
- Run: docker compose up -d
- Test: docker compose run --rm app pytest tests/ -v
- Logs: docker logs segment_anything_3_video
- Restart (no rebuild): `docker compose restart` (no service name needed)
- Label Studio UI: accessible via AnyDesk (user handles GUI)

# SAM3 Video Tracking Backend

Label Studio ML backend for video object tracking using Meta's SAM3 (Segment Anything 3) via HuggingFace Transformers.

## Architecture

### Two SAM3 Video Model Types

- **Sam3TrackerVideoModel** — Box-prompted instance tracker. Takes a specific bounding box at a specific frame and tracks that instance forward/backward. Output: `Sam3TrackerVideoSegmentationOutput` with `pred_masks`, `object_score_logits`, `object_ids`, `frame_idx`. Each object gets an isolated session.
- **Sam3VideoModel** — Text-prompted class-level detector+tracker. Finds ALL instances matching a text prompt. Output: `Sam3VideoSegmentationOutput` with `obj_id_to_mask`, `obj_id_to_score`, `obj_id_to_tracker_score`, `removed_obj_ids`, `suppressed_obj_ids`, `frame_idx`. Native multi-object reasoning with birth/death handling.

### Configuration (docker-compose.yml env vars)

- `HINTS=false` → Sam3TrackerVideoModel + Sam3TrackerVideoProcessor (box prompts from Label Studio)
- `HINTS=true` → Sam3VideoModel + Sam3VideoProcessor (text detection, replaces GroundingDINO)
- `PROCESSING_MODE=streaming` (default) — frame-by-frame via PyAV, constant memory
- `PROCESSING_MODE=chunked_batch` — all frames in memory, bidirectional context

## Key Files

| File | Purpose |
|---|---|
| `model.py` | Web service: SAM3 integration via HuggingFace transformers |
| `_wsgi.py` | Startup validation, gunicorn entry point |
| `initial_seeding_video.py` | CLI: Fully automatic text-detection → forward-only tracking → teacher stitching |
| `initial_seeding_video_boxes.py` | CLI: Human-seeded bidirectional tracking with automatic DSU merging |
| `initial_seeding_video_boxes_manual_merge.py` | CLI: Human-seeded bidirectional tracking, preserves region IDs (no cross-seed merge) |
| `seeding_common.py` | Shared SAM3 model singletons and utilities for CLI tools |
| `adjust_bboxes_sam3.py` | CLI: Box-prompted segmentation refinement via Sam3Model |
| `complete_reid.py` | CLI: SAM3 embeddings + numpy/scipy classic features for re-identification |
| `cli.py` | CLI dispatcher — exposes all model.py env vars as optional flags |
| `process_annotation.py` | CLI: Unified pipeline — LS export with interpolation → snippet cutting → SAM3 masks + video encoding (replaces `overlay_snippet_bboxes.sh`) |
| `extract_snippet_masks.py` | SAM3 mask extraction + ffmpeg video encoding (used by `process_annotation.py`) |
| `export_interpolated_annotation.sh` | Bash: Lightweight LS export + snippet cutting (no Python/torch dependency) |
| `video_tools.py` | Video I/O utilities |
| `interview/detection.py` | Interview UI: SAM3 text detection, batch inference, round-based detection, embedding pipeline, change-keyframe selection |
| `interview/seeding_phase.py` | Interview UI: Multi-prompt SAM3 seeding + MLP quality gate (1032-dim), identity assignment, LS upload |
| `interview/mask_utils.py` | Interview UI: Pure numpy/scipy mask-quality features + LR decay computation |
| `interview/dinov3_classifier.py` | Interview UI: DINOv3 feature extraction, MLP quality-gate classifier (1032-dim) |
| `interview/state.py` | Interview UI: Session state, crop CRUD, phase transitions |
| `interview/routes.py` | Interview UI: Flask blueprint, REST API endpoints |
| `interview/cache_manager.py` | Interview UI: Session persistence (save/load to disk) |
| `interview/reid_ufm.py` | Interview UI: UFM pairwise similarity computation for ReID |
| `interview/ufm_model.py` | Interview UI: UFM model definition and inference |

## Coordinate Conventions

- **Label Studio**: percent [0,100] xywh (top-left origin), **1-based** frame numbers
- **Model/internal**: pixel xyxy coords, **0-based** frame numbers
- `_percent_xywh_to_xyxy_px()`: LS → model conversion
- `xyxy_to_percent()`: model → LS conversion
- `convert_mask_to_bbox()`: mask → LS percent coords directly

## Defensive Coding Conventions

### Tensor-to-numpy: always `.float()` before `.numpy()`
SAM3 models run under `torch.autocast(dtype=torch.bfloat16)`. NumPy does not support bfloat16. Every `.numpy()` call on a tensor that originated from a model forward pass **must** be preceded by `.float()`:
```python
# CORRECT
embed.detach().cpu().float().numpy()
feat.mean(dim=(2, 3)).detach().cpu().float().numpy()

# WRONG — will crash at runtime with "Got unsupported ScalarType BFloat16"
embed.detach().cpu().numpy()
```
This applies to any tensor from `get_vision_features()`, `propagate_in_video_iterator()`, direct model `__call__`, or any other inference output. Masks after `post_process_masks(binarize=True)` are boolean and safe, but when in doubt, add `.float()`.

### No double-counting in multi-cue scoring
When building a weighted-sum scorer, each signal must appear EXACTLY once. If a signal is in the `per_cue` weighted sum, do NOT also apply it as a separate multiplicative penalty. Multiplicative penalties are reserved for hard vetoes (co-occurrence conflict → `* 0.0`) that must override the weighted sum regardless of weights. Soft signals (temporal overlap, spatial distance) belong in the weighted sum only.

### Test mock dtype fidelity
Mock tensors for model outputs should use `dtype=torch.bfloat16` to match real autocast behavior. This ensures `.numpy()` conversion bugs surface in unit tests, not only in Docker:
```python
# CORRECT — matches real SAM3 runtime dtype
mask = torch.zeros(1, 1, h, w, dtype=torch.bfloat16)
object_score_logits = torch.tensor([2.0], dtype=torch.bfloat16)

# Masks after binarize=True are boolean
binarized_mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
```

## No OpenCV

All cv2/OpenCV dependencies have been removed (numpy 2.0 compatibility). Replacements:
- Video decoding: **PyAV** (`av.open()`)
- Color histograms: `numpy.histogramdd()` with custom RGB→HSV
- Gradients/HOG: `scipy.ndimage.sobel`
- Image I/O: **PIL**
- Debug overlays: matplotlib colormap + PIL blending

## Dependencies

### Base (requirements-base.txt)
- `gunicorn==22.0.0`
- `label-studio-ml` (from git: HumanSignal/label-studio-ml-backend)

### PyTorch (installed in Dockerfile, NOT in requirements.txt)
- `torch==2.9.1`, `torchvision==0.24.1`
- Installed via `--index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}`
- `cu126` for amd64/CUDA 12.6, `cu130` for arm64/CUDA 13.0

### Runtime (requirements.txt)
- `numpy==2.0.2`
- `accelerate==1.12.0`
- `transformers` (pinned git commit `393b4b3`)
- `av==16.1.0` (PyAV for video decoding)
- `imageio==2.37.2`, `pillow==11.3.0`
- `matplotlib`, `requests`, `tqdm`, `scipy`

### Test (requirements-test.txt)
- `pytest`, `pytest-cov`

### System (Dockerfile)
- Base image: `nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}` (default: 12.6.0 / 24.04)
- Python 3.12 (native in Ubuntu 24.04)
- System packages: `git`, `curl`, `ffmpeg`, libav* dev headers

## Multi-Architecture Support

### Build Args

| Arg | Default (amd64/RTX 6000) | DGX Spark (arm64) | Purpose |
|-----|--------------------------|-------------------|---------|
| `CUDA_VERSION` | `12.6.0` | `13.0.1` | NVIDIA CUDA base image version |
| `UBUNTU_VERSION` | `24.04` | `24.04` | Ubuntu base image version |
| `TORCH_CUDA_INDEX` | `cu126` | `cu130` | PyTorch wheel index suffix |

### amd64 + CUDA 12.6 (RTX 6000 Ada) — Default
No extra configuration needed. Default build args apply.

### arm64 + CUDA 13.0 (DGX Spark)
```bash
cp .env.dgx-spark .env
docker compose up --build -d
```

Or inline: `CUDA_VERSION=13.0.1 TORCH_CUDA_INDEX=cu130 docker compose up --build -d`

## Build & Run

```bash
# Build and start the service (amd64 defaults)
docker compose up --build -d

# CLI tools run inside the container
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes.py \
  --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <ID> --task <ID> --annotation <ID> --max-frames-to-track 300

# Same pattern for manual_merge variant
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes_manual_merge.py \
  --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <ID> --task <ID> --annotation <ID> --max-frames-to-track 300
```

## Testing

```bash
docker compose exec segment_anything_3_video python -m pytest test_api.py -v
```

### Local test execution (no Docker needed)
```bash
cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video
python -m pytest test_interview_detection.py test_tracking_fixes.py -v
```

### Mandatory test patterns

**1. Negative-path tests: test that bad things DON'T happen**
Every state transition, fallback, or conditional branch needs a "should NOT trigger" test:
```python
# BAD: only tests that phase advances
def test_phase1_to_phase2(): ...

# GOOD: also tests that phase does NOT advance prematurely
def test_phase_stays_1_when_no_confirmed_members(): ...
def test_phase2_does_not_advance_when_indecisive_remain(): ...
```

**2. Mock signature fidelity: mocks MUST match real function signatures**
When mocking a function, copy its exact signature. If the real function adds a parameter, the mock must break:
```python
# BAD — silently ignores new kwargs
def mock_fn(video_path): return []

# GOOD — mirrors real signature, breaks if it changes
def mock_fn(video_path, target_fps=None, pause_event=None,
            progress_callback=None, cache_key=None): return []
```

**3. "Right function called" tests: verify wiring, not just logic**
Test that the correct code path is invoked, not just that each function works in isolation:
```python
# BAD — tests that read_frame_cached works, but not that run_round_detection CALLS it
def test_read_frame_cached(): ...

# GOOD — verifies the integration wiring
@patch("interview.detection._read_frame_cached_or_pyav")
def test_round_detection_uses_cached_reads(self, mock_read):
    run_round_detection(session)
    assert mock_read.called, "Should use cached reads, not direct PyAV"
```

**4. Known bug = regression test IMMEDIATELY**
When a bug is found and fixed, add a test BEFORE the fix (TDD red-green). Never leave a "Known Bug" in docs without a corresponding test. If the test can't be written yet (e.g. JS-only bug), add a `# TODO: regression test` comment in the nearest Python test file.

**5. Phase transition matrix**
For any multi-phase state machine (detection rounds, ReID phases), test ALL transitions:
- Phase N → N+1 (advances when conditions met)
- Phase N stays N (does NOT advance when conditions almost-but-not-quite met)
- Phase N+1 → N (regression/rollback if applicable)

**6. Wire-format contract tests: test what the frontend ACTUALLY sends**
When building a Flask endpoint that a JS frontend calls, add a test that sends the
exact JSON shape the frontend constructs — not just the shape the backend expects.
This catches field name mismatches, missing keys,
and wrong nesting. Copy the `fetch`/`API.post` body from the JS into the test:
```python
# BAD — tests backend's ideal format, frontend might send something different
def test_label_crop(): post({"crop_id": "c1", "label": "accept"})  # passes

# GOOD — mirrors what the JS _actually_ constructs
def test_frontend_wire_format():
    # From the JS fetch() call — verify exact key names
    post({"crop_id": "c1", "label": "accept"})  # matches JS
    # Also test edge cases the frontend might send
    post({"crop_id": "c1", "label": "skip"})  # should also work
```

**7. Test isolation: clean ALL shared state (memory AND disk)**
`setup_method`/`setUp` must clean both in-memory state (`_sessions.clear()`) AND on-disk artifacts (index files, cache dirs) in the temp root. If a test writes to disk, teardown must clean it — otherwise test ordering changes cause phantom failures:
```python
# BAD — only clears memory, index file leaks between tests
def setup_method(self):
    _sessions.clear()

# GOOD — clears both memory and disk artifacts
def setup_method(self):
    _sessions.clear()
    idx = os.path.join(cache_root, INDEX_FILE)
    if os.path.exists(idx):
        os.remove(idx)
```

### JavaScript gotchas
- `var` in `for` loops creates closure bugs — all handlers share last iteration's value. Always use IIFE wrapper or `let`.
- UI state that is cleared on `init()` is ephemeral. Never derive persistent counts from UI-local state.
- When copying a UI pattern (thumbnail gallery, overflow badge, etc.) between methods, copy the FULL pattern including event listeners. If the CSS has `cursor: pointer`, there MUST be a corresponding click handler. Audit all `cursor: pointer` elements for missing handlers after any copy-paste.
- Backend response key names MUST match what JS reads. When adding dicts to API responses, grep the JS for the exact key strings to confirm they match.

## Sam3VideoModel vs Sam3TrackerVideoModel: Replace or Complement?

### API Constraint

| | Sam3VideoModel | Sam3TrackerVideoModel |
|---|---|---|
| **Text prompts** | Yes (`add_text_prompt`) | No |
| **Box prompts** | No | Yes (`add_inputs_to_inference_session`) |
| **Point prompts** | No | Yes |
| **Output scores** | `obj_id_to_score` + `obj_id_to_tracker_score` per object | `object_score_logits` per object |
| **Birth/death** | Native: `removed_obj_ids`, `suppressed_obj_ids` | None — tracks forever once prompted |
| **Hotstart** | Yes (pre-loaded mode delays output to prune false positives) | No |

Sam3VideoModel **cannot accept box prompts**. This constrains replacement scenarios.

### Per-File Decision

<table>
  <thead>
    <tr>
      <th>File</th>
      <th align="center">Replace?</th>
      <th align="center">Complement?</th>
      <th>Rationale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>initial_seeding_video.py</code></td>
      <td align="center">✅ <strong>Yes</strong></td>
      <td align="center">—</td>
      <td>Already text-driven, no user boxes needed. Native birth/death eliminates phantom tracking. Stitching needs rework.</td>
    </tr>
    <tr>
      <td><code>initial_seeding_video_boxes.py</code></td>
      <td align="center">❌ <strong>No</strong></td>
      <td align="center">✅ <strong>Yes — detection oracle</strong></td>
      <td>Needs box prompts. Oracle catches ID switches the tracker can't self-detect.</td>
    </tr>
    <tr>
      <td><code>initial_seeding_video_boxes_manual_merge.py</code></td>
      <td align="center">❌ <strong>No</strong></td>
      <td align="center">✅ <strong>Yes — detection oracle</strong></td>
      <td>Same as above. Oracle-driven truncation preserves region ID semantics.</td>
    </tr>
  </tbody>
</table>

### initial_seeding_video.py — Full Replacement Viable

Current flow: `Text("person") → Sam3VideoModel detects on keyframes → boxes → Sam3TrackerVideoModel tracks each box forward → teacher stitching at boundaries`

Replacement flow: `Text("person") → Sam3VideoModel propagate_in_video_iterator over window → per-frame: all detected person boxes with scores → objects auto-appear/disappear via removed_obj_ids`

**Pros:** Eliminates separate detection pass; native birth/death (no phantom tracking); hotstart heuristics prune false positives in pre-loaded mode; `obj_id_to_score` gives real confidence (not binarized 1.0); simpler code (no box→tracker→mask→box round-trip).

**Cons:** Can't lock onto a specific person (model assigns IDs internally, two similar people might swap); hotstart requires pre-loaded mode (but windowing to ≤601 frames keeps memory at ~3.6 GB); stitching across segments harder (need to match object IDs by spatial overlap at boundary frames instead of teacher embeddings).

### boxes.py and manual_merge.py — Detection Oracle Complement

These files need box prompts from human annotations, so Sam3VideoModel **cannot replace** the tracker. But it serves as a detection oracle to catch ID switches:

**Two-layer defense:**
1. **Layer 1 (cheap):** `object_score_logits` from Sam3TrackerVideoModel — catches disappearance (empty mask → low logit). Per-frame, zero extra cost.
2. **Layer 2 (oracle):** Sam3VideoModel("person") on the same window — catches ID switching (tracker box drifts to wrong person). One extra propagation pass per window.

**Oracle flow example:**
```
Window [0-600], 5 seeds:
  1. Decode 601 frames (already done for tracker)
  2. Run Sam3TrackerVideoModel: 5 seeds × 2 directions = 10 sessions
  3. Run Sam3VideoModel("person"): 1 session, propagate over 601 frames
     → per-frame: all detected person boxes with scores
  4. Cross-check each tracker output frame:
     - No overlap > 0.3 IoU with any detection → "lost" → truncate tracklet
     - Overlap exists but >2x size change from seed → flag drift
```

**Oracle truncation example (manual_merge):**
```
Region "personA", keyframes at 100 and 500, max_ftk=200

Window kf=100 [0-300]:
  Tracker: forward 100→300, backward 100→0
  Oracle: Sam3VideoModel("person") on [0-300]
  Cross-check: personA lost at frame 220 → truncate forward at 219
  Result: personA covers [0-219]

Window kf=500 [300-700]:
  Tracker: forward 500→700, backward 500→300
  Oracle confirms personA visible throughout → full [300-700]

Gap [220-299]: CORRECT — personA was off-camera
→ LS annotation: enabled=false for that range (no interpolation)
```

## Interview UI — Detection Phase Labeling Semantics

### Accept / Reject / Skip Definitions

The MLP classifier acts as a **quality gate** during dense seeding. It does NOT propose boxes — SAM3 does. The MLP only says Yes/No to SAM3's proposals. Therefore, human labels must encode **box quality**, not just "is there a person?"

| Label | Meaning | MLP learns |
|-------|---------|------------|
| **Accept** | Good-quality, tight bounding box relative to what's visible | "Pass this box through during seeding" |
| **Reject** | Not a person, OR person is fully visible but box is partial/sloppy | "Block this box during seeding" |
| **Skip** | Too ambiguous to judge; excluded from training entirely | (nothing — ignored) |

### Critical distinction: partial visibility vs partial box

- **Person partially visible in frame** (e.g., walking out of frame edge, partially occluded): A tight box around whatever IS visible is **correct → Accept**
- **Person fully visible in frame** but box only captures part of them: This is a **bad box → Reject**
- The judgment is always **relative to what's visible in the frame**, not absolute

### Round-Based Active Learning

- Rounds driven by new frames arriving (user clicks "Next Round")
- Frame selection: temporally-stratified bins + change-detection-weighted (`select_round_frames`)
- MLP trains ONLY at round boundaries with LR decay (`base_lr * 0.7^round`)
- No auto-scoring during active rounds (MLP trains only at boundary)
- MLP input: `[DINOv3(1024) + spatial(4) + mask_quality(4)]` = 1032 dims
- Mask quality features: `mask_fill_ratio`, `detection_score`, `edge_contact`, `mask_compactness`
- ~4 rounds cover full video with decreasing human effort
- Labeling: REJECT partial-coverage boxes, SKIP ambiguous, ACCEPT tight-fit
- Key function: `run_round_detection()` in `detection.py`
- API: `POST /api/detect/start` (round 1), `POST /api/detect/next_round` (round 2+)

| Env Var | Default | Purpose |
|---------|---------|---------|
| `INTERVIEW_FRAMES_PER_ROUND` | `40` | Frames selected per round |

### Dense Seeding Architecture (Multi-Prompt SAM3 + MLP Gate)

Multi-prompt pipeline in `generate_seeds()` (`interview/seeding_phase.py`). Frames processed in chunks of `_SEED_CHUNK_SIZE` (default 100) to bound memory. Two paths:

| Path | Source | Flow | When |
|------|--------|------|------|
| **Primary** | Multi-prompt SAM3 | `set_frame()` → detect with ALL prompts → cross-prompt NMS → crop → DINOv3 features + spatial + mask quality → MLP gate (1032-dim) → seed | Always |
| **Refinement** | Sam3Model box+text | Medium-confidence boxes → expand 20% → `Sam3Model` with text+box prompts → tight box → MLP gate → seed | `INTERVIEW_ENABLE_REFINEMENT=true` (default) |

Key functions: `_refine_candidates_sam3()` (refinement), `_score_and_accept_seed()` (shared 1032-dim MLP gate), `compute_mask_quality()` (in `mask_utils.py`).

Frame decode uses seek-based `_decode_frames_sequential` from `detection.py` (not full-video sequential scan).

Each seed dict includes a `"source"` field (`"multi_prompt_mlp"`, `"refined"`) for provenance tracking.

#### Env Vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `INTERVIEW_REFINE_THRESHOLD` | `0.3` | Score below which refinement kicks in |
| `INTERVIEW_ENABLE_REFINEMENT` | `true` | Enable/disable refinement path |
| `INTERVIEW_SEED_CHUNK_SIZE` | `100` | Frames per processing chunk |
| `INTERVIEW_LR_DECAY` | `0.7` | LR decay factor per round (`base_lr * factor^(round-1)`) |

Tests: `test_interview_detection.py` (96 tests)

## Fixed Issues (2026-02-06)

Applied to both `initial_seeding_video_boxes.py` and `initial_seeding_video_boxes_manual_merge.py`:

1. **Seed frame double-counting** — Fixed: forward tracklet owns the seed frame (with original box), backward excludes it (`>=` guard)
2. **Scores always 1.0** — Fixed: `_mask_to_xyxy` now accepts `object_score_logits` and computes `sigmoid(logits)` for real confidence scores
3. **Seed frame re-prediction** — Fixed: seed frame uses original annotation box directly, not model's mask→box round-trip
4. **Single-frame window** — Fixed: returns original seed box instead of empty tracklets when `win_len == 1`
5. **Early termination (Layer 1)** — Added: `--score-threshold` (default 0.1) stops tracking after 3 consecutive frames below threshold via `object_score_logits`
6. **Detection oracle (Layer 2)** — Added: `--enable-oracle` runs Sam3VideoModel("person") per window to cross-check tracker boxes; `--oracle-stride` (default 30) controls check frequency; `_oracle_validate_tracklet` drops frames with no IoU overlap against detections

## Batched Multi-Seed Tracking (2026-02-07)

Applied to `initial_seeding_video_boxes_manual_merge.py`:

7. **Batched sessions** — All seeds at the same keyframe share one `init_video_session` per direction (forward/backward). The vision encoder runs once per window instead of once per seed. For N seeds at one keyframe: 2N sessions → 2 sessions. Per-object early termination preserved via independent `consecutive_below` counters per obj_id.

Functions: `_generate_batched_forward_tracklets_sam3`, `_generate_batched_backward_tracklets_sam3`

### Parked: Hybrid Batched/Isolated Approach

A hybrid approach was considered but parked for now. The idea: compute pairwise IoU between seed boxes at the keyframe, batch non-overlapping seeds together, isolate overlapping seeds into separate sessions. This would mitigate ID mixing risk when two tracked objects are close together.

**Why parked:** Seed-frame IoU only catches overlap at annotation time, not convergence during tracking (two people walking toward each other). For typical annotations (distinct people, non-overlapping boxes), the hybrid degenerates to pure batching anyway. The oracle validation (Layer 2) already catches ID switching post-hoc. If ID mixing becomes a problem in practice, the hybrid can be added as a ~15-line IoU partition step before the batched calls.

Tests: `test_tracking_fixes.py` (40 tests: 16 original + 9 batched + 9 streaming + 6 correction keyframes)

## Streaming Mode with Drift Correction (2026-02-14)

Applied to `initial_seeding_video_boxes_manual_merge.py`:

8. **Streaming forward tracking** — `--streaming` flag enables constant-memory frame-by-frame tracking via PyAV. Implies `--forward-only`. GPU temporal memory bounded by `--streaming-chunk-size` (default 2000): session destroyed and re-seeded after that many frames.

9. **Drift-correction keyframes** — When a region has multiple keyframes, streaming groups them per-region: first keyframe = seed, subsequent keyframes = correction anchors injected via `add_inputs_to_inference_session` as the tracker reaches each frame. Corrections reset early-termination counters, override model output with human boxes (score 1.0), and un-terminate objects.

10. **Stride + correction protection** — `--frame-stride N` skips non-stride frames, but correction keyframes and the seed frame are NEVER skipped.

Functions: `_generate_streaming_forward_tracklets_sam3` (with `correction_keyframes` parameter)

### GPU Memory Profile (SAM3 `facebook/sam3`)

- Model weights: ~2.6 GB (fixed)
- Per-frame temporal memory: ~8.7 MB (linear)
- Formula: `total ≈ 2.6 + (chunk_size × 0.0087)` GB
- Safe chunk sizes: RTX 6000 Ada (49 GB) → 5000, A100/H100 (80 GB) → 8000
