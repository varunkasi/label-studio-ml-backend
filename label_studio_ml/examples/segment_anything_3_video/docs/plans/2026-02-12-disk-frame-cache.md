# Plan: Disk Frame Cache for Interview Pipeline

## Context

The background embedding pipeline decodes ~10,141 frames (at 10 FPS from a
30 FPS video) for lightweight change detection.  Previously this caused OOM
by accumulating all PIL Images in a list (~63 GB).  A streaming fix was
deployed but frames are discarded after scoring — forcing subsequent phases
(seeding, UI) to re-decode from the video.

This plan adds a **persistent disk-based frame cache** so frames are decoded
once and reused across all pipeline phases.

## Design Decisions

- **Cache location:** `/data/adapters/{cache_key}/frames/`
- **Frame format:** JPEG quality 95 (visually lossless, ~200–400 KB per 1080p frame)
- **Which frames:** Only the sampled frames from background embedding (~10,141 at 10 FPS)
- **Naming:** `{frame_idx:08d}.jpg` — zero-padded original frame index
- **Metadata file:** `frames/meta.json`
- **Change detection:** 10-frame moving average (compare current frame to mean of previous 10)
- **Small operations** (detection rounds ~40 frames, DINOv3, ReID): untouched — keep LRU cache

## Task Breakdown

### Task 1: Disk frame cache infrastructure

**Files:** `interview/disk_frame_cache.py` (NEW)

Create the disk frame cache module with these functions:

```python
# --- Write API (used during background embedding) ---

def init_frame_cache(cache_key: str, meta: dict) -> Path:
    """Create frames/ dir and write meta.json.
    meta = {target_fps, src_fps, resolution: [W, H], created_at}
    Returns the frames directory path."""

def write_frame(cache_key: str, frame_idx: int, pil_img: Image.Image,
                quality: int = 95) -> None:
    """Save a single frame as JPEG.  File: frames/{frame_idx:08d}.jpg"""

def finalize_frame_cache(cache_key: str, sampled_indices: List[int],
                         total_size_bytes: int) -> None:
    """Update meta.json with sampled_indices list and total_size_bytes."""


# --- Read API (used by change detection, seeding, UI) ---

def frame_cache_exists(cache_key: str) -> bool:
    """Check if frames/meta.json exists and has sampled_indices."""

def get_frame_cache_meta(cache_key: str) -> Optional[dict]:
    """Read meta.json.  Returns None if cache doesn't exist."""

def read_cached_frame(cache_key: str, frame_idx: int) -> Optional[Image.Image]:
    """Read a single JPEG from disk.  Returns None if file missing."""

def iter_cached_frames(cache_key: str,
                       frame_indices: Optional[List[int]] = None
                       ) -> Iterator[Tuple[int, Image.Image]]:
    """Iterate over cached frames in order.  If frame_indices is None,
    iterate all frames in sampled_indices order from meta.json.
    Yields (frame_idx, PIL.Image) tuples."""


# --- Management API (used by UI landing page, session restore) ---

def get_frame_cache_size(cache_key: str) -> int:
    """Total bytes of frames/ directory.  Returns 0 if not present."""

def get_all_frame_cache_sizes() -> Dict[str, int]:
    """Scan all cache_keys under CACHE_ROOT, return {cache_key: bytes}."""

def delete_frame_cache(cache_key: str) -> bool:
    """Delete frames/ directory for a cache_key."""
```

The CACHE_ROOT is imported from `cache_manager.py` (already defined as
`/data/adapters`).

### Task 2: Integrate disk frame cache into background embedding

**Files:** `seeding_common.py` (modify `compute_lightweight_change_from_video`)

Modify the streaming function to ALSO write frames to disk as a side effect:

**Path A — no existing disk cache (first run):**
1. Call `init_frame_cache(cache_key, meta)`.
2. For each decoded frame:
   a. `write_frame(cache_key, frame_idx, pil_img)` — write JPEG to disk.
   b. Resize to 128×128 thumbnail.
   c. Compute change score vs 10-frame moving average of previous thumbnails.
   d. `del pil_img` — discard full-resolution image immediately.
3. Call `finalize_frame_cache(cache_key, sampled_indices, total_size)`.
4. Return `(scores, sampled_indices)` — same contract as before.

**Path B — existing disk cache (session restore / re-run):**
1. Call `frame_cache_exists(cache_key)`.  If True:
2. `iter_cached_frames(cache_key)` — read JPEGs from disk sequentially.
3. For each frame: resize to 128×128, compute change score vs 10-frame
   moving average.
4. Return `(scores, sampled_indices)` — same contract.
5. No video decode at all.

**Signature change:** Add `cache_key: Optional[str] = None` parameter.
If None, behave like today (stream-only, no disk writes).  The caller in
`run_embedding_background` (detection.py) passes `session.cache_key`.

**10-frame moving average implementation:**
Keep a deque of the last 10 thumbnails (128×128×3 uint8).  Compute
pixel-wise mean of the deque as the baseline.  Score = max(pixel_L1_diff,
histogram_chi_sq_diff) between current thumbnail and baseline mean.  When
deque has < 10 frames, use whatever is available (graceful startup).

Memory overhead: 10 × 128 × 128 × 3 = 491,520 bytes ≈ 0.5 MB.

### Task 3: Integrate disk frame cache into seeding phase

**Files:** `interview/seeding_phase.py` (modify `generate_seeds`)

Currently `generate_seeds()` decodes every frame (or uniform + change subset)
via `_decode_frames_sequential()` in chunks of 100.

**Change:** Before chunked video decode, check if disk frame cache exists:

```python
from .disk_frame_cache import frame_cache_exists, read_cached_frame

if frame_cache_exists(session.cache_key):
    # Read frames from disk cache instead of re-decoding
    for frame_idx in target_frames:
        pil_img = read_cached_frame(session.cache_key, frame_idx)
        if pil_img is None:
            # Frame not in cache (different FPS sampling) — fall back to seek
            pil_img = _read_frame_pyav(session.video_path, frame_idx)
        # ... run SAM3 detection on pil_img ...
else:
    # Current behavior: _decode_frames_sequential in chunks
    ...
```

**Seeding frame selection:** The seeding phase currently lets users pick a
frame interval.  When disk cache exists, limit candidates to the
`sampled_indices` from the cache (the change-keyframe subset).  The UI
slider becomes "% of change keyframes to seed" instead of "frame interval".

### Task 4: Integrate disk frame cache into UI frame serving

**Files:** `interview/frame_cache.py` (modify `read_frame_cached`)

Add disk frame cache as a middle tier in the lookup chain:

```
LRU memory cache → disk frame cache → PyAV seek
```

```python
def read_frame_cached(video_path: str, frame_idx: int,
                      cache_key: Optional[str] = None) -> Optional[Image.Image]:
    # 1. Check in-memory LRU
    cached = get_cached_frame(video_path, frame_idx)
    if cached is not None:
        return cached

    # 2. Check disk frame cache (NEW)
    if cache_key:
        from .disk_frame_cache import read_cached_frame
        disk_img = read_cached_frame(cache_key, frame_idx)
        if disk_img is not None:
            put_cached_frame(video_path, frame_idx, disk_img)
            return disk_img

    # 3. Fall back to PyAV seek (existing)
    pil_img = _read_frame_pyav(video_path, frame_idx)
    if pil_img is not None:
        put_cached_frame(video_path, frame_idx, pil_img)
    return pil_img
```

Update callers in `routes.py` to pass `session.cache_key` where available.

### Task 5: Landing page disk usage + session restore cache check

**Files:**
- `interview/routes.py` (new API endpoint + modify session restore)
- `interview/static/app.js` (landing page UI)

**5a. New API endpoint: `/api/disk_usage`**

```python
@interview_bp.route("/api/disk_usage", methods=["GET"])
def disk_usage():
    from .disk_frame_cache import get_all_frame_cache_sizes
    sizes = get_all_frame_cache_sizes()
    total = sum(sizes.values())
    return jsonify({
        "total_bytes": total,
        "total_human": _human_size(total),
        "per_session": {k: {"bytes": v, "human": _human_size(v)}
                        for k, v in sizes.items()},
    })
```

**5b. Landing page UI:**
On the interview landing page, show a small info card:
```
Frame Cache: 6.2 GB (2 videos cached)
[Manage Cache]
```

**5c. Session restore:**
When loading a cached session (existing flow in routes.py), also check
for frame cache.  Report to the UI:
```json
{
  "has_frame_cache": true,
  "frame_cache_size": "3.1 GB",
  "frame_cache_frames": 10141
}
```
The UI can show: "Frame cache available (3.1 GB, 10,141 frames). [Reuse] [Re-extract]"

### Task 6: Pass `cache_key` through the call chain

**Files:** `interview/detection.py` (modify `run_embedding_background`)

The current call:
```python
scores, sampled_indices = compute_lightweight_change_from_video(
    video_path, target_fps=EMBEDDING_TARGET_FPS,
    pause_event=pause_event, progress_callback=_progress_cb,
)
```

Add `cache_key=session.cache_key`:
```python
scores, sampled_indices = compute_lightweight_change_from_video(
    video_path, target_fps=EMBEDDING_TARGET_FPS,
    pause_event=pause_event, progress_callback=_progress_cb,
    cache_key=session.cache_key,
)
```

Also update the recovery path in `routes.py:_recover_embedding_if_needed`
to pass `cache_key`.

## Execution Order

Tasks 1 → 2 → 3, 4, 5, 6 (3-6 can be parallel after 1-2 are done).

Task 1 is pure infrastructure with no dependencies.
Task 2 depends on Task 1 (uses write/read APIs).
Tasks 3, 4, 5, 6 all depend on Task 1 (use read APIs) and can be done
in parallel.

## Testing Strategy

- **Task 1:** Unit tests for all disk_frame_cache functions (write, read,
  iter, size, delete).  Use tmp_path fixture.
- **Task 2:** Test both Path A (no cache → stream + write) and Path B
  (cache exists → read from disk).  Mock av.open for streaming tests.
  Verify 10-frame moving average produces smoother scores than pairwise.
- **Task 3:** Test seeding reads from disk cache when available, falls
  back to video decode when not.
- **Task 4:** Test 3-tier lookup chain (LRU → disk → PyAV).
- **Task 5:** Test /api/disk_usage endpoint, test session restore reports
  frame cache status.
- **Task 6:** Integration test that cache_key flows through to
  compute_lightweight_change_from_video.

## Risk Assessment

- **Disk space:** ~2–4 GB per video at quality 95.  User accepts this.
  Landing page shows usage for informed decisions.
- **JPEG quality loss:** Quality 95 is visually lossless.  SAM3 detection
  is robust to minor compression artifacts.
- **Cache invalidation:** If video file changes, cache_key (based on
  task_id + video_key) changes too — stale cache won't be used.
- **Concurrent access:** Background embedding writes while UI might read.
  JPEG writes are atomic-enough (write to complete file).  meta.json
  is written last via finalize_frame_cache.
