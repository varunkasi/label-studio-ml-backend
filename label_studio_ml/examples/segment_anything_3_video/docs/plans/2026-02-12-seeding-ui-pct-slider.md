# Plan: Seeding UI — % Slider + Change Keyframe Visualization

## Context

The seeding UI currently shows "Frame Interval" (every Nth video frame) and "Confidence Threshold." This doesn't leverage the disk frame cache (10fps extraction) or give the user visibility into change keyframes. Three changes needed:

1. **Replace "Frame Interval"** with a **"% of cached frames"** slider (0–100%)
2. **Display change keyframe stats**: "X change keyframes out of Y total frames"
3. **Lo-fi timeline visualization**: a sparkline/density bar showing where change keyframes are concentrated across the video

## Files Modified

| File | Change |
|------|--------|
| `interview/state.py` | Replace `SeedConfig.frame_interval` with `SeedConfig.frame_pct` (int, default 100) |
| `interview/seeding_phase.py` | Replace interval-based frame selection with %-of-cached-frames logic |
| `interview/routes.py` | Update GET/PUT `/seeds/config` to serve `frame_pct` + `change_keyframes` + `frames_count` + `cached_frame_count` |
| `interview/static/seeding_ui.js` | Replace interval input with % slider, add keyframe stats + timeline viz |

## Change 1: `interview/state.py` — SeedConfig

Replace `frame_interval` with `frame_pct`:

```python
@dataclass
class SeedConfig:
    """User-configurable seed generation parameters."""
    frame_pct: int = 100           # % of cached frames to scan (0-100)
    confidence_threshold: float = 0.8
```

Backwards compatibility: if a serialized session has `frame_interval` but not `frame_pct`, the cache_manager deserialization should ignore the old field gracefully (dataclass will use default 100).

## Change 2: `interview/seeding_phase.py` — Frame Target Selection

Replace lines 398–402:

```python
# OLD:
# interval = max(1, session.seed_config.frame_interval)
# uniform = set(range(0, session.frames_count, interval))
# change = set(session.change_keyframes) if session.embedding_complete else set()
# all_targets = sorted(uniform | change)

# NEW:
from .disk_frame_cache import frame_cache_exists, get_frame_cache_meta

change = set(session.change_keyframes) if session.embedding_complete else set()

# Determine candidate frame pool
if frame_cache_exists(session.cache_key):
    meta = get_frame_cache_meta(session.cache_key)
    cached_indices = meta.get("sampled_indices", []) if meta else []
else:
    # Fallback: generate uniform indices from video frame count
    cached_indices = list(range(0, session.frames_count, 3))  # ~10fps from 30fps

pct = max(1, min(100, session.seed_config.frame_pct))

if pct >= 100:
    sampled = set(cached_indices)
else:
    # Uniformly subsample cached frames to desired %
    n_target = max(1, int(len(cached_indices) * pct / 100))
    step = max(1, len(cached_indices) // n_target)
    sampled = set(cached_indices[::step])

# Always include change keyframes (they're guaranteed in the cache)
all_targets = sorted(sampled | change)
```

Key properties:
- At 100%, scans all cached frames (same as current behavior with interval=1 but from cache)
- At lower %, uniformly subsamples the cached frame list
- Change keyframes are always included regardless of %
- Falls back to interval-based indices if disk cache doesn't exist

## Change 3: `interview/routes.py` — Config Endpoints

### GET `/api/seeds/config`

Add `change_keyframes`, `frames_count`, and `cached_frame_count` to the response so the UI can render the stats and timeline:

```python
@interview_bp.route("/api/seeds/config", methods=["GET"])
def seeds_config_get():
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    # Count cached frames
    cached_count = 0
    try:
        from .disk_frame_cache import frame_cache_exists, get_frame_cache_meta
        if frame_cache_exists(session.cache_key):
            meta = get_frame_cache_meta(session.cache_key)
            if meta and "sampled_indices" in meta:
                cached_count = len(meta["sampled_indices"])
    except ImportError:
        pass

    return jsonify({
        "frame_pct": session.seed_config.frame_pct,
        "confidence_threshold": session.seed_config.confidence_threshold,
        "change_keyframes": session.change_keyframes,
        "frames_count": session.frames_count,
        "cached_frame_count": cached_count,
    })
```

### PUT `/api/seeds/config`

Update to accept `frame_pct` instead of `frame_interval`:

```python
if "frame_pct" in data:
    session.seed_config.frame_pct = max(1, min(100, int(data["frame_pct"])))
```

## Change 4: `interview/static/seeding_ui.js` — UI

### 4a. Load stats in `init()`

Update the config loading (lines 47–55) to capture the new fields:

```javascript
var cfg = await API.get('/seeds/config', { session_id: sessionId });
this.seedConfig.frame_pct = cfg.frame_pct || 100;
this.seedConfig.confidence_threshold = cfg.confidence_threshold || 0.8;
this.changeKeyframes = cfg.change_keyframes || [];
this.framesCount = cfg.frames_count || 0;
this.cachedFrameCount = cfg.cached_frame_count || 0;
```

### 4b. Replace `_renderConfig()` form

Remove the "Frame Interval" number input. The new layout (top to bottom):

**1. Header + explanation** (update text)

**2. Change keyframe stats bar** — a single line:
```
Change keyframes: 47 out of 24,342 frames (0.19%)
```

**3. Timeline density visualization** — a `<canvas>` element, 100% width × 40px height. Draw logic:

```javascript
_renderTimeline(canvas) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    var total = this.framesCount;
    if (total === 0) return;

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    // Bin change keyframes into columns (1 pixel = 1 bin)
    var nBins = W;
    var bins = new Array(nBins).fill(0);
    this.changeKeyframes.forEach(function (f) {
        var bin = Math.min(nBins - 1, Math.floor(f / total * nBins));
        bins[bin]++;
    });
    var maxBin = Math.max(1, Math.max.apply(null, bins));

    // Draw bars
    ctx.fillStyle = '#e94560';
    for (var i = 0; i < nBins; i++) {
        if (bins[i] > 0) {
            var barH = Math.max(1, (bins[i] / maxBin) * H);
            ctx.fillRect(i, H - barH, 1, barH);
        }
    }
}
```

This produces a red-on-dark sparkline where taller bars = more change keyframes concentrated in that portion of the video. Bin width adapts to canvas width automatically.

**4. "% of Cached Frames" slider** — range input 1–100, same styling as confidence threshold:

```javascript
// Label: "Frame Coverage"
// Hint: "Scan X% of 8,114 cached frames (always includes change keyframes)"
// Slider: range 1-100, default 100
// Value display: "100%"
```

The hint text updates dynamically as the slider moves:
```javascript
slider.addEventListener('input', function () {
    var pct = parseInt(slider.value, 10);
    var nFrames = Math.max(1, Math.round(self.cachedFrameCount * pct / 100));
    var nChange = self.changeKeyframes.length;
    var effective = Math.max(nFrames, nChange);
    valueSpan.textContent = pct + '%';
    hintDiv.textContent = 'Scan ~' + effective + ' frames (' +
        nChange + ' change keyframes always included)';
});
```

**5. Confidence threshold slider** (unchanged)

**6. Generate Seeds button** (update to pass `frame_pct` instead of `frame_interval`)

### 4c. Update `_generateSeeds()` signature

```javascript
async _generateSeeds(framePct, confidenceThreshold) {
    // ...
    await API.put('/seeds/config', {
        session_id: this.sessionId,
        frame_pct: framePct,
        confidence_threshold: confidenceThreshold,
    });
    // ... rest unchanged
}
```

## Tests

No new test file needed. Verify:
1. `SeedConfig` default is `frame_pct=100`
2. Seeding phase at 100% produces same frame count as before (all cached)
3. Seeding phase at 50% produces roughly half the cached frames + all change keyframes
4. GET `/seeds/config` returns the new fields
5. Old serialized sessions with `frame_interval` don't crash (graceful default)

Test manually via UI: slider should show timeline, stats, and produce seeds.

## Migration

Existing sessions with `frame_interval` in their serialized `SeedConfig`:
- `cache_manager.py` deserialization already uses `SeedConfig(**data)` which will raise on unknown fields
- Add a migration shim: if `frame_interval` is present in the dict but `frame_pct` is not, pop `frame_interval` and set `frame_pct = 100` (scan everything, safest default)
