# Interview UI Stale Code Cleanup

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove ~5000 lines of dead Python, ~200 lines of dead CSS, and ~200 lines of dead JS from the interview UI without affecting the active code path.

**Architecture:** The interview UI has two parallel ReID architectures: (1) the ACTIVE `reid_ufm.py` UFM+HAC pipeline wired into `routes.py`, and (2) the DEAD `reid_phase.py` pairwise-constraint model + `reid_pipeline.py` visual multi-cue pipeline, which are fully implemented and tested but never called from any production endpoint. This plan removes architecture (2) and all its dependencies in safe, incremental steps with verification after each task.

**Tech Stack:** Python, Flask, JavaScript, CSS. No new dependencies.

**Active path (MUST NOT BREAK):**
- `routes.py` → `reid_ufm.py` (all ReID endpoints)
- `routes.py` → `detection.py` (detection phase endpoints)
- `routes.py` → `seeding_phase.py` (seeding endpoints)
- `routes.py` → `cache_manager.py` → `state.py` (session persistence)
- `ufm_model.py` (UFM singleton, imported by `reid_ufm.py`)
- `dinov3_classifier.py` (MLP quality gate, imported by `detection.py`)
- `mask_utils.py` (mask features, imported by `detection.py`)
- JS: `app.js`, `reid_ui.js`, `seeding_ui.js`, `components.js`

**Verification command (run after EVERY task):**
```bash
cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py test_disk_frame_cache.py test_extract_snippet_masks.py test_process_annotation.py test_seed_frame_pct.py test_lightweight_change.py test_reid_montage.py test_reid_keybindings.py -v --tb=short 2>&1 | tail -20
```

---

### Task 1: Delete dead Python modules (reid_phase.py, reid_pipeline.py)

These two files (~3700 lines total) are never imported by any production code. `routes.py` imports only from `reid_ufm.py`. No file in `interview/` imports from either module.

**Files:**
- Delete: `interview/reid_phase.py` (~1900 lines)
- Delete: `interview/reid_pipeline.py` (~1800 lines)

**Step 1: Verify no production imports exist**

Run:
```bash
cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video
grep -rn "from.*reid_phase\|import.*reid_phase" interview/ --include="*.py"
grep -rn "from.*reid_pipeline\|import.*reid_pipeline" interview/ --include="*.py"
```
Expected: Zero matches (only routes.py importing `reid_ufm`, not `reid_phase` or `reid_pipeline`)

**Step 2: Delete the files**

```bash
rm interview/reid_phase.py
rm interview/reid_pipeline.py
```

**Step 3: Run active test suite**

```bash
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py test_disk_frame_cache.py test_extract_snippet_masks.py test_process_annotation.py test_seed_frame_pct.py test_lightweight_change.py test_reid_montage.py test_reid_keybindings.py -v --tb=short 2>&1 | tail -20
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add -u interview/reid_phase.py interview/reid_pipeline.py
git commit -m "chore: remove dead reid_phase.py and reid_pipeline.py modules

These modules (~3700 lines) implemented a pairwise-constraint ReID model
and visual multi-cue pipeline that were never wired into routes.py.
The active ReID path uses reid_ufm.py exclusively."
```

---

### Task 2: Delete dead test files

Five test files (~4500 lines) test only the removed modules. None of them test active code paths.

**Files:**
- Delete: `test_constraint_reid.py` (tests `reid_phase.py` constraint model)
- Delete: `test_interview_reid.py` (tests `reid_phase.py` functions: `spherical_kmeans`, `sample_pairs`, etc.)
- Delete: `test_reid_pipeline.py` (tests `reid_pipeline.py` visual pipeline)
- Delete: `test_visual_pipeline_integration.py` (tests visual pipeline HTTP endpoints that don't exist in `routes.py`)
- Delete: `test_proposal_persistence.py` (tests visual pipeline proposal persistence)

**Step 1: Verify these test files import only from deleted modules**

```bash
head -30 test_constraint_reid.py test_interview_reid.py test_reid_pipeline.py test_visual_pipeline_integration.py test_proposal_persistence.py | grep "from interview"
```
Expected: All imports reference `reid_phase` or `reid_pipeline` (now deleted)

**Step 2: Delete the test files**

```bash
rm test_constraint_reid.py test_interview_reid.py test_reid_pipeline.py test_visual_pipeline_integration.py test_proposal_persistence.py
```

**Step 3: Run active test suite**

```bash
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py test_disk_frame_cache.py test_extract_snippet_masks.py test_process_annotation.py test_seed_frame_pct.py test_lightweight_change.py test_reid_montage.py test_reid_keybindings.py -v --tb=short 2>&1 | tail -20
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add -u test_constraint_reid.py test_interview_reid.py test_reid_pipeline.py test_visual_pipeline_integration.py test_proposal_persistence.py
git commit -m "chore: remove test files for deleted reid_phase/reid_pipeline modules

Removes ~4500 lines of tests that tested only the removed pairwise-constraint
and visual pipeline code. Active test suite unaffected."
```

---

### Task 3: Remove dead state fields and ReIDPair class

The `InterviewSession` dataclass has 7 fields and 1 inner class (`ReIDPair`) used only by the deleted `reid_phase.py`. `reid_ufm.py` never references any of them. Cache backward compatibility is handled by making `cache_manager.py` silently ignore these fields when loading old session files.

**Files:**
- Modify: `interview/state.py` — remove `ReIDPair` class, remove 7 dead fields
- Modify: `interview/cache_manager.py` — remove serialization of dead fields, add `pop()` for backward-compat loading

**Step 1: Read state.py to identify exact lines**

Read `interview/state.py` and locate:
- `class ReIDPair` dataclass (lines ~95-115)
- `reid_pairs: Dict[str, ReIDPair]` field (line ~321)
- `reid_must_links: List[Tuple[str, str]]` field
- `reid_cannot_links: List[Tuple[str, str]]` field
- `reid_phase_stage: int` field (line ~323)
- `visual_reid_proposals: List[Dict[str, Any]]` field (line ~326)
- `visual_reid_weights: Dict[str, float]` field (line ~327)
- `visual_reid_verdicts_count: int` field (line ~328)

**Step 2: Remove ReIDPair class from state.py**

Delete the entire `class ReIDPair` dataclass definition.

**Step 3: Remove dead fields from InterviewSession in state.py**

Remove these 7 field definitions from the `InterviewSession` dataclass:
```python
reid_pairs: Dict[str, ReIDPair] = field(default_factory=dict)
reid_must_links: List[Tuple[str, str]] = field(default_factory=list)
reid_cannot_links: List[Tuple[str, str]] = field(default_factory=list)
reid_phase_stage: int = 1
visual_reid_proposals: List[Dict[str, Any]] = field(default_factory=list)
visual_reid_weights: Dict[str, float] = field(default_factory=dict)
visual_reid_verdicts_count: int = 0
```

**Step 4: Clean up imports in state.py**

Remove any unused imports that only supported `ReIDPair` (check for unused `Tuple` etc.).

**Step 5: Update cache_manager.py — remove serialization**

In the save function, remove lines that serialize:
- `reid_pairs`
- `reid_must_links` / `reid_cannot_links`
- `reid_phase_stage`
- `visual_reid_proposals` / `visual_reid_weights` / `visual_reid_verdicts_count`

In the load function, add `.pop()` calls to silently discard these keys if present in old cache files:
```python
# Backward compat: discard fields from old cache format
for stale_key in ("reid_pairs", "reid_must_links", "reid_cannot_links",
                  "reid_phase_stage", "visual_reid_proposals",
                  "visual_reid_weights", "visual_reid_verdicts_count"):
    data.pop(stale_key, None)
```

**Step 6: Run active test suite**

```bash
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py test_disk_frame_cache.py test_extract_snippet_masks.py test_process_annotation.py test_seed_frame_pct.py test_lightweight_change.py test_reid_montage.py test_reid_keybindings.py -v --tb=short 2>&1 | tail -20
```
Expected: All tests PASS. If `test_interview_state.py` or `test_interview_cache.py` reference removed fields, fix those tests (they're testing active infrastructure, just referencing stale fields).

**Step 7: Commit**

```bash
git add interview/state.py interview/cache_manager.py
# Also add any modified test files if they referenced removed fields
git commit -m "chore: remove dead ReIDPair class and 7 stale session fields

Removes reid_pairs, reid_must/cannot_links, reid_phase_stage, and
visual_reid_* fields from InterviewSession. Cache loading silently
discards these keys from old session files for backward compatibility."
```

---

### Task 4: Remove dead routes from routes.py

Four endpoints have no JavaScript callers and test no active functionality.

**Files:**
- Modify: `interview/routes.py` — remove 4 dead endpoints

**Step 1: Read routes.py and locate the endpoints**

Find these endpoint functions:
- `/api/reid/renumber` (POST) — cluster ID renumbering, no JS caller
- `/api/detect/training_status` (GET) — MLP training status, superseded by job polling
- `/api/detect/recall_strategy` (POST) — multi-prompt recall, no JS caller
- `/api/detect/crop/<crop_id>/context` (GET) — expanded context patch, no JS caller

**Step 2: Remove the 4 endpoint functions**

Delete each `@interview_bp.route(...)` + function body for the 4 dead endpoints.

**Step 3: Remove unused imports in routes.py**

If removing these endpoints leaves unused imports (e.g., `renumber_clusters` from `reid_ufm`), remove those import lines too.

**Step 4: Run active test suite**

```bash
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py test_disk_frame_cache.py test_extract_snippet_masks.py test_process_annotation.py test_seed_frame_pct.py test_lightweight_change.py test_reid_montage.py test_reid_keybindings.py -v --tb=short 2>&1 | tail -20
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add interview/routes.py
git commit -m "chore: remove 4 dead endpoints with no frontend callers

Removes /api/reid/renumber, /api/detect/training_status,
/api/detect/recall_strategy, /api/detect/crop/<id>/context."
```

---

### Task 5: Remove dead functions from detection.py

Three functions in detection.py are defined but never called.

**Files:**
- Modify: `interview/detection.py` — remove 3 dead functions + 1 dead env var

**Step 1: Read detection.py and locate the dead code**

Find:
- `uniform_indices()` (~lines 263-274) — never called
- `_get_embedding_mode()` (~lines 119-121) — never called, `EMBEDDING_MODE` used directly
- `_compute_iou_matrix()` (~lines 128-152) — never called, `nms_numpy()` computes IoU inline
- `EMBEDDING_MODE = os.environ.get(...)` (~line 116) — only read by `_get_embedding_mode()` which is dead

**Step 2: Verify no callers exist**

```bash
cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video
grep -rn "uniform_indices\|_get_embedding_mode\|_compute_iou_matrix" interview/ --include="*.py" | grep -v "def "
grep -rn "EMBEDDING_MODE" interview/ --include="*.py"
```
Expected: Only the definition lines match. `EMBEDDING_MODE` may appear in `run_embedding_background()` — if so, check whether that usage can be replaced with the literal string `"lightweight"` or if the env var is still useful.

**Step 3: Delete the dead functions**

Remove the function definitions. For `EMBEDDING_MODE`, if it's still read elsewhere, keep it; if only by `_get_embedding_mode()`, remove it.

**Step 4: Run active test suite**

```bash
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py -v --tb=short 2>&1 | tail -20
```
Expected: All tests PASS. If any detection test calls `uniform_indices` or `_compute_iou_matrix`, that test is testing dead code and should be removed too.

**Step 5: Commit**

```bash
git add interview/detection.py
git commit -m "chore: remove 3 dead functions from detection.py

Removes uniform_indices, _get_embedding_mode, _compute_iou_matrix —
all defined but never called in any production or test code path."
```

---

### Task 6: Remove dead CSS from style.css

~200 lines of CSS for UI components that were never built or were superseded.

**Files:**
- Modify: `interview/static/style.css` — remove dead CSS blocks

**Step 1: Read style.css and identify dead blocks**

Locate these dead CSS sections:
- **Pairwise ReID styles** (~lines 580-643): `.reid-frame-top`, `.reid-frame-bottom`, `.reid-comparison`, `.reid-crops`, `.reid-verdict`, `.btn-unsure`
- **Visual pipeline styles** (~lines 939-1051): `.visual-pipeline-stats`, `.visual-pipeline-stat`, `.visual-proposal-card`, `.visual-proposal-header`, `.visual-proposal-clusters`, etc.
- **Diagnostic advisory styles** (~lines 1054-1088): `.reid-diagnostics-panel`, `.reid-advisories`, `.reid-advisory-*`
- **Overflow badge** (~lines 894-912): `.reid-overflow-badge`
- **Recluster panel** (~lines 914-928): `.reid-recluster-panel`

**Step 2: Verify no JS references these classes**

```bash
cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video
grep -rn "reid-frame-top\|reid-comparison\|reid-verdict\|btn-unsure\|visual-pipeline\|visual-proposal\|reid-diagnostics\|reid-overflow-badge\|reid-recluster-panel" interview/static/ --include="*.js"
```
Expected: Zero matches

**Step 3: Delete the dead CSS blocks**

Remove each identified block from style.css.

**Step 4: Commit**

```bash
git add interview/static/style.css
git commit -m "chore: remove ~200 lines of dead CSS for pairwise ReID, visual pipeline, diagnostics"
```

---

### Task 7: Remove dead JS functions from app.js

~400 lines of JS for features that are never wired or are unreachable fallbacks.

**Files:**
- Modify: `interview/static/app.js` — remove dead functions

**Step 1: Read app.js and identify dead code**

Locate:
- `_onTrain()` (~lines 1095-1127) — MLP train callback, never called
- `_onRecall()` (~lines 1168-1215) — recall strategy modal, no backend endpoint
- Seeding fallback functions (~lines 1316-1517):
  - `_renderSeedingFallback()`
  - `_loadSeedConfig()`
  - `_generateSeeds()`
  - `_loadSeedPreview()`
  - `_uploadSeeds()`

**Step 2: Verify no callers**

```bash
grep -n "_onTrain\|_onRecall\|_renderSeedingFallback\|_loadSeedConfig\|_generateSeeds\|_loadSeedPreview\|_uploadSeeds" interview/static/app.js | grep -v "^\s*//"
```
Expected: Only the function definitions and possibly the fallback dispatch in `renderSeedingPhase()`. For the fallback dispatch, update it to remove the dead path (since `seeding_ui.js` is always loaded).

**Step 3: Delete the dead functions**

Remove each function body. For `renderSeedingPhase()`, if it has a fallback branch to `_renderSeedingFallback()`, remove that branch and keep only the `seeding_ui.js` delegation path.

Also remove any toolbar callback wiring for `onTrain` and `onRecall` if they're passed but never used by the toolbar renderer.

**Step 4: Commit**

```bash
git add interview/static/app.js
git commit -m "chore: remove dead JS functions (train, recall, seeding fallback)

Removes _onTrain, _onRecall, and 5 seeding fallback functions (~400 lines)
that were never called or unreachable in normal operation."
```

---

### Task 8: Remove diagnostic scripts and ufm_eval/

Diagnostic scripts that import from the now-deleted modules, plus the UFM evaluation suite.

**Files:**
- Delete: `grid_search_clustering.py` (imports from `reid_pipeline`)
- Delete: `inspect_clusters.py` (imports from `reid_pipeline`)
- Delete: `inspect_colors.py` (standalone diagnostic)
- Delete: `ufm_eval/` directory (entire evaluation suite)

**Step 1: Delete the files**

```bash
rm grid_search_clustering.py inspect_clusters.py inspect_colors.py
rm -rf ufm_eval/
```

**Step 2: Run active test suite**

```bash
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py -v --tb=short 2>&1 | tail -20
```
Expected: All tests PASS

**Step 3: Commit**

```bash
git add -u grid_search_clustering.py inspect_clusters.py inspect_colors.py
git add -u ufm_eval/
git commit -m "chore: remove diagnostic scripts and ufm_eval/ directory

grid_search_clustering.py, inspect_clusters.py import from deleted
reid_pipeline.py. ufm_eval/ was a one-time A/B test suite for UFM
vs DINOv3 clustering quality."
```

---

### Task 9: Remove renumber_clusters from reid_ufm.py

The `/api/reid/renumber` endpoint was removed in Task 4. The `renumber_clusters()` function in `reid_ufm.py` is now unreachable.

**Files:**
- Modify: `interview/reid_ufm.py` — remove `renumber_clusters()` function

**Step 1: Verify no remaining callers**

```bash
grep -rn "renumber_clusters" interview/ --include="*.py"
```
Expected: Only the definition in `reid_ufm.py`

**Step 2: Remove the function**

Delete `renumber_clusters()` from `reid_ufm.py`.

**Step 3: Run active test suite**

```bash
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py -v --tb=short 2>&1 | tail -20
```
Expected: All tests PASS

**Step 4: Commit**

```bash
git add interview/reid_ufm.py
git commit -m "chore: remove renumber_clusters() — endpoint was removed in earlier task"
```

---

### Task 10: Update CLAUDE.md and old plan docs

Remove stale sections from CLAUDE.md that reference deleted code.

**Files:**
- Modify: `CLAUDE.md` — remove/update stale sections
- Optionally delete: old plan docs in `docs/plans/` that reference deleted modules

**Step 1: Read CLAUDE.md and identify stale sections**

Sections to update:
- Key Files table: remove rows for `reid_phase.py` and `reid_pipeline.py`
- "ReID Constraint Architecture" section: remove (describes dead pairwise model)
- References to `reid_must_links`, `reid_cannot_links`, `reid_pairs`, `reid_phase_stage`
- Any references to "Phase 1/2/3" ReID workflow, pairwise verdicts, merge proposals

**Step 2: Update the Key Files table**

Remove rows for deleted files. Keep rows for active files.

**Step 3: Remove stale architecture sections**

Remove:
- "ReID Constraint Architecture" section
- Any "visual_reid_*" field references

**Step 4: Optionally mark old plan docs as superseded**

These plan docs reference deleted modules and are historical only:
- `docs/plans/2026-02-11-reid-adaptive-rounds.md`
- `docs/plans/2026-02-12-constraint-based-reid.md`
- `docs/plans/2026-02-12-three-phase-reid-elicitation.md`
- `docs/plans/2026-02-14-visual-reid-pipeline-design.md`
- `docs/plans/2026-02-14-visual-reid-pipeline-impl.md`

Either delete them or add a `> SUPERSEDED` header. Deleting is cleaner since git history preserves them.

**Step 5: Commit**

```bash
git add CLAUDE.md docs/plans/
git commit -m "docs: update CLAUDE.md and remove stale plan docs

Removes references to deleted reid_phase.py, reid_pipeline.py, and
the pairwise constraint/visual pipeline architecture. Deletes 5
superseded plan documents (preserved in git history)."
```

---

## Estimated Impact

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Python production code | ~8000 lines | ~4300 lines | -3700 |
| Python test code | ~8500 lines | ~4000 lines | -4500 |
| CSS | ~1100 lines | ~900 lines | -200 |
| JS | ~3200 lines | ~2800 lines | -400 |
| State fields | 20+ | 13 | -7 |
| Routes | ~20 | ~16 | -4 |
| **Total lines removed** | | | **~8800** |

## Post-Cleanup Verification

After all 10 tasks, run the full active test suite one final time:

```bash
cd /Users/triage/CascadeProjects/label-studio-ml-backend-old/label_studio_ml/examples/segment_anything_3_video
python -m pytest test_interview_detection.py test_tracking_fixes.py test_interview_state.py test_interview_cache.py test_interview_background.py test_disk_frame_cache.py test_extract_snippet_masks.py test_process_annotation.py test_seed_frame_pct.py test_lightweight_change.py test_reid_montage.py test_reid_keybindings.py test_cli.py test_api.py -v --tb=short
```

All tests must pass. The active path (detection rounds → UFM clustering → manual gallery → seeding) is completely untouched.
