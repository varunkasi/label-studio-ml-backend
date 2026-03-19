---
name: label-studio-labeling-ui
description: Build labeling/annotation UIs for reviewing and categorizing visual content with keyboard-driven binary choices. Use when the user wants to create a new labeling workflow with images, a question, binary choices with keyboard shortcuts, and a thumbnail grid. Trigger on requests like "create a labeling UI", "build a review interface", "I need to classify/categorize images".
---

# Labeling UI Skill

Generate keyboard-driven labeling UIs for reviewing visual content. Every workflow follows the same architecture; only the items, question, choices, and shortcuts change.

## The Pattern

Every labeling UI has these elements:

1. **Split panel** — left: main image viewer (maximized, no wasted space), right: labeler UI
2. **Main image viewer** — fills the left panel so users can spot small details. Minimal padding, `object-fit: contain`, dark background. This is the primary workspace.
3. **Entity crops** (optional) — at the top of the right panel, above the question. Small thumbnails of the entity being labeled (e.g., zoomed crop of a detected person). Useful when the main image shows a wide scene.
4. **Binary question + choices** — a question (e.g., "Is this a good crop?") with 2 choice buttons. Keyboard shortcuts use the first letter of each choice (e.g., Accept→`A`, Reject→`R`). Ergonomic exceptions allowed when first-letter keys cause hand strain.
5. **Auto-save + go-back** — each choice immediately saves to the backend (no explicit save button). Arrow ← navigates back so the user can change previous responses. Arrow → advances forward.
6. **Thumbnail grid** — below the question/choices. Color-coded borders per label state. Click to select. Scrollable.
7. **Stats toolbar** — counts per label, filter/sort controls, at the top of the left panel.

### Keyboard Conventions

| Key | Action |
|-----|--------|
| First letter of choice A | Apply choice A |
| First letter of choice B | Apply choice B |
| `→` | Next item |
| `←` | Previous item (change previous response) |

After labeling, auto-advance to the next unlabeled item. If no unlabeled items remain, show a toast.

## Parameters to Gather

Before generating code, ask the user for any missing parameters:

1. **What are we labeling?** — data source, item schema (images from an API, files on disk, video frames, etc.)
2. **The question** — what appears above the choices (e.g., "Is this a person?")
3. **Two choices** — label names and optional shortcut overrides (default: first letter). Example: "Person" (P) / "Not Person" (N)
4. **Image endpoint** — how to fetch the item's image (URL pattern or API path)
5. **Context viewer** — does the left panel show a different/larger image than the right panel entity crop? (e.g., full video frame vs. zoomed crop)
6. **API base path** — URL prefix for the Flask blueprint (e.g., `/review/api`)

## Workflow

Make a todo list for all tasks and work through them sequentially.

### Step 1: Read the Reference Implementation

Read these files to understand the existing patterns:

- `label_studio_ml/examples/segment_anything_3_video/interview/static/components.js` — reusable UI components: `SplitPanel`, `CropGrid`, `CropLabeler`, `FrameViewer`, `Toolbar`, `ProgressOverlay`, `Modal`
- `label_studio_ml/examples/segment_anything_3_video/interview/static/app.js` — application wiring: `API` client, `AppState` singleton, keyboard shortcuts (line ~2059), auto-advance `_advanceToNextPending()` (line ~1084), `renderDetection()` (line ~836), toast system
- `label_studio_ml/examples/segment_anything_3_video/interview/static/style.css` — dark theme CSS custom properties (`--bg-body`, `--bg-surface`, `--color-accepted`, `--color-rejected`, `--color-pending`, etc.) and all component styles
- `label_studio_ml/examples/segment_anything_3_video/interview/routes.py` — Flask blueprint pattern, REST endpoints, `_fix_passthrough` after_request handler
- `label_studio_ml/examples/segment_anything_3_video/interview/static/index.html` — minimal SPA HTML shell

### Step 2: Create the Flask Blueprint

Following `interview/routes.py`:

```python
from flask import Blueprint, jsonify, request, send_from_directory

bp = Blueprint("review", __name__, static_folder="static",
               static_url_path="", url_prefix="/review")

@bp.route("/")
def index():
    return send_from_directory(bp.static_folder, "index.html")

@bp.after_request
def _fix_passthrough(response):
    # Convert direct-passthrough to buffered (for label_studio_ml middleware)
    if response.direct_passthrough:
        response.data = response.get_data()
        response.direct_passthrough = False
    return response
```

Required endpoints:
- `GET /api/items?filter=&sort=&limit=` — list items with their current labels
- `POST /api/label` — `{"labels": {"item_id": "choice_a"}}` — batch label items (auto-save uses this with single items)
- `GET /api/item/<id>/image` — serve the item image
- `GET /api/stats` — return label counts

### Step 3: Create HTML Entry Point

Minimal SPA shell following `interview/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Review UI</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <nav class="top-nav">
    <span class="nav-title">Review UI</span>
  </nav>
  <main id="app"></main>
  <div id="toast-container" aria-live="polite"></div>
  <div id="modal-container" aria-hidden="true"></div>
  <script src="components.js" defer></script>
  <script src="app.js" defer></script>
</body>
</html>
```

### Step 4: Adapt Components

Copy and parameterize from the reference. Key adaptations:

**ItemLabeler** (from `CropLabeler` at `components.js:322-478`):

```javascript
class ItemLabeler {
    constructor(container, options) {
        this.container = container;
        this.choices = options.choices; // [{label, key, className}]
        this.question = options.question;
        this._callbacks = [];
        this._currentItem = null;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'item-preview';

        // Entity crop (optional, shown at top)
        this.cropEl = document.createElement('img');
        this.cropEl.className = 'entity-crop';
        this.cropEl.alt = 'Entity crop';
        this.cropEl.draggable = false;
        this.el.appendChild(this.cropEl);

        // Question
        this.questionEl = document.createElement('div');
        this.questionEl.className = 'labeler-question';
        this.questionEl.textContent = this.question;
        this.el.appendChild(this.questionEl);

        // Choice buttons
        this.actionsEl = document.createElement('div');
        this.actionsEl.className = 'labeler-actions';
        this.choices.forEach(choice => {
            const btn = document.createElement('button');
            btn.className = `btn ${choice.className}`;
            btn.textContent = choice.label;
            btn.addEventListener('click', () => this._fireChoice(choice.label));
            this.actionsEl.appendChild(btn);
        });
        this.el.appendChild(this.actionsEl);

        // Keyboard hints
        this.hintsEl = document.createElement('div');
        this.hintsEl.className = 'keyboard-hints';
        this.hintsEl.innerHTML = this.choices
            .map(c => `<span><kbd>${c.key.toUpperCase()}</kbd> ${c.label}</span>`)
            .join('') +
            '<span><kbd>&larr;</kbd><kbd>&rarr;</kbd> Navigate</span>';
        this.el.appendChild(this.hintsEl);

        this.container.appendChild(this.el);
    }

    showItem(item, sessionId) { /* update image + metadata */ }
    onChoice(callback) { this._callbacks.push(callback); }
    _fireChoice(label) { for (const cb of this._callbacks) cb(this._currentItem, label); }
}
```

**Reuse directly**: `SplitPanel`, `CropGrid` (rename items), `Toolbar`, `ProgressOverlay`, `Modal`.

**Left panel image viewer**: Must maximize space. Use `FrameViewer` pattern but ensure:
- `flex: 1` fills remaining height
- Background `#000`
- Image `object-fit: contain` with `max-width: 100%; max-height: 100%`
- Frame badge at bottom-left (item index or metadata)

### Step 5: Wire Auto-Save

Each choice immediately POSTs to the backend:

```javascript
async function _labelItem(item, label) {
    const result = await API.post('/label', {
        labels: { [item.id]: label }
    });
    item.label = label;
    grid.updateCardLabel(currentIndex, label);
    _advanceToNextPending();
    _renderToolbar();
}
```

Arrow ← navigates back — the user sees the previous item with its current label and can re-label it by pressing a choice key again (the new label overwrites via the same POST endpoint).

### Step 6: Wire Keyboard Shortcuts

Single global listener, guarded against input focus:

```javascript
document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

    // Choice keys (case-insensitive)
    for (const choice of CHOICES) {
        if (e.key.toLowerCase() === choice.key.toLowerCase()) {
            e.preventDefault();
            _labelCurrentItem(choice.label);
            return;
        }
    }

    // Navigation
    if (e.key === 'ArrowRight') { e.preventDefault(); _nextItem(); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); _prevItem(); }
});
```

### Step 7: Wire Auto-Advance

After labeling, find the next unlabeled item:

```javascript
function _advanceToNextPending() {
    const start = currentIndex;
    // Search forward
    for (let i = start + 1; i < items.length; i++) {
        if (items[i].label === 'pending') { _selectItem(i); return; }
    }
    // Wrap around
    for (let i = 0; i < start; i++) {
        if (items[i].label === 'pending') { _selectItem(i); return; }
    }
    // All labeled
    if (start + 1 < items.length) _selectItem(start + 1);
    else showToast('All items labeled!', 'success');
}
```

### Step 8: Add CSS

Import the existing dark theme and extend with workflow-specific styles:

```css
/* Import base theme from interview UI */
@import url('/interview/style.css');

/* Entity crop at top of right panel */
.entity-crop {
    max-width: 100%;
    max-height: 160px;
    border-radius: var(--radius-sm);
    object-fit: contain;
    margin: 8px auto;
    display: block;
}

/* Labeler question */
.labeler-question {
    text-align: center;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    padding: 8px 16px;
}

.labeler-actions {
    display: flex;
    gap: 10px;
    justify-content: center;
    padding: 8px 16px;
}

/* Add --color-* vars for new label states */
```

### Step 9: Register Blueprint

Add to the application entry point (e.g., `_wsgi.py` or wherever blueprints are registered):

```python
from .review.routes import bp as review_bp
app.register_blueprint(review_bp)
```

## Output Structure

When generating a new workflow, create this file structure:

```
<workflow_dir>/
  static/
    index.html        # SPA entry point
    app.js            # State, routing, keyboard shortcuts, auto-advance
    components.js     # Adapted UI components (ItemLabeler, ItemGrid, etc.)
    style.css         # @import base theme + workflow-specific styles
  routes.py           # Flask blueprint with REST endpoints
```

## Wrap Up

After generating all files, provide the user with:

1. **Shortcut table** — each choice mapped to its keyboard key
2. **API endpoints** — method, path, request body, response shape
3. **Registration instructions** — how to wire the blueprint into the app
4. **Test suggestion** — how to verify with a small batch of items
5. **File list** — all files created with their paths
