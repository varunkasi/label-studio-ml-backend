/* ==========================================================================
   SAM3 Interview UI - Reusable Components
   Vanilla JS UI components for the interview workflow SPA.
   No frameworks, no imports -- everything attaches to the global scope.
   ========================================================================== */

'use strict';

// ---------------------------------------------------------------------------
// SplitPanel
// ---------------------------------------------------------------------------

class SplitPanel {
    /**
     * Creates a two-column split layout inside the given container.
     * @param {HTMLElement} container - Parent element to render into.
     * @param {number} leftRatio - Fraction for the left panel width (0-1).
     */
    constructor(container, leftRatio = 0.6) {
        this.container = container;
        this.leftRatio = leftRatio;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'split-panel';

        this.leftEl = document.createElement('div');
        this.leftEl.className = 'panel-left';
        this.leftEl.style.flex = `0 0 ${this.leftRatio * 100}%`;

        this.rightEl = document.createElement('div');
        this.rightEl.className = 'panel-right';

        this.el.appendChild(this.leftEl);
        this.el.appendChild(this.rightEl);
        this.container.appendChild(this.el);
    }

    /** @returns {HTMLElement} The left panel element. */
    getLeft() {
        return this.leftEl;
    }

    /** @returns {HTMLElement} The right panel element. */
    getRight() {
        return this.rightEl;
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// FrameViewer
// ---------------------------------------------------------------------------

class FrameViewer {
    /**
     * Frame image viewer with optional canvas overlay for drawing boxes.
     * @param {HTMLElement} container - Element to render the viewer into.
     * @param {Object} options
     * @param {number} [options.width] - Video pixel width (for coordinate mapping).
     * @param {number} [options.height] - Video pixel height (for coordinate mapping).
     */
    constructor(container, options = {}) {
        this.container = container;
        this.videoWidth = options.width || 1920;
        this.videoHeight = options.height || 1080;
        this._drawMode = false;
        this._drawing = false;
        this._startX = 0;
        this._startY = 0;
        this._boxCallbacks = [];
        this._currentFrameIdx = -1;

        this._render();
        this._bindEvents();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'frame-viewer';

        // Main frame image
        this.img = document.createElement('img');
        this.img.className = 'frame-image';
        this.img.alt = 'Video frame';
        this.img.draggable = false;
        this.el.appendChild(this.img);

        // Canvas overlay for drawing boxes
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'frame-canvas-overlay';
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.cursor = 'default';
        this.el.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');

        // Frame indicator badge
        this.badge = document.createElement('div');
        this.badge.className = 'card-badge';
        this.badge.style.cssText =
            'position:absolute;bottom:8px;left:8px;font-size:0.75rem;' +
            'background:rgba(0,0,0,0.7);color:#fff;padding:2px 8px;' +
            'border-radius:4px;pointer-events:none;';
        this.badge.textContent = 'Frame --';
        this.el.appendChild(this.badge);

        this.container.appendChild(this.el);
    }

    _bindEvents() {
        // We use the canvas for mouse events in draw mode.
        this.canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this._onMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this._onMouseUp(e));

        // Resize observer to keep canvas size in sync
        this._resizeObserver = new ResizeObserver(() => this._syncCanvasSize());
        this._resizeObserver.observe(this.el);
    }

    _syncCanvasSize() {
        const rect = this.el.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    /**
     * Convert mouse event coordinates to pixel coordinates on the video frame.
     * Accounts for the image being object-fit:contain inside the viewer.
     */
    _eventToPixelCoords(e) {
        const rect = this.el.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Compute the rendered image rectangle (object-fit: contain)
        const containerW = rect.width;
        const containerH = rect.height;
        const imgAspect = this.videoWidth / this.videoHeight;
        const containerAspect = containerW / containerH;

        let renderW, renderH, offsetX, offsetY;
        if (containerAspect > imgAspect) {
            // Container is wider -- image is height-limited
            renderH = containerH;
            renderW = containerH * imgAspect;
            offsetX = (containerW - renderW) / 2;
            offsetY = 0;
        } else {
            // Container is taller -- image is width-limited
            renderW = containerW;
            renderH = containerW / imgAspect;
            offsetX = 0;
            offsetY = (containerH - renderH) / 2;
        }

        const px = ((mouseX - offsetX) / renderW) * this.videoWidth;
        const py = ((mouseY - offsetY) / renderH) * this.videoHeight;

        return {
            px: Math.max(0, Math.min(this.videoWidth, px)),
            py: Math.max(0, Math.min(this.videoHeight, py)),
            mouseX,
            mouseY,
            renderW,
            renderH,
            offsetX,
            offsetY,
        };
    }

    _onMouseDown(e) {
        if (!this._drawMode) return;
        e.preventDefault();
        const coords = this._eventToPixelCoords(e);
        this._drawing = true;
        this._startX = coords.px;
        this._startY = coords.py;
        this._startMouseX = coords.mouseX;
        this._startMouseY = coords.mouseY;
    }

    _onMouseMove(e) {
        if (!this._drawMode || !this._drawing) return;
        e.preventDefault();
        const coords = this._eventToPixelCoords(e);

        // Draw the rubber-band rectangle on the canvas
        this._syncCanvasSize();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = '#e94560';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([6, 3]);
        this.ctx.strokeRect(
            this._startMouseX,
            this._startMouseY,
            coords.mouseX - this._startMouseX,
            coords.mouseY - this._startMouseY
        );
        this.ctx.setLineDash([]);
    }

    _onMouseUp(e) {
        if (!this._drawMode || !this._drawing) return;
        this._drawing = false;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const coords = this._eventToPixelCoords(e);
        const x1 = Math.min(this._startX, coords.px);
        const y1 = Math.min(this._startY, coords.py);
        const x2 = Math.max(this._startX, coords.px);
        const y2 = Math.max(this._startY, coords.py);

        // Minimum box size: 8px in video coords
        if (x2 - x1 < 8 || y2 - y1 < 8) return;

        const box = { x1, y1, x2, y2 };
        for (const cb of this._boxCallbacks) {
            cb(box);
        }
    }

    /**
     * Load and display an annotated frame from the backend.
     * @param {number} frameIdx - 0-based frame index.
     * @param {string} sessionId - Current session ID.
     * @param {boolean} [annotated=true] - Whether to load the annotated version.
     * @param {string|null} [highlightCropId=null] - Crop ID to highlight with thick border.
     */
    loadFrame(frameIdx, sessionId, annotated = true, highlightCropId = null) {
        this._currentFrameIdx = frameIdx;
        if (annotated) {
            let path = `/interview/api/detect/frame/${frameIdx}/annotated?session_id=${sessionId}`;
            if (highlightCropId) {
                path += `&highlight=${encodeURIComponent(highlightCropId)}`;
            }
            this.img.src = path;
        } else {
            this.img.src = `/interview/api/detect/frame/${frameIdx}?session_id=${sessionId}`;
        }
        this.badge.textContent = `Frame ${frameIdx}`;
    }

    /** Reload the currently displayed frame. */
    reload(sessionId, highlightCropId = null) {
        if (this._currentFrameIdx >= 0 && sessionId) {
            this.loadFrame(this._currentFrameIdx, sessionId, true, highlightCropId);
        }
    }

    /** @returns {number} The currently displayed frame index. */
    getCurrentFrame() {
        return this._currentFrameIdx;
    }

    /** Enable draw mode -- canvas becomes interactive for drawing boxes. */
    enableDrawMode() {
        this._drawMode = true;
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.style.cursor = 'crosshair';
    }

    /** Disable draw mode -- canvas stops capturing mouse events. */
    disableDrawMode() {
        this._drawMode = false;
        this._drawing = false;
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.cursor = 'default';
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /** @returns {boolean} Whether draw mode is currently active. */
    isDrawMode() {
        return this._drawMode;
    }

    /**
     * Register a callback for when a box is drawn.
     * @param {Function} callback - Receives {x1, y1, x2, y2} in pixel coords.
     */
    onBoxDrawn(callback) {
        this._boxCallbacks.push(callback);
    }

    /**
     * Update the video dimensions used for coordinate mapping.
     * @param {number} width
     * @param {number} height
     */
    setVideoDimensions(width, height) {
        this.videoWidth = width;
        this.videoHeight = height;
    }

    destroy() {
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
        }
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// CropLabeler
// ---------------------------------------------------------------------------

class CropLabeler {
    /**
     * Right-panel component: displays a zoomed crop image with
     * accept / reject buttons and metadata.
     * @param {HTMLElement} container
     */
    constructor(container) {
        this.container = container;
        this._acceptCallbacks = [];
        this._rejectCallbacks = [];
        this._skipCallbacks = [];
        this._currentCrop = null;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'crop-preview';

        // Crop image
        this.img = document.createElement('img');
        this.img.alt = 'Crop preview';
        this.img.draggable = false;
        this.el.appendChild(this.img);

        // Metadata block
        this.metaEl = document.createElement('div');
        this.metaEl.className = 'crop-meta';
        this.metaEl.style.cssText =
            'font-size:0.75rem;color:var(--text-secondary);text-align:center;';
        this.el.appendChild(this.metaEl);

        // Action buttons
        this.actionsEl = document.createElement('div');
        this.actionsEl.className = 'crop-actions';

        this.rejectBtn = document.createElement('button');
        this.rejectBtn.className = 'btn btn-reject';
        this.rejectBtn.textContent = 'Reject';
        this.rejectBtn.addEventListener('click', () => this._fireReject());

        this.skipBtn = document.createElement('button');
        this.skipBtn.className = 'btn btn-skip';
        this.skipBtn.textContent = 'Skip';
        this.skipBtn.style.cssText = 'background:var(--text-secondary, #888);';
        this.skipBtn.addEventListener('click', () => this._fireSkip());

        this.acceptBtn = document.createElement('button');
        this.acceptBtn.className = 'btn btn-accept';
        this.acceptBtn.textContent = 'Accept';
        this.acceptBtn.addEventListener('click', () => this._fireAccept());

        this.actionsEl.appendChild(this.rejectBtn);
        this.actionsEl.appendChild(this.skipBtn);
        this.actionsEl.appendChild(this.acceptBtn);
        this.el.appendChild(this.actionsEl);

        // Keyboard hints
        this.hintsEl = document.createElement('div');
        this.hintsEl.className = 'keyboard-hints';
        this.hintsEl.innerHTML =
            '<span><kbd>Enter</kbd> Accept</span>' +
            '<span><kbd>S</kbd> Skip</span>' +
            '<span><kbd>Backspace</kbd> Reject</span>' +
            '<span><kbd>&larr;</kbd><kbd>&rarr;</kbd> Navigate</span>';
        this.el.appendChild(this.hintsEl);

        this.container.appendChild(this.el);
    }

    /**
     * Display a crop image and its metadata.
     * @param {Object} crop - Crop data object from the API.
     * @param {string} sessionId - Current session ID.
     */
    showCrop(crop, sessionId) {
        this._currentCrop = crop;
        this.img.src = `/interview/api/detect/crop/${crop.crop_id}/image?session_id=${sessionId}`;

        const labelBadge = `<span style="color:var(--color-${crop.label})">${crop.label}</span>`;
        this.metaEl.innerHTML =
            `Frame ${crop.frame_idx} | Score ${crop.score.toFixed(2)} | ` +
            `Source: ${crop.source} | ${labelBadge}` +
            (crop.cluster_id != null ? ` | Cluster ${crop.cluster_id}` : '') +
            (crop.uncertainty != null ? ` | Unc ${crop.uncertainty.toFixed(2)}` : '');

        // Visually disable buttons if already labeled
        this.acceptBtn.disabled = crop.label === 'accepted';
        this.rejectBtn.disabled = crop.label === 'rejected';
        this.skipBtn.disabled = crop.label === 'skipped';
    }

    /** Clear the crop preview. */
    clear() {
        this._currentCrop = null;
        this.img.src = '';
        this.metaEl.textContent = 'No crop selected';
        this.acceptBtn.disabled = true;
        this.rejectBtn.disabled = true;
    }

    /** @returns {Object|null} The currently displayed crop. */
    getCurrentCrop() {
        return this._currentCrop;
    }

    /**
     * Register a callback for when the Accept button is clicked.
     * @param {Function} callback - Receives the current crop object.
     */
    onAccept(callback) {
        this._acceptCallbacks.push(callback);
    }

    /**
     * Register a callback for when the Reject button is clicked.
     * @param {Function} callback - Receives the current crop object.
     */
    onReject(callback) {
        this._rejectCallbacks.push(callback);
    }

    /**
     * Register a callback for when the Skip button is clicked.
     * @param {Function} callback - Receives the current crop object.
     */
    onSkip(callback) {
        this._skipCallbacks.push(callback);
    }

    _fireAccept() {
        if (!this._currentCrop) return;
        for (const cb of this._acceptCallbacks) {
            cb(this._currentCrop);
        }
    }

    _fireReject() {
        if (!this._currentCrop) return;
        for (const cb of this._rejectCallbacks) {
            cb(this._currentCrop);
        }
    }

    _fireSkip() {
        if (!this._currentCrop) return;
        for (const cb of this._skipCallbacks) {
            cb(this._currentCrop);
        }
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// CropGrid
// ---------------------------------------------------------------------------

class CropGrid {
    /**
     * Scrollable grid of crop thumbnails with color-coded borders.
     * @param {HTMLElement} container
     */
    constructor(container) {
        this.container = container;
        this._selectCallbacks = [];
        this._selectedIndex = -1;
        this._crops = [];
        this._sessionId = null;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'crop-grid';
        this.container.appendChild(this.el);
    }

    /**
     * Render a list of crop thumbnails.
     * @param {Array} crops - Array of crop data objects.
     * @param {string} sessionId
     */
    render(crops, sessionId) {
        this._crops = crops;
        this._sessionId = sessionId;
        this.el.innerHTML = '';

        crops.forEach((crop, index) => {
            const card = document.createElement('div');
            card.className = 'crop-card';
            card.classList.add(crop.label);

            if (crop.source === 'human_drawn') card.classList.add('human');

            if (index === this._selectedIndex) card.classList.add('selected');

            const img = document.createElement('img');
            img.src = `/interview/api/detect/crop/${crop.crop_id}/image?session_id=${sessionId}`;
            img.alt = `Crop ${crop.crop_id}`;
            img.loading = 'lazy';
            card.appendChild(img);

            // Badge showing uncertainty or cluster
            const badge = document.createElement('span');
            badge.className = 'card-badge';
            if (crop.uncertainty != null && crop.uncertainty > 0) {
                badge.textContent = crop.uncertainty.toFixed(1);
            } else if (crop.cluster_id != null) {
                badge.textContent = `C${crop.cluster_id}`;
            }
            card.appendChild(badge);

            card.addEventListener('click', () => {
                this.select(index);
                for (const cb of this._selectCallbacks) {
                    cb(crop, index);
                }
            });

            this.el.appendChild(card);
        });
    }

    /**
     * Visually select a crop card by index.
     * @param {number} index
     */
    select(index) {
        if (index < 0 || index >= this._crops.length) return;

        // Remove previous selection
        const prev = this.el.querySelector('.crop-card.selected');
        if (prev) prev.classList.remove('selected');

        // Apply new selection
        this._selectedIndex = index;
        const cards = this.el.querySelectorAll('.crop-card');
        if (cards[index]) {
            cards[index].classList.add('selected');
            cards[index].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    /** @returns {number} Currently selected index. */
    getSelectedIndex() {
        return this._selectedIndex;
    }

    /** @returns {Object|null} Currently selected crop data. */
    getSelectedCrop() {
        if (this._selectedIndex < 0 || this._selectedIndex >= this._crops.length) {
            return null;
        }
        return this._crops[this._selectedIndex];
    }

    /** @returns {Array} All crops in the grid. */
    getCrops() {
        return this._crops;
    }

    /**
     * Update a single crop card's label status in place.
     * @param {number} index
     * @param {string} label - 'accepted', 'rejected', 'skipped', or 'pending'.
     */
    updateCardLabel(index, label) {
        if (index < 0 || index >= this._crops.length) return;
        this._crops[index].label = label;
        const cards = this.el.querySelectorAll('.crop-card');
        if (cards[index]) {
            cards[index].classList.remove('accepted', 'rejected', 'pending', 'skipped');
            cards[index].classList.add(label);
        }
    }

    /**
     * Register a callback for when a crop card is clicked.
     * @param {Function} callback - Receives (crop, index).
     */
    onCropSelect(callback) {
        this._selectCallbacks.push(callback);
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// ProgressOverlay
// ---------------------------------------------------------------------------

class ProgressOverlay {
    /**
     * Full-area overlay with animated progress bar, step text, and percentage.
     * @param {HTMLElement} container - Element to overlay.
     */
    constructor(container) {
        this.container = container;
        this._visible = false;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'progress-overlay';
        this.el.style.cssText =
            'position:absolute;inset:0;display:flex;flex-direction:column;' +
            'align-items:center;justify-content:center;gap:16px;' +
            'background:rgba(26,26,46,0.92);z-index:100;pointer-events:none;' +
            'opacity:0;transition:opacity 250ms ease;';

        this.stepEl = document.createElement('div');
        this.stepEl.style.cssText =
            'font-size:0.9rem;color:var(--text-primary);font-weight:600;';
        this.stepEl.textContent = 'Processing...';
        this.el.appendChild(this.stepEl);

        // Progress bar container
        const barWrapper = document.createElement('div');
        barWrapper.className = 'progress-bar-wrapper';
        barWrapper.style.cssText = 'width:320px;padding:0;background:transparent;border:none;';

        const track = document.createElement('div');
        track.className = 'progress-bar-track';

        this.fill = document.createElement('div');
        this.fill.className = 'progress-bar-fill';
        this.fill.style.width = '0%';

        track.appendChild(this.fill);
        barWrapper.appendChild(track);
        this.el.appendChild(barWrapper);

        // Percentage text
        this.pctEl = document.createElement('div');
        this.pctEl.style.cssText = 'font-size:0.8rem;color:var(--text-muted);';
        this.pctEl.textContent = '';
        this.el.appendChild(this.pctEl);

        // Ensure container is positioned for overlay
        const containerPos = getComputedStyle(this.container).position;
        if (containerPos === 'static') {
            this.container.style.position = 'relative';
        }
        this.container.appendChild(this.el);
    }

    /**
     * Show the overlay with a step description and percentage.
     * @param {string} step - Description of the current operation.
     * @param {number} percent - Progress percentage (0-100). Use -1 for indeterminate.
     */
    show(step, percent) {
        this._visible = true;
        this.el.style.opacity = '1';
        this.el.style.pointerEvents = 'auto';
        this.stepEl.textContent = step || 'Processing...';

        if (percent < 0) {
            // Indeterminate
            this.fill.classList.add('indeterminate');
            this.fill.style.width = '';
            this.pctEl.textContent = '';
        } else {
            this.fill.classList.remove('indeterminate');
            this.fill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
            this.pctEl.textContent = `${Math.round(percent)}%`;
        }
    }

    /** Hide the overlay. */
    hide() {
        this._visible = false;
        this.el.style.opacity = '0';
        this.el.style.pointerEvents = 'none';
    }

    /** @returns {boolean} Whether the overlay is currently visible. */
    isVisible() {
        return this._visible;
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

class Toolbar {
    /**
     * Top toolbar for the detection/classification phase.
     * @param {HTMLElement} container
     */
    constructor(container) {
        this.container = container;
        this._render();
    }

    _render() {
        this.el = document.createElement('div');
        this.el.className = 'toolbar';
        this.container.appendChild(this.el);
    }

    /**
     * Render the toolbar with the given options.
     * @param {Object} options
     * @param {boolean} options.drawMode - Whether draw mode is active.
     * @param {Function} options.onDrawToggle - Callback when draw toggle is clicked.
     * @param {Function} [options.onPrevFrame] - Navigate to previous sampled frame.
     * @param {Function} [options.onNextFrame] - Navigate to next sampled frame.
     * @param {Function} [options.onAdvancePhase] - Advance to next workflow phase.
     * @param {Object} [options.stats] - Session stats for display.
     * @param {string} [options.sortBy] - Current sort mode.
     * @param {Function} [options.onSortChange] - Callback when sort changes.
     * @param {string} [options.filterLabel] - Current filter label.
     * @param {Function} [options.onFilterChange] - Callback when filter changes.
     */
    render(options = {}) {
        this.el.innerHTML = '';

        // Frame navigation
        const frameNav = document.createElement('div');
        frameNav.style.cssText = 'display:flex;align-items:center;gap:6px;';

        const prevBtn = document.createElement('button');
        prevBtn.className = 'btn btn-ghost btn-small';
        prevBtn.textContent = 'Prev Frame';
        prevBtn.addEventListener('click', () => {
            if (options.onPrevFrame) options.onPrevFrame();
        });
        frameNav.appendChild(prevBtn);

        const nextBtn = document.createElement('button');
        nextBtn.className = 'btn btn-ghost btn-small';
        nextBtn.textContent = 'Next Frame';
        nextBtn.addEventListener('click', () => {
            if (options.onNextFrame) options.onNextFrame();
        });
        frameNav.appendChild(nextBtn);

        this.el.appendChild(frameNav);

        // Separator
        this.el.appendChild(this._separator());

        // Draw mode toggle
        const drawBtn = document.createElement('button');
        drawBtn.className = 'btn btn-secondary btn-small';
        if (options.drawMode) drawBtn.classList.add('active');
        drawBtn.textContent = options.drawMode ? 'Draw Mode ON' : 'Draw Mode';
        if (options.disableDrawToggle) {
            drawBtn.disabled = true;
            drawBtn.title = 'Use reject-review Draw ON / Save crop controls';
        }
        drawBtn.addEventListener('click', () => {
            if (options.onDrawToggle) options.onDrawToggle();
        });
        this.el.appendChild(drawBtn);

        // Separator
        this.el.appendChild(this._separator());

        // Round badge
        const roundBadge = document.createElement('span');
        roundBadge.className = 'round-badge';
        roundBadge.textContent = `Round ${options.currentRound || 1}`;
        this.el.appendChild(roundBadge);

        // Next Round button (scores with k-NN + detects new frames)
        const nextRoundBtn = document.createElement('button');
        nextRoundBtn.className = 'btn btn-primary btn-small';
        nextRoundBtn.textContent = 'Next Round';
        nextRoundBtn.title = 'Score crops with k-NN, then detect on new frames';
        nextRoundBtn.addEventListener('click', () => {
            if (options.onNextRound) options.onNextRound();
        });
        this.el.appendChild(nextRoundBtn);

        // Separator
        this.el.appendChild(this._separator());

        // Sort select
        const sortGroup = document.createElement('div');
        sortGroup.style.cssText = 'display:flex;align-items:center;gap:4px;';
        const sortLabel = document.createElement('span');
        sortLabel.style.cssText = 'font-size:0.75rem;color:var(--text-muted);';
        sortLabel.textContent = 'Sort:';
        sortGroup.appendChild(sortLabel);

        const sortSelect = document.createElement('select');
        sortSelect.style.cssText =
            'padding:3px 6px;font-size:0.75rem;background:var(--bg-body);' +
            'color:var(--text-primary);border:1px solid var(--border-default);' +
            'border-radius:var(--radius-sm);';
        ['uncertainty', 'cluster', 'frame'].forEach((s) => {
            const opt = document.createElement('option');
            opt.value = s;
            opt.textContent = s.charAt(0).toUpperCase() + s.slice(1);
            if (s === (options.sortBy || 'uncertainty')) opt.selected = true;
            sortSelect.appendChild(opt);
        });
        sortSelect.addEventListener('change', () => {
            if (options.onSortChange) options.onSortChange(sortSelect.value);
        });
        sortGroup.appendChild(sortSelect);
        this.el.appendChild(sortGroup);

        // Filter select
        const filterGroup = document.createElement('div');
        filterGroup.style.cssText = 'display:flex;align-items:center;gap:4px;';
        const filterLabel = document.createElement('span');
        filterLabel.style.cssText = 'font-size:0.75rem;color:var(--text-muted);';
        filterLabel.textContent = 'Filter:';
        filterGroup.appendChild(filterLabel);

        const filterSelect = document.createElement('select');
        filterSelect.style.cssText =
            'padding:3px 6px;font-size:0.75rem;background:var(--bg-body);' +
            'color:var(--text-primary);border:1px solid var(--border-default);' +
            'border-radius:var(--radius-sm);';
        ['all', 'pending', 'accepted', 'rejected', 'corrected', 'skipped'].forEach((f) => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f.charAt(0).toUpperCase() + f.slice(1);
            if (f === (options.filterLabel || 'all')) opt.selected = true;
            filterSelect.appendChild(opt);
        });
        filterSelect.addEventListener('change', () => {
            if (options.onFilterChange) options.onFilterChange(filterSelect.value);
        });
        filterGroup.appendChild(filterSelect);
        this.el.appendChild(filterGroup);

        // Separator
        this.el.appendChild(this._separator());

        // Stats display
        if (options.stats) {
            const statsEl = document.createElement('span');
            statsEl.style.cssText =
                'font-size:0.7rem;color:var(--text-muted);margin-left:auto;';
            const s = options.stats;
            statsEl.textContent =
                `${s.accepted || 0} accepted | ${s.rejected || 0} rejected | ` +
                `${s.corrected_total || 0} corrected | ` +
                `${s.skipped || 0} skipped | ` +
                `${s.pending || 0} pending | ${s.total_crops || 0} total`;
            this.el.appendChild(statsEl);
        }

        // Accuracy trend widget (validation history)
        const valHistory = (options.stats && options.stats.validation_history) || [];
        if (valHistory.length > 0) {
            this.el.appendChild(this._separator());
            this.el.appendChild(this._buildAccuracyTrend(valHistory));
        }

        // Advance to ReID button (available once at least 1 round is done)
        if (options.onAdvancePhase && (options.roundsCompleted || 0) >= 1) {
            const advBtn = document.createElement('button');
            advBtn.className = 'btn btn-secondary btn-small';
            advBtn.style.marginLeft = '8px';
            advBtn.textContent = 'Finish Labeling → ReID';
            advBtn.addEventListener('click', () => options.onAdvancePhase());
            this.el.appendChild(advBtn);
        }
    }

    _separator() {
        const sep = document.createElement('div');
        sep.style.cssText =
            'width:1px;height:20px;background:var(--border-default);flex-shrink:0;';
        return sep;
    }

    /**
     * Build a compact accuracy trend widget (sparkline bar chart + value).
     * @param {Array<{round: number, val_accuracy: number}>} history
     * @returns {HTMLElement}
     */
    _buildAccuracyTrend(history) {
        const wrap = document.createElement('div');
        wrap.style.cssText = 'display:flex;align-items:center;gap:6px;';

        const label = document.createElement('span');
        label.className = 'accuracy-trend-label';
        label.textContent = 'Val:';
        wrap.appendChild(label);

        const bars = document.createElement('div');
        bars.className = 'accuracy-trend';

        const maxH = 24;
        history.forEach((entry) => {
            const acc = entry.val_accuracy || 0;
            const pct = Math.round(acc * 100);
            const bar = document.createElement('div');
            bar.className = 'accuracy-trend-bar';
            if (pct < 60) bar.classList.add('low');
            else if (pct < 80) bar.classList.add('mid');
            else bar.classList.add('high');
            bar.style.height = Math.max(3, (acc * maxH)) + 'px';
            bar.title = `Round ${entry.round}: ${pct}%`;
            bars.appendChild(bar);
        });
        wrap.appendChild(bars);

        const latest = history[history.length - 1];
        const value = document.createElement('span');
        value.className = 'accuracy-trend-value';
        value.textContent = Math.round((latest.val_accuracy || 0) * 100) + '%';
        wrap.appendChild(value);

        return wrap;
    }

    destroy() {
        if (this.el && this.el.parentNode) {
            this.el.parentNode.removeChild(this.el);
        }
    }
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

class Modal {
    /**
     * Show a confirmation dialog. Returns a Promise that resolves
     * to true (confirmed) or false (cancelled).
     * @param {string} title
     * @param {string} message
     * @returns {Promise<boolean>}
     */
    static confirm(title, message) {
        return new Promise((resolve) => {
            const container = document.getElementById('modal-container');
            container.innerHTML = '';
            container.setAttribute('aria-hidden', 'false');

            const box = document.createElement('div');
            box.className = 'modal-box';

            const h3 = document.createElement('h3');
            h3.textContent = title;
            box.appendChild(h3);

            const p = document.createElement('p');
            p.textContent = message;
            box.appendChild(p);

            const actions = document.createElement('div');
            actions.className = 'modal-actions';

            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-ghost';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.addEventListener('click', () => {
                Modal._close(container);
                resolve(false);
            });

            const confirmBtn = document.createElement('button');
            confirmBtn.className = 'btn btn-primary';
            confirmBtn.textContent = 'Confirm';
            confirmBtn.addEventListener('click', () => {
                Modal._close(container);
                resolve(true);
            });

            actions.appendChild(cancelBtn);
            actions.appendChild(confirmBtn);
            box.appendChild(actions);
            container.appendChild(box);

            // Close on backdrop click
            container.addEventListener('click', function _backdrop(e) {
                if (e.target === container) {
                    container.removeEventListener('click', _backdrop);
                    Modal._close(container);
                    resolve(false);
                }
            });

            // Close on Escape
            const _esc = (e) => {
                if (e.key === 'Escape') {
                    document.removeEventListener('keydown', _esc);
                    Modal._close(container);
                    resolve(false);
                }
            };
            document.addEventListener('keydown', _esc);

            // Focus the confirm button
            requestAnimationFrame(() => confirmBtn.focus());
        });
    }

    /**
     * Show a generic modal with arbitrary HTML content.
     * Returns a close function.
     * @param {string} title
     * @param {string|HTMLElement} content - HTML string or DOM element.
     * @returns {Function} A function to call to close the modal.
     */
    static show(title, content) {
        const container = document.getElementById('modal-container');
        container.innerHTML = '';
        container.setAttribute('aria-hidden', 'false');

        const box = document.createElement('div');
        box.className = 'modal-box';

        const h3 = document.createElement('h3');
        h3.textContent = title;
        box.appendChild(h3);

        const body = document.createElement('div');
        body.style.marginBottom = '16px';
        if (typeof content === 'string') {
            body.innerHTML = content;
        } else if (content instanceof HTMLElement) {
            body.appendChild(content);
        }
        box.appendChild(body);

        const actions = document.createElement('div');
        actions.className = 'modal-actions';

        const closeBtn = document.createElement('button');
        closeBtn.className = 'btn btn-ghost';
        closeBtn.textContent = 'Close';
        closeBtn.addEventListener('click', () => closeFn());
        actions.appendChild(closeBtn);
        box.appendChild(actions);

        container.appendChild(box);

        const closeFn = () => Modal._close(container);

        container.addEventListener('click', function _backdrop(e) {
            if (e.target === container) {
                container.removeEventListener('click', _backdrop);
                closeFn();
            }
        });

        const _esc = (e) => {
            if (e.key === 'Escape') {
                document.removeEventListener('keydown', _esc);
                closeFn();
            }
        };
        document.addEventListener('keydown', _esc);

        return closeFn;
    }

    /** @private Close the modal container. */
    static _close(container) {
        container.setAttribute('aria-hidden', 'true');
        // Clear content after transition
        setTimeout(() => {
            container.innerHTML = '';
        }, 300);
    }
}

// ---------------------------------------------------------------------------
// BoxAdjuster
// ---------------------------------------------------------------------------

class BoxAdjuster {
    /**
     * Overlay component that renders 8 resize handles on an existing bounding
     * box and allows the user to drag them to adjust the box.  Works with the
     * FrameViewer's canvas overlay for drawing and its coordinate helpers for
     * pixel <-> screen mapping.
     *
     * @param {FrameViewer} frameViewer - The FrameViewer instance to overlay on.
     */
    constructor(frameViewer) {
        this._fv = frameViewer;
        this._active = false;
        this._box = null;               // {x1, y1, x2, y2} in video-pixel coords
        this._callbacks = [];            // onBoxChanged listeners
        this._dragging = false;
        this._dragHandle = null;         // which handle is being dragged
        this._destroyed = false;

        // Minimum box dimension in video-pixel coords
        this._MIN_SIZE = 16;

        // Handle visual size (screen px) and hit radius (screen px)
        this._HANDLE_SIZE = 8;
        this._HIT_RADIUS = 12;

        // Handle definitions: name -> which edges it controls and cursor
        this._HANDLES = [
            { name: 'tl', edgesX: 'x1', edgesY: 'y1', cursor: 'nwse-resize' },
            { name: 'tc', edgesX: null,  edgesY: 'y1', cursor: 'ns-resize'   },
            { name: 'tr', edgesX: 'x2', edgesY: 'y1', cursor: 'nesw-resize' },
            { name: 'ml', edgesX: 'x1', edgesY: null,  cursor: 'ew-resize'   },
            { name: 'mr', edgesX: 'x2', edgesY: null,  cursor: 'ew-resize'   },
            { name: 'bl', edgesX: 'x1', edgesY: 'y2', cursor: 'nesw-resize' },
            { name: 'bc', edgesX: null,  edgesY: 'y2', cursor: 'ns-resize'   },
            { name: 'br', edgesX: 'x2', edgesY: 'y2', cursor: 'nwse-resize' },
        ];

        // Bind event handlers so we can add/remove them cleanly
        this._onPointerDown = this._onPointerDown.bind(this);
        this._onPointerMove = this._onPointerMove.bind(this);
        this._onPointerUp   = this._onPointerUp.bind(this);
    }

    // -- Public API -----------------------------------------------------------

    /**
     * Activate box adjustment mode with the given bounding box.
     * @param {Object} xyxy - {x1, y1, x2, y2} in video-pixel coords.
     */
    activate(xyxy) {
        if (this._destroyed) return;
        this._box = {
            x1: Math.min(xyxy.x1, xyxy.x2),
            y1: Math.min(xyxy.y1, xyxy.y2),
            x2: Math.max(xyxy.x1, xyxy.x2),
            y2: Math.max(xyxy.y1, xyxy.y2),
        };
        this._active = true;
        this._dragging = false;
        this._dragHandle = null;

        // Enable pointer events on the canvas overlay
        this._fv.el.classList.add('box-adjuster-active');
        this._fv.canvas.style.cursor = 'default';

        // Attach listeners
        this._fv.canvas.addEventListener('pointerdown', this._onPointerDown);
        this._fv.canvas.addEventListener('pointermove', this._onPointerMove);
        this._fv.canvas.addEventListener('pointerup',   this._onPointerUp);
        this._fv.canvas.addEventListener('pointerleave', this._onPointerUp);

        this._draw();
    }

    /** Deactivate box adjustment, remove listeners, clear canvas. */
    deactivate() {
        if (!this._active) return;
        this._active = false;
        this._dragging = false;
        this._dragHandle = null;

        this._fv.el.classList.remove('box-adjuster-active');
        this._fv.canvas.style.cursor = 'default';

        // Remove listeners
        this._fv.canvas.removeEventListener('pointerdown', this._onPointerDown);
        this._fv.canvas.removeEventListener('pointermove', this._onPointerMove);
        this._fv.canvas.removeEventListener('pointerup',   this._onPointerUp);
        this._fv.canvas.removeEventListener('pointerleave', this._onPointerUp);

        // Clear canvas
        this._fv._syncCanvasSize();
        this._fv.ctx.clearRect(0, 0, this._fv.canvas.width, this._fv.canvas.height);
    }

    /** @returns {boolean} Whether box adjustment is currently active. */
    isActive() {
        return this._active;
    }

    /** @returns {Object|null} Current adjusted {x1, y1, x2, y2} or null. */
    getBox() {
        if (!this._active || !this._box) return null;
        return { x1: this._box.x1, y1: this._box.y1, x2: this._box.x2, y2: this._box.y2 };
    }

    /**
     * Register a callback for when the box changes (on handle release).
     * @param {Function} callback - Receives {x1, y1, x2, y2} in video-pixel coords.
     */
    onBoxChanged(callback) {
        this._callbacks.push(callback);
    }

    /** Permanently destroy the adjuster, cleaning up all state. */
    destroy() {
        this.deactivate();
        this._callbacks = [];
        this._destroyed = true;
    }

    // -- Coordinate Helpers ---------------------------------------------------

    /**
     * Convert video-pixel coords to canvas (screen) coords.
     * Inverse of FrameViewer._eventToPixelCoords.
     */
    _pixelToCanvas(px, py) {
        const rect = this._fv.el.getBoundingClientRect();
        const containerW = rect.width;
        const containerH = rect.height;
        const imgAspect = this._fv.videoWidth / this._fv.videoHeight;
        const containerAspect = containerW / containerH;

        let renderW, renderH, offsetX, offsetY;
        if (containerAspect > imgAspect) {
            renderH = containerH;
            renderW = containerH * imgAspect;
            offsetX = (containerW - renderW) / 2;
            offsetY = 0;
        } else {
            renderW = containerW;
            renderH = containerW / imgAspect;
            offsetX = 0;
            offsetY = (containerH - renderH) / 2;
        }

        return {
            cx: offsetX + (px / this._fv.videoWidth) * renderW,
            cy: offsetY + (py / this._fv.videoHeight) * renderH,
        };
    }

    // -- Handle Geometry ------------------------------------------------------

    /**
     * Compute the canvas-space position for each of the 8 handles.
     * @returns {Array<{def: Object, cx: number, cy: number}>}
     */
    _handlePositions() {
        const b = this._box;
        const positions = [];

        const pixelPoints = [
            { def: this._HANDLES[0], px: b.x1,                  py: b.y1 },                  // tl
            { def: this._HANDLES[1], px: (b.x1 + b.x2) / 2,    py: b.y1 },                  // tc
            { def: this._HANDLES[2], px: b.x2,                  py: b.y1 },                  // tr
            { def: this._HANDLES[3], px: b.x1,                  py: (b.y1 + b.y2) / 2 },    // ml
            { def: this._HANDLES[4], px: b.x2,                  py: (b.y1 + b.y2) / 2 },    // mr
            { def: this._HANDLES[5], px: b.x1,                  py: b.y2 },                  // bl
            { def: this._HANDLES[6], px: (b.x1 + b.x2) / 2,    py: b.y2 },                  // bc
            { def: this._HANDLES[7], px: b.x2,                  py: b.y2 },                  // br
        ];

        for (const pt of pixelPoints) {
            const c = this._pixelToCanvas(pt.px, pt.py);
            positions.push({ def: pt.def, cx: c.cx, cy: c.cy });
        }
        return positions;
    }

    /**
     * Hit-test mouse coordinates against the 8 handles.
     * @param {number} mx - Mouse X in canvas/screen space.
     * @param {number} my - Mouse Y in canvas/screen space.
     * @returns {Object|null} The handle definition, or null.
     */
    _hitTest(mx, my) {
        const positions = this._handlePositions();
        const r = this._HIT_RADIUS;
        for (const h of positions) {
            if (Math.abs(mx - h.cx) <= r && Math.abs(my - h.cy) <= r) {
                return h.def;
            }
        }
        return null;
    }

    // -- Drawing --------------------------------------------------------------

    /** Draw the box outline and all 8 handles on the FrameViewer canvas. */
    _draw() {
        if (!this._active || !this._box) return;

        this._fv._syncCanvasSize();
        const ctx = this._fv.ctx;
        ctx.clearRect(0, 0, this._fv.canvas.width, this._fv.canvas.height);

        const tl = this._pixelToCanvas(this._box.x1, this._box.y1);
        const br = this._pixelToCanvas(this._box.x2, this._box.y2);

        // Box outline -- dashed cyan
        ctx.strokeStyle = '#00bcd4';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(tl.cx, tl.cy, br.cx - tl.cx, br.cy - tl.cy);
        ctx.setLineDash([]);

        // Semi-transparent fill for visibility
        ctx.fillStyle = 'rgba(0, 188, 212, 0.08)';
        ctx.fillRect(tl.cx, tl.cy, br.cx - tl.cx, br.cy - tl.cy);

        // Draw handles
        const half = this._HANDLE_SIZE / 2;
        const positions = this._handlePositions();
        for (const h of positions) {
            ctx.fillStyle = '#00bcd4';
            ctx.fillRect(h.cx - half, h.cy - half, this._HANDLE_SIZE, this._HANDLE_SIZE);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.strokeRect(h.cx - half, h.cy - half, this._HANDLE_SIZE, this._HANDLE_SIZE);
        }
    }

    // -- Pointer Events -------------------------------------------------------

    _onPointerDown(e) {
        if (!this._active) return;
        e.preventDefault();
        e.stopPropagation();

        const rect = this._fv.el.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const handle = this._hitTest(mx, my);
        if (!handle) return;

        this._dragging = true;
        this._dragHandle = handle;
        this._fv.canvas.setPointerCapture(e.pointerId);
        this._fv.canvas.style.cursor = handle.cursor;
    }

    _onPointerMove(e) {
        if (!this._active) return;
        e.preventDefault();
        e.stopPropagation();

        const rect = this._fv.el.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        if (this._dragging && this._dragHandle) {
            // Convert mouse position to video-pixel coords
            const coords = this._fv._eventToPixelCoords(e);
            const px = coords.px;
            const py = coords.py;

            // Move the edge(s) controlled by this handle
            if (this._dragHandle.edgesX) {
                this._box[this._dragHandle.edgesX] = Math.max(0, Math.min(this._fv.videoWidth, px));
            }
            if (this._dragHandle.edgesY) {
                this._box[this._dragHandle.edgesY] = Math.max(0, Math.min(this._fv.videoHeight, py));
            }

            // Enforce minimum box size -- prevent edge inversion
            this._enforceMinSize();

            this._draw();
        } else {
            // Hover: update cursor based on which handle is under the mouse
            const handle = this._hitTest(mx, my);
            this._fv.canvas.style.cursor = handle ? handle.cursor : 'default';
        }
    }

    _onPointerUp(e) {
        if (!this._active || !this._dragging) return;
        e.preventDefault();
        e.stopPropagation();

        this._dragging = false;
        this._dragHandle = null;

        if (e.pointerId != null) {
            try { this._fv.canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
        }

        this._fv.canvas.style.cursor = 'default';

        // Normalize: ensure x1 < x2, y1 < y2
        this._normalizeBox();
        this._draw();

        // Fire callbacks
        const box = this.getBox();
        if (box) {
            for (const cb of this._callbacks) {
                cb(box);
            }
        }
    }

    // -- Box Constraints ------------------------------------------------------

    /** Normalize box so x1 <= x2 and y1 <= y2. */
    _normalizeBox() {
        if (!this._box) return;
        const b = this._box;
        const nx1 = Math.min(b.x1, b.x2);
        const ny1 = Math.min(b.y1, b.y2);
        const nx2 = Math.max(b.x1, b.x2);
        const ny2 = Math.max(b.y1, b.y2);
        b.x1 = nx1;
        b.y1 = ny1;
        b.x2 = nx2;
        b.y2 = ny2;
    }

    /**
     * Enforce minimum box size of _MIN_SIZE pixels in each dimension.
     * When a drag would make the box too small, clamp the moving edge
     * so the box stays at minimum size.
     */
    _enforceMinSize() {
        if (!this._box || !this._dragHandle) return;
        const b = this._box;
        const min = this._MIN_SIZE;
        const h = this._dragHandle;

        // X-axis enforcement
        if (h.edgesX === 'x1' && b.x1 > b.x2 - min) {
            b.x1 = b.x2 - min;
        } else if (h.edgesX === 'x2' && b.x2 < b.x1 + min) {
            b.x2 = b.x1 + min;
        }

        // Y-axis enforcement
        if (h.edgesY === 'y1' && b.y1 > b.y2 - min) {
            b.y1 = b.y2 - min;
        } else if (h.edgesY === 'y2' && b.y2 < b.y1 + min) {
            b.y2 = b.y1 + min;
        }

        // Re-clamp to frame bounds after min-size enforcement
        b.x1 = Math.max(0, b.x1);
        b.y1 = Math.max(0, b.y1);
        b.x2 = Math.min(this._fv.videoWidth, b.x2);
        b.y2 = Math.min(this._fv.videoHeight, b.y2);
    }
}
