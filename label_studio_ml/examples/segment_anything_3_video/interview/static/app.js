/* ==========================================================================
   SAM3 Interview UI - Main Application
   Vanilla JS SPA: hash-based routing, API client, state management,
   phase renderers, keyboard shortcuts, job polling, toast notifications.
   No frameworks, no imports -- everything attaches to the global scope.
   ========================================================================== */

'use strict';

// ---------------------------------------------------------------------------
// API Client
// ---------------------------------------------------------------------------

const API = {
    base: '/interview/api',

    /**
     * POST JSON to an API endpoint.
     * @param {string} path - Relative path (e.g. '/session/init').
     * @param {Object} data - JSON body.
     * @returns {Promise<Object>} Parsed JSON response.
     */
    async post(path, data) {
        try {
            const res = await fetch(`${this.base}${path}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            const json = await res.json();
            if (!res.ok) {
                const msg = json.error || `Request failed (${res.status})`;
                showToast(msg, 'error');
                throw new Error(msg);
            }
            return json;
        } catch (err) {
            if (!err.message.includes('Request failed')) {
                showToast(`Network error: ${err.message}`, 'error');
            }
            throw err;
        }
    },

    /**
     * GET from an API endpoint with optional query parameters.
     * @param {string} path - Relative path (e.g. '/detect/crops').
     * @param {Object} [params] - Query string parameters.
     * @returns {Promise<Object>} Parsed JSON response.
     */
    async get(path, params) {
        try {
            let url = `${this.base}${path}`;
            if (params) {
                const qs = new URLSearchParams(params).toString();
                url += `?${qs}`;
            }
            const res = await fetch(url);
            const json = await res.json();
            if (!res.ok) {
                const msg = json.error || `Request failed (${res.status})`;
                showToast(msg, 'error');
                throw new Error(msg);
            }
            return json;
        } catch (err) {
            if (!err.message.includes('Request failed')) {
                showToast(`Network error: ${err.message}`, 'error');
            }
            throw err;
        }
    },

    /**
     * PUT JSON to an API endpoint.
     * @param {string} path - Relative path.
     * @param {Object} data - JSON body.
     * @returns {Promise<Object>} Parsed JSON response.
     */
    async put(path, data) {
        try {
            const res = await fetch(`${this.base}${path}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            const json = await res.json();
            if (!res.ok) {
                const msg = json.error || `Request failed (${res.status})`;
                showToast(msg, 'error');
                throw new Error(msg);
            }
            return json;
        } catch (err) {
            if (!err.message.includes('Request failed')) {
                showToast(`Network error: ${err.message}`, 'error');
            }
            throw err;
        }
    },

    /**
     * DELETE with JSON body.
     * @param {string} path - Relative path.
     * @param {Object} data - JSON body.
     * @returns {Promise<Object>} Parsed JSON response.
     */
    async delete_(path, data) {
        try {
            const res = await fetch(`${this.base}${path}`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            const json = await res.json();
            if (!res.ok) {
                const msg = json.error || `Request failed (${res.status})`;
                showToast(msg, 'error');
                throw new Error(msg);
            }
            return json;
        } catch (err) {
            if (!err.message.includes('Request failed')) {
                showToast(`Network error: ${err.message}`, 'error');
            }
            throw err;
        }
    },
};

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------

const AppState = {
    sessionId: null,
    phase: 'init',       // init, detection, classification, reid, seeding
    projectId: null,
    taskId: null,
    annotationId: null,

    // Video metadata
    videoWidth: 0,
    videoHeight: 0,
    framesCount: 0,
    fps: 30,
    sampledFrames: [],

    // Detection / Classification
    crops: [],
    currentCropIndex: 0,
    currentFrameIdx: 0,
    currentFrameInSampled: 0,
    drawMode: false,
    sortBy: 'uncertainty',
    filterLabel: 'all',

    // Session stats (from backend)
    stats: {},

    // Cached options from session/init
    cacheOptions: null,

    // Reject review sub-phase
    rejectReviewMode: false,
    rejectReviewCrops: [],       // filtered list of rejected crops for current round
    rejectReviewIndex: 0,
    rejectReviewSubcategory: 'not_person',  // current subcategory selection
    rejectReviewBoxAdjusted: false,         // whether box was adjusted for current crop
    rejectReviewDrawActive: false,          // reject-review-only draw/save toggle state
    rejectReviewFixExpandPct: 0,            // expansion percent for pre-Fix prompt preview
    rejectReviewExpandBase: null,           // base box for expansion preview [x1,y1,x2,y2]
    rejectReviewSam3AutoEnabled: false,     // persistent in-session SAM3 auto mode preference
    rejectReviewSam3InFlight: false,        // true while /detect/refine_box request is active
    rejectReviewSam3QueuedPrompt: null,     // latest queued {cropId, promptXyxy, source}
    rejectReviewSam3ActiveRequestId: 0,     // monotonic request token for stale-response guard

    // Active components (for cleanup)
    _components: {},
};

// ---------------------------------------------------------------------------
// Toast Notifications
// ---------------------------------------------------------------------------

/**
 * Show a toast message.
 * @param {string} message - Toast text.
 * @param {string} [type='info'] - 'info', 'success', 'error', 'warning'.
 * @param {number} [duration=4000] - Auto-dismiss in ms.
 */
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    const dismiss = () => {
        toast.classList.add('dismissing');
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 300);
    };

    toast.addEventListener('click', dismiss);
    if (duration > 0) {
        setTimeout(dismiss, duration);
    }
}

// ---------------------------------------------------------------------------
// Phase Indicator
// ---------------------------------------------------------------------------

const PHASE_ORDER = ['init', 'detection', 'classification', 'reid', 'seeding'];

/**
 * Update the nav bar phase dots to reflect the current phase.
 * @param {string} activePhase
 */
function updatePhaseIndicator(activePhase) {
    const dots = document.querySelectorAll('.phase-dot');
    const activeIdx = PHASE_ORDER.indexOf(activePhase);

    dots.forEach((dot) => {
        const phase = dot.dataset.phase;
        const phaseIdx = PHASE_ORDER.indexOf(phase);

        dot.classList.remove('active', 'complete', 'clickable');

        if (phaseIdx < activeIdx) {
            dot.classList.add('complete', 'clickable');
        } else if (phaseIdx === activeIdx) {
            dot.classList.add('active');
        }
    });
}

/** Wire click handlers on phase dots (called once at startup). */
function _initPhaseNavigation() {
    document.querySelectorAll('.phase-dot').forEach((dot) => {
        dot.addEventListener('click', () => {
            if (!dot.classList.contains('clickable')) return;
            const phase = dot.dataset.phase;
            if (phase && phase !== AppState.phase) {
                navigate(phase);
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/**
 * Navigate to a phase and render it.
 * @param {string} phase - Phase name.
 */
function navigate(phase) {
    // Cancel background polls before changing phase
    if (AppState._embeddingPollCancel) {
        AppState._embeddingPollCancel();
        AppState._embeddingPollCancel = null;
    }
    window.location.hash = `/${phase}`;
    AppState.phase = phase;
    renderPhase(phase);
    updatePhaseIndicator(phase);
}

/**
 * Render the appropriate phase into #app.
 * @param {string} phase
 */
function renderPhase(phase) {
    // Cleanup existing components
    _cleanupComponents();

    const app = document.getElementById('app');
    app.innerHTML = '';

    switch (phase) {
        case 'init':
        case 'setup':
            renderSetup(app);
            break;
        case 'detection':
        case 'classification':
            renderDetection(app);
            break;
        case 'reid':
            renderReID(app);
            break;
        case 'seeding':
            renderSeeding(app);
            break;
        default:
            renderSetup(app);
            break;
    }
}

/** Destroy all tracked component instances. */
function _cleanupComponents() {
    for (const key of Object.keys(AppState._components)) {
        const comp = AppState._components[key];
        if (comp && typeof comp.destroy === 'function') {
            comp.destroy();
        }
        delete AppState._components[key];
    }
}

/**
 * Handle hash-change events for browser back/forward navigation.
 */
function _onHashChange() {
    const hash = window.location.hash.replace(/^#\/?/, '') || 'setup';
    // Only re-render if phase actually changed
    if (hash !== AppState.phase) {
        AppState.phase = hash;
        renderPhase(hash);
        updatePhaseIndicator(hash);
    }
}

// ---------------------------------------------------------------------------
// Job Polling
// ---------------------------------------------------------------------------

/**
 * Poll a background job until it completes or fails.
 * @param {string} jobId
 * @param {Function} onProgress - Called with progress object on each poll.
 * @param {Function} onComplete - Called with final progress when done.
 * @param {number} [interval=500] - Polling interval in ms.
 * @returns {Function} A cancel function to stop polling.
 */
function pollJob(jobId, onProgress, onComplete, interval = 500) {
    let cancelled = false;

    const poll = setInterval(async () => {
        if (cancelled) {
            clearInterval(poll);
            return;
        }
        try {
            const progress = await API.get(`/job/${jobId}/progress`);
            if (onProgress) onProgress(progress);

            if (progress.status === 'completed' || progress.status === 'failed') {
                clearInterval(poll);
                if (onComplete) onComplete(progress);
            }
        } catch (err) {
            // Silently retry on transient errors
            console.warn('Poll error for job', jobId, err);
        }
    }, interval);

    return () => {
        cancelled = true;
        clearInterval(poll);
    };
}

// ---------------------------------------------------------------------------
// Phase Renderer: Setup
// ---------------------------------------------------------------------------

/**
 * Render the session setup form (init phase).
 * @param {HTMLElement} app
 */
function renderSetup(app) {
    const wrap = document.createElement('div');
    wrap.className = 'session-setup';

    wrap.innerHTML = `
        <h2>Session Setup</h2>
        <div class="form-group">
            <label for="setup-project-id">Project ID</label>
            <input type="number" id="setup-project-id" placeholder="e.g. 42"
                   value="${AppState.projectId || ''}">
        </div>
        <div class="form-group">
            <label for="setup-task-id">Task ID</label>
            <input type="number" id="setup-task-id" placeholder="e.g. 101"
                   value="${AppState.taskId || ''}">
        </div>
        <div class="form-group">
            <label for="setup-annotation-id">Annotation ID (optional)</label>
            <input type="number" id="setup-annotation-id" placeholder="Leave blank for new"
                   value="${AppState.annotationId || ''}">
        </div>
        <div class="form-group">
            <label for="setup-prompt">Detection Prompt</label>
            <input type="text" id="setup-prompt" placeholder="e.g. person"
                   value="person">
        </div>
        <div id="setup-cache-info" class="hidden" style="margin-bottom:16px;
             font-size:0.8rem;color:var(--text-secondary);"></div>
        <div id="setup-actions" class="session-actions">
            <button id="setup-check-btn" class="btn btn-secondary">Check Cache</button>
        </div>
        <div id="setup-progress" class="hidden" style="margin-top:16px;"></div>
    `;

    app.appendChild(wrap);

    // Bind check-cache button
    document.getElementById('setup-check-btn').addEventListener('click', _onCheckCache);
}

/** Handle "Check Cache" -- calls session/init to discover cache options. */
async function _onCheckCache() {
    const projectId = document.getElementById('setup-project-id').value.trim();
    const taskId = document.getElementById('setup-task-id').value.trim();
    const annotationId = document.getElementById('setup-annotation-id').value.trim();

    if (!projectId || !taskId) {
        showToast('Project ID and Task ID are required', 'warning');
        return;
    }

    AppState.projectId = parseInt(projectId, 10);
    AppState.taskId = parseInt(taskId, 10);
    AppState.annotationId = annotationId ? parseInt(annotationId, 10) : null;

    try {
        const result = await API.post('/session/init', {
            project_id: AppState.projectId,
            task_id: AppState.taskId,
            annotation_id: AppState.annotationId,
        });

        AppState.cacheOptions = result;
        _renderCacheOptions(result);
    } catch (err) {
        // Error already toasted by API client
    }
}

/**
 * Show cache options split into two cards: Session State and Frame Cache.
 * @param {Object} result - Response from session/init.
 */
function _renderCacheOptions(result) {
    var infoEl = document.getElementById('setup-cache-info');
    var actionsEl = document.getElementById('setup-actions');

    infoEl.classList.remove('hidden');
    infoEl.innerHTML = '';
    actionsEl.innerHTML = '';

    // ------------------------------------------------------------------
    // Card 1: Session State
    // ------------------------------------------------------------------
    var sessionCard = document.createElement('div');
    sessionCard.style.cssText =
        'padding:12px 16px;margin-bottom:12px;' +
        'background:var(--bg-surface);border:1px solid var(--border-default);' +
        'border-radius:var(--radius-sm);';

    var sessionHeader = document.createElement('div');
    sessionHeader.style.cssText =
        'font-size:0.75rem;font-weight:600;text-transform:uppercase;' +
        'letter-spacing:0.05em;color:var(--text-secondary);margin-bottom:8px;';
    sessionHeader.textContent = 'Session State';
    sessionCard.appendChild(sessionHeader);

    var sessionInfo = document.createElement('div');
    sessionInfo.style.cssText = 'font-size:0.85rem;margin-bottom:10px;';

    var sessionBtns = document.createElement('div');
    sessionBtns.className = 'session-actions';
    sessionBtns.style.marginTop = '8px';

    if (result.has_cache) {
        sessionInfo.textContent = 'Existing session cache found (config, crops, model, clusters).';

        var resumeBtn = document.createElement('button');
        resumeBtn.className = 'btn btn-secondary';
        resumeBtn.textContent = 'Resume';
        resumeBtn.title = 'Continue exactly where you left off — same phase, same crops, same model.';
        resumeBtn.addEventListener('click', function () { _startSession('resume'); });
        sessionBtns.appendChild(resumeBtn);

        var buildBtn = document.createElement('button');
        buildBtn.className = 'btn btn-secondary';
        buildBtn.textContent = 'Build On';
        buildBtn.title = 'Keep crops, labels, and trained model but restart from the detection phase to add more rounds.';
        buildBtn.addEventListener('click', function () { _startSession('build_on'); });
        sessionBtns.appendChild(buildBtn);

        var freshBtn = document.createElement('button');
        freshBtn.className = 'btn btn-primary';
        freshBtn.textContent = 'Fresh Start';
        freshBtn.title = 'Delete all session data (crops, model, clusters) and start over. Frame cache can be kept.';
        freshBtn.addEventListener('click', async function () {
            var confirmed = await Modal.confirm(
                'Fresh Start',
                'This will delete the session state (config, crops, model, clusters). Continue?'
            );
            if (confirmed) {
                // Check if user wants to keep frame cache
                var keepCb = document.getElementById('keep-frame-cache-cb');
                var keepFrames = keepCb ? keepCb.checked : false;
                _startSession('fresh', keepFrames);
            }
        });
        sessionBtns.appendChild(freshBtn);
    } else if (result.other_caches && result.other_caches.length > 0) {
        sessionInfo.textContent = 'No cache for this task. Other task caches found in this project.';

        result.other_caches.forEach(function (cache) {
            var btn = document.createElement('button');
            btn.className = 'btn btn-secondary';
            btn.textContent = 'Use from Task ' + cache.task_id;
            btn.addEventListener('click', function () {
                _startSession('use_from_' + cache.task_id);
            });
            sessionBtns.appendChild(btn);
        });

        var freshBtn2 = document.createElement('button');
        freshBtn2.className = 'btn btn-primary';
        freshBtn2.textContent = 'Fresh Start';
        freshBtn2.addEventListener('click', function () { _startSession('fresh'); });
        sessionBtns.appendChild(freshBtn2);
    } else {
        sessionInfo.textContent = 'No existing session cache.';

        var startBtn = document.createElement('button');
        startBtn.className = 'btn btn-primary';
        startBtn.textContent = 'Start';
        startBtn.addEventListener('click', function () { _startSession('fresh'); });
        sessionBtns.appendChild(startBtn);
    }

    sessionCard.appendChild(sessionInfo);
    sessionCard.appendChild(sessionBtns);
    infoEl.appendChild(sessionCard);

    // ------------------------------------------------------------------
    // Card 2: Frame Cache
    // ------------------------------------------------------------------
    _renderFrameCacheCard(infoEl, result);
}

/**
 * Render the Frame Cache card showing this-task info, global usage,
 * keep-on-fresh checkbox, and delete button.
 * @param {HTMLElement} parentEl
 * @param {Object} result - session/init response
 */
async function _renderFrameCacheCard(parentEl, result) {
    var card = document.createElement('div');
    card.style.cssText =
        'padding:12px 16px;' +
        'background:var(--bg-surface);border:1px solid var(--border-default);' +
        'border-radius:var(--radius-sm);';

    var header = document.createElement('div');
    header.style.cssText =
        'font-size:0.75rem;font-weight:600;text-transform:uppercase;' +
        'letter-spacing:0.05em;color:var(--text-secondary);margin-bottom:8px;';
    header.textContent = 'Frame Cache';
    card.appendChild(header);

    var info = document.createElement('div');
    info.style.cssText = 'font-size:0.85rem;margin-bottom:8px;';

    // This-task frame cache info
    if (result.has_frame_cache) {
        info.innerHTML =
            'This task: <strong>' + result.frame_cache_size + '</strong>' +
            ' (' + result.frame_cache_frames.toLocaleString() + ' decoded frames)';
    } else {
        info.textContent = 'No frame cache for this task. Will be created during detection.';
    }
    card.appendChild(info);

    // Global disk usage (async)
    var globalLine = document.createElement('div');
    globalLine.style.cssText = 'font-size:0.8rem;color:var(--text-secondary);margin-bottom:10px;';
    globalLine.textContent = '';
    card.appendChild(globalLine);

    try {
        var data = await API.get('/disk_usage');
        if (data.total_bytes > 0) {
            globalLine.textContent =
                'Total across all tasks: ' + data.total_human +
                ' (' + data.sessions_cached +
                ' video' + (data.sessions_cached !== 1 ? 's' : '') + ' cached)';
        }
    } catch (err) { /* non-critical */ }

    // Controls row
    var controls = document.createElement('div');
    controls.style.cssText = 'display:flex;align-items:center;gap:16px;flex-wrap:wrap;';

    // Keep frame cache checkbox (only relevant if session cache exists)
    if (result.has_cache && result.has_frame_cache) {
        var label = document.createElement('label');
        label.style.cssText =
            'display:flex;align-items:center;gap:6px;font-size:0.8rem;cursor:pointer;';
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.id = 'keep-frame-cache-cb';
        cb.checked = true;  // default: keep frames
        label.appendChild(cb);
        label.appendChild(document.createTextNode('Keep on Fresh Start'));
        controls.appendChild(label);
    }

    // Delete frame cache button (only if frame cache exists)
    if (result.has_frame_cache) {
        var delBtn = document.createElement('button');
        delBtn.className = 'btn btn-ghost';
        delBtn.style.cssText = 'font-size:0.8rem;margin-left:auto;';
        delBtn.textContent = 'Delete Frame Cache';
        delBtn.addEventListener('click', async function () {
            var confirmed = await Modal.confirm(
                'Delete Frame Cache',
                'Delete ' + result.frame_cache_size +
                ' of decoded frames? They will be re-created on next detection run.'
            );
            if (confirmed) {
                try {
                    await API.delete_('/frame_cache', {
                        cache_key: result.cache_key,
                    });
                    showToast('Frame cache deleted', 'success');
                    // Refresh cache options
                    _onCheckCache();
                } catch (err) {
                    showToast('Failed to delete frame cache: ' + err.message, 'error');
                }
            }
        });
        controls.appendChild(delBtn);
    }

    if (controls.childNodes.length > 0) {
        card.appendChild(controls);
    }

    parentEl.appendChild(card);
}

/**
 * Create/resume session, fetch video info, run detection, then navigate.
 * @param {string} mode - 'resume', 'build_on', 'fresh', or 'use_from_<id>'.
 * @param {boolean} [keepFrameCache=false] - Preserve disk frame cache on fresh start.
 */
async function _startSession(mode, keepFrameCache) {
    var actionsEl = document.getElementById('setup-actions');
    var progressEl = document.getElementById('setup-progress');

    // Disable buttons
    actionsEl.querySelectorAll('.btn').forEach(function (b) { b.disabled = true; });
    progressEl.classList.remove('hidden');

    // Build inline progress widget (step text + bar + elapsed timer)
    progressEl.innerHTML = '';
    var _stepText = document.createElement('div');
    _stepText.style.cssText = 'font-weight:600;margin-bottom:8px;';
    _stepText.textContent = 'Initializing session...';
    progressEl.appendChild(_stepText);

    var _barWrapper = document.createElement('div');
    _barWrapper.className = 'progress-bar-wrapper';
    _barWrapper.style.cssText = 'width:320px;max-width:100%;';
    var _track = document.createElement('div');
    _track.className = 'progress-bar-track';
    var _fill = document.createElement('div');
    _fill.className = 'progress-bar-fill indeterminate';
    _track.appendChild(_fill);
    _barWrapper.appendChild(_track);
    progressEl.appendChild(_barWrapper);

    var _elapsed = document.createElement('div');
    _elapsed.style.cssText = 'font-size:0.8rem;color:var(--text-muted,#999);margin-top:4px;';
    _elapsed.textContent = '';
    progressEl.appendChild(_elapsed);

    /** Update the setup progress widget from a poll response. */
    function _updateSetupProgress(p) {
        _stepText.textContent = p.step || 'Processing...';
        if (typeof p.elapsed_seconds === 'number') {
            var secs = Math.round(p.elapsed_seconds);
            _elapsed.textContent = secs < 60
                ? secs + 's elapsed'
                : Math.floor(secs / 60) + 'm ' + (secs % 60) + 's elapsed';
        }
        if (typeof p.percent === 'number' && p.percent >= 0) {
            _fill.classList.remove('indeterminate');
            _fill.style.width = Math.min(100, Math.max(0, p.percent)) + '%';
        }
    }

    try {
        // 1) Resume or create session
        var payload = {
            project_id: AppState.projectId,
            task_id: AppState.taskId,
            annotation_id: AppState.annotationId,
            mode: mode,
        };
        if (mode === 'fresh' && keepFrameCache) {
            payload.keep_frame_cache = true;
        }
        var sessionResult = await API.post('/session/resume', payload);

        AppState.sessionId = sessionResult.session_id;
        AppState.stats = sessionResult;

        // If resuming into an advanced phase, go directly there
        if (mode === 'resume' && sessionResult.phase && sessionResult.phase !== 'init') {
            // Re-download video if file was lost (e.g. container rebuild)
            if (sessionResult.needs_video_info) {
                _stepText.textContent = 'Re-downloading video file...';
                var reVideoJob = await API.post('/session/' + AppState.sessionId + '/video_info', {});
                await new Promise(function (resolve, reject) {
                    pollJob(
                        reVideoJob.job_id,
                        function (p) { _updateSetupProgress(p); },
                        function (p) {
                            if (p.status === 'failed') reject(new Error(p.error || 'Video download failed'));
                            else resolve(p);
                        }
                    );
                });
            }
            _applyStats(sessionResult);
            showToast('Session resumed', 'success');
            navigate(sessionResult.phase);
            return;
        }

        // 2) Fetch video info (background job)
        _stepText.textContent = 'Fetching video info...';
        const videoJob = await API.post(`/session/${AppState.sessionId}/video_info`, {});

        await new Promise((resolve, reject) => {
            pollJob(
                videoJob.job_id,
                (p) => {
                    _updateSetupProgress(p);
                },
                (p) => {
                    if (p.status === 'failed') {
                        reject(new Error(p.error || 'Video info fetch failed'));
                    } else {
                        resolve(p);
                    }
                }
            );
        });

        // 3) Get updated session status
        const status = await API.get(`/session/${AppState.sessionId}/status`);
        _applyStats(status);

        // 4) Run detection (Stage 1 = fast, embedding = background)
        _stepText.textContent = 'Running detection on sampled frames...';
        _elapsed.textContent = '';
        _fill.classList.add('indeterminate');
        _fill.style.width = '';
        const prompt = document.getElementById('setup-prompt').value.trim() || 'person';
        const detectResult = await API.post('/detect/start', {
            session_id: AppState.sessionId,
            prompt,
        });

        // Store embedding job ID for background polling
        AppState._embeddingJobId = detectResult.embedding_job_id || null;

        // Wait only for the fast Stage 1 detection job
        await new Promise((resolve, reject) => {
            pollJob(
                detectResult.job_id,
                (p) => {
                    _updateSetupProgress(p);
                },
                (p) => {
                    if (p.status === 'failed') {
                        reject(new Error(p.error || 'Detection failed'));
                    } else {
                        resolve(p);
                    }
                }
            );
        });

        showToast('Detection complete — crops ready for labeling', 'success');
        navigate('detection');

        // Start background embedding status polling (non-blocking)
        if (AppState._embeddingJobId) {
            _startEmbeddingPoll();
        }
    } catch (err) {
        showToast(`Setup failed: ${err.message}`, 'error');
        actionsEl.querySelectorAll('.btn').forEach((b) => (b.disabled = false));
        progressEl.innerHTML = '';
        progressEl.classList.add('hidden');
    }
}

/**
 * Apply session stats to AppState.
 * @param {Object} stats
 */
function _applyStats(stats) {
    AppState.stats = stats;
    if (stats.video_frames) AppState.framesCount = stats.video_frames;
    if (stats.sampled_frames != null) {
        // Note: sampled_frames from stats is a count, not the array.
        // We may need to fetch actual sampled frame list from crops.
    }
}

// ---------------------------------------------------------------------------
// Phase Renderer: Detection / Classification
// ---------------------------------------------------------------------------

/**
 * Render the detection/classification work area.
 * Split panel: left = frame viewer, right = crop labeler + grid.
 * @param {HTMLElement} app
 */
function renderDetection(app) {
    if (!AppState.sessionId) {
        showToast('No active session. Redirecting to setup.', 'warning');
        navigate('setup');
        return;
    }

    // Create split panel
    const split = new SplitPanel(app, 0.6);
    AppState._components.splitPanel = split;

    const leftPanel = split.getLeft();
    const rightPanel = split.getRight();

    // Toolbar at top of left panel
    const toolbar = new Toolbar(leftPanel);
    AppState._components.toolbar = toolbar;

    // Frame viewer
    const frameViewer = new FrameViewer(leftPanel, {
        width: AppState.videoWidth || 1920,
        height: AppState.videoHeight || 1080,
    });
    AppState._components.frameViewer = frameViewer;

    // Progress overlay on the left panel
    const progress = new ProgressOverlay(leftPanel);
    AppState._components.progressOverlay = progress;

    // Header above crop labeler
    const header = document.createElement('h3');
    header.className = 'detection-header';
    header.textContent = 'Is this a good crop?';
    rightPanel.appendChild(header);

    // Crop labeler at top of right panel
    const cropLabeler = new CropLabeler(rightPanel);
    AppState._components.cropLabeler = cropLabeler;

    // Crop grid below the labeler in right panel
    const cropGrid = new CropGrid(rightPanel);
    AppState._components.cropGrid = cropGrid;

    // Wire up frame viewer draw mode
    frameViewer.onBoxDrawn(async (box) => {
        try {
            const result = await API.post('/detect/draw', {
                session_id: AppState.sessionId,
                frame_idx: frameViewer.getCurrentFrame(),
                xyxy: [box.x1, box.y1, box.x2, box.y2],
            });
            showToast('Box added', 'success');
            AppState.stats = result;
            frameViewer.reload(AppState.sessionId);
            await _refreshCrops();
        } catch (err) {
            // Already toasted
        }
    });

    // Wire up crop labeler callbacks
    cropLabeler.onAccept((crop) => _labelCrop(crop, 'accepted'));
    cropLabeler.onReject((crop) => _labelCrop(crop, 'rejected'));
    cropLabeler.onSkip((crop) => _labelCrop(crop, 'skipped'));

    // Wire up crop grid selection
    cropGrid.onCropSelect((crop, index) => {
        AppState.currentCropIndex = index;
        cropLabeler.showCrop(crop, AppState.sessionId);
        // Navigate frame viewer to the crop's frame
        frameViewer.loadFrame(crop.frame_idx, AppState.sessionId, true, crop.crop_id);
    });

    // Render toolbar
    _renderToolbar();

    // Load initial data
    _loadDetectionData();
}

/** Refresh the toolbar with current state. */
function _renderToolbar() {
    const toolbar = AppState._components.toolbar;
    if (!toolbar) return;

    toolbar.render({
        drawMode: AppState.drawMode,
        disableDrawToggle: AppState.rejectReviewMode,
        sortBy: AppState.sortBy,
        filterLabel: AppState.filterLabel,
        stats: AppState.stats,
        onDrawToggle: () => {
            if (AppState.rejectReviewMode) return;
            AppState.drawMode = !AppState.drawMode;
            const fv = AppState._components.frameViewer;
            if (fv) {
                if (AppState.drawMode) {
                    fv.enableDrawMode();
                } else {
                    fv.disableDrawMode();
                }
            }
            _renderToolbar();
        },

        onNextRound: _onNextRound,
        currentRound: AppState.stats.current_round || 1,
        roundsCompleted: AppState.stats.rounds_completed || 0,

        onPrevFrame: () => _navigateFrame(-1),
        onNextFrame: () => _navigateFrame(1),

        onSortChange: (sort) => {
            AppState.sortBy = sort;
            _refreshCrops();
        },

        onFilterChange: (filter) => {
            AppState.filterLabel = filter;
            _refreshCrops();
        },

        onAdvancePhase: async () => {
            const confirmed = await Modal.confirm(
                'Advance to ReID',
                'Move to the Re-Identification phase? You can return to labeling later.'
            );
            if (confirmed) {
                navigate('reid');
            }
        },
    });
}

/** Load crops and video info for detection phase. */
async function _loadDetectionData() {
    try {
        // Get session status for video dimensions
        const status = await API.get(`/session/${AppState.sessionId}/status`);
        _applyStats(status);

        // Update frame viewer dimensions if we have them
        const fv = AppState._components.frameViewer;
        if (fv && AppState.videoWidth && AppState.videoHeight) {
            fv.setVideoDimensions(AppState.videoWidth, AppState.videoHeight);
        }

        await _refreshCrops();

        // Keep frame index aligned with the selected crop; _refreshCrops()
        // already loads the highlighted frame.
        if (AppState.crops.length > 0) {
            const idx = Math.max(0, Math.min(AppState.currentCropIndex, AppState.crops.length - 1));
            AppState.currentFrameIdx = AppState.crops[idx].frame_idx;
        }
    } catch (err) {
        showToast(`Failed to load detection data: ${err.message}`, 'error');
    }
}

/** Refresh the crop list from the backend. */
async function _refreshCrops() {
    try {
        const result = await API.get('/detect/crops', {
            session_id: AppState.sessionId,
            sort: AppState.sortBy,
            filter: AppState.filterLabel,
            limit: 200,
        });
        AppState.crops = result.crops || [];

        const grid = AppState._components.cropGrid;
        if (grid) {
            grid.render(AppState.crops, AppState.sessionId);

            // Re-select current crop
            if (AppState.currentCropIndex >= 0 && AppState.currentCropIndex < AppState.crops.length) {
                grid.select(AppState.currentCropIndex);
                const labeler = AppState._components.cropLabeler;
                const fv = AppState._components.frameViewer;
                if (labeler) {
                    const selected = AppState.crops[AppState.currentCropIndex];
                    labeler.showCrop(selected, AppState.sessionId);
                    if (fv && selected) {
                        AppState.currentFrameIdx = selected.frame_idx;
                        fv.loadFrame(selected.frame_idx, AppState.sessionId, true, selected.crop_id);
                    }
                }
            } else if (AppState.crops.length > 0) {
                AppState.currentCropIndex = 0;
                grid.select(0);
                const labeler = AppState._components.cropLabeler;
                const fv = AppState._components.frameViewer;
                if (labeler) {
                    const selected = AppState.crops[0];
                    labeler.showCrop(selected, AppState.sessionId);
                    if (fv) {
                        AppState.currentFrameIdx = selected.frame_idx;
                        fv.loadFrame(selected.frame_idx, AppState.sessionId, true, selected.crop_id);
                    }
                }
            }
        }

        // Refresh stats in toolbar
        const statusResult = await API.get(`/session/${AppState.sessionId}/status`);
        _applyStats(statusResult);
        _renderToolbar();
    } catch (err) {
        // Already toasted
    }
}

/**
 * Label a crop and advance to the next one.
 * @param {Object} crop
 * @param {string} label - 'accepted', 'rejected', or 'skipped'.
 */
async function _labelCrop(crop, label) {
    try {
        const result = await API.post('/detect/label', {
            session_id: AppState.sessionId,
            labels: { [crop.crop_id]: label },
        });
        AppState.stats = result;

        // Update local crop state
        crop.label = label;

        // Update grid card in place
        const grid = AppState._components.cropGrid;
        if (grid) {
            grid.updateCardLabel(AppState.currentCropIndex, label);
        }

        // Auto-advance to next pending crop (reloads frame with highlight)
        _advanceToNextPending();

        // Update toolbar stats
        _renderToolbar();
    } catch (err) {
        // Already toasted
    }
}

/** Move to the next pending (unlabeled) crop. */
function _advanceToNextPending() {
    const crops = AppState.crops;
    const start = AppState.currentCropIndex;

    // Search forward from current position
    for (let i = start + 1; i < crops.length; i++) {
        if (crops[i].label === 'pending') {
            _selectCropByIndex(i);
            return;
        }
    }
    // Wrap around
    for (let i = 0; i < start; i++) {
        if (crops[i].label === 'pending') {
            _selectCropByIndex(i);
            return;
        }
    }
    // No pending crops left -- check for reject review
    var unreviewed = crops.filter(function (c) {
        return c.label === 'rejected' && !c.reject_reason;
    });
    if (unreviewed.length > 0) {
        _enterRejectReview(unreviewed);
        return;
    }

    // No pending and no unreviewed rejects -- just move to next
    if (start + 1 < crops.length) {
        _selectCropByIndex(start + 1);
    } else {
        // Stay on current crop but refresh frame to show updated label color
        var fv = AppState._components.frameViewer;
        var crop = crops[start];
        if (fv && crop) fv.reload(AppState.sessionId, crop.crop_id);
    }
}

/**
 * Select a crop by index, updating grid, labeler, and frame viewer.
 * @param {number} index
 */
function _selectCropByIndex(index) {
    if (index < 0 || index >= AppState.crops.length) return;
    AppState.currentCropIndex = index;

    const crop = AppState.crops[index];
    const grid = AppState._components.cropGrid;
    const labeler = AppState._components.cropLabeler;
    const fv = AppState._components.frameViewer;

    if (grid) grid.select(index);
    if (labeler) labeler.showCrop(crop, AppState.sessionId);
    if (fv) {
        // Always reload annotated frame with highlight — even on same frame,
        // the highlight target may have changed.
        fv.loadFrame(crop.frame_idx, AppState.sessionId, true, crop.crop_id);
    }
}

/**
 * Navigate to the prev/next sampled frame.
 * @param {number} direction - -1 for previous, +1 for next.
 */
function _navigateFrame(direction) {
    // Build a unique sorted list of frame indices from crops
    const frameSet = new Set(AppState.crops.map((c) => c.frame_idx));
    const frames = Array.from(frameSet).sort((a, b) => a - b);
    if (frames.length === 0) return;

    const fv = AppState._components.frameViewer;
    if (!fv) return;

    const currentFrame = fv.getCurrentFrame();
    let currentIdx = frames.indexOf(currentFrame);

    if (currentIdx === -1) {
        // Find closest frame
        currentIdx = frames.findIndex((f) => f >= currentFrame);
        if (currentIdx === -1) currentIdx = frames.length - 1;
    }

    const nextIdx = Math.max(0, Math.min(frames.length - 1, currentIdx + direction));
    const nextFrame = frames[nextIdx];

    AppState.currentFrameIdx = nextFrame;
    fv.loadFrame(nextFrame, AppState.sessionId);
}

/** Handle Next Round: score crops with k-NN, then detect on new frames. */
async function _onNextRound() {
    const progress = AppState._components.progressOverlay;

    try {
        progress.show('Scoring with k-NN and preparing next round...', -1);

        const job = await API.post('/detect/next_round', {
            session_id: AppState.sessionId,
        });

        pollJob(
            job.job_id,
            (p) => {
                progress.show(p.step || 'Processing...', p.percent || -1);
            },
            async (p) => {
                progress.hide();
                if (p.status === 'completed') {
                    await _refreshCrops();
                    const vh = (AppState.stats && AppState.stats.validation_history) || [];
                    const valStr = vh.length > 0
                        ? ` — Val: ${Math.round(vh[vh.length - 1].val_accuracy * 100)}%`
                        : '';
                    showToast(`Round ${job.round} ready` + valStr, 'success');
                } else {
                    showToast(`Round failed: ${p.error}`, 'error');
                }
            }
        );
    } catch (err) {
        progress.hide();
    }
}

// ---------------------------------------------------------------------------
// Reject Review Sub-Phase
// ---------------------------------------------------------------------------

var _SUBCATEGORIES = ['not_person', 'partial_box', 'oversized_box'];
var _SUBCATEGORY_LABELS = {
    not_person: 'Not a Person',
    partial_box: 'Partial Box',
    oversized_box: 'Oversized Box',
};

function _isRejectReviewSam3EligibleSubcat(subcat) {
    return subcat === 'partial_box' || subcat === 'oversized_box';
}

function _setRejectReviewSam3AutoEnabled(enabled) {
    AppState.rejectReviewSam3AutoEnabled = !!enabled;
    var toggle = document.getElementById('reject-review-sam3-auto-toggle');
    if (toggle) {
        toggle.checked = AppState.rejectReviewSam3AutoEnabled;
    }
}

function _setRejectReviewSam3BusyUI(busy) {
    var isBusy = !!busy;

    var fixBtn = document.getElementById('reject-review-fix-btn');
    if (fixBtn) {
        fixBtn.disabled = isBusy;
        fixBtn.textContent = isBusy ? 'Fixing...' : 'Fix (SAM3)';
    }

    var autoToggle = document.getElementById('reject-review-sam3-auto-toggle');
    if (autoToggle) {
        autoToggle.disabled = isBusy;
    }

    var subcatBtns = document.querySelectorAll('#reject-review-bar .subcat-btn');
    subcatBtns.forEach(function (btn) {
        btn.disabled = isBusy;
    });
}

function _blockRejectReviewWhileSam3Busy() {
    if (!AppState.rejectReviewSam3InFlight) return false;
    showToast('SAM3 refine in progress. Please wait.', 'warning');
    return true;
}

/**
 * Enter reject review mode, showing rejected crops one at a time.
 * @param {Array} rejectedCrops - Rejected crops missing a reject_reason.
 */
function _enterRejectReview(rejectedCrops) {
    AppState.rejectReviewMode = true;
    AppState.rejectReviewCrops = rejectedCrops;
    AppState.rejectReviewIndex = 0;
    AppState.rejectReviewSubcategory = 'not_person';
    AppState.rejectReviewBoxAdjusted = false;
    AppState.rejectReviewDrawActive = false;
    AppState.rejectReviewExpandBase = null;
    AppState.rejectReviewSam3InFlight = false;
    AppState.rejectReviewSam3QueuedPrompt = null;
    AppState.rejectReviewSam3ActiveRequestId = 0;

    // Change header
    var header = document.querySelector('.detection-header');
    if (header) header.textContent = 'Why was this rejected?';

    // Hide crop grid, show reject review UI
    var grid = AppState._components.cropGrid;
    if (grid && grid.el) grid.el.classList.add('hidden');

    // Hide normal crop labeler actions
    var labeler = AppState._components.cropLabeler;
    if (labeler && labeler.actionsEl) labeler.actionsEl.classList.add('hidden');
    if (labeler && labeler.hintsEl) labeler.hintsEl.classList.add('hidden');

    // Build the subcategory bar and hints (inserted into right panel)
    _buildRejectReviewUI();

    // Create BoxAdjuster if not already created
    if (!AppState._components.boxAdjuster) {
        var fv = AppState._components.frameViewer;
        if (fv) {
            AppState._components.boxAdjuster = new BoxAdjuster(fv);
            AppState._components.boxAdjuster.onBoxChanged(function (box) {
                AppState.rejectReviewBoxAdjusted = true;
                var ba = AppState._components.boxAdjuster;
                if (ba && ba.isActive()) {
                    var b = box || ba.getBox();
                    if (b && b.x1 != null && b.y1 != null && b.x2 != null && b.y2 != null) {
                        var promptXyxy = [b.x1, b.y1, b.x2, b.y2];
                        AppState.rejectReviewExpandBase = promptXyxy;

                        var crop = AppState.rejectReviewCrops[AppState.rejectReviewIndex];
                        if (
                            AppState.rejectReviewMode &&
                            AppState.rejectReviewSam3AutoEnabled &&
                            _isRejectReviewSam3EligibleSubcat(AppState.rejectReviewSubcategory) &&
                            crop
                        ) {
                            _queueRejectReviewSam3Fix(promptXyxy, crop.crop_id, 'auto');
                        }
                    }
                }
            });
        }
    }

    // Show first crop
    _showRejectReviewCrop(0);
}

/** Build the subcategory selector bar and hints into the right panel. */
function _buildRejectReviewUI() {
    // Remove any existing reject review UI
    var existing = document.getElementById('reject-review-ui');
    if (existing) existing.parentNode.removeChild(existing);

    var wrap = document.createElement('div');
    wrap.id = 'reject-review-ui';

    // Counter
    var counter = document.createElement('div');
    counter.className = 'reject-review-counter';
    counter.id = 'reject-review-counter';
    wrap.appendChild(counter);

    // Subcategory bar
    var bar = document.createElement('div');
    bar.className = 'reject-review-bar';
    bar.id = 'reject-review-bar';

    _SUBCATEGORIES.forEach(function (subcat) {
        var btn = document.createElement('button');
        btn.className = 'subcat-btn';
        btn.dataset.subcat = subcat;
        btn.textContent = _SUBCATEGORY_LABELS[subcat];
        if (subcat === AppState.rejectReviewSubcategory) {
            btn.classList.add('active');
        }
        btn.addEventListener('click', function () {
            _setSubcategory(subcat);
        });
        bar.appendChild(btn);
    });

    wrap.appendChild(bar);

    // Draw/save toggle for corrected boxes (reject-review scope only)
    var drawToggle = document.createElement('button');
    drawToggle.id = 'reject-review-draw-toggle';
    drawToggle.className = 'btn btn-secondary btn-small reject-review-draw-toggle';
    drawToggle.textContent = 'Draw OFF';
    drawToggle.addEventListener('click', function () {
        _toggleRejectReviewDraw();
    });
    wrap.appendChild(drawToggle);

    // Fix button — auto-refine box via Sam3TrackerModel (shown for partial/oversized)
    var fixBtn = document.createElement('button');
    fixBtn.id = 'reject-review-fix-btn';
    fixBtn.className = 'btn btn-secondary btn-small reject-review-fix-btn';
    fixBtn.textContent = 'Fix (SAM3)';
    fixBtn.style.display = 'none';  // shown/hidden by _setSubcategory
    fixBtn.addEventListener('click', function () {
        _fixOversizedBox();
    });
    wrap.appendChild(fixBtn);

    // Auto-SAM3 checkbox for commit-time refinement after box edit.
    var autoWrap = document.createElement('label');
    autoWrap.id = 'reject-review-sam3-auto-wrap';
    autoWrap.className = 'reject-review-sam3-auto-wrap';
    autoWrap.style.display = 'none';  // shown/hidden by _setSubcategory

    var autoToggle = document.createElement('input');
    autoToggle.type = 'checkbox';
    autoToggle.id = 'reject-review-sam3-auto-toggle';
    autoToggle.checked = !!AppState.rejectReviewSam3AutoEnabled;
    autoToggle.addEventListener('change', function () {
        _setRejectReviewSam3AutoEnabled(autoToggle.checked);
        if (!AppState.rejectReviewSam3AutoEnabled) {
            AppState.rejectReviewSam3QueuedPrompt = null;
        }
    });
    autoWrap.appendChild(autoToggle);

    var autoLabel = document.createElement('span');
    autoLabel.className = 'reject-review-sam3-auto-label';
    autoLabel.textContent = 'SAM3 Mode ON';
    autoWrap.appendChild(autoLabel);
    wrap.appendChild(autoWrap);

    // Expansion selector for pre-Fix prompt preview
    var expandWrap = document.createElement('div');
    expandWrap.id = 'reject-review-fix-expand-wrap';
    expandWrap.className = 'reject-review-fix-expand-wrap';
    expandWrap.style.display = 'none';  // shown/hidden by _setSubcategory

    var expandLabel = document.createElement('label');
    expandLabel.className = 'reject-review-fix-expand-label';
    expandLabel.setAttribute('for', 'reject-review-fix-expand-select');
    expandLabel.textContent = 'Expand before Fix';
    expandWrap.appendChild(expandLabel);

    var expandSelect = document.createElement('select');
    expandSelect.id = 'reject-review-fix-expand-select';
    expandSelect.className = 'reject-review-fix-expand-select';
    [0, 10, 25, 50, 75, 100].forEach(function (pct) {
        var opt = document.createElement('option');
        opt.value = String(pct);
        opt.textContent = pct + '%';
        expandSelect.appendChild(opt);
    });
    expandSelect.value = String(AppState.rejectReviewFixExpandPct || 0);
    expandSelect.addEventListener('change', function () {
        AppState.rejectReviewFixExpandPct = parseInt(expandSelect.value, 10) || 0;
        _previewRejectReviewExpansion();
    });
    expandWrap.appendChild(expandSelect);
    wrap.appendChild(expandWrap);

    // Keyboard hints
    var hints = document.createElement('div');
    hints.className = 'reject-review-hints';
    hints.innerHTML =
        '<span><kbd>J</kbd>/<kbd>K</kbd> Category</span>' +
        '<span><kbd>&darr;</kbd> Fix (SAM3)</span>' +
        '<span><kbd>&rarr;</kbd> Save & Next</span>' +
        '<span><kbd>&larr;</kbd> Previous</span>' +
        '<span><kbd>Esc</kbd> Exit</span>';
    wrap.appendChild(hints);

    // Insert before the labeler actions in the right panel
    var labeler = AppState._components.cropLabeler;
    if (labeler && labeler.el && labeler.el.parentNode) {
        labeler.el.parentNode.insertBefore(wrap, labeler.el.nextSibling);
    }
}

/**
 * Show a rejected crop at the given index.
 * @param {number} index
 */
function _showRejectReviewCrop(index) {
    var crops = AppState.rejectReviewCrops;
    if (index < 0 || index >= crops.length) return;

    AppState.rejectReviewIndex = index;
    AppState.rejectReviewBoxAdjusted = false;
    AppState.rejectReviewDrawActive = false;
    AppState.rejectReviewFixExpandPct = 0;
    AppState.rejectReviewExpandBase = null;
    AppState.rejectReviewSam3QueuedPrompt = null;

    var crop = crops[index];
    var base = crop.corrected_xyxy || crop.xyxy;
    if (base && base.length === 4) {
        AppState.rejectReviewExpandBase = [base[0], base[1], base[2], base[3]];
    }

    // Restore previously-chosen subcategory, or default to 'not_person'
    var initialSubcat = crop.reject_reason || 'not_person';
    AppState.rejectReviewSubcategory = initialSubcat;

    // Update counter
    var counter = document.getElementById('reject-review-counter');
    if (counter) {
        counter.textContent = (index + 1) + ' of ' + crops.length + ' rejected crops';
    }

    // Show the crop image
    var labeler = AppState._components.cropLabeler;
    if (labeler) labeler.showCrop(crop, AppState.sessionId);

    // Load the frame with highlight
    var fv = AppState._components.frameViewer;
    if (fv) {
        fv.loadFrame(crop.frame_idx, AppState.sessionId, true, crop.crop_id);
    }

    // Reset draw toggle first.
    _setRejectReviewDrawButton(false);
    _setRejectReviewExpandControl(0);
    _setRejectReviewSam3BusyUI(AppState.rejectReviewSam3InFlight);

    // Restore subcategory buttons — will re-activate adjuster for partial/oversized
    _setSubcategory(initialSubcat);
}

/**
 * Set the current subcategory selection.
 * @param {string} subcat - 'not_person', 'partial_box', or 'oversized_box'
 */
function _setSubcategory(subcat) {
    AppState.rejectReviewSubcategory = subcat;
    var sam3Eligible = _isRejectReviewSam3EligibleSubcat(subcat);

    // Update button active states
    var btns = document.querySelectorAll('#reject-review-bar .subcat-btn');
    btns.forEach(function (btn) {
        btn.classList.toggle('active', btn.dataset.subcat === subcat);
    });

    // Show/hide Fix button based on subcategory
    var fixBtn = document.getElementById('reject-review-fix-btn');
    if (fixBtn) {
        fixBtn.style.display = sam3Eligible ? '' : 'none';
    }
    var autoWrap = document.getElementById('reject-review-sam3-auto-wrap');
    if (autoWrap) {
        autoWrap.style.display = sam3Eligible ? 'flex' : 'none';
    }
    var expandWrap = document.getElementById('reject-review-fix-expand-wrap');
    if (expandWrap) {
        expandWrap.style.display = sam3Eligible ? '' : 'none';
    }

    // Activate/deactivate box adjuster based on subcategory
    var ba = AppState._components.boxAdjuster;
    if (sam3Eligible) {
        AppState.rejectReviewDrawActive = true;
        _setRejectReviewDrawButton(true);
        _previewRejectReviewExpansion();
    } else {
        AppState.rejectReviewSam3QueuedPrompt = null;
        if (ba && ba.isActive()) ba.deactivate();
        AppState.rejectReviewDrawActive = false;
        AppState.rejectReviewBoxAdjusted = false;
        AppState.rejectReviewFixExpandPct = 0;
        _setRejectReviewExpandControl(0);
        _setRejectReviewDrawButton(false);
    }

    _setRejectReviewSam3BusyUI(AppState.rejectReviewSam3InFlight);
}

/** Cycle the subcategory in the given direction (-1 or +1). */
function _cycleSubcategory(direction) {
    var idx = _SUBCATEGORIES.indexOf(AppState.rejectReviewSubcategory);
    idx = (idx + direction + _SUBCATEGORIES.length) % _SUBCATEGORIES.length;
    _setSubcategory(_SUBCATEGORIES[idx]);
}

/** Save the current reject review crop and advance to the next one. */
async function _saveAndAdvanceRejectReview() {
    if (_blockRejectReviewWhileSam3Busy()) return;

    var crops = AppState.rejectReviewCrops;
    var index = AppState.rejectReviewIndex;
    if (index < 0 || index >= crops.length) return;

    var crop = crops[index];
    var neverSaved = !crop.reject_reason;
    var reasonChanged = AppState.rejectReviewSubcategory !== (crop.reject_reason || 'not_person');
    var boxChanged = AppState.rejectReviewBoxAdjusted;
    if (neverSaved || reasonChanged || boxChanged) {
        var saved = await _saveRejectReviewCurrent(boxChanged, false);
        if (!saved) return;
    }

    if (index + 1 < crops.length) {
        _showRejectReviewCrop(index + 1);
    } else {
        showToast('All rejected crops reviewed', 'success');
        _exitRejectReview();
    }
}

/** Go back to the previous rejected crop (saves current first). */
async function _prevRejectReviewCrop() {
    if (_blockRejectReviewWhileSam3Busy()) return;

    var index = AppState.rejectReviewIndex;
    if (index <= 0) return;

    var crop = AppState.rejectReviewCrops[index];
    var neverSaved = !crop.reject_reason;
    var reasonChanged = AppState.rejectReviewSubcategory !== (crop.reject_reason || 'not_person');
    var boxChanged = AppState.rejectReviewBoxAdjusted;
    if (neverSaved || reasonChanged || boxChanged) {
        var saved = await _saveRejectReviewCurrent(boxChanged, false);
        if (!saved) return;
    }

    _showRejectReviewCrop(index - 1);
}

function _setRejectReviewDrawButton(active) {
    var btn = document.getElementById('reject-review-draw-toggle');
    if (!btn) return;
    if (active) {
        btn.classList.add('active');
        btn.textContent = 'Draw ON / Save crop';
    } else {
        btn.classList.remove('active');
        btn.textContent = 'Draw OFF';
    }
}

async function _toggleRejectReviewDraw() {
    var subcat = AppState.rejectReviewSubcategory;
    if (subcat !== 'partial_box' && subcat !== 'oversized_box') {
        showToast('Draw mode is only available for Partial/Oversized categories', 'warning');
        return;
    }

    var ba = AppState._components.boxAdjuster;
    var crop = AppState.rejectReviewCrops[AppState.rejectReviewIndex];
    if (!ba || !crop) return;

    if (!AppState.rejectReviewDrawActive) {
        var box = AppState.rejectReviewExpandBase || crop.corrected_xyxy || crop.xyxy;
        if (!ba.isActive() && box) {
            ba.activate({
                x1: box[0],
                y1: box[1],
                x2: box[2],
                y2: box[3],
            });
        }
        AppState.rejectReviewDrawActive = true;
        AppState.rejectReviewBoxAdjusted = false;
        _setRejectReviewDrawButton(true);
        return;
    }

    // Turning draw OFF: persist reason + adjusted box immediately.
    if (_blockRejectReviewWhileSam3Busy()) return;
    var saved = await _saveRejectReviewCurrent(true);
    if (!saved) return;

    if (ba.isActive()) ba.deactivate();
    AppState.rejectReviewDrawActive = false;
    _setRejectReviewDrawButton(false);
}

function _setRejectReviewExpandControl(value) {
    var select = document.getElementById('reject-review-fix-expand-select');
    if (!select) return;
    select.value = String(value);
}

function _expandBoxForPreview(box, pct, frameViewer) {
    if (!box || box.length !== 4) return null;
    if (!pct || pct <= 0 || !frameViewer) return [box[0], box[1], box[2], box[3]];
    var x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
    var bw = Math.max(0, x2 - x1);
    var bh = Math.max(0, y2 - y1);
    var frac = pct / 100.0;
    var dx = bw * frac;
    var dy = bh * frac;
    var w = frameViewer.videoWidth || 0;
    var h = frameViewer.videoHeight || 0;
    return [
        Math.max(0, x1 - dx),
        Math.max(0, y1 - dy),
        w > 0 ? Math.min(w, x2 + dx) : (x2 + dx),
        h > 0 ? Math.min(h, y2 + dy) : (y2 + dy),
    ];
}

function _sameBox(a, b, eps) {
    if (!a || !b) return false;
    var t = eps || 0.5;
    return (
        Math.abs(a[0] - b[0]) <= t &&
        Math.abs(a[1] - b[1]) <= t &&
        Math.abs(a[2] - b[2]) <= t &&
        Math.abs(a[3] - b[3]) <= t
    );
}

function _previewRejectReviewExpansion() {
    var subcat = AppState.rejectReviewSubcategory;
    if (subcat !== 'partial_box' && subcat !== 'oversized_box') return;

    var crops = AppState.rejectReviewCrops;
    var index = AppState.rejectReviewIndex;
    if (index < 0 || index >= crops.length) return;
    var crop = crops[index];
    if (!crop) return;

    var baseBox = AppState.rejectReviewExpandBase || crop.corrected_xyxy || crop.xyxy;
    if (!baseBox) return;

    var ba = AppState._components.boxAdjuster;
    var fv = AppState._components.frameViewer;
    if (!ba || !fv) return;

    var expanded = _expandBoxForPreview(baseBox, AppState.rejectReviewFixExpandPct || 0, fv);
    if (!expanded) return;

    if (ba.isActive()) {
        var existing = ba.getBox();
        if (existing && _sameBox(expanded, [existing.x1, existing.y1, existing.x2, existing.y2])) {
            return;
        }
        ba.deactivate();
    }
    ba.activate({
        x1: expanded[0],
        y1: expanded[1],
        x2: expanded[2],
        y2: expanded[3],
    });
    AppState.rejectReviewDrawActive = true;
    _setRejectReviewDrawButton(true);
}

/**
 * Auto-fix an oversized/partial box using Sam3TrackerModel.
 * Calls /api/detect/refine_box, applies the refined box to the BoxAdjuster,
 * and keeps Draw mode ON so the user can further adjust.
 */
function _queueRejectReviewSam3Fix(promptXyxy, cropId, source) {
    if (!promptXyxy || promptXyxy.length !== 4 || !cropId) return;

    var parsedPrompt = promptXyxy.map(function (v) { return Number(v); });
    if (!parsedPrompt.every(function (v) { return Number.isFinite(v); })) return;
    if (parsedPrompt[0] >= parsedPrompt[2] || parsedPrompt[1] >= parsedPrompt[3]) return;

    var queued = {
        cropId: cropId,
        promptXyxy: parsedPrompt,
        source: source === 'manual' ? 'manual' : 'auto',
    };

    if (AppState.rejectReviewSam3InFlight) {
        AppState.rejectReviewSam3QueuedPrompt = queued;
        return;
    }

    _runRejectReviewSam3Fix(queued.promptXyxy, queued.cropId, queued.source);
}

async function _runRejectReviewSam3Fix(promptXyxy, cropId, source) {
    AppState.rejectReviewSam3InFlight = true;
    AppState.rejectReviewSam3ActiveRequestId += 1;
    var requestId = AppState.rejectReviewSam3ActiveRequestId;
    var isManual = source === 'manual';
    _setRejectReviewSam3BusyUI(true);

    try {
        var result = await API.post('/detect/refine_box', {
            session_id: AppState.sessionId,
            crop_id: cropId,
            prompt_xyxy: promptXyxy,
        });

        if (requestId !== AppState.rejectReviewSam3ActiveRequestId) return;
        if (!AppState.rejectReviewMode) return;

        if (result.error) {
            showToast((isManual ? 'Fix failed: ' : 'Auto SAM3 failed: ') + result.error, 'error');
            return;
        }

        var refined = result.refined_xyxy;
        if (!refined || refined.length !== 4) {
            showToast(isManual ? 'Fix returned invalid box' : 'Auto SAM3 returned invalid box', 'error');
            return;
        }

        var crop = AppState.rejectReviewCrops[AppState.rejectReviewIndex];
        if (!crop || crop.crop_id !== cropId) return;

        var ba = AppState._components.boxAdjuster;
        if (ba) {
            if (ba.isActive()) ba.deactivate();
            ba.activate({
                x1: refined[0],
                y1: refined[1],
                x2: refined[2],
                y2: refined[3],
            });
        }

        AppState.rejectReviewBoxAdjusted = true;
        AppState.rejectReviewDrawActive = true;
        _setRejectReviewDrawButton(true);

        crop.corrected_xyxy = refined;
        AppState.rejectReviewExpandBase = [refined[0], refined[1], refined[2], refined[3]];

        if (isManual) {
            var conf = (result.confidence * 100).toFixed(0);
            showToast('Box refined (IoU ' + conf + '%). Adjust if needed, then save.', 'success');
        }
    } catch (err) {
        showToast(
            (isManual ? 'Fix failed: ' : 'Auto SAM3 failed: ') + (err.message || err),
            'error'
        );
    } finally {
        if (requestId !== AppState.rejectReviewSam3ActiveRequestId) return;

        var queued = AppState.rejectReviewSam3QueuedPrompt;
        AppState.rejectReviewSam3QueuedPrompt = null;

        if (queued) {
            var currentCrop = AppState.rejectReviewCrops[AppState.rejectReviewIndex];
            var canRunQueued = (
                AppState.rejectReviewMode &&
                currentCrop &&
                currentCrop.crop_id === queued.cropId &&
                _isRejectReviewSam3EligibleSubcat(AppState.rejectReviewSubcategory) &&
                (queued.source === 'manual' || AppState.rejectReviewSam3AutoEnabled)
            );
            if (canRunQueued) {
                _runRejectReviewSam3Fix(queued.promptXyxy, queued.cropId, queued.source);
                return;
            }
        }

        AppState.rejectReviewSam3InFlight = false;
        _setRejectReviewSam3BusyUI(false);
    }
}

function _fixOversizedBox() {
    var subcat = AppState.rejectReviewSubcategory;
    if (!_isRejectReviewSam3EligibleSubcat(subcat)) {
        showToast('Fix is only available for Partial/Oversized categories', 'warning');
        return;
    }

    var crop = AppState.rejectReviewCrops[AppState.rejectReviewIndex];
    if (!crop) return;

    var ba = AppState._components.boxAdjuster;
    var promptXyxy = null;
    if (ba && ba.isActive()) {
        var activeBox = ba.getBox();
        if (activeBox) {
            promptXyxy = [activeBox.x1, activeBox.y1, activeBox.x2, activeBox.y2];
        }
    }
    if (!promptXyxy) {
        promptXyxy = crop.corrected_xyxy || crop.xyxy;
    }
    if (!promptXyxy) return;

    _queueRejectReviewSam3Fix(promptXyxy, crop.crop_id, 'manual');
}

async function _saveRejectReviewCurrent(includeAdjustedBox, reloadFrame) {
    var crops = AppState.rejectReviewCrops;
    var index = AppState.rejectReviewIndex;
    if (index < 0 || index >= crops.length) return false;
    var shouldReload = (reloadFrame !== false);

    var crop = crops[index];
    var ba = AppState._components.boxAdjuster;
    var adjustedXyxy = null;
    if (includeAdjustedBox && ba && ba.isActive()) {
        var box = ba.getBox();
        if (box) {
            adjustedXyxy = [box.x1, box.y1, box.x2, box.y2];
        }
    }

    try {
        var result = await API.post('/detect/subcategorize', {
            session_id: AppState.sessionId,
            crop_id: crop.crop_id,
            reject_reason: AppState.rejectReviewSubcategory,
            adjusted_xyxy: adjustedXyxy,
        });
        crop.reject_reason = AppState.rejectReviewSubcategory;
        if (adjustedXyxy) {
            crop.corrected_xyxy = adjustedXyxy;
            AppState.rejectReviewExpandBase = [
                adjustedXyxy[0], adjustedXyxy[1], adjustedXyxy[2], adjustedXyxy[3],
            ];
        } else if (AppState.rejectReviewSubcategory === 'not_person') {
            crop.corrected_xyxy = null;
            AppState.rejectReviewExpandBase = null;
        }
        AppState.stats = result;
        _renderToolbar();
        var fv = AppState._components.frameViewer;
        if (fv && shouldReload) {
            fv.reload(AppState.sessionId, crop.crop_id);
        }
        return true;
    } catch (err) {
        return false;
    }
}

/** Exit reject review mode, restoring normal detection UI. */
function _exitRejectReview() {
    if (_blockRejectReviewWhileSam3Busy()) return;

    var unreviewed = AppState.rejectReviewCrops.filter(function (c) {
        return !c.reject_reason;
    });
    if (unreviewed.length > 0) {
        showToast('Please assign a subcategory to all rejected crops before exiting.', 'warning');
        return;
    }

    // Guard against discarding unsaved box edits on current crop (e.g., Esc exit).
    if (AppState.rejectReviewBoxAdjusted) {
        var discard = window.confirm(
            'You have unsaved box changes for this crop. Exit reject review and discard these changes?'
        );
        if (!discard) return;
    }

    AppState.rejectReviewMode = false;
    AppState.rejectReviewCrops = [];
    AppState.rejectReviewIndex = 0;
    AppState.rejectReviewBoxAdjusted = false;
    AppState.rejectReviewDrawActive = false;
    AppState.rejectReviewExpandBase = null;
    AppState.rejectReviewSam3InFlight = false;
    AppState.rejectReviewSam3QueuedPrompt = null;
    AppState.rejectReviewSam3ActiveRequestId = 0;

    // Deactivate box adjuster
    var ba = AppState._components.boxAdjuster;
    if (ba && ba.isActive()) ba.deactivate();

    // Restore header
    var header = document.querySelector('.detection-header');
    if (header) header.textContent = 'Is this a good crop?';

    // Remove reject review UI
    var ui = document.getElementById('reject-review-ui');
    if (ui && ui.parentNode) ui.parentNode.removeChild(ui);

    // Show crop grid again
    var grid = AppState._components.cropGrid;
    if (grid && grid.el) grid.el.classList.remove('hidden');

    // Show normal labeler actions
    var labeler = AppState._components.cropLabeler;
    if (labeler && labeler.actionsEl) labeler.actionsEl.classList.remove('hidden');
    if (labeler && labeler.hintsEl) labeler.hintsEl.classList.remove('hidden');

    // Refresh crops
    _refreshCrops();
}

// ---------------------------------------------------------------------------
// Phase Renderer: ReID
// ---------------------------------------------------------------------------

/**
 * Render the Re-Identification comparison interface.
 * Delegates to renderReIDPhase (defined in reid_ui.js) if available,
 * otherwise renders a basic fallback.
 * @param {HTMLElement} app
 */
function renderReID(app) {
    if (!AppState.sessionId) {
        showToast('No active session. Redirecting to setup.', 'warning');
        navigate('setup');
        return;
    }

    // If reid_ui.js provides an extended renderer, use it
    if (typeof renderReIDPhase === 'function') {
        renderReIDPhase(app);
        return;
    }

    // Fallback: reid_ui.js not loaded
    const panel = document.createElement('div');
    panel.style.cssText = 'padding:40px;text-align:center;';
    panel.innerHTML =
        '<h2>Re-Identification</h2>' +
        '<p style="color:var(--text-secondary);">reid_ui.js is required but was not loaded.</p>';
    app.appendChild(panel);
}

// ---------------------------------------------------------------------------
// Phase Renderer: Seeding
// ---------------------------------------------------------------------------

/**
 * Render the seeding configuration and upload interface.
 * Delegates to renderSeedingPhase (defined in seeding_ui.js).
 * @param {HTMLElement} app
 */
function renderSeeding(app) {
    if (!AppState.sessionId) {
        showToast('No active session. Redirecting to setup.', 'warning');
        navigate('setup');
        return;
    }

    if (typeof renderSeedingPhase === 'function') {
        renderSeedingPhase(app);
        return;
    }

    // seeding_ui.js not loaded
    const panel = document.createElement('div');
    panel.style.cssText = 'padding:40px;text-align:center;';
    panel.innerHTML =
        '<h2>Seeding</h2>' +
        '<p style="color:var(--text-secondary);">seeding_ui.js is required but was not loaded.</p>';
    app.appendChild(panel);
}

// ---------------------------------------------------------------------------
// Background Embedding Progress
// ---------------------------------------------------------------------------

/** Start polling embedding status and show a progress banner. */
function _startEmbeddingPoll() {
    if (AppState._embeddingPollCancel) {
        AppState._embeddingPollCancel();
    }

    _showEmbeddingBanner('Analyzing video in background...');

    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 5;

    const cancel = setInterval(async () => {
        try {
            const status = await API.get('/detect/embedding_status', {
                session_id: AppState.sessionId,
            });

            consecutiveErrors = 0; // Reset on success

            if (status.embedding_complete) {
                clearInterval(cancel);
                AppState._embeddingPollCancel = null;
                _hideEmbeddingBanner();
                showToast(
                    `Change detection ready (${status.change_keyframes_count} keyframes)`,
                    'success'
                );
                return;
            }

            if (status.progress) {
                const p = status.progress;
                const current = (p.current || 0).toLocaleString();
                const total = (p.total || 0).toLocaleString();
                const pct = p.percent > 0 ? ` (${Math.round(p.percent)}%)` : '';
                _showEmbeddingBanner(
                    `Background: Analyzing frames ${current} / ${total}${pct}`
                );
            }
        } catch (err) {
            consecutiveErrors++;
            if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                clearInterval(cancel);
                AppState._embeddingPollCancel = null;
                _hideEmbeddingBanner();
                console.warn('Embedding poll stopped after repeated failures');
            }
        }
    }, 2000);

    AppState._embeddingPollCancel = () => clearInterval(cancel);
}

/** Show or update the embedding progress banner. */
function _showEmbeddingBanner(text) {
    let banner = document.getElementById('embedding-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'embedding-banner';
        banner.style.cssText =
            'position:fixed;bottom:0;left:0;right:0;z-index:100;' +
            'background:var(--bg-surface,#1a1a2e);' +
            'border-top:1px solid var(--border-default,#333);' +
            'padding:6px 16px;font-size:0.8rem;' +
            'color:var(--text-secondary,#aaa);text-align:center;' +
            'transition:opacity 0.3s;';
        document.body.appendChild(banner);
    }
    banner.textContent = text;
    banner.style.opacity = '1';
}

/** Remove the embedding progress banner. */
function _hideEmbeddingBanner() {
    const banner = document.getElementById('embedding-banner');
    if (banner) {
        banner.style.opacity = '0';
        setTimeout(() => {
            if (banner.parentNode) banner.parentNode.removeChild(banner);
        }, 300);
    }
}

// ---------------------------------------------------------------------------
// Keyboard Shortcuts
// ---------------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
    // Skip if user is typing in an input or textarea
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

    const phase = AppState.phase;

    // Detection / Classification shortcuts
    if (phase === 'detection' || phase === 'classification') {
        // Reject review mode has its own shortcuts
        if (AppState.rejectReviewMode) {
            if (e.key === 'j' || e.key === 'J') {
                e.preventDefault();
                if (_blockRejectReviewWhileSam3Busy()) return;
                _cycleSubcategory(-1);
            } else if (e.key === 'k' || e.key === 'K') {
                e.preventDefault();
                if (_blockRejectReviewWhileSam3Busy()) return;
                _cycleSubcategory(1);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                _fixOversizedBox();
            } else if (e.key === 'ArrowRight' || e.key === 'Enter') {
                e.preventDefault();
                if (_blockRejectReviewWhileSam3Busy()) return;
                _saveAndAdvanceRejectReview();
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                if (_blockRejectReviewWhileSam3Busy()) return;
                _prevRejectReviewCrop();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                if (_blockRejectReviewWhileSam3Busy()) return;
                _exitRejectReview();
            }
            return;
        }

        if (e.key === 'Enter') {
            e.preventDefault();
            _acceptCurrentCrop();
        } else if (e.key === 'Backspace') {
            e.preventDefault();
            _rejectCurrentCrop();
        } else if (e.key === 's' || e.key === 'S') {
            e.preventDefault();
            _skipCurrentCrop();
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            _nextCrop();
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault();
            _prevCrop();
        }
    }

    // ReID shortcuts are handled by reid_ui.js internally
});

/** Accept the current crop via keyboard. */
function _acceptCurrentCrop() {
    const crop = AppState.crops[AppState.currentCropIndex];
    if (crop && crop.label === 'pending') {
        _labelCrop(crop, 'accepted');
    }
}

/** Reject the current crop via keyboard. */
function _rejectCurrentCrop() {
    const crop = AppState.crops[AppState.currentCropIndex];
    if (crop && crop.label === 'pending') {
        _labelCrop(crop, 'rejected');
    }
}

/** Skip the current crop via keyboard (excluded from training). */
function _skipCurrentCrop() {
    const crop = AppState.crops[AppState.currentCropIndex];
    if (crop && crop.label === 'pending') {
        _labelCrop(crop, 'skipped');
    }
}

/** Move to the next crop. */
function _nextCrop() {
    const next = Math.min(AppState.currentCropIndex + 1, AppState.crops.length - 1);
    _selectCropByIndex(next);
}

/** Move to the previous crop. */
function _prevCrop() {
    const prev = Math.max(AppState.currentCropIndex - 1, 0);
    _selectCropByIndex(prev);
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // Set up hash-based routing
    window.addEventListener('hashchange', _onHashChange);

    // Wire phase dot click navigation
    _initPhaseNavigation();

    // Parse initial hash
    const hash = window.location.hash.replace(/^#\/?/, '') || 'setup';
    AppState.phase = hash;
    renderPhase(hash);
    updatePhaseIndicator(hash);
});
