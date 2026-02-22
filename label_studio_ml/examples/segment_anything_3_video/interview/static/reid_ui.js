/**
 * ReID UI - UFM-based cluster gallery with multi-select crop assignment.
 *
 * Shows crops organized by cluster. User can select multiple crops
 * and move them to a different cluster or create a new cluster.
 * Uses UFM covisibility for initial clustering (HAC with user-specified k).
 *
 * Relies on the global API object (from app.js) for HTTP requests
 * and showToast() / pollJob() / navigate() for notifications & routing.
 */

/* exported renderReIDPhase */
/* global API, showToast, AppState, navigate, pollJob */

// ---------------------------------------------------------------------------
// Entry point called by app.js renderReID()
// ---------------------------------------------------------------------------

function renderReIDPhase(app) {
    var ui = new _ReIDUI(app);
    ui.init(AppState.sessionId);
    // Store for cleanup
    AppState._components.reidUI = ui;
}

// ---------------------------------------------------------------------------
// Main ReID class
// ---------------------------------------------------------------------------

function _ReIDUI(container) {
    this.container = container;
    this.sessionId = null;
    this.clusters = {};       // {cluster_id: {crop_ids, count}}
    this.nIdentities = 0;
    this.ufmComplete = false;
    this.warnings = [];       // co-occurrence warnings
    this.selected = {};       // {crop_id: true} — selected crops
    this._boundKeyHandler = this._handleKeyDown.bind(this);
}

function _isUfmPairsStep(step) {
    return typeof step === 'string' && step.indexOf('UFM pairs:') === 0;
}

function _formatEtaSeconds(seconds) {
    if (!isFinite(seconds) || seconds < 0) return null;
    var s = Math.max(0, Math.round(seconds));
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = s % 60;
    if (h > 0) {
        return h + ':' + String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
    }
    return m + ':' + String(sec).padStart(2, '0');
}

_ReIDUI.prototype.init = async function (sessionId) {
    this.sessionId = sessionId;
    this.selected = {};

    try {
        var data = await API.get('/reid/clusters', {
            session_id: sessionId,
        });
        this.clusters = data.clusters || {};
        this.nIdentities = data.n_identities || 0;
        this.ufmComplete = data.ufm_complete || false;
        this.warnings = data.co_occurrence_warnings || [];
    } catch (err) {
        showToast('Failed to load ReID data: ' + err.message, 'error');
        return;
    }

    // No clusters yet — show start screen
    if (this.nIdentities === 0 && Object.keys(this.clusters).length === 0) {
        this._renderStartScreen();
        return;
    }

    // Have clusters — show gallery
    document.addEventListener('keydown', this._boundKeyHandler);
    this._renderGallery();
};

_ReIDUI.prototype.destroy = function () {
    document.removeEventListener('keydown', this._boundKeyHandler);
};

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

_ReIDUI.prototype._handleKeyDown = function (e) {
    // Escape: clear selection
    if (e.key === 'Escape') {
        this.selected = {};
        this._updateSelectionUI();
        return;
    }
    // Ctrl+A / Cmd+A: select all in focused cluster
    if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        var focused = document.querySelector('.reid-cluster-section:hover');
        if (focused) {
            e.preventDefault();
            var cid = focused.dataset.clusterId;
            if (cid && this.clusters[cid]) {
                var cropIds = this.clusters[cid].crop_ids || [];
                for (var i = 0; i < cropIds.length; i++) {
                    this.selected[cropIds[i]] = true;
                }
                this._updateSelectionUI();
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Start screen (no clusters yet)
// ---------------------------------------------------------------------------

_ReIDUI.prototype._renderStartScreen = function () {
    this.container.innerHTML = '';

    var panel = document.createElement('div');
    panel.className = 'seeding-panel';
    panel.style.marginTop = '40px';

    var box = document.createElement('div');
    box.className = 'session-setup';
    box.style.maxWidth = '600px';

    var h2 = document.createElement('h2');
    h2.textContent = 'Re-Identification';
    box.appendChild(h2);

    var desc = document.createElement('p');
    desc.style.color = 'var(--text-secondary)';
    desc.style.fontSize = '0.85rem';
    desc.style.marginBottom = '20px';
    desc.textContent =
        'Cluster accepted crops into identities using UFM covisibility. ' +
        'You must specify the number of people in the video.';
    box.appendChild(desc);

    // K input (REQUIRED)
    var kGroup = document.createElement('div');
    kGroup.className = 'form-group';
    var kLabel = document.createElement('label');
    kLabel.textContent = 'Number of people in the video';
    kLabel.setAttribute('for', 'reid-k-input');
    kGroup.appendChild(kLabel);
    var kInput = document.createElement('input');
    kInput.type = 'number';
    kInput.id = 'reid-k-input';
    kInput.min = '2';
    kInput.max = '50';
    kInput.placeholder = 'e.g. 4';
    kInput.value = '';
    kInput.style.maxWidth = '120px';
    kInput.required = true;
    kGroup.appendChild(kInput);
    box.appendChild(kGroup);

    var actions = document.createElement('div');
    actions.className = 'session-actions';

    var self = this;

    var btnStart = document.createElement('button');
    btnStart.className = 'btn btn-primary';
    btnStart.textContent = 'Start Clustering';
    btnStart.addEventListener('click', function () {
        var val = kInput.value.trim();
        if (!val || parseInt(val, 10) < 2) {
            showToast('Please enter the number of people (at least 2)', 'warning');
            kInput.focus();
            return;
        }
        self._runClustering(parseInt(val, 10));
    });
    actions.appendChild(btnStart);

    var btnSkip = document.createElement('button');
    btnSkip.className = 'btn btn-ghost';
    btnSkip.textContent = 'Skip to Seeding';
    btnSkip.addEventListener('click', function () {
        if (typeof navigate === 'function') navigate('seeding');
    });
    actions.appendChild(btnSkip);

    var btnBack = document.createElement('button');
    btnBack.className = 'btn btn-ghost';
    btnBack.textContent = 'Back to Detection';
    btnBack.addEventListener('click', function () {
        if (typeof navigate === 'function') navigate('detection');
    });
    actions.appendChild(btnBack);

    box.appendChild(actions);

    // Progress area (hidden until clustering starts)
    var progressArea = document.createElement('div');
    progressArea.id = 'reid-progress-area';
    progressArea.style.marginTop = '20px';
    box.appendChild(progressArea);

    panel.appendChild(box);
    this.container.appendChild(panel);
};

// ---------------------------------------------------------------------------
// Run clustering (start or recluster)
// ---------------------------------------------------------------------------

_ReIDUI.prototype._runClustering = async function (nClusters) {
    var progressArea = document.getElementById('reid-progress-area');
    if (progressArea) {
        progressArea.innerHTML =
            '<div class="progress-bar-wrapper">' +
            '<div class="progress-bar-track"><div class="progress-bar-fill indeterminate"></div></div>' +
            '<div class="progress-text"><span>Starting UFM clustering...</span></div>' +
            '</div>';
    }

    var self = this;
    var prevCurrent = null;
    var prevTotal = null;
    var prevTs = null;
    var smoothRate = null;

    try {
        // Use recluster if UFM matrix already exists (instant), else start
        var endpoint = this.ufmComplete ? '/reid/recluster' : '/reid/start';
        var resp = await API.post(endpoint, {
            session_id: this.sessionId,
            n_clusters: nClusters,
        });

        // If instant recluster (no job_id), go straight to gallery
        if (!resp.job_id) {
            showToast('Re-clustered into ' + (resp.n_clusters || 0) + ' groups', 'success');
            self.init(self.sessionId);
            return;
        }

        // Background job — poll
        if (typeof pollJob === 'function') {
            pollJob(
                resp.job_id,
                function (p) {
                    if (progressArea) {
                        var text = progressArea.querySelector('.progress-text span');
                        if (text) {
                            var stepText = p.step || 'Clustering...';
                            if (_isUfmPairsStep(stepText)) {
                                var now = Date.now() / 1000.0;
                                if (
                                    prevTs !== null &&
                                    typeof p.current === 'number' &&
                                    typeof p.total === 'number' &&
                                    p.total === prevTotal &&
                                    p.current >= prevCurrent
                                ) {
                                    var dt = now - prevTs;
                                    var dPairs = p.current - prevCurrent;
                                    if (dt > 0 && dPairs >= 0) {
                                        var instRate = dPairs / dt;
                                        if (instRate > 0) {
                                            smoothRate = smoothRate === null
                                                ? instRate
                                                : (0.3 * instRate + 0.7 * smoothRate);
                                        }
                                    }
                                }
                                prevCurrent = (typeof p.current === 'number') ? p.current : prevCurrent;
                                prevTotal = (typeof p.total === 'number') ? p.total : prevTotal;
                                prevTs = now;

                                var rate = null;
                                if (typeof p.items_per_second === 'number' && p.items_per_second > 0) {
                                    rate = p.items_per_second;
                                } else if (smoothRate !== null && smoothRate > 0) {
                                    rate = smoothRate;
                                }

                                var eta = null;
                                if (typeof p.eta_seconds === 'number' && p.eta_seconds >= 0) {
                                    eta = p.eta_seconds;
                                } else if (
                                    rate !== null &&
                                    typeof p.total === 'number' &&
                                    typeof p.current === 'number' &&
                                    p.total >= p.current
                                ) {
                                    eta = (p.total - p.current) / rate;
                                }

                                var etaText = _formatEtaSeconds(eta);
                                if (etaText) {
                                    stepText += ' • ETA ' + etaText;
                                }
                                if (rate !== null && isFinite(rate)) {
                                    stepText += ' • ' + (rate >= 100 ? rate.toFixed(0) : rate.toFixed(1)) + ' pairs/s';
                                }
                            } else {
                                prevCurrent = null;
                                prevTotal = null;
                                prevTs = null;
                                smoothRate = null;
                            }
                            text.textContent = stepText;
                        }
                        var fill = progressArea.querySelector('.progress-bar-fill');
                        if (fill && p.percent > 0) {
                            fill.classList.remove('indeterminate');
                            fill.style.width = p.percent + '%';
                        }
                    }
                },
                function (p) {
                    if (p.status === 'completed') {
                        showToast('Clustering complete!', 'success');
                        self.init(self.sessionId);
                    } else {
                        showToast('Clustering failed: ' + (p.error || 'unknown error'), 'error');
                        if (progressArea) progressArea.innerHTML = '';
                    }
                },
                1000
            );
        }
    } catch (err) {
        showToast('Clustering failed: ' + err.message, 'error');
        if (progressArea) progressArea.innerHTML = '';
    }
};

// ---------------------------------------------------------------------------
// Cluster gallery view
// ---------------------------------------------------------------------------

_ReIDUI.prototype._renderGallery = function () {
    this.container.innerHTML = '';

    var panel = document.createElement('div');
    panel.className = 'reid-panel';
    panel.style.minWidth = '0';
    panel.style.maxWidth = '1400px';
    panel.style.margin = '0 auto';

    // --- Header bar ---
    var header = document.createElement('div');
    header.style.cssText =
        'display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:16px;';

    var h2 = document.createElement('h2');
    h2.style.fontSize = '1.1rem';
    h2.style.margin = '0';
    h2.textContent = 'Re-Identification (' + this.nIdentities + ' identities)';
    header.appendChild(h2);

    // Recluster controls
    var reclusterWrap = document.createElement('div');
    reclusterWrap.style.cssText = 'display:flex; align-items:center; gap:6px;';
    var kInput = document.createElement('input');
    kInput.type = 'number';
    kInput.min = '2';
    kInput.max = '50';
    kInput.placeholder = 'k';
    kInput.value = this.nIdentities || '';
    kInput.style.cssText = 'width:60px; padding:4px 6px; font-size:0.8rem;';
    reclusterWrap.appendChild(kInput);

    var self = this;

    var btnRecluster = document.createElement('button');
    btnRecluster.className = 'btn btn-secondary';
    btnRecluster.style.fontSize = '0.8rem';
    btnRecluster.textContent = 'Re-cluster';
    btnRecluster.addEventListener('click', function () {
        var val = kInput.value.trim();
        if (!val || parseInt(val, 10) < 2) {
            showToast('Enter k >= 2', 'warning');
            return;
        }
        self._runClustering(parseInt(val, 10));
    });
    reclusterWrap.appendChild(btnRecluster);
    header.appendChild(reclusterWrap);

    // Navigation buttons
    var btnSeeding = document.createElement('button');
    btnSeeding.className = 'btn btn-primary';
    btnSeeding.style.fontSize = '0.8rem';
    btnSeeding.style.marginLeft = 'auto';
    btnSeeding.textContent = 'Proceed to Seeding';
    btnSeeding.addEventListener('click', function () {
        if (typeof navigate === 'function') navigate('seeding');
    });
    header.appendChild(btnSeeding);

    var btnDetect = document.createElement('button');
    btnDetect.className = 'btn btn-ghost';
    btnDetect.style.fontSize = '0.8rem';
    btnDetect.textContent = 'Back to Detection';
    btnDetect.addEventListener('click', function () {
        if (typeof navigate === 'function') navigate('detection');
    });
    header.appendChild(btnDetect);

    panel.appendChild(header);

    // --- Co-occurrence warnings ---
    if (this.warnings.length > 0) {
        var warnBox = document.createElement('div');
        warnBox.style.cssText =
            'background:var(--warning-bg, #332800); border:1px solid var(--color-pending); ' +
            'border-radius:var(--radius-sm); padding:8px 12px; margin-bottom:12px; font-size:0.8rem;';
        warnBox.innerHTML =
            '<strong>Warning:</strong> ' + this.warnings.length +
            ' co-occurrence conflict(s) — crops from the same frame assigned to the same cluster. ' +
            'Consider moving them to separate clusters.';
        panel.appendChild(warnBox);
    }

    // --- Selection action bar (hidden until crops selected) ---
    var actionBar = document.createElement('div');
    actionBar.id = 'reid-action-bar';
    actionBar.style.cssText =
        'display:none; position:sticky; top:0; z-index:100; ' +
        'background:var(--bg-secondary); border:1px solid var(--highlight); ' +
        'border-radius:var(--radius-sm); padding:8px 12px; margin-bottom:12px; ' +
        'gap:8px; align-items:center; flex-wrap:wrap;';

    var selCount = document.createElement('span');
    selCount.id = 'reid-sel-count';
    selCount.style.fontSize = '0.85rem';
    selCount.style.fontWeight = '600';
    actionBar.appendChild(selCount);

    // "Move to cluster" buttons — one per existing cluster
    var moveLabel = document.createElement('span');
    moveLabel.style.cssText = 'font-size:0.8rem; color:var(--text-secondary); margin-left:8px;';
    moveLabel.textContent = 'Move to:';
    actionBar.appendChild(moveLabel);

    var clusterKeys = Object.keys(this.clusters).sort(function (a, b) {
        return parseInt(a, 10) - parseInt(b, 10);
    });
    for (var ci = 0; ci < clusterKeys.length; ci++) {
        (function (clusterId) {
            var btn = document.createElement('button');
            btn.className = 'btn btn-secondary';
            btn.style.cssText = 'font-size:0.75rem; padding:3px 8px;';
            btn.textContent = 'Cluster ' + clusterId;
            btn.addEventListener('click', function () {
                self._assignSelected(parseInt(clusterId, 10));
            });
            actionBar.appendChild(btn);
        })(clusterKeys[ci]);
    }

    var btnNew = document.createElement('button');
    btnNew.className = 'btn btn-primary';
    btnNew.style.cssText = 'font-size:0.75rem; padding:3px 8px;';
    btnNew.textContent = '+ New Cluster';
    btnNew.addEventListener('click', function () {
        self._createNewCluster();
    });
    actionBar.appendChild(btnNew);

    var btnClear = document.createElement('button');
    btnClear.className = 'btn btn-ghost';
    btnClear.style.cssText = 'font-size:0.75rem; padding:3px 8px; margin-left:auto;';
    btnClear.textContent = 'Clear Selection';
    btnClear.addEventListener('click', function () {
        self.selected = {};
        self._updateSelectionUI();
    });
    actionBar.appendChild(btnClear);

    panel.appendChild(actionBar);

    // --- Cluster sections ---
    var gallery = document.createElement('div');
    gallery.className = 'reid-cluster-gallery';
    gallery.id = 'reid-gallery';

    for (var i = 0; i < clusterKeys.length; i++) {
        var cid = clusterKeys[i];
        var cluster = this.clusters[cid];
        var section = this._buildClusterSection(cid, cluster);
        gallery.appendChild(section);
    }

    panel.appendChild(gallery);
    this.container.appendChild(panel);
};

// ---------------------------------------------------------------------------
// Build a single cluster section
// ---------------------------------------------------------------------------

_ReIDUI.prototype._buildClusterSection = function (clusterId, cluster) {
    var section = document.createElement('div');
    section.className = 'reid-cluster-section';
    section.dataset.clusterId = clusterId;

    // Header
    var header = document.createElement('div');
    header.className = 'reid-cluster-header';
    header.innerHTML =
        '<span style="font-weight:600;">Cluster ' + clusterId + '</span>' +
        '<span style="color:var(--text-secondary); font-size:0.8rem;">' +
        (cluster.count || 0) + ' crops</span>';

    // Check if this cluster has co-occurrence warnings
    var hasWarning = false;
    for (var w = 0; w < this.warnings.length; w++) {
        if (String(this.warnings[w].cluster_id) === String(clusterId)) {
            hasWarning = true;
            break;
        }
    }
    if (hasWarning) {
        var badge = document.createElement('span');
        badge.style.cssText =
            'background:var(--color-pending); color:#000; font-size:0.7rem; ' +
            'padding:1px 6px; border-radius:8px; font-weight:600;';
        badge.textContent = 'co-occurrence';
        header.appendChild(badge);
    }

    section.appendChild(header);

    // Crop thumbnails
    var row = document.createElement('div');
    row.className = 'reid-cluster-row';

    var cropIds = cluster.crop_ids || [];
    var self = this;

    for (var i = 0; i < cropIds.length; i++) {
        (function (cropId) {
            var wrap = document.createElement('div');
            wrap.className = 'reid-thumb-wrap';
            wrap.dataset.cropId = cropId;

            // Checkbox overlay
            var cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.className = 'reid-crop-checkbox';
            cb.style.cssText =
                'position:absolute; top:2px; left:2px; z-index:10; ' +
                'width:16px; height:16px; cursor:pointer;';
            cb.checked = !!self.selected[cropId];
            cb.addEventListener('change', function () {
                if (cb.checked) {
                    self.selected[cropId] = true;
                } else {
                    delete self.selected[cropId];
                }
                self._updateSelectionUI();
            });
            wrap.appendChild(cb);

            var img = document.createElement('img');
            img.src = '/interview/api/detect/crop/' + cropId + '/image?session_id=' + self.sessionId;
            img.alt = cropId;
            img.title = cropId;
            img.loading = 'lazy';
            img.addEventListener('click', function (e) {
                // Click on image toggles selection (unless clicking checkbox)
                if (e.target === cb) return;
                cb.checked = !cb.checked;
                if (cb.checked) {
                    self.selected[cropId] = true;
                } else {
                    delete self.selected[cropId];
                }
                self._updateSelectionUI();
            });
            wrap.appendChild(img);

            row.appendChild(wrap);
        })(cropIds[i]);
    }

    section.appendChild(row);
    return section;
};

// ---------------------------------------------------------------------------
// Selection UI update
// ---------------------------------------------------------------------------

_ReIDUI.prototype._updateSelectionUI = function () {
    var count = Object.keys(this.selected).length;
    var actionBar = document.getElementById('reid-action-bar');

    if (actionBar) {
        actionBar.style.display = count > 0 ? 'flex' : 'none';
        var counter = document.getElementById('reid-sel-count');
        if (counter) {
            counter.textContent = count + ' crop' + (count !== 1 ? 's' : '') + ' selected';
        }
    }

    // Update checkbox states and visual highlight
    var allWraps = document.querySelectorAll('.reid-thumb-wrap');
    for (var i = 0; i < allWraps.length; i++) {
        var wrap = allWraps[i];
        var cid = wrap.dataset.cropId;
        var cb = wrap.querySelector('.reid-crop-checkbox');
        var isSelected = !!this.selected[cid];
        if (cb) cb.checked = isSelected;

        if (isSelected) {
            wrap.style.outline = '2px solid var(--highlight)';
            wrap.style.outlineOffset = '-2px';
        } else {
            wrap.style.outline = '';
            wrap.style.outlineOffset = '';
        }
    }
};

// ---------------------------------------------------------------------------
// Assign selected crops to an existing cluster
// ---------------------------------------------------------------------------

_ReIDUI.prototype._assignSelected = async function (targetClusterId) {
    var cropIds = Object.keys(this.selected);
    if (cropIds.length === 0) return;

    try {
        var result = await API.post('/reid/assign', {
            session_id: this.sessionId,
            crop_ids: cropIds,
            target_cluster_id: targetClusterId,
        });
        this.selected = {};
        this.clusters = result.clusters || {};
        this.nIdentities = result.n_identities || 0;
        this.warnings = result.co_occurrence_warnings || [];
        showToast(
            'Moved ' + cropIds.length + ' crop(s) to cluster ' + targetClusterId,
            'success'
        );
        this._renderGallery();
    } catch (err) {
        showToast('Assignment failed: ' + err.message, 'error');
    }
};

// ---------------------------------------------------------------------------
// Create new cluster from selected crops
// ---------------------------------------------------------------------------

_ReIDUI.prototype._createNewCluster = async function () {
    var cropIds = Object.keys(this.selected);
    if (cropIds.length === 0) return;

    try {
        var result = await API.post('/reid/new_cluster', {
            session_id: this.sessionId,
            crop_ids: cropIds,
        });
        this.selected = {};
        this.clusters = result.clusters || {};
        this.nIdentities = result.n_identities || 0;
        this.warnings = result.co_occurrence_warnings || [];
        showToast(
            'Created cluster ' + (result.new_cluster_id || '?') +
            ' with ' + cropIds.length + ' crop(s)',
            'success'
        );
        this._renderGallery();
    } catch (err) {
        showToast('Create cluster failed: ' + err.message, 'error');
    }
};
