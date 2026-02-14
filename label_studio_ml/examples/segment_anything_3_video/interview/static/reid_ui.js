/**
 * ReID UI - Pairwise comparison interface for identity clustering.
 *
 * Shows pairs of crops for the user to judge: "Are they the same person?"
 * Mixes ambiguous pairs with calibration pairs (confident same/different).
 *
 * Relies on the global API object (from app.js) for HTTP requests
 * and showToast() for notifications.
 */

/* exported ReIDUI */
/* global API, showToast, AppState, navigate */

class ReIDUI {
    constructor(container) {
        this.container = container;
        this.pairs = [];
        this.currentPairIndex = 0;
        this.resolutions = {};       // {pair_id: "same"|"different"|"unsure"}
        this.sessionId = null;
        this.clusters = {};
        this.nIdentities = 0;
        this.totalPairs = 0;
        this.onComplete = null;      // callback(mergeResult) when all pairs resolved
        this.reidRound = 1;          // current ReID round number
        this.phaseStage = 1;         // 1=centroid_building, 2=ambiguous, 3=auto_assignment

        this._boundKeyHandler = this._handleKeyDown.bind(this);
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    async init(sessionId) {
        this.sessionId = sessionId;
        this.resolutions = {};
        this.currentPairIndex = 0;

        try {
            const data = await API.get('/reid/clusters', {
                session_id: sessionId,
            });
            this.pairs = data.unresolved_pairs || [];
            this.clusters = data.clusters || {};
            this.nIdentities = data.n_identities || 0;
            this.totalPairs = data.total_pairs || 0;
            this.reidRound = data.reid_round || 1;
            this.phaseStage = data.reid_phase_stage || 1;
            this.mustLinkCount = data.must_link_count || 0;
            this.cannotLinkCount = data.cannot_link_count || 0;
        } catch (err) {
            showToast('Failed to load ReID clusters: ' + err.message, 'error');
            return;
        }

        // If clustering hasn't been run yet, show start screen
        if (this.nIdentities === 0 && this.totalPairs === 0 &&
            Object.keys(this.clusters).length === 0) {
            this._renderStartClustering();
            return;
        }

        document.addEventListener('keydown', this._boundKeyHandler);
        this._render();

        if (this.pairs.length === 0) {
            this._renderSummary();
        } else {
            this._showPair(0);
        }
    }

    destroy() {
        document.removeEventListener('keydown', this._boundKeyHandler);
    }

    // ------------------------------------------------------------------
    // Start clustering (not yet run)
    // ------------------------------------------------------------------

    _renderStartClustering() {
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
            'Cluster accepted crops into identities using DINOv3 features ' +
            'and color histograms, then verify ambiguous pairs.';
        box.appendChild(desc);

        // K input
        var kGroup = document.createElement('div');
        kGroup.className = 'form-group';
        var kLabel = document.createElement('label');
        kLabel.textContent = 'Expected number of identities (leave blank for auto)';
        kLabel.setAttribute('for', 'reid-k-input');
        kGroup.appendChild(kLabel);
        var kInput = document.createElement('input');
        kInput.type = 'number';
        kInput.id = 'reid-k-input';
        kInput.min = '2';
        kInput.max = '50';
        kInput.placeholder = 'Auto';
        kInput.style.maxWidth = '120px';
        kGroup.appendChild(kInput);
        box.appendChild(kGroup);

        var actions = document.createElement('div');
        actions.className = 'session-actions';

        var btnStart = document.createElement('button');
        btnStart.className = 'btn btn-primary';
        btnStart.textContent = 'Start Clustering';
        btnStart.addEventListener('click', function () {
            var val = kInput.value.trim();
            var nClusters = val ? parseInt(val, 10) : null;
            this._runClustering(nClusters);
        }.bind(this));
        actions.appendChild(btnStart);

        var btnVisual = document.createElement('button');
        btnVisual.className = 'btn btn-secondary';
        btnVisual.textContent = 'Visual Pipeline';
        btnVisual.addEventListener('click', function () {
            this._runVisualPipeline();
        }.bind(this));
        actions.appendChild(btnVisual);

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
    }

    async _runClustering(nClusters) {
        var progressArea = document.getElementById('reid-progress-area');
        if (progressArea) {
            progressArea.innerHTML =
                '<div class="progress-bar-wrapper">' +
                '<div class="progress-bar-track"><div class="progress-bar-fill indeterminate"></div></div>' +
                '<div class="progress-text"><span>Starting clustering...</span></div>' +
                '</div>';
        }

        try {
            var payload = { session_id: this.sessionId };
            if (nClusters != null && nClusters >= 2) {
                payload.n_clusters = nClusters;
            }
            var resp = await API.post('/reid/start', payload);

            // Poll until done
            var self = this;
            if (typeof pollJob === 'function') {
                pollJob(
                    resp.job_id,
                    function (p) {
                        if (progressArea) {
                            var text = progressArea.querySelector('.progress-text span');
                            if (text) text.textContent = p.step || 'Clustering...';
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
                            // Re-init to load the new clusters and pairs
                            self.init(self.sessionId);
                        } else {
                            showToast('Clustering failed: ' + (p.error || 'Unknown'), 'error');
                            if (progressArea) progressArea.innerHTML = '';
                        }
                    },
                    1000
                );
            }
        } catch (err) {
            showToast('Failed to start clustering: ' + err.message, 'error');
            if (progressArea) progressArea.innerHTML = '';
        }
    }

    // ------------------------------------------------------------------
    // Visual Pipeline (multi-cue merge proposals)
    // ------------------------------------------------------------------

    async _runVisualPipeline() {
        this.container.innerHTML = '';

        var loading = document.createElement('div');
        loading.className = 'seeding-panel';
        loading.style.marginTop = '40px';
        loading.style.textAlign = 'center';
        loading.innerHTML =
            '<div class="spinner"></div>' +
            '<p style="color:var(--text-secondary);margin-top:12px">Running visual pipeline...</p>';
        this.container.appendChild(loading);

        try {
            var result = await API.post('/reid/visual_pipeline', {
                session_id: this.sessionId,
            });
            this._renderVisualPipelineResults(result);
        } catch (err) {
            showToast('Visual pipeline failed: ' + err.message, 'error');
            this._renderSummary();
        }
    }

    _renderVisualPipelineResults(result) {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';
        panel.style.marginTop = '20px';

        var box = document.createElement('div');
        box.className = 'session-setup';
        box.style.maxWidth = '900px';

        var h2 = document.createElement('h2');
        h2.textContent = 'Visual Pipeline';
        box.appendChild(h2);

        // -- Stats bar --
        var sessionStats = result.session_stats || {};
        var statsRow = document.createElement('div');
        statsRow.className = 'visual-pipeline-stats';

        var statItems = [
            { label: 'Crops', value: sessionStats.n_crops || '?' },
            { label: 'Runs', value: sessionStats.n_runs || '?' },
            { label: 'Clusters', value: sessionStats.n_clusters || '?' },
        ];
        for (var si = 0; si < statItems.length; si++) {
            var badge = document.createElement('span');
            badge.className = 'visual-pipeline-stat';
            badge.innerHTML = '<strong>' + statItems[si].value + '</strong> ' + statItems[si].label;
            statsRow.appendChild(badge);
        }
        box.appendChild(statsRow);

        // -- Weights display (optional) --
        var weights = result.weights;
        if (weights && typeof weights === 'object') {
            var wKeys = Object.keys(weights);
            if (wKeys.length > 0) {
                var wRow = document.createElement('div');
                wRow.style.fontSize = '0.7rem';
                wRow.style.color = 'var(--text-muted)';
                wRow.style.marginBottom = '12px';
                var wParts = [];
                for (var wi = 0; wi < wKeys.length; wi++) {
                    var wVal = weights[wKeys[wi]];
                    wParts.push(wKeys[wi] + ': ' + (typeof wVal === 'number' ? wVal.toFixed(3) : wVal));
                }
                wRow.textContent = 'Cue weights: ' + wParts.join(', ');
                box.appendChild(wRow);
            }
        }

        // -- Cluster gallery (reuse summary pattern) --
        var clusters = result.clusters || {};
        var clusterKeys = Object.keys(clusters);
        if (clusterKeys.length > 0) {
            var gallery = document.createElement('div');
            gallery.className = 'reid-cluster-gallery';

            var MAX_THUMBS = 12;
            for (var c = 0; c < clusterKeys.length; c++) {
                (function (cKey, clusterData, self) {
                    var cropIds = [];
                    // clusters can be {identity_name: [crop_ids]} or {identity_name: {crop_ids: [...]}}
                    if (Array.isArray(clusterData)) {
                        cropIds = clusterData;
                    } else if (clusterData && Array.isArray(clusterData.crop_ids)) {
                        cropIds = clusterData.crop_ids;
                    }

                    var section = document.createElement('div');
                    section.className = 'reid-cluster-section';

                    var header = document.createElement('div');
                    header.className = 'reid-cluster-header';
                    header.textContent = 'Identity ' + cKey + ' (' + cropIds.length + ' crops)';
                    section.appendChild(header);

                    var row = document.createElement('div');
                    row.className = 'reid-cluster-row';

                    var visible = Math.min(cropIds.length, MAX_THUMBS);
                    for (var t = 0; t < visible; t++) {
                        var wrap = self._createThumbWrap(cropIds[t]);
                        row.appendChild(wrap);
                    }
                    if (cropIds.length > MAX_THUMBS) {
                        var overflow = document.createElement('div');
                        overflow.className = 'reid-overflow-badge';
                        overflow.textContent = '+' + (cropIds.length - MAX_THUMBS) + ' more';
                        row.appendChild(overflow);
                    }

                    section.appendChild(row);
                    gallery.appendChild(section);
                })(clusterKeys[c], clusters[clusterKeys[c]], this);
            }
            box.appendChild(gallery);
        }

        // -- Merge proposals --
        var proposals = result.merge_proposals || [];
        if (proposals.length > 0) {
            var proposalHeader = document.createElement('h3');
            proposalHeader.style.fontSize = '0.95rem';
            proposalHeader.style.marginTop = '16px';
            proposalHeader.style.marginBottom = '10px';
            proposalHeader.textContent = 'Merge Proposals (' + proposals.length + ')';
            box.appendChild(proposalHeader);

            // Build a map of cluster -> crop_ids for thumbnails
            var clusterCropMap = {};
            for (var ck = 0; ck < clusterKeys.length; ck++) {
                var cd = clusters[clusterKeys[ck]];
                if (Array.isArray(cd)) {
                    clusterCropMap[clusterKeys[ck]] = cd;
                } else if (cd && Array.isArray(cd.crop_ids)) {
                    clusterCropMap[clusterKeys[ck]] = cd.crop_ids;
                } else {
                    clusterCropMap[clusterKeys[ck]] = [];
                }
            }

            var verdicts = {};
            var proposalContainer = document.createElement('div');

            // Show top 3 most informative
            var maxProposals = Math.min(proposals.length, 3);
            for (var pi = 0; pi < maxProposals; pi++) {
                this._renderProposalCard(proposals[pi], pi, proposalContainer, verdicts, clusterCropMap);
            }
            box.appendChild(proposalContainer);

            // -- Apply Verdicts button --
            var verdictActions = document.createElement('div');
            verdictActions.className = 'session-actions';
            verdictActions.style.marginTop = '16px';

            var self = this;
            var btnApply = document.createElement('button');
            btnApply.className = 'btn btn-primary';
            btnApply.textContent = 'Apply Verdicts';
            btnApply.addEventListener('click', function () {
                // Collect all verdicts that have been set
                var verdictList = [];
                var vKeys = Object.keys(verdicts);
                for (var vi = 0; vi < vKeys.length; vi++) {
                    verdictList.push(verdicts[vKeys[vi]]);
                }
                if (verdictList.length === 0) {
                    showToast('No verdicts to submit. Judge at least one proposal.', 'warning');
                    return;
                }
                self._submitVisualVerdicts(verdictList);
            });
            verdictActions.appendChild(btnApply);

            var btnSkipVerdicts = document.createElement('button');
            btnSkipVerdicts.className = 'btn btn-ghost';
            btnSkipVerdicts.textContent = 'Skip — View Summary';
            btnSkipVerdicts.addEventListener('click', function () {
                self._renderSummary();
            });
            verdictActions.appendChild(btnSkipVerdicts);

            box.appendChild(verdictActions);
        } else {
            var noProposals = document.createElement('p');
            noProposals.style.color = 'var(--text-secondary)';
            noProposals.style.fontSize = '0.85rem';
            noProposals.style.marginTop = '16px';
            noProposals.textContent = 'No merge proposals generated. Clusters appear well-separated.';
            box.appendChild(noProposals);

            var doneActions = document.createElement('div');
            doneActions.className = 'session-actions';
            doneActions.style.marginTop = '16px';

            var btnDone = document.createElement('button');
            btnDone.className = 'btn btn-primary';
            btnDone.textContent = 'View Summary';
            btnDone.addEventListener('click', function () {
                this._renderSummary();
            }.bind(this));
            doneActions.appendChild(btnDone);
            box.appendChild(doneActions);
        }

        panel.appendChild(box);
        this.container.appendChild(panel);
    }

    _renderProposalCard(proposal, index, container, verdicts, clusterCropMap) {
        var self = this;
        var card = document.createElement('div');
        card.className = 'visual-proposal-card';

        // -- Header: cluster names + scores --
        var header = document.createElement('div');
        header.className = 'visual-proposal-header';

        var title = document.createElement('span');
        title.style.fontWeight = '600';
        title.textContent = proposal.cluster_a + '  ↔  ' + proposal.cluster_b;
        header.appendChild(title);

        var scores = document.createElement('span');
        scores.style.color = 'var(--text-secondary)';
        var scoreText = 'Score: ' + (typeof proposal.merge_score === 'number' ? proposal.merge_score.toFixed(2) : '?');
        if (typeof proposal.information_gain === 'number') {
            scoreText += '  IG: ' + proposal.information_gain.toFixed(2);
        }
        scores.textContent = scoreText;
        header.appendChild(scores);

        card.appendChild(header);

        // -- Cluster thumbnails side by side --
        var clustersRow = document.createElement('div');
        clustersRow.className = 'visual-proposal-clusters';

        var cropsA = clusterCropMap[proposal.cluster_a] || [];
        var cropsB = clusterCropMap[proposal.cluster_b] || [];

        var thumbsA = document.createElement('div');
        thumbsA.className = 'visual-proposal-cluster-thumbs';
        var maxThumbs = 3;
        for (var ta = 0; ta < Math.min(cropsA.length, maxThumbs); ta++) {
            (function (cropId) {
                var img = document.createElement('img');
                img.src = '/interview/api/detect/crop/' + cropId +
                    '/image?session_id=' + encodeURIComponent(self.sessionId);
                img.alt = cropId;
                thumbsA.appendChild(img);
            })(cropsA[ta]);
        }
        clustersRow.appendChild(thumbsA);

        var arrow = document.createElement('span');
        arrow.className = 'visual-proposal-arrow';
        arrow.textContent = '↔';
        clustersRow.appendChild(arrow);

        var thumbsB = document.createElement('div');
        thumbsB.className = 'visual-proposal-cluster-thumbs';
        for (var tb = 0; tb < Math.min(cropsB.length, maxThumbs); tb++) {
            (function (cropId) {
                var img = document.createElement('img');
                img.src = '/interview/api/detect/crop/' + cropId +
                    '/image?session_id=' + encodeURIComponent(self.sessionId);
                img.alt = cropId;
                thumbsB.appendChild(img);
            })(cropsB[tb]);
        }
        clustersRow.appendChild(thumbsB);

        card.appendChild(clustersRow);

        // -- Per-cue evidence bars --
        var perCue = proposal.per_cue || {};
        var cueKeys = Object.keys(perCue);
        if (cueKeys.length > 0) {
            var barsDiv = document.createElement('div');
            barsDiv.className = 'visual-cue-bars';

            for (var ci = 0; ci < cueKeys.length; ci++) {
                (function (cueName, cueVal) {
                    var row = document.createElement('div');
                    row.className = 'visual-cue-row';

                    var label = document.createElement('span');
                    label.className = 'visual-cue-label';
                    label.textContent = cueName;
                    row.appendChild(label);

                    var track = document.createElement('div');
                    track.className = 'visual-cue-bar-track';

                    var fill = document.createElement('div');
                    fill.className = 'visual-cue-bar-fill';
                    var pct = Math.max(0, Math.min(100, (typeof cueVal === 'number' ? cueVal : 0) * 100));
                    fill.style.width = pct + '%';

                    // Color based on value
                    if (cueVal < 0.3) {
                        fill.style.background = 'var(--color-rejected)';
                    } else if (cueVal < 0.7) {
                        fill.style.background = 'var(--color-pending)';
                    } else {
                        fill.style.background = 'var(--color-accepted)';
                    }

                    track.appendChild(fill);
                    row.appendChild(track);

                    var valSpan = document.createElement('span');
                    valSpan.className = 'visual-cue-value';
                    valSpan.textContent = typeof cueVal === 'number' ? cueVal.toFixed(2) : '?';
                    row.appendChild(valSpan);

                    barsDiv.appendChild(row);
                })(cueKeys[ci], perCue[cueKeys[ci]]);
            }
            card.appendChild(barsDiv);
        }

        // -- Flags (co-occurrence conflict, temporal overlap) --
        var flags = [];
        if (proposal.cooccurrence_conflict) {
            flags.push('co-occurrence conflict');
        }
        if (proposal.temporal_overlap) {
            flags.push('temporal overlap');
        }
        if (flags.length > 0) {
            var flagDiv = document.createElement('div');
            flagDiv.className = 'visual-proposal-flags';
            flagDiv.textContent = '⚠ ' + flags.join(', ');
            card.appendChild(flagDiv);
        }

        // -- Verdict buttons --
        var verdictRow = document.createElement('div');
        verdictRow.className = 'visual-proposal-verdict';

        var pairKey = proposal.cluster_a + '::' + proposal.cluster_b;

        var btnSame = document.createElement('button');
        btnSame.className = 'btn btn-accept btn-small';
        btnSame.textContent = 'Same Person';
        btnSame.addEventListener('click', function () {
            verdicts[pairKey] = {
                cluster_a: proposal.cluster_a,
                cluster_b: proposal.cluster_b,
                verdict: 'same',
            };
            btnSame.style.outline = '2px solid #fff';
            btnDiff.style.outline = 'none';
            card.style.borderColor = 'var(--color-accepted)';
        });
        verdictRow.appendChild(btnSame);

        var btnDiff = document.createElement('button');
        btnDiff.className = 'btn btn-reject btn-small';
        btnDiff.textContent = 'Different Person';
        btnDiff.addEventListener('click', function () {
            verdicts[pairKey] = {
                cluster_a: proposal.cluster_a,
                cluster_b: proposal.cluster_b,
                verdict: 'different',
            };
            btnDiff.style.outline = '2px solid #fff';
            btnSame.style.outline = 'none';
            card.style.borderColor = 'var(--color-rejected)';
        });
        verdictRow.appendChild(btnDiff);

        card.appendChild(verdictRow);
        container.appendChild(card);
    }

    async _submitVisualVerdicts(verdictList) {
        this.container.innerHTML = '';
        var loading = document.createElement('div');
        loading.className = 'seeding-panel';
        loading.style.marginTop = '40px';
        loading.style.textAlign = 'center';
        loading.innerHTML =
            '<div class="spinner"></div>' +
            '<p style="color:var(--text-secondary);margin-top:12px">Applying verdicts...</p>';
        this.container.appendChild(loading);

        try {
            var result = await API.post('/reid/merge_verdict', {
                session_id: this.sessionId,
                verdicts: verdictList,
            });

            if (result.clusters) {
                this.clusters = result.clusters;
                this.nIdentities = Object.keys(this.clusters).length;
            }

            if (result.converged) {
                showToast('Visual pipeline converged!', 'success');
                this._renderSummary();
            } else {
                // Show updated proposals for next round
                this._renderVisualPipelineResults(result);
            }
        } catch (err) {
            showToast('Failed to submit verdicts: ' + err.message, 'error');
            this._renderSummary();
        }
    }

    // ------------------------------------------------------------------
    // Top-level render
    // ------------------------------------------------------------------

    _render() {
        this.container.innerHTML = '';

        var root = document.createElement('div');
        root.className = 'reid-panel';

        // Left column: two frame panels stacked
        var frameTop = document.createElement('div');
        frameTop.className = 'reid-frame-top frame-viewer';
        frameTop.id = 'reid-frame-a';

        var frameBottom = document.createElement('div');
        frameBottom.className = 'reid-frame-bottom frame-viewer';
        frameBottom.id = 'reid-frame-b';

        // Right column: crops + verdict + merge status
        var comparison = document.createElement('div');
        comparison.className = 'reid-comparison';
        comparison.id = 'reid-comparison';

        root.appendChild(frameTop);
        root.appendChild(frameBottom);
        root.appendChild(comparison);

        this.container.appendChild(root);
    }

    // ------------------------------------------------------------------
    // Show a specific pair
    // ------------------------------------------------------------------

    async _showPair(index) {
        if (index < 0 || index >= this.pairs.length) return;
        this.currentPairIndex = index;

        var pair = this.pairs[index];

        // Fetch pair frame data from backend
        var pairData;
        try {
            pairData = await API.get('/reid/pair/' + pair.pair_id + '/frames', {
                session_id: this.sessionId,
            });
        } catch (err) {
            showToast('Failed to load pair data: ' + err.message, 'error');
            return;
        }

        var cropA = pairData.crop_a;
        var cropB = pairData.crop_b;
        if (!cropA || !cropB) {
            showToast('Missing crop data for this pair.', 'warning');
            return;
        }

        // -- Frame A with highlighted box --
        this._renderFrame('reid-frame-a', cropA, 'A');

        // -- Frame B with highlighted box --
        this._renderFrame('reid-frame-b', cropB, 'B');

        // -- Comparison panel --
        this._renderComparison(pair, cropA, cropB);
    }

    _renderFrame(containerId, crop, label) {
        var container = document.getElementById(containerId);
        container.innerHTML = '';

        var wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.display = 'inline-block';
        wrapper.style.maxWidth = '100%';
        wrapper.style.maxHeight = '100%';

        var img = document.createElement('img');
        img.src = '/interview/api/detect/frame/' + crop.frame_idx +
            '?session_id=' + encodeURIComponent(this.sessionId);
        img.alt = 'Frame ' + crop.frame_idx;
        img.style.display = 'block';
        img.style.maxWidth = '100%';
        img.style.maxHeight = '100%';
        img.style.objectFit = 'contain';

        // Draw highlight box after image loads
        img.onload = function () {
            var scaleX = img.clientWidth / img.naturalWidth;
            var scaleY = img.clientHeight / img.naturalHeight;

            var box = document.createElement('div');
            box.className = 'overlay-box active';
            box.style.position = 'absolute';
            box.style.left = (crop.xyxy[0] * scaleX) + 'px';
            box.style.top = (crop.xyxy[1] * scaleY) + 'px';
            box.style.width = ((crop.xyxy[2] - crop.xyxy[0]) * scaleX) + 'px';
            box.style.height = ((crop.xyxy[3] - crop.xyxy[1]) * scaleY) + 'px';
            box.style.borderColor = label === 'A' ? '#3498db' : '#e67e22';
            box.style.borderWidth = '3px';
            box.style.borderStyle = 'solid';
            box.style.pointerEvents = 'none';
            wrapper.appendChild(box);
        };

        // Frame label badge
        var badge = document.createElement('span');
        badge.className = 'card-badge';
        badge.style.position = 'absolute';
        badge.style.top = '8px';
        badge.style.left = '8px';
        badge.style.fontSize = '0.75rem';
        badge.style.padding = '2px 8px';
        badge.style.zIndex = '10';
        badge.textContent = label + ' - Frame ' + crop.frame_idx;

        wrapper.appendChild(img);
        wrapper.appendChild(badge);
        container.appendChild(wrapper);
    }

    _renderComparison(pair, cropA, cropB) {
        var container = document.getElementById('reid-comparison');
        container.innerHTML = '';

        // -- Progress bar --
        var progressWrapper = document.createElement('div');
        progressWrapper.className = 'progress-bar-wrapper';

        var track = document.createElement('div');
        track.className = 'progress-bar-track';
        var fill = document.createElement('div');
        fill.className = 'progress-bar-fill';
        var resolved = Object.keys(this.resolutions).length;
        var pct = this.pairs.length > 0
            ? ((resolved / this.pairs.length) * 100)
            : 0;
        fill.style.width = pct + '%';
        track.appendChild(fill);

        var text = document.createElement('div');
        text.className = 'progress-text';

        var ambiguousRemaining = 0;
        for (var i = 0; i < this.pairs.length; i++) {
            if (!this.resolutions[this.pairs[i].pair_id] && this.pairs[i].pool === 'ambiguous') {
                ambiguousRemaining++;
            }
        }

        var phaseLabel = this.phaseStage === 1 ? 'Centroid Building'
            : this.phaseStage === 2 ? 'Ambiguous Resolution'
            : 'Auto-Assignment';
        text.innerHTML =
            '<span>' + phaseLabel + ' — Round ' + this.reidRound + ' — Pair ' + (this.currentPairIndex + 1) + '/' + this.pairs.length + '</span>' +
            '<span>Remaining: ' + ambiguousRemaining + ' ambiguous</span>';

        progressWrapper.appendChild(track);
        progressWrapper.appendChild(text);
        container.appendChild(progressWrapper);

        // -- Crops side by side --
        var cropsRow = document.createElement('div');
        cropsRow.className = 'reid-crops';

        var cropImgA = document.createElement('img');
        cropImgA.src = '/interview/api/detect/crop/' + cropA.crop_id +
            '/image?session_id=' + encodeURIComponent(this.sessionId);
        cropImgA.alt = 'Crop A';
        cropImgA.style.border = '3px solid #3498db';

        var cropImgB = document.createElement('img');
        cropImgB.src = '/interview/api/detect/crop/' + cropB.crop_id +
            '/image?session_id=' + encodeURIComponent(this.sessionId);
        cropImgB.alt = 'Crop B';
        cropImgB.style.border = '3px solid #e67e22';

        cropsRow.appendChild(cropImgA);
        cropsRow.appendChild(cropImgB);
        container.appendChild(cropsRow);

        // -- Pool indicator (subtle) --
        var poolIndicator = document.createElement('div');
        poolIndicator.style.display = 'flex';
        poolIndicator.style.alignItems = 'center';
        poolIndicator.style.justifyContent = 'center';
        poolIndicator.style.gap = '6px';
        poolIndicator.style.padding = '4px 0';

        var dot = document.createElement('span');
        dot.style.display = 'inline-block';
        dot.style.width = '8px';
        dot.style.height = '8px';
        dot.style.borderRadius = '50%';

        var poolLabel = document.createElement('span');
        poolLabel.style.fontSize = '0.65rem';
        poolLabel.style.color = 'var(--text-muted)';

        if (pair.pool === 'centroid_building') {
            dot.style.background = '#9b59b6';
            poolLabel.textContent = 'centroid building (intra-cluster)';
        } else if (pair.pool === 'merge_candidate') {
            dot.style.background = '#e74c3c';
            poolLabel.textContent = 'merge candidate (high similarity)';
        } else if (pair.pool === 'ambiguous') {
            dot.style.background = '#3498db';
            poolLabel.textContent = 'ambiguous';
        } else if (pair.pool === 'confident_same') {
            dot.style.background = '#2ecc71';
            poolLabel.textContent = 'calibration (likely same)';
        } else if (pair.pool === 'confident_different') {
            dot.style.background = '#e67e22';
            poolLabel.textContent = 'calibration (likely different)';
        } else {
            dot.style.background = '#95a5a6';
            poolLabel.textContent = pair.pool || 'unknown';
        }

        poolIndicator.appendChild(dot);
        poolIndicator.appendChild(poolLabel);
        container.appendChild(poolIndicator);

        // -- Similarity score --
        if (typeof pair.similarity === 'number') {
            var simRow = document.createElement('div');
            simRow.style.textAlign = 'center';
            simRow.style.fontSize = '0.75rem';
            simRow.style.color = 'var(--text-secondary)';
            simRow.textContent = 'Similarity: ' + pair.similarity.toFixed(3);
            container.appendChild(simRow);
        }

        // -- Verdict buttons --
        var verdictRow = document.createElement('div');
        verdictRow.className = 'reid-verdict';

        var btnYes = document.createElement('button');
        btnYes.className = 'btn btn-accept';
        btnYes.innerHTML = 'Yes &mdash; Same Person';
        btnYes.addEventListener('click', function () {
            this._resolvePair('same');
        }.bind(this));

        var btnNo = document.createElement('button');
        btnNo.className = 'btn btn-reject';
        btnNo.innerHTML = 'No &mdash; Different';
        btnNo.addEventListener('click', function () {
            this._resolvePair('different');
        }.bind(this));

        var btnUnsure = document.createElement('button');
        btnUnsure.className = 'btn btn-unsure';
        btnUnsure.textContent = 'Unsure';
        btnUnsure.addEventListener('click', function () {
            this._resolvePair('unsure');
        }.bind(this));

        verdictRow.appendChild(btnYes);
        verdictRow.appendChild(btnNo);
        verdictRow.appendChild(btnUnsure);
        container.appendChild(verdictRow);

        // -- Keyboard hints --
        var hints = document.createElement('div');
        hints.className = 'keyboard-hints';
        hints.innerHTML =
            '<kbd>F</kbd> Same ' +
            '<kbd>J</kbd> Different ' +
            '<kbd>Space</kbd> Unsure ' +
            '<kbd>&larr;</kbd> Previous';
        container.appendChild(hints);

        // -- Merge status panel --
        this._renderMergeStatus(container);
    }

    // ------------------------------------------------------------------
    // Resolve a pair
    // ------------------------------------------------------------------

    async _resolvePair(resolution) {
        var pair = this.pairs[this.currentPairIndex];
        if (!pair) return;

        this.resolutions[pair.pair_id] = resolution;

        // Move to next unresolved pair (no per-pair API call — batch on summary)
        var nextIndex = this._findNextUnresolved(this.currentPairIndex + 1);
        if (nextIndex !== -1) {
            this._showPair(nextIndex);
        } else {
            // All pairs resolved — batch-submit then show summary
            await this._submitAndShowSummary();
        }
    }

    async _submitAndShowSummary() {
        this.container.innerHTML = '';
        var loading = document.createElement('div');
        loading.className = 'seeding-panel';
        loading.style.marginTop = '40px';
        loading.style.textAlign = 'center';
        loading.innerHTML =
            '<div class="spinner"></div>' +
            '<p style="color:var(--text-secondary);margin-top:12px">Applying resolutions…</p>';
        this.container.appendChild(loading);

        var result = {};
        try {
            result = await API.post('/reid/resolve', {
                session_id: this.sessionId,
                resolutions: this.resolutions,
            });

            if (result.n_identities !== undefined) {
                this.nIdentities = result.n_identities;
            }
            if (result.clusters) {
                this.clusters = result.clusters;
            }
            if (result.reid_phase_stage !== undefined) {
                this.phaseStage = result.reid_phase_stage;
            }
        } catch (err) {
            showToast('Failed to submit resolutions: ' + err.message, 'error');
        }

        if (result.reid_phase_stage >= 2 && result.centroid_assignments) {
            // Phase 2: show centroid-based assignments
            this._renderCentroidAssignments(result.centroid_assignments);
        } else if (result.needs_more_rounds) {
            this._renderNextRoundPrompt(result);
        } else {
            // Rounds complete — check for auto-assignments
            this._checkAutoAssignments();
        }
    }

    _renderNextRoundPrompt(resolveResult) {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';
        panel.style.marginTop = '20px';

        var box = document.createElement('div');
        box.className = 'session-setup';
        box.style.maxWidth = '600px';

        var h2 = document.createElement('h2');
        h2.textContent = 'Round ' + this.reidRound + ' Complete';
        box.appendChild(h2);

        var info = document.createElement('div');
        info.style.marginBottom = '16px';
        info.style.color = 'var(--text-secondary)';
        info.style.fontSize = '0.85rem';
        var merges = resolveResult.merges_executed || 0;
        var vetoed = resolveResult.vetoed_pairs || 0;
        var uncovered = resolveResult.uncovered_count || 0;
        info.innerHTML =
            '<p>Merges: <strong>' + merges + '</strong> | ' +
            'Vetoed: <strong>' + vetoed + '</strong> | ' +
            'Identities: <strong>' + this.nIdentities + '</strong></p>' +
            '<p style="margin-top:8px">' + uncovered +
            ' cluster-pair relationship(s) still need verification.</p>';
        box.appendChild(info);

        var actions = document.createElement('div');
        actions.className = 'session-actions';

        var btnContinue = document.createElement('button');
        btnContinue.className = 'btn btn-primary';
        btnContinue.textContent = 'Continue — Next Round';
        btnContinue.addEventListener('click', async function () {
            await this._loadNextRound();
        }.bind(this));
        actions.appendChild(btnContinue);

        var btnFinish = document.createElement('button');
        btnFinish.className = 'btn btn-secondary';
        btnFinish.textContent = 'Finish (View Summary)';
        btnFinish.addEventListener('click', function () {
            this._renderSummary();
        }.bind(this));
        actions.appendChild(btnFinish);

        box.appendChild(actions);
        panel.appendChild(box);
        this.container.appendChild(panel);
    }

    async _loadNextRound() {
        this.container.innerHTML = '';
        var loading = document.createElement('div');
        loading.className = 'seeding-panel';
        loading.style.marginTop = '40px';
        loading.style.textAlign = 'center';
        loading.innerHTML =
            '<div class="spinner"></div>' +
            '<p style="color:var(--text-secondary);margin-top:12px">Generating next round…</p>';
        this.container.appendChild(loading);

        try {
            var data = await API.post('/reid/next_round', {
                session_id: this.sessionId,
            });

            this.reidRound = data.reid_round || (this.reidRound + 1);
            var newPairs = data.pairs || [];

            if (newPairs.length === 0) {
                showToast('All cluster-pairs resolved!', 'success');
                this._renderSummary();
                return;
            }

            // Reset for new round
            this.pairs = newPairs;
            this.resolutions = {};
            this.currentPairIndex = 0;
            this.totalPairs += newPairs.length;

            document.addEventListener('keydown', this._boundKeyHandler);
            this._render();
            this._showPair(0);

            showToast('Round ' + this.reidRound + ': ' + newPairs.length + ' pairs', 'info');
        } catch (err) {
            showToast('Failed to load next round: ' + err.message, 'error');
            this._renderSummary();
        }
    }

    _findNextUnresolved(startIndex) {
        for (var i = startIndex; i < this.pairs.length; i++) {
            if (!this.resolutions[this.pairs[i].pair_id]) {
                return i;
            }
        }
        // Wrap around to check before startIndex
        for (var j = 0; j < startIndex && j < this.pairs.length; j++) {
            if (!this.resolutions[this.pairs[j].pair_id]) {
                return j;
            }
        }
        return -1;
    }

    // ------------------------------------------------------------------
    // Phase 3: Auto-assignment quick-confirm cards
    // ------------------------------------------------------------------

    async _checkAutoAssignments() {
        // If in Phase 2, use centroid assignment view instead
        if (this.phaseStage >= 2) {
            try {
                var cData = await API.get('/reid/centroid_assignments', {
                    session_id: this.sessionId,
                });
                var nD = Object.keys(cData.decisive || {}).length;
                var nI = Object.keys(cData.indecisive || {}).length;
                if (nD > 0 || nI > 0) {
                    this._renderCentroidAssignments(cData);
                    return;
                }
            } catch (e) {
                // fall through to legacy
            }
        }

        try {
            var data = await API.get('/reid/auto_assignments', {
                session_id: this.sessionId,
            });
            var autoAssigned = data.auto_assigned || {};
            var unresolved = data.unresolved || {};
            var autoCount = Object.keys(autoAssigned).length;
            var unresolvedCount = Object.keys(unresolved).length;

            if (autoCount > 0) {
                this._renderAutoAssignments(autoAssigned, unresolvedCount);
            } else {
                this._renderSummary();
            }
        } catch (err) {
            this._renderSummary();
        }
    }

    _renderAutoAssignments(autoAssigned, unresolvedCount) {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';
        panel.style.marginTop = '20px';

        var box = document.createElement('div');
        box.className = 'session-setup';
        box.style.maxWidth = '900px';

        // Header
        var h2 = document.createElement('h2');
        h2.textContent = 'Auto-Assignment Review';
        box.appendChild(h2);

        var desc = document.createElement('p');
        desc.style.color = 'var(--text-secondary)';
        desc.style.fontSize = '0.85rem';
        desc.style.marginBottom = '16px';
        var cropIds = Object.keys(autoAssigned);
        var newCount = 0;
        var confirmCount = 0;
        for (var ci = 0; ci < cropIds.length; ci++) {
            if (autoAssigned[cropIds[ci]].already_clustered) {
                confirmCount++;
            } else {
                newCount++;
            }
        }
        var descParts = [];
        if (newCount > 0) descParts.push(newCount + ' new assignment(s)');
        if (confirmCount > 0) descParts.push(confirmCount + ' confident confirmation(s)');
        desc.textContent =
            descParts.join(', ') +
            ' across identity clusters. Review and confirm each below.' +
            (unresolvedCount > 0 ? ' (' + unresolvedCount + ' crops are too ambiguous for auto-assignment.)' : '');
        box.appendChild(desc);

        // Quick-confirm card grid
        var grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(200px, 1fr))';
        grid.style.gap = '12px';
        grid.style.marginBottom = '20px';

        var acceptedMap = {};   // crop_id -> cluster_id
        var rejectedSet = {};   // crop_id -> true

        var self = this;
        for (var i = 0; i < cropIds.length; i++) {
            (function (cropId) {
                var info = autoAssigned[cropId];

                var card = document.createElement('div');
                card.className = 'reid-auto-card';
                card.style.border = '1px solid var(--border-default)';
                card.style.borderRadius = '8px';
                card.style.padding = '8px';
                card.style.background = 'var(--bg-secondary)';
                card.style.textAlign = 'center';

                // Crop image
                var img = document.createElement('img');
                img.src = '/interview/api/detect/crop/' + cropId +
                    '/image?session_id=' + encodeURIComponent(self.sessionId);
                img.alt = cropId;
                img.style.width = '100%';
                img.style.maxHeight = '120px';
                img.style.objectFit = 'contain';
                img.style.borderRadius = '4px';
                card.appendChild(img);

                // Info text
                var infoEl = document.createElement('div');
                infoEl.style.fontSize = '0.7rem';
                infoEl.style.color = 'var(--text-secondary)';
                infoEl.style.margin = '6px 0';
                var matchNote = '';
                if (info.already_clustered) {
                    if (info.current_cluster === info.cluster_id) {
                        matchNote = '<span style="color:var(--color-accepted)"> (confirms current)</span>';
                    } else {
                        matchNote = '<span style="color:var(--color-rejected)"> (reassign from ' + info.current_cluster + ')</span>';
                    }
                }
                infoEl.innerHTML =
                    'Identity <strong>' + info.cluster_id + '</strong> ' +
                    '(conf: ' + (info.confidence != null ? info.confidence.toFixed(2) : '?') + ', ' +
                    'margin: ' + (info.margin != null ? info.margin.toFixed(2) : '?') + ')' + matchNote;
                card.appendChild(infoEl);

                // Accept / Reject buttons
                var btns = document.createElement('div');
                btns.style.display = 'flex';
                btns.style.gap = '4px';
                btns.style.justifyContent = 'center';

                var btnAccept = document.createElement('button');
                btnAccept.className = 'btn btn-accept btn-small';
                btnAccept.textContent = 'Accept';
                btnAccept.style.fontSize = '0.7rem';
                btnAccept.style.padding = '2px 10px';
                btnAccept.addEventListener('click', function () {
                    acceptedMap[cropId] = info.cluster_id;
                    delete rejectedSet[cropId];
                    card.style.borderColor = 'var(--color-accepted)';
                    card.style.borderWidth = '2px';
                });
                btns.appendChild(btnAccept);

                var btnReject = document.createElement('button');
                btnReject.className = 'btn btn-reject btn-small';
                btnReject.textContent = 'Reject';
                btnReject.style.fontSize = '0.7rem';
                btnReject.style.padding = '2px 10px';
                btnReject.addEventListener('click', function () {
                    rejectedSet[cropId] = true;
                    delete acceptedMap[cropId];
                    card.style.borderColor = 'var(--color-rejected)';
                    card.style.borderWidth = '2px';
                });
                btns.appendChild(btnReject);

                card.appendChild(btns);

                // Pre-accept by default (user can reject)
                acceptedMap[cropId] = info.cluster_id;
                card.style.borderColor = 'var(--color-accepted)';
                card.style.borderWidth = '2px';

                grid.appendChild(card);
            })(cropIds[i]);
        }

        box.appendChild(grid);

        // Action buttons
        var actions = document.createElement('div');
        actions.className = 'session-actions';

        var btnApply = document.createElement('button');
        btnApply.className = 'btn btn-primary';
        btnApply.textContent = 'Apply Assignments';
        btnApply.addEventListener('click', async function () {
            try {
                await API.post('/reid/apply_auto_assignments', {
                    session_id: self.sessionId,
                    assignments: acceptedMap,
                });
                var appliedCount = Object.keys(acceptedMap).length;
                showToast(appliedCount + ' crop(s) assigned to identities', 'success');
                self._renderSummary();
            } catch (err) {
                showToast('Failed to apply: ' + err.message, 'error');
            }
        });
        actions.appendChild(btnApply);

        var btnSkip = document.createElement('button');
        btnSkip.className = 'btn btn-ghost';
        btnSkip.textContent = 'Skip — Go to Summary';
        btnSkip.addEventListener('click', function () {
            self._renderSummary();
        });
        actions.appendChild(btnSkip);

        box.appendChild(actions);
        panel.appendChild(box);
        this.container.appendChild(panel);
    }

    // ------------------------------------------------------------------
    // Phase 2: Centroid-based crop assignment (decisive + indecisive)
    // ------------------------------------------------------------------

    _renderCentroidAssignments(data) {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';
        panel.style.marginTop = '20px';

        var box = document.createElement('div');
        box.className = 'session-setup';
        box.style.maxWidth = '1000px';

        // Header
        var h2 = document.createElement('h2');
        h2.textContent = 'Identity Assignment (Phase 2)';
        box.appendChild(h2);

        var desc = document.createElement('p');
        desc.style.color = 'var(--text-secondary)';
        desc.style.fontSize = '0.85rem';
        desc.style.marginBottom = '16px';
        var nDecisive = Object.keys(data.decisive || {}).length;
        var nIndecisive = Object.keys(data.indecisive || {}).length;
        desc.textContent = nDecisive + ' confident assignment(s), ' +
            nIndecisive + ' need your input. ' +
            'Centroids formed from ' + (data.centroid_count || '?') + ' confirmed groups.';
        box.appendChild(desc);

        var assignmentMap = {};  // crop_id -> cluster_id or null
        var self = this;

        // --- Decisive section: auto-confirm cards ---
        if (nDecisive > 0) {
            var decisiveHeader = document.createElement('h3');
            decisiveHeader.style.fontSize = '0.9rem';
            decisiveHeader.style.marginBottom = '8px';
            decisiveHeader.textContent = 'Confident Assignments (pre-accepted)';
            box.appendChild(decisiveHeader);

            var dGrid = document.createElement('div');
            dGrid.style.display = 'grid';
            dGrid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(180px, 1fr))';
            dGrid.style.gap = '10px';
            dGrid.style.marginBottom = '20px';

            var decisiveIds = Object.keys(data.decisive);
            for (var di = 0; di < decisiveIds.length; di++) {
                (function (cropId) {
                    var info = data.decisive[cropId];
                    assignmentMap[cropId] = info.cluster_id;  // pre-accept

                    var card = document.createElement('div');
                    card.style.border = '2px solid var(--color-accepted)';
                    card.style.borderRadius = '8px';
                    card.style.padding = '8px';
                    card.style.background = 'var(--bg-secondary)';
                    card.style.textAlign = 'center';

                    var img = document.createElement('img');
                    img.src = '/interview/api/detect/crop/' + cropId +
                        '/image?session_id=' + encodeURIComponent(self.sessionId);
                    img.alt = cropId;
                    img.style.width = '100%';
                    img.style.maxHeight = '100px';
                    img.style.objectFit = 'contain';
                    img.style.borderRadius = '4px';
                    card.appendChild(img);

                    var label = document.createElement('div');
                    label.style.fontSize = '0.7rem';
                    label.style.color = 'var(--text-secondary)';
                    label.style.margin = '4px 0';
                    label.innerHTML = 'Identity <strong>' + info.cluster_id +
                        '</strong> (conf: ' + info.confidence.toFixed(2) + ')';
                    card.appendChild(label);

                    // Reject button
                    var btnReject = document.createElement('button');
                    btnReject.className = 'btn btn-reject btn-small';
                    btnReject.textContent = 'Reject';
                    btnReject.style.fontSize = '0.65rem';
                    btnReject.style.padding = '2px 8px';
                    btnReject.addEventListener('click', function () {
                        delete assignmentMap[cropId];
                        card.style.borderColor = 'var(--color-rejected)';
                        btnReject.disabled = true;
                    });
                    card.appendChild(btnReject);

                    dGrid.appendChild(card);
                })(decisiveIds[di]);
            }
            box.appendChild(dGrid);
        }

        // --- Indecisive section: crop vs centroid cards ---
        if (nIndecisive > 0) {
            var indecisiveHeader = document.createElement('h3');
            indecisiveHeader.style.fontSize = '0.9rem';
            indecisiveHeader.style.marginBottom = '8px';
            indecisiveHeader.textContent = 'Needs Your Input';
            box.appendChild(indecisiveHeader);

            var iGrid = document.createElement('div');
            iGrid.style.display = 'grid';
            iGrid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(300px, 1fr))';
            iGrid.style.gap = '12px';
            iGrid.style.marginBottom = '20px';

            var indecisiveIds = Object.keys(data.indecisive);
            for (var ii = 0; ii < indecisiveIds.length; ii++) {
                (function (cropId) {
                    var info = data.indecisive[cropId];
                    var candidates = info.candidates || [];

                    var card = document.createElement('div');
                    card.style.border = '1px solid var(--border-default)';
                    card.style.borderRadius = '8px';
                    card.style.padding = '10px';
                    card.style.background = 'var(--bg-secondary)';

                    // Crop image (left)
                    var topRow = document.createElement('div');
                    topRow.style.display = 'flex';
                    topRow.style.gap = '10px';
                    topRow.style.marginBottom = '8px';

                    var cropCol = document.createElement('div');
                    cropCol.style.flex = '0 0 90px';
                    var cropImg = document.createElement('img');
                    cropImg.src = '/interview/api/detect/crop/' + cropId +
                        '/image?session_id=' + encodeURIComponent(self.sessionId);
                    cropImg.alt = cropId;
                    cropImg.style.width = '90px';
                    cropImg.style.height = '80px';
                    cropImg.style.objectFit = 'contain';
                    cropImg.style.borderRadius = '4px';
                    cropImg.style.border = '2px solid var(--border-default)';
                    cropCol.appendChild(cropImg);
                    var cropLabel = document.createElement('div');
                    cropLabel.style.fontSize = '0.65rem';
                    cropLabel.style.color = 'var(--text-secondary)';
                    cropLabel.style.textAlign = 'center';
                    cropLabel.style.marginTop = '2px';
                    cropLabel.textContent = 'This crop';
                    cropCol.appendChild(cropLabel);
                    topRow.appendChild(cropCol);

                    // Candidate centroids (right)
                    var candCol = document.createElement('div');
                    candCol.style.flex = '1';
                    candCol.style.display = 'flex';
                    candCol.style.gap = '8px';

                    for (var ci = 0; ci < candidates.length; ci++) {
                        (function (cand, candIdx) {
                            var candCard = document.createElement('div');
                            candCard.style.flex = '1';
                            candCard.style.textAlign = 'center';
                            candCard.style.padding = '4px';
                            candCard.style.border = '1px solid var(--border-default)';
                            candCard.style.borderRadius = '6px';
                            candCard.style.cursor = 'pointer';

                            // Representative thumbnails
                            var reps = cand.representatives || [];
                            var repRow = document.createElement('div');
                            repRow.style.display = 'flex';
                            repRow.style.gap = '2px';
                            repRow.style.justifyContent = 'center';
                            for (var ri = 0; ri < Math.min(reps.length, 3); ri++) {
                                var repImg = document.createElement('img');
                                repImg.src = '/interview/api/detect/crop/' + reps[ri] +
                                    '/image?session_id=' + encodeURIComponent(self.sessionId);
                                repImg.style.width = '40px';
                                repImg.style.height = '36px';
                                repImg.style.objectFit = 'contain';
                                repImg.style.borderRadius = '3px';
                                repRow.appendChild(repImg);
                            }
                            candCard.appendChild(repRow);

                            var candInfo = document.createElement('div');
                            candInfo.style.fontSize = '0.65rem';
                            candInfo.style.color = 'var(--text-secondary)';
                            candInfo.style.marginTop = '2px';
                            candInfo.textContent = 'Identity ' + cand.cluster_id +
                                ' (sim: ' + cand.similarity.toFixed(2) + ')';
                            candCard.appendChild(candInfo);

                            // Click to assign
                            candCard.addEventListener('click', function () {
                                assignmentMap[cropId] = cand.cluster_id;
                                card.style.borderColor = 'var(--color-accepted)';
                                card.style.borderWidth = '2px';
                                // Highlight selected candidate
                                var siblings = candCol.children;
                                for (var s = 0; s < siblings.length; s++) {
                                    siblings[s].style.borderColor = 'var(--border-default)';
                                    siblings[s].style.background = '';
                                }
                                candCard.style.borderColor = 'var(--color-accepted)';
                                candCard.style.background = 'var(--bg-accent)';
                            });

                            candCol.appendChild(candCard);
                        })(candidates[ci], ci);
                    }

                    topRow.appendChild(candCol);
                    card.appendChild(topRow);

                    // "Neither" button
                    var btnNeither = document.createElement('button');
                    btnNeither.className = 'btn btn-ghost btn-small';
                    btnNeither.textContent = 'Neither / New Identity';
                    btnNeither.style.fontSize = '0.65rem';
                    btnNeither.style.width = '100%';
                    btnNeither.addEventListener('click', function () {
                        assignmentMap[cropId] = null;
                        card.style.borderColor = 'var(--text-secondary)';
                        card.style.borderWidth = '2px';
                        var siblings = candCol.children;
                        for (var s = 0; s < siblings.length; s++) {
                            siblings[s].style.borderColor = 'var(--border-default)';
                            siblings[s].style.background = '';
                        }
                    });
                    card.appendChild(btnNeither);

                    iGrid.appendChild(card);
                })(indecisiveIds[ii]);
            }
            box.appendChild(iGrid);
        }

        // --- Action buttons ---
        var actions = document.createElement('div');
        actions.className = 'session-actions';

        var btnApply = document.createElement('button');
        btnApply.className = 'btn btn-primary';
        btnApply.textContent = 'Apply & Continue';
        btnApply.addEventListener('click', async function () {
            try {
                var result = await API.post('/reid/apply_associations', {
                    session_id: self.sessionId,
                    assignments: assignmentMap,
                });

                if (result.n_identities !== undefined) {
                    self.nIdentities = result.n_identities;
                }
                if (result.clusters) {
                    self.clusters = result.clusters;
                }
                if (result.reid_phase_stage !== undefined) {
                    self.phaseStage = result.reid_phase_stage;
                }

                var newDecisive = result.new_decisive || {};
                var newIndecisive = result.new_indecisive || {};
                var nNew = Object.keys(newDecisive).length + Object.keys(newIndecisive).length;

                if (result.converged || nNew === 0) {
                    showToast('Centroid assignment complete!', 'success');
                    self._renderSummary();
                } else {
                    showToast(Object.keys(newDecisive).length + ' new confident, ' +
                        Object.keys(newIndecisive).length + ' need input', 'info');
                    self._renderCentroidAssignments({
                        decisive: newDecisive,
                        indecisive: newIndecisive,
                        centroid_count: result.n_identities,
                    });
                }
            } catch (err) {
                showToast('Failed to apply: ' + err.message, 'error');
            }
        });
        actions.appendChild(btnApply);

        var btnFinish = document.createElement('button');
        btnFinish.className = 'btn btn-secondary';
        btnFinish.textContent = 'Finish ReID';
        btnFinish.addEventListener('click', function () {
            self._renderSummary();
        });
        actions.appendChild(btnFinish);

        box.appendChild(actions);
        panel.appendChild(box);
        this.container.appendChild(panel);
    }

    // ------------------------------------------------------------------
    // Merge status panel
    // ------------------------------------------------------------------

    _renderMergeStatus(container) {
        var statusPanel = document.createElement('div');
        statusPanel.style.flex = '1';
        statusPanel.style.overflowY = 'auto';
        statusPanel.style.padding = '12px 0';
        statusPanel.style.borderTop = '1px solid var(--border-default)';

        var heading = document.createElement('div');
        heading.style.fontSize = '0.75rem';
        heading.style.fontWeight = '600';
        heading.style.color = 'var(--text-secondary)';
        heading.style.marginBottom = '8px';
        heading.textContent = 'Merge Evidence';
        statusPanel.appendChild(heading);

        // Build a map of cluster-pair evidence from resolved pairs
        var clusterPairEvidence = {};
        for (var i = 0; i < this.pairs.length; i++) {
            var p = this.pairs[i];
            var res = this.resolutions[p.pair_id];
            if (!res) continue;

            var key = Math.min(p.cluster_a, p.cluster_b) + '-' +
                      Math.max(p.cluster_a, p.cluster_b);

            if (!clusterPairEvidence[key]) {
                clusterPairEvidence[key] = {
                    a: Math.min(p.cluster_a, p.cluster_b),
                    b: Math.max(p.cluster_a, p.cluster_b),
                    same: 0,
                    different: 0,
                    unsure: 0,
                };
            }
            clusterPairEvidence[key][res]++;
        }

        var keys = Object.keys(clusterPairEvidence);
        if (keys.length === 0) {
            var noEvidence = document.createElement('div');
            noEvidence.style.fontSize = '0.7rem';
            noEvidence.style.color = 'var(--text-muted)';
            noEvidence.textContent = 'No pair evidence yet. Resolve pairs to see merge status.';
            statusPanel.appendChild(noEvidence);
        } else {
            for (var k = 0; k < keys.length; k++) {
                var ev = clusterPairEvidence[keys[k]];
                var row = document.createElement('div');
                row.style.display = 'flex';
                row.style.alignItems = 'center';
                row.style.gap = '8px';
                row.style.marginBottom = '4px';
                row.style.fontSize = '0.7rem';
                row.style.color = 'var(--text-primary)';

                var label = 'Cluster ' + ev.a + ' <-> ' + ev.b + ': ';

                if (ev.different > 0) {
                    label += 'VETOED (different)';
                    row.style.color = 'var(--color-rejected)';
                } else if (ev.same >= 2) {
                    label += 'MERGED (' + ev.same + ' confirmations)';
                    row.style.color = 'var(--color-accepted)';
                } else {
                    label += ev.same + ' same, ' + ev.unsure + ' unsure';
                }

                row.textContent = label;
                statusPanel.appendChild(row);
            }
        }

        container.appendChild(statusPanel);
    }

    // ------------------------------------------------------------------
    // Summary (all pairs resolved)
    // ------------------------------------------------------------------

    _renderSummary() {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';
        panel.style.marginTop = '20px';

        var box = document.createElement('div');
        box.className = 'session-setup';
        box.style.maxWidth = '800px';

        var h2 = document.createElement('h2');
        h2.textContent = 'ReID Summary';
        box.appendChild(h2);

        // Resolved = pairs resolved in this session + pairs already resolved at init
        var localResolved = Object.keys(this.resolutions).length;
        var backendResolved = this.totalPairs - this.pairs.length;
        var resolved = backendResolved + localResolved;

        var stats = document.createElement('div');
        stats.style.marginBottom = '16px';
        var phaseNames = {1: 'Centroid Building', 2: 'Ambiguous Resolution', 3: 'Auto-Assignment'};
        var constraintInfo = '';
        if (this.mustLinkCount > 0 || this.cannotLinkCount > 0) {
            constraintInfo = '<br>Constraints: <strong>' +
                this.mustLinkCount + '</strong> same-person, <strong>' +
                this.cannotLinkCount + '</strong> different-person' +
                ' (preserved across re-clusters)';
        }
        var phaseInfo = '<br>Phase: <strong>' + (phaseNames[this.phaseStage] || 'Unknown') + '</strong>';
        var pairInfo = this.totalPairs > 0
            ? 'Resolved <strong>' + resolved + '</strong> of ' +
              '<strong>' + this.totalPairs + '</strong> pairs. '
            : '';
        stats.innerHTML =
            '<p style="margin-bottom:4px;color:var(--text-secondary);font-size:0.85rem">' +
            pairInfo +
            'Final identity count: <strong>' + this.nIdentities + '</strong>' +
            constraintInfo + phaseInfo +
            '</p>';
        box.appendChild(stats);

        // Visual cluster gallery
        var clusterKeys = Object.keys(this.clusters);
        if (clusterKeys.length > 0) {
            var gallery = document.createElement('div');
            gallery.className = 'reid-cluster-gallery';

            var MAX_THUMBS = 20;
            for (var c = 0; c < clusterKeys.length; c++) {
                (function (cKey, cluster, self) {
                    var cropIds = cluster.crop_ids || [];
                    var count = cluster.count || cropIds.length;

                    var section = document.createElement('div');
                    section.className = 'reid-cluster-section';

                    var header = document.createElement('div');
                    header.className = 'reid-cluster-header';
                    header.textContent = 'Identity ' + cKey + ' (' + count + ' crops)';
                    section.appendChild(header);

                    var row = document.createElement('div');
                    row.className = 'reid-cluster-row';

                    var visible = Math.min(cropIds.length, MAX_THUMBS);
                    for (var t = 0; t < visible; t++) {
                        var wrap = self._createThumbWrap(cropIds[t]);
                        row.appendChild(wrap);
                    }

                    if (cropIds.length > MAX_THUMBS) {
                        var overflow = document.createElement('div');
                        overflow.className = 'reid-overflow-badge';
                        var remaining = cropIds.length - MAX_THUMBS;
                        overflow.textContent = '+' + remaining + ' more';
                        row.appendChild(overflow);

                        var visibleCount = MAX_THUMBS;
                        overflow.addEventListener('click', function expandMore() {
                            var nextBatch = Math.min(visibleCount + MAX_THUMBS, cropIds.length);
                            for (var i = visibleCount; i < nextBatch; i++) {
                                var w = self._createThumbWrap(cropIds[i]);
                                row.insertBefore(w, overflow);
                            }
                            visibleCount = nextBatch;
                            var left = cropIds.length - visibleCount;
                            if (left > 0) {
                                overflow.textContent = '+' + left + ' more';
                            } else {
                                overflow.remove();
                            }
                        });
                    }

                    section.appendChild(row);
                    gallery.appendChild(section);
                })(clusterKeys[c], this.clusters[clusterKeys[c]], this);
            }

            box.appendChild(gallery);
        }

        // Action buttons
        var actions = document.createElement('div');
        actions.className = 'session-actions';

        var btnProceed = document.createElement('button');
        btnProceed.className = 'btn btn-primary';
        btnProceed.textContent = 'Proceed to Seeding';
        btnProceed.addEventListener('click', function () {
            if (typeof this.onComplete === 'function') {
                this.onComplete({
                    n_identities: this.nIdentities,
                    resolutions: this.resolutions,
                });
            }
        }.bind(this));
        actions.appendChild(btnProceed);

        var btnRecluster = document.createElement('button');
        btnRecluster.className = 'btn btn-secondary';
        btnRecluster.textContent = 'Re-cluster';
        btnRecluster.addEventListener('click', function () {
            this._toggleReclusterPanel(box);
        }.bind(this));
        actions.appendChild(btnRecluster);

        var btnVisualPipeline = document.createElement('button');
        btnVisualPipeline.className = 'btn btn-secondary';
        btnVisualPipeline.textContent = 'Visual Pipeline';
        btnVisualPipeline.addEventListener('click', function () {
            this._runVisualPipeline();
        }.bind(this));
        actions.appendChild(btnVisualPipeline);

        var btnBack = document.createElement('button');
        btnBack.className = 'btn btn-ghost';
        btnBack.textContent = 'Back to Detection';
        btnBack.addEventListener('click', function () {
            if (typeof navigate === 'function') navigate('detection');
        });
        actions.appendChild(btnBack);

        box.appendChild(actions);

        // Re-cluster panel placeholder
        var reclusterSlot = document.createElement('div');
        reclusterSlot.id = 'reid-recluster-slot';
        box.appendChild(reclusterSlot);

        // Progress area (for re-cluster)
        var progressArea = document.createElement('div');
        progressArea.id = 'reid-progress-area';
        progressArea.style.marginTop = '12px';
        box.appendChild(progressArea);

        panel.appendChild(box);
        this.container.appendChild(panel);

        this.destroy();
    }

    /**
     * Create a thumbnail wrapper with image + flag ("x") button.
     */
    _createThumbWrap(cropId) {
        var wrap = document.createElement('div');
        wrap.className = 'reid-thumb-wrap';

        var img = document.createElement('img');
        img.src = '/interview/api/detect/crop/' + cropId +
            '/image?session_id=' + encodeURIComponent(this.sessionId);
        img.alt = cropId;
        img.title = 'Crop ' + cropId;
        wrap.appendChild(img);

        var flagBtn = document.createElement('button');
        flagBtn.className = 'reid-thumb-flag';
        flagBtn.textContent = '\u00d7'; // ×
        flagBtn.title = 'Flag as outlier (move to new cluster)';
        flagBtn.addEventListener('click', async function (e) {
            e.stopPropagation();
            await this._flagOutlier(cropId);
        }.bind(this));
        wrap.appendChild(flagBtn);

        return wrap;
    }

    /**
     * Toggle the inline re-cluster settings panel.
     */
    _toggleReclusterPanel(parentBox) {
        var slot = document.getElementById('reid-recluster-slot');
        if (!slot) return;

        // Toggle: if already showing, remove
        if (slot.children.length > 0) {
            slot.innerHTML = '';
            return;
        }

        var rpanel = document.createElement('div');
        rpanel.className = 'reid-recluster-panel';

        // Auto checkbox
        var autoGroup = document.createElement('div');
        autoGroup.className = 'form-group';
        var autoLabel = document.createElement('label');
        autoLabel.style.display = 'flex';
        autoLabel.style.alignItems = 'center';
        autoLabel.style.gap = '6px';
        autoLabel.style.cursor = 'pointer';
        var autoCb = document.createElement('input');
        autoCb.type = 'checkbox';
        autoCb.id = 'recluster-auto';
        autoCb.checked = true;
        autoLabel.appendChild(autoCb);
        autoLabel.appendChild(document.createTextNode('Auto (with overclustering bias)'));
        autoGroup.appendChild(autoLabel);
        rpanel.appendChild(autoGroup);

        // K input
        var kGroup = document.createElement('div');
        kGroup.className = 'form-group';
        var kLbl = document.createElement('label');
        kLbl.textContent = 'Number of identities';
        kLbl.setAttribute('for', 'recluster-k');
        kGroup.appendChild(kLbl);
        var kInput = document.createElement('input');
        kInput.type = 'number';
        kInput.id = 'recluster-k';
        kInput.min = '2';
        kInput.max = '50';
        kInput.value = String((this.nIdentities || 2) + 2);
        kInput.style.maxWidth = '80px';
        kInput.disabled = true; // disabled when auto is checked
        kGroup.appendChild(kInput);
        rpanel.appendChild(kGroup);

        autoCb.addEventListener('change', function () {
            kInput.disabled = autoCb.checked;
        });

        // Run button
        var btnRun = document.createElement('button');
        btnRun.className = 'btn btn-primary btn-small';
        btnRun.textContent = 'Run';
        btnRun.addEventListener('click', function () {
            var useAuto = autoCb.checked;
            var k = useAuto ? null : parseInt(kInput.value, 10);
            this._runRecluster(k, useAuto);
        }.bind(this));
        rpanel.appendChild(btnRun);

        slot.appendChild(rpanel);
    }

    /**
     * Call the recluster endpoint and reload on completion.
     */
    async _runRecluster(nClusters, useAuto) {
        var progressArea = document.getElementById('reid-progress-area');
        if (progressArea) {
            progressArea.innerHTML =
                '<div class="progress-bar-wrapper">' +
                '<div class="progress-bar-track"><div class="progress-bar-fill indeterminate"></div></div>' +
                '<div class="progress-text"><span>Re-clustering...</span></div>' +
                '</div>';
        }

        try {
            var payload = { session_id: this.sessionId };
            if (!useAuto && nClusters != null && nClusters >= 2) {
                payload.n_clusters = nClusters;
            }
            var resp = await API.post('/reid/recluster', payload);

            var self = this;
            if (typeof pollJob === 'function') {
                pollJob(
                    resp.job_id,
                    function (p) {
                        if (progressArea) {
                            var text = progressArea.querySelector('.progress-text span');
                            if (text) text.textContent = p.step || 'Re-clustering...';
                            var fill = progressArea.querySelector('.progress-bar-fill');
                            if (fill && p.percent > 0) {
                                fill.classList.remove('indeterminate');
                                fill.style.width = p.percent + '%';
                            }
                        }
                    },
                    function (p) {
                        if (p.status === 'completed') {
                            showToast('Re-clustering complete!', 'success');
                            self.init(self.sessionId);
                        } else {
                            showToast('Re-clustering failed: ' + (p.error || 'Unknown'), 'error');
                            if (progressArea) progressArea.innerHTML = '';
                        }
                    },
                    1000
                );
            }
        } catch (err) {
            showToast('Failed to start re-clustering: ' + err.message, 'error');
            if (progressArea) progressArea.innerHTML = '';
        }
    }

    // ------------------------------------------------------------------
    // Flag outlier
    // ------------------------------------------------------------------

    async _flagOutlier(cropId) {
        try {
            var result = await API.post('/reid/flag_outlier', {
                session_id: this.sessionId,
                crop_id: cropId,
            });

            // Notify about new pairs
            if (result.new_pairs_count > 0) {
                showToast(
                    result.new_pairs_count + ' new pair(s) to review',
                    'warning'
                );
            } else {
                showToast('Crop moved to new cluster', 'success');
            }

            // Re-init: fetches updated clusters and unresolved pairs,
            // routes to pair comparison if new pairs exist, else summary
            this.init(this.sessionId);
        } catch (err) {
            showToast('Failed to flag outlier: ' + err.message, 'error');
        }
    }

    // ------------------------------------------------------------------
    // Keyboard handler
    // ------------------------------------------------------------------

    _handleKeyDown(e) {
        // Ignore if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        var key = e.key.toLowerCase();
        if (key === 'f') {
            e.preventDefault();
            this._resolvePair('same');
        } else if (key === 'j') {
            e.preventDefault();
            this._resolvePair('different');
        } else if (key === ' ') {
            e.preventDefault();
            this._resolvePair('unsure');
        } else if (key === 'arrowleft') {
            e.preventDefault();
            this._goBack();
        }
    }

    _goBack() {
        if (this.currentPairIndex > 0) {
            // Find previous pair (even if resolved, let user revisit)
            this._showPair(this.currentPairIndex - 1);
        }
    }
}

/**
 * Global bridge function called by app.js renderReID().
 * Instantiates ReIDUI, inits with current session, and wires
 * onComplete to navigate to the seeding phase.
 */
function renderReIDPhase(container) {
    var ui = new ReIDUI(container);
    // Register for cleanup on navigation (prevents keydown listener leak)
    if (typeof AppState !== 'undefined' && AppState._components) {
        AppState._components.reidUI = ui;
    }
    ui.onComplete = function () {
        if (typeof navigate === 'function') {
            navigate('seeding');
        }
    };
    ui.init(AppState.sessionId);
}
