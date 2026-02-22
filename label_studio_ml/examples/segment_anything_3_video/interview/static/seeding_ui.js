/**
 * Seeding UI - Configure seed generation, preview results, upload to Label Studio.
 *
 * Final phase of the interview workflow. Takes the resolved identity clusters
 * and generates keyframe seeds that can be uploaded to Label Studio, then
 * tracked offline via the CLI.
 *
 * Relies on the global API object (from app.js) for HTTP requests
 * and showToast() for notifications.
 */

/* exported SeedingUI */
/* global API, showToast, AppState, navigate */

class SeedingUI {
    constructor(container) {
        this.container = container;
        this.sessionId = null;
        this.seeds = [];
        this.identities = {};
        this.seedConfig = { frame_pct: 100, confidence_threshold: 0.8 };
        this.changeKeyframes = [];
        this.framesCount = 0;
        this.cachedFrameCount = 0;
        this.isGenerating = false;
        this.isUploading = false;
        this.currentGenerateJobId = null;
        this.stopRequested = false;
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    async init(sessionId) {
        this.sessionId = sessionId;

        // Check if ReID clusters exist — seeding requires identities
        try {
            var reidData = await API.get('/reid/clusters', {
                session_id: sessionId,
            });
            if (!reidData.n_identities || reidData.n_identities === 0) {
                this._renderNoClusterWarning();
                return;
            }
        } catch (_) {
            // If we can't check, proceed and let server-side catch it
        }

        // Load existing config + cache stats
        try {
            var cfg = await API.get('/seeds/config', {
                session_id: sessionId,
            });
            this.seedConfig.frame_pct = cfg.frame_pct || 100;
            this.seedConfig.confidence_threshold = cfg.confidence_threshold || 0.8;
            this.changeKeyframes = cfg.change_keyframes || [];
            this.framesCount = cfg.frames_count || 0;
            this.cachedFrameCount = cfg.cached_frame_count || 0;
        } catch (_) {
            // Use defaults
        }

        // Check if seeds already exist
        try {
            var existing = await API.get('/seeds/list', {
                session_id: sessionId,
            });
            if (existing.total_seeds > 0) {
                this.identities = existing.identities || {};
                this.seedConfig.frame_pct = existing.seed_config
                    ? (existing.seed_config.frame_pct || 100) : this.seedConfig.frame_pct;
                this.seedConfig.confidence_threshold = existing.seed_config
                    ? existing.seed_config.confidence_threshold : this.seedConfig.confidence_threshold;
                this._renderPreview(existing);
                return;
            }
        } catch (_) {
            // No seeds yet
        }

        this._renderConfig();
    }

    _renderNoClusterWarning() {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';
        panel.style.marginTop = '40px';

        var box = document.createElement('div');
        box.className = 'session-setup';
        box.style.maxWidth = '600px';

        var h2 = document.createElement('h2');
        h2.textContent = 'Seeding Requires ReID';
        box.appendChild(h2);

        var desc = document.createElement('p');
        desc.style.color = 'var(--text-secondary)';
        desc.style.fontSize = '0.85rem';
        desc.style.marginBottom = '20px';
        desc.textContent =
            'No identity clusters found. You need to run ReID clustering first ' +
            'so that seeds can be assigned to identities.';
        box.appendChild(desc);

        var actions = document.createElement('div');
        actions.className = 'session-actions';

        var btnBack = document.createElement('button');
        btnBack.className = 'btn btn-primary';
        btnBack.textContent = 'Go to ReID';
        btnBack.addEventListener('click', function () {
            if (typeof navigate === 'function') navigate('reid');
        });
        actions.appendChild(btnBack);

        var btnDetection = document.createElement('button');
        btnDetection.className = 'btn btn-ghost';
        btnDetection.textContent = 'Back to Detection';
        btnDetection.addEventListener('click', function () {
            if (typeof navigate === 'function') navigate('detection');
        });
        actions.appendChild(btnDetection);

        box.appendChild(actions);
        panel.appendChild(box);
        this.container.appendChild(panel);
    }

    // ------------------------------------------------------------------
    // Configuration form
    // ------------------------------------------------------------------

    _renderConfig() {
        this.container.innerHTML = '';
        var self = this;

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';

        // -- Header --
        var header = document.createElement('h2');
        header.textContent = 'Seed Generation';
        header.style.marginBottom = '12px';
        panel.appendChild(header);

        // -- Explanation --
        var explanation = document.createElement('p');
        explanation.style.color = 'var(--text-secondary)';
        explanation.style.fontSize = '0.85rem';
        explanation.style.marginBottom = '20px';
        explanation.textContent =
            'Seeds are generated from cached video frames. ' +
            'Only detections with confidence above the threshold will be kept. ' +
            'Each seed is a bounding box keyframe tied to an identity.';
        panel.appendChild(explanation);

        // -- Change keyframe coverage visualization --
        var nChange = this.changeKeyframes.length;
        var totalFrames = this.framesCount;
        var fps = (AppState.stats && AppState.stats.fps) || 25;

        if (nChange > 0 && totalFrames > 0) {
            // Compute gap statistics
            var sorted = this.changeKeyframes.slice().sort(function (a, b) { return a - b; });
            var gaps = [];
            for (var gi = 1; gi < sorted.length; gi++) {
                gaps.push(sorted[gi] - sorted[gi - 1]);
            }
            var maxGap = gaps.length > 0 ? Math.max.apply(null, gaps) : 0;
            var avgGap = gaps.length > 0 ? Math.round(gaps.reduce(function (s, v) { return s + v; }, 0) / gaps.length) : 0;

            // Stats line
            var statsBar = document.createElement('div');
            statsBar.style.padding = '10px 14px';
            statsBar.style.background = 'var(--bg-accent)';
            statsBar.style.borderRadius = 'var(--radius-sm)';
            statsBar.style.fontSize = '0.8rem';
            statsBar.style.color = 'var(--text-secondary)';
            statsBar.style.marginBottom = '12px';
            statsBar.style.display = 'flex';
            statsBar.style.flexWrap = 'wrap';
            statsBar.style.gap = '16px';
            var items = [
                nChange.toLocaleString() + ' change keyframes',
                'avg spacing: ' + avgGap + ' frames (' + (avgGap / fps).toFixed(1) + 's)',
                'largest gap: ' + maxGap + ' frames (' + (maxGap / fps).toFixed(1) + 's)',
            ];
            items.forEach(function (text) {
                var span = document.createElement('span');
                span.textContent = text;
                statsBar.appendChild(span);
            });
            panel.appendChild(statsBar);

            // Coverage strip label
            var timelineLabel = document.createElement('div');
            timelineLabel.style.fontSize = '0.7rem';
            timelineLabel.style.color = 'var(--text-muted)';
            timelineLabel.style.marginBottom = '4px';
            timelineLabel.textContent = 'Coverage across video (bright = dense keyframes, dark = gaps)';
            panel.appendChild(timelineLabel);

            // Canvas: tick row (8px) + coverage strip (24px)
            var canvas = document.createElement('canvas');
            canvas.width = 800;
            canvas.height = 32;
            canvas.style.width = '100%';
            canvas.style.height = '32px';
            canvas.style.borderRadius = 'var(--radius-sm)';
            canvas.style.marginBottom = '20px';
            panel.appendChild(canvas);

            setTimeout(function () { self._renderTimeline(canvas); }, 0);
        } else {
            var statsBar = document.createElement('div');
            statsBar.style.padding = '10px 14px';
            statsBar.style.background = 'var(--bg-accent)';
            statsBar.style.borderRadius = 'var(--radius-sm)';
            statsBar.style.fontSize = '0.8rem';
            statsBar.style.color = 'var(--text-secondary)';
            statsBar.style.marginBottom = '12px';
            statsBar.textContent = 'No change keyframes detected.';
            panel.appendChild(statsBar);
        }

        // -- Config form --
        var configBox = document.createElement('div');
        configBox.className = 'seeding-config';

        // Frame coverage (% slider)
        var pctGroup = document.createElement('div');
        pctGroup.className = 'form-group';

        var pctLabel = document.createElement('label');
        pctLabel.setAttribute('for', 'seed-frame-pct');
        pctLabel.textContent = 'Frame Coverage';

        var pctRow = document.createElement('div');
        pctRow.style.display = 'flex';
        pctRow.style.alignItems = 'center';
        pctRow.style.gap = '10px';

        var pctSlider = document.createElement('input');
        pctSlider.type = 'range';
        pctSlider.id = 'seed-frame-pct';
        pctSlider.min = '1';
        pctSlider.max = '100';
        pctSlider.step = '1';
        pctSlider.value = String(this.seedConfig.frame_pct);
        pctSlider.style.flex = '1';

        var pctValue = document.createElement('span');
        pctValue.style.fontSize = '0.85rem';
        pctValue.style.color = 'var(--text-primary)';
        pctValue.style.minWidth = '40px';
        pctValue.style.textAlign = 'right';
        pctValue.textContent = this.seedConfig.frame_pct + '%';

        var pctHint = document.createElement('div');
        pctHint.style.fontSize = '0.7rem';
        pctHint.style.color = 'var(--text-muted)';
        pctHint.style.marginTop = '4px';

        // Compute initial hint text
        var initPct = this.seedConfig.frame_pct;
        var initFrames = Math.max(1, Math.round(this.cachedFrameCount * initPct / 100));
        var initEffective = Math.max(initFrames, nChange);
        pctHint.textContent = 'Scan ~' + initEffective.toLocaleString() + ' of ' +
            this.cachedFrameCount.toLocaleString() + ' cached frames (' +
            nChange + ' change keyframes always included)';

        pctSlider.addEventListener('input', function () {
            var pct = parseInt(pctSlider.value, 10);
            var nFrames = Math.max(1, Math.round(self.cachedFrameCount * pct / 100));
            var effective = Math.max(nFrames, nChange);
            pctValue.textContent = pct + '%';
            pctHint.textContent = 'Scan ~' + effective.toLocaleString() + ' of ' +
                self.cachedFrameCount.toLocaleString() + ' cached frames (' +
                nChange + ' change keyframes always included)';
        });

        pctRow.appendChild(pctSlider);
        pctRow.appendChild(pctValue);

        pctGroup.appendChild(pctLabel);
        pctGroup.appendChild(pctRow);
        pctGroup.appendChild(pctHint);

        // Confidence threshold
        var threshGroup = document.createElement('div');
        threshGroup.className = 'form-group';

        var threshLabel = document.createElement('label');
        threshLabel.setAttribute('for', 'seed-confidence');
        threshLabel.textContent = 'Confidence Threshold';

        var threshRow = document.createElement('div');
        threshRow.style.display = 'flex';
        threshRow.style.alignItems = 'center';
        threshRow.style.gap = '10px';

        var threshSlider = document.createElement('input');
        threshSlider.type = 'range';
        threshSlider.id = 'seed-confidence';
        threshSlider.min = '0';
        threshSlider.max = '1';
        threshSlider.step = '0.05';
        threshSlider.value = String(this.seedConfig.confidence_threshold);
        threshSlider.style.flex = '1';

        var threshValue = document.createElement('span');
        threshValue.id = 'seed-confidence-value';
        threshValue.style.fontSize = '0.85rem';
        threshValue.style.color = 'var(--text-primary)';
        threshValue.style.minWidth = '40px';
        threshValue.style.textAlign = 'right';
        threshValue.textContent = this.seedConfig.confidence_threshold.toFixed(2);

        threshSlider.addEventListener('input', function () {
            threshValue.textContent = parseFloat(threshSlider.value).toFixed(2);
        });

        threshRow.appendChild(threshSlider);
        threshRow.appendChild(threshValue);

        var threshHint = document.createElement('div');
        threshHint.style.fontSize = '0.7rem';
        threshHint.style.color = 'var(--text-muted)';
        threshHint.style.marginTop = '4px';
        threshHint.textContent = 'Only keep detections above this score (default: 0.80)';

        threshGroup.appendChild(threshLabel);
        threshGroup.appendChild(threshRow);
        threshGroup.appendChild(threshHint);

        configBox.appendChild(pctGroup);
        configBox.appendChild(threshGroup);
        panel.appendChild(configBox);

        // -- Generate button --
        var actions = document.createElement('div');
        actions.className = 'seeding-actions';
        actions.style.marginTop = '20px';
        actions.id = 'seed-generate-actions';

        var btnGenerate = document.createElement('button');
        btnGenerate.className = 'btn btn-primary';
        btnGenerate.id = 'btn-generate-seeds';
        btnGenerate.textContent = 'Generate Seeds';
        btnGenerate.addEventListener('click', function () {
            var framePct = parseInt(pctSlider.value, 10);
            var threshold = parseFloat(threshSlider.value);
            self._generateSeeds(framePct, threshold);
        });

        var btnStop = document.createElement('button');
        btnStop.className = 'btn btn-reject';
        btnStop.id = 'btn-stop-generate-seeds';
        btnStop.textContent = 'Stop Generating';
        btnStop.style.display = 'none';
        btnStop.addEventListener('click', function () {
            self._stopGenerating();
        });

        actions.appendChild(btnGenerate);
        actions.appendChild(btnStop);
        panel.appendChild(actions);

        // -- Progress area (hidden initially) --
        var progressArea = document.createElement('div');
        progressArea.id = 'seed-progress-area';
        progressArea.style.marginTop = '20px';
        progressArea.style.display = 'none';
        panel.appendChild(progressArea);

        this.container.appendChild(panel);
    }

    // ------------------------------------------------------------------
    // Timeline sparkline
    // ------------------------------------------------------------------

    _renderTimeline(canvas) {
        var ctx = canvas.getContext('2d');
        var W = canvas.width;
        var H = canvas.height;
        var total = this.framesCount;
        if (total === 0) return;

        // Dark background
        ctx.fillStyle = '#0d0d1a';
        ctx.fillRect(0, 0, W, H);

        // Draw each keyframe as a full-height tick mark.
        // Clusters of ticks overlap visually → brighter regions.
        ctx.fillStyle = 'rgba(233, 69, 96, 0.35)';
        var kfs = this.changeKeyframes;
        for (var i = 0; i < kfs.length; i++) {
            var x = Math.round(kfs[i] / total * W);
            ctx.fillRect(x, 0, 2, H);
        }

        // Crisp white tick on top for exact positions
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        for (var i = 0; i < kfs.length; i++) {
            var x = Math.round(kfs[i] / total * W);
            ctx.fillRect(x, 0, 1, H);
        }
    }

    // ------------------------------------------------------------------
    // Generate seeds
    // ------------------------------------------------------------------

    async _generateSeeds(framePct, confidenceThreshold) {
        if (this.isGenerating) return;
        this.isGenerating = true;
        this.stopRequested = false;

        var btn = document.getElementById('btn-generate-seeds');
        var btnStop = document.getElementById('btn-stop-generate-seeds');
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Generating...';
        }
        if (btnStop) {
            btnStop.style.display = '';
            btnStop.disabled = false;
        }

        // Update config on the backend
        try {
            await API.put('/seeds/config', {
                session_id: this.sessionId,
                frame_pct: framePct,
                confidence_threshold: confidenceThreshold,
            });
        } catch (err) {
            showToast('Failed to update seed config: ' + err.message, 'error');
            this.isGenerating = false;
            if (btn) { btn.disabled = false; btn.textContent = 'Generate Seeds'; }
            if (btnStop) { btnStop.style.display = 'none'; }
            return;
        }

        // Start generation job
        var jobId;
        try {
            var resp = await API.post('/seeds/generate', {
                session_id: this.sessionId,
            });
            jobId = resp.job_id;
        } catch (err) {
            showToast('Failed to start seed generation: ' + err.message, 'error');
            this.isGenerating = false;
            if (btn) { btn.disabled = false; btn.textContent = 'Generate Seeds'; }
            if (btnStop) { btnStop.style.display = 'none'; }
            return;
        }
        this.currentGenerateJobId = jobId;

        // Show progress
        this._showProgress('Generating seeds...', {
            framePct: framePct,
            confidenceThreshold: confidenceThreshold,
            estimatedFrames: this._estimateTargetFrames(framePct),
        });

        // Poll for completion
        var pollResult = await this._pollJob(jobId);
        this.currentGenerateJobId = null;

        if (!pollResult || pollResult.status === 'failed') {
            this.isGenerating = false;
            this.stopRequested = false;
            this._renderConfig();
            return;
        }
        if (pollResult.status === 'cancelled') {
            // Fetch partial seeds — backend saves them before raising cancellation
            try {
                var cancelledSeedData = await API.get('/seeds/list', {
                    session_id: this.sessionId,
                });
                if (cancelledSeedData.total_seeds > 0) {
                    this.identities = cancelledSeedData.identities || {};
                    showToast('Cancelled — ' + cancelledSeedData.total_seeds + ' partial seeds saved.', 'warning');
                    this._renderPreview(cancelledSeedData);
                } else {
                    this._renderConfig();
                }
            } catch (_) {
                this._renderConfig();
            }
            this.isGenerating = false;
            this.stopRequested = false;
            return;
        }

        // Fetch seed list
        try {
            var seedData = await API.get('/seeds/list', {
                session_id: this.sessionId,
            });
            this.identities = seedData.identities || {};
            showToast('Generated ' + seedData.total_seeds + ' seeds.', 'success');
            this._renderPreview(seedData);
        } catch (err) {
            showToast('Failed to load seeds: ' + err.message, 'error');
            this._renderConfig();
        }

        this.isGenerating = false;
        this.stopRequested = false;
    }

    // ------------------------------------------------------------------
    // Preview
    // ------------------------------------------------------------------

    _renderPreview(seedData) {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';

        var header = document.createElement('h2');
        header.textContent = 'Seed Preview';
        header.style.marginBottom = '16px';
        panel.appendChild(header);

        // -- Summary bar --
        var summary = document.createElement('div');
        summary.style.display = 'flex';
        summary.style.gap = '24px';
        summary.style.marginBottom = '20px';
        summary.style.padding = '12px 16px';
        summary.style.background = 'var(--bg-accent)';
        summary.style.borderRadius = 'var(--radius-md)';
        summary.style.fontSize = '0.85rem';

        var identityKeys = Object.keys(seedData.identities || {});

        summary.innerHTML =
            '<div><span style="color:var(--text-secondary)">Total Seeds:</span> ' +
            '<strong>' + seedData.total_seeds + '</strong></div>' +
            '<div><span style="color:var(--text-secondary)">Identities:</span> ' +
            '<strong>' + identityKeys.length + '</strong></div>' +
            '<div><span style="color:var(--text-secondary)">Coverage:</span> ' +
            '<strong>' + (seedData.seed_config ? (seedData.seed_config.frame_pct || 100) : '?') +
            '%</strong></div>' +
            '<div><span style="color:var(--text-secondary)">Threshold:</span> ' +
            '<strong>' + (seedData.seed_config ? seedData.seed_config.confidence_threshold : '?') +
            '</strong></div>';

        panel.appendChild(summary);

        // -- Identity table --
        if (identityKeys.length > 0) {
            var tableWrapper = document.createElement('div');
            tableWrapper.className = 'seed-table-wrapper';

            var table = document.createElement('table');
            table.className = 'seed-table';

            var thead = document.createElement('thead');
            thead.innerHTML =
                '<tr>' +
                '<th>Identity</th>' +
                '<th>Seed Count</th>' +
                '<th>Frame Range</th>' +
                '</tr>';
            table.appendChild(thead);

            var tbody = document.createElement('tbody');
            for (var i = 0; i < identityKeys.length; i++) {
                var id = identityKeys[i];
                var info = seedData.identities[id];
                var frames = (info.frames || []).slice().sort(function (a, b) { return a - b; });
                var minFrame = frames.length > 0 ? frames[0] : '?';
                var maxFrame = frames.length > 0 ? frames[frames.length - 1] : '?';

                var tr = document.createElement('tr');
                tr.innerHTML =
                    '<td><strong>' + this._formatIdentityLabel(id) + '</strong></td>' +
                    '<td>' + info.count + '</td>' +
                    '<td>' + minFrame + ' - ' + maxFrame + '</td>';
                tbody.appendChild(tr);
            }

            table.appendChild(tbody);
            tableWrapper.appendChild(table);
            panel.appendChild(tableWrapper);
        }

        // -- Action buttons --
        var actions = document.createElement('div');
        actions.className = 'seeding-actions';

        var btnRegenerate = document.createElement('button');
        btnRegenerate.className = 'btn btn-secondary';
        btnRegenerate.textContent = 'Regenerate';
        btnRegenerate.addEventListener('click', function () {
            this._renderConfig();
        }.bind(this));

        var btnUpload = document.createElement('button');
        btnUpload.className = 'btn btn-accept';
        btnUpload.id = 'btn-upload-seeds';
        btnUpload.textContent = 'Upload to Label Studio';
        btnUpload.style.padding = '10px 24px';
        btnUpload.style.fontSize = '0.9rem';
        btnUpload.addEventListener('click', function () {
            this._confirmUpload(seedData);
        }.bind(this));

        actions.appendChild(btnRegenerate);
        actions.appendChild(btnUpload);
        panel.appendChild(actions);

        this.container.appendChild(panel);
    }

    _formatIdentityLabel(id) {
        if (id === 'unknown') return 'Unknown';
        // Make cluster IDs more readable
        return 'Identity ' + id;
    }

    // ------------------------------------------------------------------
    // Upload confirmation + execution
    // ------------------------------------------------------------------

    _confirmUpload(seedData) {
        var identityCount = Object.keys(seedData.identities || {}).length;

        var modal = document.getElementById('modal-container');
        modal.innerHTML = '';
        modal.setAttribute('aria-hidden', 'false');

        var box = document.createElement('div');
        box.className = 'modal-box';

        var h3 = document.createElement('h3');
        h3.textContent = 'Upload Seeds to Label Studio?';
        box.appendChild(h3);

        var p1 = document.createElement('p');
        p1.innerHTML =
            'Upload <strong>' + seedData.total_seeds + '</strong> seeds for ' +
            '<strong>' + identityCount + '</strong> identities to Label Studio.';
        box.appendChild(p1);

        var p2 = document.createElement('p');
        p2.style.fontSize = '0.8rem';
        p2.innerHTML =
            'Seeds will be created with <code>enabled=false</code> (no auto-interpolation). ' +
            'After upload, run the tracking CLI to fill gaps between keyframes.';
        box.appendChild(p2);

        var modalActions = document.createElement('div');
        modalActions.className = 'modal-actions';

        var btnCancel = document.createElement('button');
        btnCancel.className = 'btn btn-ghost';
        btnCancel.textContent = 'Cancel';
        btnCancel.addEventListener('click', function () {
            modal.setAttribute('aria-hidden', 'true');
            modal.innerHTML = '';
        });

        var btnConfirm = document.createElement('button');
        btnConfirm.className = 'btn btn-accept';
        btnConfirm.textContent = 'Upload';
        btnConfirm.addEventListener('click', function () {
            modal.setAttribute('aria-hidden', 'true');
            modal.innerHTML = '';
            this._uploadSeeds();
        }.bind(this));

        modalActions.appendChild(btnCancel);
        modalActions.appendChild(btnConfirm);
        box.appendChild(modalActions);

        modal.appendChild(box);
    }

    async _uploadSeeds() {
        if (this.isUploading) return;
        this.isUploading = true;

        var btn = document.getElementById('btn-upload-seeds');
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Uploading...';
        }

        var jobId;
        try {
            var resp = await API.post('/seeds/upload', {
                session_id: this.sessionId,
            });
            jobId = resp.job_id;
        } catch (err) {
            showToast('Failed to start upload: ' + err.message, 'error');
            this.isUploading = false;
            if (btn) { btn.disabled = false; btn.textContent = 'Upload to Label Studio'; }
            return;
        }

        showToast('Uploading seeds to Label Studio...', 'info');

        var result = await this._pollJob(jobId);
        this.isUploading = false;

        if (result && result.status === 'completed') {
            showToast('Seeds uploaded successfully!', 'success');
            this._renderSuccess(result.result || {});
        } else {
            showToast(
                'Upload failed: ' + (result ? result.error || 'Unknown error' : 'Lost connection'),
                'error'
            );
            if (btn) { btn.disabled = false; btn.textContent = 'Upload to Label Studio'; }
        }
    }

    // ------------------------------------------------------------------
    // Success screen
    // ------------------------------------------------------------------

    _renderSuccess(result) {
        this.container.innerHTML = '';

        var panel = document.createElement('div');
        panel.className = 'seeding-panel';
        panel.style.marginTop = '40px';

        var box = document.createElement('div');
        box.className = 'session-setup';
        box.style.maxWidth = '700px';

        var h2 = document.createElement('h2');
        h2.textContent = 'Seeds Uploaded Successfully';
        h2.style.color = 'var(--color-accepted)';
        box.appendChild(h2);

        // Stats
        var stats = document.createElement('div');
        stats.style.margin = '20px 0';
        stats.style.fontSize = '0.85rem';
        stats.style.color = 'var(--text-secondary)';

        var regionsCreated = result.regions_created || result.total_seeds || '?';
        var keyframesPerRegion = result.keyframes_per_region || '?';
        var annotationId = result.annotation_id || '?';
        var projectId = result.project_id || '?';
        var taskId = result.task_id || '?';

        stats.innerHTML =
            '<p style="margin-bottom:8px">Regions created: <strong>' + regionsCreated + '</strong></p>' +
            '<p style="margin-bottom:8px">Keyframes per region: <strong>' + keyframesPerRegion + '</strong></p>' +
            '<p style="margin-bottom:8px">Annotation ID: <strong>' + annotationId + '</strong></p>';
        box.appendChild(stats);

        // CLI command
        var cliSection = document.createElement('div');
        cliSection.style.marginTop = '20px';

        var cliLabel = document.createElement('div');
        cliLabel.style.fontSize = '0.8rem';
        cliLabel.style.fontWeight = '600';
        cliLabel.style.marginBottom = '8px';
        cliLabel.style.color = 'var(--text-primary)';
        cliLabel.textContent = 'Next step: Run tracking CLI to fill gaps between keyframes';
        cliSection.appendChild(cliLabel);

        var cliCommand = document.createElement('pre');
        cliCommand.style.background = 'var(--bg-body)';
        cliCommand.style.border = '1px solid var(--border-default)';
        cliCommand.style.borderRadius = 'var(--radius-sm)';
        cliCommand.style.padding = '12px 16px';
        cliCommand.style.fontSize = '0.75rem';
        cliCommand.style.overflowX = 'auto';
        cliCommand.style.color = 'var(--text-primary)';
        cliCommand.style.whiteSpace = 'pre-wrap';
        cliCommand.style.wordBreak = 'break-all';

        cliCommand.textContent =
            'docker compose exec segment_anything_3_video python \\\n' +
            '  /app/initial_seeding_video_boxes_manual_merge.py \\\n' +
            '  --ls-url "$LABEL_STUDIO_HOST" \\\n' +
            '  --ls-api-key "$LABEL_STUDIO_API_KEY" \\\n' +
            '  --project ' + projectId + ' \\\n' +
            '  --task ' + taskId + ' \\\n' +
            '  --annotation ' + annotationId + ' \\\n' +
            '  --max-frames-to-track 300';
        cliSection.appendChild(cliCommand);

        // Copy button
        var btnCopy = document.createElement('button');
        btnCopy.className = 'btn btn-ghost btn-small';
        btnCopy.textContent = 'Copy Command';
        btnCopy.style.marginTop = '8px';
        btnCopy.addEventListener('click', function () {
            var text = cliCommand.textContent;
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(function () {
                    showToast('Command copied to clipboard.', 'success');
                }).catch(function () {
                    showToast('Failed to copy.', 'warning');
                });
            } else {
                // Fallback
                var ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                try {
                    document.execCommand('copy');
                    showToast('Command copied to clipboard.', 'success');
                } catch (_) {
                    showToast('Failed to copy.', 'warning');
                }
                document.body.removeChild(ta);
            }
        });
        cliSection.appendChild(btnCopy);

        box.appendChild(cliSection);
        panel.appendChild(box);
        this.container.appendChild(panel);
    }

    // ------------------------------------------------------------------
    // Progress helpers
    // ------------------------------------------------------------------

    _showProgress(message, meta) {
        var area = document.getElementById('seed-progress-area');
        if (!area) return;

        area.style.display = 'block';
        area.innerHTML = '';

        var wrapper = document.createElement('div');
        wrapper.className = 'progress-bar-wrapper';

        var track = document.createElement('div');
        track.className = 'progress-bar-track';
        var fill = document.createElement('div');
        fill.className = 'progress-bar-fill indeterminate';
        track.appendChild(fill);

        var text = document.createElement('div');
        text.className = 'progress-text';
        text.innerHTML = '<span>' + message + '</span><span id="seed-progress-pct"></span>';

        wrapper.appendChild(track);
        wrapper.appendChild(text);
        area.appendChild(wrapper);

        if (meta) {
            var summary = document.createElement('div');
            summary.className = 'seeding-config';
            summary.style.marginTop = '12px';
            summary.style.padding = '12px 14px';
            summary.style.fontSize = '0.8rem';
            summary.style.color = 'var(--text-secondary)';
            summary.innerHTML =
                '<div style="display:flex;gap:16px;flex-wrap:wrap">' +
                '<span>Coverage: <strong style="color:var(--text-primary)">' + meta.framePct + '%</strong></span>' +
                '<span>Threshold: <strong style="color:var(--text-primary)">' + meta.confidenceThreshold.toFixed(2) + '</strong></span>' +
                '<span>Estimated frames: <strong style="color:var(--text-primary)">' + meta.estimatedFrames.toLocaleString() + '</strong></span>' +
                '</div>';
            area.appendChild(summary);
        }
    }

    _updateProgress(progress) {
        var fill = document.querySelector('#seed-progress-area .progress-bar-fill');
        var pctEl = document.getElementById('seed-progress-pct');
        var textEl = document.querySelector('#seed-progress-area .progress-text span:first-child');

        if (fill && progress.percent > 0) {
            fill.classList.remove('indeterminate');
            fill.style.width = progress.percent + '%';
        }
        if (pctEl) {
            pctEl.textContent = progress.percent > 0 ? progress.percent + '%' : '';
        }
        if (textEl && progress.step) {
            textEl.textContent = progress.step;
        }
    }

    async _pollJob(jobId) {
        var maxAttempts = 3600;  // 60 minutes at 1s interval
        var attempt = 0;

        while (attempt < maxAttempts) {
            await this._sleep(1000);
            attempt++;

            try {
                var progress = await API.get('/job/' + jobId + '/progress');
                this._updateProgress(progress);

                if (progress.status === 'completed') {
                    return progress;
                }
                if (progress.status === 'cancelled') {
                    return progress;
                }
                if (progress.status === 'failed') {
                    showToast('Job failed: ' + (progress.error || 'Unknown error'), 'error');
                    return progress;
                }
            } catch (err) {
                // Network error, keep polling
                if (attempt % 10 === 0) {
                    showToast('Connection issue, retrying...', 'warning');
                }
            }
        }

        showToast('Job timed out after 60 minutes.', 'error');
        return null;
    }

    _sleep(ms) {
        return new Promise(function (resolve) {
            setTimeout(resolve, ms);
        });
    }

    _estimateTargetFrames(framePct) {
        var nChange = this.changeKeyframes.length || 0;
        var cacheCount = this.cachedFrameCount || 0;
        var sampled = Math.max(1, Math.round(cacheCount * framePct / 100));
        return Math.max(sampled, nChange);
    }

    async _stopGenerating() {
        var jobId = this.currentGenerateJobId;
        if (!jobId || this.stopRequested) return;
        this.stopRequested = true;

        var btnStop = document.getElementById('btn-stop-generate-seeds');
        if (btnStop) btnStop.disabled = true;

        try {
            await API.post('/job/' + jobId + '/cancel', {});
            showToast('Stopping seed generation...', 'info');
        } catch (err) {
            showToast('Stop request failed: ' + err.message, 'warning');
        }
        // Let _generateSeeds handle cleanup when _pollJob returns
    }
}

/**
 * Global bridge function called by app.js renderSeeding().
 * Instantiates SeedingUI and inits with current session.
 */
function renderSeedingPhase(container) {
    var ui = new SeedingUI(container);
    ui.init(AppState.sessionId);
}
