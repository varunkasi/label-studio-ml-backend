"""Flask Blueprint for Interview UI — all REST endpoints.

Serves the SPA at /interview and exposes /interview/api/* for backend ops.
Long-running operations return 202 with a job_id for polling.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from flask import Blueprint, jsonify, request, send_from_directory, abort

from .state import (
    CropData, CropLabel, CropSource, InterviewSession, Phase,
    create_session, get_session, get_or_create_session, list_sessions,
    delete_session, SeedConfig,
)
from .cache_manager import (
    cache_exists, list_project_caches, save_session, load_session,
    delete_cache, cache_accepted_crop_image, delete_cached_accepted_crop_image,
)
from .background import (
    submit_job, get_job_progress, get_job_result, pause_job, resume_job,
    cancel_job,
)
from .frame_cache import read_frame_cached as _read_frame_cached

logger = logging.getLogger(__name__)


# Blueprint with static files served from interview/static/
interview_bp = Blueprint(
    "interview",
    __name__,
    static_folder="static",
    static_url_path="",
    url_prefix="/interview",
)


def _cache_accepted_crop_for_reid(session: InterviewSession, crop: CropData) -> None:
    """Persist a UFM-ready accepted crop JPEG under the session temp cache."""
    if getattr(crop, "is_imported_support", False):
        return

    try:
        frame = _read_frame_cached(session.video_path, crop.frame_idx, cache_key=session.cache_key)
        if frame is None:
            logger.warning("Could not read frame %d for accepted crop cache (%s)", crop.frame_idx, crop.crop_id)
            return

        x1, y1, x2, y2 = [int(round(v)) for v in crop.xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.width, x2)
        y2 = min(frame.height, y2)
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid accepted crop box for cache (%s): %s", crop.crop_id, crop.xyxy)
            return

        cropped = frame.crop((x1, y1, x2, y2))
        cache_accepted_crop_image(session.cache_key, crop.crop_id, cropped)
    except Exception as exc:
        # Temp cache is an optimization only; never fail endpoint logic.
        logger.warning("Failed caching accepted crop %s: %s", crop.crop_id, exc)


def _delete_cached_accepted_crop(session: InterviewSession, crop_id: str) -> None:
    delete_cached_accepted_crop_image(session.cache_key, crop_id)


def _invalidate_reid_cache(session: InterviewSession, reason: str) -> None:
    """Invalidate cached ReID/UFM artifacts when accepted crop set changes."""
    session.reid_clusters = {}
    session.n_identities = 0
    for crop in session.crops.values():
        crop.reid_cluster_id = None
    session.ufm_similarity_matrix = None
    session.ufm_crop_ids = []
    session.ufm_complete = False
    session.ufm_job_id = None
    session.touch()
    logger.info("Invalidated ReID cache for session %s: %s", session.session_id, reason)


@interview_bp.after_request
def _fix_passthrough(response):
    """Convert direct-passthrough file responses to buffered responses.

    The label_studio_ml logging middleware calls response.get_data() on
    every response, which crashes on send_from_directory's streaming
    responses with 'RuntimeError: Attempted implicit sequence conversion
    but the response object is in direct passthrough mode'.

    Also sets no-cache on JS/CSS so browsers always revalidate after deploys.
    """
    if response.direct_passthrough:
        response.direct_passthrough = False

    # Prevent stale JS/CSS after deploys — browser revalidates every request
    ct = response.content_type or ""
    if "javascript" in ct or "text/css" in ct:
        response.headers["Cache-Control"] = "no-cache"

    return response


# ===========================================================================
# SPA entry point
# ===========================================================================

@interview_bp.route("/")
@interview_bp.route("/index.html")
def index():
    return send_from_directory(interview_bp.static_folder, "index.html")


# ===========================================================================
# Session endpoints
# ===========================================================================

@interview_bp.route("/api/session/init", methods=["POST"])
def session_init():
    """Create or find a session. Check cache, fetch video info."""
    data = request.get_json(force=True)
    project_id = data.get("project_id")
    task_id = data.get("task_id")
    annotation_id = data.get("annotation_id")

    if not project_id or not task_id:
        return jsonify({"error": "project_id and task_id are required"}), 400

    project_id = int(project_id)
    task_id = int(task_id)
    annotation_id = int(annotation_id) if annotation_id else None

    cache_key = f"p{project_id}_t{task_id}"

    # Check for existing caches
    has_cache = cache_exists(cache_key)
    project_caches = list_project_caches(project_id)
    other_caches = [c for c in project_caches if c.get("cache_key") != cache_key]

    options = []
    if has_cache:
        options.extend(["resume", "build_on", "fresh"])
    elif other_caches:
        for oc in other_caches:
            options.append(f"use_from_{oc['task_id']}")
        options.append("fresh")
    else:
        options.append("fresh")

    # Check for disk frame cache
    frame_cache_info = {"has_frame_cache": False}
    try:
        from .disk_frame_cache import frame_cache_exists, get_frame_cache_size, get_frame_cache_meta
        if frame_cache_exists(cache_key):
            size = get_frame_cache_size(cache_key)
            meta = get_frame_cache_meta(cache_key)
            n_frames = len(meta.get("sampled_indices", [])) if meta else 0
            frame_cache_info = {
                "has_frame_cache": True,
                "frame_cache_size": _human_size(size),
                "frame_cache_bytes": size,
                "frame_cache_frames": n_frames,
            }
    except ImportError:
        pass

    return jsonify({
        "cache_key": cache_key,
        "has_cache": has_cache,
        "other_caches": other_caches,
        "options": options,
        **frame_cache_info,
    })


def _recover_embedding_if_needed(session: InterviewSession) -> None:
    """Re-run lightweight change detection if embedding was interrupted.

    After a container restart, sessions loaded from disk may have
    embedding_complete=False and an empty change_keyframes list.  In
    lightweight mode this takes only seconds, so we run it inline.

    If the video file is missing (e.g. LS cache wiped on restart),
    we skip gracefully — embedding will run when the video is
    re-fetched on the next ``/detect/start``.
    """
    if session.embedding_complete:
        return
    if not session.video_path:
        return
    if not os.path.isfile(session.video_path):
        logger.warning(
            "Skipping embedding recovery: video file missing (%s). "
            "Will re-run after video is re-fetched.",
            session.video_path,
        )
        # Clear stale job reference so polling doesn't 404
        session.embedding_job_id = None
        return

    from .detection import (
        EMBEDDING_MODE, EMBEDDING_TARGET_FPS,
        DEFAULT_KEYFRAME_FRAC, DEFAULT_MIN_SPACING,
    )

    if EMBEDDING_MODE != "lightweight":
        # SAM3 mode is too slow for inline recovery — leave it for the
        # background job to handle on next /detect/start.
        logger.warning("Skipping embedding recovery: SAM3 mode requires background job")
        return

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from seeding_common import (
        compute_lightweight_change_from_video,
        smooth_change_scores, select_keyframes,
    )

    logger.info("Recovering incomplete embedding for session %s (lightweight)...",
                session.session_id)

    scores, sampled_indices = compute_lightweight_change_from_video(
        session.video_path, target_fps=EMBEDDING_TARGET_FPS,
        cache_key=session.cache_key,
    )
    # Backfill video_key and video_path into disk frame cache if available
    if session.cache_key and session.video_key:
        try:
            from .disk_frame_cache import update_frame_cache_meta
            updates = {"video_key": session.video_key}
            force = set()
            if session.video_path:
                updates["video_path"] = session.video_path
                force.add("video_path")
            update_frame_cache_meta(session.cache_key, updates,
                                    force_keys=force)
        except ImportError:
            pass
    smooth = smooth_change_scores(scores, kernel_size=5)
    change_keyframes_sub = select_keyframes(
        len(scores), DEFAULT_KEYFRAME_FRAC, smooth,
        min_spacing=DEFAULT_MIN_SPACING,
    )
    change_keyframes = [
        sampled_indices[k] for k in change_keyframes_sub
        if k < len(sampled_indices)
    ]

    session.embedding_complete = True
    session.change_keyframes = change_keyframes
    session.embedding_sampled_indices = sampled_indices
    session.embedding_job_id = None  # Clear stale job reference
    session.touch()

    logger.info("Embedding recovery complete: %d keyframes", len(change_keyframes))


@interview_bp.route("/api/session/resume", methods=["POST"])
def session_resume():
    """Resume, Build On, or start Fresh from cache."""
    data = request.get_json(force=True)
    project_id = int(data["project_id"])
    task_id = int(data["task_id"])
    annotation_id = data.get("annotation_id")
    annotation_id = int(annotation_id) if annotation_id else None
    mode = data.get("mode", "fresh")  # resume, build_on, fresh, use_from_<task_id>

    cache_key = f"p{project_id}_t{task_id}"

    if mode == "resume":
        session = load_session(cache_key)
        if session is None:
            return jsonify({"error": "No cache found to resume"}), 404
        # Recover incomplete embedding (fast in lightweight mode)
        _recover_embedding_if_needed(session)
        # Register in memory
        from .state import _sessions, _registry_lock
        with _registry_lock:
            _sessions[session.session_id] = session
        # Flag if video file was lost (e.g. container rebuild)
        needs_video = bool(
            session.video_path and not os.path.isfile(session.video_path)
        )
        return jsonify({
            "session_id": session.session_id,
            "needs_video_info": needs_video,
            **session.stats(),
        })

    elif mode == "build_on":
        session = load_session(cache_key)
        if session is None:
            return jsonify({"error": "No cache found to build on"}), 404
        _recover_embedding_if_needed(session)
        session.phase = Phase.DETECTION
        from .state import _sessions, _registry_lock
        with _registry_lock:
            _sessions[session.session_id] = session
        needs_video = bool(
            session.video_path and not os.path.isfile(session.video_path)
        )
        return jsonify({
            "session_id": session.session_id,
            "needs_video_info": needs_video,
            **session.stats(),
        })

    elif mode.startswith("use_from_"):
        source_task_id = int(mode.split("_")[-1])
        source_key = f"p{project_id}_t{source_task_id}"
        source = load_session(source_key)
        if source is None:
            return jsonify({"error": f"No cache found for task {source_task_id}"}), 404

        # Create new session importing prompts from source.
        # Also copy labeled crops — their DINOv3 features form the k-NN
        # support set and are video-agnostic (semantic embeddings).
        # Frame indices/coords from the source video are irrelevant to k-NN
        # scoring — only features + labels + reject_reason matter.
        session = create_session(project_id, task_id, annotation_id)
        session.prompts = list(source.prompts)
        import copy
        transferable = {
            cid: crop for cid, crop in source.crops.items()
            if crop.label in (CropLabel.ACCEPTED, CropLabel.REJECTED) and crop.features is not None
        }
        id_map: Dict[str, str] = {}
        for cid in transferable.keys():
            new_cid = cid
            if new_cid in session.crops:
                i = 1
                while f"imp_{source_task_id}_{cid}_{i}" in session.crops:
                    i += 1
                new_cid = f"imp_{source_task_id}_{cid}_{i}"
            id_map[cid] = new_cid

        for cid, crop in transferable.items():
            transferred = copy.deepcopy(crop)
            transferred.crop_id = id_map[cid]
            if transferred.corrected_from in id_map:
                transferred.corrected_from = id_map[transferred.corrected_from]
            transferred.is_imported_support = True
            transferred.source_project_id = source.project_id
            transferred.source_task_id = source.task_id
            transferred.source_session_id = source.session_id
            transferred.source_crop_id = cid
            transferred.source_frame_idx = crop.frame_idx
            transferred.source_video_key = source.video_key
            session.add_crop(transferred)
        # Build-from starts a fresh detection round timeline while preserving
        # transferred support knowledge.
        session.current_round = 0
        session.round_history = []
        session.round_frames = {}
        session.phase = Phase.DETECTION

        return jsonify({"session_id": session.session_id, **session.stats()})

    else:
        # Fresh start — optionally preserve disk frame cache
        keep_frames = bool(data.get("keep_frame_cache", False))
        delete_cache(cache_key, project_id, keep_frame_cache=keep_frames)
        session = create_session(project_id, task_id, annotation_id)
        return jsonify({"session_id": session.session_id, **session.stats()})


@interview_bp.route("/api/session/<session_id>/status", methods=["GET"])
def session_status(session_id: str):
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session.stats())


def _origin_tuple(url: str) -> Optional[tuple]:
    """Return normalized (scheme, host, port) for absolute HTTP(S) URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return (parsed.scheme, host, port)


def _same_origin(url_a: str, url_b: str) -> bool:
    """True if both absolute URLs share scheme/host/port origin."""
    oa = _origin_tuple(url_a)
    ob = _origin_tuple(url_b)
    return oa is not None and ob is not None and oa == ob


def _download_video_with_progress(video_url: str, task_id: int,
                                  progress, *, ls_base_url: str = "",
                                  ls_api_key: str = "",
                                  cache_root: str = "/data/video_cache") -> str:
    """Download video with byte-level progress, using a deterministic cache.

    Cache path: ``/data/video_cache/{task_id}/{url_hash}.mp4``
    Returns the local path to the downloaded (or cached) video file.
    """
    import hashlib
    import requests

    url_hash = hashlib.md5(video_url.encode()).hexdigest()[:12]
    cache_dir = os.path.join(cache_root, str(task_id))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{url_hash}.mp4")

    # Cache hit
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        logger.info("Video cache hit: %s (%.1f MB)",
                     cache_path, os.path.getsize(cache_path) / 1024**2)
        return cache_path

    # Cache miss — streaming download with progress
    headers = {}
    if ls_api_key and _same_origin(video_url, ls_base_url):
        headers["Authorization"] = f"Token {ls_api_key}"

    logger.info("Downloading video from %s to %s", video_url, cache_path)
    tmp_path = cache_path + ".tmp"
    try:
        with requests.get(video_url, headers=headers, stream=True,
                          timeout=600) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("Content-Length", 0))
            if total_bytes > 0:
                progress.total = total_bytes
                size_gb = total_bytes / (1024**3)
                progress.step = f"Downloading video ({size_gb:.1f} GB)..."
            else:
                progress.step = "Downloading video..."
                progress.total = 0

            written = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    written += len(chunk)
                    progress.current = written

        os.rename(tmp_path, cache_path)
        logger.info("Video download complete: %s (%.1f MB)",
                     cache_path, written / 1024**2)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return cache_path


@interview_bp.route("/api/session/<session_id>/video_info", methods=["POST"])
def session_video_info(session_id: str):
    """Fetch video info for the session's task. Must be called after init."""
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _fetch(progress):
        progress.step = "Checking frame cache..."
        progress.total = 3

        # ---- Fast path: disk frame cache has all metadata we need ----
        try:
            from .disk_frame_cache import (
                frame_cache_exists, get_frame_cache_meta,
            )
            if frame_cache_exists(session.cache_key):
                meta = get_frame_cache_meta(session.cache_key)
                if meta and "resolution" in meta and "src_fps" in meta:
                    res = meta["resolution"]  # [W, H]
                    width, height = res[0], res[1]
                    fps = meta["src_fps"]

                    # total_frames: new caches have it; old ones estimate
                    total_frames = meta.get("total_frames")
                    if not total_frames:
                        indices = meta.get("sampled_indices", [])
                        target_fps = meta.get("target_fps", 10.0)
                        skip = max(1, int(round(fps / target_fps)))
                        total_frames = (max(indices) + skip) if indices else 0

                    video_path = meta.get("video_path", "")
                    video_key = meta.get("video_key", "video")
                    if not video_path or not os.path.isfile(video_path):
                        logger.info(
                            "Frame-cache metadata present but video_path is "
                            "missing/unreadable; falling back to task fetch "
                            "and download (cache_key=%s)",
                            session.cache_key,
                        )
                    else:
                        progress.step = "Using cached video metadata"
                        progress.current = 3

                        with session._lock:
                            session.video_path = video_path
                            session.video_key = video_key
                            session.width = width
                            session.height = height
                            session.frames_count = total_frames
                            session.fps = fps
                            session.touch()

                        logger.info(
                            "Fast path: video info from disk frame cache "
                            "(%dx%d, %d frames, %.1f fps)",
                            width, height, total_frames, fps,
                        )

                        return _cross_video_gate(
                            session, video_path, video_key,
                            width, height, total_frames, fps,
                        )
        except ImportError:
            pass

        # ---- Slow path: fetch task from LS + download video ----
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from seeding_common import (
            _build_ls_client, _fetch_task, _detect_video_key,
            _get_video_info_pyav,
        )

        progress.step = "Connecting to Label Studio..."
        progress.current = 1
        ls_url = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL", "")
        ls_api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
        ls = _build_ls_client(ls_url, ls_api_key)

        progress.step = "Fetching task data..."
        progress.current = 2
        task = _fetch_task(ls, session.project_id, session.task_id)
        data = task.get("data") or {}
        video_key, video_url = _detect_video_key(data)

        # Resolve relative URL to absolute
        if not video_url.startswith("http") and video_url.startswith("/"):
            from urllib.parse import urljoin
            video_url = urljoin(ls_url.rstrip("/"), video_url)

        # Download with byte-level progress
        video_path = _download_video_with_progress(
            video_url, session.task_id, progress,
            ls_base_url=ls_url,
            ls_api_key=ls_api_key,
        )

        progress.step = "Reading video metadata..."
        progress.current = 3
        width, height, frames_count, fps = _get_video_info_pyav(video_path)

        with session._lock:
            session.video_path = video_path
            session.video_key = video_key
            session.width = width
            session.height = height
            session.frames_count = frames_count
            session.fps = fps
            session.touch()

        # Backfill disk frame cache meta.json so future starts use fast path.
        # Force-overwrite video_path: if an old/stale path was already stored,
        # the add-only default would skip it, leaving the fast path broken.
        try:
            from .disk_frame_cache import update_frame_cache_meta
            if update_frame_cache_meta(session.cache_key, {
                "video_path": video_path,
                "video_key": video_key,
                "total_frames": frames_count,
            }, force_keys={"video_path"}):
                logger.info("Backfilled meta.json for %s", session.cache_key)
        except Exception as exc:
            logger.debug("Could not backfill meta.json: %s", exc)

        return _cross_video_gate(
            session, video_path, video_key,
            width, height, frames_count, fps,
        )

    job_id = submit_job(_fetch, name="fetch_video_info")
    return jsonify({"job_id": job_id}), 202


def _cross_video_gate(session, video_path, video_key,
                      width, height, frames_count, fps):
    """Strip imported supports from a different video and build result dict."""
    cross_video_ids = [
        cid for cid, crop in session.crops.items()
        if crop.is_imported_support
        and crop.source_video_key
        and crop.source_video_key != video_key
    ]
    warning = None
    if cross_video_ids:
        logger.warning(
            "Cross-video import: %d imported supports from a different "
            "video removed (source_video_key != %s). Cross-task import "
            "with different videos is not yet supported.",
            len(cross_video_ids), video_key,
        )
        with session._lock:
            for cid in cross_video_ids:
                session.remove_crop(cid)
        warning = (
            f"Removed {len(cross_video_ids)} imported supports — they "
            f"came from a different video. Cross-video transfer is not "
            f"yet supported."
        )

    result = {
        "video_path": video_path,
        "width": width,
        "height": height,
        "frames_count": frames_count,
        "fps": fps,
    }
    if warning:
        result["warning"] = warning
    return result


# ===========================================================================
# Detection endpoints (Phase 1)
# ===========================================================================

@interview_bp.route("/api/detect/start", methods=["POST"])
def detect_start():
    """Start Round 1: detection on stratified frames + background embedding.

    Round 1 (fast): select ~40 temporally-stratified frames, batch-detect.
    Background: GPU-batch embed all frames, run change detection.
    Returns both job IDs so the frontend can poll each independently.
    """
    data = request.get_json(force=True)
    session_id = data["session_id"]
    prompt = data.get("prompt", "person")

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    # Round 1: detection on stratified frames
    def _detect_round1(progress):
        from .detection import run_round_detection
        return run_round_detection(session, prompt, progress, round_num=1)

    detect_job_id = submit_job(_detect_round1, name="round_1_detection")

    # Background: embed all frames + change detection (concurrent)
    def _embed_bg(progress):
        from .detection import run_embedding_background
        return run_embedding_background(session, progress)

    embed_job_id = submit_job(_embed_bg, name="embedding_background")

    # Store embedding job ID on session for status polling
    with session._lock:
        session.embedding_job_id = embed_job_id
        session.touch()

    return jsonify({
        "job_id": detect_job_id,
        "embedding_job_id": embed_job_id,
        "round": 1,
    }), 202


@interview_bp.route("/api/detect/embedding_status", methods=["GET"])
def detect_embedding_status():
    """Poll background embedding progress."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    result = {
        "embedding_complete": session.embedding_complete,
        "change_keyframes_count": len(session.change_keyframes),
    }

    # Include job progress if available
    if session.embedding_job_id:
        progress = get_job_progress(session.embedding_job_id)
        if progress:
            result["progress"] = {
                "current": progress.get("current", 0),
                "total": progress.get("total", 0),
                "percent": progress.get("percent", 0),
                "step": progress.get("step", ""),
                "status": progress.get("status", "unknown"),
            }

    return jsonify(result)


@interview_bp.route("/api/detect/next_round", methods=["POST"])
def detect_next_round():
    """Score pending crops with k-NN, then start next round of detection.

    Two-phase job:
      1. Re-score all pending crops using k-NN (no training step)
      2. Select new frames, detect on them
    """
    data = request.get_json(force=True)
    session_id = data["session_id"]

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    next_round = session.current_round + 1
    prompt = session.prompts[0] if session.prompts else "person"

    def _next_round(progress):
        from .knn_classifier import compute_uncertainties
        from .dinov3_classifier import _ensure_crop_features
        from .detection import run_round_detection

        # Pause background embedding to free GPU for detection
        embedding_paused = False
        if session.embedding_job_id:
            embedding_paused = pause_job(session.embedding_job_id)

        try:
            # Ensure all crops have DINOv3 + context features
            progress.step = f"Extracting features before round {next_round}..."
            _ensure_crop_features(session, list(session.crops.keys()), progress)

            # Re-score pending crops with k-NN (instant, no training)
            progress.step = f"Scoring crops with k-NN before round {next_round}..."
            n_scored = compute_uncertainties(session)
            save_session(session)

            progress.step = f"Starting round {next_round} detection..."
            detect_result = run_round_detection(
                session, prompt, progress, round_num=next_round,
            )
        finally:
            # Resume embedding after detection completes
            if embedding_paused and session.embedding_job_id:
                resume_job(session.embedding_job_id)

        return {
            "scoring": {"n_scored": n_scored},
            "detection": detect_result,
            "round": next_round,
        }

    job_id = submit_job(_next_round, name=f"round_{next_round}")
    return jsonify({"job_id": job_id, "round": next_round}), 202


@interview_bp.route("/api/detect/crops", methods=["GET"])
def detect_crops():
    """List crops with filtering and sorting."""
    session_id = request.args.get("session_id")
    filter_label = request.args.get("filter", "all")
    sort_by = request.args.get("sort", "uncertainty")  # uncertainty, cluster, frame
    include_imported = str(request.args.get("include_imported", "0")).lower() in {"1", "true", "yes", "on"}
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 50))

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    crops = list(session.crops.values())
    if not include_imported:
        crops = [c for c in crops if not getattr(c, "is_imported_support", False)]

    # Filter
    if filter_label != "all":
        if filter_label == "corrected":
            crops = [c for c in crops if c.source == CropSource.BOX_CORRECTED]
        else:
            try:
                label = CropLabel(filter_label)
                crops = [c for c in crops if c.label == label]
            except ValueError:
                pass

    # Sort
    if sort_by == "uncertainty":
        crops.sort(key=lambda c: -c.uncertainty)
    elif sort_by == "cluster":
        crops.sort(key=lambda c: (c.cluster_id or 9999, -c.uncertainty))
    elif sort_by == "frame":
        crops.sort(key=lambda c: (c.frame_idx, c.xyxy[0] if c.xyxy is not None else 0))

    total = len(crops)
    crops = crops[offset:offset + limit]

    return jsonify({
        "total": total,
        "offset": offset,
        "limit": limit,
        "include_imported": include_imported,
        "crops": [c.to_dict() for c in crops],
    })


@interview_bp.route("/api/detect/frame/<int:frame_idx>", methods=["GET"])
def detect_frame(frame_idx: int):
    """Serve a frame as JPEG (raw, no annotations)."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        abort(404)

    pil_img = _read_frame_cached(session.video_path, frame_idx, cache_key=session.cache_key)
    if pil_img is None:
        abort(404)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.getvalue(), 200, {
        "Content-Type": "image/jpeg",
        "Cache-Control": "public, max-age=86400",
    }


@interview_bp.route("/api/detect/frame/<int:frame_idx>/annotated", methods=["GET"])
def detect_frame_annotated(frame_idx: int):
    """Frame with color-coded boxes drawn.

    Optional query param ``highlight`` — crop_id to draw with a thick border.
    """
    session_id = request.args.get("session_id")
    highlight_id = request.args.get("highlight")
    session = get_session(session_id)
    if session is None:
        abort(404)

    pil_img = _read_frame_cached(session.video_path, frame_idx, cache_key=session.cache_key)
    if pil_img is None:
        abort(404)

    # Draw on a copy so the LRU-cached original stays clean
    pil_img = pil_img.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil_img)

    color_map = {
        CropLabel.ACCEPTED: "#00ff00",
        CropLabel.REJECTED: "#ff0000",
        CropLabel.PENDING: "#ffff00",
        CropLabel.SKIPPED: "#888888",
    }
    source_color_override = {
        CropSource.HUMAN_DRAWN: "#ff8800",
    }

    for crop in session.get_crops_by_frame(frame_idx):
        is_highlighted = (highlight_id and crop.crop_id == highlight_id)
        color = source_color_override.get(crop.source, color_map.get(crop.label, "#ffff00"))
        x1, y1, x2, y2 = crop.xyxy
        if is_highlighted:
            # Thick cyan border + semi-transparent fill for highlighted crop
            draw.rectangle([x1, y1, x2, y2], outline="#00ffff", width=4)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    # No long cache — box colors change with labels
    return buf.getvalue(), 200, {"Content-Type": "image/jpeg"}


@interview_bp.route("/api/detect/crop/<crop_id>/image", methods=["GET"])
def detect_crop_image(crop_id: str):
    """Serve a cropped box as JPEG."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        abort(404)

    crop = session.get_crop(crop_id)
    if crop is None:
        abort(404)

    pil_img = _read_frame_cached(session.video_path, crop.frame_idx, cache_key=session.cache_key)
    if pil_img is None:
        abort(404)

    x1, y1, x2, y2 = [int(round(v)) for v in crop.xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(pil_img.width, x2)
    y2 = min(pil_img.height, y2)
    cropped = pil_img.crop((x1, y1, x2, y2))

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue(), 200, {
        "Content-Type": "image/jpeg",
        "Cache-Control": "public, max-age=86400",
    }


@interview_bp.route("/api/detect/label", methods=["POST"])
def detect_label():
    """Batch label crops (accept/reject)."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    labels = data.get("labels", {})  # {crop_id: "accepted" | "rejected"}

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    updated = 0
    reid_changed = False
    for crop_id, label_str in labels.items():
        try:
            label = CropLabel(label_str)
            existing = session.get_crop(crop_id)
            old_label = existing.label if existing is not None else None
            if session.label_crop(crop_id, label):
                updated += 1
                crop = session.get_crop(crop_id)
                if crop is not None:
                    if label == CropLabel.ACCEPTED:
                        _cache_accepted_crop_for_reid(session, crop)
                    else:
                        _delete_cached_accepted_crop(session, crop_id)
                if old_label is not None and old_label != label:
                    if old_label == CropLabel.ACCEPTED or label == CropLabel.ACCEPTED:
                        reid_changed = True
        except ValueError:
            pass

    if reid_changed:
        _invalidate_reid_cache(session, reason="accepted label set changed")

    save_session(session)
    return jsonify({"updated": updated, **session.stats()})


@interview_bp.route("/api/detect/draw", methods=["POST"])
def detect_draw():
    """Add a human-drawn box (Draw Mode) — auto-accepted."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    frame_idx = int(data["frame_idx"])
    xyxy = data["xyxy"]  # [x1, y1, x2, y2] in pixel coords

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    import numpy as np
    import uuid

    crop = CropData(
        crop_id=str(uuid.uuid4())[:12],
        frame_idx=frame_idx,
        xyxy=np.array(xyxy, dtype=np.float32),
        score=1.0,
        label=CropLabel.ACCEPTED,
        source=CropSource.HUMAN_DRAWN,
        prompt="human_drawn",
    )
    session.add_crop(crop)
    _cache_accepted_crop_for_reid(session, crop)
    _invalidate_reid_cache(session, reason="manual accepted crop added")
    save_session(session)

    return jsonify({"crop": crop.to_dict(), **session.stats()})


@interview_bp.route("/api/detect/subcategorize", methods=["POST"])
def detect_subcategorize():
    """Tag a rejected crop with a subcategory and optionally create a corrected crop.

    Body: {session_id, crop_id, reject_reason, adjusted_xyxy (nullable)}
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    crop_id = data.get("crop_id")
    reject_reason = data.get("reject_reason")
    adjusted_xyxy = data.get("adjusted_xyxy")  # [x1, y1, x2, y2] or null

    if not session_id or not crop_id or not reject_reason:
        return jsonify({"error": "session_id, crop_id, and reject_reason are required"}), 400

    valid_reasons = {"not_person", "partial_box", "oversized_box"}
    if reject_reason not in valid_reasons:
        return jsonify({"error": f"Invalid reject_reason. Must be one of: {sorted(valid_reasons)}"}), 400

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    crop = session.get_crop(crop_id)
    if crop is None:
        return jsonify({"error": "Crop not found"}), 404

    if crop.label != CropLabel.REJECTED:
        return jsonify({"error": f"Crop {crop_id} is not rejected (label={crop.label.value})"}), 400

    # Contradictory: "not a person" should never produce a corrected box.
    if reject_reason == "not_person" and adjusted_xyxy is not None:
        return jsonify({"error": "Cannot provide adjusted_xyxy with not_person reason"}), 400

    # Set reject_reason on the original crop
    crop.reject_reason = reject_reason

    new_crop_id = None

    existing_corrected = [
        cid for cid, c in session.crops.items()
        if getattr(c, "corrected_from", None) == crop_id
    ]
    reid_changed = False

    # Saving "not_person" removes any corrected counterpart.
    if reject_reason == "not_person":
        for cid in existing_corrected:
            _delete_cached_accepted_crop(session, cid)
            session.remove_crop(cid)
            reid_changed = True

    # If an adjusted box was provided, create a corrected crop.
    # Features for the corrected crop are extracted lazily during
    # the next round's _ensure_crop_features() call, not here.
    if adjusted_xyxy is not None:
        if not isinstance(adjusted_xyxy, (list, tuple)) or len(adjusted_xyxy) != 4:
            return jsonify({"error": "adjusted_xyxy must be a 4-element array [x1, y1, x2, y2]"}), 400
        try:
            coords = [float(v) for v in adjusted_xyxy]
        except (TypeError, ValueError):
            return jsonify({"error": "adjusted_xyxy values must be numeric"}), 400
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            return jsonify({"error": "Invalid box: x1 must be < x2 and y1 must be < y2"}), 400

        import numpy as np
        import uuid

        # Remove any previously-created corrected crop for this original
        # (idempotency: re-subcategorize replaces, not duplicates)
        for cid in existing_corrected:
            _delete_cached_accepted_crop(session, cid)
            session.remove_crop(cid)
            reid_changed = True

        new_crop = CropData(
            crop_id=str(uuid.uuid4())[:12],
            frame_idx=crop.frame_idx,
            xyxy=np.array(coords, dtype=np.float32),
            score=1.0,
            label=CropLabel.ACCEPTED,
            source=CropSource.BOX_CORRECTED,
            prompt="box_corrected",
            corrected_from=crop_id,
        )
        session.add_crop(new_crop)
        _cache_accepted_crop_for_reid(session, new_crop)
        new_crop_id = new_crop.crop_id
        reid_changed = True

    if reid_changed:
        _invalidate_reid_cache(session, reason="corrected accepted crops changed")

    save_session(session)

    return jsonify({
        "crop_id": crop_id,
        "reject_reason": reject_reason,
        "new_crop_id": new_crop_id,
        **session.stats(),
    })


@interview_bp.route("/api/detect/refine_box", methods=["POST"])
def detect_refine_box():
    """Refine an oversized/partial box using Sam3TrackerModel (single-image PVS).

    Body: {session_id, crop_id, prompt_xyxy?}
    Returns: {refined_xyxy: [x1, y1, x2, y2], confidence: float}

    Uses Sam3TrackerModel with box prompt — segments the object *within* the
    crop's bounding box and returns a tighter box derived from the mask.
    This is a preview-only endpoint; persistence happens via /detect/subcategorize.
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    crop_id = data.get("crop_id")
    prompt_xyxy = data.get("prompt_xyxy")

    if not session_id or not crop_id:
        return jsonify({"error": "session_id and crop_id are required"}), 400

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    crop = session.get_crop(crop_id)
    if crop is None:
        return jsonify({"error": "Crop not found"}), 404

    # Read the frame
    from .detection import _read_frame_cached_or_pyav
    frame = _read_frame_cached_or_pyav(
        session.video_path, crop.frame_idx, cache_key=session_id,
    )
    if frame is None:
        return jsonify({"error": f"Could not read frame {crop.frame_idx}"}), 500

    # Run Sam3TrackerModel with box prompt
    import numpy as np
    import torch
    from seeding_common import _get_sam3_tracker_image_model, DEVICE, DTYPE

    model, processor = _get_sam3_tracker_image_model()
    if prompt_xyxy is not None:
        if not isinstance(prompt_xyxy, (list, tuple)) or len(prompt_xyxy) != 4:
            return jsonify({"error": "prompt_xyxy must be a 4-element array [x1, y1, x2, y2]"}), 400
        try:
            box_xyxy = [float(v) for v in prompt_xyxy]
        except (TypeError, ValueError):
            return jsonify({"error": "prompt_xyxy values must be numeric"}), 400
        if box_xyxy[0] >= box_xyxy[2] or box_xyxy[1] >= box_xyxy[3]:
            return jsonify({"error": "Invalid prompt_xyxy: x1 must be < x2 and y1 must be < y2"}), 400
    else:
        box_xyxy = [float(v) for v in crop.xyxy]

    try:
        # Clamp prompt box to frame bounds.
        w, h = frame.size
        box_xyxy = [
            max(0.0, min(float(w), box_xyxy[0])),
            max(0.0, min(float(h), box_xyxy[1])),
            max(0.0, min(float(w), box_xyxy[2])),
            max(0.0, min(float(h), box_xyxy[3])),
        ]
        if box_xyxy[0] >= box_xyxy[2] or box_xyxy[1] >= box_xyxy[3]:
            return jsonify({"error": "prompt_xyxy is outside frame bounds"}), 400

        # Sam3TrackerProcessor: input_boxes is 3D [[box_xyxy]]
        inputs = processor(
            images=frame,
            input_boxes=[[box_xyxy]],
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode():
            if DEVICE != "cpu":
                with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                    outputs = model(**inputs, multimask_output=False)
            else:
                outputs = model(**inputs, multimask_output=False)

        # Post-process mask to original size
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"],
        )[0]  # (objects=1, num_masks=1, H, W)

        iou_score = float(outputs.iou_scores[0, 0, 0].item())

        # Extract tight bounding box from mask
        best_mask = masks[0, 0]  # (H, W)
        if hasattr(best_mask, "cpu"):
            best_mask = best_mask.cpu().float().numpy()

        mask_bool = best_mask.astype(bool)
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)

        if not rows.any() or not cols.any():
            # Empty mask — return original box
            return jsonify({
                "refined_xyxy": box_xyxy,
                "confidence": 0.0,
            })

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        refined_xyxy = [
            max(0.0, float(x_indices[0])),
            max(0.0, float(y_indices[0])),
            min(float(w), float(x_indices[-1] + 1)),
            min(float(h), float(y_indices[-1] + 1)),
        ]

        return jsonify({
            "refined_xyxy": refined_xyxy,
            "confidence": iou_score,
        })

    except Exception as exc:
        logger.exception("Sam3TrackerModel refine_box failed")
        return jsonify({"error": str(exc)}), 500


@interview_bp.route("/api/detect/train", methods=["POST"])
def detect_train():
    """Re-score pending crops with k-NN (replaces MLP training)."""
    data = request.get_json(force=True)
    session_id = data["session_id"]

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _score(progress):
        from .knn_classifier import compute_uncertainties
        from .dinov3_classifier import _ensure_crop_features

        progress.step = "Extracting features..."
        _ensure_crop_features(session, list(session.crops.keys()), progress)

        progress.step = "Scoring with k-NN..."
        n_scored = compute_uncertainties(session)
        save_session(session)

        return {"n_scored": n_scored, "method": "knn"}

    job_id = submit_job(_score, name="knn_scoring")
    return jsonify({"job_id": job_id}), 202


# ===========================================================================
# ReID endpoints (Phase 2)
# ===========================================================================

@interview_bp.route("/api/reid/start", methods=["POST"])
def reid_start():
    """Start UFM-based ReID clustering.

    Body: {session_id, n_clusters} — n_clusters is REQUIRED (user specifies k).
    Runs UFM pairwise similarity + HAC as a background job.
    """
    data = request.get_json(force=True)
    session_id = data["session_id"]
    n_clusters = data.get("n_clusters")

    if not n_clusters or int(n_clusters) < 2:
        return jsonify({"error": "n_clusters >= 2 is required"}), 400

    n_clusters = int(n_clusters)

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _cluster(progress):
        from .reid_ufm import run_ufm_reid_pipeline
        return run_ufm_reid_pipeline(
            session, n_clusters, progress, _read_frame_cached,
        )

    job_id = submit_job(_cluster, name="reid_ufm_clustering")
    session.ufm_job_id = job_id
    return jsonify({"job_id": job_id}), 202


@interview_bp.route("/api/reid/recluster", methods=["POST"])
def reid_recluster():
    """Re-run HAC clustering with a different k.

    If UFM similarity matrix is already computed, re-uses it (instant).
    Otherwise, re-computes from scratch.

    Body: {session_id, n_clusters}
    """
    data = request.get_json(force=True)
    session_id = data["session_id"]
    n_clusters = data.get("n_clusters")

    if not n_clusters or int(n_clusters) < 2:
        return jsonify({"error": "n_clusters >= 2 is required"}), 400

    n_clusters = int(n_clusters)

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    # Clear existing cluster assignments
    session.reid_clusters = {}
    session.n_identities = 0
    for crop in session.crops.values():
        crop.reid_cluster_id = None

    # If we already have a UFM similarity matrix, just re-cluster (instant)
    if session.ufm_similarity_matrix is not None and session.ufm_complete:
        from .reid_ufm import cluster_hac, silhouette_score, compute_co_occurrence_warnings

        sim = session.ufm_similarity_matrix
        crop_ids = session.ufm_crop_ids
        n = len(crop_ids)
        k = max(2, min(n_clusters, n - 1))
        labels = cluster_hac(sim, k)
        sil = silhouette_score(sim, labels)

        clusters = {}
        for idx, cid in enumerate(crop_ids):
            cluster_id = int(labels[idx])
            clusters.setdefault(cluster_id, []).append(cid)
            crop = session.get_crop(cid)
            if crop is not None:
                crop.reid_cluster_id = cluster_id

        session.reid_clusters = clusters
        session.n_identities = len(clusters)
        session.touch()
        save_session(session)

        warnings = compute_co_occurrence_warnings(clusters, session.crops)
        return jsonify({
            "n_clusters": len(clusters),
            "n_crops": n,
            "cluster_sizes": {cid: len(m) for cid, m in clusters.items()},
            "silhouette": round(sil, 4),
            "co_occurrence_warnings": warnings,
        })

    # No matrix yet — run full pipeline as background job
    def _recluster(progress):
        from .reid_ufm import run_ufm_reid_pipeline
        return run_ufm_reid_pipeline(
            session, n_clusters, progress, _read_frame_cached,
        )

    job_id = submit_job(_recluster, name="reid_recluster")
    session.ufm_job_id = job_id
    return jsonify({"job_id": job_id}), 202


@interview_bp.route("/api/reid/clusters", methods=["GET"])
def reid_clusters():
    """Cluster list with co-occurrence warnings and UFM status."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    clusters_info = {}
    for cid, crop_ids in session.reid_clusters.items():
        clusters_info[str(cid)] = {
            "crop_ids": crop_ids,
            "count": len(crop_ids),
        }

    # Co-occurrence warnings
    from .reid_ufm import compute_co_occurrence_warnings
    warnings = compute_co_occurrence_warnings(session.reid_clusters, session.crops)

    return jsonify({
        "clusters": clusters_info,
        "n_identities": session.n_identities,
        "cluster_sizes": {str(cid): len(m) for cid, m in session.reid_clusters.items()},
        "ufm_complete": session.ufm_complete,
        "co_occurrence_warnings": warnings,
    })


@interview_bp.route("/api/reid/assign", methods=["POST"])
def reid_assign():
    """Move selected crops into an existing cluster.

    Body: {session_id, crop_ids: [str], target_cluster_id: int}
    """
    data = request.get_json(force=True)
    session_id = data["session_id"]
    crop_ids = data.get("crop_ids", [])
    target_cluster_id = data.get("target_cluster_id")

    if not crop_ids:
        return jsonify({"error": "crop_ids is required"}), 400
    if target_cluster_id is None:
        return jsonify({"error": "target_cluster_id is required"}), 400

    target_cluster_id = int(target_cluster_id)

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    from .reid_ufm import assign_crops_to_cluster
    result = assign_crops_to_cluster(session, crop_ids, target_cluster_id)
    save_session(session)
    return jsonify(result)


@interview_bp.route("/api/reid/new_cluster", methods=["POST"])
def reid_new_cluster():
    """Create a new cluster from selected crops.

    Body: {session_id, crop_ids: [str]}
    """
    data = request.get_json(force=True)
    session_id = data["session_id"]
    crop_ids = data.get("crop_ids", [])

    if not crop_ids:
        return jsonify({"error": "crop_ids is required"}), 400

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    from .reid_ufm import create_new_cluster
    result = create_new_cluster(session, crop_ids)
    save_session(session)
    return jsonify(result)


# ===========================================================================
# Seeding + Upload endpoints (Phase 3)
# ===========================================================================

@interview_bp.route("/api/seeds/generate", methods=["POST"])
def seeds_generate():
    """Start seed generation job."""
    data = request.get_json(force=True)
    session_id = data["session_id"]

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _generate(progress):
        from .seeding_phase import generate_seeds
        return generate_seeds(session, progress)

    job_id = submit_job(_generate, name="seed_generation")
    return jsonify({"job_id": job_id}), 202


@interview_bp.route("/api/seeds/list", methods=["GET"])
def seeds_list():
    """Seed list with identity assignments."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    from .seeding_phase import filter_seeds_by_threshold, summarise_seeds_by_identity

    threshold = max(0.0, min(1.0, float(session.seed_config.confidence_threshold)))
    filtered = filter_seeds_by_threshold(session.seeds, threshold)
    identity_summary = summarise_seeds_by_identity(filtered)
    generated_identity_summary = summarise_seeds_by_identity(session.seeds)

    return jsonify({
        # Backward-compatible: total_seeds remains the currently visible
        # (threshold-filtered) count shown in preview/upload flows.
        "total_seeds": len(filtered),
        "total_generated_seeds": len(session.seeds),
        "identities": identity_summary,
        "generated_identities": generated_identity_summary,
        "target_frames": len(session.seed_target_frames),
        "cached_frames_selected": len(session.seed_cached_frames),
        "seed_config": {
            "frame_pct": session.seed_config.frame_pct,
            "confidence_threshold": session.seed_config.confidence_threshold,
        },
    })


@interview_bp.route("/api/frame_cache", methods=["DELETE"])
def delete_frame_cache_endpoint():
    """Delete just the disk frame cache for a specific task."""
    data = request.get_json(force=True)
    cache_key = data.get("cache_key")
    if not cache_key:
        return jsonify({"error": "cache_key is required"}), 400

    try:
        from .disk_frame_cache import delete_frame_cache
        deleted = delete_frame_cache(cache_key)
    except ImportError:
        deleted = False

    return jsonify({"deleted": deleted, "cache_key": cache_key})


@interview_bp.route("/api/seeds/config", methods=["GET"])
def seeds_config_get():
    """Current seed generation config."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    # Determine cached frame count from disk cache
    cached_frame_count = 0
    try:
        from .disk_frame_cache import frame_cache_exists as _fce, get_frame_cache_meta as _gfcm
        if _fce(session.cache_key):
            meta = _gfcm(session.cache_key)
            if meta and "sampled_indices" in meta:
                cached_frame_count = len(meta["sampled_indices"])
    except ImportError:
        pass

    return jsonify({
        "frame_pct": session.seed_config.frame_pct,
        "confidence_threshold": session.seed_config.confidence_threshold,
        "change_keyframes": session.change_keyframes,
        "frames_count": session.frames_count,
        "cached_frame_count": cached_frame_count,
    })


@interview_bp.route("/api/seeds/config", methods=["PUT"])
def seeds_config_put():
    """Update seed generation config."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    if "frame_pct" in data:
        session.seed_config.frame_pct = max(1, min(100, int(data["frame_pct"])))
    if "confidence_threshold" in data:
        session.seed_config.confidence_threshold = max(
            0.0, min(1.0, float(data["confidence_threshold"])),
        )

    session.touch()
    save_session(session)

    return jsonify({
        "frame_pct": session.seed_config.frame_pct,
        "confidence_threshold": session.seed_config.confidence_threshold,
    })


@interview_bp.route("/api/seeds/upload", methods=["POST"])
def seeds_upload():
    """Upload seed regions to LS with enabled=false keyframes."""
    data = request.get_json(force=True)
    session_id = data["session_id"]

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    if "confidence_threshold" in data:
        session.seed_config.confidence_threshold = max(
            0.0, min(1.0, float(data["confidence_threshold"])),
        )
        session.touch()
        save_session(session)

    def _upload(progress):
        from .seeding_phase import upload_seeds
        return upload_seeds(session, progress)

    job_id = submit_job(_upload, name="seed_upload")
    return jsonify({"job_id": job_id}), 202


# ===========================================================================
# Shared endpoints
# ===========================================================================

@interview_bp.route("/api/job/<job_id>/progress", methods=["GET"])
def job_progress(job_id: str):
    """Poll background job status + progress."""
    progress = get_job_progress(job_id)
    if progress is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(progress)


@interview_bp.route("/api/job/<job_id>/cancel", methods=["POST"])
def job_cancel(job_id: str):
    """Request cooperative cancellation for a running background job."""
    cancelled = cancel_job(job_id)
    if cancelled is None:
        return jsonify({"error": "Job not found"}), 404
    if cancelled is False:
        return jsonify({"error": "Job is not running"}), 409
    return jsonify({"job_id": job_id, "cancel_requested": True})


@interview_bp.route("/api/session/<session_id>/save", methods=["POST"])
def session_save(session_id: str):
    """Manually save session to cache."""
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    save_session(session)
    return jsonify({"saved": True})


# ===========================================================================
# Disk frame cache endpoints
# ===========================================================================

def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


@interview_bp.route("/api/disk_usage", methods=["GET"])
def disk_usage():
    """Report disk frame cache usage across all sessions."""
    try:
        from .disk_frame_cache import get_all_frame_cache_sizes
    except ImportError:
        return jsonify({
            "total_bytes": 0,
            "total_human": "0 B",
            "per_session": {},
        })

    sizes = get_all_frame_cache_sizes()
    total = sum(sizes.values())
    return jsonify({
        "total_bytes": total,
        "total_human": _human_size(total),
        "sessions_cached": len(sizes),
        "per_session": {
            k: {"bytes": v, "human": _human_size(v)}
            for k, v in sizes.items()
        },
    })
