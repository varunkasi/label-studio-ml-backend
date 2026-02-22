"""Disk cache for interview sessions.

Persists session state under /data/adapters/{cache_key}/ so that
sessions can survive container restarts and be reused across tasks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .state import (
    CropData, CropLabel, CropSource, InterviewSession, Phase,
    SeedConfig,
)

logger = logging.getLogger(__name__)

CACHE_ROOT = os.getenv("INTERVIEW_CACHE_ROOT", "/data/adapters")
PROJECT_INDEX_FILE = "_project_index.json"
ACCEPTED_CROPS_DIRNAME = "accepted_crops"
ACCEPTED_CROP_SIZE: Tuple[int, int] = (224, 224)

_index_lock = threading.Lock()


def _cache_dir(cache_key: str) -> Path:
    return Path(CACHE_ROOT) / cache_key


def _accepted_crops_dir(cache_key: str) -> Path:
    return _cache_dir(cache_key) / ACCEPTED_CROPS_DIRNAME


def _accepted_crop_path(cache_key: str, crop_id: str) -> Path:
    # Prevent path traversal; crop IDs are expected to be short IDs.
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(crop_id))
    return _accepted_crops_dir(cache_key) / f"{safe_id}.jpg"


def cache_accepted_crop_image(
    cache_key: str,
    crop_id: str,
    pil_img: Image.Image,
    target_size: Tuple[int, int] = ACCEPTED_CROP_SIZE,
    quality: int = 90,
) -> Path:
    """Write a UFM-ready accepted crop JPEG to temp cache."""
    out_dir = _accepted_crops_dir(cache_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _accepted_crop_path(cache_key, crop_id)

    img = pil_img.convert("RGB")
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)

    tmp = out_path.with_suffix(".tmp")
    img.save(tmp, "JPEG", quality=quality)
    tmp.rename(out_path)
    return out_path


def read_cached_accepted_crop_image(
    cache_key: str,
    crop_id: str,
    target_size: Tuple[int, int] = ACCEPTED_CROP_SIZE,
) -> Optional[Image.Image]:
    """Read a cached accepted crop JPEG, or None if absent/unreadable."""
    path = _accepted_crop_path(cache_key, crop_id)
    if not path.is_file():
        return None
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
        if img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)
        return img
    except (OSError, IOError) as exc:
        logger.warning("Failed reading accepted crop cache %s: %s", path, exc)
        return None


def delete_cached_accepted_crop_image(cache_key: str, crop_id: str) -> bool:
    """Delete cached accepted crop JPEG if present."""
    path = _accepted_crop_path(cache_key, crop_id)
    try:
        if path.is_file():
            path.unlink()
            return True
    except OSError as exc:
        logger.warning("Failed deleting accepted crop cache %s: %s", path, exc)
    return False


def cache_exists(cache_key: str) -> bool:
    return (_cache_dir(cache_key) / "config.json").is_file()


def list_project_caches(project_id: int) -> List[Dict[str, Any]]:
    """Find all caches belonging to a project via the project index."""
    index = _read_project_index()
    return index.get(str(project_id), [])


# ---------------------------------------------------------------------------
# Project index
# ---------------------------------------------------------------------------

def _index_path() -> Path:
    return Path(CACHE_ROOT) / PROJECT_INDEX_FILE


def _read_project_index() -> Dict[str, List[Dict[str, Any]]]:
    path = _index_path()
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read project index: %s", e)
        return {}


def _write_project_index(index: Dict[str, List[Dict[str, Any]]]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    tmp.rename(path)


def _update_project_index(project_id: int, cache_key: str, task_id: int, phase: str) -> None:
    with _index_lock:
        index = _read_project_index()
        pid = str(project_id)
        entries = index.get(pid, [])

        # Update or insert
        found = False
        for entry in entries:
            if entry.get("cache_key") == cache_key:
                entry["phase"] = phase
                entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                found = True
                break
        if not found:
            entries.append({
                "cache_key": cache_key,
                "task_id": task_id,
                "phase": phase,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

        index[pid] = entries
        _write_project_index(index)


def _remove_from_project_index(project_id: int, cache_key: str) -> None:
    with _index_lock:
        index = _read_project_index()
        pid = str(project_id)
        entries = index.get(pid, [])
        entries = [e for e in entries if e.get("cache_key") != cache_key]
        if entries:
            index[pid] = entries
        else:
            index.pop(pid, None)
        _write_project_index(index)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_session(session: InterviewSession) -> None:
    """Persist session state to disk."""
    d = _cache_dir(session.cache_key)
    d.mkdir(parents=True, exist_ok=True)

    # config.json — lightweight session metadata
    config = {
        "session_id": session.session_id,
        "project_id": session.project_id,
        "task_id": session.task_id,
        "annotation_id": session.annotation_id,
        "cache_key": session.cache_key,
        "phase": session.phase.value,
        "video_path": session.video_path,
        "video_key": session.video_key,
        "width": session.width,
        "height": session.height,
        "frames_count": session.frames_count,
        "fps": session.fps,
        "prompts": session.prompts,
        "sampled_frames": session.sampled_frames,
        "model_trained": session.model_trained,
        "training_epochs": session.training_epochs,
        "training_accuracy": session.training_accuracy,
        "n_identities": session.n_identities,
        "embedding_job_id": session.embedding_job_id,
        "embedding_complete": session.embedding_complete,
        "change_keyframes": session.change_keyframes,
        "embedding_sampled_indices": session.embedding_sampled_indices,
        "current_round": session.current_round,
        "round_history": session.round_history,
        "round_frames": {str(k): v for k, v in session.round_frames.items()},
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "seed_config": {
            "frame_pct": session.seed_config.frame_pct,
            "confidence_threshold": session.seed_config.confidence_threshold,
        },
        "seed_cached_frames": session.seed_cached_frames,
        "seed_target_frames": session.seed_target_frames,
    }
    _write_json(d / "config.json", config)

    # crops_metadata.json — all crop data (excluding numpy arrays)
    crops_meta = {cid: c.to_dict() for cid, c in session.crops.items()}
    _write_json(d / "crops_metadata.json", crops_meta)

    # labels.json — just the labels for quick access
    labels = {cid: c.label.value for cid, c in session.crops.items()}
    _write_json(d / "labels.json", labels)

    # prompts.json
    _write_json(d / "prompts.json", session.prompts)

    # features.npz — DINOv3 features (float16 for space)
    _save_features(d, session)

    # clusters.json — ReID data
    reid_data = {
        "clusters": {str(k): v for k, v in session.reid_clusters.items()},
        "ufm_complete": session.ufm_complete,
        "ufm_crop_ids": session.ufm_crop_ids,
    }
    _write_json(d / "clusters.json", reid_data)

    # ufm_similarity.npz — UFM pairwise similarity matrix
    if session.ufm_similarity_matrix is not None:
        np.savez_compressed(
            d / "ufm_similarity.npz",
            matrix=session.ufm_similarity_matrix,
            crop_ids=np.array(session.ufm_crop_ids, dtype=object),
        )

    # seeds.json — generated seed regions (survive container restarts)
    if session.seeds:
        _write_json(d / "seeds.json", session.seeds)

    # Update project index
    _update_project_index(
        session.project_id, session.cache_key, session.task_id, session.phase.value
    )

    logger.info("Saved session %s to %s", session.session_id, d)


def load_session(cache_key: str) -> Optional[InterviewSession]:
    """Load session state from disk cache."""
    d = _cache_dir(cache_key)
    config_path = d / "config.json"
    if not config_path.is_file():
        return None

    config = _read_json(config_path)
    if config is None:
        return None

    session = InterviewSession(
        session_id=config.get("session_id", ""),
        project_id=config.get("project_id", 0),
        task_id=config.get("task_id", 0),
        annotation_id=config.get("annotation_id"),
        cache_key=cache_key,
        video_path=config.get("video_path", ""),
        video_key=config.get("video_key", ""),
        width=config.get("width", 0),
        height=config.get("height", 0),
        frames_count=config.get("frames_count", 0),
        fps=config.get("fps", 30.0),
        phase=Phase(config.get("phase", "init")),
        prompts=config.get("prompts", []),
        sampled_frames=config.get("sampled_frames", []),
        model_trained=config.get("model_trained", False),
        training_epochs=config.get("training_epochs", 0),
        training_accuracy=config.get("training_accuracy", 0.0),
        n_identities=config.get("n_identities", 0),
        embedding_job_id=config.get("embedding_job_id"),
        embedding_complete=config.get("embedding_complete", False),
        change_keyframes=config.get("change_keyframes", []),
        embedding_sampled_indices=config.get("embedding_sampled_indices", []),
        current_round=config.get("current_round", 0),
        round_history=config.get("round_history", []),
        created_at=config.get("created_at", time.time()),
        updated_at=config.get("updated_at", time.time()),
    )

    # Restore round_frames (keys must be ints)
    raw_rf = config.get("round_frames", {})
    session.round_frames = {int(k): v for k, v in raw_rf.items()}

    sc = config.get("seed_config", {})
    # Migration shim: old sessions may have "frame_interval" instead of "frame_pct"
    if "frame_interval" in sc and "frame_pct" not in sc:
        sc.pop("frame_interval")
        sc["frame_pct"] = 100
    session.seed_config = SeedConfig(
        frame_pct=sc.get("frame_pct", 100),
        confidence_threshold=sc.get("confidence_threshold", 0.8),
    )
    session.seed_cached_frames = list(config.get("seed_cached_frames", []))
    session.seed_target_frames = list(config.get("seed_target_frames", []))

    # Load crops
    crops_meta = _read_json(d / "crops_metadata.json")
    if crops_meta:
        for cid, cdata in crops_meta.items():
            session.crops[cid] = CropData.from_dict(cdata)

    # Load features
    _load_features(d, session)

    # Load ReID data
    reid_data = _read_json(d / "clusters.json")
    if reid_data:
        # Backward compat: discard stale fields from old cache format
        for stale_key in ("pairs", "must_links", "cannot_links",
                          "phase_stage", "visual_reid_proposals",
                          "visual_reid_weights", "visual_reid_verdicts_count"):
            reid_data.pop(stale_key, None)

        clusters = reid_data.get("clusters", {})
        session.reid_clusters = {int(k): v for k, v in clusters.items()}
        session.ufm_complete = reid_data.get("ufm_complete", False)
        session.ufm_crop_ids = reid_data.get("ufm_crop_ids", [])

    # Load UFM similarity matrix
    ufm_path = d / "ufm_similarity.npz"
    if ufm_path.is_file():
        try:
            data = np.load(ufm_path, allow_pickle=True)
            session.ufm_similarity_matrix = data["matrix"].astype(np.float32)
            session.ufm_crop_ids = list(data["crop_ids"])
            session.ufm_complete = True
        except Exception as e:
            logger.warning("Failed to load UFM similarity from %s: %s", ufm_path, e)

    # Load seeds
    seeds_data = _read_json(d / "seeds.json")
    if seeds_data:
        session.seeds = seeds_data

    logger.info("Loaded session %s from %s (phase=%s)", session.session_id, d, session.phase.value)
    return session


def delete_cache(
    cache_key: str,
    project_id: Optional[int] = None,
    keep_frame_cache: bool = False,
) -> bool:
    """Remove a cache directory and its project index entry.

    If *keep_frame_cache* is True, the ``frames/`` subdirectory (decoded
    JPEG frames) is preserved while all session state files are deleted.
    """
    import shutil
    d = _cache_dir(cache_key)
    if d.is_dir():
        if keep_frame_cache:
            # Delete everything EXCEPT the frames/ subdirectory
            for child in d.iterdir():
                if child.name == "frames":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            logger.info("Deleted session state for %s (kept frame cache)", cache_key)
        else:
            shutil.rmtree(d)
            logger.info("Deleted cache %s", cache_key)

    # Also clean up the video download cache if present.
    # cache_key is "p{project}_t{task}" — extract task_id.
    if not keep_frame_cache:
        try:
            parts = cache_key.split("_t", 1)
            if len(parts) == 2:
                task_id = parts[1]
                video_cache_dir = Path("/data/video_cache") / task_id
                if video_cache_dir.is_dir():
                    shutil.rmtree(video_cache_dir)
                    logger.info("Deleted video download cache %s", video_cache_dir)
        except Exception as exc:
            logger.debug("Could not clean video cache for %s: %s", cache_key, exc)

    if project_id is not None:
        _remove_from_project_index(project_id, cache_key)

    return True



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.rename(path)


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _save_features(d: Path, session: InterviewSession) -> None:
    """Save DINOv3 features, context features, and metadata arrays to npz."""
    ids = []
    feats = []
    metas = []
    ctx_feats = []
    has_ctx = False
    for cid, crop in session.crops.items():
        if crop.features is not None:
            ids.append(cid)
            feats.append(crop.features)
            if crop.metadata is not None:
                metas.append(crop.metadata)
            if crop.context_features is not None:
                ctx_feats.append(crop.context_features)
                has_ctx = True
            else:
                ctx_feats.append(np.zeros(1024, dtype=np.float32))

    if feats:
        feat_arr = np.stack(feats).astype(np.float16)
        meta_arr = np.stack(metas).astype(np.float16) if metas and len(metas) == len(feats) else np.array([])
        save_kwargs = dict(
            ids=np.array(ids, dtype=object),
            features=feat_arr,
            metadata=meta_arr,
        )
        if has_ctx:
            save_kwargs["context_features"] = np.stack(ctx_feats).astype(np.float16)
        np.savez_compressed(d / "features.npz", **save_kwargs)


def _load_features(d: Path, session: InterviewSession) -> None:
    """Load DINOv3 features (and context features) back into session crops."""
    path = d / "features.npz"
    if not path.is_file():
        return
    try:
        data = np.load(path, allow_pickle=True)
        ids = data["ids"]
        feats = data["features"].astype(np.float32)
        metas = data.get("metadata")
        if metas is not None and metas.size > 0:
            metas = metas.astype(np.float32)
        ctx = data.get("context_features")
        if ctx is not None and ctx.size > 0:
            ctx = ctx.astype(np.float32)

        for i, cid in enumerate(ids):
            cid_str = str(cid)
            crop = session.crops.get(cid_str)
            if crop is not None:
                crop.features = feats[i]
                if metas is not None and metas.size > 0 and i < len(metas):
                    crop.metadata = metas[i]
                if ctx is not None and ctx.size > 0 and i < len(ctx):
                    vec = ctx[i]
                    # Skip zero-vectors (placeholder for missing context)
                    if np.any(vec != 0):
                        crop.context_features = vec
    except Exception as e:
        logger.warning("Failed to load features from %s: %s", path, e)
