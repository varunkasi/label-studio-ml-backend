"""Disk-based frame cache for decoded video frames.

Three-tier cache hierarchy for frame access:

  1. LRU memory cache (frame_cache.py) — OrderedDict, ~64 frames, instant
  2. Disk frame cache (this module) — JPEG files under frames/, ~ms seek
  3. PyAV seek (seeding_common._read_frame_pyav) — video container seek, ~10-50ms

During background embedding, frames are decoded once and written to disk as
JPEG files under ``/data/adapters/{cache_key}/frames/``.  Subsequent reads
(change detection, seeding, UI) hit the disk cache instead of re-decoding
from the video container.

File layout::

    /data/adapters/{cache_key}/
        frames/
            meta.json          — cache metadata (fps, resolution, sampled_indices)
            00000000.jpg       — frame at index 0
            00000030.jpg       — frame at index 30
            ...
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from PIL import Image

from .cache_manager import CACHE_ROOT

logger = logging.getLogger(__name__)

_meta_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frames_dir(cache_key: str) -> Path:
    return Path(CACHE_ROOT) / cache_key / "frames"


def _meta_path(cache_key: str) -> Path:
    return _frames_dir(cache_key) / "meta.json"


def _frame_path(cache_key: str, frame_idx: int) -> Path:
    return _frames_dir(cache_key) / f"{frame_idx:08d}.jpg"


def _write_meta(path: Path, obj: dict) -> None:
    """Atomic JSON write: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.rename(path)


def _read_meta(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Write API (used during background embedding)
# ---------------------------------------------------------------------------

def init_frame_cache(cache_key: str, meta: dict) -> Path:
    """Create frames/ dir and write initial meta.json.

    Parameters
    ----------
    cache_key : str
        Session cache key.
    meta : dict
        Must contain: target_fps, src_fps, resolution ([W, H]), created_at.

    Returns
    -------
    Path
        The frames directory path.
    """
    fdir = _frames_dir(cache_key)
    fdir.mkdir(parents=True, exist_ok=True)
    with _meta_lock:
        _write_meta(_meta_path(cache_key), meta)
    logger.info("Initialized frame cache for %s at %s", cache_key, fdir)
    return fdir


def write_frame(cache_key: str, frame_idx: int, pil_img: Image.Image,
                quality: int = 95) -> None:
    """Save a single frame as JPEG.

    File is saved as ``frames/{frame_idx:08d}.jpg``.  Called once per frame
    during streaming decode — must be fast (no ``optimize=True``).
    """
    path = _frame_path(cache_key, frame_idx)
    pil_img.save(path, "JPEG", quality=quality)


def finalize_frame_cache(cache_key: str, sampled_indices: List[int],
                         total_size_bytes: int) -> None:
    """Update meta.json with sampled_indices and total_size_bytes.

    Called once after all frames have been written.
    """
    with _meta_lock:
        meta = _read_meta(_meta_path(cache_key))
        if meta is None:
            meta = {}
        meta["sampled_indices"] = sampled_indices
        meta["total_size_bytes"] = total_size_bytes
        _write_meta(_meta_path(cache_key), meta)
    logger.info(
        "Finalized frame cache for %s: %d frames, %d bytes",
        cache_key, len(sampled_indices), total_size_bytes,
    )


def update_frame_cache_meta(cache_key: str, updates: dict,
                           force_keys: Optional[set] = None) -> bool:
    """Merge *updates* into an existing meta.json.

    By default, only writes keys that are new (does not overwrite existing).
    Keys listed in *force_keys* will overwrite even if they already exist —
    used for repairing stale ``video_path`` entries after container rebuilds.

    Returns True if meta.json was modified, False otherwise.
    """
    if force_keys is None:
        force_keys = set()
    with _meta_lock:
        existing = _read_meta(_meta_path(cache_key))
        if existing is None:
            return False
        changed_keys = []
        for k, v in updates.items():
            if k not in existing:
                existing[k] = v
                changed_keys.append(k)
            elif k in force_keys and existing[k] != v:
                existing[k] = v
                changed_keys.append(k)
        if changed_keys:
            _write_meta(_meta_path(cache_key), existing)
            logger.info("Updated meta.json for %s: changed keys %s",
                        cache_key, changed_keys)
    return bool(changed_keys)


# ---------------------------------------------------------------------------
# Read API (used by change detection, seeding, UI)
# ---------------------------------------------------------------------------

def frame_cache_exists(cache_key: str) -> bool:
    """Check if frame cache is complete (finalized).

    Returns True only if meta.json exists and contains ``sampled_indices``
    (meaning ``finalize_frame_cache`` was called).  Returns False for
    partial or incomplete caches.
    """
    meta = _read_meta(_meta_path(cache_key))
    if meta is None:
        return False
    return "sampled_indices" in meta


def get_frame_cache_meta(cache_key: str) -> Optional[dict]:
    """Read and return meta.json contents, or None if not present."""
    return _read_meta(_meta_path(cache_key))


def read_cached_frame(cache_key: str, frame_idx: int) -> Optional[Image.Image]:
    """Read a single cached frame from disk.

    Returns the PIL Image or None if the file does not exist.
    """
    path = _frame_path(cache_key, frame_idx)
    if not path.is_file():
        return None
    try:
        img = Image.open(path)
        img.load()  # force read into memory so file handle is released
        return img
    except (OSError, IOError) as e:
        logger.warning("Failed to read cached frame %s: %s", path, e)
        return None


def iter_cached_frames(cache_key: str,
                       frame_indices: Optional[List[int]] = None
                       ) -> Iterator[Tuple[int, Image.Image]]:
    """Yield (frame_idx, PIL.Image) tuples in order.

    If *frame_indices* is None, iterate all frames listed in
    ``sampled_indices`` from meta.json.  If *frame_indices* is provided,
    iterate only those indices (skipping any that are missing on disk).
    """
    if frame_indices is None:
        meta = _read_meta(_meta_path(cache_key))
        if meta is None:
            return
        frame_indices = meta.get("sampled_indices", [])

    for idx in frame_indices:
        img = read_cached_frame(cache_key, idx)
        if img is not None:
            yield idx, img


# ---------------------------------------------------------------------------
# Management API (used by UI landing page)
# ---------------------------------------------------------------------------

def get_frame_cache_size(cache_key: str) -> int:
    """Total bytes of the frames/ directory.

    Uses ``total_size_bytes`` from meta.json if available (fast path).
    Falls back to walking the directory.  Returns 0 if not present.
    """
    meta = _read_meta(_meta_path(cache_key))
    if meta is not None and "total_size_bytes" in meta:
        return meta["total_size_bytes"]

    fdir = _frames_dir(cache_key)
    if not fdir.is_dir():
        return 0

    total = 0
    for entry in fdir.iterdir():
        if entry.is_file():
            total += entry.stat().st_size
    return total


def get_all_frame_cache_sizes() -> Dict[str, int]:
    """Scan all cache_key dirs under CACHE_ROOT that have a frames/ subdir.

    Returns a dict mapping ``{cache_key: size_bytes}``.
    """
    root = Path(CACHE_ROOT)
    if not root.is_dir():
        return {}

    result: Dict[str, int] = {}
    for entry in root.iterdir():
        if entry.is_dir() and (entry / "frames").is_dir():
            result[entry.name] = get_frame_cache_size(entry.name)
    return result


def delete_frame_cache(cache_key: str) -> bool:
    """Delete the frames/ directory for a cache_key.

    Returns True if the directory was deleted, False if it did not exist.
    """
    fdir = _frames_dir(cache_key)
    if fdir.is_dir():
        shutil.rmtree(fdir)
        logger.info("Deleted frame cache for %s", cache_key)
        return True
    return False
