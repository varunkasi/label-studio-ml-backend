"""Shared LRU frame cache for the Interview UI.

Extracted from routes.py to avoid circular imports — both routes.py
(HTTP endpoints) and dinov3_classifier.py (feature extraction) need
frame caching.

Keyed by ``(video_path, frame_idx)``.  Default 64 entries ~ 384 MB.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

_FRAME_CACHE_SIZE = int(os.getenv("INTERVIEW_FRAME_CACHE_SIZE", "64"))
_frame_cache: OrderedDict = OrderedDict()
_frame_cache_lock = threading.Lock()


def get_cached_frame(video_path: str, frame_idx: int) -> Optional[Image.Image]:
    """Return cached PIL Image or None."""
    key = (video_path, frame_idx)
    with _frame_cache_lock:
        if key in _frame_cache:
            _frame_cache.move_to_end(key)
            return _frame_cache[key]
    return None


def put_cached_frame(video_path: str, frame_idx: int, pil_img: Image.Image) -> None:
    """Store a PIL Image in the LRU cache."""
    key = (video_path, frame_idx)
    with _frame_cache_lock:
        _frame_cache[key] = pil_img
        _frame_cache.move_to_end(key)
        while len(_frame_cache) > _FRAME_CACHE_SIZE:
            _frame_cache.popitem(last=False)


def read_frame_cached(video_path: str, frame_idx: int) -> Optional[Image.Image]:
    """Read a frame with LRU caching.  Falls back to _read_frame_pyav."""
    cached = get_cached_frame(video_path, frame_idx)
    if cached is not None:
        return cached

    import sys
    _parent = os.path.dirname(os.path.dirname(__file__))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from seeding_common import _read_frame_pyav

    pil_img = _read_frame_pyav(video_path, frame_idx)
    if pil_img is not None:
        put_cached_frame(video_path, frame_idx, pil_img)
    return pil_img
