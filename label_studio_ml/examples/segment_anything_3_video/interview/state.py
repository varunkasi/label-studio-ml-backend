"""In-memory session state for Interview UI.

Manages InterviewSession objects with thread-safe access. Each session
represents a single interview workflow (detection → classification →
ReID → seeding) for one Label Studio task.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    """Interview workflow phases."""
    INIT = "init"
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    REID = "reid"
    SEEDING = "seeding"
    COMPLETE = "complete"


class CropLabel(str, Enum):
    """Label state for a detection crop."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"  # Excluded from classifier training


class CropSource(str, Enum):
    """How a crop was generated."""
    TEXT_DETECT = "text_detect"       # Sam3TextBasedDetector
    MULTI_PROMPT = "multi_prompt"     # Strategy A
    HUMAN_DRAWN = "human_drawn"       # Manual draw mode
    CHANGE_DETECT = "change_detect"   # Detection on change-detected keyframes


@dataclass
class CropData:
    """A single detected/drawn bounding box crop."""
    crop_id: str
    frame_idx: int
    xyxy: np.ndarray          # pixel coords [x1, y1, x2, y2]
    score: float
    label: CropLabel = CropLabel.PENDING
    source: CropSource = CropSource.TEXT_DETECT
    prompt: str = ""
    cluster_id: Optional[int] = None    # detection-phase cluster
    reid_cluster_id: Optional[int] = None  # ReID identity cluster
    uncertainty: float = 0.5            # classifier uncertainty (0=certain, 1=uncertain)
    features: Optional[np.ndarray] = None  # DINOv3 CLS token (1024,)
    metadata: Optional[np.ndarray] = None  # [norm_cx, norm_cy, scale, aspect] (4,)
    mask_quality: Optional[np.ndarray] = None  # [fill_ratio, det_score, edge_contact, compactness] (4,)
    histogram: Optional[np.ndarray] = None  # HSV color histogram for ReID

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON (excludes numpy arrays)."""
        return {
            "crop_id": self.crop_id,
            "frame_idx": self.frame_idx,
            "xyxy": self.xyxy.tolist(),
            "score": self.score,
            "label": self.label.value,
            "source": self.source.value,
            "prompt": self.prompt,
            "cluster_id": self.cluster_id,
            "reid_cluster_id": self.reid_cluster_id,
            "uncertainty": self.uncertainty,
            "mask_quality": self.mask_quality.tolist() if self.mask_quality is not None else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CropData":
        """Deserialize from JSON dict."""
        return CropData(
            crop_id=d["crop_id"],
            frame_idx=d["frame_idx"],
            xyxy=np.array(d["xyxy"], dtype=np.float32),
            score=d["score"],
            label=CropLabel(d.get("label", "pending")),
            source=CropSource(d.get("source", "text_detect")),
            prompt=d.get("prompt", ""),
            cluster_id=d.get("cluster_id"),
            reid_cluster_id=d.get("reid_cluster_id"),
            uncertainty=d.get("uncertainty", 0.5),
            mask_quality=np.array(d["mask_quality"], dtype=np.float32) if d.get("mask_quality") is not None else None,
        )


@dataclass
class ReIDPair:
    """A pair of crops presented for identity comparison."""
    pair_id: str
    crop_id_a: str
    crop_id_b: str
    cluster_a: int
    cluster_b: int
    pool: str  # "ambiguous", "confident_same", "confident_different"
    similarity: float
    resolution: Optional[str] = None  # "same", "different", "unsure"


@dataclass
class SeedConfig:
    """User-configurable seed generation parameters."""
    frame_interval: int = 5
    confidence_threshold: float = 0.8


@dataclass
class InterviewSession:
    """Full state for one interview session."""
    session_id: str
    project_id: int
    task_id: int
    annotation_id: Optional[int] = None
    cache_key: str = ""

    # Video info
    video_path: str = ""
    video_key: str = ""
    width: int = 0
    height: int = 0
    frames_count: int = 0
    fps: float = 30.0

    # Workflow
    phase: Phase = Phase.INIT
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Detection
    prompts: List[str] = field(default_factory=list)
    sampled_frames: List[int] = field(default_factory=list)
    crops: Dict[str, CropData] = field(default_factory=dict)

    # Background embedding state
    embedding_job_id: Optional[str] = None
    embedding_complete: bool = False
    change_keyframes: List[int] = field(default_factory=list)
    embedding_sampled_indices: List[int] = field(default_factory=list)

    # Round-based active learning
    current_round: int = 0
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    round_frames: Dict[int, List[int]] = field(default_factory=dict)

    # Held-out validation (selected in Round 1, never used for training)
    validation_frames: List[int] = field(default_factory=list)
    validation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Classification
    model_trained: bool = False
    training_epochs: int = 0
    training_accuracy: float = 0.0

    # ReID
    reid_pairs: Dict[str, ReIDPair] = field(default_factory=dict)
    reid_clusters: Dict[int, List[str]] = field(default_factory=dict)  # cluster_id -> [crop_ids]
    n_identities: int = 0
    reid_round: int = 0  # current ReID pair resolution round

    # Seeding
    seed_config: SeedConfig = field(default_factory=SeedConfig)
    seeds: List[Dict[str, Any]] = field(default_factory=list)  # final seed regions
    upload_result: Optional[Dict[str, Any]] = None

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def touch(self) -> None:
        """Update the last-modified timestamp."""
        self.updated_at = time.time()

    # -- Crop CRUD --

    def add_crop(self, crop: CropData) -> None:
        self.crops[crop.crop_id] = crop
        self.touch()

    def get_crop(self, crop_id: str) -> Optional[CropData]:
        return self.crops.get(crop_id)

    def label_crop(self, crop_id: str, label: CropLabel) -> bool:
        crop = self.crops.get(crop_id)
        if crop is None:
            return False
        crop.label = label
        self.touch()
        return True

    def get_crops_by_label(self, label: CropLabel) -> List[CropData]:
        return [c for c in self.crops.values() if c.label == label]

    def get_crops_by_frame(self, frame_idx: int) -> List[CropData]:
        return [c for c in self.crops.values() if c.frame_idx == frame_idx]

    def get_pending_crops_sorted(self) -> List[CropData]:
        """Return pending crops sorted by uncertainty (most uncertain first),
        with stratified class balance for alternating pos/neg."""
        pending = [c for c in self.crops.values() if c.label == CropLabel.PENDING]
        return sorted(pending, key=lambda c: -c.uncertainty)

    def get_validation_crop_ids(self) -> List[str]:
        """Return crop IDs on validation frames (for held-out evaluation)."""
        val_set = set(self.validation_frames)
        return [c.crop_id for c in self.crops.values() if c.frame_idx in val_set]

    # -- Phase transitions --

    def advance_to(self, phase: Phase) -> None:
        self.phase = phase
        self.touch()

    # -- Stats --

    def stats(self) -> Dict[str, Any]:
        accepted = len(self.get_crops_by_label(CropLabel.ACCEPTED))
        rejected = len(self.get_crops_by_label(CropLabel.REJECTED))
        pending = len(self.get_crops_by_label(CropLabel.PENDING))
        skipped = len(self.get_crops_by_label(CropLabel.SKIPPED))
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "video_frames": self.frames_count,
            "sampled_frames": len(self.sampled_frames),
            "total_crops": len(self.crops),
            "accepted": accepted,
            "rejected": rejected,
            "skipped": skipped,
            "pending": pending,
            "model_trained": self.model_trained,
            "training_accuracy": self.training_accuracy,
            "n_identities": self.n_identities,
            "prompts": self.prompts,
            "current_round": self.current_round,
            "rounds_completed": len(self.round_history),
            "validation_history": self.validation_history,
        }


# ---------------------------------------------------------------------------
# Session registry (in-memory, thread-safe)
# ---------------------------------------------------------------------------

_sessions: Dict[str, InterviewSession] = {}
_registry_lock = threading.Lock()


def create_session(project_id: int, task_id: int, annotation_id: Optional[int] = None) -> InterviewSession:
    """Create and register a new interview session."""
    session_id = str(uuid.uuid4())[:12]
    cache_key = f"p{project_id}_t{task_id}"
    session = InterviewSession(
        session_id=session_id,
        project_id=project_id,
        task_id=task_id,
        annotation_id=annotation_id,
        cache_key=cache_key,
    )
    with _registry_lock:
        _sessions[session_id] = session
    logger.info("Created session %s (cache_key=%s)", session_id, cache_key)
    return session


def get_session(session_id: str) -> Optional[InterviewSession]:
    """Retrieve a session by ID."""
    with _registry_lock:
        session = _sessions.get(session_id)
        if session is None and session_id:
            logger.warning(
                "get_session(%r) → None. Registry has %d sessions: %s",
                session_id, len(_sessions), list(_sessions.keys()),
            )
        return session


def list_sessions() -> List[Dict[str, Any]]:
    """Return summary of all active sessions."""
    with _registry_lock:
        return [s.stats() for s in _sessions.values()]


def delete_session(session_id: str) -> bool:
    """Remove a session from the registry."""
    with _registry_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            logger.info("Deleted session %s", session_id)
            return True
        return False


def get_or_create_session(project_id: int, task_id: int, annotation_id: Optional[int] = None) -> Tuple[InterviewSession, bool]:
    """Find existing session for this task or create a new one.

    Returns (session, is_new).
    """
    cache_key = f"p{project_id}_t{task_id}"
    with _registry_lock:
        for s in _sessions.values():
            if s.cache_key == cache_key:
                return s, False
    return create_session(project_id, task_id, annotation_id), True
