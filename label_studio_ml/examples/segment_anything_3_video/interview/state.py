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
    BOX_CORRECTED = "box_corrected"   # Adjusted box from reject review


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
    context_features: Optional[np.ndarray] = None  # DINOv3 CLS on 50%-expanded crop (1024,)
    metadata: Optional[np.ndarray] = None  # [norm_cx, norm_cy, scale, aspect] (4,)
    mask_quality: Optional[np.ndarray] = None  # [fill_ratio, det_score, edge_contact, compactness] (4,)
    histogram: Optional[np.ndarray] = None  # HSV color histogram for ReID
    reject_reason: Optional[str] = None  # "not_person", "partial_box", "oversized_box"
    corrected_from: Optional[str] = None  # crop_id of original bad box (for BOX_CORRECTED crops)
    # Cross-task support transfer provenance (use_from_<task_id> mode)
    is_imported_support: bool = False
    source_project_id: Optional[int] = None
    source_task_id: Optional[int] = None
    source_session_id: Optional[str] = None
    source_crop_id: Optional[str] = None
    source_frame_idx: Optional[int] = None
    source_video_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON (excludes numpy arrays)."""
        d = {
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
        if self.reject_reason is not None:
            d["reject_reason"] = self.reject_reason
        if self.corrected_from is not None:
            d["corrected_from"] = self.corrected_from
        if self.is_imported_support:
            d["is_imported_support"] = True
        if self.source_project_id is not None:
            d["source_project_id"] = self.source_project_id
        if self.source_task_id is not None:
            d["source_task_id"] = self.source_task_id
        if self.source_session_id is not None:
            d["source_session_id"] = self.source_session_id
        if self.source_crop_id is not None:
            d["source_crop_id"] = self.source_crop_id
        if self.source_frame_idx is not None:
            d["source_frame_idx"] = self.source_frame_idx
        if self.source_video_key is not None:
            d["source_video_key"] = self.source_video_key
        return d

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
            reject_reason=d.get("reject_reason"),
            corrected_from=d.get("corrected_from"),
            is_imported_support=bool(d.get("is_imported_support", False)),
            source_project_id=d.get("source_project_id"),
            source_task_id=d.get("source_task_id"),
            source_session_id=d.get("source_session_id"),
            source_crop_id=d.get("source_crop_id"),
            source_frame_idx=d.get("source_frame_idx"),
            source_video_key=d.get("source_video_key"),
        )


@dataclass
class SeedConfig:
    """User-configurable seed generation parameters."""
    frame_pct: int = 100           # % of cached frames to scan (0-100)
    confidence_threshold: float = 0.8


# ---------------------------------------------------------------------------
# Visual ReID Pipeline Data Structures
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    """Video modality detected from pixel content."""
    RGB = "rgb"
    GRAYSCALE = "grayscale"
    IR = "ir"


@dataclass
class CueWeights:
    """Per-cue weights for the visual ReID similarity function.

    Default weights sum to 1.0. Use normalize() after manual edits to
    ensure the constraint holds.
    """
    appearance: float = 0.40
    spatial: float = 0.15
    body: float = 0.05
    color: float = 0.20
    context: float = 0.10
    temporal: float = 0.10

    def normalize(self, floor: float = 0.0) -> None:
        """Floor negative values and rescale so weights sum to 1.0."""
        self.appearance = max(self.appearance, floor)
        self.spatial = max(self.spatial, floor)
        self.body = max(self.body, floor)
        self.color = max(self.color, floor)
        self.context = max(self.context, floor)
        self.temporal = max(self.temporal, floor)
        total = (
            self.appearance + self.spatial + self.body
            + self.color + self.context + self.temporal
        )
        if total > 1e-12:
            self.appearance /= total
            self.spatial /= total
            self.body /= total
            self.color /= total
            self.context /= total
            self.temporal /= total

    def to_dict(self) -> Dict[str, float]:
        return {
            "appearance": self.appearance,
            "spatial": self.spatial,
            "body": self.body,
            "color": self.color,
            "context": self.context,
            "temporal": self.temporal,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "CueWeights":
        return cls(
            appearance=d["appearance"],
            spatial=d["spatial"],
            body=d["body"],
            color=d["color"],
            context=d["context"],
            temporal=d["temporal"],
        )


@dataclass
class EnrichedCrop:
    """Crop with full multi-cue feature descriptor for visual ReID.

    Built from CropData by enrich_crops_t1() (Tier 1) and later tiers.
    """
    crop_id: str
    frame_idx: int
    modality: str                               # "rgb", "grayscale", "ir"
    dinov3_cls: np.ndarray                      # (1024,) DINOv3 CLS token
    body_props: np.ndarray                      # (4,) [aspect, area_norm, torso_ratio, limb_spread]
    scene_pos: Optional[np.ndarray] = None      # (2,) [norm_cx, norm_cy]
    quadrant: int = 0                           # 0-8 spatial quadrant
    context_cls: Optional[np.ndarray] = None    # (1024,) context patch DINOv3
    color_hist: Optional[np.ndarray] = None     # (512,) HSV histogram
    lighting_ctx: Optional[np.ndarray] = None   # (4,) lighting context
    relative_color: Optional[np.ndarray] = None # (3,) relative color stats


@dataclass
class RunGroup:
    """A temporal run of consecutive crops from the same identity track.

    Used for temporal co-occurrence analysis and run-level averaging.
    """
    run_id: int
    crop_ids: List[str]
    mean_features: np.ndarray                   # (1024,) mean DINOv3
    mean_scene_pos: np.ndarray                  # (2,) mean normalized position
    frame_range: Tuple[int, int]                # (start_frame, end_frame)
    mean_body_props: Optional[np.ndarray] = None
    mean_color_hist: Optional[np.ndarray] = None
    mean_context_cls: Optional[np.ndarray] = None


@dataclass
class MergeProposal:
    """A proposal to merge two clusters, with per-cue evidence.

    Generated by the merge proposal scorer and ranked by information_gain
    for human presentation.
    """
    cluster_a: int
    cluster_b: int
    merge_score: float
    per_cue: Dict[str, float]
    cooccurrence_conflict: bool = False
    temporal_overlap: float = 0.0
    information_gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_a": self.cluster_a,
            "cluster_b": self.cluster_b,
            "merge_score": self.merge_score,
            "per_cue": self.per_cue,
            "cooccurrence_conflict": self.cooccurrence_conflict,
            "temporal_overlap": self.temporal_overlap,
            "information_gain": self.information_gain,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MergeProposal":
        return cls(
            cluster_a=d["cluster_a"],
            cluster_b=d["cluster_b"],
            merge_score=d["merge_score"],
            per_cue=d["per_cue"],
            cooccurrence_conflict=d.get("cooccurrence_conflict", False),
            temporal_overlap=d.get("temporal_overlap", 0.0),
            information_gain=d.get("information_gain", 0.0),
        )


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
    _crops_by_frame: Dict[int, Dict[str, CropData]] = field(default_factory=dict, repr=False)
    _crop_index_count: int = field(default=0, repr=False)

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

    # ReID (UFM-based)
    reid_clusters: Dict[int, List[str]] = field(default_factory=dict)  # cluster_id -> [crop_ids]
    n_identities: int = 0

    # UFM similarity matrix (precomputed pairwise covisibility)
    ufm_similarity_matrix: Optional[np.ndarray] = None  # (N, N) float32
    ufm_crop_ids: List[str] = field(default_factory=list)  # ordered crop IDs matching matrix rows
    ufm_complete: bool = False
    ufm_job_id: Optional[str] = None

    # Seeding
    seed_config: SeedConfig = field(default_factory=SeedConfig)
    # All generated seed candidates (unfiltered). Confidence filtering is applied
    # at list/upload time using ``seed_config.confidence_threshold``.
    seeds: List[Dict[str, Any]] = field(default_factory=list)
    # Cached-frame subset selected by the frame-coverage slider for the current
    # generation state. Used to compute incremental regeneration deltas.
    seed_cached_frames: List[int] = field(default_factory=list)
    # Full target frame set currently represented in ``seeds`` (coverage subset
    # plus any forced change keyframes).
    seed_target_frames: List[int] = field(default_factory=list)
    upload_result: Optional[Dict[str, Any]] = None

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def touch(self) -> None:
        """Update the last-modified timestamp."""
        self.updated_at = time.time()

    # -- Crop CRUD --

    def add_crop(self, crop: CropData) -> None:
        existing = self.crops.get(crop.crop_id)
        if existing is not None:
            frame_map = self._crops_by_frame.get(existing.frame_idx)
            if frame_map is not None and crop.crop_id in frame_map:
                frame_map.pop(crop.crop_id, None)
                if not frame_map:
                    self._crops_by_frame.pop(existing.frame_idx, None)
        else:
            self._crop_index_count += 1
        self.crops[crop.crop_id] = crop
        self._crops_by_frame.setdefault(crop.frame_idx, {})[crop.crop_id] = crop
        self.touch()

    def remove_crop(self, crop_id: str) -> Optional[CropData]:
        crop = self.crops.pop(crop_id, None)
        if crop is None:
            return None
        frame_map = self._crops_by_frame.get(crop.frame_idx)
        if frame_map is not None:
            frame_map.pop(crop_id, None)
            if not frame_map:
                self._crops_by_frame.pop(crop.frame_idx, None)
        self._crop_index_count = max(0, self._crop_index_count - 1)
        self.touch()
        return crop

    def get_crop(self, crop_id: str) -> Optional[CropData]:
        return self.crops.get(crop_id)

    def label_crop(self, crop_id: str, label: CropLabel) -> bool:
        crop = self.crops.get(crop_id)
        if crop is None:
            return False
        crop.label = label
        self.touch()
        return True

    def get_crops_by_label(self, label: CropLabel, include_imported: bool = True) -> List[CropData]:
        return [
            c for c in self.crops.values()
            if c.label == label and (include_imported or not c.is_imported_support)
        ]

    def get_crops_by_frame(self, frame_idx: int) -> List[CropData]:
        self._ensure_crop_index()
        return list(self._crops_by_frame.get(frame_idx, {}).values())

    def get_pending_crops_sorted(self) -> List[CropData]:
        """Return pending crops sorted by uncertainty (most uncertain first),
        with stratified class balance for alternating pos/neg."""
        pending = [c for c in self.crops.values() if c.label == CropLabel.PENDING]
        return sorted(pending, key=lambda c: -c.uncertainty)

    def _ensure_crop_index(self) -> None:
        # Guard against direct writes to session.crops that bypass add/remove APIs.
        if self._crop_index_count == len(self.crops):
            return
        self._rebuild_crop_index()

    def _rebuild_crop_index(self) -> None:
        self._crops_by_frame = {}
        for crop in self.crops.values():
            self._crops_by_frame.setdefault(crop.frame_idx, {})[crop.crop_id] = crop
        self._crop_index_count = len(self.crops)

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
        local_crops = [c for c in self.crops.values() if not c.is_imported_support]
        imported_crops = [c for c in self.crops.values() if c.is_imported_support]

        accepted = sum(1 for c in local_crops if c.label == CropLabel.ACCEPTED)
        rejected = sum(1 for c in local_crops if c.label == CropLabel.REJECTED)
        pending = sum(1 for c in local_crops if c.label == CropLabel.PENDING)
        skipped = sum(1 for c in local_crops if c.label == CropLabel.SKIPPED)
        corrected_total = sum(
            1 for c in local_crops if c.source == CropSource.BOX_CORRECTED
        )
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "video_frames": self.frames_count,
            "sampled_frames": len(self.sampled_frames),
            "total_crops": len(local_crops),
            "total_crops_all": len(self.crops),
            "imported_support_total": len(imported_crops),
            "accepted": accepted,
            "rejected": rejected,
            "skipped": skipped,
            "pending": pending,
            "corrected_total": corrected_total,
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
