"""Thread-based background job executor with progress callbacks.

Long-running operations (detection, training, feature extraction) run in
daemon threads. Frontend polls /job/{id}/progress for status updates.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCancelledError(RuntimeError):
    """Raised when a background job is cooperatively cancelled."""


@dataclass
class JobProgress:
    """Progress state for a background job."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    step: str = ""
    current: int = 0
    total: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    eta_seconds: Optional[float] = None
    items_per_second: Optional[float] = None
    _pause_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def __post_init__(self):
        self._pause_event.set()  # Not paused by default

    def pause(self):
        """Pause the job — worker must call wait_if_paused() to block."""
        self._pause_event.clear()

    def resume(self):
        """Resume a paused job."""
        self._pause_event.set()

    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """Block until resumed.  Returns True if resumed, False if timed out."""
        return self._pause_event.wait(timeout=timeout)

    @property
    def paused(self) -> bool:
        return not self._pause_event.is_set()

    def cancel(self):
        """Request cooperative cancellation for this job."""
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise if cancellation was requested."""
        if self.cancelled:
            raise JobCancelledError("Job cancelled")

    def to_dict(self) -> Dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at
        pct = (self.current / self.total * 100) if self.total > 0 else 0
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "step": self.step,
            "current": self.current,
            "total": self.total,
            "percent": round(pct, 1),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": (round(self.eta_seconds, 1) if self.eta_seconds is not None else None),
            "items_per_second": (
                round(self.items_per_second, 2)
                if self.items_per_second is not None else None
            ),
            "error": self.error,
            "paused": self.paused,
            "cancel_requested": self.cancelled,
        }


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

_jobs: Dict[str, JobProgress] = {}
_jobs_lock = threading.Lock()

# Limit retained completed jobs to prevent memory leak
_MAX_COMPLETED_JOBS = 100


def _prune_old_jobs() -> None:
    """Remove oldest completed/failed jobs if over limit."""
    completed = [
        (jid, j) for jid, j in _jobs.items()
        if j.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
    ]
    if len(completed) <= _MAX_COMPLETED_JOBS:
        return
    completed.sort(key=lambda x: x[1].finished_at or 0)
    for jid, _ in completed[:len(completed) - _MAX_COMPLETED_JOBS]:
        _jobs.pop(jid, None)


def submit_job(
    fn: Callable[[JobProgress], Any],
    name: str = "background_job",
) -> str:
    """Submit a function to run in a background thread.

    The function receives a JobProgress object it can update with progress.
    Returns the job_id for polling.
    """
    job_id = str(uuid.uuid4())[:12]
    progress = JobProgress(job_id=job_id, step=f"Starting {name}...")

    with _jobs_lock:
        _jobs[job_id] = progress
        _prune_old_jobs()

    def _worker():
        progress.status = JobStatus.RUNNING
        progress.started_at = time.time()
        try:
            result = fn(progress)
            if progress.cancelled:
                progress.status = JobStatus.CANCELLED
                progress.step = progress.step or "Cancelled"
            else:
                progress.result = result
                progress.status = JobStatus.COMPLETED
        except JobCancelledError:
            progress.status = JobStatus.CANCELLED
            progress.step = progress.step or "Cancelled"
        except Exception as e:
            logger.error("Job %s (%s) failed: %s", job_id, name, e, exc_info=True)
            progress.error = str(e)
            progress.status = JobStatus.FAILED
        finally:
            progress.finished_at = time.time()

    t = threading.Thread(target=_worker, name=f"job-{job_id}-{name}", daemon=True)
    t.start()
    logger.info("Submitted job %s (%s)", job_id, name)
    return job_id


def get_job_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """Get current progress for a job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return None
    return job.to_dict()


def get_job_result(job_id: str) -> Any:
    """Get the result of a completed job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return None
    return job.result


def is_job_running(job_id: str) -> bool:
    """Check if a job is still running."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    return job is not None and job.status == JobStatus.RUNNING


def get_job(job_id: str) -> Optional[JobProgress]:
    """Get the raw JobProgress object (for pause_event access)."""
    with _jobs_lock:
        return _jobs.get(job_id)


def pause_job(job_id: str) -> bool:
    """Pause a running job.  Returns True if the job was found and paused."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job.status != JobStatus.RUNNING:
        return False
    job.pause()
    logger.info("Paused job %s", job_id)
    return True


def resume_job(job_id: str) -> bool:
    """Resume a paused job.  Returns True if the job was found and resumed."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job.status != JobStatus.RUNNING:
        return False
    job.resume()
    logger.info("Resumed job %s", job_id)
    return True


def cancel_job(job_id: str) -> Optional[bool]:
    """Request cancellation.

    Returns:
        True: cancellation requested for a running job.
        False: job exists but is already terminal.
        None: job was not found.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        if job.status != JobStatus.RUNNING:
            return False
        job.cancel()
    logger.info("Cancellation requested for job %s", job_id)
    return True
