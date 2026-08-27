"""In-process job store for video runs.

Processing a clip takes tens of seconds, which is far too long to hold an HTTP
request open. So uploads return a job id immediately and the client polls.

Deliberately in-memory and single-process: this matches the single-container
deployment it runs in. A multi-replica deployment would swap this for Redis and
a worker queue, and nothing outside this module would need to change.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    frames_done: int = 0
    frames_total: int = 0
    message: str = ""
    error: str | None = None
    stats: dict | None = None
    result_path: Path | None = None
    source_path: Path | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict:
        body = {
            "job_id": self.id,
            "state": self.state.value,
            "progress": round(self.progress, 3),
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "message": self.message,
            "elapsed_s": round((self.finished_at or time.time()) - self.created_at, 1),
        }
        if self.error:
            body["error"] = self.error
        if self.stats:
            body["stats"] = self.stats
        if self.state is JobState.DONE:
            body["result_url"] = f"/api/video/{self.id}/result"
        return body


class JobStore:
    """Thread-safe job registry with TTL cleanup of finished work."""

    def __init__(self, root: Path = config.JOB_DIR, ttl_s: int = config.VIDEO_JOB_TTL_S):
        self.root = root
        self.ttl_s = ttl_s
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        self.purge_expired()
        job = Job(id=uuid.uuid4().hex[:12])
        (self.root / job.id).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def dir_for(self, job: Job) -> Path:
        return self.root / job.id

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def set_progress(self, job_id: str, done: int, total: int) -> None:
        # total is an estimate for containers that do not report a frame count,
        # so clamp rather than letting the bar run past 100%.
        ratio = min(done / total, 1.0) if total else 0.0
        self.update(job_id, frames_done=done, frames_total=total, progress=ratio)

    def purge_expired(self) -> int:
        """Drop finished jobs past their TTL, and their files with them."""
        cutoff = time.time() - self.ttl_s
        removed = 0
        with self._lock:
            stale = [
                j for j in self._jobs.values()
                if j.finished_at and j.finished_at < cutoff
            ]
            for job in stale:
                self._jobs.pop(job.id, None)
        for job in stale:
            shutil.rmtree(self.root / job.id, ignore_errors=True)
            removed += 1
        if removed:
            log.info("Purged %d expired job(s)", removed)
        return removed

    def stats(self) -> dict:
        with self._lock:
            jobs = list(self._jobs.values())
        return {
            "total": len(jobs),
            "running": sum(1 for j in jobs if j.state is JobState.RUNNING),
            "queued": sum(1 for j in jobs if j.state is JobState.QUEUED),
        }
