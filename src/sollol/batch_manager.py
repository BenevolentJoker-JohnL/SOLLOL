"""
Batch Job Manager for SOLLOL.
Tracks and manages asynchronous batch processing jobs.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class BatchJobStatus(Enum):
    """Status of a batch job."""

    PENDING = "pending"  # Job created, not started
    RUNNING = "running"  # Job is executing
    COMPLETED = "completed"  # Job finished successfully
    FAILED = "failed"  # Job failed with errors
    CANCELLED = "cancelled"  # Job was cancelled


@dataclass
class BatchJob:
    """Represents a batch processing job."""

    job_id: str
    job_type: str  # "embed", "chat", "generate"
    status: BatchJobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    results: List[Any] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def get_progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    def get_duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": {
                "total_items": self.total_items,
                "completed_items": self.completed_items,
                "failed_items": self.failed_items,
                "percent": round(self.get_progress_percent(), 2),
            },
            "duration_seconds": self.get_duration_seconds(),
            "metadata": self.metadata,
        }


class BatchJobManager:
    """
    Manager for batch processing jobs.

    Features:
    - Job tracking with unique IDs
    - Status monitoring
    - Result storage
    - Job cleanup (configurable TTL)
    """

    def __init__(self, max_jobs: int = 1000, job_ttl_seconds: int = 3600):
        """
        Initialize batch job manager.

        Args:
            max_jobs: Maximum number of jobs to keep in memory
            job_ttl_seconds: Time-to-live for completed jobs (default: 1 hour)
        """
        self.jobs: Dict[str, BatchJob] = {}
        self.max_jobs = max_jobs
        self.job_ttl_seconds = job_ttl_seconds

        # Statistics
        self.total_jobs_created = 0
        self.total_jobs_completed = 0
        self.total_jobs_failed = 0

    def create_job(
        self,
        job_type: str,
        total_items: int,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a new batch job.

        Args:
            job_type: Type of job (embed, chat, generate)
            total_items: Number of items in batch
            metadata: Optional metadata (model, options, etc.)

        Returns:
            Job ID (UUID)
        """
        job_id = str(uuid.uuid4())

        job = BatchJob(
            job_id=job_id,
            job_type=job_type,
            status=BatchJobStatus.PENDING,
            created_at=datetime.now(),
            total_items=total_items,
            metadata=metadata or {},
        )

        self.jobs[job_id] = job
        self.total_jobs_created += 1

        # Cleanup old jobs if needed
        if len(self.jobs) > self.max_jobs:
            self._cleanup_old_jobs()

        return job_id

    def start_job(self, job_id: str) -> None:
        """Mark job as started."""
        if job_id in self.jobs:
            self.jobs[job_id].status = BatchJobStatus.RUNNING
            self.jobs[job_id].started_at = datetime.now()

    def update_progress(
        self,
        job_id: str,
        completed_items: int,
        failed_items: int = 0,
    ) -> None:
        """Update job progress."""
        if job_id in self.jobs:
            self.jobs[job_id].completed_items = completed_items
            self.jobs[job_id].failed_items = failed_items

    def complete_job(
        self,
        job_id: str,
        results: List[Any],
        errors: Optional[List[Dict]] = None,
    ) -> None:
        """Mark job as completed."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = BatchJobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.results = results
            job.errors = errors or []
            job.completed_items = len(results)
            job.failed_items = len(errors) if errors else 0

            self.total_jobs_completed += 1

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.now()
            job.errors.append({"error": error, "timestamp": datetime.now().isoformat()})

            self.total_jobs_failed += 1

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job if it's pending or running.

        Returns:
            True if cancelled, False if not found or already completed
        """
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
            return False

        job.status = BatchJobStatus.CANCELLED
        job.completed_at = datetime.now()
        return True

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status as dictionary."""
        job = self.get_job(job_id)
        return job.to_dict() if job else None

    def get_job_results(self, job_id: str) -> Optional[Dict]:
        """Get job results."""
        job = self.get_job(job_id)
        if not job:
            return None

        return {
            "job_id": job_id,
            "status": job.status.value,
            "results": job.results,
            "errors": job.errors,
            "total_items": job.total_items,
            "completed_items": job.completed_items,
            "failed_items": job.failed_items,
        }

    def list_jobs(self, limit: int = 100) -> List[Dict]:
        """List recent jobs."""
        jobs = sorted(
            self.jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        return [j.to_dict() for j in jobs[:limit]]

    def _cleanup_old_jobs(self) -> None:
        """Remove completed jobs older than TTL."""
        now = datetime.now()
        to_remove = []

        for job_id, job in self.jobs.items():
            if job.status in [
                BatchJobStatus.COMPLETED,
                BatchJobStatus.FAILED,
                BatchJobStatus.CANCELLED,
            ]:
                if job.completed_at:
                    age_seconds = (now - job.completed_at).total_seconds()
                    if age_seconds > self.job_ttl_seconds:
                        to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]

    def get_stats(self) -> Dict:
        """Get manager statistics."""
        active_jobs = sum(1 for j in self.jobs.values() if j.status == BatchJobStatus.RUNNING)
        pending_jobs = sum(1 for j in self.jobs.values() if j.status == BatchJobStatus.PENDING)

        return {
            "total_jobs_created": self.total_jobs_created,
            "total_jobs_completed": self.total_jobs_completed,
            "total_jobs_failed": self.total_jobs_failed,
            "active_jobs": active_jobs,
            "pending_jobs": pending_jobs,
            "stored_jobs": len(self.jobs),
            "max_jobs": self.max_jobs,
            "job_ttl_seconds": self.job_ttl_seconds,
        }
