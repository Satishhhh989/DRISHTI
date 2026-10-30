"""PHOTO → 3D ENGINE - Job Tracker
Version 1.2.0 (Phase 04: Progressive Engine & Cancellation)
"""

import uuid
from typing import Dict, Optional
from app.events.emitter import event_emitter
from app.schemas.events import JobStatus, PipelineStageIdentifier


class JobTracker:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}

    def create_job(self, scene_id: str) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "scene_id": scene_id,
            "status": "pending",
            "current_stage": "image_validation",
            "cancelled": False,
        }
        # Emit initial acceptance event
        event_emitter.emit(
            job_id=job_id,
            scene_id=scene_id,
            event_type="job_accepted",
            stage_identifier="image_validation",
            status="pending",
            message="Processing job queued for execution",
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancels an active or pending job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.get("status") in ("completed", "failed", "cancelled"):
            return False

        job["cancelled"] = True
        job["status"] = "cancelled"

        event_emitter.emit(
            job_id=job_id,
            scene_id=job["scene_id"],
            event_type="job_cancelled",
            stage_identifier=job.get("current_stage", "completed"),
            status="cancelled",
            message="Processing job cancelled by user intervention",
        )
        return True

    def is_cancelled(self, job_id: str) -> bool:
        """Checks if a job has been cancelled."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        return bool(job.get("cancelled", False))

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        current_stage: Optional[PipelineStageIdentifier] = None,
    ) -> None:
        """Updates the tracked status and current stage of a job."""
        job = self._jobs.get(job_id)
        if not job:
            return
        job["status"] = status
        if current_stage:
            job["current_stage"] = current_stage


job_tracker = JobTracker()
