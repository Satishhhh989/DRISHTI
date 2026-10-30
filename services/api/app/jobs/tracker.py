"""PHOTO → 3D ENGINE - Job Tracker
"""

import uuid
from typing import Dict, Optional
from app.events.emitter import event_emitter
from app.schemas.events import JobStatus, PipelineStageIdentifier, ProcessingEvent


class JobTracker:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}

    def create_job(self, scene_id: str) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "scene_id": scene_id,
            "status": "pending",
            "current_stage": "image_analysis",
        }
        # Emit initial acceptance event
        event_emitter.emit(
            job_id=job_id,
            scene_id=scene_id,
            event_type="job_accepted",
            stage_identifier="image_analysis",
            status="pending",
            message="Processing job queued for execution"
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self._jobs.get(job_id)


job_tracker = JobTracker()
