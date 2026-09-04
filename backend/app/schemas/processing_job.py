"""
Pydantic Schemas for Processing Job status.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict
from backend.app.models.processing_job import JobStatus, PipelineStep


class ProcessingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    status: JobStatus
    step: PipelineStep
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_metadata: Dict[str, Any] = {}
