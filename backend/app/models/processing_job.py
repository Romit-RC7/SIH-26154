"""
SQLAlchemy Model for ProcessingJob.
Tracks execution states, processing pipeline steps, timings, and error telemetry.
"""

import enum
import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Enum, ForeignKey, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.document import Document


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PipelineStep(str, enum.Enum):
    INIT = "INIT"
    PARSING = "PARSING"
    STRUCTURE_ANALYSIS = "STRUCTURE_ANALYSIS"
    EXTRACTION = "EXTRACTION"
    SEMANTIC_FUSION = "SEMANTIC_FUSION"
    PERSISTENCE = "PERSISTENCE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProcessingJob(Base, TimestampMixin):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True
    )
    step: Mapped[PipelineStep] = mapped_column(
        Enum(PipelineStep, native_enum=False),
        nullable=False,
        default=PipelineStep.INIT
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    processing_metadata: Mapped[dict] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=False,
        default=dict
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="jobs"
    )

    def __repr__(self) -> str:
        return f"<ProcessingJob id={self.id} doc={self.document_id} status={self.status} step={self.step}>"
