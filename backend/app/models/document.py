"""
SQLAlchemy Model for Document.
Stores uploaded document metadata, processing status, and the unified Semantic Document JSON.
"""

import enum
import uuid
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, Enum, JSON
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.document_element import DocumentElement
    from backend.app.models.processing_job import ProcessingJob
    from backend.app.models.document_chunk import DocumentChunk


class DocumentStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False, default="application/pdf")
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, native_enum=False),
        nullable=False,
        default=DocumentStatus.PENDING,
        index=True
    )
    
    # Store complete Semantic Document JSON representation
    # PostgreSQL JSONB is used when available, falling back to JSON
    semantic_json: Mapped[Optional[dict]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=True
    )
    
    # Extra processing or document metadata
    processing_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=True,
        default=dict
    )

    # Relationships
    elements: Mapped[List["DocumentElement"]] = relationship(
        "DocumentElement",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentElement.element_index"
    )
    
    jobs: Mapped[List["ProcessingJob"]] = relationship(
        "ProcessingJob",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="desc(ProcessingJob.created_at)"
    )

    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename='{self.filename}' status={self.status}>"
