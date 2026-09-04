"""
Pydantic Schemas for Document API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from backend.app.models.document import DocumentStatus
from backend.app.schemas.semantic_document import SemanticDocument


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_size: int
    mime_type: str
    page_count: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    element_count: Optional[int] = 0


class DocumentDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    stored_path: str
    file_size: int
    mime_type: str
    page_count: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    element_count: int = 0
    processing_metadata: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    job_id: str
    status: DocumentStatus
    filename: str


class DocumentListResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[DocumentSummary]
