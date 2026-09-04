"""
Unified Semantic Document Schema (System Contract).
All future AI modules (Embedding Engine, Knowledge Engine, Qwen2.5-VL,
Content Orchestrator, LLM Generation) consume this unified structure.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    FIGURE = "figure"


class DocumentMetadata(BaseModel):
    """Metadata describing the origin document."""
    file_name: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    mime_type: str = Field(default="application/pdf", description="MIME type")
    page_count: int = Field(default=1, ge=1, description="Total number of pages")
    title: Optional[str] = Field(default=None, description="Inferred or extracted document title")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Upload timestamp (UTC)")
    sha256: Optional[str] = Field(default=None, description="SHA-256 hash of the original file")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Custom document level metadata")


class ElementContent(BaseModel):
    """
    Flexible content payload supporting text, tables, figures, charts, and diagrams.
    """
    text: Optional[str] = Field(default=None, description="Extracted raw text or OCR text")
    markdown: Optional[str] = Field(default=None, description="Markdown formatted content (e.g. formatted table)")
    html: Optional[str] = Field(default=None, description="HTML formatted representation (e.g. table cells)")
    image_path: Optional[str] = Field(default=None, description="Path to cropped figure/chart image file")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="OCR/Recognition confidence score")
    reading_order: Optional[int] = Field(default=None, description="Natural reading order sequence")
    caption: Optional[str] = Field(default=None, description="Associated figure/table caption")
    table_structure: Optional[Dict[str, Any]] = Field(default=None, description="Structured cell/row matrix if table")
    raw_attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Engine specific raw attributes")


class SemanticElement(BaseModel):
    """Individual atomic document element."""
    id: str = Field(..., description="Deterministic or unique element identifier")
    type: Literal["text", "table", "image", "chart", "figure"] = Field(
        ...,
        description="Categorized element type"
    )
    page: int = Field(..., ge=1, description="1-based page number where element appears")
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Bounding box [x1, y1, x2, y2] in points or pixels"
    )
    content: ElementContent = Field(..., description="Element content payload")


# Future Contract Stubs (Phase 2-4: Populated by Knowledge Engine / Qwen2.5-VL)
class EntityItem(BaseModel):
    """Named entity recognized across document elements."""
    id: str = Field(..., description="Entity identifier")
    name: str = Field(..., description="Entity surface name")
    category: str = Field(..., description="Entity category (e.g., PERSON, ORG, METRIC, EVENT)")
    mentions: List[str] = Field(default_factory=list, description="IDs of elements referencing this entity")
    confidence: Optional[float] = Field(default=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ClaimItem(BaseModel):
    """Extracted factual claim or proposition."""
    id: str = Field(..., description="Claim identifier")
    statement: str = Field(..., description="Proposition statement")
    source_element_ids: List[str] = Field(default_factory=list, description="Evidence elements supporting this claim")
    confidence: Optional[float] = Field(default=1.0)


class RelationshipItem(BaseModel):
    """Semantic graph triple or relationship."""
    id: str = Field(..., description="Relationship identifier")
    subject_id: str = Field(..., description="Subject entity ID")
    predicate: str = Field(..., description="Relation / predicate type")
    object_id: str = Field(..., description="Object entity ID or literal value")
    confidence: Optional[float] = Field(default=1.0)


class SourceReference(BaseModel):
    """Source reference / citation tracking."""
    id: str = Field(..., description="Source reference ID")
    title: Optional[str] = Field(default=None)
    page: Optional[int] = Field(default=None)
    citation: Optional[str] = Field(default=None)
    url: Optional[str] = Field(default=None)


class SemanticDocument(BaseModel):
    """
    Unified Semantic Document JSON Contract.
    Strictly adheres to the SIH architecture specification.
    """
    version: str = Field(default="1.0.0", description="Semantic Document Schema Version")
    document_id: str = Field(..., description="Unique document ID (UUID)")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    elements: List[SemanticElement] = Field(
        default_factory=list,
        description="Ordered list of extracted layout elements"
    )
    entities: List[EntityItem] = Field(
        default_factory=list,
        description="Extracted entities (Phase 3 Knowledge Engine)"
    )
    claims: List[ClaimItem] = Field(
        default_factory=list,
        description="Extracted claims/facts (Phase 3 Knowledge Engine)"
    )
    relationships: List[RelationshipItem] = Field(
        default_factory=list,
        description="Extracted knowledge graph edges (Phase 3 Knowledge Engine)"
    )
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Source tracking and citations"
    )
