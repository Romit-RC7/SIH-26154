"""
SQLAlchemy Model for DocumentElement.
Stores relational representations of extracted layout elements (text, table, image, chart, figure).
"""

import enum
import uuid
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, Enum, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.document import Document


class ElementType(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    FIGURE = "figure"
    AUDIO = "audio"


class DocumentElement(Base, TimestampMixin):
    __tablename__ = "document_elements"

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
    element_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type: Mapped[ElementType] = mapped_column(
        Enum(ElementType, native_enum=False),
        nullable=False,
        index=True
    )
    page: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    
    # Bounding Box: [x1, y1, x2, y2]
    bbox: Mapped[Optional[list]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=True
    )
    
    # Normalized content object containing text, markdown, html, image_path, confidence, etc.
    content: Mapped[dict] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=False,
        default=dict
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="elements"
    )

    def __repr__(self) -> str:
        return f"<DocumentElement id={self.id} doc={self.document_id} type={self.type} page={self.page}>"
