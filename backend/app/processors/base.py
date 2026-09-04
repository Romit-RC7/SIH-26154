"""
Base definitions and abstract contracts for document structure analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from PIL import Image


@dataclass
class RawDocumentElement:
    """Intermediate extracted element before semantic fusion."""
    type: str  # text, table, image, chart, figure
    page: int
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    text: Optional[str] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    confidence: Optional[float] = None
    image: Optional[Image.Image] = None  # In-memory crop
    caption: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPage:
    """Rasterized page image and text representation."""
    page_number: int
    width: float
    height: float
    image: Optional[Image.Image] = None
    raw_text: Optional[str] = None


@dataclass
class DocumentParseResult:
    """Complete output of document parsing and structure analysis."""
    document_id: str
    file_path: str
    file_name: str
    page_count: int
    title: Optional[str] = None
    pages: List[ParsedPage] = field(default_factory=list)
    elements: List[RawDocumentElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStructureAnalyzer(ABC):
    """Abstract interface for structure analysis engines (PP-StructureV3, etc.)."""

    @abstractmethod
    def analyze_page(
        self,
        page_image: Image.Image,
        page_number: int,
        page_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RawDocumentElement]:
        """
        Analyze a single page image and return recognized layout elements.
        """
        pass
