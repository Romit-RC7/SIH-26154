"""
Semantic Document Builder Service.
Constructs, validates, and finalizes the unified Semantic Document JSON (System Contract).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.processors.base import RawDocumentElement
from backend.app.services.semantic_fusion import semantic_fusion_engine
from backend.app.schemas.semantic_document import (
    SemanticDocument,
    DocumentMetadata,
)
from backend.app.utils.file_utils import compute_sha256, detect_mime_type
from backend.app.core.logging import logger


class SemanticDocumentBuilder:
    """
    Builds and validates the unified Semantic Document JSON representation
    from normalized extracted layout elements.
    """

    def build(
        self,
        document_id: str,
        file_path: Path,
        raw_elements: List[RawDocumentElement],
        extraction_metadata: Optional[Dict[str, Any]] = None
    ) -> SemanticDocument:
        """
        Assembles all components into the validated Semantic Document system contract.
        """
        meta = extraction_metadata or {}
        file_name = file_path.name
        file_size = file_path.stat().st_size if file_path.exists() else 0
        mime_type = detect_mime_type(file_name)
        sha256 = compute_sha256(file_path) if file_path.exists() else None
        page_count = meta.get("page_count", 1)
        title = meta.get("title") or file_path.stem

        # 1. Fuse layout elements, reading order, captions, and source references
        elements, sources = semantic_fusion_engine.fuse_elements(
            raw_elements=raw_elements,
            document_id=document_id,
            file_name=file_name
        )

        # 2. Build Document Metadata
        metadata = DocumentMetadata(
            file_name=file_name,
            file_size=file_size,
            mime_type=mime_type,
            page_count=page_count,
            title=title,
            created_at=datetime.now(timezone.utc),
            sha256=sha256,
            extra={
                "extracted_elements": len(elements),
                "author": meta.get("author"),
                "creator": meta.get("creator"),
            }
        )

        # 3. Instantiate and validate against Semantic Document Contract
        semantic_doc = SemanticDocument(
            version="1.0.0",
            document_id=document_id,
            metadata=metadata,
            elements=elements,
            entities=[],         # Ready for Phase 3 Knowledge Engine
            claims=[],           # Ready for Phase 3 Knowledge Engine
            relationships=[],    # Ready for Phase 3 Knowledge Engine
            sources=sources
        )

        logger.info(
            f"Successfully built Semantic Document JSON for doc_id={document_id} "
            f"({len(semantic_doc.elements)} elements)"
        )
        return semantic_doc


semantic_document_builder = SemanticDocumentBuilder()
