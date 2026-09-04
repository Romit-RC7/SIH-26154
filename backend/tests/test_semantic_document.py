"""
Unit tests for the Unified Semantic Document Schema (System Contract)
and SemanticDocumentBuilder.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

from backend.app.schemas.semantic_document import (
    SemanticDocument,
    SemanticElement,
    ElementContent,
    DocumentMetadata,
    EntityItem,
    ClaimItem,
    RelationshipItem,
    SourceReference,
)
from backend.app.processors.base import RawDocumentElement
from backend.app.services.semantic_builder import semantic_document_builder
from backend.app.services.semantic_fusion import semantic_fusion_engine


def test_valid_semantic_document_schema():
    """Verify standard contract construction and serialization."""
    doc = SemanticDocument(
        document_id="doc-1234-test",
        metadata=DocumentMetadata(
            file_name="report.pdf",
            file_size=204800,
            mime_type="application/pdf",
            page_count=2,
            title="Q3 Performance Report",
            created_at=datetime.now(timezone.utc),
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        elements=[
            SemanticElement(
                id="elem_1",
                type="text",
                page=1,
                bbox=[50.0, 50.0, 500.0, 100.0],
                content=ElementContent(
                    text="Executive Overview",
                    reading_order=1,
                    confidence=0.99
                )
            ),
            SemanticElement(
                id="elem_2",
                type="table",
                page=1,
                bbox=[50.0, 120.0, 500.0, 300.0],
                content=ElementContent(
                    markdown="| Revenue | Costs |\n| --- | --- |\n| $1M | $600K |",
                    html="<table><tr><th>Revenue</th><th>Costs</th></tr><tr><td>$1M</td><td>$600K</td></tr></table>",
                    reading_order=2,
                    caption="Table 1: Financial summary"
                )
            ),
            SemanticElement(
                id="elem_3",
                type="figure",
                page=2,
                bbox=[50.0, 50.0, 450.0, 350.0],
                content=ElementContent(
                    image_path="uploads/extracted/doc-1234/elem_3.png",
                    caption="Figure 1: Growth Trend 2026",
                    reading_order=3
                )
            )
        ],
        entities=[
            EntityItem(id="ent_1", name="SIH-2026", category="ORGANIZATION")
        ],
        claims=[
            ClaimItem(id="clm_1", statement="Revenue reached $1M", source_element_ids=["elem_2"])
        ],
        relationships=[
            RelationshipItem(id="rel_1", subject_id="ent_1", predicate="HOSTS", object_id="Hackathon")
        ],
        sources=[
            SourceReference(id="src_1", title="report.pdf")
        ]
    )

    doc_dict = doc.model_dump(mode="json")
    assert doc_dict["document_id"] == "doc-1234-test"
    assert len(doc_dict["elements"]) == 3
    assert doc_dict["elements"][0]["type"] == "text"
    assert doc_dict["elements"][1]["type"] == "table"
    assert doc_dict["elements"][2]["type"] == "figure"
    assert len(doc_dict["entities"]) == 1
    assert len(doc_dict["claims"]) == 1
    assert len(doc_dict["relationships"]) == 1
    assert len(doc_dict["sources"]) == 1


def test_invalid_element_type_rejected():
    """Verify that unsupported element types fail schema validation."""
    with pytest.raises(ValidationError):
        SemanticElement(
            id="elem_bad",
            type="audio_clip",  # Not in allowed literal
            page=1,
            content=ElementContent(text="Invalid")
        )


def test_semantic_document_builder(tmp_path: Path):
    """Test full builder assembly from raw extracted elements."""
    dummy_file = tmp_path / "sample.pdf"
    dummy_file.write_bytes(b"%PDF-1.4 test content")

    raw_elems = [
        RawDocumentElement(
            type="text",
            page=1,
            bbox=[50.0, 80.0, 400.0, 100.0],
            text="Introductory text",
            confidence=0.98
        ),
        RawDocumentElement(
            type="figure",
            page=1,
            bbox=[50.0, 120.0, 400.0, 300.0],
            confidence=0.95
        ),
        RawDocumentElement(
            type="text",
            page=1,
            bbox=[50.0, 310.0, 400.0, 330.0],
            text="Figure 1: System Flowchart",
            confidence=0.99
        )
    ]

    semantic_doc = semantic_document_builder.build(
        document_id="test-doc-uuid",
        file_path=dummy_file,
        raw_elements=raw_elems,
        extraction_metadata={"page_count": 1, "title": "Test Paper"}
    )

    assert semantic_doc.document_id == "test-doc-uuid"
    assert semantic_doc.metadata.title == "Test Paper"
    assert len(semantic_doc.elements) == 3
    # Check that caption association fused "Figure 1: System Flowchart" to the figure element
    figure_elem = next(e for e in semantic_doc.elements if e.type == "figure")
    assert figure_elem.content.caption == "Figure 1: System Flowchart"
