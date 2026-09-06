"""
Unit tests for DocumentChunker.
"""

import pytest
from backend.app.schemas.semantic_document import (
    SemanticDocument,
    DocumentMetadata,
    SemanticElement,
    ElementContent,
)
from backend.app.models.document_chunk import ChunkType
from backend.app.services.embedding.chunker import DocumentChunker


@pytest.fixture
def sample_semantic_doc() -> SemanticDocument:
    return SemanticDocument(
        document_id="doc_test_123",
        metadata=DocumentMetadata(
            file_name="financial_report.pdf",
            file_size=2048,
            title="Q3 Financial Overview",
            page_count=2
        ),
        elements=[
            SemanticElement(
                id="elem_title_1",
                type="text",
                page=1,
                bbox=[50, 50, 500, 80],
                content=ElementContent(
                    text="Q3 2026 Financial Results and Strategic Overview",
                    confidence=0.99,
                    reading_order=1,
                    raw_attributes={"role": "title"}
                )
            ),
            SemanticElement(
                id="elem_para_1",
                type="text",
                page=1,
                bbox=[50, 100, 500, 300],
                content=ElementContent(
                    text="Revenue increased by 24.5% year-over-year reaching $4.8 billion. Operating margin expanded to 32.1%. Strategic investments in AI infrastructure accelerated platform delivery across all core enterprise segments.",
                    confidence=0.95,
                    reading_order=2,
                    raw_attributes={"role": "paragraph"}
                )
            ),
            SemanticElement(
                id="elem_table_1",
                type="table",
                page=2,
                bbox=[50, 50, 500, 250],
                content=ElementContent(
                    text="Table data",
                    markdown="| Segment | Revenue | Growth |\n|---|---|---|\n| Cloud | $2.4B | +35% |\n| AI Services | $1.2B | +65% |\n| Enterprise | $1.2B | +10% |",
                    caption="Segment Revenue Breakdown Q3",
                    confidence=0.98,
                    reading_order=3,
                )
            ),
            SemanticElement(
                id="elem_chart_1",
                type="chart",
                page=2,
                bbox=[50, 300, 500, 500],
                content=ElementContent(
                    caption="Quarterly Revenue Trajectory",
                    image_path="uploads/extracted/doc_test_123/elem_chart_1.png",
                    confidence=0.92,
                    reading_order=4,
                    raw_attributes={"visual_analysis": "Upward trajectory from Q1 $3.2B to Q3 $4.8B."}
                )
            )
        ]
    )


def test_chunk_document_generates_correct_types(sample_semantic_doc: SemanticDocument):
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(sample_semantic_doc)

    assert len(chunks) >= 4
    types = [ch.chunk_type for ch in chunks]
    assert ChunkType.HEADER in types
    assert ChunkType.TEXT in types
    assert ChunkType.TABLE in types
    assert ChunkType.CHART_DATA in types


def test_chunk_document_metadata_and_prefix(sample_semantic_doc: SemanticDocument):
    chunker = DocumentChunker(chunk_size=500, include_context_prefix=True)
    chunks = chunker.chunk_document(sample_semantic_doc)

    title_chunk = next(ch for ch in chunks if ch.element_id == "elem_title_1")
    assert title_chunk.page == 1
    assert "[Document: Q3 Financial Overview | Page 1 | Type: title]" in title_chunk.cleaned_text
    assert "Q3 2026 Financial Results" in title_chunk.cleaned_text


def test_chunk_table_preserves_markdown_and_caption(sample_semantic_doc: SemanticDocument):
    chunker = DocumentChunker(chunk_size=500)
    chunks = chunker.chunk_document(sample_semantic_doc)

    table_chunk = next(ch for ch in chunks if ch.chunk_type == ChunkType.TABLE)
    assert table_chunk.page == 2
    assert "Cloud" in table_chunk.cleaned_text
    assert "AI Services" in table_chunk.cleaned_text
    assert table_chunk.chunk_metadata.get("caption") == "Segment Revenue Breakdown Q3"


def test_chunk_large_text_splits_with_overlap():
    long_text = "Sentence one is clear and simple. " * 30
    elem = SemanticElement(
        id="elem_long_1",
        type="text",
        page=1,
        content=ElementContent(text=long_text, raw_attributes={"role": "paragraph"})
    )
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_element(elem, doc_title="Test Doc")

    assert len(chunks) > 1
    for idx, ch in enumerate(chunks):
        assert ch.element_id == "elem_long_1"
        assert len(ch.content) <= 300
