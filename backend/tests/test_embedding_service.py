"""
Integration tests for EmbeddingService.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk, ChunkType
from backend.app.schemas.semantic_document import (
    SemanticDocument,
    DocumentMetadata,
    SemanticElement,
    ElementContent,
)
from backend.app.services.embedding.embedding_service import embedding_service


@pytest.mark.asyncio
async def test_embed_and_store_document(db_session: AsyncSession):
    # 1. Create test document
    doc = Document(
        id="doc_embed_test_1",
        filename="test_report.pdf",
        stored_path="uploads/raw/test_report.pdf",
        file_size=1024,
        status=DocumentStatus.COMPLETED,
        page_count=1,
    )
    db_session.add(doc)
    await db_session.commit()

    # 2. Build semantic document
    semantic_doc = SemanticDocument(
        document_id="doc_embed_test_1",
        metadata=DocumentMetadata(
            file_name="test_report.pdf",
            file_size=1024,
            title="AI Benchmark Report",
            page_count=1
        ),
        elements=[
            SemanticElement(
                id="elem_1",
                type="text",
                page=1,
                content=ElementContent(
                    text="The new neural network model achieved 98.4% accuracy on the evaluation set.",
                    reading_order=1,
                    raw_attributes={"role": "paragraph"}
                )
            ),
            SemanticElement(
                id="elem_2",
                type="table",
                page=1,
                content=ElementContent(
                    markdown="| Model | Score |\n|---|---|\n| Baseline | 85% |\n| Proposed | 98.4% |",
                    caption="Model Benchmark Comparison",
                    reading_order=2
                )
            )
        ]
    )

    # 3. Run embedding service
    chunks = await embedding_service.embed_and_store_document(
        document_id="doc_embed_test_1",
        semantic_doc=semantic_doc,
        db=db_session
    )

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.document_id == "doc_embed_test_1"
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384

    # 4. Verify persisted in DB
    result = await db_session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == "doc_embed_test_1")
    )
    saved_chunks = result.scalars().all()
    assert len(saved_chunks) == 2


@pytest.mark.asyncio
async def test_embed_query_and_texts():
    query_vec = embedding_service.embed_query("machine learning benchmark")
    assert len(query_vec) == 384

    batch_vecs = embedding_service.embed_texts(["Text A", "Text B"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384
