"""
Unit and integration tests for RetrievalService.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk, ChunkType
from backend.app.services.model_initializer.bge_initializer import bge_initializer
from backend.app.services.retrieval_service import retrieval_service


@pytest.mark.asyncio
async def test_retrieval_search_ranks_relevant_chunks(db_session: AsyncSession):
    doc_id = "doc_retrieval_test_1"
    doc = Document(
        id=doc_id,
        filename="market_report.pdf",
        stored_path="uploads/raw/market_report.pdf",
        file_size=1024,
        status=DocumentStatus.COMPLETED,
    )
    db_session.add(doc)

    # Pre-generate embeddings for test chunks
    emb_finance = bge_initializer.encode(["Financial quarterly earnings and revenue growth increased by 20%"])[0]
    emb_sports = bge_initializer.encode(["The football team won the championship final in overtime"])[0]

    chunk1 = DocumentChunk(
        id="ch_1",
        document_id=doc_id,
        element_id="elem_1",
        chunk_index=0,
        chunk_type=ChunkType.TEXT,
        page=1,
        content="Financial quarterly earnings and revenue growth increased by 20%",
        cleaned_text="Financial quarterly earnings and revenue growth increased by 20%",
        embedding=emb_finance
    )
    chunk2 = DocumentChunk(
        id="ch_2",
        document_id=doc_id,
        element_id="elem_2",
        chunk_index=1,
        chunk_type=ChunkType.TEXT,
        page=2,
        content="The football team won the championship final in overtime",
        cleaned_text="The football team won the championship final in overtime",
        embedding=emb_sports
    )
    db_session.add_all([chunk1, chunk2])
    await db_session.commit()

    # Search for financial terms
    results = await retrieval_service.search(
        query="quarterly revenue growth",
        db=db_session,
        document_id=doc_id,
        top_k=2
    )

    assert len(results) >= 1
    # Top result should be the financial chunk
    assert results[0].id == "ch_1"
    assert results[0].similarity_score > 0.0


@pytest.mark.asyncio
async def test_retrieval_search_with_filters(db_session: AsyncSession):
    doc_id = "doc_filter_test"
    doc = Document(
        id=doc_id,
        filename="test.pdf",
        stored_path="uploads/raw/test.pdf",
        file_size=500,
        status=DocumentStatus.COMPLETED,
    )
    db_session.add(doc)

    emb = bge_initializer.encode(["Table showing user statistics"])[0]
    chunk_tbl = DocumentChunk(
        id="ch_tbl",
        document_id=doc_id,
        chunk_index=0,
        chunk_type=ChunkType.TABLE,
        page=3,
        content="| User | Count |",
        cleaned_text="| User | Count |",
        embedding=emb
    )
    db_session.add(chunk_tbl)
    await db_session.commit()

    # Filter for table type
    results = await retrieval_service.search(
        query="user statistics",
        db=db_session,
        document_id=doc_id,
        chunk_types=[ChunkType.TABLE]
    )
    assert len(results) == 1
    assert results[0].chunk_type == ChunkType.TABLE

    # Filter for non-matching chunk type
    empty_results = await retrieval_service.search(
        query="user statistics",
        db=db_session,
        document_id=doc_id,
        chunk_types=[ChunkType.FIGURE_CAPTION]
    )
    assert len(empty_results) == 0
