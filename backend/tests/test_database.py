"""
Database layer tests for SQLAlchemy models, relationships, cascading deletes,
and JSONB storage.
"""

import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_element import DocumentElement, ElementType
from backend.app.models.processing_job import ProcessingJob, JobStatus, PipelineStep


@pytest.mark.asyncio
async def test_document_lifecycle(db_session: AsyncSession):
    """Verify document insertion, status transition, JSONB update, and retrieval."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="financials.pdf",
        stored_path="/uploads/raw/financials.pdf",
        file_size=512000,
        mime_type="application/pdf",
        page_count=5,
        status=DocumentStatus.PENDING,
        processing_metadata={"uploader": "admin"}
    )
    db_session.add(doc)
    await db_session.commit()

    # 1. Retrieve and verify
    query = await db_session.execute(select(Document).where(Document.id == doc_id))
    fetched_doc = query.scalar_one_or_none()
    assert fetched_doc is not None
    assert fetched_doc.filename == "financials.pdf"
    assert fetched_doc.status == DocumentStatus.PENDING

    # 2. Update status and attach semantic JSON
    fetched_doc.status = DocumentStatus.COMPLETED
    fetched_doc.semantic_json = {
        "document_id": doc_id,
        "metadata": {"title": "Financial Report 2026"},
        "elements": [{"id": "elem_1", "type": "text", "page": 1, "content": {"text": "Profit Margin: 24%"}}]
    }
    await db_session.commit()

    # Verify update
    query_updated = await db_session.execute(select(Document).where(Document.id == doc_id))
    updated_doc = query_updated.scalar_one()
    assert updated_doc.status == DocumentStatus.COMPLETED
    assert updated_doc.semantic_json["metadata"]["title"] == "Financial Report 2026"
    assert len(updated_doc.semantic_json["elements"]) == 1


@pytest.mark.asyncio
async def test_document_elements_and_cascade(db_session: AsyncSession):
    """Verify relational elements, foreign keys, and cascading deletion (no orphans)."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="cascade_test.pdf",
        stored_path="/uploads/raw/cascade_test.pdf",
        file_size=1024,
        mime_type="application/pdf",
        page_count=2,
        status=DocumentStatus.COMPLETED,
    )
    db_session.add(doc)
    await db_session.commit()

    # Add 3 elements
    elem1 = DocumentElement(
        id=f"elem_{doc_id[:8]}_1",
        document_id=doc_id,
        element_index=1,
        type=ElementType.TEXT,
        page=1,
        bbox=[10.0, 20.0, 300.0, 50.0],
        content={"text": "Introduction"}
    )
    elem2 = DocumentElement(
        id=f"elem_{doc_id[:8]}_2",
        document_id=doc_id,
        element_index=2,
        type=ElementType.TABLE,
        page=1,
        bbox=[10.0, 60.0, 400.0, 200.0],
        content={"markdown": "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |"}
    )
    elem3 = DocumentElement(
        id=f"elem_{doc_id[:8]}_3",
        document_id=doc_id,
        element_index=3,
        type=ElementType.FIGURE,
        page=2,
        bbox=[50.0, 50.0, 450.0, 300.0],
        content={"image_path": "uploads/extracted/elem3.png"}
    )
    db_session.add_all([elem1, elem2, elem3])
    await db_session.commit()

    # Verify elements count
    count_q = await db_session.execute(
        select(func.count(DocumentElement.id)).where(DocumentElement.document_id == doc_id)
    )
    assert count_q.scalar_one() == 3

    # Delete the parent document
    await db_session.delete(doc)
    await db_session.commit()

    # Verify no orphaned DocumentElement records remain
    post_del_q = await db_session.execute(
        select(func.count(DocumentElement.id)).where(DocumentElement.document_id == doc_id)
    )
    assert post_del_q.scalar_one() == 0


@pytest.mark.asyncio
async def test_processing_job_lifecycle(db_session: AsyncSession):
    """Verify ProcessingJob step transitions and cascade delete."""
    doc_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    doc = Document(
        id=doc_id,
        filename="job_test.pdf",
        stored_path="/uploads/raw/job_test.pdf",
        file_size=2048,
        status=DocumentStatus.PENDING,
    )
    job = ProcessingJob(
        id=job_id,
        document_id=doc_id,
        status=JobStatus.QUEUED,
        step=PipelineStep.INIT,
    )
    db_session.add_all([doc, job])
    await db_session.commit()

    # Step transitions
    job.status = JobStatus.PROCESSING
    job.step = PipelineStep.PARSING
    job.started_at = datetime.now(timezone.utc)
    await db_session.commit()

    job.step = PipelineStep.SEMANTIC_FUSION
    await db_session.commit()

    job.status = JobStatus.COMPLETED
    job.step = PipelineStep.COMPLETED
    job.completed_at = datetime.now(timezone.utc)
    job.processing_metadata = {"duration_sec": 1.25}
    await db_session.commit()

    # Verify
    fetched = (await db_session.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))).scalar_one()
    assert fetched.status == JobStatus.COMPLETED
    assert fetched.step == PipelineStep.COMPLETED
    assert fetched.processing_metadata["duration_sec"] == 1.25

    # Cascade test on job
    await db_session.delete(doc)
    await db_session.commit()

    orphan_job = (await db_session.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))).scalar_one_or_none()
    assert orphan_job is None
