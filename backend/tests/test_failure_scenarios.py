"""
Failure, Edge Case, and Resilience Tests (Task 7):
Verifies graceful error handling, proper HTTP status codes, error logging, and pipeline fault tolerance.
"""

from pathlib import Path
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import Document, DocumentStatus
from backend.app.models.processing_job import ProcessingJob, JobStatus, PipelineStep
from backend.app.services.pipeline_service import pipeline_service
from backend.app.services.storage_service import storage_service
from backend.app.core.config import settings


@pytest.mark.asyncio
async def test_upload_empty_filename(client: AsyncClient):
    """Verify upload fails with 400 when filename is empty."""
    files = {"file": ("", b"content", "application/pdf")}
    res = await client.post("/api/v1/documents/upload", files=files)
    assert res.status_code in (400, 422)


@pytest.mark.asyncio
async def test_upload_unsupported_extension(client: AsyncClient):
    """Verify upload fails with 400 when file type is not supported."""
    files = {"file": ("script.py", b"print('hello')", "text/x-python")}
    res = await client.post("/api/v1/documents/upload", files=files)
    assert res.status_code == 400
    assert "Unsupported file extension" in res.json()["detail"]


@pytest.mark.asyncio
async def test_upload_file_exceeding_size_limit(client: AsyncClient, monkeypatch):
    """Verify upload rejects files larger than MAX_UPLOAD_SIZE_MB with 413."""
    # Temporarily set max size to 1MB for test
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

    large_payload = b"0" * (1024 * 1024 + 1024)  # 1MB + 1KB
    files = {"file": ("oversized.pdf", large_payload, "application/pdf")}
    res = await client.post("/api/v1/documents/upload", files=files)
    assert res.status_code == 413
    assert "exceeds maximum allowed size" in res.json()["detail"]


@pytest.mark.asyncio
async def test_get_semantic_json_while_processing(client: AsyncClient, db_session: AsyncSession):
    """Verify GET /semantic returns 202 Accepted with status when document is still processing."""
    doc_id = "doc-still-processing"
    doc = Document(
        id=doc_id,
        filename="in_progress.pdf",
        stored_path="uploads/raw/in_progress.pdf",
        file_size=2048,
        status=DocumentStatus.PROCESSING,
    )
    db_session.add(doc)
    await db_session.commit()

    res = await client.get(f"/api/v1/documents/{doc_id}/semantic")
    assert res.status_code == 202
    data = res.json()["detail"]
    assert data["status"] == "PROCESSING"
    assert "still being processed" in data["message"]


@pytest.mark.asyncio
async def test_get_semantic_json_when_failed(client: AsyncClient, db_session: AsyncSession):
    """Verify GET /semantic returns 422 with error telemetry when processing failed."""
    doc_id = "doc-processing-failed"
    doc = Document(
        id=doc_id,
        filename="bad.pdf",
        stored_path="uploads/raw/bad.pdf",
        file_size=2048,
        status=DocumentStatus.FAILED,
        processing_metadata={"error": "Corrupted header"}
    )
    db_session.add(doc)
    await db_session.commit()

    res = await client.get(f"/api/v1/documents/{doc_id}/semantic")
    assert res.status_code == 422
    data = res.json()["detail"]
    assert data["status"] == "FAILED"
    assert "failed" in data["message"]


@pytest.mark.asyncio
async def test_pipeline_failure_on_corrupted_file(db_session: AsyncSession, tmp_path: Path):
    """
    Verify pipeline execution gracefully marks document and job as FAILED
    with error telemetry when given an unparseable/corrupted file.
    """
    bad_file = tmp_path / "damaged.pdf"
    bad_file.write_bytes(b"%PDF corrupted non-standard binary bytes 999999")

    doc_id = "doc-corrupted-pipe-test"
    job_id = "job-corrupted-pipe-test"

    db_doc = Document(
        id=doc_id,
        filename="damaged.pdf",
        stored_path=str(bad_file),
        file_size=bad_file.stat().st_size,
        status=DocumentStatus.PENDING,
    )
    db_job = ProcessingJob(
        id=job_id,
        document_id=doc_id,
        status=JobStatus.QUEUED,
        step=PipelineStep.INIT,
    )
    db_session.add_all([db_doc, db_job])
    await db_session.commit()

    # Run pipeline
    await pipeline_service.process_document(doc_id, job_id, session=db_session)

    # Document and Job must be marked FAILED, no unhandled exception
    refreshed_doc = (await db_session.execute(select(Document).where(Document.id == doc_id))).scalar_one()
    refreshed_job = (await db_session.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))).scalar_one()

    assert refreshed_doc.status == DocumentStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED
    assert refreshed_job.step == PipelineStep.FAILED
    assert refreshed_job.error_message is not None
    assert refreshed_job.completed_at is not None


@pytest.mark.asyncio
async def test_pipeline_missing_records_graceful(db_session: AsyncSession):
    """Verify pipeline handles non-existent IDs cleanly without throwing unhandled exceptions."""
    # Should log error and return cleanly
    await pipeline_service.process_document("non-existent-doc", "non-existent-job", session=db_session)
