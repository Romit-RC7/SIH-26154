"""
Integration tests for the complete asynchronous document processing pipeline:
Upload -> Extraction -> PP-Structure/Fallback -> Semantic Fusion -> PostgreSQL Storage -> API Retrieval.
"""

from pathlib import Path
import pytest
from httpx import AsyncClient
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import pymupdf as fitz

from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_element import DocumentElement
from backend.app.models.processing_job import ProcessingJob, JobStatus, PipelineStep
from backend.app.services.pipeline_service import pipeline_service
from backend.app.services.storage_service import storage_service
from backend.app.schemas.semantic_document import SemanticDocument


@pytest.mark.asyncio
async def test_full_pipeline_service_execution(db_session: AsyncSession, tmp_path: Path):
    """
    Execute DocumentPipelineService end-to-end on a generated PDF with tables and text.
    Validates status transitions, semantic JSON generation, and relational element persistence.
    """
    # 1. Create realistic multi-modal test PDF
    doc = fitz.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 50), "Quarterly Financial Analysis", fontsize=16)
    page1.insert_text((50, 80), "This report evaluates quarterly metrics and projections.", fontsize=11)
    page1.insert_text((50, 120), "Table 1: Key Performance Metrics", fontsize=12)
    page1.insert_text((50, 140), "Metric | Q1 | Q2 | Target", fontsize=10)
    page1.insert_text((50, 160), "Revenue | $4.2M | $5.1M | $5.0M", fontsize=10)
    page1.insert_text((50, 180), "Retention | 92% | 95% | 94%", fontsize=10)

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 50), "Strategic Recommendations", fontsize=14)
    page2.insert_text((50, 80), "Expand enterprise tier and enhance multi-modal visual inference.", fontsize=11)

    test_pdf = tmp_path / "financial_audit.pdf"
    doc.save(str(test_pdf))
    doc.close()

    # 2. Insert initial Document and ProcessingJob records
    doc_id = "test-pipe-doc-1"
    job_id = "test-pipe-job-1"

    db_doc = Document(
        id=doc_id,
        filename="financial_audit.pdf",
        stored_path=str(test_pdf),
        file_size=test_pdf.stat().st_size,
        mime_type="application/pdf",
        page_count=0,
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

    # 3. Execute pipeline processing synchronously in test
    await pipeline_service.process_document(doc_id, job_id, session=db_session)

    # 4. Verify Document record in PostgreSQL
    refreshed_doc = (await db_session.execute(select(Document).where(Document.id == doc_id))).scalar_one()
    assert refreshed_doc.status == DocumentStatus.COMPLETED
    assert refreshed_doc.page_count == 2
    assert refreshed_doc.semantic_json is not None

    # Validate that semantic_json strictly adheres to the SemanticDocument contract
    semantic_contract = SemanticDocument.model_validate(refreshed_doc.semantic_json)
    assert semantic_contract.document_id == doc_id
    assert semantic_contract.metadata.file_name == "financial_audit.pdf"
    assert len(semantic_contract.elements) >= 3
    assert len(semantic_contract.sources) >= 1

    # 5. Verify relational DocumentElement records
    elements_query = await db_session.execute(
        select(DocumentElement).where(DocumentElement.document_id == doc_id).order_by(DocumentElement.element_index)
    )
    db_elements = elements_query.scalars().all()
    assert len(db_elements) == len(semantic_contract.elements)
    assert db_elements[0].type.value in ["text", "table", "figure", "chart", "image"]

    # 6. Verify ProcessingJob record
    refreshed_job = (await db_session.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))).scalar_one()
    assert refreshed_job.status == JobStatus.COMPLETED
    assert refreshed_job.step == PipelineStep.COMPLETED
    assert refreshed_job.started_at is not None
    assert refreshed_job.completed_at is not None
    assert refreshed_job.processing_metadata["elements_count"] == len(db_elements)

    # Cleanup artifacts
    storage_service.delete_document_artifacts(doc_id)


@pytest.mark.asyncio
async def test_api_upload_to_retrieval_integration(client: AsyncClient, db_session: AsyncSession, tmp_path: Path):
    """
    Test uploading a PDF via API, executing pipeline, and querying GET /documents/{id}/semantic.
    """
    # Create test PDF
    doc = fitz.open()
    p = doc.new_page(width=500, height=700)
    p.insert_text((50, 50), "API End-to-End Test Document", fontsize=14)
    p.insert_text((50, 80), "Verifying HTTP multipart upload to Semantic JSON retrieval.", fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()

    # 1. POST /api/v1/documents/upload
    upload_res = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("e2e_test.pdf", pdf_bytes, "application/pdf")}
    )
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["document_id"]
    job_id = upload_data["job_id"]

    # 2. Process pipeline for this document
    await pipeline_service.process_document(doc_id, job_id, session=db_session)

    # 3. GET /api/v1/documents/{id}
    status_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "COMPLETED"
    assert status_res.json()["element_count"] > 0

    # 4. GET /api/v1/documents/{id}/semantic
    semantic_res = await client.get(f"/api/v1/documents/{doc_id}/semantic")
    assert semantic_res.status_code == 200
    sem_json = semantic_res.json()
    assert sem_json["document_id"] == doc_id
    assert "API End-to-End Test Document" in sem_json["elements"][0]["content"]["text"]

    # Cleanup
    storage_service.delete_document_artifacts(doc_id)
