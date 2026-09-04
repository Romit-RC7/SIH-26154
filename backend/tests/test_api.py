"""
Integration tests for FastAPI endpoints.
Tests upload, document retrieval, semantic JSON retrieval, list, and health check.
"""

import pytest
from httpx import AsyncClient
from backend.app.models.document import Document, DocumentStatus
from backend.app.schemas.semantic_document import (
    SemanticDocument,
    DocumentMetadata,
    SemanticElement,
    ElementContent,
)
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Verify health endpoint returns 200 OK and expected keys."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


@pytest.mark.asyncio
async def test_upload_invalid_extension(client: AsyncClient):
    """Verify that unsupported extensions are rejected with 400."""
    files = {"file": ("malicious.exe", b"binary content", "application/octet-stream")}
    response = await client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_valid_pdf(client: AsyncClient, sample_pdf_bytes: bytes):
    """Verify PDF upload creates document record and enqueues job."""
    files = {"file": ("test_report.pdf", sample_pdf_bytes, "application/pdf")}
    response = await client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert "job_id" in data
    assert data["filename"] == "test_report.pdf"


@pytest.mark.asyncio
async def test_get_document_and_semantic_json(client: AsyncClient, db_session):
    """Verify retrieving document metadata and semantic JSON contract."""
    # Pre-populate a completed document in the database
    doc_id = "doc-completed-test-123"

    sample_semantic = SemanticDocument(
        document_id=doc_id,
        metadata=DocumentMetadata(
            file_name="completed.pdf",
            file_size=1024,
            page_count=1,
            title="Completed Doc",
            created_at=datetime.now(timezone.utc)
        ),
        elements=[
            SemanticElement(
                id="elem_1",
                type="text",
                page=1,
                content=ElementContent(text="Heading Text", reading_order=1)
            )
        ]
    )

    doc = Document(
        id=doc_id,
        filename="completed.pdf",
        stored_path="uploads/raw/completed.pdf",
        file_size=1024,
        mime_type="application/pdf",
        page_count=1,
        status=DocumentStatus.COMPLETED,
        semantic_json=sample_semantic.model_dump(mode="json")
    )
    db_session.add(doc)
    await db_session.commit()

    # 1. Test GET /documents/{id}
    res = await client.get(f"/api/v1/documents/{doc_id}")
    assert res.status_code == 200
    doc_data = res.json()
    assert doc_data["id"] == doc_id
    assert doc_data["status"] == "COMPLETED"

    # 2. Test GET /documents/{id}/semantic
    semantic_res = await client.get(f"/api/v1/documents/{doc_id}/semantic")
    assert semantic_res.status_code == 200
    sem_data = semantic_res.json()
    assert sem_data["document_id"] == doc_id
    assert len(sem_data["elements"]) == 1
    assert sem_data["elements"][0]["content"]["text"] == "Heading Text"

    # 3. Test GET /documents (List)
    list_res = await client.get("/api/v1/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == doc_id for item in list_data["items"])


@pytest.mark.asyncio
async def test_get_nonexistent_document(client: AsyncClient):
    """Verify 404 for unknown document ID."""
    res = await client.get("/api/v1/documents/non-existent-uuid")
    assert res.status_code == 404
