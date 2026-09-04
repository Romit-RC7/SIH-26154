"""
Tests for Document List API pagination and status filtering.
"""

import uuid
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.document import Document, DocumentStatus


@pytest.mark.asyncio
async def test_list_documents_filtering_and_pagination(client: AsyncClient, db_session: AsyncSession):
    """Verify listing documents with status filters and offset/limit pagination."""
    # Insert 5 documents with different statuses
    for i in range(3):
        doc = Document(
            id=f"doc-completed-{i}",
            filename=f"comp_{i}.pdf",
            stored_path=f"/uploads/raw/comp_{i}.pdf",
            file_size=1000,
            status=DocumentStatus.COMPLETED,
        )
        db_session.add(doc)

    for i in range(2):
        doc = Document(
            id=f"doc-failed-{i}",
            filename=f"fail_{i}.pdf",
            stored_path=f"/uploads/raw/fail_{i}.pdf",
            file_size=1000,
            status=DocumentStatus.FAILED,
        )
        db_session.add(doc)

    await db_session.commit()

    # 1. Query all
    res_all = await client.get("/api/v1/documents?skip=0&limit=10")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert data_all["total"] == 5
    assert len(data_all["items"]) == 5

    # 2. Query filter status=COMPLETED
    res_comp = await client.get("/api/v1/documents?status=COMPLETED")
    assert res_comp.status_code == 200
    data_comp = res_comp.json()
    assert data_comp["total"] == 3
    assert all(item["status"] == "COMPLETED" for item in data_comp["items"])

    # 3. Query filter status=FAILED
    res_fail = await client.get("/api/v1/documents?status=FAILED")
    assert res_fail.status_code == 200
    data_fail = res_fail.json()
    assert data_fail["total"] == 2
    assert all(item["status"] == "FAILED" for item in data_fail["items"])

    # 4. Pagination limit=2, skip=1
    res_page = await client.get("/api/v1/documents?skip=1&limit=2")
    assert res_page.status_code == 200
    data_page = res_page.json()
    assert len(data_page["items"]) == 2
    assert data_page["skip"] == 1
    assert data_page["limit"] == 2
