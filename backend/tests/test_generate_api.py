"""
API tests for Content Generation endpoints (POST /api/v1/generate/{document_id}).
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.document import Document, DocumentStatus


@pytest.mark.asyncio
async def test_generate_api_success(
    client: AsyncClient,
    db_session: AsyncSession
):
    doc_id = "doc_api_gen_1"
    doc = Document(
        id=doc_id,
        filename="threat_advisory.pdf",
        stored_path="uploads/raw/threat_advisory.pdf",
        file_size=2048,
        status=DocumentStatus.COMPLETED,
        page_count=1,
        semantic_json={
            "version": "1.0.0",
            "document_id": doc_id,
            "metadata": {
                "file_name": "threat_advisory.pdf",
                "file_size": 2048,
                "title": "Cyber Threat Intelligence Briefing",
                "page_count": 1,
            },
            "elements": [
                {
                    "id": "elem_th_1",
                    "type": "text",
                    "page": 1,
                    "content": {
                        "text": "Critical zero-day vulnerability detected in network edge routers. Immediate patching required.",
                        "confidence": 0.99,
                        "reading_order": 1,
                    }
                }
            ]
        }
    )
    db_session.add(doc)
    await db_session.commit()

    # Call POST /api/v1/generate/{doc_id}
    payload = {
        "output_types": ["linkedin_post", "twitter_thread", "executive_summary"],
        "audience": "technical",
        "tone": "authoritative",
        "detail_level": "moderate",
        "objective": "Issue emergency technical advisory and action items.",
        "focus_keywords": ["Vulnerability", "Patching", "Zero-day"]
    }

    response = await client.post(f"/api/v1/generate/{doc_id}", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["document_id"] == doc_id
    assert len(data["artefacts"]) == 3
    assert data["total_generation_time_seconds"] >= 0.0

    output_types_received = [a["output_type"] for a in data["artefacts"]]
    assert "linkedin_post" in output_types_received
    assert "twitter_thread" in output_types_received
    assert "executive_summary" in output_types_received


@pytest.mark.asyncio
async def test_generate_api_nonexistent_doc(client: AsyncClient):
    response = await client.post(
        "/api/v1/generate/non_existent_doc_id",
        json={"output_types": ["linkedin_post"]}
    )
    assert response.status_code == 404
