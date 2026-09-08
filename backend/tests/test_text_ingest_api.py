"""
Integration tests for Direct Text & Prompt Ingestion API endpoint (POST /api/v1/documents/text).
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_text_ingest_valid_article(client: AsyncClient):
    payload = {
        "title": "Zero Trust Security Briefing",
        "text": (
            "Zero Trust Architecture mandates continuous verification of all assets and users. "
            "Microsegmentation policies should be strictly enforced across hybrid cloud environments "
            "to prevent lateral movement following any credential compromise."
        )
    }
    response = await client.post("/api/v1/documents/text", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert "job_id" in data
    assert data["status"] == "PENDING"
    assert data["filename"] == "Zero Trust Security Briefing.txt"
    assert "RAW_ARTICLE" in data["message"]


@pytest.mark.asyncio
async def test_text_ingest_valid_prompt(client: AsyncClient):
    payload = {
        "text": (
            "Write a concise executive briefing summarizing the security implications of quantum computing "
            "on existing RSA public-key cryptographic infrastructures."
        )
    }
    response = await client.post("/api/v1/documents/text", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "FREEFORM_PROMPT" in data["message"]


@pytest.mark.asyncio
async def test_text_ingest_greeting_typo_rejected(client: AsyncClient):
    payload = {
        "text": "good gorming everyone hope all is well today"
    }
    response = await client.post("/api/v1/documents/text", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "errors" in detail
    assert any("greeting" in err.lower() for err in detail["errors"])


@pytest.mark.asyncio
async def test_text_ingest_too_short_rejected(client: AsyncClient):
    payload = {
        "text": "Too short"
    }
    response = await client.post("/api/v1/documents/text", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("too short" in err.lower() for err in detail["errors"])


@pytest.mark.asyncio
async def test_text_ingest_gibberish_rejected(client: AsyncClient):
    payload = {
        "text": "asdfghjkl qwertyuiop zxcvbnm qwrtypsdfg"
    }
    response = await client.post("/api/v1/documents/text", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any("gibberish" in err.lower() or "unpronounceable" in err.lower() for err in detail["errors"])
