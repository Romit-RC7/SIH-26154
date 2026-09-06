"""
API tests for Knowledge and Retrieval Layer endpoints.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.document import Document, DocumentStatus


@pytest.mark.asyncio
async def test_knowledge_api_embed_and_search_and_assemble(
    client: AsyncClient,
    db_session: AsyncSession
):
    doc_id = "doc_api_knowledge_1"
    semantic_dict = {
        "version": "1.0.0",
        "document_id": doc_id,
        "metadata": {
            "file_name": "ai_tech_stack.pdf",
            "file_size": 1500,
            "title": "Modern AI Architecture Overview",
            "page_count": 1,
        },
        "elements": [
            {
                "id": "elem_stack_1",
                "type": "text",
                "page": 1,
                "content": {
                    "text": "The platform incorporates PyMuPDF rasterization, PP-Structure OCR, BGE embeddings, and Qwen3 reasoning models.",
                    "reading_order": 1,
                    "confidence": 0.99
                }
            },
            {
                "id": "elem_stack_2",
                "type": "table",
                "page": 1,
                "content": {
                    "markdown": "| Layer | Component |\n|---|---|\n| Embeddings | BGE-small-en-v1.5 |\n| Reasoning | Qwen3-4B |",
                    "caption": "Core AI Layer Architecture",
                    "reading_order": 2
                }
            }
        ],
        "entities": [],
        "claims": [],
        "relationships": [],
        "sources": []
    }

    doc = Document(
        id=doc_id,
        filename="ai_tech_stack.pdf",
        stored_path="uploads/raw/ai_tech_stack.pdf",
        file_size=1500,
        status=DocumentStatus.COMPLETED,
        page_count=1,
        semantic_json=semantic_dict
    )
    db_session.add(doc)
    await db_session.commit()

    # 1. POST /api/v1/knowledge/embed/{document_id}
    embed_resp = await client.post(f"/api/v1/knowledge/embed/{doc_id}")
    assert embed_resp.status_code == 200
    embed_data = embed_resp.json()
    assert embed_data["document_id"] == doc_id
    assert embed_data["chunks_created"] >= 2
    assert embed_data["status"] == "completed"

    # 2. POST /api/v1/knowledge/search
    search_resp = await client.post(
        "/api/v1/knowledge/search",
        json={
            "query": "Qwen3 reasoning models and BGE embeddings",
            "document_id": doc_id,
            "top_k": 3
        }
    )
    assert search_resp.status_code == 200
    results = search_resp.json()
    assert len(results) >= 1
    assert "BGE" in results[0]["content"] or "Qwen3" in results[0]["content"]

    # 3. POST /api/v1/knowledge/assemble
    assemble_resp = await client.post(
        "/api/v1/knowledge/assemble",
        json={
            "document_id": doc_id,
            "output_type": "presentation_deck",
            "audience": "technical",
            "tone": "authoritative",
            "language": "English",
            "objective": "Present the end-to-end AI architecture to engineering leads.",
            "detail_level": "comprehensive",
            "focus_keywords": ["BGE", "Qwen3", "PP-Structure"]
        }
    )
    assert assemble_resp.status_code == 200
    package = assemble_resp.json()
    assert package["document_id"] == doc_id
    assert package["intent"]["output_type"] == "presentation_deck"
    assert "orchestrator_prompt_context" in package
    assert len(package["strategy"]["suggested_structure"]) > 0
