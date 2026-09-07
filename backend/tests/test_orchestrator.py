"""
Integration tests for OrchestratorService and REST API endpoints.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.document import Document, DocumentStatus
from backend.app.schemas.intent import OutputType, AudienceType, ToneType, DetailLevel
from backend.app.schemas.generated_artefact import GenerateRequest
from backend.app.services.orchestrator.orchestrator_service import orchestrator_service


@pytest.mark.asyncio
async def test_orchestrator_service_multi_output_generation(db_session: AsyncSession):
    doc_id = "doc_orch_test_1"
    
    # Create sample document with semantic JSON
    doc = Document(
        id=doc_id,
        filename="market_report.pdf",
        stored_path="uploads/raw/market_report.pdf",
        file_size=1024,
        status=DocumentStatus.COMPLETED,
        page_count=1,
        semantic_json={
            "version": "1.0.0",
            "document_id": doc_id,
            "metadata": {
                "file_name": "market_report.pdf",
                "file_size": 1024,
                "title": "2026 AI Market Report",
                "page_count": 1,
            },
            "elements": [
                {
                    "id": "elem_1",
                    "type": "text",
                    "page": 1,
                    "content": {
                        "text": "The enterprise generative AI market reached $45 billion in 2026 with a compound annual growth rate of 38%.",
                        "confidence": 0.99,
                        "reading_order": 1,
                    }
                }
            ]
        }
    )
    db_session.add(doc)
    await db_session.commit()

    # Request multi-format generation: LinkedIn Post + Executive Summary
    req = GenerateRequest(
        output_types=[OutputType.LINKEDIN_POST, OutputType.EXECUTIVE_SUMMARY],
        audience=AudienceType.EXECUTIVE,
        tone=ToneType.PROFESSIONAL,
        detail_level=DetailLevel.MODERATE,
        objective="Summarize market size and growth trajectory.",
        focus_keywords=["Market Size", "$45 billion", "Growth"],
    )

    response = await orchestrator_service.generate_artefacts(
        document=doc,
        request=req,
        db=db_session
    )

    assert response.document_id == doc_id
    assert len(response.artefacts) == 2

    # Check LinkedIn artefact
    linkedin_art = next(a for a in response.artefacts if a.output_type == OutputType.LINKEDIN_POST)
    assert linkedin_art.status in ("success", "fallback")
    assert "hook" in linkedin_art.content
    assert "body" in linkedin_art.content
    assert "hashtags" in linkedin_art.content

    # Check Executive Summary artefact
    exec_art = next(a for a in response.artefacts if a.output_type == OutputType.EXECUTIVE_SUMMARY)
    assert exec_art.status in ("success", "fallback")
    assert "title" in exec_art.content
    assert "overview" in exec_art.content
    assert "key_findings" in exec_art.content
