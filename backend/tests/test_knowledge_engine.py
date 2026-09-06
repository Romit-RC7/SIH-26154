"""
Unit tests for KnowledgeEngine and KnowledgePackage assembly.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.document import Document, DocumentStatus
from backend.app.schemas.semantic_document import (
    SemanticDocument,
    DocumentMetadata,
    SemanticElement,
    ElementContent,
)
from backend.app.schemas.intent import (
    IntentAndPersonalization,
    OutputType,
    AudienceType,
    ToneType,
    DetailLevel,
)
from backend.app.services.knowledge_engine import knowledge_engine


@pytest.mark.asyncio
async def test_assemble_knowledge_produces_valid_package(db_session: AsyncSession):
    doc_id = "doc_ke_test_1"
    
    # Setup semantic document payload
    semantic_dict = {
        "version": "1.0.0",
        "document_id": doc_id,
        "metadata": {
            "file_name": "ai_adoption_2026.pdf",
            "file_size": 2048,
            "title": "State of Enterprise AI Adoption 2026",
            "page_count": 2,
        },
        "elements": [
            {
                "id": "elem_t1",
                "type": "text",
                "page": 1,
                "content": {
                    "text": "Enterprise generative AI adoption reached 78% in 2026, delivering an average productivity gain of 34%.",
                    "confidence": 0.98,
                    "reading_order": 1,
                    "raw_attributes": {"role": "title"}
                }
            },
            {
                "id": "elem_tbl1",
                "type": "table",
                "page": 1,
                "content": {
                    "markdown": "| Sector | Adoption Rate | ROI |\n|---|---|---|\n| Financial Services | 84% | 4.2x |\n| Healthcare | 71% | 3.1x |\n| Technology | 92% | 5.0x |",
                    "caption": "AI Adoption and ROI by Industry Sector",
                    "reading_order": 2
                }
            },
            {
                "id": "elem_fig1",
                "type": "chart",
                "page": 2,
                "content": {
                    "caption": "5-Year AI Investment Trajectory",
                    "image_path": "uploads/extracted/doc_ke_test_1/elem_fig1.png",
                    "reading_order": 3,
                    "raw_attributes": {"visual_analysis": "Investments surged from $12B in 2022 to $85B in 2026."}
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
        filename="ai_adoption_2026.pdf",
        stored_path="uploads/raw/ai_adoption_2026.pdf",
        file_size=2048,
        status=DocumentStatus.COMPLETED,
        page_count=2,
        semantic_json=semantic_dict
    )
    db_session.add(doc)
    await db_session.commit()

    # User Intent & Personalization
    intent = IntentAndPersonalization(
        document_id=doc_id,
        output_type=OutputType.LINKEDIN_POST,
        audience=AudienceType.EXECUTIVE,
        tone=ToneType.PERSUASIVE,
        language="English",
        objective="Highlight rapid enterprise AI adoption and exceptional ROI across sectors.",
        detail_level=DetailLevel.MODERATE,
        focus_keywords=["generative AI", "ROI", "Financial Services"]
    )

    # Execute Knowledge Engine
    package = await knowledge_engine.assemble_knowledge(
        intent=intent,
        document=doc,
        db=db_session
    )

    # Validate output contract
    assert package.document_id == doc_id
    assert package.document_title == "State of Enterprise AI Adoption 2026"
    assert package.intent.output_type == OutputType.LINKEDIN_POST
    assert package.strategy is not None
    assert len(package.strategy.suggested_structure) > 0
    assert len(package.tables) == 1
    assert len(package.visual_insights) == 1
    assert len(package.key_metrics) >= 1

    # Verify high-density Orchestrator Prompt Context Markdown block
    prompt_ctx = package.orchestrator_prompt_context
    assert "# Knowledge Context for Content Orchestration" in prompt_ctx
    assert "## 1. Content Strategy Blueprint" in prompt_ctx
    assert "Financial Services" in prompt_ctx or "78%" in prompt_ctx or "ROI" in prompt_ctx
    assert "elem_tbl1" in prompt_ctx or "elem_t1" in prompt_ctx
