"""
Unit tests for Meme & Informal Graphic Recognition, Knowledge Extraction, and Fact-Checking filtering.
"""

import json
from unittest.mock import MagicMock
from PIL import Image
import pytest

from backend.app.processors.base import RawDocumentElement
from backend.app.services.recognition.image_service import image_recognition_service
from backend.app.services.knowledge_engine import knowledge_engine
from backend.app.services.validation.fact_checker import fact_checker
from backend.app.schemas.semantic_document import (
    SemanticDocument,
    DocumentMetadata,
    SemanticElement,
    ElementContent,
)
from backend.app.schemas.knowledge_package import KnowledgePackage, VisualInsightItem, ContentStrategy
from backend.app.schemas.intent import IntentAndPersonalization, OutputType
from backend.app.schemas.generated_artefact import GeneratedArtefact


def test_meme_recognition_payload_parsing():
    """Verify that ImageRecognitionService extracts meme attributes from Qwen output."""
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "visual_type": "meme",
                        "description": "Satirical programming meme showing server room on fire.",
                        "visible_text": ["Deploying to Prod at 5 PM on Friday", "What could go wrong?"],
                        "overlay_text": ["Deploying to Prod at 5 PM on Friday", "What could go wrong?"],
                        "core_concept_or_humor_theme": "High outage risk of unverified late-Friday production deployments",
                        "key_details": ["Animated character smiling in flames"]
                    })
                }
            }
        ]
    }

    dummy_image = Image.new("RGB", (300, 300), color="blue")
    element = RawDocumentElement(
        type="image",
        page=1,
        image=dummy_image,
        attributes={}
    )

    image_recognition_service._recognize_element(mock_model, element)

    assert element.attributes.get("is_informal_graphic") is True
    assert "Deploying to Prod at 5 PM on Friday" in element.attributes.get("overlay_text")
    assert element.attributes.get("core_concept") == "High outage risk of unverified late-Friday production deployments"


def test_knowledge_engine_meme_visual_insight():
    """Verify KnowledgeEngine transforms meme attributes into a structured takeaway."""
    doc_id = "doc-meme-test-01"
    meme_element = SemanticElement(
        id="elem-meme-1",
        page=1,
        type="image",
        content=ElementContent(
            caption="Friday deploy meme",
            raw_attributes={
                "is_informal_graphic": True,
                "overlay_text": ["Deploying on Friday", "Zero Tests Passed"],
                "core_concept": "Deployment hygiene and automated CI/CD gating",
                "visual_analysis": {
                    "visual_type": "meme",
                    "description": "Meme regarding skipping tests before release",
                    "overlay_text": ["Deploying on Friday", "Zero Tests Passed"],
                    "core_concept_or_humor_theme": "Deployment hygiene and automated CI/CD gating"
                }
            }
        )
    )

    semantic_doc = SemanticDocument(
        document_id=doc_id,
        metadata=DocumentMetadata(
            file_name="devops_humor.png",
            file_size=2048,
            page_count=1
        ),
        elements=[meme_element]
    )

    insights = knowledge_engine._extract_visual_insights(semantic_doc, [])
    assert len(insights) == 1
    takeaway = insights[0].takeaway

    assert "Informal Graphic/Meme Overlay Text" in takeaway
    assert "Deploying on Friday, Zero Tests Passed" in takeaway
    assert "Core Theme: Deployment hygiene and automated CI/CD gating" in takeaway


def test_fact_checker_filters_irrelevant_visuals():
    """Verify fact checker excludes unrelated visuals (selfies/irrelevant graphics) from grounding pool."""
    kp = KnowledgePackage(
        document_id="doc-fact-01",
        document_title="Infrastructure Report",
        intent=IntentAndPersonalization(document_id="doc-fact-01", output_type=OutputType.EXECUTIVE_SUMMARY),
        retrieved_evidence=[],
        visual_insights=[
            VisualInsightItem(
                element_id="vis-meme-1",
                page=1,
                element_type="image",
                takeaway="Informal Graphic/Meme Overlay Text: 'Deploy on Friday'. Core Theme: CI/CD test gates",
            ),
            VisualInsightItem(
                element_id="vis-unrelated-2",
                page=1,
                element_type="image",
                takeaway="Unrelated selfie in cafeteria of office building",
            )
        ],
        strategy=ContentStrategy(
            headline_hook="Infrastructure Alert",
            key_themes=["CI/CD", "Testing"],
            suggested_structure=["Overview", "Remediation"],
            recommended_cta="Deploy safeguards immediately."
        ),
        orchestrator_prompt_context="Infrastructure Report: Mandate automated CI/CD test gates."
    )

    artefact = GeneratedArtefact(
        artefact_id="art-1",
        document_id="doc-fact-01",
        output_type=OutputType.EXECUTIVE_SUMMARY,
        content={
            "summary": "Mandating automated CI/CD test gates prevents production deploy incidents on Friday."
        }
    )

    trust_score, is_grounded, verifications = fact_checker.verify_factuality(artefact, kp)
    assert len(verifications) > 0
    assert trust_score > 0
    # Confirm the unrelated selfie was not the matched text for this claim
    for v in verifications:
        if v.matched_source_text:
            assert "selfie" not in v.matched_source_text.lower()
