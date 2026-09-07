"""
Unit tests for PromptBuilder.
Verifies format-specific prompt generation across all supported deliverable types.
"""

import pytest
from backend.app.schemas.intent import (
    IntentAndPersonalization,
    OutputType,
    AudienceType,
    ToneType,
    DetailLevel,
)
from backend.app.schemas.knowledge_package import (
    KnowledgePackage,
    ContentStrategy,
    EvidenceItem,
    KeyMetricItem,
)
from backend.app.services.orchestrator.prompt_builder import prompt_builder


@pytest.fixture
def mock_knowledge_package() -> KnowledgePackage:
    intent = IntentAndPersonalization(
        document_id="doc_test_123",
        output_type=OutputType.LINKEDIN_POST,
        audience=AudienceType.EXECUTIVE,
        tone=ToneType.PROFESSIONAL,
        objective="Highlight enterprise productivity gains from AI transformation.",
        detail_level=DetailLevel.MODERATE,
        focus_keywords=["Generative AI", "ROI", "Enterprise"],
    )
    return KnowledgePackage(
        document_id="doc_test_123",
        document_title="AI Enterprise 2026",
        intent=intent,
        retrieved_evidence=[
            EvidenceItem(
                chunk_id="ch_1",
                page=1,
                chunk_type="text",
                text="Generative AI adoption delivered an average productivity gain of 34% in 2026.",
                relevance_score=0.92,
            )
        ],
        key_metrics=[
            KeyMetricItem(label="Productivity Gain", value="34%", context="Across 500 enterprises")
        ],
        strategy=ContentStrategy(
            headline_hook="How Generative AI Delivered 34% Productivity Surges in 2026",
            key_themes=["Enterprise Scale", "Productivity Gains"],
            suggested_structure=["The Hook", "Key Metrics", "Strategic Takeaways"],
            recommended_cta="Adopt modern content transformation pipelines today.",
            tone_guidelines="Authoritative and data-backed",
        ),
        orchestrator_prompt_context=(
            "### Summary Context\n"
            "- Generative AI adoption reached 78%.\n"
            "- Productivity surged by 34% across enterprises."
        ),
    )


def test_prompt_builder_linkedin(mock_knowledge_package: KnowledgePackage):
    prompt = prompt_builder.build_prompt(OutputType.LINKEDIN_POST, mock_knowledge_package)
    assert "LinkedIn Thought Leadership Post" in prompt
    assert "34%" in prompt
    assert "hook" in prompt
    assert "hashtags" in prompt


def test_prompt_builder_twitter(mock_knowledge_package: KnowledgePackage):
    prompt = prompt_builder.build_prompt(OutputType.TWITTER_THREAD, mock_knowledge_package)
    assert "Twitter / X Insight Thread" in prompt
    assert "280 characters" in prompt
    assert "tweets" in prompt


def test_prompt_builder_executive_summary(mock_knowledge_package: KnowledgePackage):
    prompt = prompt_builder.build_prompt(OutputType.EXECUTIVE_SUMMARY, mock_knowledge_package)
    assert "Executive Strategic Briefing" in prompt
    assert "key_findings" in prompt
    assert "recommendations" in prompt


def test_prompt_builder_presentation_deck(mock_knowledge_package: KnowledgePackage):
    prompt = prompt_builder.build_prompt(OutputType.PRESENTATION_DECK, mock_knowledge_package)
    assert "Presentation Slide Deck" in prompt
    assert "slides" in prompt
    assert "speaker_notes" in prompt


def test_prompt_builder_infographic(mock_knowledge_package: KnowledgePackage):
    prompt = prompt_builder.build_prompt(OutputType.INFOGRAPHIC_BRIEF, mock_knowledge_package)
    assert "Infographic Blueprint" in prompt
    assert "key_stats" in prompt
    assert "visual_sections" in prompt


def test_prompt_builder_video_script(mock_knowledge_package: KnowledgePackage):
    prompt = prompt_builder.build_prompt(OutputType.VIDEO_SCRIPT, mock_knowledge_package)
    assert "Video Package" in prompt
    assert "scenes" in prompt
    assert "narration" in prompt
