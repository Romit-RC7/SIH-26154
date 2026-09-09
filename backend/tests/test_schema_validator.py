"""
Unit tests for SchemaValidator and RepairService (Stage 5).
"""

import pytest
from backend.app.schemas.intent import OutputType, AudienceType, ToneType, DetailLevel, IntentAndPersonalization
from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.schemas.knowledge_package import KnowledgePackage, ContentStrategy
from backend.app.services.validation.schema_validator import schema_validator
from backend.app.services.validation.repair_service import repair_service


@pytest.fixture
def mock_kp_repair() -> KnowledgePackage:
    intent = IntentAndPersonalization(document_id="doc_r1", output_type=OutputType.TWITTER_THREAD)
    return KnowledgePackage(
        document_id="doc_r1",
        intent=intent,
        strategy=ContentStrategy(headline_hook="Twitter Hook", key_themes=[], suggested_structure=[]),
        orchestrator_prompt_context="Context info"
    )


def test_schema_validator_twitter_over_length():
    long_tweet = "A" * 300  # Exceeds 280 chars
    artefact = GeneratedArtefact(
        artefact_id="art_tw_over",
        document_id="doc_r1",
        output_type=OutputType.TWITTER_THREAD,
        content={
            "tweets": [
                {"index": 1, "text": "1/3 Valid short tweet."},
                {"index": 2, "text": long_tweet},
                {"index": 3, "text": "3/3 Valid closing tweet."}
            ]
        }
    )

    is_compliant, violations = schema_validator.validate_schema(artefact)
    assert is_compliant is False
    assert any("exceeds 280 characters" in v for v in violations)


def test_repair_service_fixes_twitter_over_length(mock_kp_repair: KnowledgePackage):
    long_tweet = "B" * 300
    artefact = GeneratedArtefact(
        artefact_id="art_tw_fix",
        document_id="doc_r1",
        output_type=OutputType.TWITTER_THREAD,
        content={
            "tweets": [
                {"index": 1, "text": "1/3 Valid short tweet."},
                {"index": 2, "text": long_tweet},
                {"index": 3, "text": "3/3 Valid closing tweet."}
            ]
        }
    )

    is_comp, violations = schema_validator.validate_schema(artefact)
    assert is_comp is False

    repaired_art, attempts, is_now_comp = repair_service.repair_artefact(artefact, violations, mock_kp_repair)
    assert attempts >= 1
    assert is_now_comp is True
    assert len(repaired_art.content["tweets"][1]["text"]) <= 280


def test_schema_validator_linkedin_exact_violations():
    """Verifies that schema validation logs exact detailed violation messages."""
    artefact = GeneratedArtefact(
        artefact_id="art_li_viol",
        document_id="doc_r1",
        output_type=OutputType.LINKEDIN_POST,
        content={
            "hook": "🚀 Good hook for test.",
            # "body" is intentionally omitted
            "cta": "Click to read more.",
            "hashtags": "single_string_not_list",  # Invalid type
        }
    )
    is_compliant, violations = schema_validator.validate_schema(artefact)
    assert is_compliant is False
    assert any("Missing field: body" in v for v in violations)
    assert any("Invalid field type: hashtags" in v for v in violations)
    assert any("Word count below minimum" in v for v in violations)


def test_repair_service_fixes_linkedin_missing_body_and_hashtags():
    """Verifies that repair service fixes missing body, invalid hashtags, and short length."""
    intent = IntentAndPersonalization(document_id="doc_li_fix", output_type=OutputType.LINKEDIN_POST)
    kp = KnowledgePackage(
        document_id="doc_li_fix",
        document_title="AI Efficiency Report 2026",
        intent=intent,
        strategy=ContentStrategy(
            headline_hook="AI Delivers 34% Productivity Gains Across Enterprises",
            key_themes=["Efficiency", "Productivity"],
            suggested_structure=["Overview", "Details", "CTA"],
            recommended_cta="Contact us today for full benchmark access.",
        ),
        orchestrator_prompt_context="Benchmarked across 500 enterprises showing 34% efficiency improvements.",
    )

    artefact = GeneratedArtefact(
        artefact_id="art_li_repair",
        document_id="doc_li_fix",
        output_type=OutputType.LINKEDIN_POST,
        content={
            "hook": "🚀 AI Delivers 34% Productivity Gains Across Enterprises.",
            # "body" missing
            "cta": "Contact us today for full benchmark access.",
            "hashtags": "invalid_type",
        }
    )

    is_comp, violations = schema_validator.validate_schema(artefact)
    assert is_comp is False

    repaired_art, attempts, is_now_comp = repair_service.repair_artefact(artefact, violations, kp)
    assert attempts >= 1
    assert is_now_comp is True
    assert "body" in repaired_art.content
    assert isinstance(repaired_art.content["hashtags"], list)
    assert len(repaired_art.content["hashtags"]) >= 2
    words = len(f"{repaired_art.content['hook']} {repaired_art.content['body']} {repaired_art.content['cta']}".split())
    assert words >= 80


def test_trust_service_original_and_repaired_scores():
    """Verifies that TrustService calculates original_trust_score and repaired_trust_score."""
    from backend.app.services.validation.trust_service import trust_service

    intent = IntentAndPersonalization(document_id="doc_ts", output_type=OutputType.LINKEDIN_POST)
    kp = KnowledgePackage(
        document_id="doc_ts",
        document_title="AI Benchmark",
        intent=intent,
        strategy=ContentStrategy(
            headline_hook="Major AI Milestone Reached",
            key_themes=[],
            suggested_structure=[],
            recommended_cta="Visit our website to read more.",
        ),
        orchestrator_prompt_context="Empirical evidence validates operational enhancements.",
    )

    # Artefact with missing body requiring repair
    artefact = GeneratedArtefact(
        artefact_id="art_ts_test",
        document_id="doc_ts",
        output_type=OutputType.LINKEDIN_POST,
        content={
            "hook": "🚀 Major AI Milestone Reached Across Multiple Organizations.",
            "cta": "Visit our website to read more.",
            "hashtags": ["#AI", "#Innovation"],
        }
    )

    verified = trust_service.validate_and_enforce(artefact, kp, auto_repair=True)
    report = verified.validation_report
    assert report.repaired is True
    assert report.repair_attempts >= 1
    assert report.original_trust_score is not None
    assert report.repaired_trust_score is not None
    assert report.schema_compliance is True
