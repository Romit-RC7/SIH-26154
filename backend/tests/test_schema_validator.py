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
