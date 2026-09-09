"""
Unit tests for Qwen Generation Hardening, ChatML formatting, Finish Reason Diagnostics,
Debug Mode Raw Parse Failure Surfacing, and Penalty-Based Trust Scoring.
"""

import json
from unittest.mock import MagicMock, patch
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
    KeyMetricItem,
)
from backend.app.schemas.generated_artefact import (
    GeneratedArtefact,
    GenerateRequest,
)
from backend.app.services.orchestrator.qwen3_generation_service import (
    Qwen3GenerationService,
    qwen3_generation_service,
)
from backend.app.services.orchestrator.response_parser import (
    ResponseParser,
    response_parser,
)
from backend.app.services.validation.trust_service import (
    TrustService,
    trust_service,
)


@pytest.fixture
def sample_kp() -> KnowledgePackage:
    intent = IntentAndPersonalization(
        document_id="doc_h1",
        output_type=OutputType.LINKEDIN_POST,
        audience=AudienceType.EXECUTIVE,
        tone=ToneType.PROFESSIONAL,
        objective="Analyze quarterly performance.",
        detail_level=DetailLevel.CONCISE,
        focus_keywords=["Growth", "AI"],
    )
    return KnowledgePackage(
        document_id="doc_h1",
        document_title="Enterprise Growth Benchmark 2026",
        intent=intent,
        key_metrics=[KeyMetricItem(label="Efficiency Gain", value="34%", context="Empirical study")],
        strategy=ContentStrategy(
            headline_hook="Enterprise AI Delivers 34% Productivity Gains",
            key_themes=["Efficiency", "Scalability"],
            suggested_structure=["Hook", "Evidence", "CTA"],
            recommended_cta="Download the full benchmark report today.",
        ),
        orchestrator_prompt_context="Empirical evidence demonstrates 34% productivity gains across 500 enterprise deployments.",
    )


# --- 1. LinkedIn Hyperparameters & Structured JSON Configuration ---

def test_linkedin_hyperparams_lowered_temp_and_increased_tokens():
    """Verifies LinkedIn post temperature is 0.2 and max_tokens is 1200."""
    params = qwen3_generation_service.FORMAT_HYPERPARAMS[OutputType.LINKEDIN_POST]
    assert params["temperature"] == 0.2
    assert params["max_tokens"] == 1200


def test_generation_passes_structured_json_and_stop_tokens():
    """Verifies that create_chat_completion is invoked with json_object response_format and stop tokens."""
    service = Qwen3GenerationService()
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"content": '{"hook": "Test", "body": "Body", "cta": "CTA", "hashtags": ["#A", "#B"]}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 80},
    }

    with patch("backend.app.services.orchestrator.qwen3_generation_service.qwen_orchestrator_initializer") as mock_init:
        mock_init.is_available.return_value = True
        mock_init.active_model_name = "Qwen3-4B"
        mock_init.load.return_value = mock_model

        raw, meta = service.generate("Test prompt", OutputType.LINKEDIN_POST)

        mock_model.create_chat_completion.assert_called_once()
        kwargs = mock_model.create_chat_completion.call_args[1]

        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 1200
        assert "<|im_end|>" in kwargs["stop"]
        assert "<|endoftext|>" in kwargs["stop"]
        assert meta["finish_reason"] == "stop"
        assert meta["stop_reason"] == "stop token"


# --- 2. Finish Reason Logging & Categorization ---

@pytest.mark.parametrize(
    "raw_finish_reason,expected_stop_reason",
    [
        ("length", "max_tokens"),
        ("stop", "stop token"),
        ("eos", "EOS"),
    ],
)
def test_finish_reason_and_stop_reason_categorization(raw_finish_reason, expected_stop_reason):
    """Verifies finish_reason properly maps to stop_reason ('stop token', 'max_tokens', 'EOS')."""
    service = Qwen3GenerationService()
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"content": '{"hook": "test"}'},
                "finish_reason": raw_finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    with patch("backend.app.services.orchestrator.qwen3_generation_service.qwen_orchestrator_initializer") as mock_init:
        mock_init.is_available.return_value = True
        mock_init.active_model_name = "Qwen3-4B"
        mock_init.load.return_value = mock_model

        _, meta = service.generate("Prompt", OutputType.LINKEDIN_POST)

        assert meta["finish_reason"] == raw_finish_reason
        assert meta["stop_reason"] == expected_stop_reason


def test_offline_fallback_records_stop_reason():
    """Verifies offline fallback mode also populates finish_reason and stop_reason."""
    service = Qwen3GenerationService()
    with patch("backend.app.services.orchestrator.qwen3_generation_service.qwen_orchestrator_initializer") as mock_init:
        mock_init.is_available.return_value = False
        mock_init.active_model_name = "Qwen3-4B"

        _, meta = service.generate("Prompt", OutputType.LINKEDIN_POST)
        assert meta["finish_reason"] == "stop"
        assert meta["stop_reason"] == "stop token"


# --- 3. Debug Mode: Disabling Fallback & Surfacing Raw Parse Failures ---

def test_debug_mode_disables_deterministic_fallback(sample_kp: KnowledgePackage):
    """In debug mode, unparseable output must not invoke fallback; it surfaces raw parse failure."""
    raw_invalid = "This is raw unformatted text with no JSON braces."
    status, content, diag = response_parser.parse_response(
        raw_invalid, OutputType.LINKEDIN_POST, sample_kp, return_diagnostics=True, debug_mode=True
    )
    assert status == "parse_error"
    assert diag["fallback_disabled"] is True
    assert diag["json_extracted"] is False
    assert diag["raw_parse_failure"] is not None
    assert content["error"] == diag["parse_failure_reason"]
    assert content["raw_output"] == raw_invalid


def test_non_debug_mode_still_uses_fallback(sample_kp: KnowledgePackage):
    """When debug_mode is False, deterministic fallback is preserved for resilience."""
    raw_invalid = "This is raw unformatted text with no JSON braces."
    status, content, diag = response_parser.parse_response(
        raw_invalid, OutputType.LINKEDIN_POST, sample_kp, return_diagnostics=True, debug_mode=False
    )
    assert status == "fallback"
    assert diag["fallback_disabled"] is False
    assert "hook" in content
    assert "body" in content


def test_debug_mode_raise_on_error_option(sample_kp: KnowledgePackage):
    """When debug_mode=True and raise_on_error=True, parse_response raises ValueError."""
    raw_invalid = "Malformed text: { unfinished JSON..."
    with pytest.raises(ValueError) as exc_info:
        response_parser.parse_response(
            raw_invalid, OutputType.LINKEDIN_POST, sample_kp, debug_mode=True, raise_on_error=True
        )
    assert "Raw parse failure for linkedin_post" in str(exc_info.value)


# --- 4. Trust Scoring Penalties ---

def test_trust_score_no_penalties_on_perfect_output(sample_kp: KnowledgePackage):
    """A clean, well-formed artefact receives 0 penalties and keeps full trust score."""
    content = {
        "hook": "🚀 Enterprise AI Delivers 34% Productivity Gains across organizations.",
        "body": "Empirical evidence demonstrates 34% productivity gains across 500 enterprise deployments. Modern organizations adopting these methodologies consistently achieve measurable performance improvements and reduced operational risks.",
        "cta": "Download the full benchmark report today to explore the insights.",
        "hashtags": ["#AI", "#Innovation", "#Growth"],
        "word_count": 90,
    }
    artefact = GeneratedArtefact(
        artefact_id="art_clean",
        document_id="doc_h1",
        output_type=OutputType.LINKEDIN_POST,
        status="success",
        content=content,
        raw_llm_output=json.dumps(content),
        generation_metadata={"json_extracted": True, "parse_failure_reason": None},
    )

    score, penalties = trust_service.calculate_adjusted_trust_score(
        base_grounding_score=1.0, artefact=artefact, repair_attempts=0
    )
    assert score == 1.0
    assert len(penalties) == 0


def test_trust_score_penalized_for_reasoning_leakage(sample_kp: KnowledgePackage):
    """Presence of <think> in raw_llm_output deducts reasoning leakage penalty (0.20)."""
    raw_with_think = "<think>I should write a post about AI</think>{\"hook\": \"AI Insights\"}"
    artefact = GeneratedArtefact(
        artefact_id="art_think",
        document_id="doc_h1",
        output_type=OutputType.LINKEDIN_POST,
        status="success",
        content={"hook": "AI Insights"},
        raw_llm_output=raw_with_think,
        generation_metadata={"json_extracted": True, "reasoning_detected": True},
    )

    score, penalties = trust_service.calculate_adjusted_trust_score(
        base_grounding_score=1.0, artefact=artefact, repair_attempts=0
    )
    assert "reasoning_leakage" in penalties
    assert penalties["reasoning_leakage"] == 0.20
    assert score == 0.80


def test_trust_score_penalized_for_malformed_json(sample_kp: KnowledgePackage):
    """Malformed JSON (parse_error status or parse_failure_reason) deducts 0.25."""
    artefact = GeneratedArtefact(
        artefact_id="art_malformed",
        document_id="doc_h1",
        output_type=OutputType.LINKEDIN_POST,
        status="parse_error",
        content={"error": "Malformed JSON syntax"},
        raw_llm_output="{ broken json",
        generation_metadata={"json_extracted": False, "parse_failure_reason": "Malformed JSON syntax"},
    )

    score, penalties = trust_service.calculate_adjusted_trust_score(
        base_grounding_score=1.0, artefact=artefact, repair_attempts=0
    )
    assert "malformed_json" in penalties
    assert penalties["malformed_json"] == 0.25
    assert score == 0.75


def test_trust_score_penalized_for_fallback_generation(sample_kp: KnowledgePackage):
    """Fallback generation deducts 0.35 from trust score."""
    artefact = GeneratedArtefact(
        artefact_id="art_fallback",
        document_id="doc_h1",
        output_type=OutputType.LINKEDIN_POST,
        status="fallback",
        content={"hook": "Fallback Hook", "body": "Fallback Body"},
        raw_llm_output="unparseable text",
        generation_metadata={"json_extracted": False, "fallback_generated": True},
    )

    score, penalties = trust_service.calculate_adjusted_trust_score(
        base_grounding_score=1.0, artefact=artefact, repair_attempts=0
    )
    assert "fallback_generation" in penalties
    assert penalties["fallback_generation"] == 0.35
    assert "malformed_json" in penalties  # json_extracted is False
    assert score == 0.40


def test_trust_score_penalized_for_repair_attempts(sample_kp: KnowledgePackage):
    """Each repair iteration deducts 0.10 from trust score."""
    artefact = GeneratedArtefact(
        artefact_id="art_repaired",
        document_id="doc_h1",
        output_type=OutputType.LINKEDIN_POST,
        status="success",
        content={"hook": "Hook", "body": "Body"},
        raw_llm_output='{"hook": "Hook", "body": "Body"}',
        generation_metadata={"json_extracted": True},
    )

    score_1_attempt, penalties_1 = trust_service.calculate_adjusted_trust_score(
        base_grounding_score=1.0, artefact=artefact, repair_attempts=1
    )
    assert penalties_1["repair_attempts"] == 0.10
    assert score_1_attempt == 0.90

    score_2_attempts, penalties_2 = trust_service.calculate_adjusted_trust_score(
        base_grounding_score=1.0, artefact=artefact, repair_attempts=2
    )
    assert penalties_2["repair_attempts"] == 0.20
    assert score_2_attempts == 0.80


def test_trust_score_cumulative_penalties_with_clamping(sample_kp: KnowledgePackage):
    """Cumulative penalties correctly accumulate and clamp at 0.0."""
    artefact = GeneratedArtefact(
        artefact_id="art_all_penalties",
        document_id="doc_h1",
        output_type=OutputType.LINKEDIN_POST,
        status="fallback",
        content={"hook": "Hook", "body": "Body"},
        raw_llm_output="<think>Thinking</think> broken string",
        generation_metadata={
            "json_extracted": False,
            "parse_failure_reason": "Syntax error",
            "reasoning_detected": True,
            "fallback_generated": True,
        },
    )

    # Base grounding 0.50 with all penalties:
    # 0.50 - 0.20 (reasoning) - 0.25 (malformed) - 0.35 (fallback) - 0.20 (2 repairs) = -0.50 -> clamped to 0.0
    score, penalties = trust_service.calculate_adjusted_trust_score(
        base_grounding_score=0.50, artefact=artefact, repair_attempts=2
    )
    assert score == 0.0
    assert "reasoning_leakage" in penalties
    assert "malformed_json" in penalties
    assert "fallback_generation" in penalties
    assert "repair_attempts" in penalties


def test_validate_and_enforce_stores_trust_penalties(sample_kp: KnowledgePackage):
    """Validation report records trust_penalties in report and reflects reductions."""
    artefact = GeneratedArtefact(
        artefact_id="art_enforce_test",
        document_id="doc_h1",
        output_type=OutputType.LINKEDIN_POST,
        status="fallback",
        content={
            "hook": "🚀 Enterprise AI Delivers 34% Productivity Gains Across Organizations.",
            "body": "Empirical evidence demonstrates 34% productivity gains across 500 enterprise deployments. Modern organizations report sustained competitive advantages and operational reliability.",
            "cta": "Download the full benchmark report today.",
            "hashtags": ["#AI", "#Innovation"],
        },
        raw_llm_output="<think>Reasoning</think> Fallback text",
        generation_metadata={"json_extracted": False, "parse_failure_reason": "No JSON", "fallback_generated": True},
    )

    verified = trust_service.validate_and_enforce(artefact, sample_kp, auto_repair=False)
    report = verified.validation_report

    assert report.trust_penalties is not None
    assert "reasoning_leakage" in report.trust_penalties
    assert "malformed_json" in report.trust_penalties
    assert "fallback_generation" in report.trust_penalties
    assert report.trust_score <= 0.40
