"""
Unit tests for ResponseParser.
Verifies JSON parsing from markdown code blocks, raw strings, and deterministic schema fallbacks.
"""

import json
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
from backend.app.services.orchestrator.response_parser import response_parser


@pytest.fixture
def sample_kp() -> KnowledgePackage:
    intent = IntentAndPersonalization(
        document_id="doc_sample_1",
        output_type=OutputType.LINKEDIN_POST,
        audience=AudienceType.EXECUTIVE,
        tone=ToneType.PROFESSIONAL,
        objective="Analyze quarterly growth.",
        detail_level=DetailLevel.CONCISE,
        focus_keywords=["Growth", "Q3"],
    )
    return KnowledgePackage(
        document_id="doc_sample_1",
        document_title="Q3 Growth Report",
        intent=intent,
        key_metrics=[KeyMetricItem(label="Revenue", value="$10M", context="Q3 Record")],
        strategy=ContentStrategy(
            headline_hook="Q3 Results Deliver Record $10M Revenue Surge",
            key_themes=["Revenue Growth"],
            suggested_structure=["Overview", "Data", "Outlook"],
            recommended_cta="Contact us for partnership inquiries.",
        ),
        orchestrator_prompt_context="Q3 Revenue surged to $10M.",
    )


def test_parse_clean_json_linkedin(sample_kp: KnowledgePackage):
    raw = json.dumps({
        "hook": "🚀 Enterprise AI has unlocked unprecedented productivity gains.",
        "body": "In 2026, companies adopting automated workflows reported an average of 34% efficiency gains across multiple sectors.",
        "cta": "Read our latest benchmark study to learn more.",
        "hashtags": ["#AI", "#Innovation", "#TechTrends"],
        "word_count": 180,
    })
    status, content = response_parser.parse_response(raw, OutputType.LINKEDIN_POST, sample_kp)
    assert status == "success"
    assert content["hook"].startswith("🚀")
    assert len(content["hashtags"]) == 3


def test_parse_markdown_fenced_json_twitter(sample_kp: KnowledgePackage):
    raw = """
Here is the requested Twitter thread:
```json
{
  "tweets": [
    {"index": 1, "text": "1/5 🚀 AI is revolutionizing business workflows in 2026."},
    {"index": 2, "text": "2/5 📊 Revenue jumped to $10M in Q3 according to latest benchmarks."}
  ],
  "tweet_count": 2
}
```
Hope this helps!
"""
    status, content = response_parser.parse_response(raw, OutputType.TWITTER_THREAD, sample_kp)
    assert status == "success"
    assert len(content["tweets"]) == 2
    assert content["tweets"][0]["text"].startswith("1/5")


def test_parse_fallback_when_invalid_json(sample_kp: KnowledgePackage):
    raw = "This is a purely unformatted text summary without JSON braces at all."
    status, content = response_parser.parse_response(raw, OutputType.EXECUTIVE_SUMMARY, sample_kp)
    assert status == "fallback"
    assert "title" in content
    assert "overview" in content
    assert "key_findings" in content


# --- Explicit Test Cases 1 through 5 for Qwen3-4B LinkedIn Pipeline ---

def test_case_1_think_block_removal_with_valid_json(sample_kp: KnowledgePackage):
    """Case 1: <think>...</think>{valid json}"""
    raw = """<think>
I need to craft a high-impact LinkedIn post based on the Q3 Growth Report.
The document highlights $10M in revenue.
Let's structure this with a bold opening hook, 2 dense paragraphs, a CTA, and hashtags.
Ensure no markdown code fences are used.
</think>
{
  "hook": "🚀 Q3 Results Deliver Record $10M Revenue Surge across enterprise sectors.",
  "body": "In 2026, companies adopting automated workflows reported an average of 34% efficiency gains across multiple sectors. Our benchmark study examined over 500 enterprise deployments to establish scalable execution patterns.\\n\\nOrganizations implementing these strategic transformations demonstrate substantial capacity gains, reduced cycle latency, and sustained operational excellence.",
  "cta": "Read our latest benchmark study to explore the detailed findings.",
  "hashtags": ["#AI", "#Innovation", "#TechTrends"],
  "word_count": 92
}"""
    status, content, diag = response_parser.parse_response(
        raw, OutputType.LINKEDIN_POST, sample_kp, return_diagnostics=True
    )
    assert status == "success"
    assert diag["json_extracted"] is True
    assert diag["parse_failure_reason"] is None
    assert diag["raw_output_length"] == len(raw)
    assert diag["cleaned_output_length"] < len(raw)
    # Ensure NO <think> leakage exists anywhere in content
    assert "<think>" not in str(content)
    assert "</think>" not in str(content)
    assert "I need to craft" not in str(content)
    assert content["hook"].startswith("🚀")
    assert len(content["hashtags"]) == 3


def test_case_2_text_before_and_after_valid_json(sample_kp: KnowledgePackage):
    """Case 2: Text before JSON + valid JSON (+ trailing text)"""
    raw = """Here is your response for the requested LinkedIn thought leadership post:

{
  "hook": "🚀 Scaling AI systems in 2026 requires grounded evidence and structural metrics.",
  "body": "According to the latest Q3 benchmark, organizations deploying automated workflows achieve a 34% acceleration in operational throughput. The empirical analysis demonstrates that unified data pipelines significantly diminish cross-team execution friction.\\n\\nSenior leaders are leveraging these findings to optimize capital allocation and enhance core product reliability.",
  "cta": "Connect with our team to benchmark your organization's AI maturity.",
  "hashtags": ["#Leadership", "#EnterpriseAI", "#Strategy"],
  "word_count": 88
}

I hope this meets all your requirements! Let me know if you need any adjustments."""

    status, content, diag = response_parser.parse_response(
        raw, OutputType.LINKEDIN_POST, sample_kp, return_diagnostics=True
    )
    assert status == "success"
    assert diag["json_extracted"] is True
    assert diag["parse_failure_reason"] is None
    assert "Here is your response" not in content["hook"]
    assert content["hook"].startswith("🚀")
    assert len(content["hashtags"]) == 3
    assert "body" in content


def test_case_3_malformed_json(sample_kp: KnowledgePackage):
    """Case 3: Malformed JSON that cannot be recovered"""
    raw = """<think>Thinking about generating JSON</think>
This is completely unparseable output:
{ hook: 'unquoted key', body: "unfinished string without closure...
"""
    status, content, diag = response_parser.parse_response(
        raw, OutputType.LINKEDIN_POST, sample_kp, return_diagnostics=True
    )
    assert status == "fallback"
    assert diag["json_extracted"] is False
    assert diag["parse_failure_reason"] is not None
    # Fallback still produces a valid schema object
    assert "hook" in content
    assert "body" in content
    assert "cta" in content
    assert "hashtags" in content


def test_case_4_missing_schema_fields(sample_kp: KnowledgePackage):
    """Case 4: Recoverable JSON but with missing schema fields and invalid types"""
    from backend.app.services.validation.schema_validator import schema_validator
    from backend.app.schemas.generated_artefact import GeneratedArtefact

    raw = """{
  "hook": "🚀 Quick hook for announcement.",
  "cta": "Check this out today.",
  "hashtags": "invalid_string_instead_of_list"
}"""
    status, content, diag = response_parser.parse_response(
        raw, OutputType.LINKEDIN_POST, sample_kp, return_diagnostics=True
    )
    assert status == "success"
    assert diag["json_extracted"] is True

    # Validate schema: must report exact missing fields and invalid types
    artefact = GeneratedArtefact(
        artefact_id="art_case4",
        document_id="doc_sample_1",
        output_type=OutputType.LINKEDIN_POST,
        status=status,
        content=content,
        raw_llm_output=raw,
        generation_metadata=diag,
    )
    is_compliant, violations = schema_validator.validate_schema(artefact)
    assert is_compliant is False
    # Check that exact detailed violation descriptions are reported
    assert any("Missing field: body" in v for v in violations)
    assert any("Invalid field type: hashtags" in v for v in violations)
    assert any("Word count below minimum" in v for v in violations)


def test_case_5_perfect_json(sample_kp: KnowledgePackage):
    """Case 5: Perfect JSON matching LinkedInPost schema directly"""
    from backend.app.services.validation.schema_validator import schema_validator
    from backend.app.services.validation.trust_service import trust_service
    from backend.app.schemas.generated_artefact import GeneratedArtefact

    raw = json.dumps({
        "hook": "🚀 Enterprise AI has unlocked unprecedented productivity gains across modern organizations in 2026.",
        "body": "In 2026, companies adopting automated workflows reported an average of 34% efficiency gains across multiple sectors. Our benchmark study examined over 500 enterprise deployments to establish scalable execution patterns.\n\nOrganizations implementing these strategic transformations demonstrate substantial capacity gains, reduced cycle latency, and sustained operational excellence across cross-functional engineering and product divisions.\n\nFurthermore, leadership teams report accelerated time-to-market and enhanced decision accuracy when integrating grounded machine intelligence into mission-critical workflows.",
        "cta": "Read our latest benchmark study to explore the detailed findings and frameworks.",
        "hashtags": ["#AI", "#Innovation", "#TechTrends"],
        "word_count": 95,
    })

    status, content, diag = response_parser.parse_response(
        raw, OutputType.LINKEDIN_POST, sample_kp, return_diagnostics=True
    )
    assert status == "success"
    assert diag["json_extracted"] is True
    assert diag["parse_failure_reason"] is None

    artefact = GeneratedArtefact(
        artefact_id="art_case5",
        document_id="doc_sample_1",
        output_type=OutputType.LINKEDIN_POST,
        status=status,
        content=content,
        raw_llm_output=raw,
        generation_metadata=diag,
    )

    # Schema validation should pass with zero violations
    is_compliant, violations = schema_validator.validate_schema(artefact)
    assert is_compliant is True
    assert len(violations) == 0

    # Trust service should NOT trigger repair
    verified = trust_service.validate_and_enforce(artefact, sample_kp, auto_repair=True)
    assert verified.validation_report.repaired is False
    assert verified.validation_report.repair_attempts == 0
    assert verified.validation_report.schema_compliance is True
    assert verified.validation_report.original_trust_score is not None
    assert verified.validation_report.repaired_trust_score is None

