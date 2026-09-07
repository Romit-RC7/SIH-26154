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
