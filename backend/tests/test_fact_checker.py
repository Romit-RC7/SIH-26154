"""
Unit tests for FactChecker and TrustService (Stage 5).
"""

import pytest
from backend.app.schemas.intent import OutputType, AudienceType, ToneType, DetailLevel, IntentAndPersonalization
from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.schemas.knowledge_package import KnowledgePackage, EvidenceItem, ContentStrategy
from backend.app.services.validation.fact_checker import fact_checker
from backend.app.services.validation.trust_service import trust_service


@pytest.fixture
def mock_kp_for_trust() -> KnowledgePackage:
    intent = IntentAndPersonalization(
        document_id="doc_trust_1",
        output_type=OutputType.LINKEDIN_POST,
    )
    return KnowledgePackage(
        document_id="doc_trust_1",
        document_title="Cybersecurity Benchmark 2026",
        intent=intent,
        retrieved_evidence=[
            EvidenceItem(
                chunk_id="ch_1",
                page=1,
                chunk_type="text",
                text="Zero-day router vulnerability detected. Over 450 network nodes affected globally.",
                relevance_score=0.95
            ),
            EvidenceItem(
                chunk_id="ch_2",
                page=1,
                chunk_type="text",
                text="Patches deployed reduced incidence rate by 98%. Immediate upgrade is mandatory.",
                relevance_score=0.90
            ),
        ],
        strategy=ContentStrategy(
            headline_hook="Emergency Router Vulnerability Advisory",
            key_themes=["Zero-day", "Network Safety"],
            suggested_structure=["Overview", "Action Required"],
            recommended_cta="Upgrade router firmware immediately."
        ),
        orchestrator_prompt_context="Zero-day router vulnerability detected. Patches reduced incidence by 98%."
    )


def test_fact_checker_grounded_statement(mock_kp_for_trust: KnowledgePackage):
    artefact = GeneratedArtefact(
        artefact_id="art_trust_1",
        document_id="doc_trust_1",
        output_type=OutputType.LINKEDIN_POST,
        content={
            "hook": "Emergency zero-day router vulnerability detected across networks.",
            "body": "Patches deployed reduced incidence rate by 98%. Immediate upgrade is mandatory.",
            "cta": "Upgrade router firmware immediately.",
            "hashtags": ["#Cybersecurity", "#Advisory"]
        }
    )

    trust_score, is_grounded, verifications = fact_checker.verify_factuality(artefact, mock_kp_for_trust)
    assert trust_score >= 0.70
    assert len(verifications) > 0
    assert verifications[0].is_grounded is True


def test_trust_service_validation(mock_kp_for_trust: KnowledgePackage):
    artefact = GeneratedArtefact(
        artefact_id="art_trust_2",
        document_id="doc_trust_1",
        output_type=OutputType.LINKEDIN_POST,
        content={
            "hook": "Critical router security advisory for network operators across global financial infrastructure.",
            "body": (
                "A critical zero-day vulnerability has been detected across enterprise router hardware globally. "
                "Immediate patch deployment is required to mitigate potential remote code execution threats. "
                "Network administrators should verify that firmware updates are systematically applied to all border routers. "
                "Security teams must monitor network traffic logs for anomalous inbound connections and enforce perimeter isolation. "
                "Following manufacturer recommendations, backup configuration profiles must be archived securely before installing hotfixes. "
                "Our telemetry confirms that deployed patches successfully reduced attack incidence by 98 percent."
            ),
            "cta": "Upgrade router firmware immediately to maintain network integrity.",
            "hashtags": ["#Security", "#Network", "#CyberDefense"]
        }
    )

    verified = trust_service.validate_and_enforce(artefact, mock_kp_for_trust, auto_repair=True)
    assert verified.validation_report.trust_score > 0.0
    assert verified.validation_report.schema_compliance is True
