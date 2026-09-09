"""
Trust & Validation Master Service (Stage 5).
Coordinates fact verification, hallucination scoring, schema validation, and repair execution.
"""

from typing import List, Dict, Any, Optional
from backend.app.core.logging import logger
from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.schemas.knowledge_package import KnowledgePackage
from backend.app.schemas.validation import ValidationReport, VerifiedArtefact
from backend.app.services.validation.fact_checker import fact_checker
from backend.app.services.validation.schema_validator import schema_validator
from backend.app.services.validation.repair_service import repair_service


class TrustService:
    """
    Master service for Trust, Fact Checking, and Schema Enforcement.
    """

    def validate_and_enforce(
        self,
        artefact: GeneratedArtefact,
        kp: KnowledgePackage,
        auto_repair: bool = True
    ) -> VerifiedArtefact:
        """
        Validates factual grounding and schema compliance for an artefact,
        executes auto-repair loop if violations are present, and returns a VerifiedArtefact.
        """
        logger.info("Executing Trust & Validation Layer for artefact %s", artefact.artefact_id)

        # 1. Fact Verification & Grounding Check on original parsed LLM artefact
        original_trust_score, is_fully_grounded, claim_verifications = fact_checker.verify_factuality(artefact, kp)

        # 2. Schema Compliance Check
        is_schema_compliant, violations = schema_validator.validate_schema(artefact)

        target_artefact = artefact
        was_repaired = False
        repair_attempts = 0
        repaired_trust_score = None

        # 3. Trigger Repair Loop if schema violations exist and auto_repair is True
        if not is_schema_compliant and auto_repair:
            target_artefact, repair_attempts, is_schema_compliant = repair_service.repair_artefact(
                artefact=artefact,
                violations=violations,
                kp=kp
            )
            was_repaired = True
            # Re-verify schema violations on repaired artefact
            _, violations = schema_validator.validate_schema(target_artefact)
            # Compute trust score on repaired artefact
            repaired_trust_score, _, _ = fact_checker.verify_factuality(target_artefact, kp)

        report = ValidationReport(
            artefact_id=target_artefact.artefact_id,
            output_type=target_artefact.output_type.value,
            trust_score=original_trust_score,
            original_trust_score=original_trust_score,
            repaired_trust_score=repaired_trust_score,
            is_fully_grounded=is_fully_grounded,
            schema_compliance=is_schema_compliant,
            claim_verifications=claim_verifications,
            violations=violations,
            repaired=was_repaired,
            repair_attempts=repair_attempts,
        )

        return VerifiedArtefact(
            artefact=target_artefact,
            validation_report=report
        )


trust_service = TrustService()
