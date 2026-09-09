"""
Trust & Validation Master Service (Stage 5).
Coordinates fact verification, hallucination scoring, schema validation, and repair execution.
"""

from typing import List, Dict, Any, Optional, Tuple
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
    Incorporates factual grounding along with penalty deductions for
    malformed JSON, reasoning leakage (<think>), deterministic fallback,
    and schema repair iterations.
    """

    REASONING_LEAKAGE_PENALTY: float = 0.20
    MALFORMED_JSON_PENALTY: float = 0.25
    FALLBACK_GENERATION_PENALTY: float = 0.35
    REPAIR_ATTEMPT_PENALTY: float = 0.10  # per attempt, max 0.30

    def calculate_adjusted_trust_score(
        self,
        base_grounding_score: float,
        artefact: GeneratedArtefact,
        repair_attempts: int = 0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Deducts penalties from base factual grounding score for:
        1. Reasoning leakage (<think>...</think> in raw text or detected in metadata)
        2. Malformed JSON (parse failure, parse_error status, or unextracted JSON)
        3. Fallback generation (deterministic fallback used)
        4. Schema repair attempts (each repair iteration)
        Returns (clamped_trust_score, penalties_dict).
        """
        penalties: Dict[str, float] = {}
        score = base_grounding_score
        meta = artefact.generation_metadata or {}
        raw_output = artefact.raw_llm_output or ""

        # 1. Reasoning leakage penalty
        has_reasoning = (
            "<think>" in raw_output.lower()
            or "</think>" in raw_output.lower()
            or bool(meta.get("reasoning_detected", False))
        )
        if has_reasoning:
            penalties["reasoning_leakage"] = self.REASONING_LEAKAGE_PENALTY
            score -= self.REASONING_LEAKAGE_PENALTY

        # 2. Malformed JSON penalty
        has_malformed_json = (
            artefact.status == "parse_error"
            or meta.get("json_extracted") is False
            or bool(meta.get("parse_failure_reason"))
        )
        if has_malformed_json:
            penalties["malformed_json"] = self.MALFORMED_JSON_PENALTY
            score -= self.MALFORMED_JSON_PENALTY

        # 3. Fallback generation penalty
        is_fallback = (
            artefact.status == "fallback"
            or meta.get("fallback_generated") is True
            or (isinstance(artefact.content, dict) and artefact.content.get("metadata", {}).get("fallback_generated") is True)
        )
        if is_fallback:
            penalties["fallback_generation"] = self.FALLBACK_GENERATION_PENALTY
            score -= self.FALLBACK_GENERATION_PENALTY

        # 4. Repair attempts penalty
        if repair_attempts > 0:
            attempt_penalty = round(min(0.30, repair_attempts * self.REPAIR_ATTEMPT_PENALTY), 3)
            penalties["repair_attempts"] = attempt_penalty
            score -= attempt_penalty

        final_score = max(0.0, min(1.0, round(score, 3)))
        return final_score, penalties

    def validate_and_enforce(
        self,
        artefact: GeneratedArtefact,
        kp: KnowledgePackage,
        auto_repair: bool = True
    ) -> VerifiedArtefact:
        """
        Validates factual grounding and schema compliance for an artefact,
        executes auto-repair loop if violations are present, and returns a VerifiedArtefact
        with comprehensive trust scoring and penalty deductions.
        """
        logger.info("Executing Trust & Validation Layer for artefact %s", artefact.artefact_id)

        # 1. Fact Verification & Grounding Check on original parsed LLM artefact
        base_original_score, _, claim_verifications = fact_checker.verify_factuality(artefact, kp)
        original_trust_score, original_penalties = self.calculate_adjusted_trust_score(
            base_original_score, artefact, repair_attempts=0
        )

        # 2. Schema Compliance Check
        is_schema_compliant, violations = schema_validator.validate_schema(artefact)

        target_artefact = artefact
        was_repaired = False
        repair_attempts = 0
        repaired_trust_score = None
        final_penalties = original_penalties

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
            base_repaired_score, _, _ = fact_checker.verify_factuality(target_artefact, kp)
            repaired_trust_score, repaired_penalties = self.calculate_adjusted_trust_score(
                base_repaired_score, target_artefact, repair_attempts=repair_attempts
            )
            final_penalties = repaired_penalties

        final_trust_score = repaired_trust_score if was_repaired else original_trust_score
        is_fully_grounded = final_trust_score >= 0.85 and is_schema_compliant

        report = ValidationReport(
            artefact_id=target_artefact.artefact_id,
            output_type=target_artefact.output_type.value,
            trust_score=final_trust_score,
            original_trust_score=original_trust_score,
            repaired_trust_score=repaired_trust_score,
            is_fully_grounded=is_fully_grounded,
            schema_compliance=is_schema_compliant,
            claim_verifications=claim_verifications,
            violations=violations,
            repaired=was_repaired,
            repair_attempts=repair_attempts,
            trust_penalties=final_penalties,
        )

        return VerifiedArtefact(
            artefact=target_artefact,
            validation_report=report
        )


trust_service = TrustService()
