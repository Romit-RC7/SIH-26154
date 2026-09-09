"""
Validation & Trust Schemas (Stage 5).
Defines schemas for fact checking, claim verification, hallucination scoring, and schema compliance.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.generated_artefact import GeneratedArtefact


class ClaimVerification(BaseModel):
    """Result of sentence/claim grounding verification against source evidence."""
    statement: str = Field(..., description="Sentence or claim statement evaluated")
    is_grounded: bool = Field(..., description="True if evidence similarity meets grounding threshold")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score against evidence")
    matched_chunk_id: Optional[str] = Field(default=None, description="Source document chunk ID matched")
    matched_source_text: Optional[str] = Field(default=None, description="Source chunk text supporting statement")


class ValidationReport(BaseModel):
    """Comprehensive validation report for a generated artefact."""
    artefact_id: str = Field(..., description="ID of the validated artefact")
    output_type: str = Field(..., description="Output format type")
    trust_score: float = Field(..., ge=0.0, le=1.0, description="Grounding confidence ratio (0.0 to 1.0)")
    is_fully_grounded: bool = Field(..., description="True if all claims meet threshold")
    schema_compliance: bool = Field(..., description="True if format rules are fully met")
    claim_verifications: List[ClaimVerification] = Field(default_factory=list, description="Claim-by-claim analysis")
    violations: List[str] = Field(default_factory=list, description="Format rule violations detected")
    original_trust_score: Optional[float] = Field(default=None, description="Trust score calculated on original LLM output")
    repaired_trust_score: Optional[float] = Field(default=None, description="Trust score calculated on repaired artefact if repair was performed")
    repaired: bool = Field(default=False, description="True if automatic repair was executed")
    repair_attempts: int = Field(default=0, description="Number of repair iterations executed")
    trust_penalties: Dict[str, float] = Field(default_factory=dict, description="Itemized trust score deductions")


class VerifiedArtefact(BaseModel):
    """Container wrapping the generated artefact along with its validation report."""
    artefact: GeneratedArtefact = Field(..., description="Target deliverable artefact")
    validation_report: ValidationReport = Field(..., description="Trust and schema validation report")
