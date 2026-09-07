"""
Trust & Validation Endpoints (Stage 5).
Exposes REST endpoints to inspect factual grounding, Trust Scores, claim verifications,
and schema enforcement for generated deliverables.
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Body, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.api.deps import get_db
from backend.app.models.document import Document
from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.schemas.validation import VerifiedArtefact, ValidationReport
from backend.app.services.knowledge_engine import knowledge_engine
from backend.app.services.validation.trust_service import trust_service
from backend.app.schemas.intent import IntentAndPersonalization, OutputType

router = APIRouter()


@router.post(
    "/{document_id}",
    response_model=VerifiedArtefact,
    status_code=status.HTTP_200_OK,
    summary="Validate factual grounding and schema compliance of a generated artefact",
    description="Cross-references claims against document evidence, scores trust (0-100%), and executes auto-repair if rules are violated."
)
async def validate_artefact_endpoint(
    document_id: str = Path(..., description="Target document ID"),
    artefact: GeneratedArtefact = Body(...),
    db: AsyncSession = Depends(get_db)
) -> VerifiedArtefact:
    """
    Evaluates factual grounding and schema compliance for the provided artefact against document context.
    """
    clean_id = document_id.strip()
    result = await db.execute(
        select(Document).where(Document.id == clean_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{clean_id}' not found."
        )

    # Assemble KnowledgePackage context for verification
    intent = IntentAndPersonalization(
        document_id=document.id,
        output_type=artefact.output_type,
    )
    kp = await knowledge_engine.assemble_knowledge(intent, document, db)

    verified = trust_service.validate_and_enforce(artefact, kp, auto_repair=True)
    return verified
