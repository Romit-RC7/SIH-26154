"""
Content Orchestration & Generation Endpoints (Phase 4).
Exposes REST endpoints to transform documents and KnowledgePackages into
tailored multi-format artefacts (LinkedIn, Twitter, Exec Summary, PPT, Infographic, Video Script).
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Body, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.api.deps import get_db
from backend.app.models.document import Document
from backend.app.schemas.generated_artefact import GenerateRequest, GenerateResponse
from backend.app.services.orchestrator.orchestrator_service import orchestrator_service

router = APIRouter()


@router.post(
    "/{document_id}",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate multi-format content deliverables from document context",
    description=(
        "Consumes the document's Semantic JSON & pgvector evidence via Knowledge Engine, "
        "and runs Qwen3-8B generation for all specified output types."
    )
)
async def generate_content(
    document_id: str = Path(..., description="Target document ID"),
    request: GenerateRequest = Body(...),
    db: AsyncSession = Depends(get_db)
) -> GenerateResponse:
    """
    Generates structured communication deliverables for one or more output types
    (e.g., LinkedIn Post, Twitter Thread, Executive Summary, Slide Deck, Infographic, Video Script).
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

    if not document.semantic_json:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document '{clean_id}' has not been processed into a Semantic Document yet."
        )

    try:
        response = await orchestrator_service.generate_artefacts(
            document=document,
            request=request,
            db=db
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content: {str(exc)}"
        )
