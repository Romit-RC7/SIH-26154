"""
Content Orchestration & Generation Endpoints (Phase 4).
Exposes REST endpoints to transform documents and KnowledgePackages into
tailored multi-format artefacts (LinkedIn, Twitter, Exec Summary, PPT, Infographic, Video Script).
"""

import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Body, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.api.deps import get_db
from backend.app.models.document import Document
from backend.app.models.processing_job import ProcessingJob, JobStatus
from backend.app.services.pipeline_service import pipeline_service
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
    request_started = time.perf_counter()
    result = await db.execute(
        select(Document).where(Document.id == clean_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{clean_id}' not found."
        )

    pipeline_timings = {}
    if not document.semantic_json:
        job_result = await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.document_id == clean_id)
            .order_by(ProcessingJob.created_at.desc())
            .limit(1)
        )
        job = job_result.scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Document has no processing job.")
        if job.status == JobStatus.PROCESSING:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document processing is already in progress; retry generation when it completes.")

        # A queued/failed document can be processed inline, making this endpoint
        # an end-to-end path from stored source file to generated artefact.
        await pipeline_service.process_document(clean_id, job.id, db)
        await db.refresh(document)
        await db.refresh(job)
        if not document.semantic_json:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document processing did not produce semantic data.")
        pipeline_timings = (job.processing_metadata or {}).get("stage_timings", {})
        pipeline_timings["document_processing_total_seconds"] = (job.processing_metadata or {}).get("duration_seconds", 0)

    # Include timings from a previously completed inline-processing run too.
    if not pipeline_timings:
        job_result = await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.document_id == clean_id)
            .order_by(ProcessingJob.created_at.desc())
            .limit(1)
        )
        latest_job = job_result.scalar_one_or_none()
        if latest_job:
            pipeline_timings = dict((latest_job.processing_metadata or {}).get("stage_timings", {}))
            if (latest_job.processing_metadata or {}).get("duration_seconds") is not None:
                pipeline_timings["document_processing_total_seconds"] = (latest_job.processing_metadata or {}).get("duration_seconds")

    try:
        response = await orchestrator_service.generate_artefacts(
            document=document,
            request=request,
            db=db
        )
        response.timings = {
            **pipeline_timings,
            **response.timings,
            "end_to_end_total_seconds": round(time.perf_counter() - request_started, 2),
        }
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content: {str(exc)}"
        )
