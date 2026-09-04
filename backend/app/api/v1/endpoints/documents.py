"""
Document API Endpoints.
Provides endpoints to upload documents (PDF/DOCX), query status, retrieve the
unified Semantic Document JSON contract, and list processed documents.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.api.deps import get_db
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_element import DocumentElement
from backend.app.models.processing_job import ProcessingJob, JobStatus, PipelineStep
from backend.app.schemas.document import (
    DocumentUploadResponse,
    DocumentDetail,
    DocumentSummary,
    DocumentListResponse,
)
from backend.app.schemas.semantic_document import SemanticDocument
from backend.app.services.storage_service import storage_service
from backend.app.services.pipeline_service import pipeline_service
from backend.app.utils.file_utils import validate_upload_filename, detect_mime_type

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF or DOCX Document",
    description="Uploads a PDF or DOCX file, creates database records, and triggers the background Semantic Document Processing pipeline."
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Validate extension
    is_valid, err_msg = validate_upload_filename(file.filename, settings.ALLOWED_EXTENSIONS)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg
        )

    document_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    try:
        # 2. Save uploaded file to disk
        target_path = await storage_service.save_uploaded_file(file, document_id)
        file_size = target_path.stat().st_size

        # Check maximum allowed size
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            storage_service.delete_document_artifacts(document_id)
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB"
            )

        mime_type = detect_mime_type(file.filename)

        # 3. Create Document DB record
        doc = Document(
            id=document_id,
            filename=file.filename,
            stored_path=str(target_path),
            file_size=file_size,
            mime_type=mime_type,
            page_count=0,
            status=DocumentStatus.PENDING,
            processing_metadata={"original_filename": file.filename}
        )
        db.add(doc)

        # 4. Create ProcessingJob DB record
        job = ProcessingJob(
            id=job_id,
            document_id=document_id,
            status=JobStatus.QUEUED,
            step=PipelineStep.INIT,
            processing_metadata={"upload_filename": file.filename}
        )
        db.add(job)

        await db.commit()
        await db.refresh(doc)

        # 5. Enqueue background pipeline processing
        background_tasks.add_task(pipeline_service.process_document, document_id, job_id)

        logger.info(f"Enqueued document {document_id} for processing (job: {job_id})")

        return DocumentUploadResponse(
            message="Document accepted and enqueued for semantic processing",
            document_id=document_id,
            job_id=job_id,
            status=doc.status,
            filename=doc.filename
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unexpected error during upload: {exc}")
        storage_service.delete_document_artifacts(document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(exc)}"
        )


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Get Document Status and Details",
    description="Retrieves document metadata, upload details, processing status, and extracted element counts."
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    query = await db.execute(select(Document).where(Document.id == document_id))
    doc = query.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    # Count elements
    elem_count_query = await db.execute(
        select(func.count(DocumentElement.id)).where(DocumentElement.document_id == document_id)
    )
    elem_count = elem_count_query.scalar_one() or 0

    return DocumentDetail(
        id=doc.id,
        filename=doc.filename,
        stored_path=doc.stored_path,
        file_size=doc.file_size,
        mime_type=doc.mime_type,
        page_count=doc.page_count,
        status=doc.status,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        element_count=elem_count,
        processing_metadata=doc.processing_metadata
    )


@router.get(
    "/{document_id}/semantic",
    response_model=SemanticDocument,
    summary="Get Unified Semantic Document JSON (System Contract)",
    description="Returns the complete Semantic Document JSON representation consumed by all future AI modules."
)
async def get_document_semantic_json(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    query = await db.execute(select(Document).where(Document.id == document_id))
    doc = query.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found"
        )

    if doc.status == DocumentStatus.PENDING or doc.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail={
                "message": "Document is still being processed",
                "document_id": doc.id,
                "status": doc.status
            }
        )

    if doc.status == DocumentStatus.FAILED:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Document processing failed",
                "document_id": doc.id,
                "status": doc.status,
                "metadata": doc.processing_metadata
            }
        )

    if not doc.semantic_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Semantic Document JSON has not been generated for this document"
        )

    # Return semantic JSON conforming to the contract
    return doc.semantic_json


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="Retrieves a paginated list of uploaded documents, optionally filtered by status."
)
async def list_documents(
    skip: int = Query(0, ge=0, description="Offset"),
    limit: int = Query(20, ge=1, le=100, description="Page limit"),
    status: Optional[DocumentStatus] = Query(None, description="Filter by DocumentStatus"),
    db: AsyncSession = Depends(get_db)
):
    base_query = select(Document)
    count_query = select(func.count(Document.id))

    if status:
        base_query = base_query.where(Document.status == status)
        count_query = count_query.where(Document.status == status)

    total_res = await db.execute(count_query)
    total = total_res.scalar_one()

    results = await db.execute(
        base_query.options(selectinload(Document.elements)).order_by(Document.created_at.desc()).offset(skip).limit(limit)
    )
    docs = results.scalars().all()

    items = [
        DocumentSummary(
            id=d.id,
            filename=d.filename,
            file_size=d.file_size,
            mime_type=d.mime_type,
            page_count=d.page_count,
            status=d.status,
            created_at=d.created_at,
            updated_at=d.updated_at,
            element_count=len(d.elements) if d.elements else 0
        )
        for d in docs
    ]

    return DocumentListResponse(
        total=total,
        skip=skip,
        limit=limit,
        items=items
    )
