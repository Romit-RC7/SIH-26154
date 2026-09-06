"""
Knowledge and Retrieval Layer API Endpoints.
Exposes endpoints for BGE vector embeddings generation, pgvector similarity search,
and Knowledge Engine context assembly for the Content Orchestrator.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.api.deps import get_db
from backend.app.models.document import Document
from backend.app.models.document_chunk import DocumentChunk
from backend.app.schemas.semantic_document import SemanticDocument
from backend.app.schemas.intent import (
    IntentAndPersonalization,
    OutputType,
    AudienceType,
    ToneType,
    DetailLevel,
)
from backend.app.schemas.knowledge_package import KnowledgePackage
from backend.app.services.embedding.embedding_service import embedding_service
from backend.app.services.retrieval_service import retrieval_service, RetrievedChunk
from backend.app.services.knowledge_engine import knowledge_engine

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    document_id: Optional[str] = Field(default=None, description="Optional document ID filter")
    top_k: int = Field(default=5, ge=1, le=50, description="Max number of results to return")
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity score")


class AssembleRequest(BaseModel):
    """Optional overrides for intent configuration when calling assemble/{document_id}."""
    output_type: OutputType = Field(default=OutputType.EXECUTIVE_SUMMARY, description="Target output format")
    audience: AudienceType = Field(default=AudienceType.EXECUTIVE, description="Target persona")
    tone: ToneType = Field(default=ToneType.PROFESSIONAL, description="Tone of delivery")
    language: str = Field(default="English", description="Target language")
    objective: str = Field(
        default="Summarize key insights, data points, and recommendations.",
        description="Primary goal or core thesis"
    )
    detail_level: DetailLevel = Field(default=DetailLevel.MODERATE, description="Content depth")
    focus_keywords: List[str] = Field(default_factory=list, description="Keywords to prioritize")
    custom_instructions: Optional[str] = Field(default=None, description="Custom guidelines")
    use_llm: bool = Field(
        default=True,
        description="If true, uses Qwen3-4B for advanced reasoning (may take 30-60s on first call). "
                    "Set to false for instant deterministic extraction (no model load needed)."
    )


class SearchResponseItem(BaseModel):
    id: str
    document_id: str
    element_id: Optional[str]
    chunk_index: int
    chunk_type: str
    page: int
    content: str
    similarity_score: float


class EmbedResponse(BaseModel):
    document_id: str
    chunks_created: int
    status: str


@router.post(
    "/embed/{document_id}",
    response_model=EmbedResponse,
    summary="Generate BGE Vector Embeddings",
    description="Cleans, chunks, and creates 384-dimensional dense BGE embeddings for a document, saving to pgvector."
)
async def embed_document(
    document_id: str = Path(..., description="Target document ID to embed"),
    db: AsyncSession = Depends(get_db)
):
    clean_doc_id = document_id.strip()
    result = await db.execute(select(Document).where(Document.id == clean_doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{clean_doc_id}' not found"
        )

    if not doc.semantic_json:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document does not contain semantic JSON. Process the document before embedding."
        )

    semantic_doc = SemanticDocument.model_validate(doc.semantic_json)
    chunks = await embedding_service.embed_and_store_document(
        document_id=clean_doc_id,
        semantic_doc=semantic_doc,
        db=db
    )

    return EmbedResponse(
        document_id=clean_doc_id,
        chunks_created=len(chunks),
        status="completed"
    )


@router.post(
    "/search",
    response_model=List[SearchResponseItem],
    summary="Semantic Vector Search (POST Body)",
    description="Searches document chunks using cosine similarity over BGE dense vector embeddings."
)
async def search_knowledge(
    req: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    target_doc_id = req.document_id.strip() if req.document_id else None
    chunks = await retrieval_service.search(
        query=req.query,
        db=db,
        document_id=target_doc_id,
        top_k=req.top_k,
        min_similarity=req.min_similarity
    )

    return [
        SearchResponseItem(
            id=c.id,
            document_id=c.document_id,
            element_id=c.element_id,
            chunk_index=c.chunk_index,
            chunk_type=str(c.chunk_type.value if hasattr(c.chunk_type, "value") else c.chunk_type),
            page=c.page,
            content=c.content,
            similarity_score=c.similarity_score
        )
        for c in chunks
    ]


@router.get(
    "/search",
    response_model=List[SearchResponseItem],
    summary="Semantic Vector Search (Query Parameters)",
    description="Interactive query endpoint with dedicated UI parameter fields for document ID, query, top_k, and min_similarity."
)
async def search_knowledge_query(
    query: str = Query(..., description="Search query string"),
    document_id: Optional[str] = Query(None, description="Optional target document ID filter"),
    top_k: int = Query(5, ge=1, le=50, description="Max number of results to return"),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0, description="Minimum cosine similarity score"),
    db: AsyncSession = Depends(get_db)
):
    target_doc_id = document_id.strip() if document_id else None
    chunks = await retrieval_service.search(
        query=query,
        db=db,
        document_id=target_doc_id,
        top_k=top_k,
        min_similarity=min_similarity
    )

    return [
        SearchResponseItem(
            id=c.id,
            document_id=c.document_id,
            element_id=c.element_id,
            chunk_index=c.chunk_index,
            chunk_type=str(c.chunk_type.value if hasattr(c.chunk_type, "value") else c.chunk_type),
            page=c.page,
            content=c.content,
            similarity_score=c.similarity_score
        )
        for c in chunks
    ]


@router.post(
    "/assemble/{document_id}",
    response_model=KnowledgePackage,
    summary="Assemble Knowledge Context by Document ID",
    description="Dedicated path endpoint providing a direct document_id input field in Swagger UI to assemble the Knowledge Package for the Content Orchestrator."
)
async def assemble_knowledge_by_id(
    document_id: str = Path(..., description="Target document ID"),
    req: AssembleRequest = Body(default_factory=AssembleRequest),
    db: AsyncSession = Depends(get_db)
):
    clean_doc_id = document_id.strip()
    intent = IntentAndPersonalization(
        document_id=clean_doc_id,
        output_type=req.output_type,
        audience=req.audience,
        tone=req.tone,
        language=req.language,
        objective=req.objective,
        detail_level=req.detail_level,
        focus_keywords=req.focus_keywords,
        custom_instructions=req.custom_instructions
    )
    return await _execute_assemble(intent, db)


@router.post(
    "/assemble",
    response_model=KnowledgePackage,
    summary="Assemble Knowledge Package (Full JSON Body)",
    description="Processes user intent & personalization, performs semantic retrieval, executes Qwen3-4B reasoning, and returns structured context for the Content Orchestrator."
)
async def assemble_knowledge_package(
    intent: IntentAndPersonalization,
    db: AsyncSession = Depends(get_db)
):
    intent.document_id = intent.document_id.strip()
    return await _execute_assemble(intent, db)


async def _execute_assemble(intent: IntentAndPersonalization, db: AsyncSession) -> KnowledgePackage:
    clean_doc_id = intent.document_id.strip()
    result = await db.execute(select(Document).where(Document.id == clean_doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{clean_doc_id}' not found"
        )

    # Automatically trigger embedding if not already embedded
    chunk_check = await db.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == clean_doc_id).limit(1)
    )
    if not chunk_check.scalar_one_or_none() and doc.semantic_json:
        semantic_doc = SemanticDocument.model_validate(doc.semantic_json)
        await embedding_service.embed_and_store_document(
            document_id=clean_doc_id,
            semantic_doc=semantic_doc,
            db=db
        )

    package = await knowledge_engine.assemble_knowledge(
        intent=intent,
        document=doc,
        db=db
    )

    return package
