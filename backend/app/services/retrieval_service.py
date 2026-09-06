"""
Retrieval Service.
Performs semantic vector search and hybrid filtering against document_chunks in PostgreSQL / pgvector.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logging import logger
from backend.app.models.document_chunk import DocumentChunk, ChunkType
from backend.app.services.embedding.embedding_service import embedding_service


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved via semantic vector search with relevance metrics."""
    id: str
    document_id: str
    element_id: Optional[str]
    chunk_index: int
    chunk_type: ChunkType
    page: int
    content: str
    cleaned_text: str
    chunk_metadata: Dict[str, Any]
    similarity_score: float


class RetrievalService:
    """
    Vector search engine providing cosine similarity search and metadata filtering.
    """

    async def search(
        self,
        query: str,
        db: AsyncSession,
        document_id: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
        chunk_types: Optional[List[ChunkType]] = None,
        page_range: Optional[tuple[int, int]] = None
    ) -> List[RetrievedChunk]:
        """
        Executes a semantic vector similarity search for the given query string.
        """
        if not query.strip():
            return []

        # 1. Compute query vector (384-dim normalized)
        query_vector = embedding_service.embed_query(query)

        # 2. Build base query with filters
        conditions = []
        if document_id:
            conditions.append(DocumentChunk.document_id == document_id)
        if chunk_types:
            conditions.append(DocumentChunk.chunk_type.in_(chunk_types))
        if page_range:
            min_p, max_p = page_range
            conditions.append(and_(DocumentChunk.page >= min_p, DocumentChunk.page <= max_p))

        stmt = select(DocumentChunk)
        if conditions:
            stmt = stmt.where(and_(*conditions))

        result = await db.execute(stmt)
        chunks = result.scalars().all()

        if not chunks:
            return []

        # 3. Score chunks using cosine similarity
        scored_chunks: List[RetrievedChunk] = []
        for ch in chunks:
            score = 0.0
            if ch.embedding is not None and isinstance(ch.embedding, (list, tuple)):
                score = self._compute_cosine_similarity(query_vector, list(ch.embedding))
            
            if score >= min_similarity:
                scored_chunks.append(
                    RetrievedChunk(
                        id=ch.id,
                        document_id=ch.document_id,
                        element_id=ch.element_id,
                        chunk_index=ch.chunk_index,
                        chunk_type=ch.chunk_type,
                        page=ch.page,
                        content=ch.content,
                        cleaned_text=ch.cleaned_text,
                        chunk_metadata=ch.chunk_metadata or {},
                        similarity_score=round(score, 4)
                    )
                )

        # 4. Sort descending by similarity score and take top_k
        scored_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored_chunks[:top_k]

    def _compute_cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine similarity between two float vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# Global retrieval service instance
retrieval_service = RetrievalService()

__all__ = ["RetrievedChunk", "RetrievalService", "retrieval_service"]
