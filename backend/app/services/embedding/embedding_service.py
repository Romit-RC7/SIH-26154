"""
Embedding Service.
Coordinates document text cleaning, chunking, BGE vector embedding generation,
and persistence to PostgreSQL with pgvector.
"""

from typing import List, Optional
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logging import logger
from backend.app.models.document_chunk import DocumentChunk
from backend.app.schemas.semantic_document import SemanticDocument
from backend.app.services.embedding.chunker import DocumentChunker, ChunkItem
from backend.app.services.model_initializer.bge_initializer import bge_initializer


class EmbeddingService:
    """
    Manages semantic document chunking, BGE-small vector creation,
    and pgvector database storage.
    """

    def __init__(self, chunker: Optional[DocumentChunker] = None):
        self.chunker = chunker or DocumentChunker()

    async def embed_and_store_document(
        self,
        document_id: str,
        semantic_doc: SemanticDocument,
        db: AsyncSession
    ) -> List[DocumentChunk]:
        """
        Chunks the document, generates 384-dimensional dense BGE embeddings,
        and saves all chunks to the database.
        """
        # 1. Chunk document
        chunk_items: List[ChunkItem] = self.chunker.chunk_document(semantic_doc)
        if not chunk_items:
            logger.warning("No chunks generated for document %s", document_id)
            return []

        logger.info("Generated %d chunks for document %s", len(chunk_items), document_id)

        # 2. Extract texts for batch embedding
        texts_to_embed = [ch.cleaned_text for ch in chunk_items]

        # 3. Generate dense vectors using BGE
        embeddings = bge_initializer.encode(texts_to_embed, normalize_embeddings=True)

        # 4. Remove previous chunks for this document (idempotent re-processing)
        await db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

        # 5. Build DocumentChunk ORM objects
        db_chunks: List[DocumentChunk] = []
        for item, emb in zip(chunk_items, embeddings):
            chunk_orm = DocumentChunk(
                document_id=document_id,
                element_id=item.element_id,
                chunk_index=item.chunk_index,
                chunk_type=item.chunk_type,
                page=item.page,
                content=item.content,
                cleaned_text=item.cleaned_text,
                chunk_metadata=item.chunk_metadata,
                embedding=emb,
            )
            db.add(chunk_orm)
            db_chunks.append(chunk_orm)

        await db.commit()
        logger.info("Successfully persisted %d embedded chunks into database for doc %s", len(db_chunks), document_id)
        return db_chunks

    def embed_query(self, query: str) -> List[float]:
        """Generates a 384-dim normalized query vector with BGE instruction prefix."""
        return bge_initializer.encode_query(query)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vectors for arbitrary text strings."""
        return bge_initializer.encode(texts, normalize_embeddings=True)


# Global embedding service instance
embedding_service = EmbeddingService()

__all__ = ["EmbeddingService", "embedding_service"]
