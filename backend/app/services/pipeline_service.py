"""
End-to-End Document Processing Pipeline Service.
Orchestrates document extraction, PP-Structure analysis, semantic fusion,
and stores the finalized Semantic Document JSON in PostgreSQL.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.database.session import AsyncSessionLocal
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_element import DocumentElement, ElementType
from backend.app.models.processing_job import ProcessingJob, JobStatus, PipelineStep
from backend.app.services.semantic_builder import semantic_document_builder
from backend.app.core.logging import logger


class DocumentPipelineService:
    """Coordinates asynchronous pipeline execution for uploaded documents."""

    def __init__(self, session_factory=None):
        self._session_factory = session_factory

    @property
    def session_factory(self):
        return self._session_factory or AsyncSessionLocal

    @session_factory.setter
    def session_factory(self, factory):
        self._session_factory = factory

    async def process_document(
        self,
        document_id: str,
        job_id: str,
        session: Optional[AsyncSession] = None
    ):
        """
        Main pipeline entrypoint. Runs in FastAPI BackgroundTasks or worker process.
        """
        if session is not None:
            await self._run_pipeline(session, document_id, job_id)
        else:
            async with self.session_factory() as managed_session:
                await self._run_pipeline(managed_session, document_id, job_id)

    async def _run_pipeline(self, session: AsyncSession, document_id: str, job_id: str):
        try:
            # 1. Fetch document and job records
            doc_query = await session.execute(select(Document).where(Document.id == document_id))
            doc = doc_query.scalar_one_or_none()

            job_query = await session.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
            job = job_query.scalar_one_or_none()

            if not doc or not job:
                logger.error(f"Cannot process: Document {document_id} or Job {job_id} not found.")
                return

            # Update job to PROCESSING
            job.status = JobStatus.PROCESSING
            job.step = PipelineStep.PARSING
            job.started_at = datetime.now(timezone.utc)
            doc.status = DocumentStatus.PROCESSING
            await session.commit()

            file_path = Path(doc.stored_path)
            logger.info(f"Starting processing for Document {document_id} ({file_path.name})")

            # 2. Document Extraction & Structure Analysis (PP-StructureV3 / Fallback)
            job.step = PipelineStep.STRUCTURE_ANALYSIS
            await session.commit()

            from backend.app.processors.extractor import document_extractor
            raw_elements, meta = document_extractor.extract_document(
                file_path=file_path,
                document_id=document_id
            )

            # 3. Semantic Fusion & Document Contract Generation
            job.step = PipelineStep.SEMANTIC_FUSION
            await session.commit()

            semantic_doc = semantic_document_builder.build(
                document_id=document_id,
                file_path=file_path,
                raw_elements=raw_elements,
                extraction_metadata=meta
            )

            # 4. Persistence Layer: Store in PostgreSQL
            job.step = PipelineStep.PERSISTENCE
            await session.commit()

            # Update Document record with complete Semantic Document JSON
            doc.semantic_json = semantic_doc.model_dump(mode="json")
            doc.page_count = semantic_doc.metadata.page_count
            doc.status = DocumentStatus.COMPLETED
            doc.processing_metadata = {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "total_elements": len(semantic_doc.elements),
                "sources_count": len(semantic_doc.sources),
            }

            # Clear existing elements if reprocessing
            existing_elems = await session.execute(
                select(DocumentElement).where(DocumentElement.document_id == document_id)
            )
            for el in existing_elems.scalars():
                await session.delete(el)

            # Insert relational DocumentElement rows for fast querying
            for idx, elem in enumerate(semantic_doc.elements):
                # Map string type to ElementType enum
                try:
                    elem_enum_type = ElementType(elem.type)
                except ValueError:
                    elem_enum_type = ElementType.TEXT

                db_elem = DocumentElement(
                    id=elem.id,
                    document_id=document_id,
                    element_index=idx + 1,
                    type=elem_enum_type,
                    page=elem.page,
                    bbox=elem.bbox,
                    content=elem.content.model_dump(mode="json"),
                )
                session.add(db_elem)

            # Complete Job
            job.status = JobStatus.COMPLETED
            job.step = PipelineStep.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.processing_metadata = {
                "elements_count": len(semantic_doc.elements),
                "duration_seconds": (job.completed_at - job.started_at).total_seconds()
                if job.started_at else 0
            }

            await session.commit()
            logger.info(f"Pipeline completed successfully for Document {document_id}")

        except Exception as exc:
            logger.exception(f"Pipeline failed for Document {document_id}: {exc}")
            await session.rollback()

            # Re-fetch and record failure on current session
            try:
                doc_query = await session.execute(select(Document).where(Document.id == document_id))
                doc = doc_query.scalar_one_or_none()
                job_query = await session.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
                job = job_query.scalar_one_or_none()

                if doc:
                    doc.status = DocumentStatus.FAILED
                if job:
                    job.status = JobStatus.FAILED
                    job.step = PipelineStep.FAILED
                    job.error_message = str(exc)
                    job.completed_at = datetime.now(timezone.utc)
                await session.commit()
            except Exception as inner_e:
                logger.error(f"Failed to record failure status: {inner_e}")


pipeline_service = DocumentPipelineService()
