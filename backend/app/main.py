"""
FastAPI Application Entrypoint for SIH-26154 Semantic Document Processing System.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.core.config import settings
from backend.app.core.logging import setup_logging, logger
from backend.app.database.session import init_db
from backend.app.api.v1.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Setup logging and ensure database tables exist
    setup_logging()
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    try:
        await init_db()
        logger.info("Database schema initialized successfully.")

        # Startup recovery: reset any documents/jobs orphaned in PROCESSING due to previous worker crash
        from backend.app.database.session import AsyncSessionLocal
        from backend.app.models.document import Document, DocumentStatus
        from backend.app.models.processing_job import ProcessingJob, JobStatus, PipelineStep
        from sqlalchemy import update
        from datetime import datetime, timezone

        async with AsyncSessionLocal() as recovery_session:
            now = datetime.now(timezone.utc)
            doc_update = await recovery_session.execute(
                update(Document)
                .where(Document.status == DocumentStatus.PROCESSING)
                .values(status=DocumentStatus.FAILED)
            )
            job_update = await recovery_session.execute(
                update(ProcessingJob)
                .where(ProcessingJob.status.in_([JobStatus.PROCESSING, JobStatus.QUEUED]))
                .values(
                    status=JobStatus.FAILED,
                    step=PipelineStep.FAILED,
                    error_message="Worker terminated unexpectedly during processing (server restarted/interrupted).",
                    completed_at=now,
                )
            )
            await recovery_session.commit()
            if doc_update.rowcount > 0 or job_update.rowcount > 0:
                logger.warning(
                    "Startup recovery cleaned up %d orphaned document(s) and %d job(s) from previous interrupted run.",
                    doc_update.rowcount,
                    job_update.rowcount,
                )
    except Exception as e:
        logger.warning(f"Database initialization / recovery warning (will retry on requests): {e}")

    yield

    # Shutdown
    logger.info("Shutting down Semantic Document Processing System.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Offline semantic document processing API for PDF and DOCX files. "
        "Uploads are queued into bounded batches. The pipeline runs PP-Structure "
        "layout/OCR/table extraction, formula and chart recognition, and optional "
        "Qwen vision enrichment before deterministic semantic fusion and schema validation."
    ),
    openapi_tags=[
        {"name": "Documents", "description": "Upload, monitor, list, and retrieve processed documents."},
        {"name": "System", "description": "Service health and runtime diagnostics."},
        {"name": "Root", "description": "Service metadata and documentation links."},
    ],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local uploads directory for static retrieval of cropped images / artifacts
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"], summary="Service Information")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "documentation": "/docs",
        "endpoints": {
            "upload_document": f"{settings.API_V1_STR}/documents/upload",
            "list_documents": f"{settings.API_V1_STR}/documents",
            "get_document": f"{settings.API_V1_STR}/documents/{{id}}",
            "get_semantic_json": f"{settings.API_V1_STR}/documents/{{id}}/semantic",
            "health": f"{settings.API_V1_STR}/health",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
