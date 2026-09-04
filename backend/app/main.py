"""
FastAPI Application Entrypoint for SIH-26154 Semantic Document Processing System.
"""

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
    except Exception as e:
        logger.warning(f"Database initialization warning (will retry on requests): {e}")

    yield

    # Shutdown
    logger.info("Shutting down Semantic Document Processing System.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Foundational Semantic Document Processing Engine for Smart India Hackathon (SIH-26154).",
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


@app.get("/", tags=["Root"])
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
