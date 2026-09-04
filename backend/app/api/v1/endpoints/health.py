"""
Health check and system diagnostics endpoint.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.api.deps import get_db
from backend.app.core.config import settings
from backend.app.processors.pp_structure import pp_structure_analyzer

router = APIRouter()


@router.get("", summary="System Health & Diagnostic Status")
async def health_check(db: AsyncSession = Depends(get_db)):
    # Verify DB connectivity
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "pp_structure_available": pp_structure_analyzer.is_available(),
        "configured_analyzer": settings.DOC_ANALYZER_ENGINE,
        "uploads_directory": str(settings.UPLOAD_DIR),
    }
