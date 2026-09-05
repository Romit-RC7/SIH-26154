"""
Health check and system diagnostics endpoint.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db
from backend.app.core.config import settings
from backend.app.services.system_diagnostics import system_diagnostics

router = APIRouter()


@router.get("", summary="System Health & Diagnostic Status")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Returns:
    - API status
    - Database connectivity
    - Model availability
    - Engine availability
    - Storage availability
    """

    db_connected = True
    db_error = None

    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_connected = False
        db_error = str(e)

    diagnostics = system_diagnostics.get_status()

    return {
        "status": "online" if db_connected else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,

        "database": {
            "connected": db_connected,
            "error": db_error,
        },

        **diagnostics,
    }