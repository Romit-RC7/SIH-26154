"""
API Dependencies.
Provides database sessions and common dependency injection components.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.database.session import get_db

# Re-export get_db for cleaner imports in endpoints
__all__ = ["get_db"]
