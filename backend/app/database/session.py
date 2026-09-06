"""
Database session management with SQLAlchemy 2.0.
Supports asynchronous database access with asyncpg and synchronous access for migrations/fallback.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import settings
from backend.app.database.base import Base

# Async Engine and Session
async_engine = create_async_engine(
    settings.async_database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync Engine and Session (useful for sync workers, Alembic, or tests)
sync_engine = create_engine(
    settings.sync_database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for obtaining an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db() -> Session:
    """Helper for synchronous session."""
    db = SyncSessionLocal()
    try:
        return db
    finally:
        pass


async def init_db() -> None:
    """Initialize database tables."""
    # Import all models to ensure they are registered with Base.metadata
    from backend.app.models import document, document_element, document_chunk, processing_job  # noqa
    async with async_engine.begin() as conn:
        try:
            from sqlalchemy import text
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception:
            pass
        await conn.run_sync(Base.metadata.create_all)
