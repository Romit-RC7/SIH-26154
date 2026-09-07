"""
API v1 Router Aggregator.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import documents, health, knowledge, generate, validate, export

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(knowledge.router, prefix="/knowledge", tags=["Knowledge & Retrieval"])
api_router.include_router(generate.router, prefix="/generate", tags=["Content Generation"])
api_router.include_router(validate.router, prefix="/validate", tags=["Trust & Validation"])
api_router.include_router(export.router, prefix="/export", tags=["Multi-Format Export"])


