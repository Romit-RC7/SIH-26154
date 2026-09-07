"""
Export Endpoints (Stage 6).
Exposes REST endpoints to generate and download multi-format deliverable files
(.docx, .pptx, .pdf, .zip video/infographic packages, .json).
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path as FastPath, Body, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.services.formatters.export_coordinator import export_coordinator

router = APIRouter()


MIME_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "pdf": "application/pdf",
    "zip": "application/zip",
    "json": "application/json",
    "txt": "text/plain",
}


class DirectExportRequest(BaseModel):
    artefact: GeneratedArtefact = Field(..., description="Generated deliverable artefact")
    format: Optional[str] = Field(default=None, description="docx, pptx, pdf, zip, json, or txt")


@router.post(
    "/download",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Download formatted deliverable file (.docx, .pptx, .pdf, .zip, .json)",
    description="Generates physical deliverable file and streams download response with appropriate headers."
)
async def download_artefact_file(
    request: DirectExportRequest = Body(...)
) -> FileResponse:
    """
    Generates and returns physical deliverable file for download.
    """
    try:
        exported_path = export_coordinator.export_artefact(
            artefact=request.artefact,
            target_format=request.format
        )

        ext = exported_path.suffix.lstrip(".").lower()
        media_type = MIME_TYPES.get(ext, "application/octet-stream")
        filename = f"{request.artefact.output_type.value}_{request.artefact.artefact_id[:8]}.{ext}"

        return FileResponse(
            path=str(exported_path),
            media_type=media_type,
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building export file: {str(exc)}"
        )
