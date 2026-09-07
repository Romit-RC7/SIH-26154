"""
API tests for Export endpoints (POST /api/v1/export/download).
"""

import pytest
from httpx import AsyncClient
from backend.app.schemas.intent import OutputType
from backend.app.schemas.generated_artefact import GeneratedArtefact


@pytest.mark.asyncio
async def test_export_api_download_docx(client: AsyncClient):
    artefact = GeneratedArtefact(
        artefact_id="art_api_exp_1",
        document_id="doc_api_exp_1",
        output_type=OutputType.EXECUTIVE_SUMMARY,
        content={
            "title": "API Export Test Report",
            "overview": "Overview text.",
            "key_findings": ["Finding 1"]
        }
    )

    payload = {
        "artefact": artefact.model_dump(),
        "format": "docx"
    }

    response = await client.post("/api/v1/export/download", json=payload)
    assert response.status_code == 200
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in response.headers.get("content-type", "")
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_export_api_download_video_zip(client: AsyncClient):
    artefact = GeneratedArtefact(
        artefact_id="art_api_exp_2",
        document_id="doc_api_exp_1",
        output_type=OutputType.VIDEO_SCRIPT,
        content={
            "video_title": "Video Package API Test",
            "scenes": [
                {
                    "scene_number": 1,
                    "visual_cue": "Intro animation",
                    "narration": "Narration script text.",
                    "duration_sec": 15
                }
            ]
        }
    )

    payload = {
        "artefact": artefact.model_dump(),
        "format": "zip"
    }

    response = await client.post("/api/v1/export/download", json=payload)
    assert response.status_code == 200
    assert "application/zip" in response.headers.get("content-type", "")
    assert len(response.content) > 0
