"""
Unit tests for Stage 6 Formatters (Docx, Pptx, Pdf, VideoPackage, InfographicPackage).
"""

import pytest
import zipfile
from pathlib import Path
from backend.app.schemas.intent import OutputType
from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.services.formatters.docx_formatter import docx_formatter
from backend.app.services.formatters.pptx_formatter import pptx_formatter
from backend.app.services.formatters.pdf_formatter import pdf_formatter
from backend.app.services.formatters.video_package_builder import video_package_builder
from backend.app.services.formatters.infographic_package_builder import infographic_package_builder


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "exports_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_docx_formatter(tmp_output_dir: Path):
    artefact = GeneratedArtefact(
        artefact_id="art_docx_1",
        document_id="doc_f_1",
        output_type=OutputType.EXECUTIVE_SUMMARY,
        content={
            "title": "Executive AI Report",
            "overview": "Overview text summarizing findings.",
            "key_findings": ["Finding A", "Finding B"],
            "recommendations": ["Recommendation 1"]
        }
    )
    target_path = tmp_output_dir / "report.docx"
    res_path = docx_formatter.generate_docx(artefact, target_path)

    assert res_path.exists()
    assert res_path.stat().st_size > 0


def test_pptx_formatter(tmp_output_dir: Path):
    artefact = GeneratedArtefact(
        artefact_id="art_pptx_1",
        document_id="doc_f_1",
        output_type=OutputType.PRESENTATION_DECK,
        content={
            "presentation_title": "AI Strategy Deck",
            "tagline": "2026 Vision",
            "slides": [
                {
                    "slide_number": 1,
                    "title": "Introduction",
                    "bullet_points": ["Point 1", "Point 2"],
                    "speaker_notes": "Welcome team.",
                    "visual_suggestion": "Bar chart"
                }
            ]
        }
    )
    target_path = tmp_output_dir / "deck.pptx"
    res_path = pptx_formatter.generate_pptx(artefact, target_path)

    assert res_path.exists()
    assert res_path.stat().st_size > 0


def test_pdf_formatter(tmp_output_dir: Path):
    artefact = GeneratedArtefact(
        artefact_id="art_pdf_1",
        document_id="doc_f_1",
        output_type=OutputType.EXECUTIVE_SUMMARY,
        content={
            "title": "PDF Advisory",
            "overview": "Advisory text summary.",
            "key_findings": ["Finding 1"]
        }
    )
    target_path = tmp_output_dir / "advisory.pdf"
    res_path = pdf_formatter.generate_pdf(artefact, target_path)

    assert res_path.exists()
    assert res_path.stat().st_size > 0


def test_video_package_builder(tmp_output_dir: Path):
    artefact = GeneratedArtefact(
        artefact_id="art_video_1",
        document_id="doc_f_1",
        output_type=OutputType.VIDEO_SCRIPT,
        content={
            "video_title": "AI Video Package",
            "target_duration_sec": 60,
            "scenes": [
                {
                    "scene_number": 1,
                    "visual_cue": "Intro animation",
                    "narration": "Welcome to the video package.",
                    "duration_sec": 15
                }
            ]
        }
    )
    target_path = tmp_output_dir / "video.zip"
    res_path = video_package_builder.generate_video_package(artefact, target_path)

    assert res_path.exists()
    assert zipfile.is_zipfile(res_path)

    with zipfile.ZipFile(res_path, "r") as z:
        names = z.namelist()
        assert "subtitles.srt" in names
        assert "script_storyboard.json" in names
        assert "visual_recommendations.md" in names


def test_infographic_package_builder(tmp_output_dir: Path):
    artefact = GeneratedArtefact(
        artefact_id="art_info_1",
        document_id="doc_f_1",
        output_type=OutputType.INFOGRAPHIC_BRIEF,
        content={
            "headline": "Visual Infographic Summary",
            "subheadline": "2026 Trends",
            "key_stats": [{"label": "Adoption", "value": "85%", "context": "Enterprise"}],
            "visual_sections": [{"title": "Growth", "description": "Section desc", "visual_type": "barchart"}]
        }
    )
    target_path = tmp_output_dir / "infographic.zip"
    res_path = infographic_package_builder.generate_infographic_package(artefact, target_path)

    assert res_path.exists()
    assert zipfile.is_zipfile(res_path)

    with zipfile.ZipFile(res_path, "r") as z:
        names = z.namelist()
        assert "infographic_blueprint.json" in names
        assert "render_template.html" in names
        assert "key_metrics_summary.md" in names
