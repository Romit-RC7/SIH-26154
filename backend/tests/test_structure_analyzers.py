"""
Unit tests for PP-StructureV3 integration, Layout recognition, HTML table conversion,
and Automatic Fallback Analyzer execution.
"""

from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
import pymupdf as fitz

from backend.app.processors.pp_structure import PPStructureAnalyzer, pp_structure_analyzer
from backend.app.processors.fallback_analyzer import FallbackStructureAnalyzer, fallback_analyzer
from backend.app.processors.base import RawDocumentElement


def test_html_to_markdown_table_converter():
    """Verify PPStructureAnalyzer static table conversion from HTML to clean Markdown table."""
    html_sample = (
        "<table>"
        "<tr><th>Region</th><th>Revenue</th><th>Growth</th></tr>"
        "<tr><td>North America</td><td>$12M</td><td>14%</td></tr>"
        "<tr><td>Europe</td><td>$8M</td><td>9%</td></tr>"
        "<tr><td>Asia Pacific</td><td>$15M</td><td>22%</td></tr>"
        "</table>"
    )
    md = PPStructureAnalyzer._html_to_markdown_table(html_sample)
    assert md is not None
    assert "| Region | Revenue | Growth |" in md
    assert "| --- | --- | --- |" in md
    assert "| North America | $12M | 14% |" in md
    assert "| Asia Pacific | $15M | 22% |" in md


def test_html_to_markdown_table_empty():
    """Verify converter returns None on empty or invalid table strings."""
    assert PPStructureAnalyzer._html_to_markdown_table("") is None
    assert PPStructureAnalyzer._html_to_markdown_table("<div>No table here</div>") is None


def test_pp_structure_analyzer_parsing_mock():
    """
    Verify PPStructureAnalyzer layout parsing and element extraction using
    representative PP-StructureV3 inference payloads.
    """
    analyzer = PPStructureAnalyzer()
    analyzer._initialized = True

    # Mock PPStructure engine output format
    mock_engine_output = [
        {
            "type": "title",
            "bbox": [50, 40, 500, 80],
            "res": "Autonomous AI Multi-Modal Engine"
        },
        {
            "type": "text",
            "bbox": [50, 90, 500, 160],
            "res": [
                ({"text": "This document outlines the system architecture.", "confidence": 0.98}),
                ({"text": "All modules communicate via Semantic Document JSON.", "confidence": 0.96})
            ]
        },
        {
            "type": "table",
            "bbox": [50, 180, 500, 320],
            "res": {
                "html": "<table><tr><th>Component</th><th>Tech</th></tr><tr><td>OCR</td><td>PP-Structure</td></tr></table>",
                "text": "Component Tech OCR PP-Structure"
            }
        },
        {
            "type": "figure",
            "bbox": [60, 340, 450, 550],
            "res": []
        }
    ]

    analyzer.engine = MagicMock(return_value=mock_engine_output)

    # Test analyze_page on a sample test image
    test_img = Image.new("RGB", (600, 800), color="white")
    elements = analyzer.analyze_page(test_img, page_number=1)

    assert len(elements) == 4

    # Element 1: Title -> mapped to 'text'
    assert elements[0].type == "text"
    assert elements[0].text == "Autonomous AI Multi-Modal Engine"
    assert elements[0].bbox == [50.0, 40.0, 500.0, 80.0]

    # Element 2: Text block OCR
    assert elements[1].type == "text"
    assert "Autonomous AI" not in elements[1].text
    assert "system architecture" in elements[1].text
    assert elements[1].confidence >= 0.95

    # Element 3: Table with markdown and HTML
    assert elements[2].type == "table"
    assert elements[2].markdown is not None
    assert "| Component | Tech |" in elements[2].markdown
    assert elements[2].image is not None  # Cropped region

    # Element 4: Figure
    assert elements[3].type == "figure"
    assert elements[3].bbox == [60.0, 340.0, 450.0, 550.0]
    assert elements[3].image is not None  # Cropped region


def test_fallback_analyzer_direct_page_analysis():
    """Verify FallbackStructureAnalyzer on a live PyMuPDF page with text and table."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), "Fallback Architecture Verification", fontsize=16)
    page.insert_text((50, 90), "Section 1: Automatic switch to rule-based analyzer when PP-Structure is absent.", fontsize=11)

    img = Image.new("RGB", (595, 842), color="white")
    elements = fallback_analyzer.analyze_pdf_page_directly(page, page_number=1, rendered_image=img)

    assert len(elements) >= 2
    assert any(e.type == "text" and "Fallback Architecture" in e.text for e in elements)
    doc.close()


def test_fallback_analyzer_when_pp_structure_unavailable():
    """
    Force PP-Structure to be unavailable and verify fallback analyzer executes smoothly
    without crashing the pipeline.
    """
    analyzer = PPStructureAnalyzer()
    analyzer._initialized = False
    analyzer.engine = None

    assert analyzer.is_available() is False

    with pytest.raises(RuntimeError, match="PP-Structure engine is not available"):
        test_img = Image.new("RGB", (200, 200), color="white")
        analyzer.analyze_page(test_img, page_number=1)

    # Fallback analyzer runs independently
    fallback = FallbackStructureAnalyzer()
    fb_elements = fallback.analyze_page(test_img, page_number=1)
    assert len(fb_elements) > 0
    assert fb_elements[0].type == "text"
