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


def test_pp_structure_v3_native_parsing():
    """
    Verify PPStructureAnalyzer parsing of native PaddleOCR 3.7.0 output structures
    including LayoutBlock objects, reading order, tables, images, charts, headers, and footers.
    """
    class MockLayoutBlock:
        def __init__(self, label, bbox, content="", order_index=None, image=None):
            self.label = label
            self.bbox = bbox
            self.content = content
            self.order_index = order_index
            self.image = image

    sample_crop = Image.new("RGB", (100, 100), color="blue")

    mock_v3_res = {
        "parsing_res_list": [
            MockLayoutBlock(
                label="header",
                bbox=[50, 20, 500, 40],
                content="CONFIDENTIAL REPORT - Q3",
                order_index=0
            ),
            MockLayoutBlock(
                label="paragraph_title",
                bbox=[50, 50, 400, 80],
                content="Executive Summary",
                order_index=1
            ),
            MockLayoutBlock(
                label="image",
                bbox=[50, 90, 250, 250],
                content="Architecture Diagram",
                order_index=2,
                image={"img": sample_crop, "path": "imgs/crop1.png"}
            ),
            MockLayoutBlock(
                label="chart",
                bbox=[260, 90, 500, 250],
                content="Growth Trends Chart",
                order_index=3
            ),
            MockLayoutBlock(
                label="figure_title",
                bbox=[50, 260, 500, 280],
                content="Figure 1: High Level System Architecture and Q3 Growth",
                order_index=4
            ),
            MockLayoutBlock(
                label="table",
                bbox=[50, 300, 500, 450],
                content="<table><tr><th>Metric</th><th>Score</th></tr><tr><td>Accuracy</td><td>99.4%</td></tr></table>",
                order_index=5
            ),
            MockLayoutBlock(
                label="footer",
                bbox=[50, 780, 500, 800],
                content="Page 1 of 10",
                order_index=6
            )
        ]
    }

    analyzer = PPStructureAnalyzer()
    analyzer._initialized = True
    analyzer.engine = MagicMock()
    analyzer.engine.predict.return_value = [mock_v3_res]

    test_img = Image.new("RGB", (600, 850), color="white")
    elements = analyzer.analyze_page(test_img, page_number=1)

    assert len(elements) == 7

    # Element 0: Header -> text with role 'header'
    assert elements[0].type == "text"
    assert elements[0].text == "CONFIDENTIAL REPORT - Q3"
    assert elements[0].attributes["role"] == "header"
    assert elements[0].attributes["reading_order"] == 0

    # Element 1: Title -> text with role 'title'
    assert elements[1].type == "text"
    assert elements[1].text == "Executive Summary"
    assert elements[1].attributes["role"] == "title"

    # Element 2: Image -> image with pre-cropped image
    assert elements[2].type == "image"
    assert elements[2].image is not None
    assert elements[2].attributes["role"] == "image"

    # Element 3: Chart -> chart with cropped region
    assert elements[3].type == "chart"
    assert elements[3].image is not None
    assert elements[3].attributes["role"] == "chart"

    # Element 4: Caption -> caption
    assert elements[4].type == "text"
    assert elements[4].attributes["role"] == "caption"

    # Element 5: Table -> table with html and markdown
    assert elements[5].type == "table"
    assert elements[5].html is not None
    assert elements[5].markdown is not None
    assert "| Metric | Score |" in elements[5].markdown
    assert "| Accuracy | 99.4% |" in elements[5].markdown
    assert elements[5].image is not None

    # Element 6: Footer -> text with role 'footer'
    assert elements[6].type == "text"
    assert elements[6].text == "Page 1 of 10"
    assert elements[6].attributes["role"] == "footer"


def test_pp_structure_initialization_failure_fallback():
    """Verify analyzer handles initialization errors cleanly without crashing."""
    with patch("paddleocr.PPStructureV3", side_effect=ImportError("Mocked missing module")):
        analyzer = PPStructureAnalyzer()
        assert analyzer.is_available() is False
        assert analyzer.engine is None


def test_extractor_runtime_failure_fallback(tmp_path):
    """
    Verify that if PPStructureAnalyzer fails at runtime during extraction,
    DocumentExtractor catches the exception, logs it, and falls back to PyMuPDF.
    """
    from backend.app.processors.extractor import DocumentExtractor

    # Create a small valid test PDF
    pdf_path = tmp_path / "test_runtime_fail.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), "Runtime Fallback Recovery Test", fontsize=14)
    doc.save(str(pdf_path))
    doc.close()

    extractor = DocumentExtractor()

    # Mock pp_structure_analyzer to be available but fail during analyze_page
    with patch("backend.app.processors.extractor.pp_structure_analyzer.is_available", return_value=True), \
         patch("backend.app.processors.extractor.pp_structure_analyzer.analyze_page", side_effect=RuntimeError("Simulated C++ runtime fault")):
        elements, meta = extractor.extract_document(pdf_path, document_id="doc_failover_test")

        # Must succeed via PyMuPDF fallback
        assert len(elements) > 0
        assert any("Runtime Fallback Recovery Test" in (e.text or "") for e in elements)
        assert meta["extracted_elements_count"] == len(elements)

