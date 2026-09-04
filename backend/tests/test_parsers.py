"""
Tests for PDF and DOCX document parsers and extraction layer.
"""

from pathlib import Path
import pytest
from backend.app.processors.pdf_parser import pdf_parser
from backend.app.processors.docx_parser import docx_parser
from backend.app.processors.extractor import document_extractor


def test_pdf_parser(tmp_path: Path, sample_pdf_bytes: bytes):
    """Verify PyMuPDF rasterization and text extraction."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(sample_pdf_bytes)

    pages, raw_elements, metadata = pdf_parser.parse(pdf_file)

    assert len(pages) == 1
    assert pages[0].image is not None
    assert "AI Content Transformation" in pages[0].raw_text
    assert len(raw_elements) > 0
    assert metadata["page_count"] == 1


def test_docx_parser(tmp_path: Path, sample_docx_bytes: bytes):
    """Verify python-docx paragraph and table extraction."""
    docx_file = tmp_path / "sample.docx"
    docx_file.write_bytes(sample_docx_bytes)

    pages, raw_elements, metadata = docx_parser.parse(docx_file)

    assert len(raw_elements) >= 2
    # One of the elements must be a table with markdown representation
    table_elems = [e for e in raw_elements if e.type == "table"]
    assert len(table_elems) == 1
    assert "PP-StructureV3" in table_elems[0].markdown
    assert "<table>" in table_elems[0].html


def test_document_extractor_end_to_end(tmp_path: Path, sample_pdf_bytes: bytes):
    """Verify DocumentExtractor processes PDF and crops elements."""
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(sample_pdf_bytes)

    elements, meta = document_extractor.extract_document(pdf_file, "doc-test-extract")

    assert len(elements) > 0
    assert any(e.type == "text" for e in elements)
