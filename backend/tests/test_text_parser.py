"""
Unit tests for TextParser.
Tests segmenting raw string input and .txt files into ParsedPage and RawDocumentElement instances.
"""

from pathlib import Path
import pytest
from backend.app.processors.text_parser import text_parser


def test_text_parser_basic_segmentation():
    sample_text = (
        "Cloud Security Advisory: Critical Vulnerability Discovered\n\n"
        "Security researchers have uncovered a severe flaw affecting Kubernetes clusters.\n\n"
        "All operators must update their ingress controllers immediately to version 1.9.0."
    )
    pages, elements, meta = text_parser.parse_text(sample_text, title="Kubernetes Advisory", mode="RAW_ARTICLE")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert len(elements) == 3

    assert elements[0].text.startswith("Cloud Security Advisory")
    assert elements[0].attributes["reading_order"] == 1
    assert elements[0].attributes["input_mode"] == "RAW_ARTICLE"

    assert elements[1].attributes["reading_order"] == 2
    assert elements[2].attributes["reading_order"] == 3

    assert meta["title"] == "Kubernetes Advisory"
    assert meta["paragraph_count"] == 3
    assert meta["word_count"] > 20


def test_text_parser_markdown_headers():
    sample_markdown = (
        "# Executive Summary: Q3 Financial Results\n\n"
        "Revenue increased by 28% year-over-year, driven by enterprise AI adoption.\n\n"
        "## Key Operational Milestones\n\n"
        "Over 500 new institutional clients were onboarded during the quarter."
    )
    pages, elements, meta = text_parser.parse_text(sample_markdown, mode="RAW_ARTICLE")

    assert len(elements) == 4
    assert elements[0].attributes.get("is_header") is True
    assert elements[1].attributes.get("is_header") is False
    assert elements[2].attributes.get("is_header") is True
    assert elements[3].attributes.get("is_header") is False
    assert meta["title"] == "Executive Summary: Q3 Financial Results"


def test_text_parser_file_parsing(tmp_path: Path):
    test_file = tmp_path / "incident_report.txt"
    test_file.write_text(
        "Incident 1042: API Gateway Throttling\n\n"
        "Traffic spikes exceeded auto-scaling limits by 300% between 14:00 and 14:45 UTC.\n\n"
        "Redundancy circuits successfully restored normal latency within 12 minutes.",
        encoding="utf-8"
    )

    pages, elements, meta = text_parser.parse(test_file)
    assert len(pages) == 1
    assert len(elements) == 3
    assert meta["title"] == "Incident Report"
