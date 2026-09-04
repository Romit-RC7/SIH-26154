"""
Unit tests for file utility functions:
- compute_sha256
- detect_mime_type
- validate_upload_filename
"""

from pathlib import Path
import pytest
from backend.app.utils.file_utils import compute_sha256, detect_mime_type, validate_upload_filename


def test_compute_sha256(tmp_path: Path):
    """Verify SHA-256 hash calculation."""
    f = tmp_path / "hash_test.txt"
    f.write_text("Hello SIH 2026", encoding="utf-8")
    hash_val = compute_sha256(f)
    assert len(hash_val) == 64
    assert isinstance(hash_val, str)


def test_detect_mime_type():
    """Verify MIME detection for pdf, docx, and fallback."""
    assert detect_mime_type("report.pdf") == "application/pdf"
    assert "wordprocessingml" in detect_mime_type("doc.docx")
    assert detect_mime_type("photo.png") == "image/png"
    assert detect_mime_type("unknown_ext.xyz123") == "application/octet-stream"


def test_validate_upload_filename():
    """Verify allowed and disallowed extensions."""
    allowed = [".pdf", ".docx"]

    valid_1, _ = validate_upload_filename("test.pdf", allowed)
    assert valid_1 is True

    valid_2, _ = validate_upload_filename("TEST.DOCX", allowed)
    assert valid_2 is True

    invalid_1, msg_1 = validate_upload_filename("", allowed)
    assert invalid_1 is False
    assert "cannot be empty" in msg_1

    invalid_2, msg_2 = validate_upload_filename("bad.exe", allowed)
    assert invalid_2 is False
    assert "Unsupported file extension" in msg_2
