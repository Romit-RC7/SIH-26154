"""
File utility helpers for validation, hashing, and MIME type resolution.
"""

import hashlib
import mimetypes
from pathlib import Path
from typing import List, Tuple
from fastapi import HTTPException, status


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def detect_mime_type(filename: str) -> str:
    """Determine MIME type from filename extension."""
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    if filename.lower().endswith(".pdf"):
        return "application/pdf"
    if filename.lower().endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def validate_upload_filename(filename: str, allowed_extensions: List[str]) -> Tuple[bool, str]:
    """Validate file extension against allowed list."""
    if not filename:
        return False, "Filename cannot be empty"
    ext = Path(filename).suffix.lower()
    if ext not in [e.lower() for e in allowed_extensions]:
        return False, f"Unsupported file extension '{ext}'. Allowed: {allowed_extensions}"
    return True, ""
