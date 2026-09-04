"""
Unit tests for StorageService:
- Saving uploaded file stream
- Cropping and persisting PIL images
- Saving raw image bytes
- Deletion of raw and extracted document artifacts
"""

from pathlib import Path
from PIL import Image
import pytest
from backend.app.services.storage_service import storage_service
from backend.app.core.config import settings


def test_save_image_crop_pil_and_bytes(tmp_path: Path):
    """Verify saving image crops both from PIL Image objects and byte strings."""
    doc_id = "test-doc-crop-123"
    elem_id_1 = "crop_pil_1"
    elem_id_2 = "crop_bytes_2"

    # 1. PIL Image
    pil_img = Image.new("RGB", (100, 100), color="blue")
    rel_path_1 = storage_service.save_image_crop(pil_img, doc_id, elem_id_1, ext="png")
    assert rel_path_1.endswith(f"{elem_id_1}.png")
    full_path_1 = settings.BASE_DIR / rel_path_1
    assert full_path_1.exists()

    # 2. Raw Bytes
    raw_png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    rel_path_2 = storage_service.save_image_crop(raw_png_bytes, doc_id, elem_id_2, ext="png")
    assert rel_path_2.endswith(f"{elem_id_2}.png")
    full_path_2 = settings.BASE_DIR / rel_path_2
    assert full_path_2.exists()

    # 3. Clean up
    storage_service.delete_document_artifacts(doc_id)
    assert not full_path_1.exists()
    assert not full_path_2.exists()
