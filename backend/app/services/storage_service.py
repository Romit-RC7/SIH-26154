"""
Local Filesystem Storage Service.
Handles safe saving of uploaded PDFs/DOCX and cropped element artifacts (figures, charts, tables).
"""

import shutil
from pathlib import Path
from typing import Union
from fastapi import UploadFile
from PIL import Image
from backend.app.core.config import settings
from backend.app.core.logging import logger


class StorageService:
    """Manages raw document uploads and extracted visual artifacts."""

    def __init__(self):
        self.raw_dir = settings.RAW_UPLOAD_DIR
        self.extracted_dir = settings.EXTRACTED_UPLOAD_DIR
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    async def save_uploaded_file(self, file: UploadFile, document_id: str) -> Path:
        """
        Saves an uploaded FastAPI UploadFile asynchronously to disk using document_id.
        """
        extension = Path(file.filename).suffix
        target_filename = f"{document_id}{extension}"
        target_path = self.raw_dir / target_filename

        try:
            with open(target_path, "wb") as buffer:
                while content := await file.read(1024 * 1024):  # 1MB chunks
                    buffer.write(content)
            logger.info(f"Saved uploaded file to {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"Failed to save upload {file.filename}: {str(e)}")
            if target_path.exists():
                target_path.unlink()
            raise

    def save_image_crop(
        self,
        image: Union[Image.Image, bytes],
        document_id: str,
        element_id: str,
        ext: str = "png"
    ) -> str:
        """
        Saves a cropped PIL Image or bytes to the extracted artifact folder.
        Returns the relative path for client consumption and semantic JSON reference.
        """
        doc_extract_dir = self.extracted_dir / document_id
        doc_extract_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{element_id}.{ext}"
        file_path = doc_extract_dir / filename

        if isinstance(image, Image.Image):
            image.save(file_path, format=ext.upper())
        elif isinstance(image, bytes):
            with open(file_path, "wb") as f:
                f.write(image)

        # Return path relative to base directory
        relative_path = str(file_path.relative_to(settings.BASE_DIR)).replace("\\", "/")
        return relative_path

    def delete_document_artifacts(self, document_id: str):
        """Cleans up raw file and extracted directory for a document."""
        # Clean raw files matching document_id
        for f in self.raw_dir.glob(f"{document_id}.*"):
            try:
                f.unlink()
            except OSError:
                pass

        # Clean extracted directory
        doc_extract_dir = self.extracted_dir / document_id
        if doc_extract_dir.exists():
            shutil.rmtree(doc_extract_dir, ignore_errors=True)

    def save_image_crop(
        self,
        image,
        document_id: str,
        element_id: str,
        ext: str = "png"
    ) -> str:

        doc_extract_dir = self.extracted_dir / document_id
        doc_extract_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{element_id}.{ext}"
        file_path = doc_extract_dir / filename

        logger.info(f"Saving image crop -> {file_path}")

        if isinstance(image, Image.Image):
            image.save(file_path, format=ext.upper())

        elif isinstance(image, bytes):
            with open(file_path, "wb") as f:
                f.write(image)

        logger.info(f"Saved image crop -> {file_path}")

        return str(file_path.relative_to(settings.BASE_DIR)).replace("\\", "/")


storage_service = StorageService()
