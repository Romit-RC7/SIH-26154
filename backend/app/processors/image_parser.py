"""
Image Document Parser.

Treats a standalone image as a single-page document and emits
a RawDocumentElement for downstream semantic processing.
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image

from backend.app.processors.base import ParsedPage, RawDocumentElement
from backend.app.core.logging import logger


class ImageParser:

    def parse(
        self,
        file_path: Path
    ) -> Tuple[List[ParsedPage], List[RawDocumentElement], dict]:

        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        img = Image.open(file_path).convert("RGB")

        width, height = img.size

        page = ParsedPage(
            page_number=1,
            width=float(width),
            height=float(height),
            image=img,
            raw_text=None
        )

        element = RawDocumentElement(
            type="image",
            page=1,
            bbox=[0, 0, width, height],
            image=img,
            confidence=1.0,
            attributes={
                "source": "uploaded_image"
            }
        )

        metadata = {
            "title": file_path.stem,
            "page_count": 1,
            "image_width": width,
            "image_height": height
        }

        logger.info(f"Parsed image {file_path.name}")

        return [page], [element], metadata


image_parser = ImageParser()