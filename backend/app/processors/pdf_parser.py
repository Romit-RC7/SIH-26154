"""
PDF Document Parser.
Renders PDF pages to high-resolution images for PP-StructureV3 and extracts base text & layout elements.
Uses PyMuPDF (fitz) for speed and fidelity.
"""

from pathlib import Path
from typing import List, Tuple
from PIL import Image
import fitz  # PyMuPDF
from backend.app.processors.base import ParsedPage, RawDocumentElement
from backend.app.core.config import settings
from backend.app.core.logging import logger


class PDFParser:
    """Renders PDF pages to images and extracts raw blocks and metadata."""

    def __init__(self, dpi: int | None = None):
        self.dpi = dpi if dpi is not None else settings.PDF_RENDER_DPI
        self.zoom = self.dpi / 72.0  # Default PDF resolution is 72 dpi

    def parse(self, file_path: Path) -> Tuple[List[ParsedPage], List[RawDocumentElement], dict]:
        """
        Parses a PDF file, rasterizing pages into PIL Images and extracting vector text blocks.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = fitz.open(str(file_path))
        pages: List[ParsedPage] = []
        raw_elements: List[RawDocumentElement] = []
        
        pdf_metadata = {
            "title": doc.metadata.get("title") or file_path.stem,
            "author": doc.metadata.get("author"),
            "subject": doc.metadata.get("subject"),
            "creator": doc.metadata.get("creator"),
            "page_count": len(doc),
        }

        matrix = fitz.Matrix(self.zoom, self.zoom)

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Rasterize page for PP-StructureV3
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n >= 3 else "L"
            page_image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            raw_text = page.get_text("text")

            pages.append(
                ParsedPage(
                    page_number=page_num,
                    width=page.rect.width,
                    height=page.rect.height,
                    image=page_image,
                    raw_text=raw_text
                )
            )

            # 2. Extract PyMuPDF text blocks (for hybrid fallback or supplementary layout)
            blocks = page.get_text("blocks")
            for b in blocks:
                # b: (x0, y0, x1, y1, "text", block_no, block_type)
                # block_type == 0: text, block_type == 1: image
                bbox = [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)]
                block_text = b[4].strip()
                block_type = "image" if b[6] == 1 else "text"

                if block_text or block_type == "image":
                    raw_elements.append(
                        RawDocumentElement(
                            type=block_type,
                            page=page_num,
                            bbox=bbox,
                            text=block_text if block_type == "text" else None,
                            confidence=1.0,
                            attributes={"source": "pymupdf_blocks", "block_no": b[5]}
                        )
                    )

        doc.close()
        logger.info(f"Parsed PDF {file_path.name}: {len(pages)} pages, {len(raw_elements)} raw blocks")
        return pages, raw_elements, pdf_metadata


pdf_parser = PDFParser()
