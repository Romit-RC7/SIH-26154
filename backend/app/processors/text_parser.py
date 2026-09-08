"""
Text Document Parser.
Ingests direct text strings and .txt files, parsing them into structured
ParsedPage and RawDocumentElement instances for semantic processing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from backend.app.core.logging import logger
from backend.app.processors.base import ParsedPage, RawDocumentElement


class TextParser:
    """Parses raw text submissions and .txt files into document elements."""

    DEFAULT_PAGE_WIDTH: float = 612.0
    DEFAULT_PAGE_HEIGHT: float = 792.0

    def parse(self, file_path: Path) -> Tuple[List[ParsedPage], List[RawDocumentElement], dict]:
        """Parses a .txt file from disk."""
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1", errors="replace")

        title = file_path.stem.replace("_", " ").title()
        return self.parse_text(content, title=title, mode="RAW_ARTICLE")

    def parse_text(
        self,
        text: str,
        title: Optional[str] = None,
        mode: str = "RAW_ARTICLE"
    ) -> Tuple[List[ParsedPage], List[RawDocumentElement], dict]:
        """
        Segments raw text into structured ParsedPage and RawDocumentElement objects.
        """
        raw_text = text.strip()
        if not raw_text:
            return [], [], {"title": title or "Empty Document", "page_count": 0}

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw_text) if p.strip()]
        if not paragraphs:
            paragraphs = [raw_text]

        # Extract title from first line if not explicitly provided
        derived_title = title
        if not derived_title:
            first_line = paragraphs[0].split("\n")[0].strip()
            # Clean markdown formatting from title candidate
            first_line_clean = re.sub(r"^[#\s\-*]+", "", first_line).strip()
            if 3 <= len(first_line_clean) <= 80:
                derived_title = first_line_clean
            else:
                derived_title = "Direct Ingested Document"

        raw_elements: List[RawDocumentElement] = []
        reading_order = 1
        page_num = 1

        for idx, paragraph in enumerate(paragraphs):
            # Check if paragraph is markdown header
            is_header = bool(re.match(r"^#{1,6}\s+", paragraph))
            elem_type = "text"

            elem = RawDocumentElement(
                type=elem_type,
                page=page_num,
                text=paragraph,
                confidence=1.0,
                attributes={
                    "reading_order": reading_order,
                    "input_mode": mode,
                    "is_header": is_header,
                    "paragraph_index": idx + 1,
                }
            )
            raw_elements.append(elem)
            reading_order += 1

        page = ParsedPage(
            page_number=1,
            width=self.DEFAULT_PAGE_WIDTH,
            height=self.DEFAULT_PAGE_HEIGHT,
            image=None,
            raw_text=raw_text,
        )

        words = re.findall(r"\b\w+\b", raw_text)
        metadata = {
            "title": derived_title,
            "page_count": 1,
            "input_mode": mode,
            "word_count": len(words),
            "char_count": len(raw_text),
            "paragraph_count": len(paragraphs),
        }

        logger.info(
            "Parsed text document '%s': %d paragraphs, %d words (mode: %s)",
            derived_title, len(paragraphs), len(words), mode
        )

        return [page], raw_elements, metadata


text_parser = TextParser()

__all__ = ["TextParser", "text_parser"]
