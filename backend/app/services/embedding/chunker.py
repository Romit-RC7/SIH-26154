"""
Semantic and Structural Document Chunker.
Transforms SemanticDocument JSON elements into dense-embedding-ready chunks while preserving
provenance (element IDs, page numbers, types, bounding boxes) and table headers across splits.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from backend.app.schemas.semantic_document import SemanticDocument, SemanticElement
from backend.app.models.document_chunk import ChunkType
from backend.app.services.embedding.text_cleaner import TextCleaner


@dataclass
class ChunkItem:
    """Represents a generated semantic chunk ready for vector embedding and pgvector persistence."""
    element_id: str
    chunk_index: int
    chunk_type: ChunkType
    page: int
    content: str
    cleaned_text: str
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """
    Splits multi-modal elements from SemanticDocument into coherent semantic chunks.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        min_chunk_size: int = 30,
        include_context_prefix: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.include_context_prefix = include_context_prefix

    def chunk_document(self, doc: SemanticDocument) -> List[ChunkItem]:
        """
        Processes all elements of a SemanticDocument into a flat list of ChunkItem instances.
        """
        all_chunks: List[ChunkItem] = []
        global_index = 0
        doc_title = doc.metadata.title or doc.metadata.file_name

        for elem in doc.elements:
            elem_chunks = self.chunk_element(elem, doc_title=doc_title)
            for ch in elem_chunks:
                ch.chunk_index = global_index
                global_index += 1
                all_chunks.append(ch)

        return all_chunks

    def chunk_element(self, elem: SemanticElement, doc_title: Optional[str] = None) -> List[ChunkItem]:
        """
        Dispatches element chunking based on its semantic type.
        """
        elem_type = elem.type.lower()
        if elem_type == "table":
            return self._chunk_table_element(elem, doc_title)
        elif elem_type in ("figure", "chart", "image"):
            return self._chunk_visual_element(elem, doc_title)
        else:
            return self._chunk_text_element(elem, doc_title)

    def _chunk_text_element(self, elem: SemanticElement, doc_title: Optional[str]) -> List[ChunkItem]:
        """
        Chunks text paragraphs and headings with sentence-boundary awareness.
        """
        raw_text = elem.content.text or ""
        cleaned = TextCleaner.clean(raw_text)
        if not cleaned or len(cleaned) < self.min_chunk_size:
            if not cleaned:
                return []

        # Determine role (title, heading, body)
        raw_attrs = elem.content.raw_attributes or {}
        role = raw_attrs.get("role", "paragraph")
        chunk_type = ChunkType.HEADER if role in ("title", "heading", "header") else ChunkType.TEXT

        # If text is small enough, return as single chunk
        if len(cleaned) <= self.chunk_size:
            prefix = self._build_prefix(doc_title, elem.page, role, elem.id) if self.include_context_prefix else ""
            emb_text = f"{prefix}{cleaned}".strip()
            return [
                ChunkItem(
                    element_id=elem.id,
                    chunk_index=0,
                    chunk_type=chunk_type,
                    page=elem.page,
                    content=raw_text,
                    cleaned_text=emb_text,
                    chunk_metadata={
                        "role": role,
                        "bbox": elem.bbox,
                        "reading_order": elem.content.reading_order,
                        "confidence": elem.content.confidence,
                        "raw_length": len(raw_text),
                    }
                )
            ]

        # Split larger text into overlapping window chunks
        splits = self._split_text_with_overlap(cleaned, self.chunk_size, self.chunk_overlap)
        chunks: List[ChunkItem] = []
        for idx, segment in enumerate(splits):
            prefix = self._build_prefix(doc_title, elem.page, role, elem.id) if self.include_context_prefix else ""
            emb_text = f"{prefix}{segment}".strip()
            chunks.append(
                ChunkItem(
                    element_id=elem.id,
                    chunk_index=idx,
                    chunk_type=chunk_type,
                    page=elem.page,
                    content=segment,
                    cleaned_text=emb_text,
                    chunk_metadata={
                        "role": role,
                        "bbox": elem.bbox,
                        "reading_order": elem.content.reading_order,
                        "confidence": elem.content.confidence,
                        "split_index": idx,
                        "total_splits": len(splits),
                    }
                )
            )
        return chunks

    def _chunk_table_element(self, elem: SemanticElement, doc_title: Optional[str]) -> List[ChunkItem]:
        """
        Chunks structured tables while preserving markdown header syntax across chunks.
        """
        table_md = elem.content.markdown or elem.content.text or ""
        cleaned_md = TextCleaner.clean_table_markdown(table_md)
        caption = elem.content.caption or ""

        if not cleaned_md and not caption:
            return []

        # If entire table fits in chunk_size * 1.5, keep intact
        if len(cleaned_md) <= self.chunk_size * 1.5:
            full_table_text = f"Table Caption: {caption}\n{cleaned_md}" if caption else cleaned_md
            prefix = self._build_prefix(doc_title, elem.page, "table", elem.id) if self.include_context_prefix else ""
            emb_text = f"{prefix}{full_table_text}".strip()
            return [
                ChunkItem(
                    element_id=elem.id,
                    chunk_index=0,
                    chunk_type=ChunkType.TABLE,
                    page=elem.page,
                    content=cleaned_md,
                    cleaned_text=emb_text,
                    chunk_metadata={
                        "caption": caption,
                        "bbox": elem.bbox,
                        "reading_order": elem.content.reading_order,
                        "table_structure": elem.content.table_structure,
                    }
                )
            ]

        # For very large tables: chunk by rows while repeating the header
        lines = cleaned_md.splitlines()
        header_lines = lines[:2] if len(lines) >= 2 and "|" in lines[0] and "-" in lines[1] else []
        data_lines = lines[2:] if header_lines else lines

        chunks: List[ChunkItem] = []
        current_rows: List[str] = []
        current_len = 0
        part = 0

        header_str = "\n".join(header_lines) + "\n" if header_lines else ""

        for row in data_lines:
            row_len = len(row) + 1
            if current_rows and (current_len + row_len > self.chunk_size):
                table_chunk_str = header_str + "\n".join(current_rows)
                if caption:
                    table_chunk_str = f"Table (Part {part+1}): {caption}\n{table_chunk_str}"
                prefix = self._build_prefix(doc_title, elem.page, "table", elem.id) if self.include_context_prefix else ""
                emb_text = f"{prefix}{table_chunk_str}".strip()

                chunks.append(
                    ChunkItem(
                        element_id=elem.id,
                        chunk_index=part,
                        chunk_type=ChunkType.TABLE,
                        page=elem.page,
                        content=table_chunk_str,
                        cleaned_text=emb_text,
                        chunk_metadata={
                            "caption": caption,
                            "bbox": elem.bbox,
                            "part": part + 1,
                            "table_structure": elem.content.table_structure,
                        }
                    )
                )
                part += 1
                current_rows = []
                current_len = 0

            current_rows.append(row)
            current_len += row_len

        if current_rows:
            table_chunk_str = header_str + "\n".join(current_rows)
            if caption:
                table_chunk_str = f"Table (Part {part+1}): {caption}\n{table_chunk_str}"
            prefix = self._build_prefix(doc_title, elem.page, "table", elem.id) if self.include_context_prefix else ""
            emb_text = f"{prefix}{table_chunk_str}".strip()

            chunks.append(
                ChunkItem(
                    element_id=elem.id,
                    chunk_index=part,
                    chunk_type=ChunkType.TABLE,
                    page=elem.page,
                    content=table_chunk_str,
                    cleaned_text=emb_text,
                    chunk_metadata={
                        "caption": caption,
                        "bbox": elem.bbox,
                        "part": part + 1,
                        "table_structure": elem.content.table_structure,
                    }
                )
            )

        return chunks

    def _chunk_visual_element(self, elem: SemanticElement, doc_title: Optional[str]) -> List[ChunkItem]:
        """
        Chunks figure, chart, or image elements using caption, OCR text, and visual analysis.
        """
        parts = []
        caption = elem.content.caption or ""
        ocr_text = elem.content.text or ""
        raw_attrs = elem.content.raw_attributes or {}
        visual_analysis = raw_attrs.get("visual_analysis", "") or raw_attrs.get("description", "")
        if isinstance(visual_analysis, dict):
            visual_analysis = visual_analysis.get("_raw_text") or visual_analysis.get("summary") or " ".join(str(v) for v in visual_analysis.values())
        elif isinstance(visual_analysis, list):
            visual_analysis = " ".join(str(v) for v in visual_analysis)
        else:
            visual_analysis = str(visual_analysis) if visual_analysis else ""

        if caption:
            parts.append(f"Caption: {caption}")
        if visual_analysis:
            parts.append(f"Visual Analysis: {visual_analysis}")
        if ocr_text:
            parts.append(f"Visual Text Content: {ocr_text}")

        combined_desc = "\n".join(parts)
        if not combined_desc:
            combined_desc = f"{elem.type.title()} on page {elem.page}"

        cleaned = TextCleaner.clean(combined_desc)
        chunk_type = ChunkType.CHART_DATA if elem.type == "chart" else ChunkType.FIGURE_CAPTION

        prefix = self._build_prefix(doc_title, elem.page, elem.type, elem.id) if self.include_context_prefix else ""
        emb_text = f"{prefix}{cleaned}".strip()

        return [
            ChunkItem(
                element_id=elem.id,
                chunk_index=0,
                chunk_type=chunk_type,
                page=elem.page,
                content=combined_desc,
                cleaned_text=emb_text,
                chunk_metadata={
                    "element_type": elem.type,
                    "caption": caption,
                    "image_path": elem.content.image_path,
                    "bbox": elem.bbox,
                    "reading_order": elem.content.reading_order,
                }
            )
        ]

    def _build_prefix(self, doc_title: Optional[str], page: int, element_type: str, elem_id: str) -> str:
        """
        Constructs a lightweight metadata header prefixed to the chunk embedding text.
        """
        title_part = f"Document: {doc_title} | " if doc_title else ""
        return f"[{title_part}Page {page} | Type: {element_type}] "

    def _split_text_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Splits text into chunks of at most `chunk_size` characters with `overlap` overlap,
        breaking at sentences or word boundaries where possible.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Try to break at sentence or paragraph boundary if not at end
            if end < text_len:
                # Look backwards for sentence end
                boundary = -1
                for punct in (". ", ".\n", "!\n", "?\n", "! ", "? ", "\n\n", "\n", "; ", ", "):
                    pos = text.rfind(punct, start + int(chunk_size * 0.5), end)
                    if pos != -1:
                        boundary = pos + len(punct)
                        break
                if boundary != -1:
                    end = boundary

            chunk_str = text[start:end].strip()
            if chunk_str and len(chunk_str) >= self.min_chunk_size:
                chunks.append(chunk_str)

            if end >= text_len:
                break

            start = max(end - overlap, start + 1)

        return chunks
