"""
DOCX Document Parser.
Extracts text paragraphs, headings, tables (as structured markdown/html),
and embedded images from Microsoft Word .docx documents.
"""

import io
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import docx
from docx.table import Table as DocxTable
from backend.app.processors.base import ParsedPage, RawDocumentElement
from backend.app.core.logging import logger


class DOCXParser:
    """Extracts text blocks, structured tables, and media from DOCX files."""

    def parse(self, file_path: Path) -> Tuple[List[ParsedPage], List[RawDocumentElement], dict]:
        if not file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        doc = docx.Document(str(file_path))
        raw_elements: List[RawDocumentElement] = []
        full_text_parts: List[str] = []

        page_num = 1  # DOCX is flow-based, treated as logical sections or single page sequence
        reading_idx = 1

        # Iterate over block elements in the document
        for block in doc.element.body:
            tag = block.tag

            # 1. Paragraphs (Text / Headings)
            if tag.endswith("p"):
                # Find matching paragraph
                p_text = block.text.strip() if block.text else ""
                if not p_text:
                    continue

                full_text_parts.append(p_text)
                raw_elements.append(
                    RawDocumentElement(
                        type="text",
                        page=page_num,
                        text=p_text,
                        confidence=1.0,
                        attributes={"reading_order": reading_idx}
                    )
                )
                reading_idx += 1

            # 2. Tables
            elif tag.endswith("tbl"):
                # Convert tbl to DocxTable
                tbl = DocxTable(block, doc)
                rows_data = []
                for row in tbl.rows:
                    row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    rows_data.append(row_cells)

                if rows_data:
                    # Construct markdown representation
                    headers = rows_data[0]
                    md_table = "| " + " | ".join(headers) + " |\n"
                    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in rows_data[1:]:
                        # Pad row if columns differ
                        padded_row = row + [""] * (len(headers) - len(row))
                        md_table += "| " + " | ".join(padded_row[:len(headers)]) + " |\n"

                    # HTML representation
                    html_table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
                    for row in rows_data[1:]:
                        html_table += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                    html_table += "</tbody></table>"

                    raw_elements.append(
                        RawDocumentElement(
                            type="table",
                            page=page_num,
                            text="\n".join(" | ".join(r) for r in rows_data),
                            markdown=md_table,
                            html=html_table,
                            confidence=1.0,
                            table_data={"rows": rows_data},
                            attributes={"reading_order": reading_idx}
                        )
                    )
                    reading_idx += 1

        # 3. Extract Embedded Images
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    img_part = rel.target_part
                    img_bytes = img_part.blob
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    raw_elements.append(
                        RawDocumentElement(
                            type="image",
                            page=page_num,
                            image=pil_img,
                            confidence=1.0,
                            attributes={"source": "docx_media"}
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not extract docx image: {e}")

        metadata = {
            "title": file_path.stem,
            "page_count": 1,
            "paragraphs_count": len(full_text_parts),
        }

        pages = [
            ParsedPage(
                page_number=1,
                width=612.0,  # Standard letter
                height=792.0,
                image=None,
                raw_text="\n\n".join(full_text_parts)
            )
        ]

        logger.info(f"Parsed DOCX {file_path.name}: {len(raw_elements)} elements")
        return pages, raw_elements, metadata


docx_parser = DOCXParser()
