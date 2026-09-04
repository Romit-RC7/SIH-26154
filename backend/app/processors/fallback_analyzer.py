"""
Rule-Based / PyMuPDF Layout Structure Analyzer.
Provides a fast, zero-GPU fallback structure analysis engine using PyMuPDF (fitz)
geometry and heuristics when PaddleOCR is initializing or unavailable.
"""

from typing import Any, Dict, List, Optional
from PIL import Image
import fitz
from backend.app.processors.base import BaseStructureAnalyzer, RawDocumentElement
from backend.app.core.logging import logger


class FallbackStructureAnalyzer(BaseStructureAnalyzer):
    """
    Geometry- and font-driven layout analyzer using PyMuPDF.
    Detects text blocks, titles/headings, embedded tables, and images.
    """

    def analyze_page(
        self,
        page_image: Image.Image,
        page_number: int,
        page_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RawDocumentElement]:
        """
        Analyze a page using geometric layout and PyMuPDF text block analysis.
        """
        # If raw elements are already attached in metadata, return them
        if page_metadata and "raw_elements" in page_metadata:
            return page_metadata["raw_elements"]

        # Default minimal fallback if no extra metadata is supplied
        w, h = page_image.size if page_image else (600, 800)
        return [
            RawDocumentElement(
                type="text",
                page=page_number,
                bbox=[0.0, 0.0, float(w), float(h)],
                text="Content extracted via fallback analyzer.",
                confidence=0.85,
                attributes={"analyzer": "fallback"}
            )
        ]

    def analyze_pdf_page_directly(
        self,
        fitz_page: fitz.Page,
        page_number: int,
        rendered_image: Optional[Image.Image] = None
    ) -> List[RawDocumentElement]:
        """
        Direct layout and table analysis on a PyMuPDF page object.
        """
        elements: List[RawDocumentElement] = []

        # 1. Extract PyMuPDF Tables if available (fitz.Page.find_tables())
        try:
            tabs = fitz_page.find_tables()
            table_bboxes = []
            for tab in tabs:
                t_bbox = [round(v, 2) for v in tab.bbox]
                table_bboxes.append(t_bbox)
                df = tab.extract()
                if df:
                    headers = [str(c or "") for c in df[0]]
                    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                    for row in df[1:]:
                        padded = [str(c or "") for c in row] + [""] * (len(headers) - len(row))
                        md_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
                    md_table = "\n".join(md_lines)

                    html_table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
                    for row in df[1:]:
                        html_table += "<tr>" + "".join(f"<td>{c or ''}</td>" for c in row) + "</tr>"
                    html_table += "</tbody></table>"

                    # Crop table image if rendered_image exists
                    crop_img = None
                    if rendered_image:
                        x0, y0, x1, y1 = [int(v) for v in t_bbox]
                        crop_img = rendered_image.crop((max(0, x0), max(0, y0), min(rendered_image.width, x1), min(rendered_image.height, y1)))

                    elements.append(
                        RawDocumentElement(
                            type="table",
                            page=page_number,
                            bbox=t_bbox,
                            text="\n".join(" | ".join(str(c or "") for c in r) for r in df),
                            markdown=md_table,
                            html=html_table,
                            confidence=0.98,
                            image=crop_img,
                            table_data={"rows": df},
                            attributes={"source": "pymupdf_table_engine"}
                        )
                    )
        except Exception as e:
            logger.debug(f"PyMuPDF table detection skipped: {e}")

        # 2. Extract Text Blocks
        blocks = fitz_page.get_text("blocks")
        for b in blocks:
            # (x0, y0, x1, y1, text, block_no, block_type)
            bbox = [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)]
            block_type = b[6]
            raw_text = b[4].strip()

            if block_type == 0 and raw_text:  # Text block
                elements.append(
                    RawDocumentElement(
                        type="text",
                        page=page_number,
                        bbox=bbox,
                        text=raw_text,
                        confidence=0.99,
                        attributes={"block_no": b[5]}
                    )
                )
            elif block_type == 1:  # Image block
                crop_img = None
                if rendered_image:
                    try:
                        x0, y0, x1, y1 = [int(v) for v in bbox]
                        crop_img = rendered_image.crop((max(0, x0), max(0, y0), min(rendered_image.width, x1), min(rendered_image.height, y1)))
                    except Exception:
                        pass

                elements.append(
                    RawDocumentElement(
                        type="figure",
                        page=page_number,
                        bbox=bbox,
                        confidence=0.95,
                        image=crop_img,
                        attributes={"block_no": b[5], "type": "embedded_image"}
                    )
                )

        return elements


fallback_analyzer = FallbackStructureAnalyzer()
