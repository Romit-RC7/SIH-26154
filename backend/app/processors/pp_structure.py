"""
PP-StructureV3 Analyzer Integration.
Integrates PaddleOCR / PP-Structure for layout analysis, table recognition, and OCR.
Provides seamless fallback to RuleBasedStructureAnalyzer if PaddleOCR runtime is not present.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from PIL import Image
from backend.app.processors.base import BaseStructureAnalyzer, RawDocumentElement
from backend.app.core.config import settings
from backend.app.core.logging import logger


class PPStructureAnalyzer(BaseStructureAnalyzer):
    """
    PP-StructureV3 layout analysis and table recognition engine.
    Wraps PaddleOCR's PPStructure pipeline.
    """

    def __init__(self):
        self.engine = None
        self._initialized = False
        self._initialize_engine()

    def _initialize_engine(self):
        """Lazy load or initialize PPStructure."""
        if settings.DOC_ANALYZER_ENGINE != "pp_structure":
            logger.info("PP-Structure analyzer disabled by configuration (DOC_ANALYZER_ENGINE != pp_structure)")
            return

        try:
            from paddleocr import PPStructure
            # Initialize PPStructure with layout analysis, table structure, and OCR enabled
            self.engine = PPStructure(
                table=True,
                ocr=True,
                show_log=False,
                layout=True,
                recovery=True,
            )
            self._initialized = True
            logger.info("PP-StructureV3 engine initialized successfully.")
        except Exception as e:
            logger.warning(
                f"PaddleOCR PP-Structure could not be initialized ({e}). "
                "Will automatically route layout analysis to FallbackStructureAnalyzer."
            )
            self.engine = None
            self._initialized = False

    def is_available(self) -> bool:
        return self._initialized and self.engine is not None

    def analyze_page(
        self,
        page_image: Image.Image,
        page_number: int,
        page_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RawDocumentElement]:
        """
        Analyze a rasterized page image using PP-StructureV3.
        """
        if not self.is_available():
            raise RuntimeError("PP-Structure engine is not available.")

        # Convert PIL Image to BGR numpy array expected by OpenCV / PaddleOCR
        img_np = np.array(page_image.convert("RGB"))[:, :, ::-1]

        results = self.engine(img_np)
        extracted: List[RawDocumentElement] = []

        for idx, item in enumerate(results):
            raw_type = item.get("type", "text").lower()
            raw_bbox = item.get("bbox", None)  # [x1, y1, x2, y2]
            res_content = item.get("res", [])

            # Map PP-Structure layout labels to system contract types:
            # "text", "table", "image", "chart", "figure"
            if raw_type in ["text", "title", "header", "footer", "reference"]:
                elem_type = "text"
            elif raw_type == "table":
                elem_type = "table"
            elif raw_type in ["figure", "image"]:
                # If caption or keywords indicate a chart, label as chart, else figure
                elem_type = "figure"
            else:
                elem_type = "text"

            text_val = None
            html_val = None
            markdown_val = None
            crop_img = None
            confidence = 0.95

            # Handle Table Structure
            if elem_type == "table" and isinstance(res_content, dict):
                html_val = res_content.get("html", "")
                text_val = res_content.get("text", "")
                # Clean up or convert HTML to Markdown representation
                markdown_val = self._html_to_markdown_table(html_val)
            # Handle Text Blocks
            elif elem_type == "text":
                if isinstance(res_content, list):
                    # res_content is list of ((text, conf), ...) or OCR lines
                    line_texts = []
                    confs = []
                    for line in res_content:
                        if isinstance(line, dict):
                            line_texts.append(line.get("text", ""))
                            confs.append(line.get("confidence", 0.9))
                        elif isinstance(line, (list, tuple)) and len(line) >= 2:
                            ocr_data = line[1]
                            if isinstance(ocr_data, (list, tuple)):
                                line_texts.append(str(ocr_data[0]))
                                confs.append(float(ocr_data[1]))
                            else:
                                line_texts.append(str(ocr_data))
                    text_val = "\n".join(t for t in line_texts if t.strip())
                    if confs:
                        confidence = sum(confs) / len(confs)
                elif isinstance(res_content, str):
                    text_val = res_content

            # Crop visual regions (figures, tables, images)
            if raw_bbox and elem_type in ["figure", "chart", "table", "image"]:
                try:
                    x1, y1, x2, y2 = [int(v) for v in raw_bbox]
                    w, h = page_image.size
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    crop_img = page_image.crop((x1, y1, x2, y2))
                except Exception as crop_err:
                    logger.warning(f"Failed to crop region {raw_bbox}: {crop_err}")

            extracted.append(
                RawDocumentElement(
                    type=elem_type,
                    page=page_number,
                    bbox=[float(b) for b in raw_bbox] if raw_bbox else None,
                    text=text_val,
                    markdown=markdown_val,
                    html=html_val,
                    confidence=round(confidence, 3),
                    image=crop_img,
                    attributes={"pp_type": raw_type, "layout_index": idx}
                )
            )

        return extracted

    @staticmethod
    def _html_to_markdown_table(html_str: str) -> Optional[str]:
        """Simple, fast converter from standard HTML table tags to Markdown table."""
        if not html_str or "<table>" not in html_str:
            return None
        import re
        rows = re.findall(r"<tr>(.*?)</tr>", html_str, flags=re.DOTALL | re.IGNORECASE)
        if not rows:
            return None

        table_matrix = []
        for row in rows:
            cells = re.findall(r"<t[dh]>(.*?)</t[dh]>", row, flags=re.DOTALL | re.IGNORECASE)
            cleaned_cells = [re.sub(r"<.*?>", "", c).strip().replace("\n", " ") for c in cells]
            if cleaned_cells:
                table_matrix.append(cleaned_cells)

        if not table_matrix:
            return None

        headers = table_matrix[0]
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for r in table_matrix[1:]:
            padded = r + [""] * (len(headers) - len(r))
            md += "| " + " | ".join(padded[:len(headers)]) + " |\n"
        return md


pp_structure_analyzer = PPStructureAnalyzer()
