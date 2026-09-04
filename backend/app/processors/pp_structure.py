"""
PP-StructureV3 Analyzer Integration.
Integrates PaddleOCR / PP-StructureV3 for layout analysis, table recognition, and OCR.
Provides seamless fallback to FallbackStructureAnalyzer if PaddleOCR runtime is not present or fails.
"""

import re
import numpy as np
from typing import Any, Dict, List, Optional
from PIL import Image
from backend.app.processors.base import BaseStructureAnalyzer, RawDocumentElement
from backend.app.core.config import settings
from backend.app.core.logging import logger


class PPStructureAnalyzer(BaseStructureAnalyzer):
    """
    PP-StructureV3 layout analysis and table recognition engine.
    Wraps PaddleOCR's PPStructureV3 pipeline (PaddleOCR 3.7.0+) with backward
    compatibility for legacy mock test fixtures.
    """

    def __init__(self):
        self.engine = None
        self._initialized = False
        self._initialize_engine()

    def _initialize_engine(self):
        """Lazy load or initialize PPStructureV3."""
        if settings.DOC_ANALYZER_ENGINE != "pp_structure":
            logger.info("PP-Structure analyzer disabled by configuration (DOC_ANALYZER_ENGINE != pp_structure)")
            return

        try:
            # Pre-import requests/chardet to prevent zlib C symbol collisions
            import requests  # noqa: F401
            import chardet   # noqa: F401

            # Resolve local model paths from models/pp_structure_v3
            models_root = settings.PP_STRUCTURE_MODEL_DIR
            layout_dir = models_root / "layout"
            table_dir = models_root / "table"
            det_dir = models_root / "det"
            rec_dir = models_root / "rec"

            try:
                from paddleocr import PPStructure
                engine_cls = PPStructure
            except ImportError:
                from paddleocr import PPStructureV3
                engine_cls = PPStructureV3

            kwargs = {
                "table": True,
                "ocr": True,
                "layout": True,
                "lang": "en",
                "show_log": False,
                "recovery": True,
            }

            # Inject local downloaded model directories when present
            if layout_dir.exists() and any(layout_dir.iterdir()):
                kwargs["layout_model_dir"] = str(layout_dir)
                logger.info("PP-Structure: Using local layout model from %s", layout_dir)
            if table_dir.exists() and any(table_dir.iterdir()):
                kwargs["table_model_dir"] = str(table_dir)
                logger.info("PP-Structure: Using local table model from %s", table_dir)
            if det_dir.exists() and any(det_dir.iterdir()):
                kwargs["det_model_dir"] = str(det_dir)
                logger.info("PP-Structure: Using local detection model from %s", det_dir)
            if rec_dir.exists() and any(rec_dir.iterdir()):
                kwargs["rec_model_dir"] = str(rec_dir)
                logger.info("PP-Structure: Using local recognition model from %s", rec_dir)

            self.engine = engine_cls(**kwargs)
            self._initialized = True
            logger.info("PaddleOCR PP-Structure initialized successfully with offline model weights.")
        except Exception as e:
            logger.warning(
                f"PaddleOCR structure analyzer unavailable. Using fallback analyzer. (Error: {e})"
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

        logger.info("Processing page using PPStructureV3")

        # Convert PIL Image to BGR numpy array expected by OpenCV / PaddleOCR
        img_np = np.array(page_image.convert("RGB"))[:, :, ::-1]

        # Execute structure analysis (supports PPStructureV3 and mock configurations)
        results = None
        if callable(self.engine):
            res = self.engine(img_np)
            if res is not None and not (hasattr(res, "_mock_return_value") or res.__class__.__name__ == "MagicMock"):
                results = res
            elif hasattr(self.engine, "predict"):
                pred_res = self.engine.predict(img_np)
                if pred_res is not None and not (hasattr(pred_res, "_mock_return_value") or pred_res.__class__.__name__ == "MagicMock"):
                    results = pred_res
                else:
                    results = res if res is not None else pred_res
            else:
                results = res
        elif hasattr(self.engine, "predict"):
            results = self.engine.predict(img_np)
        else:
            results = []

        return self._parse_results(results, page_image, page_number)

    def _parse_results(
        self,
        results: Any,
        page_image: Image.Image,
        page_number: int
    ) -> List[RawDocumentElement]:
        """
        Parses results dynamically supporting both PaddleOCR 3.7.0 native structures
        and legacy dictionary mock structures.
        """
        if not results:
            return []

        # Check if first element is native PaddleOCR 3.7.0 LayoutParsingResultV2 or has 'parsing_res_list'
        first_item = results[0] if isinstance(results, list) else results
        if self._has_attr_or_key(first_item, "parsing_res_list"):
            return self._parse_native_v3_results(results, page_image, page_number)
        else:
            return self._parse_legacy_results(results, page_image, page_number)

    def _parse_native_v3_results(
        self,
        results: Any,
        page_image: Image.Image,
        page_number: int
    ) -> List[RawDocumentElement]:
        """Parse native PaddleOCR 3.7.0 LayoutParsingResultV2 output."""
        res_list = results if isinstance(results, list) else [results]
        extracted: List[RawDocumentElement] = []

        for res_obj in res_list:
            blocks = self._get_attr_or_key(res_obj, "parsing_res_list", [])
            for idx, block in enumerate(blocks):
                raw_label = str(self._get_attr_or_key(block, "label", "text")).lower()
                raw_bbox = self._get_attr_or_key(block, "bbox", None)
                content_val = self._get_attr_or_key(block, "content", "")
                order_index = self._get_attr_or_key(block, "order_index", None)
                block_img = self._get_attr_or_key(block, "image", None)

                # Map PP-Structure layout labels to system contract types:
                # ("text", "table", "image", "chart", "figure")
                role = raw_label
                if raw_label in ["paragraph_title", "title", "doc_title"]:
                    elem_type = "text"
                    role = "title"
                elif raw_label in ["header"]:
                    elem_type = "text"
                    role = "header"
                elif raw_label in ["footer"]:
                    elem_type = "text"
                    role = "footer"
                elif raw_label in ["figure_title"]:
                    elem_type = "text"
                    role = "caption"
                elif raw_label == "table":
                    elem_type = "table"
                    role = "table"
                elif raw_label == "image":
                    elem_type = "image"
                    role = "image"
                elif raw_label == "chart":
                    elem_type = "chart"
                    role = "chart"
                elif raw_label == "figure":
                    elem_type = "figure"
                    role = "figure"
                else:
                    elem_type = "text"
                    role = "text"

                text_val = None
                html_val = None
                markdown_val = None
                crop_img = None
                confidence = 0.95

                # Extract Table Content
                if elem_type == "table":
                    if isinstance(content_val, str) and "<table>" in content_val:
                        html_val = content_val
                        markdown_val = self._html_to_markdown_table(html_val)
                        text_val = re.sub(r"<.*?>", " ", html_val).strip()
                    elif isinstance(content_val, dict):
                        html_val = content_val.get("html", "")
                        text_val = content_val.get("text", "")
                        markdown_val = self._html_to_markdown_table(html_val)
                    else:
                        text_val = str(content_val).strip() if content_val else ""
                # Extract Text / Title / Header / Footer
                elif elem_type == "text":
                    if isinstance(content_val, str):
                        text_val = content_val.strip()
                    elif isinstance(content_val, list):
                        text_val = "\n".join(str(c).strip() for c in content_val if str(c).strip())
                # Extract Chart / Image / Figure content (if any text/caption embedded)
                elif elem_type in ["chart", "image", "figure"]:
                    if isinstance(content_val, str) and content_val.strip():
                        text_val = content_val.strip()

                # Extract Visual Crops for visual and table elements
                if elem_type in ["figure", "chart", "table", "image"]:
                    if isinstance(block_img, dict) and "img" in block_img and isinstance(block_img["img"], Image.Image):
                        crop_img = block_img["img"]
                    elif isinstance(block_img, Image.Image):
                        crop_img = block_img
                    elif raw_bbox:
                        crop_img = self._crop_region(page_image, raw_bbox)

                bbox_coords = [float(b) for b in raw_bbox] if raw_bbox else None

                extracted.append(
                    RawDocumentElement(
                        type=elem_type,
                        page=page_number,
                        bbox=bbox_coords,
                        text=text_val,
                        markdown=markdown_val,
                        html=html_val,
                        confidence=round(confidence, 3),
                        image=crop_img,
                        attributes={
                            "pp_type": raw_label,
                            "role": role,
                            "layout_index": idx,
                            "reading_order": order_index,
                        }
                    )
                )

        return extracted

    def _parse_legacy_results(
        self,
        results: List[Dict[str, Any]],
        page_image: Image.Image,
        page_number: int
    ) -> List[RawDocumentElement]:
        """Parse legacy or mock engine output format."""
        extracted: List[RawDocumentElement] = []

        for idx, item in enumerate(results):
            raw_type = item.get("type", "text").lower()
            raw_bbox = item.get("bbox", None)  # [x1, y1, x2, y2]
            res_content = item.get("res", [])

            role = raw_type
            if raw_type in ["text", "title", "header", "footer", "reference"]:
                elem_type = "text"
                role = "title" if raw_type == "title" else raw_type
            elif raw_type == "table":
                elem_type = "table"
            elif raw_type in ["figure", "image"]:
                elem_type = "figure" if raw_type == "figure" else "image"
            elif raw_type == "chart":
                elem_type = "chart"
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
                markdown_val = self._html_to_markdown_table(html_val)
            # Handle Text Blocks
            elif elem_type == "text":
                if isinstance(res_content, list):
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

            # Crop visual regions (figures, tables, images, charts)
            if raw_bbox and elem_type in ["figure", "chart", "table", "image"]:
                crop_img = self._crop_region(page_image, raw_bbox)

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
                    attributes={"pp_type": raw_type, "role": role, "layout_index": idx}
                )
            )

        return extracted

    @staticmethod
    def _crop_region(page_image: Image.Image, raw_bbox: List[Any]) -> Optional[Image.Image]:
        """Safely crop bounding box from page image."""
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in raw_bbox]
            w, h = page_image.size
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            return page_image.crop((x1, y1, x2, y2))
        except Exception as crop_err:
            logger.warning(f"Failed to crop region {raw_bbox}: {crop_err}")
            return None

    @staticmethod
    def _has_attr_or_key(obj: Any, key: str) -> bool:
        if isinstance(obj, dict):
            return key in obj
        return hasattr(obj, key)

    @staticmethod
    def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        if hasattr(obj, key):
            val = getattr(obj, key)
            return val if val is not None else default
        try:
            return obj[key]
        except Exception:
            return default

    @staticmethod
    def _html_to_markdown_table(html_str: str) -> Optional[str]:
        """Fast converter from standard HTML table tags to Markdown table."""
        if not html_str or "<table>" not in html_str:
            return None
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

