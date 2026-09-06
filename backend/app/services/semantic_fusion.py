"""
Semantic Fusion Engine.
Fuses multi-modal extracted elements, sorts reading order, links captions with tables/figures,
normalizes content representations, and prepares unified semantic elements for the document contract.
"""

from typing import List, Tuple
from backend.app.processors.base import RawDocumentElement
from backend.app.schemas.semantic_document import (
    SemanticElement,
    ElementContent,
    SourceReference,
)
from backend.app.core.logging import logger


class SemanticFusionEngine:
    """Fuses multi-modal document elements into a coherent, structured sequence."""

    def fuse_elements(
        self,
        raw_elements: List[RawDocumentElement],
        document_id: str,
        file_name: str
    ) -> Tuple[List[SemanticElement], List[SourceReference]]:
        """
        Processes raw elements:
        1. Sorts into natural reading order per page.
        2. Deduplicates text elements overlapping with tables.
        3. Links captions to figures, charts, and tables.
        4. Normalizes content payloads into SemanticElement models.
        5. Initializes SourceReferences.
        """
        # Sort reading order
        sorted_raw = self._sort_reading_order(raw_elements)

        # Deduplicate text elements that overlap with table bounding boxes
        deduped_raw = self._deduplicate_table_text(sorted_raw)

        # Associate captions with adjacent visual/table elements with distance checking
        linked_raw = self._link_captions(deduped_raw)

        # Build SemanticElements
        semantic_elements: List[SemanticElement] = []
        for idx, item in enumerate(linked_raw):
            saved_image = item.attributes.get("saved_image_path")
            has_text = bool(item.text and item.text.strip())
            has_formatted = bool(item.markdown or item.html)
            has_image = bool(saved_image)
            has_caption = bool(item.caption and item.caption.strip())

            # Skip empty background layout artifacts that contain zero content
            if not (has_text or has_formatted or has_image or has_caption):
                continue

            elem_id = item.attributes.get("element_id") or f"elem_{document_id[:8]}_{item.page}_{idx + 1}"

            # Validate type to contract
            elem_type = item.type
            if elem_type not in ["text", "table", "image", "chart", "figure"]:
                elem_type = "text"

            raw_attributes = {
                key: value
                for key, value in item.attributes.items()
                if key not in ["element_id", "saved_image_path"]
            }
            recognition_status = self._recognition_status(item.attributes)
            if recognition_status:
                raw_attributes["recognition"] = recognition_status

            content = ElementContent(
                text=item.text,
                markdown=item.markdown,
                html=item.html,
                image_path=saved_image,
                confidence=item.confidence,
                reading_order=idx + 1,
                caption=item.caption,
                table_structure=item.table_data,
                raw_attributes=raw_attributes
            )

            semantic_elements.append(
                SemanticElement(
                    id=elem_id,
                    type=elem_type,
                    page=item.page,
                    bbox=item.bbox,
                    content=content
                )
            )

        # Generate base source references
        sources = [
            SourceReference(
                id=f"src_{document_id[:8]}_1",
                title=file_name,
                citation=f"Original Document: {file_name}"
            )
        ]

        logger.info(f"Fused {len(semantic_elements)} semantic elements for document {document_id}")
        return semantic_elements, sources

    @staticmethod
    def _recognition_status(attributes: dict) -> dict:
        """Group specialist status/model/error fields without losing raw output."""
        stages = {}
        for key, value in attributes.items():
            if key.endswith("_recognition_status"):
                stage = key.removesuffix("_recognition_status")
                stages.setdefault(stage, {})["status"] = value
            elif key.endswith("_recognition_model"):
                stage = key.removesuffix("_recognition_model")
                stages.setdefault(stage, {})["model"] = value
            elif key.endswith("_recognition_error"):
                stage = key.removesuffix("_recognition_error")
                stages.setdefault(stage, {})["error"] = value
            elif key.endswith("_analysis_status"):
                stage = key.removesuffix("_analysis_status")
                stages.setdefault(stage, {})["status"] = value
        return stages

    def _sort_reading_order(self, elements: List[RawDocumentElement]) -> List[RawDocumentElement]:
        """
        Sorts elements by page, top-down position (y0), and left-to-right (x0).
        """
        def sort_key(elem: RawDocumentElement):
            page = elem.page
            if elem.bbox and len(elem.bbox) >= 4:
                y0_band = int(elem.bbox[1] / 10) * 10
                x0 = elem.bbox[0]
                return (page, y0_band, x0)
            reading_order = elem.attributes.get("reading_order", 0)
            return (page, reading_order, 0)

        return sorted(elements, key=sort_key)

    def _deduplicate_table_text(self, elements: List[RawDocumentElement]) -> List[RawDocumentElement]:
        """
        Removes loose text elements that fall completely within a structured table bounding box.
        """
        tables = [e for e in elements if e.type == "table" and e.bbox and len(e.bbox) >= 4]
        if not tables:
            return elements

        filtered: List[RawDocumentElement] = []
        for elem in elements:
            if elem.type == "text" and elem.bbox and len(elem.bbox) >= 4:
                ex1, ey1, ex2, ey2 = elem.bbox
                cx, cy = (ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0
                is_inside_table = False
                for tbl in tables:
                    if tbl.page == elem.page:
                        tx1, ty1, tx2, ty2 = tbl.bbox
                        # Allow 5-point margin for alignment
                        if (tx1 - 5) <= cx <= (tx2 + 5) and (ty1 - 5) <= cy <= (ty2 + 5):
                            is_inside_table = True
                            break
                if is_inside_table:
                    continue  # Skip redundant cell text string
            filtered.append(elem)
        return filtered

    def _link_captions(self, elements: List[RawDocumentElement]) -> List[RawDocumentElement]:
        """
        Detects caption paragraphs and matches them strictly by element type and vertical proximity (< 60pt).
        """
        i = 0
        n = len(elements)
        while i < n:
            elem = elements[i]
            if elem.type == "text" and elem.text:
                lower_text = elem.text.strip().lower()
                target_types = []
                if lower_text.startswith(("table", "tab.")):
                    target_types = ["table"]
                elif lower_text.startswith(("chart", "graph")):
                    target_types = ["chart", "figure"]
                elif lower_text.startswith(("fig", "figure", "image", "illustration")):
                    target_types = ["figure", "image"]

                if target_types:
                    # Check adjacent candidate elements on the same page
                    candidates = []
                    if i > 0 and elements[i - 1].page == elem.page and elements[i - 1].type in target_types:
                        candidates.append(elements[i - 1])
                    if i + 1 < n and elements[i + 1].page == elem.page and elements[i + 1].type in target_types:
                        candidates.append(elements[i + 1])

                    # Attach to closest candidate within 60 points vertical proximity
                    best_cand = None
                    min_dist = 60.0
                    for cand in candidates:
                        if elem.bbox and cand.bbox and len(elem.bbox) >= 4 and len(cand.bbox) >= 4:
                            # Vertical distance between bounding boxes
                            dist = min(
                                abs(elem.bbox[1] - cand.bbox[3]),  # caption below candidate
                                abs(cand.bbox[1] - elem.bbox[3]),  # caption above candidate
                            )
                            if dist < min_dist:
                                min_dist = dist
                                best_cand = cand

                    if best_cand:
                        best_cand.caption = elem.text.strip()
            i += 1
        return elements


semantic_fusion_engine = SemanticFusionEngine()

