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
        2. Links captions to figures, charts, and tables.
        3. Normalizes content payloads into SemanticElement models.
        4. Initializes SourceReferences.
        """
        # Group and sort by page and coordinates
        sorted_raw = self._sort_reading_order(raw_elements)

        # Associate captions with adjacent visual/table elements
        linked_raw = self._link_captions(sorted_raw)

        # Build SemanticElements
        semantic_elements: List[SemanticElement] = []
        for idx, item in enumerate(linked_raw):
            elem_id = item.attributes.get("element_id") or f"elem_{document_id[:8]}_{item.page}_{idx + 1}"
            saved_image = item.attributes.get("saved_image_path")

            # Validate type to contract
            elem_type = item.type
            if elem_type not in ["text", "table", "image", "chart", "figure"]:
                elem_type = "text"

            content = ElementContent(
                text=item.text,
                markdown=item.markdown,
                html=item.html,
                image_path=saved_image,
                confidence=item.confidence,
                reading_order=idx + 1,
                caption=item.caption,
                table_structure=item.table_data,
                raw_attributes={k: v for k, v in item.attributes.items() if k not in ["element_id", "saved_image_path"]}
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

    def _sort_reading_order(self, elements: List[RawDocumentElement]) -> List[RawDocumentElement]:
        """
        Sorts elements by:
        1. Page number
        2. Top-down position (y0), with a line-height threshold to group horizontal items
        3. Left-to-right position (x0)
        """
        def sort_key(elem: RawDocumentElement):
            page = elem.page
            if elem.bbox and len(elem.bbox) >= 4:
                # Quantize y0 to 10-point bands to handle slight alignment differences in 2-column or tabular flows
                y0_band = int(elem.bbox[1] / 10) * 10
                x0 = elem.bbox[0]
                return (page, y0_band, x0)
            # If no bbox, use reading_order attribute if present, else 0
            reading_order = elem.attributes.get("reading_order", 0)
            return (page, reading_order, 0)

        return sorted(elements, key=sort_key)

    def _link_captions(self, elements: List[RawDocumentElement]) -> List[RawDocumentElement]:
        """
        Detects caption paragraphs (e.g. starting with 'Figure 1', 'Table 2', 'Chart A')
        and attaches them to adjacent table or figure elements.
        """
        caption_prefixes = ("fig", "figure", "table", "chart", "diagram", "exhibit")
        i = 0
        n = len(elements)
        while i < n:
            elem = elements[i]
            if elem.type == "text" and elem.text:
                lower_text = elem.text.strip().lower()
                if lower_text.startswith(caption_prefixes):
                    # Check if previous element is a figure or table on the same page
                    if i > 0 and elements[i - 1].page == elem.page and elements[i - 1].type in ["figure", "chart", "table", "image"]:
                        elements[i - 1].caption = elem.text.strip()
                    # Or check next element
                    elif i + 1 < n and elements[i + 1].page == elem.page and elements[i + 1].type in ["figure", "chart", "table", "image"]:
                        elements[i + 1].caption = elem.text.strip()
            i += 1
        return elements


semantic_fusion_engine = SemanticFusionEngine()
