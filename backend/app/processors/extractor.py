"""
Extraction Layer.
Coordinates document parsing, layout recognition with PP-StructureV3 (or fallback),
persists visual crops (figures, charts, tables) to disk, and normalizes layout objects.
"""

from pathlib import Path
from typing import List, Tuple
from backend.app.processors.base import RawDocumentElement, ParsedPage
from backend.app.processors.pdf_parser import pdf_parser
from backend.app.processors.docx_parser import docx_parser
from backend.app.processors.pp_structure import pp_structure_analyzer
from backend.app.processors.fallback_analyzer import fallback_analyzer
from backend.app.services.storage_service import storage_service
from backend.app.core.config import settings
from backend.app.core.logging import logger
import fitz


class DocumentExtractor:
    """Orchestrates parsing and layout extraction across PDF and DOCX formats."""

    def extract_document(
        self,
        file_path: Path,
        document_id: str
    ) -> Tuple[List[RawDocumentElement], dict]:
        """
        Runs multi-modal layout analysis, saves visual crops, and returns normalized elements.
        """
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self._extract_pdf(file_path, document_id)
        elif extension == ".docx":
            return self._extract_docx(file_path, document_id)
        else:
            raise ValueError(f"Unsupported document extension: {extension}")

    def _extract_pdf(
        self,
        file_path: Path,
        document_id: str
    ) -> Tuple[List[RawDocumentElement], dict]:
        """Process PDF through page rendering and structure analysis."""
        pages, raw_blocks, meta = pdf_parser.parse(file_path)
        all_elements: List[RawDocumentElement] = []

        use_pp = settings.DOC_ANALYZER_ENGINE == "pp_structure" and pp_structure_analyzer.is_available()

        if use_pp:
            logger.info("Extracting PDF layout using PP-StructureV3...")
            for page in pages:
                if page.image:
                    page_elems = pp_structure_analyzer.analyze_page(page.image, page.page_number)
                    all_elements.extend(page_elems)
        else:
            logger.info("Extracting PDF layout using Fallback / PyMuPDF Structure Analyzer...")
            doc = fitz.open(str(file_path))
            for page_idx, page in enumerate(pages):
                fitz_page = doc[page_idx]
                page_elems = fallback_analyzer.analyze_pdf_page_directly(
                    fitz_page=fitz_page,
                    page_number=page.page_number,
                    rendered_image=page.image
                )
                all_elements.extend(page_elems)
            doc.close()

        # Save visual crops (figures, charts, images, tables) to local disk via StorageService
        processed_elements: List[RawDocumentElement] = []
        for idx, elem in enumerate(all_elements):
            element_id = f"elem_{document_id[:8]}_{elem.page}_{idx + 1}"
            elem.attributes["element_id"] = element_id

            if elem.image is not None:
                try:
                    rel_img_path = storage_service.save_image_crop(
                        image=elem.image,
                        document_id=document_id,
                        element_id=element_id,
                        ext="png"
                    )
                    elem.attributes["saved_image_path"] = rel_img_path
                except Exception as e:
                    logger.warning(f"Could not persist crop for element {element_id}: {e}")

            processed_elements.append(elem)

        meta["extracted_elements_count"] = len(processed_elements)
        return processed_elements, meta

    def _extract_docx(
        self,
        file_path: Path,
        document_id: str
    ) -> Tuple[List[RawDocumentElement], dict]:
        """Process DOCX paragraphs, tables, and images."""
        _, raw_elements, meta = docx_parser.parse(file_path)
        processed: List[RawDocumentElement] = []

        for idx, elem in enumerate(raw_elements):
            element_id = f"elem_{document_id[:8]}_{elem.page}_{idx + 1}"
            elem.attributes["element_id"] = element_id

            if elem.image is not None:
                try:
                    rel_path = storage_service.save_image_crop(
                        image=elem.image,
                        document_id=document_id,
                        element_id=element_id,
                        ext="png"
                    )
                    elem.attributes["saved_image_path"] = rel_path
                except Exception as e:
                    logger.warning(f"Could not persist crop for DOCX image {element_id}: {e}")

            processed.append(elem)

        meta["extracted_elements_count"] = len(processed)
        return processed, meta


document_extractor = DocumentExtractor()
