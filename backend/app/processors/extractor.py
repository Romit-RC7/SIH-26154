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
from backend.app.services.recognition import recognition_coordinator
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.processors.ppt_parser import ppt_parser
from backend.app.processors.image_parser import image_parser
from backend.app.processors.video_parser import video_parser
import fitz


class DocumentExtractor:
    """Orchestrates parsing and layout extraction across PDF and DOCX formats."""

    def extract_document(
        self,
        file_path: Path,
        document_id: str,
        run_specialist_recognition: bool = True,
        unload_structure: bool = True,
    ) -> Tuple[List[RawDocumentElement], dict]:
        """
        Runs multi-modal layout analysis, saves visual crops, and returns normalized elements.
        """
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self._extract_pdf(file_path, document_id, run_specialist_recognition, unload_structure)
        elif extension == ".docx":
            return self._extract_docx(file_path, document_id, run_specialist_recognition, unload_structure)
            return self._extract_docx(file_path, document_id)

        elif extension == ".pptx":
            return self._extract_pptx(file_path, document_id, run_specialist_recognition, unload_structure)

        elif extension in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tiff"
        }:
            return self._extract_image(file_path, document_id, run_specialist_recognition, unload_structure)

        elif extension in {".mp4", ".mov", ".webm"}:
            return self._extract_video(file_path, document_id, run_specialist_recognition, unload_structure)

        else:
            raise ValueError(
                f"Unsupported document extension: {extension}"
            )

    def _extract_pdf(
        self,
        file_path: Path,
        document_id: str,
        run_specialist_recognition: bool = True,
        unload_structure: bool = True,
    ) -> Tuple[List[RawDocumentElement], dict]:
        """Process PDF through page rendering and structure analysis."""
        pages, raw_blocks, meta = pdf_parser.parse(file_path)
        all_elements: List[RawDocumentElement] = []

        use_pp = settings.DOC_ANALYZER_ENGINE == "pp_structure" and pp_structure_analyzer.is_available()
        extracted_with_pp = False

        if use_pp:
            try:
                logger.info("Extracting PDF layout using PP-StructureV3...")
                for page in pages:
                    if page.image:
                        page_elems = pp_structure_analyzer.analyze_page(page.image, page.page_number)
                        all_elements.extend(page_elems)
                extracted_with_pp = True
            except Exception as e:
                logger.error(
                    f"PaddleOCR structure analysis failed. Falling back to PyMuPDF analyzer: {e}"
                )
                all_elements = []
                extracted_with_pp = False

        if not extracted_with_pp:
            if not use_pp:
                logger.warning("PaddleOCR structure analyzer unavailable. Using fallback analyzer.")
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

        if unload_structure:
            pp_structure_analyzer.unload()
        if run_specialist_recognition:
            recognition_coordinator.recognize(processed_elements)
        meta["extracted_elements_count"] = len(processed_elements)
        return processed_elements, meta

    def _extract_docx(
        self,
        file_path: Path,
        document_id: str,
        run_specialist_recognition: bool = True,
        unload_structure: bool = True,
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

        if unload_structure:
            pp_structure_analyzer.unload()
        if run_specialist_recognition:
            recognition_coordinator.recognize(processed)

        meta["extracted_elements_count"] = len(processed)
        return processed, meta


    def _extract_pptx(
        self,
        file_path: Path,
        document_id: str,
        run_specialist_recognition: bool = True,
        unload_structure: bool = True,
    ):
        _, raw_elements, meta = ppt_parser.parse(file_path)

        processed = []

        for idx, elem in enumerate(raw_elements):

            element_id = (
                f"elem_{document_id[:8]}_{elem.page}_{idx + 1}"
            )

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
                    logger.warning(
                        f"Could not persist PPTX image {element_id}: {e}"
                    )

            processed.append(elem)
        if unload_structure:
            pp_structure_analyzer.unload()
        if run_specialist_recognition:
            recognition_coordinator.recognize(processed)
        meta["extracted_elements_count"] = len(processed)

        return processed, meta

    def _extract_image(
        self,
        file_path: Path,
        document_id: str,
        run_specialist_recognition: bool = True,
        unload_structure: bool = True,
    ):
        _, raw_elements, meta = image_parser.parse(file_path)

        processed = []

        for idx, elem in enumerate(raw_elements):

            element_id = (
                f"elem_{document_id[:8]}_{elem.page}_{idx + 1}"
            )

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
                    logger.warning(
                        f"Could not persist PPTX image {element_id}: {e}"
                    )

            processed.append(elem)
        if unload_structure:
            pp_structure_analyzer.unload()
        if run_specialist_recognition:
            recognition_coordinator.recognize(processed)
        meta["extracted_elements_count"] = len(processed)

        return processed, meta

    def _extract_video(
        self,
        file_path: Path,
        document_id: str,
        run_specialist_recognition: bool = True,
        unload_structure: bool = True,
    ):
        """Extract a mono WAV track and visual frame samples from an uploaded video."""
        audio_output_path = storage_service.extracted_dir / document_id / "audio.wav"
        _, raw_elements, meta = video_parser.parse(file_path, audio_output_path)
        processed: List[RawDocumentElement] = []
        for idx, elem in enumerate(raw_elements):
            element_id = f"elem_{document_id[:8]}_{elem.page}_{idx + 1}"
            elem.attributes["element_id"] = element_id
            if elem.image is not None:
                self._attach_video_frame_ocr(elem)
                try:
                    elem.attributes["saved_image_path"] = storage_service.save_image_crop(elem.image, document_id, element_id, "png")
                except Exception as exc:
                    logger.warning("Could not persist video frame %s: %s", element_id, exc)
            processed.append(elem)
        if unload_structure:
            pp_structure_analyzer.unload()
        if run_specialist_recognition:
            recognition_coordinator.recognize(processed)
        meta["extracted_elements_count"] = len(processed)
        return processed, meta

    @staticmethod
    def _attach_video_frame_ocr(element: RawDocumentElement) -> None:
        """Attach PP-Structure OCR evidence to one sampled video frame."""
        if element.image is None or not pp_structure_analyzer.is_available():
            return
        try:
            ocr_elements = pp_structure_analyzer.analyze_page(element.image, element.page)
            text = "\n".join(
                item.text.strip()
                for item in ocr_elements
                if item.type == "text" and item.text and item.text.strip()
            )
            if text:
                element.attributes["ocr_text"] = text
                element.attributes["ocr_model"] = "PP-StructureV3"
                element.attributes["ocr_status"] = "completed"
        except Exception as exc:
            logger.warning("Video-frame OCR failed on page %s: %s", element.page, exc)
            element.attributes["ocr_status"] = "failed"
            element.attributes["ocr_error"] = str(exc)


document_extractor = DocumentExtractor()
