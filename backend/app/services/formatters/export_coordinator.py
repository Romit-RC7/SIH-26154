"""
Export Coordinator Service (Stage 6).
Master coordinator mapping GeneratedArtefact deliverables to their target physical file representations.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.generated_artefact import GeneratedArtefact
from backend.app.schemas.intent import OutputType
from backend.app.services.formatters.docx_formatter import docx_formatter
from backend.app.services.formatters.pptx_formatter import pptx_formatter
from backend.app.services.formatters.pdf_formatter import pdf_formatter
from backend.app.services.formatters.video_package_builder import video_package_builder
from backend.app.services.formatters.infographic_package_builder import infographic_package_builder


class ExportCoordinator:
    """
    Coordinates file generation across docx, pptx, pdf, zip, and json formats.
    """

    EXPORTS_DIR = settings.BASE_DIR / "uploads" / "exports"

    def export_artefact(
        self,
        artefact: GeneratedArtefact,
        target_format: Optional[str] = None
    ) -> Path:
        """
        Exports artefact to physical disk file under uploads/exports/{document_id}/{artefact_id}.{ext}.
        Infers ideal default file format if target_format is not specified.
        """
        fmt = (target_format or "").lower().strip()
        if not fmt:
            fmt = self._default_format_for_type(artefact.output_type)

        export_folder = self.EXPORTS_DIR / str(artefact.document_id)
        export_folder.mkdir(parents=True, exist_ok=True)
        file_path = export_folder / f"{artefact.artefact_id}.{fmt}"

        logger.info("Exporting artefact %s (%s) to format '%s' at %s", artefact.artefact_id, artefact.output_type.value, fmt, file_path)

        if fmt == "docx":
            return docx_formatter.generate_docx(artefact, file_path)

        elif fmt == "pptx":
            return pptx_formatter.generate_pptx(artefact, file_path)

        elif fmt == "pdf":
            return pdf_formatter.generate_pdf(artefact, file_path)

        elif fmt == "zip":
            if artefact.output_type == OutputType.VIDEO_SCRIPT:
                return video_package_builder.generate_video_package(artefact, file_path)
            elif artefact.output_type == OutputType.INFOGRAPHIC_BRIEF:
                return infographic_package_builder.generate_infographic_package(artefact, file_path)
            else:
                # Default ZIP container with JSON + Markdown
                return video_package_builder.generate_video_package(artefact, file_path)

        elif fmt == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(artefact.model_dump(), f, indent=2)
            return file_path

        else:
            # Fallback text file export
            txt_path = export_folder / f"{artefact.artefact_id}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"DELIVERABLE: {artefact.output_type.value.upper()}\n\n")
                f.write(json.dumps(artefact.content, indent=2))
            return txt_path

    def _default_format_for_type(self, output_type: OutputType) -> str:
        """Maps output format type to ideal default binary export format."""
        mapping = {
            OutputType.EXECUTIVE_SUMMARY: "pdf",
            OutputType.PRESENTATION_DECK: "pptx",
            OutputType.VIDEO_SCRIPT: "zip",
            OutputType.INFOGRAPHIC_BRIEF: "zip",
            OutputType.LINKEDIN_POST: "json",
            OutputType.TWITTER_THREAD: "json",
            OutputType.BLOG_POST: "docx",
            OutputType.CUSTOM: "json",
        }
        return mapping.get(output_type, "json")


export_coordinator = ExportCoordinator()
