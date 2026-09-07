"""
DOCX Formatter Service (Stage 6).
Renders validated structured content into professionally formatted Microsoft Word (.docx) documents.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from backend.app.core.logging import logger
from backend.app.schemas.generated_artefact import GeneratedArtefact


class DocxFormatter:
    """
    Constructs styled .docx files from GeneratedArtefact payload using python-docx.
    """

    def generate_docx(self, artefact: GeneratedArtefact, output_path: Path) -> Path:
        """
        Builds a .docx document and saves it to output_path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = artefact.content or {}
        out_type = artefact.output_type.value

        try:
            import docx
            from docx.shared import Inches, Pt, RGBColor
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.enum.table import WD_TABLE_ALIGNMENT

            doc = docx.Document()

            # Set standard margins (1 inch)
            sections = doc.sections
            for section in sections:
                section.top_margin = Inches(1.0)
                section.bottom_margin = Inches(1.0)
                section.left_margin = Inches(1.0)
                section.right_margin = Inches(1.0)

            # Title
            title_text = content.get("title", content.get("headline", content.get("presentation_title", f"Deliverable: {out_type.upper()}")))
            title = doc.add_heading(title_text, level=0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # Subtitle / Hook / Tagline
            hook = content.get("hook", content.get("overview", content.get("tagline", "")))
            if hook:
                p_hook = doc.add_paragraph()
                p_hook.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p_hook.add_run(hook)
                run.italic = True
                run.font.size = Pt(11)
                run.font.color.rgb = RGBColor(80, 80, 80)

            doc.add_paragraph()  # spacing

            # Executive Summary / Report Layout
            if "overview" in content or "key_findings" in content:
                if content.get("overview"):
                    h = doc.add_heading("Executive Overview", level=1)
                    doc.add_paragraph(content["overview"])

                if content.get("key_findings"):
                    doc.add_heading("Key Findings", level=1)
                    for kf in content["key_findings"]:
                        doc.add_paragraph(f"• {kf}")

                if content.get("data_highlights"):
                    doc.add_heading("Quantitative Data Highlights", level=1)
                    for dh in content["data_highlights"]:
                        doc.add_paragraph(f"▪ {dh}")

                if content.get("recommendations"):
                    doc.add_heading("Strategic Recommendations", level=1)
                    for rec in content["recommendations"]:
                        doc.add_paragraph(f"1. {rec}", style='List Number' if 'List Number' in doc.styles else None)

                if content.get("conclusion"):
                    doc.add_heading("Conclusion", level=1)
                    doc.add_paragraph(content["conclusion"])

            # Generic / Narrative Layout
            elif "body" in content or "sections" in content:
                if content.get("body"):
                    doc.add_heading("Main Narrative", level=1)
                    doc.add_paragraph(content["body"])

                if content.get("sections"):
                    for sec in content["sections"]:
                        if isinstance(sec, dict):
                            doc.add_heading(sec.get("heading", "Section"), level=1)
                            doc.add_paragraph(sec.get("content", ""))

                if content.get("cta"):
                    doc.add_heading("Actionable Call to Action", level=1)
                    doc.add_paragraph(content["cta"])

            else:
                # Direct JSON dump fallback
                doc.add_heading("Generated Content", level=1)
                for k, v in content.items():
                    doc.add_heading(str(k).replace("_", " ").title(), level=2)
                    doc.add_paragraph(str(v))

            doc.save(str(output_path))
            logger.info("Generated DOCX deliverable at %s", output_path)
            return output_path

        except Exception as exc:
            logger.warning("python-docx rendering failed (%s); writing raw text file fallback", exc)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# {artefact.output_type.value.upper()} DELIVERABLE\n\n")
                for k, v in content.items():
                    f.write(f"## {k}\n{v}\n\n")
            return output_path


docx_formatter = DocxFormatter()
