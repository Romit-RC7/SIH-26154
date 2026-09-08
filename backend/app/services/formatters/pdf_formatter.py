"""
PDF Formatter Service (Stage 6).
Generates publication-ready PDF documents from validated artefacts using ReportLab or PyMuPDF.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from backend.app.core.logging import logger
from backend.app.schemas.generated_artefact import GeneratedArtefact


class PdfFormatter:
    """
    Constructs PDF documents from GeneratedArtefact payload.
    """

    def generate_pdf(self, artefact: GeneratedArtefact, output_path: Path) -> Path:
        """
        Builds a PDF file and saves it to output_path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = artefact.content or {}
        out_type = artefact.output_type.value

        # 1. Try ReportLab if installed
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=letter,
                rightMargin=54,
                leftMargin=54,
                topMargin=54,
                bottomMargin=54
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Heading1'],
                fontSize=22,
                leading=26,
                textColor=colors.HexColor('#14233C'),
                alignment=1, # Center
                spaceAfter=12
            )
            h1_style = ParagraphStyle(
                'DocH1',
                parent=styles['Heading2'],
                fontSize=14,
                leading=18,
                textColor=colors.HexColor('#0066CC'),
                spaceBefore=14,
                spaceAfter=6
            )
            body_style = ParagraphStyle(
                'DocBody',
                parent=styles['Normal'],
                fontSize=10.5,
                leading=15,
                textColor=colors.HexColor('#222222'),
                spaceAfter=8
            )

            story = []

            # Title
            title_text = content.get("title", content.get("headline", content.get("presentation_title", f"{out_type.upper()} DELIVERABLE")))
            story.append(Paragraph(title_text, title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0066CC'), spaceAfter=15))

            # Overview / Subtitle
            hook = content.get("hook", content.get("overview", content.get("tagline", "")))
            if hook:
                story.append(Paragraph(f"<i><b>Overview:</b> {hook}</i>", body_style))
                story.append(Spacer(1, 10))

            # Key Findings / Content Sections
            if "key_findings" in content:
                story.append(Paragraph("Key Findings", h1_style))
                for kf in content["key_findings"]:
                    story.append(Paragraph(f"• {kf}", body_style))

            if "data_highlights" in content:
                story.append(Paragraph("Data Highlights", h1_style))
                for dh in content["data_highlights"]:
                    story.append(Paragraph(f"▪ {dh}", body_style))

            if "recommendations" in content:
                story.append(Paragraph("Strategic Recommendations", h1_style))
                for rec in content["recommendations"]:
                    story.append(Paragraph(f"1. {rec}", body_style))

            if "body" in content:
                story.append(Paragraph("Main Content", h1_style))
                story.append(Paragraph(content["body"], body_style))

            if "conclusion" in content:
                story.append(Paragraph("Conclusion", h1_style))
                story.append(Paragraph(content["conclusion"], body_style))

            doc.build(story)
            logger.info("Generated PDF deliverable via ReportLab at %s", output_path)
            return output_path

        except Exception as exc:
            logger.info("ReportLab unavailable or failed (%s); using PyMuPDF canvas renderer", exc)

        # 2. PyMuPDF (fitz) fallback canvas renderer
        try:
            import pymupdf as fitz

            pdf_doc = fitz.open()
            page = pdf_doc.new_page(width=612, height=792)  # Letter size
            rect = fitz.Rect(54, 54, 558, 738)
            
            text_flow = f"{artefact.output_type.value.upper()} DELIVERABLE\n\n"
            for k, v in content.items():
                text_flow += f"=== {k.upper()} ===\n{v}\n\n"

            page.insert_textbox(rect, text_flow, fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
            pdf_doc.save(str(output_path))
            pdf_doc.close()
            logger.info("Generated PDF deliverable via PyMuPDF at %s", output_path)
            return output_path
        except Exception as exc2:
            logger.error("Failed to generate PDF (%s); writing txt fallback", exc2)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"PDF Deliverable: {out_type}\n\n{content}")
            return output_path


pdf_formatter = PdfFormatter()
