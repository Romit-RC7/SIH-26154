"""
Infographic Package Builder Service (Stage 6).
Packages infographic layouts, key metrics, and HTML/SVG visual blueprints into a downloadable .zip package.
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from backend.app.core.logging import logger
from backend.app.schemas.generated_artefact import GeneratedArtefact


class InfographicPackageBuilder:
    """
    Assembles Infographic Brief deliverables into a complete visual blueprint ZIP package.
    """

    def generate_infographic_package(self, artefact: GeneratedArtefact, output_path: Path) -> Path:
        """
        Builds a .zip package containing JSON blueprint, interactive HTML preview, and key metrics summary.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = artefact.content or {}

        # 1. Interactive HTML Blueprint Preview Template
        html_content = self._build_html_template(content)

        # 2. Key Metrics Summary Markdown
        metrics_md = f"# Infographic Key Metrics & Callouts\n\n"
        metrics_md += f"## Headline: {content.get('headline', 'Infographic Brief')}\n"
        if content.get("subheadline"):
            metrics_md += f"*{content['subheadline']}*\n\n"

        metrics_md += "### Core Quantitative Callouts\n"
        for stat in content.get("key_stats", []):
            if isinstance(stat, dict):
                metrics_md += f"- **{stat.get('label', 'Metric')}**: `{stat.get('value', '')}` — {stat.get('context', '')}\n"

        metrics_md += "\n### Visual Sections\n"
        for sec in content.get("visual_sections", []):
            if isinstance(sec, dict):
                metrics_md += f"#### {sec.get('title')}\n"
                metrics_md += f"- Type: `{sec.get('visual_type', 'barchart')}`\n"
                metrics_md += f"- Description: {sec.get('description')}\n\n"

        # 3. Pack into ZIP archive
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("infographic_blueprint.json", json.dumps(content, indent=2))
            zipf.writestr("render_template.html", html_content)
            zipf.writestr("key_metrics_summary.md", metrics_md)

        logger.info("Generated Infographic Package ZIP at %s", output_path)
        return output_path

    def _build_html_template(self, content: Dict[str, Any]) -> str:
        headline = content.get("headline", "Infographic Dashboard")
        subheadline = content.get("subheadline", "Visual Analytics & Key Callouts")
        stats = content.get("key_stats", [])
        sections = content.get("visual_sections", [])

        stats_html = ""
        for s in stats:
            if isinstance(s, dict):
                stats_html += f"""
                <div class="stat-card">
                    <div class="stat-value">{s.get('value', '0%')}</div>
                    <div class="stat-label">{s.get('label', 'Metric')}</div>
                    <div class="stat-context">{s.get('context', '')}</div>
                </div>
                """

        sections_html = ""
        for sec in sections:
            if isinstance(sec, dict):
                sections_html += f"""
                <div class="section-card">
                    <h3>{sec.get('title', 'Visual Section')}</h3>
                    <p>{sec.get('description', '')}</p>
                    <div class="badge">Visual Layout: {sec.get('visual_type', 'barchart')}</div>
                </div>
                """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{headline}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; color: #333; margin: 0; padding: 40px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
        h1 {{ color: #14233C; font-size: 28px; margin-bottom: 8px; text-align: center; }}
        .subtitle {{ text-align: center; color: #666; font-size: 16px; margin-bottom: 36px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background: #f8fafc; border: 2px solid #e2e8f0; border-radius: 10px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #0066CC; margin-bottom: 4px; }}
        .stat-label {{ font-size: 14px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-context {{ font-size: 12px; color: #94a3b8; margin-top: 6px; }}
        .sections-grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
        .section-card {{ background: #ffffff; border-left: 4px solid #0066CC; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); border-radius: 0 8px 8px 0; }}
        .section-card h3 {{ margin: 0 0 8px 0; color: #1e293b; }}
        .badge {{ display: inline-block; background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{headline}</h1>
        <div class="subtitle">{subheadline}</div>
        <div class="stats-grid">
            {stats_html}
        </div>
        <div class="sections-grid">
            {sections_html}
        </div>
    </div>
</body>
</html>
"""


infographic_package_builder = InfographicPackageBuilder()
