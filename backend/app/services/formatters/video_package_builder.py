"""
Video Package Builder Service (Stage 6).
Packages video scripts, scene storyboards, SRT subtitles, and visual cues into a downloadable .zip package.
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from backend.app.core.logging import logger
from backend.app.schemas.generated_artefact import GeneratedArtefact


class VideoPackageBuilder:
    """
    Assembles Video Script deliverables into a complete production ZIP package.
    """

    def generate_video_package(self, artefact: GeneratedArtefact, output_path: Path) -> Path:
        """
        Builds a .zip package containing script JSON, .srt subtitles, clean narration, and visual cues MD.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = artefact.content or {}
        scenes = content.get("scenes", [])

        # 1. Build .srt Subtitle File
        srt_content = self._build_srt_subtitles(scenes)

        # 2. Build Clean Narration Text
        narration_text = "\n\n".join(
            f"Scene {s.get('scene_number', idx)}:\n{s.get('narration', '')}"
            for idx, s in enumerate(scenes, start=1)
        )

        # 3. Build Visual Recommendations Markdown
        visual_md = f"# Visual B-Roll & Storyboard Recommendations\n\nTitle: {content.get('video_title', 'Video Brief')}\n\n"
        for idx, s in enumerate(scenes, start=1):
            visual_md += f"## Scene {s.get('scene_number', idx)} ({s.get('duration_sec', 15)}s)\n"
            visual_md += f"- **Visual Cue**: {s.get('visual_cue', 'B-Roll footage')}\n"
            visual_md += f"- **Narration**: \"{s.get('narration', '')}\"\n\n"

        # 4. Pack into ZIP archive
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add script_storyboard.json
            zipf.writestr("script_storyboard.json", json.dumps(content, indent=2))
            # Add subtitles.srt
            zipf.writestr("subtitles.srt", srt_content)
            # Add narration_teleprompter.txt
            zipf.writestr("narration_teleprompter.txt", narration_text)
            # Add visual_recommendations.md
            zipf.writestr("visual_recommendations.md", visual_md)

        logger.info("Generated Video Package ZIP at %s", output_path)
        return output_path

    def _build_srt_subtitles(self, scenes: List[Dict[str, Any]]) -> str:
        """Generates formatted SRT subtitles with sequential timestamps."""
        srt_entries = []
        current_time_sec = 0.0

        for idx, scene in enumerate(scenes, start=1):
            narration = scene.get("narration", "").strip()
            duration = float(scene.get("duration_sec", 15))
            if not narration:
                continue

            start_str = self._format_srt_timestamp(current_time_sec)
            end_time = current_time_sec + duration
            end_str = self._format_srt_timestamp(end_time)

            srt_entries.append(f"{idx}\n{start_str} --> {end_str}\n{narration}\n")
            current_time_sec = end_time

        return "\n".join(srt_entries)

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Formats seconds float into SRT timestamp string: HH:MM:SS,mmm"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


video_package_builder = VideoPackageBuilder()
