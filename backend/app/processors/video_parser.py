"""Video parser that extracts a WAV track and samples visual frames over time."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.processors.base import ParsedPage, RawDocumentElement


class VideoParser:
    """Uses FFmpeg/FFprobe already present in the Docker runtime."""

    def parse(self, file_path: Path, audio_output_path: Path) -> Tuple[List[ParsedPage], List[RawDocumentElement], dict]:
        if not file_path.is_file():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        self._require_ffmpeg()
        probe = self._probe(file_path)
        duration = float(probe.get("format", {}).get("duration") or 0.0)
        video_stream = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"), {})
        has_audio = any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))
        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)

        relative_audio_path = None
        if has_audio:
            audio_output_path.parent.mkdir(parents=True, exist_ok=True)
            self._extract_audio(file_path, audio_output_path)
            relative_audio_path = str(audio_output_path.relative_to(settings.BASE_DIR)).replace("\\", "/")
        timestamps = self._sample_timestamps(duration)
        pages: List[ParsedPage] = []
        elements: List[RawDocumentElement] = []
        if relative_audio_path:
            elements.append(RawDocumentElement(
                type="text", page=1, confidence=1.0,
                attributes={"source": "video_audio", "recognition_type": "audio", "audio_path": relative_audio_path, "duration_seconds": duration},
            ))

        for index, timestamp in enumerate(timestamps, start=1):
            frame = self._extract_frame(file_path, timestamp)
            frame_width, frame_height = frame.size
            pages.append(ParsedPage(page_number=index, width=float(frame_width), height=float(frame_height), image=frame))
            elements.append(RawDocumentElement(
                type="image", page=index, bbox=[0, 0, frame_width, frame_height], image=frame, confidence=1.0,
                attributes={"source": "video_frame", "timestamp_seconds": round(timestamp, 3)},
            ))

        metadata = {"title": file_path.stem, "page_count": len(pages), "duration_seconds": duration,
                    "video_width": width, "video_height": height, "sampled_frames": len(pages), "audio_path": relative_audio_path}
        logger.info("Parsed video %s: %d sampled frames", file_path.name, len(pages))
        return pages, elements, metadata

    def duration_seconds(self, file_path: Path) -> float:
        """Read duration without extracting audio or frames, for upload validation."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        self._require_ffmpeg()
        return float(self._probe(file_path).get("format", {}).get("duration") or 0.0)

    @staticmethod
    def _require_ffmpeg() -> None:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("FFmpeg and FFprobe are required for video processing")

    @staticmethod
    def _probe(file_path: Path) -> dict:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(file_path)], check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    @staticmethod
    def _extract_audio(file_path: Path, output_path: Path) -> None:
        subprocess.run(["ffmpeg", "-y", "-i", str(file_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_path)], check=True, capture_output=True)

    @staticmethod
    def _sample_timestamps(duration: float) -> List[float]:
        """Return one frame every 60 / configured-rate seconds (six per minute by default)."""
        interval = 60.0 / max(1, settings.VIDEO_FRAME_SAMPLE_RATE_PER_MINUTE)
        if duration <= 0:
            return [0.0]
        timestamps = []
        timestamp = 0.0
        while timestamp < duration:
            timestamps.append(timestamp)
            timestamp += interval
        return timestamps

    @staticmethod
    def _extract_frame(file_path: Path, timestamp: float) -> Image.Image:
        result = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(file_path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], check=True, capture_output=True)
        with Image.open(io.BytesIO(result.stdout)) as frame:
            image = frame.convert("RGB")
        if image.width > settings.VIDEO_FRAME_MAX_WIDTH:
            height = round(image.height * settings.VIDEO_FRAME_MAX_WIDTH / image.width)
            return image.resize((settings.VIDEO_FRAME_MAX_WIDTH, height), Image.Resampling.LANCZOS)
        return image


video_parser = VideoParser()

__all__ = ["VideoParser", "video_parser"]
