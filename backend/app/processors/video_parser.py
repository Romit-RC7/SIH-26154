"""Video parser that extracts a WAV track and samples key visual frames using pixel-difference thresholding."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.processors.base import ParsedPage, RawDocumentElement


class VideoParser:
    """Uses FFmpeg/FFprobe with low-resolution NumPy pixel-diff filtering to extract keyframes efficiently."""

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

        keyframes = self.extract_keyframes(file_path, duration)
        pages: List[ParsedPage] = []
        elements: List[RawDocumentElement] = []

        if relative_audio_path:
            elements.append(RawDocumentElement(
                type="text", page=1, confidence=1.0,
                attributes={
                    "source": "video_audio",
                    "recognition_type": "audio",
                    "audio_path": relative_audio_path,
                    "duration_seconds": duration,
                },
            ))

        for index, (timestamp, frame) in enumerate(keyframes, start=1):
            frame_width, frame_height = frame.size
            pages.append(ParsedPage(page_number=index, width=float(frame_width), height=float(frame_height), image=frame))
            elements.append(RawDocumentElement(
                type="image", page=index, bbox=[0, 0, frame_width, frame_height], image=frame, confidence=1.0,
                attributes={"source": "video_frame", "timestamp_seconds": round(timestamp, 3)},
            ))

        metadata = {
            "title": file_path.stem,
            "page_count": len(pages),
            "duration_seconds": duration,
            "video_width": width,
            "video_height": height,
            "sampled_frames": len(pages),
            "audio_path": relative_audio_path,
            "frame_diff_threshold": settings.VIDEO_FRAME_DIFF_THRESHOLD,
        }
        logger.info("Parsed video %s: %d keyframes selected via frame-diff filtering", file_path.name, len(pages))
        return pages, elements, metadata

    def extract_keyframes(self, file_path: Path, duration: float) -> List[Tuple[float, Image.Image]]:
        """
        Samples candidate frames at higher frequency and filters visual duplicates using pixel difference.
        """
        candidate_timestamps = self._sample_candidate_timestamps(duration)
        if not candidate_timestamps:
            return []

        keyframes: List[Tuple[float, Image.Image]] = []
        last_keyframe: Optional[Image.Image] = None

        for ts in candidate_timestamps:
            frame = self._extract_frame(file_path, ts)
            if last_keyframe is None:
                keyframes.append((ts, frame))
                last_keyframe = frame
            else:
                diff = self._compute_frame_diff(last_keyframe, frame)
                if diff >= settings.VIDEO_FRAME_DIFF_THRESHOLD:
                    keyframes.append((ts, frame))
                    last_keyframe = frame

        logger.info(
            "Frame-diff filtering on %s: %d candidate frames -> %d unique keyframes (threshold=%.3f)",
            file_path.name, len(candidate_timestamps), len(keyframes), settings.VIDEO_FRAME_DIFF_THRESHOLD
        )

        # Cap keyframes to VIDEO_MAX_KEYFRAMES if exceeded
        max_k = getattr(settings, "VIDEO_MAX_KEYFRAMES", 20)
        if len(keyframes) > max_k:
            step = len(keyframes) / float(max_k)
            keyframes = [keyframes[int(i * step)] for i in range(max_k)]
            logger.info("Capped keyframes to max limit of %d", max_k)

        return keyframes

    @staticmethod
    def _compute_frame_diff(img1: Image.Image, img2: Image.Image) -> float:
        """
        Computes mean normalized pixel difference ratio between two images at 320x180 resolution.
        """
        a = np.asarray(img1.resize((320, 180))).astype(np.float32)
        b = np.asarray(img2.resize((320, 180))).astype(np.float32)
        return float(np.mean(np.abs(a - b)) / 255.0)

    @staticmethod
    def _sample_candidate_timestamps(duration: float) -> List[float]:
        """
        Samples candidate timestamps every 1 / VIDEO_CANDIDATE_FPS seconds (~3 seconds).
        """
        interval = 1.0 / max(0.05, getattr(settings, "VIDEO_CANDIDATE_FPS", 0.333))
        if duration <= 0:
            return [0.0]
        timestamps = []
        timestamp = 0.0
        while timestamp < duration:
            timestamps.append(timestamp)
            timestamp += interval
        return timestamps

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
