"""Offline Faster-Whisper transcription for extracted video audio."""

from __future__ import annotations

from pathlib import Path
from typing import List

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.processors.base import RawDocumentElement
from backend.app.services.model_initializer import faster_whisper_initializer
from backend.app.services.recognition.resource_manager import ModelResourceManager


class SpeechRecognitionService:
    model_name = "Faster-Whisper-small"

    def recognize(self, elements: List[RawDocumentElement]) -> None:
        targets = [
            element for element in elements
            if element.attributes.get("recognition_type") == "audio"
            and self._audio_path(element) is not None
        ]
        if not targets:
            logger.info("Skipping Faster-Whisper stage: no extracted audio in this batch")
            return
        if not faster_whisper_initializer.is_available():
            self._mark_unavailable(targets)
            return

        manager = ModelResourceManager(
            faster_whisper_initializer.load,
            self.model_name,
            unloader=faster_whisper_initializer.unload,
        )
        try:
            with manager.loaded() as model:
                for element in targets:
                    audio_path = self._audio_path(element)
                    segments, info = model.transcribe(str(audio_path), vad_filter=True)
                    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
                    element.text = transcript
                    element.attributes["audio_transcription"] = {
                        "language": getattr(info, "language", None),
                        "language_probability": getattr(info, "language_probability", None),
                        "text": transcript,
                    }
                    element.attributes["audio_transcription_model"] = self.model_name
                    element.attributes["audio_transcription_status"] = "completed"
        except Exception as exc:
            logger.warning("Faster-Whisper stage unavailable: %s", exc)
            for element in targets:
                self._mark_error(element, str(exc))

    @staticmethod
    def _audio_path(element: RawDocumentElement) -> Path | None:
        audio_path = element.attributes.get("audio_path")
        if not audio_path:
            return None
        candidate = settings.BASE_DIR / Path(audio_path)
        return candidate if candidate.is_file() else None

    def _mark_unavailable(self, elements: List[RawDocumentElement]) -> None:
        for element in elements:
            element.attributes["audio_transcription_status"] = "unavailable"
            element.attributes["audio_transcription_model"] = self.model_name
            element.attributes["audio_transcription_error"] = "Local Faster-Whisper model is unavailable"

    def _mark_error(self, element: RawDocumentElement, error: str) -> None:
        element.attributes["audio_transcription_status"] = "failed"
        element.attributes["audio_transcription_model"] = self.model_name
        element.attributes["audio_transcription_error"] = error


speech_recognition_service = SpeechRecognitionService()

__all__ = ["SpeechRecognitionService", "speech_recognition_service"]
