"""Lazy initializer for the local Faster-Whisper speech-recognition stage."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger


class FasterWhisperInitializer:
    """Loads local CTranslate2 Whisper weights only for audio-bearing batches."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or settings.FASTER_WHISPER_MODEL_DIR
        self.model: Optional[Any] = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return self.model_dir.is_dir() and all(
            (self.model_dir / filename).is_file()
            for filename in ("config.json", "model.bin", "tokenizer.json")
        )

    def _resolve_device_and_compute_type(self) -> tuple[str, str]:
        requested_device = str(settings.FASTER_WHISPER_DEVICE).lower()
        requested_compute = str(settings.FASTER_WHISPER_COMPUTE_TYPE)

        if requested_device == "cuda" and settings.USE_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    _ = torch.cuda.device_count()
                    return ("cuda", requested_compute)
                else:
                    logger.warning(
                        "CUDA requested for Faster-Whisper, but CUDA driver/runtime is unavailable "
                        "or mismatched. Falling back to CPU (int8)."
                    )
            except Exception as exc:
                logger.warning(
                    "CUDA probe failed for Faster-Whisper: %s. Falling back to CPU (int8).",
                    exc,
                )
        return ("cpu", "int8")

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        if not self.is_available():
            raise FileNotFoundError(
                f"Local Faster-Whisper model package is incomplete: {self.model_dir}"
            )
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        from faster_whisper import WhisperModel

        device, compute_type = self._resolve_device_and_compute_type()
        logger.info(
            "Loading local Faster-Whisper model from %s on %s (%s)",
            self.model_dir,
            device,
            compute_type,
        )
        try:
            self.model = WhisperModel(
                str(self.model_dir),
                device=device,
                compute_type=compute_type,
                cpu_threads=2,
            )
        except Exception as exc:
            if device != "cpu":
                logger.warning(
                    "Faster-Whisper failed on %s (%s): %s. Retrying on CPU (float32).",
                    device,
                    compute_type,
                    exc,
                )
                self.model = WhisperModel(
                    str(self.model_dir),
                    device="cpu",
                    compute_type="float32",
                    cpu_threads=2,
                )
            else:
                logger.warning("Faster-Whisper int8 failed on CPU: %s. Retrying with float32.", exc)
                self.model = WhisperModel(
                    str(self.model_dir),
                    device="cpu",
                    compute_type="float32",
                    cpu_threads=2,
                )
        return self.model

    def unload(self) -> None:
        if self.model is None:
            return
        self.model = None
        gc.collect()
        logger.info("Unloaded local Faster-Whisper model")


faster_whisper_initializer = FasterWhisperInitializer()

__all__ = ["FasterWhisperInitializer", "faster_whisper_initializer"]
