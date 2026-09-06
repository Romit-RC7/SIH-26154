"""Lazy initializer for the local UniChart chart-comprehension model."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger


class UniChartInitializer:
    """Loads UniChart locally only when the chart stage requests it."""

    def __init__(self, model_dir: Optional[Path] = None, device: Optional[str] = None):
        self.model_dir = model_dir or settings.MODELS_DIR / "unichart_base_960"
        self.device_name = device
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.device: Optional[Any] = None

    def is_available(self) -> bool:
        try:
            import torch
            import transformers
        except ImportError:
            return False

        required_files = (
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "pytorch_model.bin",
        )
        return self.model_dir.is_dir() and all(
            (self.model_dir / filename).exists() for filename in required_files
        )

    def load(self) -> tuple[Any, Any]:
        if self.model is not None and self.processor is not None:
            return self.model, self.processor
        if not self.is_available():
            raise FileNotFoundError(
                f"Local UniChart model package is incomplete: {self.model_dir}"
            )

        try:
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required to load local UniChart weights"
            ) from exc

        self.device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Loading local UniChart model from %s on %s", self.model_dir, self.device)
        self.processor = DonutProcessor.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        return self.model, self.processor

    def unload(self) -> None:
        if self.model is None and self.processor is None:
            return
        model = self.model
        self.model = None
        self.processor = None
        self.device = None
        if model is not None:
            del model
        gc.collect()
        logger.info("Unloaded local UniChart model")


unichart_initializer = UniChartInitializer()

__all__ = ["UniChartInitializer", "unichart_initializer"]
