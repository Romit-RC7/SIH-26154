"""Lazy initializers for the local Qwen GGUF model stages."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger


class QwenModelInitializer:
    """Validates and lazily loads one local Qwen GGUF model."""

    def __init__(self, model_dir: Path, model_pattern: str, name: str, n_ctx: int = 4096):
        self.model_dir = model_dir
        self.model_pattern = model_pattern
        self.name = name
        self.n_ctx = n_ctx
        self.model: Optional[Any] = None

    @property
    def model_path(self) -> Optional[Path]:
        matches = sorted(self.model_dir.glob(self.model_pattern))
        return matches[0] if matches else None

    def is_available(self) -> bool:
        return self.model_path is not None

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        model_path = self.model_path
        if model_path is None:
            raise FileNotFoundError(f"Local {self.name} GGUF model not found in {self.model_dir}")

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to load local Qwen GGUF models"
            ) from exc

        logger.info("Loading local Qwen model: %s (n_gpu_layers=%s)", model_path, settings.N_GPU_LAYERS)
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=settings.N_GPU_LAYERS,
            verbose=False,
        )
        return self.model

    def unload(self) -> None:
        if self.model is None:
            return
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
        self.model = None
        gc.collect()
        logger.info("Unloaded local Qwen model: %s", self.name)


class QwenVisionInitializer(QwenModelInitializer):
    """Lazy Qwen2.5-VL initializer requiring a local multimodal projector."""

    def __init__(self, model_dir: Optional[Path] = None, n_ctx: int = 4096):
        super().__init__(
            model_dir=model_dir or settings.MODELS_DIR / "qwen2.5_vl_3b_q4",
            model_pattern="Qwen2.5-VL*.gguf",
            name="Qwen2.5-VL-3B",
            n_ctx=n_ctx,
        )

    @property
    def projector_path(self) -> Optional[Path]:
        matches = sorted(self.model_dir.glob("*mmproj*.gguf"))
        return matches[0] if matches else None

    def is_available(self) -> bool:
        return super().is_available() and self.projector_path is not None

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        model_path = self.model_path
        projector_path = self.projector_path
        if model_path is None or projector_path is None:
            raise FileNotFoundError(
                f"Local {self.name} model and projector are required in {self.model_dir}"
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to load local Qwen vision models"
            ) from exc

        logger.info("Loading local Qwen vision model: %s (n_gpu_layers=%s)", model_path, settings.N_GPU_LAYERS)
        self.model = Llama(
            model_path=str(model_path),
            clip_model_path=str(projector_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=settings.N_GPU_LAYERS,
            verbose=False,
        )
        return self.model


class QwenFusionInitializer(QwenModelInitializer):
    """Lazy Qwen3-4B initializer used for structured fusion."""

    def __init__(self, model_dir: Optional[Path] = None, n_ctx: int = 8192):
        super().__init__(
            model_dir=model_dir or settings.MODELS_DIR / "qwen3_4b_q4",
            model_pattern="Qwen3-4B*.gguf",
            name="Qwen3-4B",
            n_ctx=n_ctx,
        )


qwen_vision_initializer = QwenVisionInitializer()
qwen_fusion_initializer = QwenFusionInitializer()

__all__ = [
    "QwenModelInitializer",
    "QwenVisionInitializer",
    "QwenFusionInitializer",
    "qwen_vision_initializer",
    "qwen_fusion_initializer",
]
