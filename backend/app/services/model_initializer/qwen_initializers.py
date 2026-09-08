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
        self.chat_handler: Optional[Any] = None

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
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to load local Qwen vision models"
            ) from exc

        logger.info("Loading local Qwen vision model: %s (n_gpu_layers=%s)", model_path, settings.N_GPU_LAYERS)
        self.chat_handler = Qwen25VLChatHandler(
            clip_model_path=str(projector_path),
            verbose=False,
        )
        self.model = Llama(
            model_path=str(model_path),
            chat_handler=self.chat_handler,
            n_ctx=self.n_ctx,
            n_gpu_layers=settings.N_GPU_LAYERS,
            verbose=False,
        )
        return self.model

    def unload(self) -> None:
        super().unload()
        self.chat_handler = None


class QwenFusionInitializer(QwenModelInitializer):
    """Lazy Qwen3-4B initializer used for structured fusion."""

    def __init__(self, model_dir: Optional[Path] = None, n_ctx: int = 8192):
        super().__init__(
            model_dir=model_dir or settings.MODELS_DIR / "qwen3_4b_q4",
            model_pattern="Qwen3-4B*.gguf",
            name="Qwen3-4B",
            n_ctx=n_ctx,
        )


class QwenOrchestratorInitializer(QwenModelInitializer):
    """
    Lazy Qwen3-8B initializer used for multi-format content orchestration and generation.
    Includes VRAM-aware GPU layer allocation and graceful fallback to CPU or Qwen3-4B.
    """

    def __init__(self, model_dir: Optional[Path] = None, n_ctx: int = 8192):
        super().__init__(
            model_dir=model_dir or settings.MODELS_DIR / "qwen3_8b_q4",
            model_pattern="Qwen3-8B*.gguf",
            name="Qwen3-8B",
            n_ctx=n_ctx,
        )
        self.active_model_name = "Qwen3-8B"

    def _determine_gpu_layers(self) -> int:
        """Calculates safe n_gpu_layers based on available VRAM."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("Detected CUDA device with %.2f GB VRAM", vram_gb)
            if vram_gb >= 5.5:
                return settings.N_GPU_LAYERS  # Full offload (-1)
            elif vram_gb >= 3.0:
                return 16  # Partial offload
            else:
                logger.warning("Low VRAM (%.2f GB) detected. Offloading 0 layers to GPU (CPU mode)", vram_gb)
                return 0
        except Exception as exc:
            logger.warning("Could not probe VRAM (%s); using default config N_GPU_LAYERS", exc)
            return settings.N_GPU_LAYERS

    def load(self) -> Any:
        if self.model is not None:
            return self.model

        # 1. Check primary 8B model path
        model_path = self.model_path
        if model_path is None:
            # Fallback to Qwen3-4B if 8B is not downloaded
            fallback_dir = settings.MODELS_DIR / "qwen3_4b_q4"
            fallback_matches = sorted(fallback_dir.glob("Qwen3-4B*.gguf")) if fallback_dir.exists() else []
            if fallback_matches:
                logger.warning("Qwen3-8B not found in %s; falling back to Qwen3-4B at %s", self.model_dir, fallback_matches[0])
                model_path = fallback_matches[0]
                self.active_model_name = "Qwen3-4B (Fallback)"
            else:
                raise FileNotFoundError(
                    f"Neither Qwen3-8B ({self.model_dir}) nor Qwen3-4B ({fallback_dir}) GGUF model files exist."
                )

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to load Qwen GGUF generation models"
            ) from exc

        n_gpu_layers = self._determine_gpu_layers()
        logger.info("Loading Qwen Orchestrator: %s (n_gpu_layers=%s, n_ctx=%s)", model_path, n_gpu_layers, self.n_ctx)

        try:
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            if n_gpu_layers != 0:
                logger.warning("Failed to load on GPU (%s); attempting fallback on CPU...", exc)
                self.model = Llama(
                    model_path=str(model_path),
                    n_ctx=self.n_ctx,
                    n_gpu_layers=0,
                    verbose=False,
                )
            else:
                raise exc

        return self.model


qwen_vision_initializer = QwenVisionInitializer()
qwen_fusion_initializer = QwenFusionInitializer()
qwen_orchestrator_initializer = QwenOrchestratorInitializer()

__all__ = [
    "QwenModelInitializer",
    "QwenVisionInitializer",
    "QwenFusionInitializer",
    "QwenOrchestratorInitializer",
    "qwen_vision_initializer",
    "qwen_fusion_initializer",
    "qwen_orchestrator_initializer",
]
