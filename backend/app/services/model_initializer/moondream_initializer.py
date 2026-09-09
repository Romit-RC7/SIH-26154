"""Lazy offline initializer for Moondream2 (1.6B Vision-Language Model)."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional, Tuple

from backend.app.core.config import settings
from backend.app.core.logging import logger


class MoondreamInitializer:
    """Loads and unloads the local Moondream2 1.6B VLM stage."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or getattr(settings, "MOONDREAM_MODEL_DIR", settings.MODELS_DIR / "moondream2")
        self.name = "Moondream2-1.6B"
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def is_available(self) -> bool:
        if not self.model_dir.is_dir():
            return False
        # Check for config.json, model files, or GGUF weights
        has_config = (self.model_dir / "config.json").exists()
        has_weights = any(self.model_dir.glob("*.safetensors")) or any(self.model_dir.glob("*.bin")) or any(self.model_dir.glob("*.gguf"))
        return has_config or has_weights

    def load(self) -> Tuple[Any, Any]:
        """
        Lazily loads Moondream2 model and tokenizer onto available device (GPU or CPU).
        Returns tuple of (model, tokenizer).
        """
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer

        if not self.is_available():
            raise FileNotFoundError(f"Local {self.name} weights not found in {self.model_dir}")

        try:
            
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required to load local Moondream2 VLM model"
            ) from exc

        device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading local %s from %s (device=%s)", self.name, self.model_dir, device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True,local_files_only=True,)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
                dtype=dtype,
                local_files_only=True,
            )
            self.model = self.model.to(device)
            if device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()
            return self.model, self.tokenizer
        except Exception as exc:
            logger.error("Failed to load %s: %s", self.name, exc)
            self.unload()
            raise exc

    def unload(self) -> None:
        """Unloads model from memory and clears CUDA cache."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Unloaded local %s VLM model", self.name)


moondream_initializer = MoondreamInitializer()

__all__ = ["MoondreamInitializer", "moondream_initializer"]
