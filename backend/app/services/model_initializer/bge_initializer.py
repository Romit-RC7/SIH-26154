"""
Lazy initializer and offline inference runner for BGE-small-en-v1.5 vector embeddings.
Generates 384-dimensional dense semantic vectors from local weights in models/bge_small_en_v1.5.
"""

from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
from typing import Any, List, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger


class BGEModelInitializer:
    """
    Validates, lazily loads, and runs the local BAAI/bge-small-en-v1.5 embedding model.
    Produces 384-dimensional normalized vector embeddings.
    """

    DIMENSION: int = 384

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or (settings.MODELS_DIR / "bge_small_en_v1.5")
        self.name = "BGE-small-en-v1.5"
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._backend: Optional[str] = None

    def is_available(self) -> bool:
        """Checks if local BGE weights directory exists and has model files."""
        if not self.model_dir.exists() or not self.model_dir.is_dir():
            return False
        # Check for safetensors, bin, or onnx
        has_weights = any(
            (self.model_dir / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin", "onnx/model.onnx")
        )
        has_config = (self.model_dir / "config.json").exists()
        return has_weights and has_config

    def load(self) -> Any:
        """
        Loads the BGE model using available runtimes in priority order:
        1. sentence_transformers
        2. transformers + torch
        3. onnxruntime
        4. deterministic fallback (for offline CI / non-torch test environments)
        """
        if self.model is not None:
            return self.model

        if not self.is_available():
            logger.warning("BGE model weights not found at %s; using deterministic fallback", self.model_dir)
            self._backend = "fallback"
            self.model = "fallback"
            return self.model

        # Attempt 1: sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading BGE model via sentence_transformers from %s", self.model_dir)
            self.model = SentenceTransformer(str(self.model_dir))
            self._backend = "sentence_transformers"
            return self.model
        except (ImportError, Exception) as e:
            logger.debug("sentence_transformers unavailable or failed (%s), trying transformers", e)

        # Attempt 2: transformers + torch
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            logger.info("Loading BGE model via transformers from %s", self.model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModel.from_pretrained(str(self.model_dir))
            self.model.eval()
            self._backend = "transformers"
            return self.model
        except (ImportError, Exception) as e:
            logger.debug("transformers/torch unavailable or failed (%s), trying onnxruntime", e)

        # Attempt 3: onnxruntime
        onnx_file = self.model_dir / "onnx" / "model.onnx"
        if onnx_file.exists():
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                logger.info("Loading BGE model via ONNX from %s", onnx_file)
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = ort.InferenceSession(str(onnx_file))
                self._backend = "onnx"
                return self.model
            except (ImportError, Exception) as e:
                logger.debug("ONNX runtime unavailable (%s)", e)

        # Fallback
        logger.info("Using deterministic 384-dim semantic embedding fallback for BGE")
        self._backend = "fallback"
        self.model = "fallback"
        return self.model

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Encodes a list of strings into 384-dimensional dense float vectors.
        """
        if not texts:
            return []

        if self.model is None:
            self.load()

        if self._backend == "sentence_transformers":
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            )
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

        elif self._backend == "transformers":
            import torch
            embeddings: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded_input = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    # Sentence-transformers mean pooling:
                    token_embeddings = model_output[0]
                    input_mask_expanded = encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    sentence_embeddings = sum_embeddings / sum_mask

                    if normalize_embeddings:
                        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

                    embeddings.extend(sentence_embeddings.cpu().numpy().tolist())
            return embeddings

        # Fallback deterministic pseudo-embedding (unit normalized 384-dim vector based on sha256 + token hashes)
        return [self._generate_deterministic_embedding(t) for t in texts]

    def encode_query(self, query: str) -> List[float]:
        """
        Encodes a search query with BGE recommended retrieval prompt instruction.
        """
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        results = self.encode([query_text], normalize_embeddings=True)
        return results[0]

    def _generate_deterministic_embedding(self, text: str) -> List[float]:
        """
        Deterministic 384-dim pseudo-vector generator for fallback & test environments.
        Computes stable normalized vectors based on token frequencies and hashing.
        """
        import re
        vec = [0.0] * self.DIMENSION
        if not text:
            return vec

        clean = text
        prefix = "Represent this sentence for searching relevant passages: "
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

        words = re.findall(r"\w+", clean.lower())
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            slot = h % self.DIMENSION
            vec[slot] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def unload(self) -> None:
        """Unloads model weights from memory."""
        self.model = None
        self.tokenizer = None
        self._backend = None
        gc.collect()
        logger.info("Unloaded local BGE embedding model")


# Global lazy initializer instance
bge_initializer = BGEModelInitializer()

__all__ = ["BGEModelInitializer", "bge_initializer"]
