"""Resource-aware lifecycle for local recognition models."""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from backend.app.core.logging import logger


class ModelResourceManager:
    """Loads one specialist model at a time and releases native resources."""

    def __init__(self, loader: Callable[[], Any], name: str, unloader: Optional[Callable[[], None]] = None):
        self._loader = loader
        self._name = name
        self._unloader = unloader
        self._model: Optional[Any] = None

    @contextmanager
    def loaded(self) -> Iterator[Any]:
        if self._model is None:
            logger.info("Loading offline recognition model: %s", self._name)
            load_started = time.perf_counter()
            self._model = self._loader()
            logger.info(
                "Loaded offline recognition model: %s | elapsed=%.2fs",
                self._name,
                time.perf_counter() - load_started,
            )
        try:
            yield self._model
        finally:
            self.unload()

    def unload(self) -> None:
        if self._model is None:
            return
        model = self._model
        self._model = None
        if self._unloader is not None:
            self._unloader()
            logger.info("Unloaded offline recognition model: %s", self._name)
            return
        close = getattr(model, "close", None)
        if callable(close):
            close()
        del model
        gc.collect()
        logger.info("Unloaded offline recognition model: %s", self._name)


__all__ = ["ModelResourceManager"]
