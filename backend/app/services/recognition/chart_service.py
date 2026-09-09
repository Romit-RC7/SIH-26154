"""Offline chart recognition for chart regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import warnings

from PIL import Image

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.processors.base import RawDocumentElement
from backend.app.services.recognition.resource_manager import ModelResourceManager
from backend.app.services.model_initializer import unichart_initializer


class ChartRecognitionService:
    """Runs PP-Chart2Table once across all chart elements in a stage."""

    model_name = "PP-Chart2Table"

    def __init__(self, models_root: Optional[Path] = None):
        self.model_dir = (models_root or settings.PP_STRUCTURE_MODEL_DIR) / "chart"

    def recognize(self, elements: List[RawDocumentElement]) -> None:
        targets = [element for element in elements if element.type == "chart"]
        if not targets:
            return
        if not self._is_ready():
            self._recognize_with_unichart(targets)
            return

        manager = ModelResourceManager(self._load_model, self.model_name)
        try:
            with manager.loaded() as model:
                for element in targets:
                    self._recognize_element(model, element)
        except Exception as exc:
            logger.warning("Chart recognition unavailable: %s", exc)
            for element in targets:
                self._mark_error(element, str(exc))

    def recognize_loaded(self, model: Any, elements: List[RawDocumentElement]) -> None:
        """Process chart elements with a model owned by a shared stage."""
        for element in (item for item in elements if item.type == "chart"):
            try:
                self._recognize_element(model, element)
            except Exception as exc:
                self._mark_error(element, str(exc))

    def _is_ready(self) -> bool:
        if not (
            self.model_dir.is_dir()
            and (self.model_dir / "model_state.pdparams").exists()
            and (self.model_dir / "inference.yml").exists()
        ):
            return False
        try:
            from paddle.incubate.nn.functional import fused_rms_norm_ext  # noqa: F401
            return True
        except ImportError:
            logger.warning("PP-Chart2Table unavailable: Paddle runtime lacks fused_rms_norm_ext")
            return False

    def _load_model(self) -> Any:
        from paddlex import create_model

        return create_model(model_name=self.model_name, model_dir=str(self.model_dir))

    def _recognize_with_unichart(self, targets: List[RawDocumentElement]) -> None:
        """Use the staged UniChart weights when PP-Chart2Table is unavailable."""
        if not unichart_initializer.is_available():
            self._mark_unavailable(targets)
            return
        started = time.perf_counter()
        try:
            model, processor = unichart_initializer.load()
            import torch
            for element in targets:
                image = element.image
                if image is None:
                    self._mark_error(element, "No chart crop available")
                    continue
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                pixel_values = processor(image, return_tensors="pt").pixel_values.to(model.device)
                decoder_input_ids = processor.tokenizer(
                    "<extract_data_table> <s_answer>", add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(model.device)
                with torch.inference_mode():
                    # UniChart's saved generation config does not use beam
                    # search, so `early_stopping` is ignored by Transformers.
                    # Its legacy cache warning is emitted by this older model
                    # implementation and does not affect output correctness.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Passing a tuple of `past_key_values` is deprecated.*",
                        )
                        output_ids = model.generate(
                            pixel_values,
                            decoder_input_ids=decoder_input_ids,
                            max_length=model.decoder.config.max_position_embeddings,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            use_cache=True,
                            num_beams=4,
                            bad_words_ids=[[processor.tokenizer.unk_token_id]],
                        )
                sequence = processor.batch_decode(output_ids)[0]
                sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                answer = sequence.split("<s_answer>", 1)[-1].strip()
                if not answer or len(answer.split()) > 16 and len(set(answer.split())) <= 2:
                    raise RuntimeError("UniChart returned an unusable chart extraction")
                element.attributes["chart_recognition"] = {"answer": answer}
                element.attributes["chart_recognition_model"] = "UniChart Base 960"
                element.attributes["chart_recognition_status"] = "completed"
            logger.info("Chart recognition fallback completed | model=UniChart Base 960 | charts=%d | elapsed=%.2fs", len(targets), time.perf_counter() - started)
        except Exception as exc:
            logger.warning("UniChart fallback unavailable: %s", exc)
            for element in targets:
                self._mark_error(element, str(exc))
        finally:
            unichart_initializer.unload()

    @staticmethod
    def _recognize_element(model: Any, element: RawDocumentElement) -> None:
        source = element.image or element.attributes.get("saved_image_path")
        if source is None:
            ChartRecognitionService._mark_error(element, "No chart crop available")
            return
        result = next(iter(model.predict(input=source, batch_size=1)), None)
        element.attributes["chart_recognition"] = ChartRecognitionService._payload(result)
        element.attributes["chart_recognition_model"] = ChartRecognitionService.model_name
        element.attributes["chart_recognition_status"] = "completed"

    @staticmethod
    def _payload(result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        value = getattr(result, "json", None)
        if callable(value):
            value = value()
        return value if isinstance(value, dict) else {"raw": str(value or result)}

    def _mark_unavailable(self, elements: List[RawDocumentElement]) -> None:
        for element in elements:
            element.attributes["chart_recognition_status"] = "unavailable"
            element.attributes["chart_recognition_model"] = self.model_name
            element.attributes["chart_recognition_error"] = f"Local model package not found: {self.model_dir}"

    @staticmethod
    def _mark_error(element: RawDocumentElement, error: str) -> None:
        element.attributes["chart_recognition_status"] = "failed"
        element.attributes["chart_recognition_model"] = ChartRecognitionService.model_name
        element.attributes["chart_recognition_error"] = error


chart_recognition_service = ChartRecognitionService()

__all__ = ["ChartRecognitionService", "chart_recognition_service"]
