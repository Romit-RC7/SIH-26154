"""Sequential offline recognition for formula and chart regions."""

from __future__ import annotations

from pathlib import Path
from contextlib import ExitStack
from typing import Any, Dict, List, Optional

from PIL import Image

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.processors.base import RawDocumentElement
from backend.app.services.recognition.resource_manager import ModelResourceManager
from backend.app.services.recognition.chart_service import chart_recognition_service
from backend.app.services.recognition.image_service import image_recognition_service
from backend.app.services.recognition.speech_service import speech_recognition_service


class RecognitionCoordinator:
    """Runs specialist models only on matching cropped document elements."""

    def __init__(self, models_root: Optional[Path] = None):
        self.models_root = models_root or settings.PP_STRUCTURE_MODEL_DIR

    def recognize(self, elements: List[RawDocumentElement]) -> List[RawDocumentElement]:
        self._run_formula_chart_stage(elements)
        speech_recognition_service.recognize(elements)
        image_recognition_service.recognize(elements)
        return elements

    def recognize_batch(
        self,
        documents: List[List[RawDocumentElement]],
    ) -> List[List[RawDocumentElement]]:
        """Run each specialist once across all regions from a document batch."""
        elements = [element for document in documents for element in document]
        self._run_formula_chart_stage(elements)
        speech_recognition_service.recognize(elements)
        image_recognition_service.recognize(elements)
        return documents

    def _run_formula_chart_stage(self, elements: List[RawDocumentElement]) -> None:
        """Keep formula and chart models resident during one shared stage."""
        formula_targets = [
            element for element in elements
            if element.type == "formula"
            or element.attributes.get("recognition_type") == "formula"
        ]
        chart_targets = [element for element in elements if element.type == "chart"]
        if not formula_targets and not chart_targets:
            return

        formula_dir = self.models_root / "formula"
        formula_ready = self._model_files_ready(formula_dir, "formula")
        chart_ready = chart_recognition_service._is_ready()
        if not formula_ready:
            self._mark_unavailable(formula_targets, "formula", "PP-FormulaNet_plus-M", formula_dir)
        if not chart_ready:
            chart_recognition_service._mark_unavailable(chart_targets)

        formula_manager = ModelResourceManager(
            loader=lambda: self._create_model("PP-FormulaNet_plus-M", formula_dir),
            name="PP-FormulaNet_plus-M",
        ) if formula_ready and formula_targets else None
        chart_manager = ModelResourceManager(
            loader=chart_recognition_service._load_model,
            name="PP-Chart2Table",
        ) if chart_ready and chart_targets else None

        try:
            with ExitStack() as stack:
                formula_model = formula_manager and stack.enter_context(formula_manager.loaded())
                chart_model = chart_manager and stack.enter_context(chart_manager.loaded())
                if formula_model is not None:
                    for element in formula_targets:
                        self._recognize_element(formula_model, element, "formula")
                if chart_model is not None:
                    chart_recognition_service.recognize_loaded(chart_model, chart_targets)
        except Exception as exc:
            logger.warning("Formula/chart recognition stage unavailable: %s", exc)
            for element in formula_targets:
                self._mark_error(element, "PP-FormulaNet_plus-M", str(exc))
            for element in chart_targets:
                chart_recognition_service._mark_error(element, str(exc))

    def _run_stage(
        self,
        elements: List[RawDocumentElement],
        element_type: str,
        model_name: str,
    ) -> None:
        targets = [
            element for element in elements
            if element.type == element_type
            or element.attributes.get("recognition_type") == element_type
        ]
        if not targets:
            return

        model_dir = self.models_root / element_type
        if not self._model_files_ready(model_dir, element_type):
            self._mark_unavailable(targets, element_type, model_name, model_dir)
            return

        manager = ModelResourceManager(
            loader=lambda: self._create_model(model_name, model_dir),
            name=model_name,
        )
        try:
            with manager.loaded() as model:
                for element in targets:
                    self._recognize_element(model, element, element_type)
        except Exception as exc:
            logger.warning("%s recognition stage unavailable: %s", model_name, exc)
            for element in targets:
                self._mark_error(element, model_name, str(exc))

    @staticmethod
    def _create_model(model_name: str, model_dir: Path) -> Any:
        from paddlex import create_model

        return create_model(model_name=model_name, model_dir=str(model_dir))

    @staticmethod
    def _model_files_ready(model_dir: Path, element_type: str) -> bool:
        if not model_dir.is_dir():
            return False
        if element_type == "chart":
            return (model_dir / "model_state.pdparams").exists() and (model_dir / "inference.yml").exists()
        return (model_dir / "inference.pdiparams").exists() and (model_dir / "inference.yml").exists()

    def _recognize_element(self, model: Any, element: RawDocumentElement, element_type: str) -> None:
        source = element.image
        if source is None:
            saved_path = element.attributes.get("saved_image_path")
            source = self._absolute_artifact_path(saved_path)
        if source is None:
            self._mark_error(element, element_type, "No image crop available")
            return

        result = next(iter(model.predict(input=source, batch_size=1)), None)
        payload = self._result_payload(result)
        element.attributes[f"{element_type}_recognition"] = payload
        element.attributes[f"{element_type}_recognition_model"] = element_type
        element.attributes[f"{element_type}_recognition_status"] = "completed"

    def _absolute_artifact_path(self, saved_path: Optional[str]) -> Optional[str]:
        if not saved_path:
            return None
        candidate = settings.BASE_DIR / saved_path
        return str(candidate) if candidate.exists() else None

    @staticmethod
    def _result_payload(result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        value = getattr(result, "json", None)
        if callable(value):
            value = value()
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return {"raw": value}
        return {"raw": str(result)}

    @staticmethod
    def _mark_unavailable(
        elements: List[RawDocumentElement],
        element_type: str,
        model_name: str,
        model_dir: Path,
    ) -> None:
        for element in elements:
            element.attributes[f"{element_type}_recognition_status"] = "unavailable"
            element.attributes[f"{element_type}_recognition_model"] = model_name
            element.attributes[f"{element_type}_recognition_error"] = f"Local model package not found: {model_dir}"

    @staticmethod
    def _mark_error(element: RawDocumentElement, model_name: str, error: str) -> None:
        element_type = element.attributes.get("recognition_type", element.type)
        element.attributes[f"{element_type}_recognition_status"] = "failed"
        element.attributes[f"{element_type}_recognition_model"] = model_name
        element.attributes[f"{element_type}_recognition_error"] = error


recognition_coordinator = RecognitionCoordinator()

__all__ = ["RecognitionCoordinator", "recognition_coordinator"]
