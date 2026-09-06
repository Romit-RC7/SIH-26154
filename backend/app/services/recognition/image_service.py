"""Offline Qwen vision recognition for image and figure regions."""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, List, Optional

from PIL import Image

from backend.app.core.logging import logger
from backend.app.processors.base import RawDocumentElement
from backend.app.services.model_initializer import qwen_vision_initializer
from backend.app.services.recognition.resource_manager import ModelResourceManager


class ImageRecognitionService:
    """Describes figures/images using one local Qwen2.5-VL model stage."""

    model_name = "Qwen2.5-VL-3B"

    def recognize(self, elements: List[RawDocumentElement]) -> None:
        targets = [element for element in elements if element.type in ("image", "figure")]
        if not targets:
            return
        if not qwen_vision_initializer.is_available():
            self._mark_unavailable(targets)
            return

        manager = ModelResourceManager(
            qwen_vision_initializer.load,
            self.model_name,
            unloader=qwen_vision_initializer.unload,
        )
        try:
            with manager.loaded() as model:
                for element in targets:
                    self._recognize_element(model, element)
        except Exception as exc:
            logger.warning("Visual image recognition unavailable: %s", exc)
            for element in targets:
                self._mark_error(element, str(exc))

    def _recognize_element(self, model: Any, element: RawDocumentElement) -> None:
        if element.image is None:
            self._mark_error(element, "No image crop available")
            return
        prompt = (
            "Examine this cropped document image precisely. "
            "1. Classify its visual category (e.g. 'natural_scenery', 'illustration', 'line_chart', 'bar_chart', 'flowchart', 'architecture_diagram', 'table_graphic', 'screenshot', 'logo').\n"
            "2. Provide an accurate, objective description of what is actually depicted in the image.\n"
            "3. Extract any visible text labels, titles, or numbers faithfully.\n"
            "4. DO NOT invent or assume unmentioned external technologies or components.\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "visual_type": "<detected category>",\n'
            '  "description": "<objective description of visible content>",\n'
            '  "visible_text": ["<extracted text/labels if present>"],\n'
            '  "key_details": ["<key visual characteristics>"]\n'
            "}"
        )
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._data_uri(element.image)}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=512,
        )
        payload = self._response_payload(response)
        element.attributes["visual_analysis"] = payload
        element.attributes["visual_analysis_model"] = self.model_name
        element.attributes["visual_analysis_status"] = "completed"

        # Dynamically classify element type based on VLM ground-truth recognition
        if isinstance(payload, dict):
            vis_type = str(payload.get("visual_type", "")).lower()
            if "chart" in vis_type or "graph" in vis_type or "plot" in vis_type:
                element.type = "chart"
            elif "diagram" in vis_type or "flowchart" in vis_type:
                element.type = "figure"

    @staticmethod
    def _data_uri(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @classmethod
    def _response_payload(cls, response: Any) -> Dict[str, Any]:
        text_content = ""
        raw_response = response
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text_content = message.get("content", "")
            else:
                text_content = str(response)
        else:
            text_content = str(response)

        parsed_json = cls._extract_json(text_content)
        if parsed_json:
            parsed_json["_raw_text"] = text_content
            return parsed_json
        return {"text": text_content, "raw": raw_response}

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from raw text or markdown codeblocks."""
        if not text:
            return None
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = match.group(1) if match else text.strip()
        try:
            val = json.loads(candidate)
            if isinstance(val, dict):
                return val
        except Exception:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end > start:
                try:
                    val = json.loads(candidate[start : end + 1])
                    if isinstance(val, dict):
                        return val
                except Exception:
                    pass
        return None

    def _mark_unavailable(self, elements: List[RawDocumentElement]) -> None:
        for element in elements:
            element.attributes["visual_analysis_status"] = "unavailable"
            element.attributes["visual_analysis_model"] = self.model_name
            element.attributes["visual_analysis_error"] = "Local Qwen vision model or projector is unavailable"

    @staticmethod
    def _mark_error(element: RawDocumentElement, error: str) -> None:
        element.attributes["visual_analysis_status"] = "failed"
        element.attributes["visual_analysis_model"] = ImageRecognitionService.model_name
        element.attributes["visual_analysis_error"] = error


image_recognition_service = ImageRecognitionService()

__all__ = ["ImageRecognitionService", "image_recognition_service"]
