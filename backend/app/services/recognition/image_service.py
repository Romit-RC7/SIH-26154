"""Offline Qwen vision recognition for image and figure regions."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from backend.app.core.logging import logger
from backend.app.core.config import settings
from backend.app.processors.base import RawDocumentElement
from backend.app.services.model_initializer import qwen_vision_initializer
from backend.app.services.recognition.resource_manager import ModelResourceManager


class ImageRecognitionService:
    """Describes figures/images using one local Qwen2.5-VL model stage."""

    model_name = "Qwen2.5-VL-3B"

    def recognize(self, elements: List[RawDocumentElement]) -> None:
        # Do this check before loading Qwen: a layout label alone is not enough
        # to justify a multi-GB model stage when no visual crop is available.
        targets = [
            element
            for element in elements
            if element.type in ("image", "figure") and self._has_image_source(element)
        ]
        if not targets:
            logger.info("Skipping Qwen vision stage: no image or figure crops in this batch")
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
        image = self._image_source(element)
        if image is None:
            self._mark_error(element, "No image crop available")
            return

        # Reset llama.cpp KV cache between calls to avoid token/context leakage across frames
        reset_fn = getattr(model, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass

        is_video_frame = element.attributes.get("source") == "video_frame"
        subject = (
            "a sampled video frame"
            if is_video_frame else "a cropped document image"
        )
        ocr_text = str(element.attributes.get("ocr_text") or "").strip()
        ocr_evidence = (
            "\nOCR evidence from this image is quoted below. Use it only "
            "to verify text that is visibly present:\n"
            f"---\n{ocr_text}\n---\n"
            if ocr_text else ""
        )
        prompt = (
            f"Analyze {subject} precisely based strictly on what is directly visible.\n"
            "CRITICAL RULES:\n"
            "1. Identify the actual visual category in a short label (e.g. 'person', 'natural_scenery', 'document', 'illustration', 'screenshot', 'diagram', 'chart').\n"
            "2. Describe ONLY what is directly visible in THIS specific frame. Do NOT invent or repeat generic stock descriptions.\n"
            "3. Transcribe only labels, titles, or numbers that are visibly present.\n"
            "4. If text is absent, use an empty visible_text list.\n"
            f"{ocr_evidence}\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "visual_type": "<detected category>",\n'
            '  "description": "<objective description of visible content in this frame>",\n'
            '  "visible_text": ["<extracted text/labels if present>"],\n'
            '  "key_details": ["<key visual characteristics>"]\n'
            "}"
        )
        try:
            response = model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise computer vision system. Describe strictly what is "
                            "directly visible in the image. Never invent unmentioned objects or repeat stock templates."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": self._data_uri(image)}},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=384,
                repeat_penalty=1.1,
            )
            payload = self._response_payload(response)
        except Exception as exc:
            logger.warning("Error during Qwen vision inference: %s", exc)
            payload = {"error": str(exc)}

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
    def _data_uri(image: Image.Image, max_dim: int = 672) -> str:
        img = image.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _has_image_source(element: RawDocumentElement) -> bool:
        if element.image is not None:
            return True
        saved_path = element.attributes.get("saved_image_path")
        return bool(saved_path and (settings.BASE_DIR / Path(saved_path)).is_file())

    @staticmethod
    def _image_source(element: RawDocumentElement) -> Optional[Image.Image]:
        if element.image is not None:
            return element.image
        saved_path = element.attributes.get("saved_image_path")
        if not saved_path:
            return None
        image_path = settings.BASE_DIR / Path(saved_path)
        if not image_path.is_file():
            return None
        with Image.open(image_path) as image:
            return image.copy()

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
