"""Offline Qwen vision recognition for image and figure regions."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

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
        """Recognize a single element with comprehensive debug logging and hallucination prevention."""
        
        # ===== STEP 1: Load image and verify source =====
        image = self._image_source(element)
        if image is None:
            self._mark_error(element, "No image crop available")
            return

        element_id = element.attributes.get("element_id", "unknown")
        element_type = element.type
        source = element.attributes.get("source", "document")
        saved_image_path = element.attributes.get("saved_image_path", "N/A")
        
        logger.info(
            "Qwen vision pipeline START | element_id=%s | type=%s | source=%s | path=%s",
            element_id, element_type, source, saved_image_path
        )

        # ===== STEP 2: Verify image crop source =====
        logger.info(
            "Image ready for Qwen vision | element_id=%s | path=%s | size=%dx%d",
            element_id, saved_image_path, image.width, image.height
        )

        # ===== STEP 3: Extract and log OCR evidence =====
        ocr_text = str(element.attributes.get("ocr_text") or "").strip()
        ocr_model = element.attributes.get("ocr_model", "N/A")
        ocr_status = element.attributes.get("ocr_status", "N/A")
        logger.info(
            "OCR metadata | element_id=%s | status=%s | model=%s | text_length=%d | text_preview=%s",
            element_id, ocr_status, ocr_model, len(ocr_text), ocr_text[:100] if ocr_text else "(empty)"
        )

        # ===== STEP 4: Reset KV cache =====
        reset_fn = getattr(model, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass

        # ===== STEP 5: Build simplified prompt (no OCR evidence by default to reduce contamination) =====
        # To debug OCR contamination, set INCLUDE_OCR_EVIDENCE=true in environment
        include_ocr = os.environ.get("INCLUDE_OCR_EVIDENCE", "false").lower() == "true"
        
        ocr_section = (
            f"\nOCR detected: {ocr_text}\n"
            if (include_ocr and ocr_text)
            else ""
        )

        # Grounded multimodal prompt schema
        prompt = (
            "Inspect the image using ONLY the pixels visible in the provided image.\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. Describe only what is directly visible. Do not guess, infer, assume, or add information not present.\n"
            "2. Identify the visual category (e.g. natural_scenery, person, photograph, illustration, screenshot, diagram, chart, document, meme, social_media_graphic).\n"
            "3. Extract only text that is actually visible. If no text is visible, return an empty list.\n"
            "4. If the image is a meme or social media graphic, extract overlay text into 'overlay_text' and the theme/concept into 'core_concept_or_humor_theme'. Otherwise set overlay_text to [] and core_concept_or_humor_theme to null.\n"
            f"{ocr_section}\n"
            "Return only a JSON object:\n\n"
            "{\n"
            '  "visual_type": "<detected category>",\n'
            '  "description": "<short objective description containing only directly visible elements>",\n'
            '  "visible_text": [],\n'
            '  "overlay_text": [],\n'
            '  "core_concept_or_humor_theme": null,\n'
            '  "key_details": []\n'
            "}"
        )

        logger.info(
            "Prompt prepared | element_id=%s | include_ocr=%s | prompt_len=%d",
            element_id, include_ocr, len(prompt)
        )

        # ===== STEP 6: Call Qwen =====
        try:
            logger.info("Qwen inference START | element_id=%s", element_id)
            response = model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a visual inspection system.\n"
                            "Report only information directly visible in the image.\n"
                            "Never guess, infer hidden context, or invent details.\n"
                            "If uncertain, omit the information."
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
            logger.info("Qwen inference END | element_id=%s | response_type=%s", element_id, type(response))
            payload = self._response_payload(response)
        except Exception as exc:
            logger.warning("Error during Qwen vision inference | element_id=%s: %s", element_id, exc)
            payload = {"error": str(exc)}

        # ===== STEP 7: Log raw response =====
        raw_text = payload.get("_raw_text", "N/A")
        logger.info(
            "Raw Qwen response | element_id=%s | text_preview=%s",
            element_id, raw_text[:200] if raw_text else "(empty)"
        )

        # ===== STEP 8: Attach results to element =====
        element.attributes["visual_analysis"] = payload
        element.attributes["visual_analysis_model"] = self.model_name
        element.attributes["visual_analysis_status"] = "completed"

        logger.info(
            "Qwen vision pipeline END | element_id=%s | visual_type=%s | visible_text_count=%d",
            element_id,
            payload.get("visual_type", "N/A"),
            len(payload.get("visible_text", []))
        )

        # ===== STEP 9: Dynamic type classification =====
        if isinstance(payload, dict):
            vis_type = str(payload.get("visual_type", "")).lower()
            if "chart" in vis_type or "graph" in vis_type or "plot" in vis_type:
                element.type = "chart"
            elif "diagram" in vis_type or "flowchart" in vis_type:
                element.type = "figure"
            elif "meme" in vis_type or "social_media" in vis_type:
                element.attributes["is_informal_graphic"] = True
                element.attributes["overlay_text"] = (
                    payload.get("overlay_text")
                    or payload.get("visible_text")
                    or []
                )
                element.attributes["core_concept"] = payload.get("core_concept_or_humor_theme")

    @classmethod
    def _resolve_image_path(cls, saved_path: Optional[str]) -> Optional[Path]:
        """Resolve saved_image_path to an existing file in extracted storage or filesystem."""
        if not saved_path:
            return None
        candidate = Path(saved_path)
        if candidate.is_file():
            return candidate
        base_candidate = settings.BASE_DIR / candidate
        if base_candidate.is_file():
            return base_candidate
        return None

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

    @classmethod
    def _has_image_source(cls, element: RawDocumentElement) -> bool:
        saved_path = element.attributes.get("saved_image_path")
        if cls._resolve_image_path(saved_path) is not None:
            return True
        return element.image is not None

    @classmethod
    def _image_source(cls, element: RawDocumentElement) -> Optional[Image.Image]:
        """Load image from extracted storage path first, falling back to in-memory crop."""
        saved_path = element.attributes.get("saved_image_path")
        image_path = cls._resolve_image_path(saved_path)
        if image_path is not None:
            with Image.open(image_path) as image:
                return image.copy()
        if element.image is not None:
            return element.image
        return None

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
            vis_text = parsed_json.get("visible_text")
            if isinstance(vis_text, str):
                parsed_json["visible_text"] = [vis_text] if vis_text.strip() else []
            elif not isinstance(vis_text, list):
                parsed_json["visible_text"] = []

            key_details = parsed_json.get("key_details")
            if isinstance(key_details, dict):
                parsed_json["key_details"] = [f"{k}: {v}" for k, v in key_details.items()]
            elif isinstance(key_details, str):
                parsed_json["key_details"] = [key_details] if key_details.strip() else []
            elif not isinstance(key_details, list):
                parsed_json["key_details"] = []

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
