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
            f"Inspect {subject} using ONLY the pixels visible in the provided image.\n\n"
            "STRICT VISUAL GROUNDING RULES:\n"
            "1. First determine what is actually visible in this specific image. "
            "Do not guess what the image might represent from context or prior knowledge.\n"
            "2. Identify the visual type conservatively. Choose the simplest accurate category "
            "such as: person, natural_scenery, photograph, illustration, screenshot, diagram, "
            "flowchart, technical_drawing, scientific_figure, chart, document, map, meme, "
            "social_media_graphic, or other.\n"
            "3. Describe ONLY objects, shapes, structures, people, text, and relationships that "
            "are directly visible. Never invent details to make the description more complete.\n"
            "4. If an object or detail is unclear, cropped, too small, or not visibly supported, "
            "DO NOT identify or guess it.\n"
            "5. Do not infer hidden context, location, purpose, identity, profession, software, "
            "technology, materials, events, or causes unless they are explicitly visible.\n"
            "6. Do not use generic descriptions learned from similar images. The description "
            "must refer to THIS exact image.\n"
            "7. For visible_text, include ONLY text that can actually be read in the image. "
            "Never invent text. If no text is clearly readable, return an empty list.\n"
            "8. For key_details, include only concrete visual characteristics that can be "
            "verified directly from the image.\n"
            "9. If the image does not contain enough information to answer something, leave it "
            "out rather than guessing.\n"
            "10. Before answering, internally check every claim: "
            "\"Can I point to visible evidence for this claim in this image?\" "
            "If not, remove the claim.\n"
            "11. If the image is a meme, comic, or social media post: identify visual_type as 'meme' "
            "or 'social_media_graphic'. Extract all overlay caption text into overlay_text. "
            "In core_concept_or_humor_theme, describe the underlying subject, problem, or message "
            "(e.g. 'Production outage during Friday deploy', 'Legacy code refactoring friction'). "
            "For standard documents/diagrams, set overlay_text to [] and core_concept_or_humor_theme to null.\n\n"
            f"{ocr_evidence}\n"
            "Return ONLY a valid JSON object. Do not include markdown, explanations, or text "
            "outside the JSON.\n\n"
            "Schema:\n"
            "{\n"
            '  "visual_type": "<single conservative category>",\n'
            '  "description": "<short objective description containing only directly visible information>",\n'
            '  "visible_text": ["<only clearly readable text>"],\n'
            '  "overlay_text": ["<extracted meme/graphic overlay text phrases>"],\n'
            '  "core_concept_or_humor_theme": "<underlying subject or concept if meme/graphic, else null>",\n'
            '  "key_details": ["<only directly observable visual details>"]\n'
            "}"
        )
        
        try:
            response = model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative visual inspection system. "
                            "Your job is to report only evidence directly visible in the supplied image. "
                            "Never guess, infer, complete, or embellish missing visual information. "
                            "When uncertain, omit the claim rather than guessing. "
                            "Do not rely on generic descriptions of similar images. "
                            "Every statement in the output must be supported by visible pixels in THIS image."
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
            elif "meme" in vis_type or "social_media" in vis_type:
                element.attributes["is_informal_graphic"] = True
                element.attributes["overlay_text"] = (
                    payload.get("overlay_text")
                    or payload.get("visible_text")
                    or []
                )
                element.attributes["core_concept"] = payload.get("core_concept_or_humor_theme")

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
