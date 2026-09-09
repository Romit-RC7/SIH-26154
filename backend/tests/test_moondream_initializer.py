"""Unit tests for Moondream2 model initializer and VLM engine toggle."""

from pathlib import Path
import pytest
from backend.app.core.config import settings
from backend.app.services.model_initializer.moondream_initializer import MoondreamInitializer, moondream_initializer
from backend.app.services.recognition.image_service import ImageRecognitionService, image_recognition_service
from backend.app.processors.base import RawDocumentElement


def test_moondream_initializer_paths():
    init = MoondreamInitializer(model_dir=Path("/nonexistent/moondream2"))
    assert init.is_available() is False


def test_vlm_engine_config_toggle(monkeypatch):
    monkeypatch.setattr(settings, "VLM_ENGINE", "moondream2")
    assert getattr(settings, "VLM_ENGINE", "").lower() == "moondream2"

    monkeypatch.setattr(settings, "VLM_ENGINE", "qwen2.5_vl")
    assert getattr(settings, "VLM_ENGINE", "").lower() == "qwen2.5_vl"


def test_image_recognition_service_vlm_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "VLM_ENGINE", "moondream2")
    
    elem = RawDocumentElement(
        type="image",
        page=1,
        bbox=[0, 0, 100, 100],
        confidence=1.0,
        attributes={"saved_image_path": "nonexistent_image.png"}
    )
    
    # Run recognition: should gracefully handle missing model weights or image
    image_recognition_service.recognize([elem])
    assert "visual_analysis_status" in elem.attributes
