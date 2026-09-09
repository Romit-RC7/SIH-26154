"""Unit tests for Visual Asset Deduplication and Decorative Noise Filtering."""

import pytest
from PIL import Image, ImageDraw
from backend.app.processors.base import RawDocumentElement
from backend.app.services.recognition.image_deduplicator import VisualDeduplicator, visual_deduplicator
from backend.app.services.recognition.image_service import ImageRecognitionService, image_recognition_service


def test_is_decorative_noise_micro_icon():
    tiny_img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    is_noise, category = visual_deduplicator.is_decorative_noise(tiny_img)
    assert is_noise is True
    assert category == "micro_icon_or_emoji"


def test_is_decorative_noise_thin_line():
    line_img = Image.new("RGB", (500, 10), color=(0, 0, 0))
    is_noise, category = visual_deduplicator.is_decorative_noise(line_img)
    assert is_noise is True
    assert category in ("micro_icon_or_emoji", "thin_line_separator")


def test_is_decorative_noise_valid_image():
    valid_img = Image.new("RGB", (300, 200), color=(100, 150, 200))
    is_noise, category = visual_deduplicator.is_decorative_noise(valid_img)
    assert is_noise is False
    assert category == "valid_visual"


def test_compute_dhash_and_similarity():
    img1 = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([20, 20, 100, 100], fill=(0, 0, 0))

    img2 = img1.copy()

    hash1 = visual_deduplicator.compute_dhash(img1)
    hash2 = visual_deduplicator.compute_dhash(img2)

    assert len(hash1) == 64
    assert visual_deduplicator.hamming_distance(hash1, hash2) == 0


def test_process_elements_deduplication_and_recurring_logo():
    logo_img = Image.new("RGB", (200, 100), color=(0, 100, 200))
    draw = ImageDraw.Draw(logo_img)
    draw.text((10, 10), "CORP LOGO", fill=(255, 255, 255))

    # Create 5 elements on 5 distinct pages with the same logo
    elements = [
        RawDocumentElement(
            type="image",
            page=p,
            bbox=[0, 0, 200, 100],
            image=logo_img.copy(),
            confidence=1.0,
            attributes={"element_id": f"elem_logo_p{p}"}
        )
        for p in range(1, 6)
    ]

    processed = visual_deduplicator.process_elements(elements, page_count=5)

    # First instance on Page 1 must be primary
    assert processed[0].attributes["is_primary_visual"] is True
    assert processed[0].attributes["is_duplicate"] is False

    # Remaining instances must be marked as duplicates
    for elem in processed[1:]:
        assert elem.attributes["is_duplicate"] is True
        assert elem.attributes["is_primary_visual"] is False
        assert elem.attributes["primary_element_id"] == "elem_logo_p1"

    # Recurring template logo flag should be set for all 5 instances
    for elem in processed:
        assert elem.attributes["is_recurring_template_asset"] is True
        assert elem.attributes["recurring_pages_ratio"] == 1.0


def test_propagate_duplicate_analysis():
    primary_elem = RawDocumentElement(
        type="image",
        page=1,
        bbox=[0, 0, 200, 100],
        confidence=1.0,
        attributes={
            "element_id": "elem_primary_1",
            "is_primary_visual": True,
            "is_duplicate": False,
            "visual_analysis": {"visual_type": "logo", "summary": "Corporate Brand Logo"},
            "visual_analysis_model": "Qwen2.5-VL-3B",
            "visual_analysis_status": "completed",
        }
    )

    duplicate_elem = RawDocumentElement(
        type="image",
        page=2,
        bbox=[0, 0, 200, 100],
        confidence=1.0,
        attributes={
            "element_id": "elem_dup_2",
            "is_primary_visual": False,
            "is_duplicate": True,
            "primary_element_id": "elem_primary_1",
        }
    )

    elements = [primary_elem, duplicate_elem]
    image_recognition_service._propagate_duplicate_analysis(elements)

    assert "visual_analysis" in duplicate_elem.attributes
    assert duplicate_elem.attributes["visual_analysis"]["summary"] == "Corporate Brand Logo"
    assert duplicate_elem.attributes["visual_analysis_status"] == "completed"
