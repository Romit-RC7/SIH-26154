"""
Visual Asset Deduplicator & Decorative Noise Filter Service.
Detects emojis, micro-icons, and recurring master slide/template logos across multi-page documents
using perceptual hashing (dHash) and spatial frequency analysis.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.processors.base import RawDocumentElement


class VisualDeduplicator:
    """
    Filters micro-icons/emojis and deduplicates recurring master slide logos across documents.
    """

    @staticmethod
    def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
        """
        Computes 64-bit difference hash (dHash) for perceptual image similarity matching.
        """
        img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(img, dtype=np.int32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return "".join(["1" if b else "0" for b in diff.flatten()])

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Calculates bitwise Hamming distance between two binary hash strings.
        """
        if len(hash1) != len(hash2):
            return 64
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    @staticmethod
    def is_decorative_noise(image: Image.Image) -> Tuple[bool, str]:
        """
        Checks if an image is a micro-icon, emoji, bullet point, or thin line separator.
        """
        width, height = image.size
        area = width * height
        min_dim = getattr(settings, "VISUAL_MIN_DIMENSION_PX", 48)
        min_area = getattr(settings, "VISUAL_MIN_AREA_PX", 2304)

        if width < min_dim or height < min_dim:
            return True, "micro_icon_or_emoji"
        if area < min_area:
            return True, "small_decorative_asset"

        aspect_ratio = float(width) / float(height) if height > 0 else 1.0
        if aspect_ratio > 10.0 or aspect_ratio < 0.1:
            return True, "thin_line_separator"

        return False, "valid_visual"

    def process_elements(
        self,
        elements: List[RawDocumentElement],
        page_count: int = 1,
    ) -> List[RawDocumentElement]:
        """
        Deduplicates visual elements and flags emojis/icons and recurring template assets across document pages.
        """
        visual_targets = [
            elem for elem in elements
            if elem.type in ("image", "figure", "chart")
        ]
        if not visual_targets:
            return elements

        # 1. Filter micro-icons / emojis & compute hashes
        primary_registry: List[Dict[str, Any]] = []
        # Group tracker: primary_id -> list of all element instances in group
        group_elements: Dict[str, List[RawDocumentElement]] = {}

        for elem in visual_targets:
            img = self._resolve_image(elem)
            if img is None:
                continue

            is_noise, noise_type = self.is_decorative_noise(img)
            if is_noise:
                elem.attributes["is_decorative_noise"] = True
                elem.attributes["noise_category"] = noise_type
                elem.attributes["is_duplicate"] = False
                elem.attributes["is_primary_visual"] = False
                logger.debug("Flagged decorative noise (%s) for element %s on page %s", noise_type, elem.type, elem.page)
                continue

            # Valid visual asset: compute hashes
            elem_dhash = self.compute_dhash(img)
            elem_sha256 = self._compute_sha256(img)

            # Match against existing primary registry
            matched_primary_id: Optional[str] = None
            dhash_threshold = getattr(settings, "VISUAL_DHASH_THRESHOLD", 4)

            for reg in primary_registry:
                # Exact byte match or perceptual dHash within Hamming threshold
                if reg["sha256"] == elem_sha256 or self.hamming_distance(reg["dhash"], elem_dhash) <= dhash_threshold:
                    matched_primary_id = reg["id"]
                    break

            elem_id = getattr(elem, "id", None) or f"elem_p{elem.page}_{id(elem)}"

            if matched_primary_id:
                # Mark as duplicate
                elem.attributes["is_duplicate"] = True
                elem.attributes["is_primary_visual"] = False
                elem.attributes["primary_element_id"] = matched_primary_id
                elem.attributes["dhash"] = elem_dhash

                if matched_primary_id in group_elements:
                    group_elements[matched_primary_id].append(elem)
                logger.debug("Element %s on page %d is a duplicate of primary visual %s", elem_id, elem.page, matched_primary_id)
            else:
                # Register new primary visual
                elem.attributes["is_duplicate"] = False
                elem.attributes["is_primary_visual"] = True
                elem.attributes["primary_element_id"] = elem_id
                elem.attributes["dhash"] = elem_dhash

                primary_registry.append({
                    "id": elem_id,
                    "dhash": elem_dhash,
                    "sha256": elem_sha256,
                    "element": elem,
                })
                group_elements[elem_id] = [elem]

        # 2. Check document-wide recurring frequency (e.g. Master Slide / Header Logos)
        freq_threshold = getattr(settings, "VISUAL_RECURRING_FREQ_THRESHOLD", 0.30)

        for primary_id, instance_list in group_elements.items():
            # Count distinct pages where this visual appears
            distinct_pages = len(set(e.page for e in instance_list))
            recurring_ratio = float(distinct_pages) / float(max(1, page_count))

            is_template_asset = recurring_ratio >= freq_threshold and (page_count > 1 or len(instance_list) > 2)

            for instance in instance_list:
                instance.attributes["recurring_count"] = len(instance_list)
                instance.attributes["recurring_pages_ratio"] = round(recurring_ratio, 2)
                instance.attributes["is_recurring_template_asset"] = is_template_asset

            if is_template_asset:
                logger.info(
                    "Visual group %s flagged as recurring template asset (appears on %d/%d pages, ratio=%.2f)",
                    primary_id, distinct_pages, page_count, recurring_ratio
                )

        return elements

    @staticmethod
    def _resolve_image(element: RawDocumentElement) -> Optional[Image.Image]:
        if element.image is not None:
            return element.image
        saved_path = element.attributes.get("saved_image_path")
        if not saved_path:
            return None
        full_path = settings.BASE_DIR / Path(saved_path)
        if not full_path.is_file():
            return None
        try:
            with Image.open(full_path) as img:
                return img.copy()
        except Exception:
            return None

    @staticmethod
    def _compute_sha256(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return hashlib.sha256(buffer.getvalue()).hexdigest()


visual_deduplicator = VisualDeduplicator()

__all__ = ["VisualDeduplicator", "visual_deduplicator"]
