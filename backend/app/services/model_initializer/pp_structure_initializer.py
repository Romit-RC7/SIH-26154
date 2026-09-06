"""Lazy offline initializer for the PP-StructureV3 layout/OCR stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger


class PPStructureInitializer:
    """Loads and unloads the complete local PP-StructureV3 stage."""

    def __init__(self, models_root: Optional[Path] = None):
        self.models_root = models_root or settings.PP_STRUCTURE_MODEL_DIR
        self.engine: Optional[Any] = None

    @property
    def model_dirs(self) -> dict[str, Path]:
        return {
            "layout": self.models_root / "layout",
            "table": self.models_root / "table",
            "det": self.models_root / "det",
            "rec": self.models_root / "rec",
            "table_cls": self.models_root / "table_cls",
            "wired_table_cells": self.models_root / "wired_table_cells",
            "wireless_table_cells": self.models_root / "wireless_table_cells",
            "chart": self.models_root / "chart",
            "doc_ori": self.models_root / "doc_ori",
            "textline_ori": self.models_root / "textline_ori",
        }

    def is_available(self) -> bool:
        if settings.DOC_ANALYZER_ENGINE != "pp_structure":
            return False
        try:
            import paddleocr  # noqa: F401
        except Exception:
            return False
        required = ("layout", "table", "det", "rec", "table_cls", "wired_table_cells", "wireless_table_cells")
        return all(
            (model_dir := self.model_dirs[name]).is_dir()
            and (model_dir / "inference.yml").exists()
            for name in required
        )

    def load(self) -> Any:
        if self.engine is not None:
            return self.engine
        if not self.is_available():
            raise RuntimeError("Required local PP-Structure model packages are unavailable")

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLEX_NO_DOWNLOAD", "True")
        from paddleocr import PPStructureV3

        device = "cpu"
        if settings.USE_GPU:
            try:
                import paddle
                if paddle.is_compiled_with_cuda():
                    device = "gpu:0"
            except Exception:
                pass

        dirs = self.model_dirs
        kwargs = {
            "device": device,
            "use_table_recognition": True,
            "use_chart_recognition": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "use_formula_recognition": False,
            "use_seal_recognition": False,
            "use_region_detection": False,
            "lang": "en",
            "layout_detection_model_name": "PP-DocLayout-L",
            "layout_detection_model_dir": str(dirs["layout"]),
            "wired_table_structure_recognition_model_name": "SLANet_plus",
            "wired_table_structure_recognition_model_dir": str(dirs["table"]),
            "wireless_table_structure_recognition_model_name": "SLANet_plus",
            "wireless_table_structure_recognition_model_dir": str(dirs["table"]),
            "table_classification_model_name": "PP-LCNet_x1_0_table_cls",
            "table_classification_model_dir": str(dirs["table_cls"]),
            "wired_table_cells_detection_model_name": "RT-DETR-L_wired_table_cell_det",
            "wired_table_cells_detection_model_dir": str(dirs["wired_table_cells"]),
            "wireless_table_cells_detection_model_name": "RT-DETR-L_wireless_table_cell_det",
            "wireless_table_cells_detection_model_dir": str(dirs["wireless_table_cells"]),
            "text_detection_model_name": "PP-OCRv4_server_det",
            "text_detection_model_dir": str(dirs["det"]),
            "text_recognition_model_name": "PP-OCRv4_server_rec",
            "text_recognition_model_dir": str(dirs["rec"]),
        }

        chart_dir = dirs["chart"]
        chart_ready = (
            (chart_dir / "inference.pdiparams").exists()
            or (chart_dir / "model_state.pdparams").exists()
        ) and (chart_dir / "inference.yml").exists()
        if chart_ready:
            try:
                from paddle.incubate.nn.functional import fused_rms_norm_ext  # noqa: F401
                kwargs.update(
                    use_chart_recognition=True,
                    chart_recognition_model_name="PP-Chart2Table",
                    chart_recognition_model_dir=str(chart_dir),
                )
            except ImportError:
                logger.warning(
                    "PP-Structure chart recognition disabled: Paddle runtime lacks fused_rms_norm_ext"
                )
        else:
            logger.warning("PP-Structure chart model is not complete: %s", chart_dir)

        if (dirs["doc_ori"] / "inference.yml").exists():
            kwargs.update(
                doc_orientation_model_name="PP-LCNet_x1_0_doc_ori",
                doc_orientation_model_dir=str(dirs["doc_ori"]),
                doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
                doc_orientation_classify_model_dir=str(dirs["doc_ori"]),
            )
        if (dirs["textline_ori"] / "inference.yml").exists():
            kwargs.update(
                textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
                textline_orientation_model_dir=str(dirs["textline_ori"]),
            )

        logger.info("Loading offline PP-StructureV3 models from %s", self.models_root)
        self.engine = PPStructureV3(**kwargs)
        return self.engine

    def unload(self) -> None:
        if self.engine is None:
            return
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()
        self.engine = None
        logger.info("Unloaded offline PP-StructureV3 models")


pp_structure_initializer = PPStructureInitializer()

__all__ = ["PPStructureInitializer", "pp_structure_initializer"]
