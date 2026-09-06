# backend/app/services/system_diagnostics.py

from pathlib import Path
from backend.app.core.config import settings
from backend.app.processors.pp_structure import pp_structure_analyzer


class SystemDiagnostics:

    @staticmethod
    def model_exists(path: Path) -> bool:
        return path.exists() and any(path.iterdir())

    @classmethod
    def get_status(cls):

        pp_root = settings.PP_STRUCTURE_MODEL_DIR
        models_root = settings.BASE_DIR / "models"

        return {
            "engines": {
                "pp_structure_initialized": pp_structure_analyzer.is_available(),
                "fallback_available": True,
            },

            "models": {

                "pp_structure": pp_structure_analyzer.is_available(),

                "layout_model":
                    cls.model_exists(pp_root / "layout"),

                "table_model":
                    cls.model_exists(pp_root / "table"),

                "ocr_det_model":
                    cls.model_exists(pp_root / "det"),

                "ocr_rec_model":
                    cls.model_exists(pp_root / "rec"),

                "bge_small_en_v1.5":
                    cls.model_exists(models_root / "bge_small_en_v1.5"),

                "qwen2.5_vl_3b_q4":
                    cls.model_exists(models_root / "qwen2.5_vl_3b_q4"),

                "qwen3_4b_q4":
                    cls.model_exists(models_root / "qwen3_4b_q4"),

                "qwen3_8b_q4":
                    cls.model_exists(models_root / "qwen3_8b_q4"),

                "unichart_base_960":
                    cls.model_exists(models_root / "unichart_base_960"),

                "faster_whisper_small":
                    cls.model_exists(settings.FASTER_WHISPER_MODEL_DIR),
            },

            "storage": {
                "raw_uploads":
                    settings.RAW_UPLOAD_DIR.exists(),

                "extracted_uploads":
                    settings.EXTRACTED_UPLOAD_DIR.exists(),
            }
        }


system_diagnostics = SystemDiagnostics()
