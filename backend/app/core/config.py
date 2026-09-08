"""
Application Configuration Module
Defines settings loaded from environment variables using Pydantic Settings.
"""

from pathlib import Path
from typing import List, Optional, ClassVar, Set

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project Info
    PROJECT_NAME: str = "SIH-26154 Semantic Document Processing System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "sih_content_platform"
    DATABASE_URL: Optional[str] = None

    @property
    def sync_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL.replace(
                "postgresql+asyncpg://",
                "postgresql://",
            )

        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def async_database_url(self) -> str:
        if self.DATABASE_URL:
            if not self.DATABASE_URL.startswith("postgresql+asyncpg://"):
                return self.DATABASE_URL.replace(
                    "postgresql://",
                    "postgresql+asyncpg://",
                )
            return self.DATABASE_URL

        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # File Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[3]
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    RAW_UPLOAD_DIR: Path = UPLOAD_DIR / "raw"
    EXTRACTED_UPLOAD_DIR: Path = UPLOAD_DIR / "extracted"

    # Model Storage Paths
    MODELS_DIR: Path = BASE_DIR / "models"
    PP_STRUCTURE_MODEL_DIR: Path = MODELS_DIR / "pp_structure_v3"
    FASTER_WHISPER_MODEL_DIR: Path = MODELS_DIR / "faster_whisper_small"

    # Processing & OCR Configuration
    # Options: 'pp_structure' (production PaddleOCR) or 'rule_based' (fast PyMuPDF/fallback)
    DOC_ANALYZER_ENGINE: str = "pp_structure"
    ENABLE_OCR: bool = True
    # PDF pages are rasterized once for PP-Structure and any visual crops passed
    # to Qwen. 120 DPI keeps those shared images within the intended 100-120
    # DPI range while retaining enough detail for ordinary digital documents.
    PDF_RENDER_DPI: int = 120
    VIDEO_FRAME_SAMPLE_RATE_PER_MINUTE: int = 6
    VIDEO_FRAME_MAX_WIDTH: int = 1280
    VIDEO_MAX_DURATION_SECONDS: int = 120
    VIDEO_FRAME_DIFF_THRESHOLD: float = 0.03  # 3% average pixel difference threshold
    VIDEO_MAX_KEYFRAMES: int = 20              # Maximum keyframes to pass to visual recognition
    VIDEO_CANDIDATE_FPS: float = 0.333         # Sample candidate frames every ~3 seconds (fps=1/3)
    VISUAL_MIN_DIMENSION_PX: int = 48          # Minimum width/height in px for VLM visual reasoning
    VISUAL_MIN_AREA_PX: int = 2304             # Minimum area (48x48) to avoid processing emojis/micro-icons
    VISUAL_DHASH_THRESHOLD: int = 4            # Hamming distance threshold for dHash perceptual image similarity
    VISUAL_RECURRING_FREQ_THRESHOLD: float = 0.30  # Frequency threshold to flag repeating slide/template logos
    FASTER_WHISPER_DEVICE: str = "cuda"
    FASTER_WHISPER_COMPUTE_TYPE: str = "float16"
    USE_GPU: bool = True
    N_GPU_LAYERS: int = -1  # -1 offloads all layers to GPU in llama-cpp-python
    RECOGNITION_BATCH_SIZE: int = 4
    RECOGNITION_BATCH_WAIT_SECONDS: float = 0.5
    MAX_UPLOAD_SIZE_MB: int = 100

    ALLOWED_EXTENSIONS: ClassVar[Set[str]] = {
        ".pdf",
        ".docx",
        ".pptx",
        ".txt",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".webp",
        ".mp4",
        ".mov",
        ".webm",
    }

    # LLM & Vision Model Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:latest"
    QWEN_VISION_N_CTX: int = 4096

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]


settings = Settings()

# Ensure directories exist
settings.RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.EXTRACTED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
