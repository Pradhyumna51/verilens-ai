import os
from pathlib import Path
from typing import Set, Optional

# Load .env file if present in workspace root
_env_file = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if _env_file.exists():
    try:
        with open(_env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass


class Settings:
    PROJECT_NAME: str = "VeriLens AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Upload limits
    MAX_MEDIA_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 Megabytes
    MAX_TEXT_CHARS: int = 100_000

    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta/models"

    # Optional Google Fact Check Tools API (100% Free)
    FACT_CHECK_API_KEY: Optional[str] = os.getenv("FACT_CHECK_API_KEY", None)
    FACT_CHECK_API_BASE: str = "https://factchecktools.googleapis.com/v1alpha1"
    
    # Perceptual hash cache
    ENABLE_REUSE_CACHE: bool = True
    MAX_CACHE_ENTRIES: int = 500
    
    # Allowed MIME types
    ALLOWED_IMAGE_TYPES: Set[str] = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/gif",
    }
    
    ALLOWED_VIDEO_TYPES: Set[str] = {
        "video/mp4",
        "video/quicktime",
        "video/webm",
        "video/x-msvideo",
    }
    
    @property
    def ALLOWED_MEDIA_TYPES(self) -> Set[str]:
        return self.ALLOWED_IMAGE_TYPES | self.ALLOWED_VIDEO_TYPES
    
    # Registered default detectors
    DEFAULT_DETECTORS = [
        "consistency",
        "media_reuse",
        "synthetic_image",
        "synthetic_text",
        "deepfake",
        "ocr_detector",
    ]

settings = Settings()
