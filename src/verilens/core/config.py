"""
Configuration settings for VeriLens AI.
"""

from typing import Set

class Settings:
    PROJECT_NAME: str = "VeriLens AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Upload limits
    MAX_MEDIA_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 Megabytes
    MAX_TEXT_CHARS: int = 100_000
    
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
    ]

settings = Settings()
