"""
Detectors package for VeriLens AI.
"""

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.detectors.consistency import ConsistencyDetector
from verilens.detectors.media_reuse import MediaReuseDetector
from verilens.detectors.synthetic_image import SyntheticImageDetector
from verilens.detectors.synthetic_text import SyntheticTextDetector
from verilens.detectors.deepfake import DeepfakeDetector
from verilens.detectors.ocr import OCRDetector
from verilens.detectors.registry import DetectorRegistry, registry

__all__ = [
    "BaseDetector",
    "DetectorInput",
    "ConsistencyDetector",
    "MediaReuseDetector",
    "SyntheticImageDetector",
    "SyntheticTextDetector",
    "DeepfakeDetector",
    "OCRDetector",
    "DetectorRegistry",
    "registry",
]
