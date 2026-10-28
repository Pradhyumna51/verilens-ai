"""
Top-level synthetic_text detector module alias.
"""

from verilens.detectors.synthetic_text import (
    SyntheticTextDetector,
    PretrainedTextClassifier,
    CalibratedTextFeatureClassifier,
)

__all__ = [
    "SyntheticTextDetector",
    "PretrainedTextClassifier",
    "CalibratedTextFeatureClassifier",
]
