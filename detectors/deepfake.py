"""
Top-level deepfake detector module alias.
"""

from verilens.detectors.deepfake import (
    DeepfakeDetector,
    PretrainedDeepfakeClassifier,
    CalibratedFaceArtifactClassifier,
    extract_bounded_video_frames,
)

__all__ = [
    "DeepfakeDetector",
    "PretrainedDeepfakeClassifier",
    "CalibratedFaceArtifactClassifier",
    "extract_bounded_video_frames",
]
