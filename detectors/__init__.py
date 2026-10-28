"""
Top-level detectors package alias forwarding to verilens.detectors.
"""

from verilens.detectors.reuse import (
    MediaReuseDetector,
    ImageReuseDetector,
    ReuseDetector,
    ImageReuseIndex,
    ReuseMatch,
    compute_phash,
    compute_dhash,
    compute_ahash,
    hamming_distance,
    default_reuse_index,
)
from verilens.detectors.synthetic_image import (
    SyntheticImageDetector,
    PretrainedImageClassifier,
    CalibratedFeatureClassifier,
)
from verilens.detectors.synthetic_text import (
    SyntheticTextDetector,
    PretrainedTextClassifier,
    CalibratedTextFeatureClassifier,
)
from verilens.detectors.deepfake import (
    DeepfakeDetector,
    PretrainedDeepfakeClassifier,
    CalibratedFaceArtifactClassifier,
    extract_bounded_video_frames,
)

__all__ = [
    "MediaReuseDetector",
    "ImageReuseDetector",
    "ReuseDetector",
    "ImageReuseIndex",
    "ReuseMatch",
    "compute_phash",
    "compute_dhash",
    "compute_ahash",
    "hamming_distance",
    "default_reuse_index",
    "SyntheticImageDetector",
    "PretrainedImageClassifier",
    "CalibratedFeatureClassifier",
    "SyntheticTextDetector",
    "PretrainedTextClassifier",
    "CalibratedTextFeatureClassifier",
    "DeepfakeDetector",
    "PretrainedDeepfakeClassifier",
    "CalibratedFaceArtifactClassifier",
    "extract_bounded_video_frames",
]
