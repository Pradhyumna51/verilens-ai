"""
Detector registry for discovering, configuring, and executing analysis modules.
"""

from typing import Dict, List, Optional
from verilens.detectors.base import BaseDetector
from verilens.detectors.consistency import ConsistencyDetector
from verilens.detectors.media_reuse import MediaReuseDetector
from verilens.detectors.synthetic_image import SyntheticImageDetector
from verilens.detectors.synthetic_text import SyntheticTextDetector
from verilens.detectors.deepfake import DeepfakeDetector
from verilens.schemas.response import DetectorMeta

class DetectorRegistry:
    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Instantiate and register standard detectors."""
        for detector_cls in [
            ConsistencyDetector,
            MediaReuseDetector,
            SyntheticImageDetector,
            SyntheticTextDetector,
            DeepfakeDetector,
        ]:
            instance = detector_cls()
            self._detectors[instance.detector_id] = instance

    def get(self, detector_id: str) -> Optional[BaseDetector]:
        return self._detectors.get(detector_id)

    def list_all(self) -> List[BaseDetector]:
        return list(self._detectors.values())

    def list_metadata(self) -> List[DetectorMeta]:
        return [
            DetectorMeta(
                id=d.detector_id,
                name=d.detector_name,
                description=d.description,
                required_modalities=d.required_modalities,
            )
            for d in self._detectors.values()
        ]

# Global singleton registry
registry = DetectorRegistry()
