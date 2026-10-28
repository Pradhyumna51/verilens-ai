"""
Base detector interface and typed input/output specifications.
"""

from abc import ABC, abstractmethod
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)
from verilens.core.logger import get_logger

logger = get_logger("BaseDetector")

class DetectorInput(BaseModel):
    """Encapsulates all incoming multimodal content."""
    text: Optional[str] = None
    media_bytes: Optional[bytes] = None
    media_filename: Optional[str] = None
    media_content_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())
        
    @property
    def has_image(self) -> bool:
        if not self.media_bytes:
            return False
        ct = (self.media_content_type or "").lower()
        fn = (self.media_filename or "").lower()
        return ct.startswith("image/") or any(fn.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"])

    @property
    def has_video(self) -> bool:
        if not self.media_bytes:
            return False
        ct = (self.media_content_type or "").lower()
        fn = (self.media_filename or "").lower()
        return ct.startswith("video/") or any(fn.endswith(ext) for ext in [".mp4", ".mov", ".webm", ".avi"])

class BaseDetector(ABC):
    """
    Abstract Base Class for all VeriLens multimodal detectors.
    Ensures support for NOT_RUN, UNCERTAIN, ERROR, and SUCCESS status codes.
    """
    detector_id: str
    detector_name: str
    description: str
    required_modalities: List[str]  # e.g. ["text"], ["image"], ["text", "image"]

    def can_run(self, input_data: DetectorInput) -> bool:
        """Determines if the minimum necessary modalities are present."""
        for mod in self.required_modalities:
            if mod == "text" and not input_data.has_text:
                return False
            if mod == "image" and not (input_data.has_image or input_data.has_video):
                return False
            if mod == "video" and not input_data.has_video:
                return False
        return True

    def run(self, input_data: DetectorInput) -> tuple[DetectorFinding, List[EvidenceItem]]:
        """
        Orchestrates detector execution with guaranteed status handling:
        NOT_RUN, UNCERTAIN, ERROR, or SUCCESS.
        """
        start_time = time.perf_counter()
        
        # Check applicability
        if not self.can_run(input_data):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            finding = DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.NOT_RUN,
                manipulation_score=None,
                confidence=0.0,
                summary=f"Skipped: Required modalities ({', '.join(self.required_modalities)}) not provided.",
                metrics={},
                execution_time_ms=round(elapsed_ms, 2),
            )
            return finding, []

        try:
            finding, evidence = self._execute(input_data)
            finding.execution_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return finding, evidence
        except Exception as exc:
            logger.exception(f"Detector [{self.detector_id}] failed with error: {exc}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            error_finding = DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary=f"Detector failed during execution: {str(exc)}",
                metrics={},
                execution_time_ms=round(elapsed_ms, 2),
                error_message=str(exc),
            )
            return error_finding, []

    @abstractmethod
    def _execute(self, input_data: DetectorInput) -> tuple[DetectorFinding, List[EvidenceItem]]:
        """Concrete execution logic to be implemented by each detector."""
        pass
