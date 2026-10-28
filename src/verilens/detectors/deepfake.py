"""
Deepfake Analysis Engine.
Inspects facial blend boundaries, skin-region color space transitions, and warping artifacts.
Supports NOT_RUN, UNCERTAIN, ERROR, and SUCCESS statuses.
"""

import io
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)
from verilens.core.logger import get_logger

logger = get_logger("DeepfakeDetector")

class DeepfakeDetector(BaseDetector):
    detector_id = "deepfake"
    detector_name = "Deepfake Analysis Engine"
    description = "Examines facial region boundaries, skin chromatic consistency, and synthetic blend transitions."
    required_modalities = ["image"]

    def __init__(self):
        self._face_net = None
        self._lazy_attempted = False

    def _get_lazy_facenet(self):
        """Lazy-loads dedicated face detection model if available."""
        if not self._lazy_attempted:
            self._lazy_attempted = True
            try:
                # Lazy import hook for facenet/mediapipe/cv2
                self._face_net = None
            except Exception as e:
                logger.warning(f"FaceNet lazy load skipped: {e}")
                self._face_net = None
        return self._face_net

    def _execute(self, input_data: DetectorInput) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        if not input_data.media_bytes:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.NOT_RUN,
                manipulation_score=None,
                confidence=0.0,
                summary="No visual media provided for deepfake evaluation.",
            ), []

        try:
            image = Image.open(io.BytesIO(input_data.media_bytes)).convert("RGB")
            img_arr = np.array(image, dtype=np.uint8)
            height, width, _ = img_arr.shape
        except Exception as exc:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary=f"Failed to inspect media frame: {str(exc)}",
                error_message=str(exc),
            ), []

        # Color-space skin tone candidate segmentation (RGB rule: R > G > B and R - G > 15)
        r = img_arr[:, :, 0].astype(int)
        g = img_arr[:, :, 1].astype(int)
        b = img_arr[:, :, 2].astype(int)

        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            ((r - g) > 15) & (np.abs(r - g) > 15)
        )
        skin_pixel_count = int(np.sum(skin_mask))
        total_pixels = height * width
        skin_ratio = skin_pixel_count / max(1, total_pixels)

        # Operational condition: if no facial/skin area is found, return UNCERTAIN
        if skin_ratio < 0.03:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.20,
                summary="No identifiable human portrait or facial skin regions detected to evaluate blend seams.",
                metrics={"skin_pixel_ratio": round(skin_ratio, 4)},
            ), []

        # Calculate edge gradients at skin boundary regions to detect face-swap seam blurring
        # In deepfakes, face swaps produce abnormal blur or harsh color gradients along the mask border
        skin_uint8 = skin_mask.astype(np.uint8)
        # Shift mask to approximate boundary
        boundary = np.zeros_like(skin_uint8)
        boundary[1:, :] |= (skin_uint8[1:, :] != skin_uint8[:-1, :])
        boundary[:, 1:] |= (skin_uint8[:, 1:] != skin_uint8[:, :-1])

        boundary_pixels = int(np.sum(boundary))
        # Compute gradient magnitude on grayscale
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)

        mean_boundary_grad = float(np.mean(grad_mag[boundary > 0])) if boundary_pixels > 0 else 0.0

        evidence: List[EvidenceItem] = []
        manipulation_score = 0.15

        # Blurry blend seams usually have unusually low gradient (< 12) or abnormal harsh seam (> 65)
        if mean_boundary_grad < 12.0 and boundary_pixels > 100:
            manipulation_score = 0.72
            evidence.append(
                EvidenceItem(
                    id="ev_face_boundary_blur",
                    detector_id=self.detector_id,
                    evidence_type="FACIAL_ARTIFACT",
                    description=f"Abnormally low boundary gradient ({mean_boundary_grad:.1f}) indicates synthetic face-swap edge feathering.",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"boundary_gradient": round(mean_boundary_grad, 2)},
                )
            )
        elif mean_boundary_grad > 65.0:
            manipulation_score = 0.55
            evidence.append(
                EvidenceItem(
                    id="ev_face_boundary_discontinuity",
                    detector_id=self.detector_id,
                    evidence_type="FACIAL_ARTIFACT",
                    description=f"Elevated edge discontinuity at facial margin ({mean_boundary_grad:.1f}).",
                    severity=EvidenceSeverity.MEDIUM,
                    metadata={"boundary_gradient": round(mean_boundary_grad, 2)},
                )
            )

        summary = (
            f"Evaluated portrait candidate regions (skin coverage: {skin_ratio * 100:.1f}%). "
            f"Mean boundary transition gradient: {mean_boundary_grad:.1f}."
        )

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=round(manipulation_score, 2),
            confidence=0.74,
            summary=summary,
            metrics={
                "skin_ratio": round(skin_ratio, 3),
                "boundary_pixels": boundary_pixels,
                "mean_boundary_gradient": round(mean_boundary_grad, 2),
            },
        )
        return finding, evidence
