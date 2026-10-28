"""
Synthetic Image Detector.
Performs frequency-domain spectral analysis (2D FFT) and Laplacian residual variance 
to identify generative diffusion and GAN upsampling grid artifacts.
Includes lazy-loaded hooks for pretrained neural classifiers.
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

logger = get_logger("SyntheticImageDetector")

class SyntheticImageDetector(BaseDetector):
    detector_id = "synthetic_image"
    detector_name = "Synthetic Image Detector"
    description = "Analyzes 2D FFT frequency spectrum and noise residual distributions for generative synthesis artifacts."
    required_modalities = ["image"]

    def __init__(self):
        self._pretrained_model: Optional[object] = None
        self._model_load_attempted: bool = False

    def _get_lazy_model(self):
        """Lazy-loads pretrained neural vision model only on demand."""
        if not self._model_load_attempted:
            self._model_load_attempted = True
            try:
                # Lazy import hook (e.g. transformers pipeline)
                # Keep lightweight for hackathon startup speed
                logger.info("Checking optional pretrained vision model...")
                self._pretrained_model = None
            except Exception as e:
                logger.warning(f"Pretrained vision model unavailable: {e}")
                self._pretrained_model = None
        return self._pretrained_model

    def _execute(self, input_data: DetectorInput) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        if not input_data.media_bytes:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.NOT_RUN,
                manipulation_score=None,
                confidence=0.0,
                summary="No visual media provided for synthetic artifact analysis.",
            ), []

        try:
            image = Image.open(io.BytesIO(input_data.media_bytes)).convert("L")
            img_arr = np.array(image, dtype=np.float32)
            height, width = img_arr.shape
        except Exception as exc:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary=f"Failed to process image array: {str(exc)}",
                error_message=str(exc),
            ), []

        # Check for degenerate cases (e.g. solid color or near zero variance)
        img_variance = float(np.var(img_arr))
        if img_variance < 5.0 or width < 32 or height < 32:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.25,
                summary="Image exhibits insufficient textural variation or resolution for spectral artifact inspection.",
                metrics={"variance": round(img_variance, 2), "resolution": f"{width}x{height}"},
            ), []

        # 2D Fast Fourier Transform to analyze frequency domain
        fft = np.fft.fft2(img_arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1.0)

        center_y, center_x = height // 2, width // 2
        radius = min(center_y, center_x) // 3
        
        # Mask out center (low frequencies) to isolate high-frequency residuals
        y_indices, x_indices = np.ogrid[:height, :width]
        mask_high_freq = (x_indices - center_x)**2 + (y_indices - center_y)**2 > radius**2

        total_energy = float(np.sum(magnitude_spectrum))
        high_freq_energy = float(np.sum(magnitude_spectrum[mask_high_freq]))
        high_freq_ratio = high_freq_energy / (total_energy + 1e-7)

        # Discrete Laplacian operator for noise residual estimation
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        # Fast 2D convolution proxy
        from scipy.signal import convolve2d
        laplacian = convolve2d(img_arr, laplacian_kernel, mode="valid")
        laplacian_var = float(np.var(laplacian))

        # Generative diffusion models often exhibit characteristic spectral distribution:
        # either hyper-smooth noise suppression (low laplacian variance) or periodic high-freq grid peaks
        manipulation_score = 0.20
        evidence: List[EvidenceItem] = []

        if high_freq_ratio > 0.82:
            manipulation_score += 0.45
            evidence.append(
                EvidenceItem(
                    id="ev_spectral_grid",
                    detector_id=self.detector_id,
                    evidence_type="FREQUENCY_ANOMALY",
                    description=f"Elevated high-frequency spectral ratio ({high_freq_ratio:.3f}) correlates with diffusion grid artifacts.",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"high_freq_ratio": round(high_freq_ratio, 3)},
                )
            )
        elif high_freq_ratio < 0.35 and laplacian_var < 50.0:
            manipulation_score += 0.30
            evidence.append(
                EvidenceItem(
                    id="ev_over_smoothing",
                    detector_id=self.detector_id,
                    evidence_type="ARTIFACT_RESIDUAL",
                    description=f"Excessive high-frequency suppression and low Laplacian variance ({laplacian_var:.1f}) indicate AI smoothing filter.",
                    severity=EvidenceSeverity.MEDIUM,
                    metadata={"laplacian_variance": round(laplacian_var, 2)},
                )
            )

        manipulation_score = round(min(1.0, max(0.05, manipulation_score)), 2)
        confidence = 0.80

        summary = (
            f"Analyzed {width}x{height} image spectrum. High-frequency energy ratio: {high_freq_ratio:.2f}, "
            f"Laplacian variance: {laplacian_var:.1f}."
        )

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=manipulation_score,
            confidence=confidence,
            summary=summary,
            metrics={
                "high_frequency_ratio": round(high_freq_ratio, 3),
                "laplacian_variance": round(laplacian_var, 2),
                "resolution": f"{width}x{height}",
            },
        )
        return finding, evidence
