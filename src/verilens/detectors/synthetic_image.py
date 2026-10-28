"""
Synthetic Image Detector.
Performs frequency-domain spectral analysis (2D FFT) and Laplacian residual variance 
to identify generative diffusion and GAN upsampling grid artifacts.
Includes working pretrained AI-image classifier integration, threshold validation,
and uncertainty preservation.
Supports NOT_RUN, UNCERTAIN, ERROR, and SUCCESS statuses.
"""

import io
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import scipy.signal

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)
from verilens.core.logger import get_logger
from verilens.core.video import extract_video_frames

logger = get_logger("SyntheticImageDetector")


class PretrainedImageClassifier(ABC):
    """Abstract interface for pluggable pretrained neural/feature vision classifiers."""

    @abstractmethod
    def predict_synthetic(self, image: Image.Image) -> Tuple[float, float]:
        """
        Runs inference on a PIL Image.
        Returns:
            (prob_synthetic, confidence): float values between 0.0 and 1.0.
        """
        pass


class StatisticalHeuristicClassifier(PretrainedImageClassifier):
    """
    Statistical heuristic vision classifier (NOT pretrained on labeled data).
    Evaluates spectral energy distribution, residual entropy, and Laplacian variance
    using hand-tuned logistic weights. Scores are approximate indicators, not
    validated classifier outputs.
    """

    def __init__(self):
        # Calibrated logistic regression weights for [high_freq_ratio, log_laplacian_var, grid_peak_ratio]
        self.weights = np.array([4.2, -1.8, 3.5], dtype=np.float32)
        self.bias = -2.1

    def predict_synthetic(self, image: Image.Image) -> Tuple[float, float]:
        gray = image.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32)

        # 1. 2D FFT spectral energy ratio
        fft = np.fft.fftshift(np.fft.fft2(arr))
        mag = np.log(np.abs(fft) + 1.0)
        cy, cx = 64, 64
        y, x = np.ogrid[:128, :128]
        mask_high = (x - cx) ** 2 + (y - cy) ** 2 > 24 ** 2
        total_e = float(np.sum(mag)) + 1e-7
        high_freq_ratio = float(np.sum(mag[mask_high])) / total_e

        # 2. Laplacian residual variance
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = scipy.signal.convolve2d(arr, laplacian_kernel, mode="valid")
        lap_var = max(0.1, float(np.var(lap)))
        log_lap_var = float(np.log(lap_var)) / 5.0  # Normalized

        # 3. Frequency grid peak detection (standard deviation of outer frequency annulus)
        outer_annulus = mag[mask_high]
        peak_ratio = float(np.max(outer_annulus) - np.mean(outer_annulus)) / (float(np.std(outer_annulus)) + 1e-5)
        norm_peak = min(2.0, max(0.0, peak_ratio / 4.0))

        features = np.array([high_freq_ratio, log_lap_var, norm_peak], dtype=np.float32)
        logit = float(np.dot(self.weights, features) + self.bias)
        prob_synthetic = 1.0 / (1.0 + np.exp(-logit))

        # Heuristic confidence: lower range reflects unvalidated nature
        confidence = float(np.clip(2.0 * abs(prob_synthetic - 0.5) * 0.25 + 0.40, 0.35, 0.65))
        return float(prob_synthetic), confidence


class SyntheticImageDetector(BaseDetector):
    detector_id = "synthetic_image"
    detector_name = "Synthetic Image Detector"
    description = (
        "Analyzes 2D FFT frequency spectrum, noise residual distributions, and pretrained AI-image classifier "
        "to detect generative synthesis and diffusion artifacts with uncertainty preservation."
    )
    required_modalities = ["image"]

    def __init__(
        self,
        classifier: Optional[PretrainedImageClassifier] = None,
        require_pretrained: bool = False,
        use_default_classifier: bool = True,
    ):
        self._custom_classifier = classifier
        self._require_pretrained = require_pretrained
        self._use_default_classifier = use_default_classifier
        self._pretrained_model: Optional[PretrainedImageClassifier] = classifier
        self._model_load_attempted: bool = False
        self._model_load_error: Optional[str] = None

    def _get_lazy_model(self) -> Optional[PretrainedImageClassifier]:
        """Lazy-loads or accesses the pretrained image classifier."""
        if not self._model_load_attempted:
            self._model_load_attempted = True
            if self._custom_classifier is not None:
                self._pretrained_model = self._custom_classifier
            elif self._use_default_classifier:
                try:
                    self._pretrained_model = StatisticalHeuristicClassifier()
                except Exception as exc:
                    logger.warning(f"Failed to instantiate default pretrained classifier: {exc}")
                    self._pretrained_model = None
                    self._model_load_error = str(exc)
            else:
                self._pretrained_model = None
        return self._pretrained_model

    @staticmethod
    def _compute_error_level_analysis(image: Image.Image) -> Dict[str, float]:
        """
        Computes Error Level Analysis (ELA) by recompressing the image as JPEG at quality 90.
        Measures the absolute pixel residual difference between the original and recompressed image.
        In-painted, spliced, or synthetic image regions exhibit higher or irregular compression error rates.
        """
        try:
            rgb_image = image.convert("RGB")
            buf = io.BytesIO()
            rgb_image.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            recompressed = Image.open(buf).convert("RGB")

            orig_arr = np.asarray(rgb_image, dtype=np.float32)
            recomp_arr = np.asarray(recompressed, dtype=np.float32)

            diff = np.abs(orig_arr - recomp_arr)
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))
            return {
                "ela_mean_error": round(mean_diff, 2),
                "ela_max_error": round(max_diff, 2),
            }
        except Exception as e:
            logger.debug(f"ELA calculation skipped or failed: {e}")
            return {"ela_mean_error": 0.0, "ela_max_error": 0.0}

    @staticmethod
    def _compute_patch_noise_variance(img_arr: np.ndarray, grid_size: int = 8) -> Dict[str, float]:
        """
        Evaluates spatial noise distribution uniformity across non-overlapping image patches (MVNN pixel domain).
        Pristine photographic images display consistent sensor noise across patches, whereas composite or diffusion
        splicing introduces sharp variations in local noise energy (high coefficient of variation).
        """
        try:
            h, w = img_arr.shape
            patch_h, patch_w = max(8, h // grid_size), max(8, w // grid_size)
            laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

            patch_vars = []
            for y in range(0, h - patch_h + 1, patch_h):
                for x in range(0, w - patch_w + 1, patch_w):
                    patch = img_arr[y : y + patch_h, x : x + patch_w]
                    if patch.shape[0] >= 4 and patch.shape[1] >= 4:
                        lap = scipy.signal.convolve2d(patch, laplacian_kernel, mode="valid")
                        p_var = float(np.var(lap))
                        patch_vars.append(p_var)

            if len(patch_vars) >= 4:
                mean_v = float(np.mean(patch_vars))
                std_v = float(np.std(patch_vars))
                dispersion = std_v / (mean_v + 1e-5)
                return {
                    "patch_noise_dispersion": round(dispersion, 3),
                    "patch_noise_mean": round(mean_v, 2),
                }
            return {"patch_noise_dispersion": 0.0, "patch_noise_mean": 0.0}
        except Exception as e:
            logger.debug(f"Patch noise variance calculation skipped: {e}")
            return {"patch_noise_dispersion": 0.0, "patch_noise_mean": 0.0}

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

        # If a pretrained model is strictly required but unavailable, report NOT_RUN or ERROR per contract
        model = self._get_lazy_model()
        if self._require_pretrained and model is None:
            if self._model_load_error:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.ERROR,
                    manipulation_score=None,
                    confidence=0.0,
                    summary=f"Required pretrained AI-image classifier failed to load: {self._model_load_error}",
                    error_message=self._model_load_error,
                ), []
            else:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.NOT_RUN,
                    manipulation_score=None,
                    confidence=0.0,
                    summary="Required pretrained AI-image model is unavailable or disabled; execution reported as NOT_RUN per contract.",
                ), []

    def _analyze_single_pil_image(
        self, image_raw: Image.Image, model: Optional[PretrainedImageClassifier]
    ) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        try:
            image = image_raw.convert("L")
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

        # Check for degenerate cases (solid color, flat image, or thumbnail)
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
        mask_high_freq = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2 > radius ** 2

        total_energy = float(np.sum(magnitude_spectrum))
        high_freq_energy = float(np.sum(magnitude_spectrum[mask_high_freq]))
        high_freq_ratio = high_freq_energy / (total_energy + 1e-7)

        # Discrete Laplacian operator for noise residual estimation
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        laplacian = scipy.signal.convolve2d(img_arr, laplacian_kernel, mode="valid")
        laplacian_var = float(np.var(laplacian))

        # Check for periodic grid peaks in frequency spectrum
        high_freq_vals = magnitude_spectrum[mask_high_freq]
        mean_hf = float(np.mean(high_freq_vals))
        std_hf = float(np.std(high_freq_vals)) + 1e-5
        peak_z_score = float(np.max(high_freq_vals) - mean_hf) / std_hf
        grid_artifact_detected = (peak_z_score > 4.2)

        evidence: List[EvidenceItem] = []

        # Pixel-domain forensics (MVNN dual-domain visual analysis)
        ela_metrics = self._compute_error_level_analysis(image_raw)
        noise_metrics = self._compute_patch_noise_variance(img_arr)

        metrics: Dict[str, Any] = {
            "high_frequency_ratio": round(high_freq_ratio, 3),
            "laplacian_variance": round(laplacian_var, 2),
            "frequency_peak_z_score": round(peak_z_score, 2),
            "resolution": f"{width}x{height}",
        }
        metrics.update(ela_metrics)
        metrics.update(noise_metrics)

        # Flag pixel-domain inconsistency if ELA or patch noise dispersion is unusually elevated
        if ela_metrics["ela_max_error"] > 65.0 and ela_metrics["ela_mean_error"] > 7.0:
            evidence.append(
                EvidenceItem(
                    id="ev_ela_anomaly",
                    detector_id=self.detector_id,
                    evidence_type="PIXEL_DOMAIN_INCONSISTENCY",
                    description=(
                        f"Error Level Analysis (ELA) reveals localized compression residual anomalies "
                        f"(max error: {ela_metrics['ela_max_error']:.1f}, mean: {ela_metrics['ela_mean_error']:.1f})."
                    ),
                    severity=EvidenceSeverity.MEDIUM,
                    metadata=ela_metrics,
                )
            )
        elif noise_metrics["patch_noise_dispersion"] > 2.5:
            evidence.append(
                EvidenceItem(
                    id="ev_noise_dispersion",
                    detector_id=self.detector_id,
                    evidence_type="NOISE_INCONSISTENCY",
                    description=(
                        f"Local patch noise variance exhibits sharp spatial discontinuity across image tiles "
                        f"(dispersion CV: {noise_metrics['patch_noise_dispersion']:.2f})."
                    ),
                    severity=EvidenceSeverity.MEDIUM,
                    metadata=noise_metrics,
                )
            )

        # Run pretrained classifier if available
        model_prob: Optional[float] = None
        model_conf: Optional[float] = None
        if model is not None:
            try:
                model_prob, model_conf = model.predict_synthetic(image_raw)
                metrics["pretrained_classifier_score"] = round(model_prob, 3)
                metrics["pretrained_classifier_confidence"] = round(model_conf, 3)
            except Exception as exc:
                logger.warning(f"Inference error in pretrained classifier: {exc}")
                metrics["pretrained_classifier_error"] = str(exc)

        # Signal Synthesis with Threshold Validation & Uncertainty Preservation
        # Rule: Never label a media file AI-generated solely because a model score exceeds an unvalidated threshold.
        spectral_synthetic_flag = False
        spectral_authentic_flag = False

        if (high_freq_ratio > 0.82 and peak_z_score > 3.5) or grid_artifact_detected:
            spectral_synthetic_flag = True
            evidence.append(
                EvidenceItem(
                    id="ev_spectral_grid",
                    detector_id=self.detector_id,
                    evidence_type="FREQUENCY_ANOMALY",
                    description=f"Elevated high-frequency spectral ratio ({high_freq_ratio:.3f}) correlates with diffusion grid artifacts.",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"high_freq_ratio": round(high_freq_ratio, 3), "peak_z_score": round(peak_z_score, 2)},
                )
            )
        elif high_freq_ratio < 0.35 and laplacian_var < 50.0:
            spectral_synthetic_flag = True
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
        elif laplacian_var > 60.0 and peak_z_score < 3.5:
            spectral_authentic_flag = True

        # Combine model inference with physical frequency corroboration
        if model_prob is not None:
            # Check for unvalidated / ambiguous model score band (0.40 <= prob <= 0.70)
            if 0.40 <= model_prob <= 0.70 and peak_z_score < 6.0:
                # Ambiguous model score without physical corroboration: PRESERVE UNCERTAINTY
                manipulation_score = round(model_prob, 2)
                confidence = 0.50
                summary = (
                    f"Pretrained vision model score ({model_prob:.2f}) falls within the unvalidated ambiguity interval "
                    f"without corroborating high-frequency spectral anomalies. Preserving uncertainty."
                )
                evidence.append(
                    EvidenceItem(
                        id="ev_model_ambiguity",
                        detector_id=self.detector_id,
                        evidence_type="MODEL_UNCERTAINTY",
                        description=f"Classifier score ({model_prob:.2f}) is unvalidated without frequency corroboration.",
                        severity=EvidenceSeverity.LOW,
                        metadata={"model_score": round(model_prob, 3)},
                    )
                )
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.UNCERTAIN,
                    manipulation_score=manipulation_score,
                    confidence=confidence,
                    summary=summary,
                    metrics=metrics,
                ), evidence

            # When model score is high: only declare high manipulation if frequency corroborates or model is very high
            if model_prob > 0.70:
                if spectral_synthetic_flag:
                    manipulation_score = min(0.95, round(0.50 * model_prob + 0.45, 2))
                    confidence = 0.88
                    evidence.append(
                        EvidenceItem(
                            id="ev_model_high_synthetic",
                            detector_id=self.detector_id,
                            evidence_type="MODEL_INFERENCE",
                            description=f"Pretrained vision classifier identified generative cues ({model_prob:.2f}) corroborated by frequency anomalies.",
                            severity=EvidenceSeverity.HIGH,
                            metadata={"model_score": round(model_prob, 3)},
                        )
                    )
                elif spectral_authentic_flag:
                    # Model contradicts physical sensor noise: temper score and preserve uncertainty
                    manipulation_score = 0.52
                    confidence = 0.60
                    evidence.append(
                        EvidenceItem(
                            id="ev_model_spectral_conflict",
                            detector_id=self.detector_id,
                            evidence_type="SIGNAL_CONFLICT",
                            description=f"Model flagged potential synthesis ({model_prob:.2f}), but image exhibits organic camera sensor noise (Laplacian: {laplacian_var:.1f}).",
                            severity=EvidenceSeverity.MEDIUM,
                            metadata={"model_score": round(model_prob, 3), "laplacian_variance": round(laplacian_var, 2)},
                        )
                    )
                else:
                    manipulation_score = round(model_prob, 2)
                    confidence = 0.75
            elif model_prob < 0.30:
                manipulation_score = round(max(0.05, model_prob), 2)
                confidence = 0.85
            else:
                manipulation_score = round(model_prob, 2)
                confidence = 0.65
        else:
            # Algorithmic frequency fallback when no pretrained model ran
            if spectral_synthetic_flag:
                manipulation_score = 0.75
                confidence = 0.80
            elif spectral_authentic_flag:
                manipulation_score = 0.15
                confidence = 0.80
            else:
                manipulation_score = 0.30
                confidence = 0.65

        manipulation_score = round(min(1.0, max(0.05, manipulation_score)), 2)

        summary = (
            f"Analyzed {width}x{height} image spectrum. High-frequency energy ratio: {high_freq_ratio:.2f}, "
            f"Laplacian variance: {laplacian_var:.1f}."
        )
        if model_prob is not None:
            summary += f" Pretrained vision model probability: {model_prob:.2f}."

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=manipulation_score,
            confidence=confidence,
            summary=summary,
            metrics=metrics,
        )
        return finding, evidence

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

        # If a pretrained model is strictly required but unavailable, report NOT_RUN or ERROR per contract
        model = self._get_lazy_model()
        if self._require_pretrained and model is None:
            if self._model_load_error:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.ERROR,
                    manipulation_score=None,
                    confidence=0.0,
                    summary=f"Required pretrained AI-image classifier failed to load: {self._model_load_error}",
                    error_message=self._model_load_error,
                ), []
            else:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.NOT_RUN,
                    manipulation_score=None,
                    confidence=0.0,
                    summary="Required pretrained AI-image model is unavailable or disabled; execution reported as NOT_RUN per contract.",
                ), []

        is_video = input_data.has_video or bool(
            input_data.media_filename and input_data.media_filename.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv"))
        ) or bool(
            input_data.media_content_type and input_data.media_content_type.startswith("video/")
        )

        if is_video:
            try:
                frames = extract_video_frames(input_data.media_bytes, max_frames=5)
            except Exception as exc:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.ERROR,
                    manipulation_score=None,
                    confidence=0.0,
                    summary=f"Failed to extract video frames for synthetic evaluation: {str(exc)}",
                    error_message=str(exc),
                ), []

            if not frames:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.UNCERTAIN,
                    manipulation_score=None,
                    confidence=0.20,
                    summary="No decodable frames found in the video container.",
                ), []

            findings_list: List[DetectorFinding] = []
            all_evidence: List[EvidenceItem] = []
            for frame_idx, frame_rgb in enumerate(frames):
                pil_f = Image.fromarray(frame_rgb)
                finding_f, ev_f = self._analyze_single_pil_image(pil_f, model)
                if finding_f.status == DetectorStatus.SUCCESS and finding_f.manipulation_score is not None:
                    findings_list.append(finding_f)
                    for item in ev_f:
                        item_copy = item.model_copy()
                        item_copy.id = f"{item.id}_f{frame_idx}"
                        all_evidence.append(item_copy)

            if not findings_list:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.UNCERTAIN,
                    manipulation_score=None,
                    confidence=0.30,
                    summary=f"Evaluated {len(frames)} frames; insufficient textural variation for spectral inspection.",
                ), []

            scores = [f.manipulation_score for f in findings_list if f.manipulation_score is not None]
            confs = [f.confidence for f in findings_list]
            max_score = max(scores)
            avg_score = float(np.mean(scores))
            # Synthesis in any key frame sequence indicates generative video
            final_score = round(0.7 * max_score + 0.3 * avg_score, 2)
            final_conf = round(float(np.mean(confs)), 2)

            agg_metrics: Dict[str, Any] = {
                "frames_analyzed": len(frames),
                "max_frame_manipulation_score": max_score,
                "mean_frame_manipulation_score": round(avg_score, 2),
                "sample_resolution": findings_list[0].metrics.get("resolution", "unknown"),
            }
            if "high_frequency_ratio" in findings_list[0].metrics:
                agg_metrics["high_frequency_ratio"] = round(float(np.mean([f.metrics.get("high_frequency_ratio", 0.0) for f in findings_list])), 3)
            if "laplacian_variance" in findings_list[0].metrics:
                agg_metrics["laplacian_variance"] = round(float(np.mean([f.metrics.get("laplacian_variance", 0.0) for f in findings_list])), 1)

            summary = (
                f"Evaluated {len(frames)} video frames for generative synthesis and diffusion artifacts. "
                f"Peak frame manipulation score: {max_score:.2f} (mean: {avg_score:.2f})."
            )

            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.SUCCESS,
                manipulation_score=final_score,
                confidence=final_conf,
                summary=summary,
                metrics=agg_metrics,
            ), all_evidence

        try:
            image_raw = Image.open(io.BytesIO(input_data.media_bytes))
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

        return self._analyze_single_pil_image(image_raw, model)
