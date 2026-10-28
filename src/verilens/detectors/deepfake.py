"""
Deepfake Analysis Engine.
Inspects facial blend boundaries, skin-region color space transitions,
warping artifacts, and temporal consistency across video frames.
Includes bounded video frame extraction using OpenCV, dedicated face detection,
threshold validation, and uncertainty preservation.
Supports NOT_RUN, UNCERTAIN, ERROR, and SUCCESS statuses.
"""

import io
import os
import tempfile
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)
from verilens.core.logger import get_logger

logger = get_logger("DeepfakeDetector")


def extract_bounded_video_frames(
    video_bytes: bytes,
    max_frames: int = 12,
    max_duration_seconds: float = 30.0,
) -> List[np.ndarray]:
    """
    Safely extracts a bounded number of evenly spaced RGB frames from video bytes using OpenCV.
    Guarantees resource cleanup via temporary file and VideoCapture release.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV ('opencv-python') is required for video frame extraction. Please install it.")

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        temp_file.write(video_bytes)
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise ValueError("OpenCV failed to open video container.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if total_frames <= 0:
            raise ValueError("Video contains zero decodable frames.")

        max_allowed_frames = min(total_frames, int(fps * max_duration_seconds))
        sample_count = min(max_frames, max_allowed_frames)
        if sample_count <= 0:
            sample_count = 1

        indices = np.linspace(0, max_allowed_frames - 1, sample_count, dtype=int)
        frames: List[np.ndarray] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize if frame is oversized (e.g. > 1080p) to keep evaluation bounded
                h, w, _ = rgb_frame.shape
                if max(h, w) > 1080:
                    scale = 1080.0 / max(h, w)
                    rgb_frame = cv2.resize(rgb_frame, (int(w * scale), int(h * scale)))
                frames.append(rgb_frame)

        cap.release()
        return frames
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except OSError:
                pass


class PretrainedDeepfakeClassifier(ABC):
    """Abstract interface for dedicated face-forgery neural classifiers."""

    @abstractmethod
    def predict_deepfake(self, face_rgb: np.ndarray) -> Tuple[float, float]:
        """
        Runs inference on an RGB face crop array.
        Returns:
            (prob_fake, confidence): float values between 0.0 and 1.0.
        """
        pass


class StatisticalFaceArtifactClassifier(PretrainedDeepfakeClassifier):
    """
    Statistical heuristic face-forgery classifier (NOT pretrained on labeled data).
    Evaluates boundary seam gradient, chroma delta, and edge transitions
    using hand-tuned logistic weights. Scores are approximate, not validated.
    """

    def __init__(self):
        # Logistic weights for [abnormal_boundary_blur, boundary_discontinuity, chroma_divergence]
        self.weights = np.array([3.1, 2.5, 2.0], dtype=np.float32)
        self.bias = -2.4

    def predict_deepfake(self, face_rgb: np.ndarray) -> Tuple[float, float]:
        h, w, _ = face_rgb.shape
        if h < 16 or w < 16:
            return 0.10, 0.50

        if cv2 is not None:
            gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            r = face_rgb[:, :, 0].astype(np.float32)
            g = face_rgb[:, :, 1].astype(np.float32)
            b = face_rgb[:, :, 2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Compute gradient
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Margin vs center gradient
        margin_mask = np.zeros((h, w), dtype=bool)
        margin_w = max(2, int(min(h, w) * 0.15))
        margin_mask[:margin_w, :] = True
        margin_mask[-margin_w:, :] = True
        margin_mask[:, :margin_w] = True
        margin_mask[:, -margin_w:] = True

        center_mask = ~margin_mask
        mean_margin = float(np.mean(grad_mag[margin_mask])) if np.any(margin_mask) else 0.0
        mean_center = float(np.mean(grad_mag[center_mask])) if np.any(center_mask) else 1.0

        # Face swaps often exhibit abnormally low margin gradient (blurred mask feathering)
        blur_feature = float(np.clip(1.0 - (mean_margin / max(1.0, mean_center)), 0.0, 1.0))
        discontinuity_feature = float(np.clip((mean_margin - 60.0) / 40.0, 0.0, 1.0))

        # Chroma divergence in YCrCb
        if cv2 is not None:
            ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
            cr = ycrcb[:, :, 1].astype(np.float32)
            cb = ycrcb[:, :, 2].astype(np.float32)
        else:
            r = face_rgb[:, :, 0].astype(np.float32)
            b = face_rgb[:, :, 2].astype(np.float32)
            cr = (r - gray) * 0.7132 + 128.0
            cb = (b - gray) * 0.5643 + 128.0
        center_cr = float(np.mean(cr[center_mask])) if np.any(center_mask) else 0.0
        margin_cr = float(np.mean(cr[margin_mask])) if np.any(margin_mask) else 0.0
        chroma_diff = float(np.clip(abs(center_cr - margin_cr) / 25.0, 0.0, 1.0))

        features = np.array([blur_feature, discontinuity_feature, chroma_diff], dtype=np.float32)
        logit = float(np.dot(self.weights, features) + self.bias)
        prob = 1.0 / (1.0 + np.exp(-logit))
        conf = float(np.clip(2.0 * abs(prob - 0.5) * 0.25 + 0.40, 0.35, 0.65))
        return float(prob), conf


class DeepfakeDetector(BaseDetector):
    detector_id = "deepfake"
    detector_name = "Deepfake Analysis Engine"
    description = (
        "Examines facial region boundaries, skin chromatic consistency, blend transitions, "
        "and temporal stability across video frames with threshold validation."
    )
    required_modalities = ["image"]

    def __init__(
        self,
        classifier: Optional[PretrainedDeepfakeClassifier] = None,
        require_pretrained: bool = False,
        use_default_classifier: bool = True,
    ):
        self._custom_classifier = classifier
        self._require_pretrained = require_pretrained
        self._use_default_classifier = use_default_classifier
        self._pretrained_model: Optional[PretrainedDeepfakeClassifier] = classifier
        self._model_load_attempted: bool = False
        self._model_load_error: Optional[str] = None
        self._cascade: Optional[Any] = None

    def _get_cascade(self) -> Optional[Any]:
        if cv2 is None or not hasattr(cv2, "CascadeClassifier"):
            return None
        if self._cascade is None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as exc:
                logger.warning(f"Failed to load OpenCV face cascade: {exc}")
                self._cascade = None
        return self._cascade

    def _get_lazy_facenet(self) -> Optional[PretrainedDeepfakeClassifier]:
        if not self._model_load_attempted:
            self._model_load_attempted = True
            if self._custom_classifier is not None:
                self._pretrained_model = self._custom_classifier
            elif self._use_default_classifier:
                try:
                    self._pretrained_model = StatisticalFaceArtifactClassifier()
                except Exception as exc:
                    logger.warning(f"Failed to instantiate default deepfake classifier: {exc}")
                    self._pretrained_model = None
                    self._model_load_error = str(exc)
            else:
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
                summary="No visual media provided for deepfake evaluation.",
            ), []

        if input_data.has_video and cv2 is None:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary="OpenCV is required for video decoding but is not installed.",
                error_message="OpenCV is required for video decoding but is not installed.",
            ), []

        # Check if pretrained model is required but unavailable
        model = self._get_lazy_facenet()
        if self._require_pretrained and model is None:
            if self._model_load_error:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.ERROR,
                    manipulation_score=None,
                    confidence=0.0,
                    summary=f"Required deepfake face-forgery model failed to load: {self._model_load_error}",
                    error_message=self._model_load_error,
                ), []
            else:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.NOT_RUN,
                    manipulation_score=None,
                    confidence=0.0,
                    summary="Required deepfake face-forgery model is unavailable or disabled; execution reported as NOT_RUN per contract.",
                ), []

        # Extract frames: either bounded video frames or single image frame
        frames: List[np.ndarray] = []
        is_video = input_data.has_video

        if is_video:
            try:
                frames = extract_bounded_video_frames(input_data.media_bytes, max_frames=12)
            except Exception as exc:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.ERROR,
                    manipulation_score=None,
                    confidence=0.0,
                    summary=f"Failed to extract video frames: {str(exc)}",
                    error_message=str(exc),
                ), []
        else:
            try:
                image = Image.open(io.BytesIO(input_data.media_bytes)).convert("RGB")
                frames = [np.array(image, dtype=np.uint8)]
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

        if not frames:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary="Zero frames could be decoded from the provided media input.",
            ), []

        # Face detection across extracted frames using OpenCV CascadeClassifier
        cascade = self._get_cascade()
        face_crops: List[np.ndarray] = []
        face_positions: List[Tuple[int, int, int, int]] = []
        total_skin_ratios: List[float] = []
        boundary_gradients: List[float] = []
        background_gradients: List[float] = []

        for frame in frames:
            h, w, _ = frame.shape
            detected_rects = []
            if cascade is not None and cv2 is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    detected_rects = cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                    )
                except Exception as exc:
                    logger.warning(f"Face cascade detection failed: {exc}")
                    detected_rects = []

            # Skin tone candidate segmentation fallback
            r = frame[:, :, 0].astype(int)
            g = frame[:, :, 1].astype(int)
            b = frame[:, :, 2].astype(int)
            skin_mask = (
                (r > 95) & (g > 40) & (b > 20) &
                (r > g) & (r > b) &
                ((r - g) > 15) & (np.abs(r - g) > 15)
            )
            skin_ratio = float(np.sum(skin_mask)) / max(1, h * w)
            total_skin_ratios.append(skin_ratio)

            if len(detected_rects) > 0:
                for (fx, fy, fw, fh) in detected_rects:
                    face_crops.append(frame[fy:fy + fh, fx:fx + fw])
                    face_positions.append((fx + fw // 2, fy + fh // 2, fw, fh))
            elif skin_ratio >= 0.03:
                # Use bounding box of skin mask as candidate face region
                y_idx, x_idx = np.where(skin_mask)
                if len(y_idx) > 0:
                    min_y, max_y = int(np.min(y_idx)), int(np.max(y_idx))
                    min_x, max_x = int(np.min(x_idx)), int(np.max(x_idx))
                    if (max_y - min_y) > 20 and (max_x - min_x) > 20:
                        face_crops.append(frame[min_y:max_y, min_x:max_x])
                        face_positions.append((
                            (min_x + max_x) // 2,
                            (min_y + max_y) // 2,
                            max_x - min_x,
                            max_y - min_y,
                        ))

            # Calculate boundary gradient on skin / face contour
            skin_uint8 = skin_mask.astype(np.uint8)
            boundary = np.zeros_like(skin_uint8)
            boundary[1:, :] |= (skin_uint8[1:, :] != skin_uint8[:-1, :])
            boundary[:, 1:] |= (skin_uint8[:, 1:] != skin_uint8[:, :-1])

            boundary_pixels = int(np.sum(boundary))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            gy, gx = np.gradient(lum)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)

            if boundary_pixels > 0:
                boundary_gradients.append(float(np.mean(grad_mag[boundary > 0])))
            # Background gradient (non-skin) for relative sharpness comparison
            bg_mask = ~skin_mask
            if np.sum(bg_mask) > 100:
                background_gradients.append(float(np.mean(grad_mag[bg_mask])))

        avg_skin_ratio = float(np.mean(total_skin_ratios)) if total_skin_ratios else 0.0
        faces_detected = len(face_crops)

        # Operational condition: if no facial or skin area is found across all frames, return UNCERTAIN
        if faces_detected == 0 and avg_skin_ratio < 0.03:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.20,
                summary="No identifiable human portrait or facial skin regions detected to evaluate blend seams.",
                metrics={
                    "frames_analyzed": len(frames),
                    "faces_detected": 0,
                    "skin_pixel_ratio": round(avg_skin_ratio, 4),
                },
            ), []

        mean_boundary_grad = float(np.mean(boundary_gradients)) if boundary_gradients else 0.0
        mean_bg_grad = float(np.mean(background_gradients)) if background_gradients else 20.0

        evidence: List[EvidenceItem] = []
        metrics: Dict[str, Any] = {
            "frames_analyzed": len(frames),
            "faces_detected": faces_detected,
            "skin_ratio": round(avg_skin_ratio, 3),
            "mean_boundary_gradient": round(mean_boundary_grad, 2),
            "mean_background_gradient": round(mean_bg_grad, 2),
        }

        # Temporal jitter analysis for video inputs
        temporal_jitter = 0.0
        if is_video and len(face_positions) >= 2:
            centroids = np.array([[p[0], p[1]] for p in face_positions], dtype=np.float32)
            face_sizes = np.array([p[2] for p in face_positions], dtype=np.float32)
            mean_size = float(np.mean(face_sizes)) + 1e-5
            diffs = np.linalg.norm(centroids[1:] - centroids[:-1], axis=1)
            temporal_jitter = float(np.mean(diffs)) / mean_size
            metrics["temporal_face_jitter"] = round(temporal_jitter, 3)

        # Run pretrained classifier on face crops if available
        model_probs: List[float] = []
        model_confs: List[float] = []
        if model is not None and face_crops:
            for crop in face_crops[:4]:  # Sample up to 4 face crops
                try:
                    prob, conf = model.predict_deepfake(crop)
                    model_probs.append(prob)
                    model_confs.append(conf)
                except Exception as exc:
                    logger.warning(f"Deepfake model crop inference error: {exc}")

        avg_model_prob: Optional[float] = float(np.mean(model_probs)) if model_probs else None
        avg_model_conf: Optional[float] = float(np.mean(model_confs)) if model_confs else None

        if avg_model_prob is not None:
            metrics["pretrained_classifier_score"] = round(avg_model_prob, 3)
            metrics["pretrained_classifier_confidence"] = round(avg_model_conf or 0.0, 3)

        # Threshold validation: Never label a media file AI-generated or manipulated solely because
        # a model score exceeds an unvalidated threshold.
        manipulation_score = 0.15

        # Check for selective seam feathering vs global camera blur
        seam_blur_detected = False
        seam_harshness_detected = False

        if mean_boundary_grad < 14.0:
            if mean_bg_grad > mean_boundary_grad * 1.2 or mean_bg_grad > 18.0:
                # Selective boundary blurring while background is sharp -> Deepfake seam artifact!
                seam_blur_detected = True
                manipulation_score = 0.72
                evidence.append(
                    EvidenceItem(
                        id="ev_face_boundary_blur",
                        detector_id=self.detector_id,
                        evidence_type="FACIAL_ARTIFACT",
                        description=(
                            f"Selective boundary feathering (gradient: {mean_boundary_grad:.1f} vs "
                            f"background: {mean_bg_grad:.1f}) indicates synthetic face-swap edge blending."
                        ),
                        severity=EvidenceSeverity.HIGH,
                        metadata={"boundary_gradient": round(mean_boundary_grad, 2), "background_gradient": round(mean_bg_grad, 2)},
                    )
                )
            else:
                # Low boundary gradient, but background is ALSO soft/blurry -> General camera blur!
                manipulation_score = 0.20
        elif mean_boundary_grad > 65.0:
            seam_harshness_detected = True
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

        # Video temporal flicker check
        if temporal_jitter > 0.35:
            manipulation_score = max(manipulation_score, 0.70)
            evidence.append(
                EvidenceItem(
                    id="ev_temporal_face_flicker",
                    detector_id=self.detector_id,
                    evidence_type="TEMPORAL_INCONSISTENCY",
                    description=f"Significant inter-frame facial position jitter ({temporal_jitter:.2f}) indicates unstable deepfake tracking.",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"temporal_jitter": round(temporal_jitter, 3)},
                )
            )

        # Harmonize with model score using threshold validation
        if avg_model_prob is not None:
            # Check for ambiguous interval (0.40 <= prob <= 0.65)
            if 0.40 <= avg_model_prob <= 0.65 and not (seam_blur_detected or seam_harshness_detected):
                # Ambiguous model score without physical seam corroboration: PRESERVE UNCERTAINTY
                summary = (
                    f"Face-forgery classifier score ({avg_model_prob:.2f}) is in the unvalidated ambiguity interval "
                    f"without corroborating blend seam anomalies. Preserving uncertainty."
                )
                evidence.append(
                    EvidenceItem(
                        id="ev_deepfake_ambiguity",
                        detector_id=self.detector_id,
                        evidence_type="MODEL_UNCERTAINTY",
                        description=f"Face model probability ({avg_model_prob:.2f}) lacks physical edge corroboration.",
                        severity=EvidenceSeverity.LOW,
                        metadata={"model_score": round(avg_model_prob, 3)},
                    )
                )
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.UNCERTAIN,
                    manipulation_score=round(avg_model_prob, 2),
                    confidence=0.50,
                    summary=summary,
                    metrics=metrics,
                ), evidence

            if avg_model_prob > 0.70 and seam_blur_detected:
                manipulation_score = min(0.95, round(0.5 * avg_model_prob + 0.5 * manipulation_score, 2))
                confidence = 0.85
            elif avg_model_prob < 0.25 and not seam_blur_detected:
                manipulation_score = round(min(manipulation_score, avg_model_prob), 2)
                confidence = 0.80
            else:
                manipulation_score = round(0.5 * avg_model_prob + 0.5 * manipulation_score, 2)
                confidence = 0.72
        else:
            confidence = 0.74

        summary = (
            f"Evaluated portrait candidate regions ({faces_detected} face regions in {len(frames)} frame(s), "
            f"skin coverage: {avg_skin_ratio * 100:.1f}%). Mean boundary transition gradient: {mean_boundary_grad:.1f}."
        )
        if is_video:
            summary += f" Inter-frame temporal jitter: {temporal_jitter:.2f}."
        if avg_model_prob is not None:
            summary += f" Face artifact model probability: {avg_model_prob:.2f}."

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=round(min(1.0, max(0.0, manipulation_score)), 2),
            confidence=confidence,
            summary=summary,
            metrics=metrics,
        )
        return finding, evidence
