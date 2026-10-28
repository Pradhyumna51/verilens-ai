"""
Separate test suite for SyntheticImageDetector.
Tests both real pretrained model inference and mocked checks,
clearly distinguishing between them as requested by the API contract.
Also tests threshold validation, uncertainty preservation, and unavailable model reporting.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.detectors.base import DetectorInput
from verilens.detectors.synthetic_image import (
    SyntheticImageDetector,
    PretrainedImageClassifier,
    CalibratedFeatureClassifier,
)
from verilens.schemas.response import DetectorStatus, EvidenceSeverity


def create_sample_image(width=128, height=128, mode="grid") -> bytes:
    """Creates deterministic test images."""
    buf = io.BytesIO()
    if mode == "grid":
        # Periodic grid pattern typical of GAN/diffusion upsampling artifacts
        arr = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                arr[y, x] = int((x * 7 + y * 9) % 256)
        img = Image.fromarray(arr).convert("RGB")
    elif mode == "organic":
        # Natural photographic scene with organic camera sensor noise
        arr = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                arr[y, x] = [128 + 50 * np.sin(x / 10.0), 128 + 40 * np.cos(y / 10.0), 128 + 30 * np.sin((x + y) / 15.0)]
        noise = np.random.RandomState(42).normal(0, 10, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    elif mode == "smooth":
        # Clean photographic gradient without high-frequency grid artifacts
        arr = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                arr[y, x] = [128 + 60 * np.sin(x / 10.0), 128 + 50 * np.cos(y / 10.0), 128 + 40 * np.sin((x + y) / 15.0)]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    elif mode == "flat":
        img = Image.new("RGB", (width, height), color=(120, 120, 120))
    else:
        arr = np.random.RandomState(0).randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestRealModelInference:
    """
    SECTION 1: REAL-MODEL INFERENCE (No mocks).
    Validates actual execution of the built-in calibrated pretrained feature classifier.
    """

    def test_real_model_inference_executes_successfully(self):
        detector = SyntheticImageDetector(use_default_classifier=True)
        img_bytes = create_sample_image(128, 128, mode="organic")
        inp = DetectorInput(media_bytes=img_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        # Real model inference verified
        assert finding.status == DetectorStatus.SUCCESS
        assert "pretrained_classifier_score" in finding.metrics
        assert isinstance(finding.metrics["pretrained_classifier_score"], float)
        assert 0.0 <= finding.metrics["pretrained_classifier_score"] <= 1.0
        assert "pretrained_classifier_confidence" in finding.metrics
        assert finding.confidence >= 0.60
        assert finding.execution_time_ms > 0.0

    def test_real_model_inference_on_grid_artifact_pattern(self):
        detector = SyntheticImageDetector(use_default_classifier=True)
        grid_bytes = create_sample_image(128, 128, mode="grid")
        inp = DetectorInput(media_bytes=grid_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        # High frequency grid energy detected
        assert finding.metrics["high_frequency_ratio"] > 0.70
        assert any(e.evidence_type == "FREQUENCY_ANOMALY" for e in evidence)


class TestMockedClassifierChecks:
    """
    SECTION 2: MOCKED MODEL CHECKS.
    Simulates various external neural model predictions to verify threshold calibration,
    evidence routing, and uncertainty preservation without depending on non-deterministic weights.
    """

    def test_mocked_classifier_high_confidence_with_corroboration(self):
        """Mock: High model synthetic probability (0.92) with high-frequency grid artifact."""
        mock_model = MagicMock(spec=PretrainedImageClassifier)
        mock_model.predict_synthetic.return_value = (0.92, 0.90)

        detector = SyntheticImageDetector(classifier=mock_model)
        grid_bytes = create_sample_image(128, 128, mode="grid")
        inp = DetectorInput(media_bytes=grid_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score is not None
        assert finding.manipulation_score >= 0.85
        assert finding.confidence >= 0.85
        assert any(e.evidence_type == "MODEL_INFERENCE" for e in evidence)

    def test_mocked_classifier_unvalidated_threshold_preserves_uncertainty(self):
        """
        Mock: Model probability is 0.60 (in ambiguity interval [0.40, 0.70]) on natural image.
        Rule: Never label AI-generated solely because a model score exceeds an unvalidated threshold.
        Expected: Detector MUST preserve uncertainty (status UNCERTAIN).
        """
        mock_model = MagicMock(spec=PretrainedImageClassifier)
        mock_model.predict_synthetic.return_value = (0.60, 0.55)

        detector = SyntheticImageDetector(classifier=mock_model)
        organic_bytes = create_sample_image(128, 128, mode="smooth")
        inp = DetectorInput(media_bytes=organic_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.UNCERTAIN
        assert finding.confidence <= 0.55
        assert "unvalidated ambiguity interval" in finding.summary
        assert any(e.evidence_type == "MODEL_UNCERTAINTY" for e in evidence)

    def test_mocked_classifier_conflicting_signal_with_sensor_noise(self):
        """
        Mock: Model asserts 0.80 synthetic, but spatial Laplacian variance shows organic sensor noise.
        Expected: Detector tempers score to ~0.50 and flags SIGNAL_CONFLICT evidence.
        """
        mock_model = MagicMock(spec=PretrainedImageClassifier)
        mock_model.predict_synthetic.return_value = (0.80, 0.75)

        detector = SyntheticImageDetector(classifier=mock_model)
        organic_bytes = create_sample_image(128, 128, mode="organic")
        inp = DetectorInput(media_bytes=organic_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score <= 0.60  # Tempered from 0.80 due to conflict
        assert any(e.evidence_type == "SIGNAL_CONFLICT" for e in evidence)


class TestUnavailableModelHandling:
    """
    SECTION 3: UNAVAILABLE MODEL REPORTING (NOT_RUN or ERROR per contract).
    """

    def test_unavailable_required_model_reports_not_run(self):
        detector = SyntheticImageDetector(
            classifier=None,
            require_pretrained=True,
            use_default_classifier=False,
        )
        img_bytes = create_sample_image(128, 128, mode="organic")
        inp = DetectorInput(media_bytes=img_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.NOT_RUN
        assert finding.manipulation_score is None
        assert "unavailable" in finding.summary.lower()

    def test_failed_required_model_reports_error(self):
        detector = SyntheticImageDetector(
            classifier=None,
            require_pretrained=True,
            use_default_classifier=False,
        )
        detector._model_load_attempted = True
        detector._model_load_error = "Weights file missing or network connection refused"

        img_bytes = create_sample_image(128, 128, mode="organic")
        inp = DetectorInput(media_bytes=img_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.ERROR
        assert finding.manipulation_score is None
        assert "failed to load" in finding.summary.lower()


class TestOperationalBoundaries:
    """
    SECTION 4: OPERATIONAL BOUNDARY & DEGENERATE INPUTS.
    """

    def test_not_run_when_no_media(self):
        detector = SyntheticImageDetector()
        inp = DetectorInput(text="Text only query")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.NOT_RUN
        assert finding.manipulation_score is None

    def test_error_status_on_corrupt_bytes(self):
        detector = SyntheticImageDetector()
        inp = DetectorInput(media_bytes=b"CORRUPT_NON_IMAGE_DATA", media_content_type="image/png")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.ERROR
        assert finding.manipulation_score is None
        assert finding.error_message is not None

    def test_uncertain_status_on_flat_color_image(self):
        detector = SyntheticImageDetector()
        flat_bytes = create_sample_image(64, 64, mode="flat")
        inp = DetectorInput(media_bytes=flat_bytes, media_content_type="image/png")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.UNCERTAIN
        assert finding.manipulation_score is None
        assert "insufficient textural variation" in finding.summary
