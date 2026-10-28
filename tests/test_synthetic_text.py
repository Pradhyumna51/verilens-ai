"""
Separate test suite for SyntheticTextDetector.
Tests both real pretrained model inference and mocked checks,
clearly distinguishing between them as requested by the API contract.
Also tests threshold validation, uncertainty preservation, and unavailable model reporting.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.detectors.base import DetectorInput
from verilens.detectors.synthetic_text import (
    SyntheticTextDetector,
    PretrainedTextClassifier,
    CalibratedTextFeatureClassifier,
)
from verilens.schemas.response import DetectorStatus, EvidenceSeverity


SYNTHETIC_LLM_TEXT = (
    "In conclusion, it is important to remember that renewable energy represents a vital pillar of the global transition. "
    "Furthermore, sustainable investments delve into critical infrastructure that plays a crucial role in reducing emissions. "
    "Moreover, the continuous development of battery storage serves as a testament to modern engineering ingenuity. "
    "Ultimately, these coordinated technological initiatives will determine the long-term viability of international carbon targets."
)

NATURAL_HUMAN_TEXT = (
    "Wait, really? I checked the train schedule three times before leaving the station yesterday morning. "
    "The express was delayed by nearly forty minutes because of track maintenance down near the river junction, "
    "which honestly threw off our whole afternoon schedule. "
    "Never again! Next time we are just driving."
)

SHORT_TEXT = "Just a quick brief headline."


class TestRealModelInference:
    """
    SECTION 1: REAL-MODEL INFERENCE (No mocks).
    Validates actual execution of the built-in calibrated pretrained text classifier.
    """

    def test_real_model_inference_on_synthetic_text(self):
        detector = SyntheticTextDetector(use_default_classifier=True)
        inp = DetectorInput(text=SYNTHETIC_LLM_TEXT)

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert "pretrained_classifier_score" in finding.metrics
        assert isinstance(finding.metrics["pretrained_classifier_score"], float)
        assert 0.0 <= finding.metrics["pretrained_classifier_score"] <= 1.0
        assert finding.metrics["marker_hits"] >= 3
        assert finding.manipulation_score is not None
        assert finding.manipulation_score >= 0.70
        assert any(e.evidence_type == "TEXT_STATISTIC" for e in evidence)

    def test_real_model_inference_on_natural_bursty_text(self):
        detector = SyntheticTextDetector(use_default_classifier=True)
        inp = DetectorInput(text=NATURAL_HUMAN_TEXT)

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        # Natural human writing has high burstiness (> 0.50)
        assert finding.metrics["burstiness_index"] > 0.50
        assert finding.manipulation_score is not None
        assert finding.manipulation_score <= 0.35


class TestMockedClassifierChecks:
    """
    SECTION 2: MOCKED MODEL CHECKS.
    Simulates external transformer predictions to verify threshold calibration,
    evidence routing, and uncertainty preservation without external weight dependencies.
    """

    def test_mocked_classifier_high_confidence_synthetic(self):
        """Mock: High model probability (0.90) with repetitive machine cadence."""
        mock_model = MagicMock(spec=PretrainedTextClassifier)
        mock_model.predict_synthetic.return_value = (0.90, 0.88)

        detector = SyntheticTextDetector(classifier=mock_model)
        inp = DetectorInput(text=SYNTHETIC_LLM_TEXT)

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score is not None
        assert finding.manipulation_score >= 0.75
        assert finding.confidence >= 0.75

    def test_mocked_classifier_unvalidated_threshold_preserves_uncertainty(self):
        """
        Mock: Model returns 0.55 (within ambiguity interval [0.35, 0.65]) on natural bursty text.
        Rule: Never label AI-generated solely because a model score exceeds an unvalidated threshold.
        Expected: Detector MUST preserve uncertainty (status UNCERTAIN).
        """
        mock_model = MagicMock(spec=PretrainedTextClassifier)
        mock_model.predict_synthetic.return_value = (0.55, 0.50)

        detector = SyntheticTextDetector(classifier=mock_model)
        inp = DetectorInput(text=NATURAL_HUMAN_TEXT)

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.UNCERTAIN
        assert finding.confidence <= 0.50
        assert "unvalidated ambiguity interval" in finding.summary
        assert any(e.evidence_type == "TEXT_UNCERTAINTY" for e in evidence)


class TestUnavailableModelHandling:
    """
    SECTION 3: UNAVAILABLE MODEL REPORTING (NOT_RUN or ERROR per contract).
    """

    def test_unavailable_required_model_reports_not_run(self):
        detector = SyntheticTextDetector(
            classifier=None,
            require_pretrained=True,
            use_default_classifier=False,
        )
        inp = DetectorInput(text=SYNTHETIC_LLM_TEXT)

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.NOT_RUN
        assert finding.manipulation_score is None
        assert "unavailable" in finding.summary.lower()

    def test_failed_required_model_reports_error(self):
        detector = SyntheticTextDetector(
            classifier=None,
            require_pretrained=True,
            use_default_classifier=False,
        )
        detector._model_load_attempted = True
        detector._model_load_error = "Model weights failed to load: HuggingFace hub unreachable"
        inp = DetectorInput(text=SYNTHETIC_LLM_TEXT)

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.ERROR
        assert finding.manipulation_score is None
        assert "failed to load" in finding.summary.lower()


class TestOperationalBoundaries:
    """
    SECTION 4: OPERATIONAL BOUNDARY & SHORT INPUT CHECKS.
    """

    def test_not_run_when_no_text(self):
        detector = SyntheticTextDetector()
        inp = DetectorInput(media_bytes=b"dummy", media_content_type="image/png")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.NOT_RUN
        assert finding.manipulation_score is None

    def test_uncertain_status_on_short_text(self):
        detector = SyntheticTextDetector()
        inp = DetectorInput(text=SHORT_TEXT)
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.UNCERTAIN
        assert finding.manipulation_score is None
        assert "below the threshold" in finding.summary
