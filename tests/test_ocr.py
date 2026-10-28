"""
Unit test suite for OCR Detector (Member 2: Multimodal Analysis).
Tests EasyOCR visual text extraction, sensational meme overlay detection,
temporal conflict cross-referencing, and graceful error handling.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import numpy as np
from PIL import Image

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.detectors.base import DetectorInput
from verilens.detectors.ocr import OCRDetector
from verilens.schemas.response import DetectorStatus, EvidenceSeverity

def make_test_image(width=128, height=128, color=(70, 70, 70)) -> bytes:
    buf = io.BytesIO()
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [color[0], (color[1] + x) % 256, (color[2] + y) % 256]
    img = Image.fromarray(arr)
    img.save(buf, format="PNG")
    return buf.getvalue()

class TestOCRDetector:
    @pytest.fixture
    def detector(self):
        return OCRDetector()

    def test_not_run_when_no_image(self, detector):
        res, evidence = detector.run(DetectorInput(text="Sample claim text without media."))
        assert res.status == DetectorStatus.NOT_RUN
        assert res.manipulation_score is None

    def test_uncertain_on_sub_resolution_thumbnail(self, detector):
        micro_buf = io.BytesIO()
        Image.new("RGB", (16, 16), color=(100, 100, 100)).save(micro_buf, format="PNG")
        res, evidence = detector.run(DetectorInput(
            media_bytes=micro_buf.getvalue(),
            media_content_type="image/png"
        ))
        assert res.status == DetectorStatus.UNCERTAIN
        assert "too low" in res.summary

    def test_error_on_corrupt_image_bytes(self, detector):
        res, evidence = detector.run(DetectorInput(
            media_bytes=b"CORRUPTED_NON_IMAGE_BYTES_12345",
            media_content_type="image/png"
        ))
        assert res.status == DetectorStatus.ERROR
        assert res.error_message is not None

    def test_success_on_clean_image_without_text(self, detector):
        img = make_test_image()
        res, evidence = detector.run(DetectorInput(
            media_bytes=img,
            media_content_type="image/png"
        ))
        assert res.status == DetectorStatus.SUCCESS
        assert res.manipulation_score == 0.05
        assert "extracted_text" in res.metrics
        assert "epistemic_disclaimer" in res.metrics

    def test_mocked_easyocr_detection(self, detector):
        mock_reader = MagicMock()
        # EasyOCR returns list of (bbox, text, prob)
        bbox = [[10, 10], [100, 10], [100, 30], [10, 30]]
        mock_reader.readtext.return_value = [
            (bbox, "BREAKING ALERT: CLASSIFIED ARCHIVE", 0.94)
        ]

        detector._reader = mock_reader
        detector._reader_attempted = True

        img = make_test_image()
        inp = DetectorInput(
            text="General caption regarding the photo.",
            media_bytes=img,
            media_content_type="image/png"
        )
        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.metrics["model_backend"] == "easyocr"
        assert "BREAKING ALERT" in finding.metrics["extracted_text"]
        assert finding.metrics["has_sensational_overlay"] is True
        assert finding.manipulation_score >= 0.50

        # Check evidence
        ev_types = [e.evidence_type for e in evidence]
        assert "EMBEDDED_TEXT_EXTRACTED" in ev_types
        ev_item = [e for e in evidence if e.evidence_type == "EMBEDDED_TEXT_EXTRACTED"][0]
        assert ev_item.severity == EvidenceSeverity.MEDIUM

    def test_mocked_ocr_temporal_conflict_detection(self, detector):
        mock_reader = MagicMock()
        bbox = [[0, 0], [50, 0], [50, 20], [0, 20]]
        mock_reader.readtext.return_value = [
            (bbox, "DOCUMENTED INCIDENT 2018", 0.92)
        ]

        detector._reader = mock_reader
        detector._reader_attempted = True

        img = make_test_image()
        inp = DetectorInput(
            text="Breaking news: Dangerous event occurring right now in 2026 across downtown.",
            media_bytes=img,
            media_content_type="image/png"
        )
        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score >= 0.70
        ev_types = [e.evidence_type for e in evidence]
        assert "OCR_CLAIM_CONFLICT" in ev_types
        conflict_ev = [e for e in evidence if e.evidence_type == "OCR_CLAIM_CONFLICT"][0]
        assert conflict_ev.severity == EvidenceSeverity.HIGH
        assert "2018" in conflict_ev.description and "2026" in conflict_ev.description

