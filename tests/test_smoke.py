"""
Smoke test suite for VeriLens AI.
Tests /health, /analyze, detector statuses (NOT_RUN, UNCERTAIN, ERROR, SUCCESS),
upload limits, and schema conformance.
"""

import io
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.api.app import app
from verilens.detectors.base import DetectorInput
from verilens.detectors.synthetic_text import SyntheticTextDetector
from verilens.detectors.media_reuse import MediaReuseDetector
from verilens.detectors.synthetic_image import SyntheticImageDetector
from verilens.detectors.consistency import ConsistencyDetector
from verilens.detectors.deepfake import DeepfakeDetector
from verilens.detectors.ocr import OCRDetector
from verilens.schemas.response import DetectorStatus, Verdict, RiskLevel
from verilens.core.config import settings

client = TestClient(app)

def create_dummy_image(width=128, height=128, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [(x * 3) % 256, (y * 4) % 256, (x + y * 2) % 256]
    img = Image.fromarray(arr)
    img.save(buf, format=fmt)
    return buf.getvalue()

def create_flat_color_image(width=64, height=64, color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()

class TestHealthEndpoint:
    def test_health_check_returns_200_and_metadata(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "limits" in data
        assert data["limits"]["max_media_size_bytes"] == settings.MAX_MEDIA_SIZE_BYTES
        assert len(data["detectors"]) == 6
        detector_ids = [d["id"] for d in data["detectors"]]
        assert "consistency" in detector_ids
        assert "media_reuse" in detector_ids
        assert "synthetic_image" in detector_ids
        assert "synthetic_text" in detector_ids
        assert "deepfake" in detector_ids
        assert "ocr_detector" in detector_ids

class TestAnalyzeEndpoint:
    def test_analyze_empty_request_returns_400(self):
        response = client.post("/analyze")
        assert response.status_code == 400
        assert "At least one input modality" in response.json()["detail"]

    def test_analyze_unsupported_media_type_returns_415(self):
        dummy_file = ("bad.exe", b"MZexecutablecontent", "application/x-msdownload")
        response = client.post(
            "/analyze",
            data={"text": "Check this file"},
            files={"media_file": dummy_file},
        )
        assert response.status_code == 415
        assert "Unsupported media type" in response.json()["detail"]

    def test_analyze_text_only_request(self):
        text = (
            "Scientists have announced the sudden discovery of a subterranean ocean beneath the Sahara Desert. "
            "The expedition team documented significant freshwater reserves that could potentially transform agriculture. "
            "Further geological surveys are scheduled for the upcoming fiscal quarter to verify aquifer capacity."
        )
        response = client.post("/analyze", data={"text": text})
        assert response.status_code == 200
        data = response.json()

        # Schema checks
        assert "request_id" in data
        assert "overall_assessment" in data
        assert "findings" in data
        assert "evidence" in data
        assert "explanation" in data
        assert "limitations" in data

        assert "text" in data["modalities_received"]
        assert "image" not in data["modalities_received"]

        # Synthetic text detector ran successfully
        assert data["findings"]["synthetic_text"]["status"] == "SUCCESS"
        # Visual detectors should be NOT_RUN
        assert data["findings"]["synthetic_image"]["status"] == "NOT_RUN"
        assert data["findings"]["media_reuse"]["status"] == "NOT_RUN"
        assert data["findings"]["consistency"]["status"] == "NOT_RUN"
        assert data["findings"]["ocr_detector"]["status"] == "NOT_RUN"

    def test_analyze_multimodal_request(self):
        text = (
            "Breaking report: Major volcanic ash plume observed rising over mountain ridge during afternoon aerial survey. "
            "Emergency authorities recommend immediate evacuation for surrounding communities."
        )
        img_bytes = create_dummy_image(100, 100)
        
        response = client.post(
            "/analyze",
            data={"text": text},
            files={"media_file": ("test_ash.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data["modalities_received"]
        assert "image" in data["modalities_received"]
        
        # Consistency, media_reuse, synthetic_image ran
        assert data["findings"]["consistency"]["status"] == "SUCCESS"
        assert data["findings"]["media_reuse"]["status"] == "SUCCESS"
        assert data["findings"]["synthetic_image"]["status"] == "SUCCESS"
        assert data["findings"]["synthetic_text"]["status"] == "SUCCESS"
        assert data["findings"]["ocr_detector"]["status"] == "SUCCESS"

class TestDetectorStatuses:
    """Verifies that all detectors correctly support NOT_RUN, UNCERTAIN, ERROR, and SUCCESS."""

    def test_not_run_status(self):
        empty_input = DetectorInput(text=None, media_bytes=None)
        
        for det in [ConsistencyDetector(), MediaReuseDetector(), SyntheticImageDetector(), SyntheticTextDetector(), DeepfakeDetector(), OCRDetector()]:
            finding, evidence = det.run(empty_input)
            assert finding.status == DetectorStatus.NOT_RUN
            assert finding.manipulation_score is None

    def test_uncertain_status_on_short_text(self):
        det = SyntheticTextDetector()
        short_input = DetectorInput(text="Just two words.")
        finding, evidence = det.run(short_input)
        assert finding.status == DetectorStatus.UNCERTAIN
        assert "below the threshold" in finding.summary

    def test_uncertain_status_on_micro_thumbnail(self):
        det = MediaReuseDetector()
        micro_image = create_dummy_image(width=16, height=16)
        micro_input = DetectorInput(media_bytes=micro_image, media_content_type="image/png")
        finding, evidence = det.run(micro_input)
        assert finding.status == DetectorStatus.UNCERTAIN
        assert "too small" in finding.summary

    def test_error_status_on_corrupted_media_bytes(self):
        det = MediaReuseDetector()
        corrupt_input = DetectorInput(
            media_bytes=b"CORRUPTED_NOT_AN_IMAGE_HEADER_XYZ",
            media_content_type="image/png",
            media_filename="corrupted.png",
        )
        finding, evidence = det.run(corrupt_input)
        assert finding.status == DetectorStatus.ERROR
        assert finding.error_message is not None

    def test_uncertain_status_on_zero_variance_image(self):
        det = SyntheticImageDetector()
        flat_image = create_flat_color_image(width=64, height=64)
        flat_input = DetectorInput(media_bytes=flat_image, media_content_type="image/png")
        finding, evidence = det.run(flat_input)
        assert finding.status == DetectorStatus.UNCERTAIN
        assert "insufficient textural variation" in finding.summary

    def test_success_status(self):
        det = SyntheticTextDetector()
        valid_input = DetectorInput(
            text="The municipal council voted unanimously to approve the expansion of the light rail system yesterday afternoon."
        )
        finding, evidence = det.run(valid_input)
        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score is not None
        assert 0.0 <= finding.manipulation_score <= 1.0
