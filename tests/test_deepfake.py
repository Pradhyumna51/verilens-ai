"""
Separate test suite for DeepfakeDetector.
Tests bounded video frame extraction using OpenCV,
dedicated face-forgery detection, real-model inference,
mocked classifier checks, threshold validation, and unavailable model reporting.
"""

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.detectors.base import DetectorInput
from verilens.detectors.deepfake import (
    DeepfakeDetector,
    PretrainedDeepfakeClassifier,
    StatisticalFaceArtifactClassifier,
    extract_bounded_video_frames,
)
from verilens.schemas.response import DetectorStatus, EvidenceSeverity


def create_portrait_image(width=128, height=128, with_face=True) -> bytes:
    """Generates synthetic portrait image bytes with or without human skin/face tones."""
    buf = io.BytesIO()
    if with_face:
        # Natural portrait with normal boundary gradient (14-25, no blur, no harsh seam)
        arr = np.full((height, width, 3), 120, dtype=np.uint8)
        cy, cx = height // 2, width // 2
        ry, rx = 35, 28
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
        for yi in range(height):
            for xi in range(width):
                d = dist[yi, xi]
                if d <= 1.0:
                    alpha = 1.0 if d < 0.93 else (1.0 - (d - 0.93) / 0.07)
                    skin = np.array([215, 145, 110], dtype=np.float32)
                    bg = np.array([120, 120, 120], dtype=np.float32)
                    arr[yi, xi] = (alpha * skin + (1.0 - alpha) * bg).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        # Green/blue landscape without skin
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 1] = 160
        arr[:, :, 2] = 200
        img = Image.fromarray(arr)
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_synthetic_video(num_frames=20, width=128, height=128, with_face=True) -> bytes:
    """Generates a small MP4 video in memory using OpenCV VideoWriter."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, 10.0, (width, height))
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if with_face:
                # Add face patch with slight movement (tracking motion)
                cy, cx = height // 2 + int(np.sin(i / 2.0) * 3), width // 2 + int(np.cos(i / 2.0) * 3)
                ry, rx = 30, 24
                y, x = np.ogrid[:height, :width]
                face_mask = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0
                frame[face_mask] = [110, 145, 215]  # BGR skin tone
            else:
                frame[:, :, 1] = 150  # BGR green
            writer.write(frame)
        writer.release()

        with open(temp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


class TestBoundedVideoFrameExtraction:
    """
    SECTION 1: BOUNDED VIDEO FRAME EXTRACTION (OpenCV).
    """

    def test_bounded_frame_extraction_caps_frames(self):
        video_bytes = create_synthetic_video(num_frames=25, with_face=True)
        # Request at most 6 frames
        frames = extract_bounded_video_frames(video_bytes, max_frames=6, max_duration_seconds=10.0)

        assert isinstance(frames, list)
        assert len(frames) == 6
        assert isinstance(frames[0], np.ndarray)
        assert frames[0].shape[2] == 3  # RGB channels

    def test_corrupt_video_bytes_returns_error(self):
        detector = DeepfakeDetector()
        corrupt_input = DetectorInput(
            media_bytes=b"CORRUPTED_MP4_HEADER_NOT_A_VIDEO",
            media_content_type="video/mp4",
            media_filename="test.mp4",
        )
        finding, evidence = detector.run(corrupt_input)

        assert finding.status == DetectorStatus.ERROR
        assert finding.manipulation_score is None
        assert finding.error_message is not None


class TestRealModelInference:
    """
    SECTION 2: REAL-MODEL INFERENCE (No mocks).
    Validates execution of OpenCV face segmentation + StatisticalFaceArtifactClassifier.
    """

    def test_real_model_inference_on_portrait_image(self):
        detector = DeepfakeDetector(use_default_classifier=True)
        portrait_bytes = create_portrait_image(width=128, height=128, with_face=True)
        inp = DetectorInput(media_bytes=portrait_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status in (DetectorStatus.SUCCESS, DetectorStatus.UNCERTAIN)
        assert finding.metrics["faces_detected"] >= 1
        assert "pretrained_classifier_score" in finding.metrics
        assert isinstance(finding.metrics["pretrained_classifier_score"], float)
        assert 0.0 <= finding.metrics["pretrained_classifier_score"] <= 1.0
        assert finding.confidence >= 0.50
        assert finding.execution_time_ms > 0.0

    def test_real_model_inference_on_video(self):
        detector = DeepfakeDetector(use_default_classifier=True)
        video_bytes = create_synthetic_video(num_frames=12, with_face=True)
        inp = DetectorInput(
            media_bytes=video_bytes,
            media_content_type="video/mp4",
            media_filename="interview.mp4",
        )

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.metrics["frames_analyzed"] > 1
        assert finding.metrics["faces_detected"] >= 1


class TestMockedClassifierChecks:
    """
    SECTION 3: MOCKED MODEL CHECKS.
    Simulates various neural face-forgery classifier predictions to verify threshold validation,
    blur seam detection, and uncertainty preservation.
    """

    def test_mocked_classifier_high_fake_with_seam_blur(self):
        """Mock: High fake probability (0.90) on portrait with selective boundary feathering."""
        mock_model = MagicMock(spec=PretrainedDeepfakeClassifier)
        mock_model.predict_deepfake.return_value = (0.90, 0.85)

        detector = DeepfakeDetector(classifier=mock_model)
        # Create portrait where face boundary is blurred while background is sharp
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        for y in range(128):
            for x in range(128):
                arr[y, x] = [30, int(((x * 20) ^ (y * 20)) % 180 + 40), 220]
        cy, cx = 64, 64
        ry, rx = 35, 28
        y, x = np.ogrid[:128, :128]
        dist = np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
        for yi in range(128):
            for xi in range(128):
                d = dist[yi, xi]
                if d <= 1.0:
                    alpha = 1.0 if d < 0.70 else (1.0 - (d - 0.70) / 0.30)
                    skin = np.array([215, 145, 110], dtype=np.float32)
                    bg = np.array([140, 140, 140], dtype=np.float32)
                    arr[yi, xi] = (alpha * skin + (1.0 - alpha) * bg).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")

        inp = DetectorInput(media_bytes=buf.getvalue(), media_content_type="image/png")
        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score is not None
        assert finding.manipulation_score >= 0.70

    def test_mocked_classifier_unvalidated_threshold_preserves_uncertainty(self):
        """
        Mock: Model outputs 0.55 (within ambiguity interval [0.40, 0.65]) without boundary seams.
        Rule: Never label AI-generated solely because a model score exceeds an unvalidated threshold.
        Expected: Detector MUST preserve uncertainty (status UNCERTAIN).
        """
        mock_model = MagicMock(spec=PretrainedDeepfakeClassifier)
        mock_model.predict_deepfake.return_value = (0.55, 0.50)

        detector = DeepfakeDetector(classifier=mock_model)
        portrait_bytes = create_portrait_image(with_face=True)
        inp = DetectorInput(media_bytes=portrait_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.UNCERTAIN
        assert finding.confidence <= 0.55
        assert "unvalidated ambiguity interval" in finding.summary
        assert any(e.evidence_type == "MODEL_UNCERTAINTY" for e in evidence)

    def test_global_blur_is_not_falsely_labeled_deepfake(self):
        """
        Physical check: When an entire image is soft/blurry (camera defocus or motion blur),
        low boundary gradient is NOT flagged as a face swap.
        """
        detector = DeepfakeDetector(use_default_classifier=True)
        # Create universally blurred image
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        arr[34:94, 34:94] = [215, 145, 110]
        blurred = cv2.GaussianBlur(arr, (15, 15), 5.0)

        buf = io.BytesIO()
        Image.fromarray(blurred).save(buf, format="PNG")

        inp = DetectorInput(media_bytes=buf.getvalue(), media_content_type="image/png")
        finding, evidence = detector.run(inp)

        # Should not falsely assert high manipulation solely from motion/defocus blur
        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score is not None
        assert finding.manipulation_score <= 0.35


class TestUnavailableModelHandling:
    """
    SECTION 4: UNAVAILABLE MODEL REPORTING (NOT_RUN or ERROR per contract).
    """

    def test_unavailable_required_model_reports_not_run(self):
        detector = DeepfakeDetector(
            classifier=None,
            require_pretrained=True,
            use_default_classifier=False,
        )
        portrait_bytes = create_portrait_image(with_face=True)
        inp = DetectorInput(media_bytes=portrait_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.NOT_RUN
        assert finding.manipulation_score is None
        assert "unavailable" in finding.summary.lower()

    def test_failed_required_model_reports_error(self):
        detector = DeepfakeDetector(
            classifier=None,
            require_pretrained=True,
            use_default_classifier=False,
        )
        detector._model_load_attempted = True
        detector._model_load_error = "Dedicated face-forgery network weights not found"

        portrait_bytes = create_portrait_image(with_face=True)
        inp = DetectorInput(media_bytes=portrait_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.ERROR
        assert finding.manipulation_score is None
        assert "failed to load" in finding.summary.lower()


class TestOperationalBoundaries:
    """
    SECTION 5: OPERATIONAL BOUNDARY & ZERO-FACE INPUTS.
    """

    def test_not_run_when_no_media(self):
        detector = DeepfakeDetector()
        inp = DetectorInput(text="No visual media here")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.NOT_RUN
        assert finding.manipulation_score is None

    def test_uncertain_status_when_no_faces_detected(self):
        detector = DeepfakeDetector()
        landscape_bytes = create_portrait_image(with_face=False)
        inp = DetectorInput(media_bytes=landscape_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.UNCERTAIN
        assert finding.manipulation_score is None
        assert "no identifiable human portrait" in finding.summary.lower()
