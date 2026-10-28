"""
Unit test suite for Cross-Modal Consistency Analyzer (Member 2: Multimodal Analysis).
Tests CLIP semantic similarity, BLIP generative captioning, environmental alignment,
graceful error handling, and epistemic neutrality (no truth-value claim).
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
from verilens.detectors.consistency import ConsistencyDetector
from verilens.schemas.response import DetectorStatus, EvidenceSeverity

def make_test_image(width=100, height=100, color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Keep small variance so it's not zero-variance, while preserving base color
            arr[y, x] = [
                min(255, max(0, color[0] + ((x + y) % 5))),
                min(255, max(0, color[1] + ((x - y) % 5))),
                min(255, max(0, color[2] + ((x * y) % 5))),
            ]
    img = Image.fromarray(arr)
    img.save(buf, format="PNG")
    return buf.getvalue()

def make_flat_image(width=50, height=50, color=(100, 100, 100)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()

class TestConsistencyDetector:
    @pytest.fixture
    def detector(self):
        return ConsistencyDetector()

    def test_not_run_when_missing_modality(self, detector):
        # Missing text
        img = make_test_image()
        res_no_text, _ = detector.run(DetectorInput(media_bytes=img, media_content_type="image/png"))
        assert res_no_text.status == DetectorStatus.NOT_RUN

        # Missing image
        res_no_img, _ = detector.run(DetectorInput(text="A test claim about an event."))
        assert res_no_img.status == DetectorStatus.NOT_RUN

    def test_uncertain_on_short_or_generic_caption(self, detector):
        img = make_test_image()
        res_short, _ = detector.run(DetectorInput(text="Hi", media_bytes=img, media_content_type="image/png"))
        assert res_short.status == DetectorStatus.UNCERTAIN

        res_generic, _ = detector.run(DetectorInput(text="look at this", media_bytes=img, media_content_type="image/png"))
        assert res_generic.status == DetectorStatus.UNCERTAIN

    def test_uncertain_on_flat_zero_variance_image(self, detector):
        flat_img = make_flat_image()
        res, _ = detector.run(DetectorInput(
            text="A detailed description of an event occurring in a public square.",
            media_bytes=flat_img,
            media_content_type="image/png"
        ))
        assert res.status == DetectorStatus.UNCERTAIN
        assert "zero textural variance" in res.summary

    def test_error_on_corrupt_media_bytes(self, detector):
        res, _ = detector.run(DetectorInput(
            text="Valid claim describing the scene in detail.",
            media_bytes=b"CORRUPT_BYTES_XYZ_123",
            media_content_type="image/png"
        ))
        assert res.status == DetectorStatus.ERROR
        assert res.error_message is not None

    def test_success_with_environmental_luminance_mismatch(self, detector):
        # Bright image (high luminance) with night caption
        bright_img = make_test_image(color=(220, 220, 220))
        inp = DetectorInput(
            text="Midnight darkness covered the silent streets during the midnight operation.",
            media_bytes=bright_img,
            media_content_type="image/png"
        )
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score is not None
        assert finding.manipulation_score >= 0.45

        # Check evidence item
        ev_types = [e.evidence_type for e in evidence]
        assert "CROSS_MODAL_MISMATCH" in ev_types

    def test_epistemic_disclaimer_present(self, detector):
        img = make_test_image()
        inp = DetectorInput(
            text="A group of people standing outside a library building on a clear afternoon.",
            media_bytes=img,
            media_content_type="image/png"
        )
        finding, _ = detector.run(inp)
        assert finding.status == DetectorStatus.SUCCESS
        assert "epistemic_disclaimer" in finding.metrics
        # Verify it explicitly acknowledges semantic similarity does not prove truth/falsity
        disclaimer = finding.metrics["epistemic_disclaimer"]
        assert "does NOT prove" in disclaimer or "not real-world" in disclaimer

    def test_mocked_clip_high_similarity(self, detector):
        from unittest.mock import patch
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_item = MagicMock()
        mock_item.item.return_value = 0.85
        mock_torch.matmul.return_value = [[mock_item]]
        detector._clip_model = mock_model
        detector._clip_processor = mock_processor
        detector._clip_attempted = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            img = make_test_image()
            inp = DetectorInput(
                text="A blue coastal seascape with ocean waves washing onto the shore.",
                media_bytes=img,
                media_content_type="image/png",
            )
            finding, evidence = detector.run(inp)
            assert finding.status == DetectorStatus.SUCCESS
            assert finding.metrics["clip_backend"] == "pretrained_clip"
            ev_types = [e.evidence_type for e in evidence]
            assert "CLIP_SEMANTIC_ALIGNMENT" in ev_types

    def test_mocked_clip_low_similarity_divergence(self, detector):
        from unittest.mock import patch
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_item = MagicMock()
        mock_item.item.return_value = -0.10
        mock_torch.matmul.return_value = [[mock_item]]
        detector._clip_model = mock_model
        detector._clip_processor = mock_processor
        detector._clip_attempted = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            img = make_test_image()
            inp = DetectorInput(
                text="An indoor conference room with a presentation whiteboard and office chairs.",
                media_bytes=img,
                media_content_type="image/png",
            )
            finding, evidence = detector.run(inp)
            assert finding.status == DetectorStatus.SUCCESS
            ev_types = [e.evidence_type for e in evidence]
            assert "CLIP_SEMANTIC_DIVERGENCE" in ev_types
            assert finding.manipulation_score > 0.40
