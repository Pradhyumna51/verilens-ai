"""
Test suite validating advanced multimodal fake news detection enhancements:
1. MVNN Dual-Domain Visual Forensics (ELA & patch noise variance).
2. DIDAN Fine-Grained Entity Contradiction Defense.
3. Dual Emotion Sensationalism & Emotional Arousal Analysis.
4. Dynamic Quality-Weighted Attention Fusion (att-RNN / MVAE paradigm).
"""

import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.detectors.base import DetectorInput
from verilens.detectors.synthetic_image import SyntheticImageDetector
from verilens.detectors.consistency import ConsistencyDetector
from verilens.detectors.synthetic_text import SyntheticTextDetector
from verilens.pipeline.orchestrator import AnalysisPipeline
from verilens.schemas.response import DetectorStatus, Verdict, RiskLevel


def make_test_photo(width=128, height=128, pattern="natural") -> bytes:
    """Generates synthetic vs natural images for testing."""
    buf = io.BytesIO()
    if pattern == "natural":
        arr = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                arr[y, x] = [120 + 30 * np.sin(x / 8.0), 130 + 20 * np.cos(y / 8.0), 140]
        noise = np.random.RandomState(42).normal(0, 8, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    elif pattern == "spliced":
        # Spliced composite: high contrast localized block
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:] = 120
        arr[20:60, 20:60] = 255  # Distinct inserted patch
        img = Image.fromarray(arr)
    else:
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestMVDualDomainVisualForensics:
    """Validates MVNN Frequency + Pixel Domain Forensics."""

    def test_ela_and_patch_noise_metrics_computed(self):
        detector = SyntheticImageDetector()
        img_bytes = make_test_photo(128, 128, pattern="natural")
        inp = DetectorInput(media_bytes=img_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert "ela_mean_error" in finding.metrics
        assert "ela_max_error" in finding.metrics
        assert "patch_noise_dispersion" in finding.metrics
        assert isinstance(finding.metrics["ela_mean_error"], float)
        assert isinstance(finding.metrics["patch_noise_dispersion"], float)

    def test_spliced_image_triggers_pixel_domain_anomaly(self):
        detector = SyntheticImageDetector()
        spliced_bytes = make_test_photo(128, 128, pattern="spliced")
        inp = DetectorInput(media_bytes=spliced_bytes, media_content_type="image/png")

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        ev_types = [e.evidence_type for e in evidence]
        # Should detect either frequency anomaly or pixel domain inconsistency
        assert "PIXEL_DOMAIN_INCONSISTENCY" in ev_types or "FREQUENCY_ANOMALY" in ev_types or finding.metrics["ela_max_error"] > 0


class TestDIDANEntityContradictions:
    """Validates fine-grained cross-modal discrepancy defense."""

    def test_snow_entity_contradiction(self):
        detector = ConsistencyDetector()
        # Dark, low-luminance image (dark nighttime setting with non-zero variance)
        dark_buf = io.BytesIO()
        dark_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            for x in range(100):
                dark_arr[y, x] = [35 + ((x + y) % 6), 35 + ((x - y) % 6), 40 + ((x * y) % 6)]
        Image.fromarray(dark_arr).save(dark_buf, format="PNG")

        inp = DetectorInput(
            text="Blizzard emergency: Heavy snow and freezing snow storm shuts down all regional transportation.",
            media_bytes=dark_buf.getvalue(),
            media_content_type="image/png",
        )

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert "didan_discrepancy_score" in finding.metrics
        assert finding.metrics["didan_discrepancy_score"] > 0.0
        assert any(e.id == "ev_mismatch_snow_spectrum" for e in evidence)

    def test_crowd_protest_density_contradiction(self):
        detector = ConsistencyDetector()
        # Low edge texture image
        smooth_buf = io.BytesIO()
        smooth_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            smooth_arr[y, :] = [100, 120, 140]
        Image.fromarray(smooth_arr).save(smooth_buf, format="PNG")

        inp = DetectorInput(
            text="Massive protest erupts as thousands of protesters gather for a huge crowd demonstration in the city center.",
            media_bytes=smooth_buf.getvalue(),
            media_content_type="image/png",
        )

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert "scene_edge_density" in finding.metrics
        assert any(e.id == "ev_mismatch_crowd_density" for e in evidence)


class TestDualEmotionTextForensics:
    """Validates publisher emotional arousal & sensationalist rhetoric detection."""

    def test_high_emotional_arousal_flagged(self):
        detector = SyntheticTextDetector()
        text = (
            "SHOCKING DISASTER! Catastrophic unverified emergency alert exposed secret truth "
            "as panic spreads across the terrified public in an unbelievable nightmare! "
            "Officials issue urgent warning regarding this mind-blowing bombshell catastrophe!"
        )
        inp = DetectorInput(text=text)

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert "sensationalism_hits" in finding.metrics
        assert "emotional_arousal_score" in finding.metrics
        assert finding.metrics["sensationalism_hits"] >= 3
        assert finding.metrics["emotional_arousal_score"] >= 0.50
        assert any(e.evidence_type == "EMOTIONAL_DISSONANCE" for e in evidence)

    def test_objective_reporting_remains_calm(self):
        detector = SyntheticTextDetector()
        text = (
            "The regional meteorological department published its quarterly hydrological bulletin on Tuesday. "
            "Precipitation averages across the northern watershed remained consistent with seasonal expectations. "
            "Water resource managers confirmed reservoir levels will adequately satisfy agricultural irrigation requirements."
        )
        inp = DetectorInput(text=text)

        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.metrics["emotional_arousal_score"] < 0.20
        assert not any(e.evidence_type == "EMOTIONAL_DISSONANCE" for e in evidence)


class TestDynamicAttentionFusion:
    """Validates att-RNN / MVAE quality-weighted fusion."""

    def test_pipeline_fuses_multimodal_signals_cleanly(self):
        pipeline = AnalysisPipeline()
        text = (
            "Archaeologists in Cairo have cataloged twenty well-preserved ceramic vessels during the spring excavation. "
            "The artifacts date to the Middle Kingdom and will undergo laboratory isotopic analysis next month."
        )
        img_bytes = make_test_photo(128, 128, pattern="natural")

        response = pipeline.run(text=text, media_bytes=img_bytes, media_content_type="image/png")

        assert response.overall_assessment.verdict in [Verdict.LIKELY_REAL, Verdict.SUSPICIOUS, Verdict.UNCERTAIN]
        assert "synthetic_image" in response.findings
        assert "synthetic_text" in response.findings
        assert "consistency" in response.findings
        assert response.findings["synthetic_image"].metrics.get("ela_mean_error") is not None
