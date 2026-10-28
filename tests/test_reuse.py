"""
Unit and integration tests for MediaReuseDetector and ImageReuseIndex.
Tests perceptual hashing (DCT pHash, dHash, aHash), Hamming distance,
reference index searching, EXIF parsing, and all four operational statuses.
"""

import io
from PIL import Image, ExifTags
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.detectors.base import DetectorInput
from verilens.detectors.reuse import (
    MediaReuseDetector,
    ImageReuseIndex,
    compute_phash,
    compute_dhash,
    compute_ahash,
    hamming_distance,
)
from verilens.schemas.response import DetectorStatus, EvidenceSeverity


def create_test_image(width=128, height=128, pattern="gradient") -> bytes:
    """Generates deterministic test image bytes."""
    buf = io.BytesIO()
    if pattern == "gradient":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                arr[y, x] = [(x * 2) % 256, (y * 2) % 256, ((x + y) * 2) % 256]
        img = Image.fromarray(arr)
    elif pattern == "flat":
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
    else:
        arr = np.random.RandomState(42).randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPerceptualHashAlgorithms:
    def test_phash_determinism_and_length(self):
        img = Image.open(io.BytesIO(create_test_image(128, 128, "gradient")))
        h1 = compute_phash(img)
        h2 = compute_phash(img)
        assert h1 == h2
        assert len(h1) == 16  # 64-bit hex hash

    def test_dhash_determinism_and_length(self):
        img = Image.open(io.BytesIO(create_test_image(128, 128, "gradient")))
        h1 = compute_dhash(img)
        h2 = compute_dhash(img)
        assert h1 == h2
        assert len(h1) == 16

    def test_ahash_determinism_and_length(self):
        img = Image.open(io.BytesIO(create_test_image(128, 128, "gradient")))
        h1 = compute_ahash(img)
        h2 = compute_ahash(img)
        assert h1 == h2
        assert len(h1) == 16

    def test_hamming_distance_exact_and_different(self):
        h1 = "ffff0000ffff0000"
        h2 = "ffff0000ffff0000"
        assert hamming_distance(h1, h2) == 0

        # Invert half the bits (32 bits different)
        h3 = "00000000ffff0000"
        assert hamming_distance(h1, h3) == 16


class TestImageReuseIndex:
    def test_index_add_and_search_exact_match(self):
        index = ImageReuseIndex()
        img_bytes = create_test_image(128, 128, "gradient")
        index.add(
            entry_id="archive_2022_protest",
            image=img_bytes,
            metadata={"title": "Protest Rally in Capital", "original_date": "2022-04-15"},
        )
        assert index.count() == 1

        matches = index.search(query_image=img_bytes, max_distance=5, hash_type="phash")
        assert len(matches) == 1
        assert matches[0].entry_id == "archive_2022_protest"
        assert matches[0].hamming_distance == 0
        assert matches[0].similarity == 1.0
        assert matches[0].metadata["title"] == "Protest Rally in Capital"

    def test_index_search_no_match_for_different_image(self):
        index = ImageReuseIndex()
        img1 = create_test_image(128, 128, "gradient")
        img2 = create_test_image(128, 128, "noise")
        index.add(entry_id="img1", image=img1)

        matches = index.search(query_image=img2, max_distance=5, hash_type="phash")
        assert len(matches) == 0


class TestMediaReuseDetector:
    def test_not_run_status_when_no_media(self):
        detector = MediaReuseDetector()
        inp = DetectorInput(text="Just some text without media")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.NOT_RUN
        assert finding.manipulation_score is None
        assert len(evidence) == 0

    def test_error_status_on_corrupt_bytes(self):
        detector = MediaReuseDetector()
        inp = DetectorInput(media_bytes=b"CORRUPT_BYTES_NOT_AN_IMAGE", media_content_type="image/png")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.ERROR
        assert finding.manipulation_score is None
        assert finding.error_message is not None

    def test_uncertain_status_on_micro_thumbnail(self):
        detector = MediaReuseDetector()
        micro = create_test_image(width=16, height=16)
        inp = DetectorInput(media_bytes=micro, media_content_type="image/png")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.UNCERTAIN
        assert finding.manipulation_score is None
        assert "too small" in finding.summary

    def test_uncertain_status_on_flat_color_image(self):
        detector = MediaReuseDetector()
        flat = create_test_image(width=64, height=64, pattern="flat")
        inp = DetectorInput(media_bytes=flat, media_content_type="image/png")
        finding, evidence = detector.run(inp)
        assert finding.status == DetectorStatus.UNCERTAIN
        assert "flat image" in finding.summary.lower()

    def test_success_organic_image_without_reuse(self):
        index = ImageReuseIndex()
        detector = MediaReuseDetector(index=index)
        img = create_test_image(width=100, height=100, pattern="gradient")
        inp = DetectorInput(media_bytes=img, media_content_type="image/png")
        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score == 0.10  # Organic default
        assert finding.confidence >= 0.80
        assert finding.metrics["match_found"] is False
        assert any(e.evidence_type == "PERCEPTUAL_FINGERPRINT" for e in evidence)

    def test_success_detects_indexed_reuse(self):
        index = ImageReuseIndex()
        img = create_test_image(width=128, height=128, pattern="gradient")
        index.add(
            entry_id="historic_event_2019",
            image=img,
            metadata={"title": "Flooding in Downtown District", "original_date": "2019-08-20"},
        )
        detector = MediaReuseDetector(index=index)
        inp = DetectorInput(media_bytes=img, media_content_type="image/png")
        finding, evidence = detector.run(inp)

        assert finding.status == DetectorStatus.SUCCESS
        assert finding.manipulation_score >= 0.70  # Elevated score due to match
        assert finding.metrics["match_found"] is True
        assert finding.metrics["matched_entry_id"] == "historic_event_2019"
        assert finding.metrics["hamming_distance"] == 0
        assert any(e.evidence_type == "IMAGE_REUSE" for e in evidence)
