"""
Out-of-Context Media Reuse Detector.
Computes perceptual visual fingerprints (dHash/aHash) and extracts EXIF provenance telemetry.
Supports NOT_RUN, UNCERTAIN, ERROR, and SUCCESS statuses.
"""

import io
from typing import List, Tuple, Dict, Any
from PIL import Image, ExifTags
import numpy as np

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)

class MediaReuseDetector(BaseDetector):
    detector_id = "media_reuse"
    detector_name = "Out-of-Context Media Reuse Detector"
    description = "Generates perceptual hashes (dHash, aHash) and examines EXIF provenance to flag recycled or edited media."
    required_modalities = ["image"]

    @staticmethod
    def _compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
        """Difference hash for visual similarity matching."""
        resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(np.asarray(resized).flatten())
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                difference.append(pixel_left > pixel_right)
        decimal_val = 0
        hex_string = []
        for index, val in enumerate(difference):
            if val:
                decimal_val += 2 ** (index % 4)
            if index % 4 == 3:
                hex_string.append(hex(decimal_val)[2:])
                decimal_val = 0
        return "".join(hex_string)

    @staticmethod
    def _compute_ahash(image: Image.Image, hash_size: int = 8) -> str:
        """Average hash for coarse visual similarity matching."""
        resized = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32)
        avg = float(np.mean(arr))
        pixels = arr.flatten()
        bits = "".join(["1" if p > avg else "0" for p in pixels])
        return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"

    def _execute(self, input_data: DetectorInput) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        if not input_data.media_bytes:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.NOT_RUN,
                manipulation_score=None,
                confidence=0.0,
                summary="No visual media provided for reuse inspection.",
            ), []

        try:
            image = Image.open(io.BytesIO(input_data.media_bytes))
            width, height = image.size
        except Exception as exc:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary=f"Unable to decode image bytes: {str(exc)}",
                error_message=str(exc),
            ), []

        # Resolution boundary check
        if width < 32 or height < 32:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.20,
                summary=f"Image dimensions ({width}x{height}) are too small for reliable provenance hashing.",
                metrics={"width": width, "height": height},
            ), []

        dhash_val = self._compute_dhash(image)
        ahash_val = self._compute_ahash(image)

        # Inspect EXIF metadata
        exif_data: Dict[str, Any] = {}
        editing_software_detected = False
        software_name = ""
        try:
            raw_exif = image.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, int, float)):
                        exif_data[tag_name] = str(value)
                        if tag_name.lower() in ("software", "processingsoftware"):
                            editing_software_detected = True
                            software_name = str(value)
        except Exception:
            pass  # EXIF parsing failure is non-fatal

        evidence: List[EvidenceItem] = []
        evidence.append(
            EvidenceItem(
                id="ev_perceptual_hash",
                detector_id=self.detector_id,
                evidence_type="PERCEPTUAL_FINGERPRINT",
                description=f"Generated dHash: {dhash_val} and aHash: {ahash_val}.",
                severity=EvidenceSeverity.LOW,
                metadata={"dhash": dhash_val, "ahash": ahash_val, "resolution": f"{width}x{height}"},
            )
        )

        manipulation_score = 0.10
        summary_points = [f"Computed perceptual visual fingerprints ({width}x{height})."]

        if editing_software_detected:
            manipulation_score += 0.40
            summary_points.append(f"Detected editor signature: {software_name}.")
            evidence.append(
                EvidenceItem(
                    id="ev_exif_software",
                    detector_id=self.detector_id,
                    evidence_type="METADATA",
                    description=f"EXIF metadata indicates manipulation or export via {software_name}.",
                    severity=EvidenceSeverity.MEDIUM,
                    metadata={"software": software_name},
                )
            )

        if not exif_data:
            summary_points.append("EXIF metadata stripped (common in social media republishing).")

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=round(min(1.0, manipulation_score), 2),
            confidence=0.85,
            summary=" ".join(summary_points),
            metrics={
                "dhash": dhash_val,
                "ahash": ahash_val,
                "exif_tags_found": len(exif_data),
                "has_exif": bool(exif_data),
                "editing_software_detected": editing_software_detected,
                "resolution": f"{width}x{height}",
            },
        )
        return finding, evidence
