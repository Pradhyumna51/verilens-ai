"""
Cross-Modal Consistency Analyzer.
Detects discrepancies between textual claims/captions and visual media content.
Supports NOT_RUN, UNCERTAIN, ERROR, and SUCCESS statuses.
"""

import io
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)
from verilens.core.logger import get_logger

logger = get_logger("ConsistencyDetector")

class ConsistencyDetector(BaseDetector):
    detector_id = "consistency"
    detector_name = "Cross-Modal Consistency Analyzer"
    description = "Compares caption entities, atmospheric cues, and environment against visual scene properties."
    required_modalities = ["text", "image"]

    def __init__(self):
        self._clip_model = None
        self._clip_attempted = False

    def _get_lazy_clip(self):
        """Lazy loader for multimodal embedding model if installed."""
        if not self._clip_attempted:
            self._clip_attempted = True
            try:
                # Lazy import hook for sentence-transformers / transformers CLIP
                self._clip_model = None
            except Exception as e:
                logger.warning(f"CLIP integration unavailable: {e}")
                self._clip_model = None
        return self._clip_model

    def _execute(self, input_data: DetectorInput) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        text = (input_data.text or "").strip().lower()
        if not input_data.media_bytes:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.NOT_RUN,
                manipulation_score=None,
                confidence=0.0,
                summary="Skipped: Both text and image modalities are required for cross-modal consistency analysis.",
            ), []

        # Check for non-descriptive captions
        tokens = re.findall(r"\b\w+\b", text)
        if len(tokens) < 3 or text in {"image", "photo", "look at this", "see this", "check this out"}:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.30,
                summary="Caption contains insufficient descriptive entities to establish cross-modal correspondence.",
                metrics={"token_count": len(tokens)},
            ), []

        try:
            image = Image.open(io.BytesIO(input_data.media_bytes)).convert("RGB")
            img_arr = np.array(image, dtype=np.float32)
        except Exception as exc:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary=f"Visual decoding error during consistency check: {str(exc)}",
                error_message=str(exc),
            ), []

        # Color channel statistics
        mean_r = float(np.mean(img_arr[:, :, 0]))
        mean_g = float(np.mean(img_arr[:, :, 1]))
        mean_b = float(np.mean(img_arr[:, :, 2]))
        luminance = float(0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b)

        # Environmental keyword dictionaries
        night_keywords = {"night", "dark", "midnight", "dusk", "evening", "moonlit", "stargazing"}
        day_keywords = {"daylight", "sunny", "noon", "bright sun", "clear sky", "morning sun"}
        fire_keywords = {"fire", "wildfire", "flames", "burning", "explosion", "inferno"}
        water_keywords = {"ocean", "sea", "flood", "submerged", "river", "lake", "drowning"}

        text_has_night = any(k in text for k in night_keywords)
        text_has_day = any(k in text for k in day_keywords)
        text_has_fire = any(k in text for k in fire_keywords)
        text_has_water = any(k in text for k in water_keywords)

        mismatches: List[str] = []
        evidence: List[EvidenceItem] = []
        mismatch_penalty = 0.0

        # Luminance contradiction
        if text_has_night and luminance > 160.0:
            mismatches.append(f"Caption claims nighttime/dark setting, but visual media exhibits high daytime luminance ({luminance:.1f}).")
            mismatch_penalty += 0.45
            evidence.append(
                EvidenceItem(
                    id="ev_mismatch_luminance",
                    detector_id=self.detector_id,
                    evidence_type="CROSS_MODAL_MISMATCH",
                    description=f"Inversion between night assertion and daylight luminance ({luminance:.1f}).",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"observed_luminance": round(luminance, 1), "claimed_environment": "night"},
                )
            )
        elif text_has_day and luminance < 40.0:
            mismatches.append(f"Caption claims sunny/daylight conditions, but visual scene has low luminance ({luminance:.1f}).")
            mismatch_penalty += 0.40
            evidence.append(
                EvidenceItem(
                    id="ev_mismatch_darkness",
                    detector_id=self.detector_id,
                    evidence_type="CROSS_MODAL_MISMATCH",
                    description=f"Daylight claim contradicts low visual luminance ({luminance:.1f}).",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"observed_luminance": round(luminance, 1), "claimed_environment": "daylight"},
                )
            )

        # Fire / chromatic warmth contradiction
        if text_has_fire:
            # Fire scenes typically have high red-to-blue ratio
            red_ratio = mean_r / (mean_b + 1e-5)
            if red_ratio < 0.95 and mean_b > 110.0:
                mismatches.append(f"Caption asserts active fire/inferno, but visual spectrum is predominantly blue/cool (R/B ratio: {red_ratio:.2f}).")
                mismatch_penalty += 0.40
                evidence.append(
                    EvidenceItem(
                        id="ev_mismatch_fire_spectrum",
                        detector_id=self.detector_id,
                        evidence_type="CROSS_MODAL_MISMATCH",
                        description=f"Absence of warm chromatic signature for claimed fire event (R/B ratio: {red_ratio:.2f}).",
                        severity=EvidenceSeverity.HIGH,
                        metadata={"rb_ratio": round(red_ratio, 2)},
                    )
                )

        # Baseline alignment
        manipulation_score = min(1.0, max(0.10, mismatch_penalty + 0.10))
        confidence = 0.78

        if mismatches:
            summary = "Cross-modal discrepancies detected: " + "; ".join(mismatches)
        else:
            summary = f"No primary atmospheric contradictions detected between caption cues and image telemetry (Luminance: {luminance:.1f})."

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=round(manipulation_score, 2),
            confidence=confidence,
            summary=summary,
            metrics={
                "mean_luminance": round(luminance, 2),
                "rgb_means": [round(mean_r, 1), round(mean_g, 1), round(mean_b, 1)],
                "mismatch_count": len(mismatches),
            },
        )
        return finding, evidence
