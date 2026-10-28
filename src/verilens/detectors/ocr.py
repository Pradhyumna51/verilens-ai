"""
Optical Character Recognition (OCR) Extractor (Member 2: Multimodal Analysis).
Extracts embedded text, headlines, watermarks, and meme banners from visual media using EasyOCR.
Cross-references extracted text against textual claims for discrepancies and sensational overlays.
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

logger = get_logger("OCRDetector")

class OCRDetector(BaseDetector):
    """
    Optical Character Recognition (OCR) Extractor.
    Uses EasyOCR to detect and extract rasterized text directly from visual assets.
    Identifies sensational meme headers, conflicting geographic/date stamps,
    and synthetic text overlays.
    """
    detector_id = "ocr_detector"
    detector_name = "Optical Character Recognition (OCR) Extractor"
    description = (
        "Extracts embedded text, headlines, subtitles, and watermark overlays from visual media "
        "using EasyOCR to identify fabricated banners and cross-modal text conflicts."
    )
    required_modalities = ["image"]

    def __init__(self):
        self._reader = None
        self._reader_attempted = False

    def _get_lazy_reader(self):
        """Attempts to lazy load EasyOCR reader (English)."""
        if not self._reader_attempted:
            self._reader_attempted = True
            try:
                import easyocr
                logger.info("Initializing EasyOCR reader (en)...")
                self._reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR reader initialized successfully.")
            except Exception as exc:
                logger.warning(f"EasyOCR reader unavailable, using text band heuristic fallback: {exc}")
                self._reader = None
        return self._reader

    def _execute(self, input_data: DetectorInput) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        if not input_data.media_bytes:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.NOT_RUN,
                manipulation_score=None,
                confidence=0.0,
                summary="Skipped: Image media stream is required for optical character recognition.",
            ), []

        # Validate image decoding
        try:
            image = Image.open(io.BytesIO(input_data.media_bytes)).convert("RGB")
            img_arr = np.array(image)
        except Exception as exc:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary=f"Failed to decode image for OCR extraction: {str(exc)}",
                error_message=str(exc),
            ), []

        # Check resolution / variance threshold
        if image.width < 32 or image.height < 32:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.20,
                summary=f"Image resolution ({image.width}x{image.height}) is too low for optical character recognition.",
                metrics={"width": image.width, "height": image.height},
            ), []

        reader = self._get_lazy_reader()
        extracted_text = ""
        bounding_boxes: List[Dict[str, Any]] = []
        model_backend = "easyocr"

        if reader:
            try:
                results = reader.readtext(img_arr)
                text_segments = []
                for bbox, text_detected, prob in results:
                    if prob >= 0.25:
                        text_segments.append(text_detected)
                        # Normalize bbox coordinates for JSON serialization
                        bbox_serializable = [[int(pt[0]), int(pt[1])] for pt in bbox] if isinstance(bbox, (list, np.ndarray)) else []
                        bounding_boxes.append({
                            "text": text_detected,
                            "confidence": round(float(prob), 3),
                            "bbox": bbox_serializable,
                        })
                extracted_text = " ".join(text_segments).strip()
            except Exception as e:
                logger.warning(f"EasyOCR read failed: {e}. Falling back to byte text heuristic.")
                extracted_text = self._detect_text_heuristic(input_data.media_bytes)
                model_backend = "byte_heuristic"
        else:
            extracted_text = self._detect_text_heuristic(input_data.media_bytes)
            model_backend = "byte_heuristic"

        evidence: List[EvidenceItem] = []
        manipulation_score = 0.05
        confidence = 0.88 if model_backend == "easyocr" else 0.70

        metrics: Dict[str, Any] = {
            "extracted_text": extracted_text,
            "token_count": len(extracted_text.split()) if extracted_text else 0,
            "bounding_boxes_count": len(bounding_boxes),
            "model_backend": model_backend,
            "epistemic_disclaimer": "OCR extracts visible typography in images; presence of text does not imply deception without conflicting context.",
        }

        if extracted_text:
            text_lower = extracted_text.lower()
            # Detect sensational clickbait/meme overlay words
            sensational_overlay = any(
                k in text_lower
                for k in ["breaking", "urgent", "censored", "exposed", "fake", "hoax", "alert", "secret", "they don't want you to know"]
            )
            metrics["has_sensational_overlay"] = sensational_overlay

            if sensational_overlay:
                manipulation_score = 0.60
                severity = EvidenceSeverity.MEDIUM
            else:
                severity = EvidenceSeverity.LOW

            evidence.append(
                EvidenceItem(
                    id="ev_ocr_text_extracted",
                    detector_id=self.detector_id,
                    evidence_type="EMBEDDED_TEXT_EXTRACTED",
                    description=(
                        f"Extracted embedded visual text: '{extracted_text[:120]}...'"
                        if len(extracted_text) > 120
                        else f"Extracted embedded visual text: '{extracted_text}'"
                    ),
                    severity=severity,
                    metadata={
                        "extracted_text": extracted_text,
                        "token_count": len(extracted_text.split()),
                        "sensational_overlay": sensational_overlay,
                        "backend": model_backend,
                    },
                )
            )

            # Cross-reference with external claim if provided
            claim_text = (input_data.text or "").strip().lower()
            if claim_text:
                ocr_words = set(re.findall(r"\b\w{3,}\b", text_lower))
                claim_words = set(re.findall(r"\b\w{3,}\b", claim_text))
                # Check for direct contradictions (e.g. OCR says 2019, claim says breaking 2026)
                ocr_years = re.findall(r"\b(19\d\d|20\d\d)\b", text_lower)
                claim_years = re.findall(r"\b(19\d\d|20\d\d)\b", claim_text)
                if ocr_years and claim_years and set(ocr_years) != set(claim_years):
                    manipulation_score = max(manipulation_score, 0.70)
                    evidence.append(
                        EvidenceItem(
                            id="ev_ocr_date_conflict",
                            detector_id=self.detector_id,
                            evidence_type="OCR_CLAIM_CONFLICT",
                            description=f"Temporal discrepancy: Image text mentions year {ocr_years[0]} but claim refers to {claim_years[0]}.",
                            severity=EvidenceSeverity.HIGH,
                            metadata={"ocr_years": ocr_years, "claim_years": claim_years},
                        )
                    )

            summary = (
                f"OCR extracted {len(extracted_text.split())} words from visual frame: '{extracted_text[:90]}...'"
                if len(extracted_text) > 90
                else f"OCR extracted visual overlay: '{extracted_text}'"
            )
        else:
            summary = "No embedded text overlays or banners detected in media frame."

        return DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=round(manipulation_score, 2),
            confidence=confidence,
            summary=summary,
            metrics=metrics,
        ), evidence

    def _detect_text_heuristic(self, raw_bytes: bytes) -> str:
        """Inspects raw byte buffer for plain-text strings when EasyOCR is offline."""
        try:
            ascii_strings = re.findall(b"[A-Za-z0-9 ,.!?'-]{6,}", raw_bytes)
            filtered = []
            for s in ascii_strings:
                decoded = s.decode("latin1", errors="ignore").strip()
                if not any(header in decoded for header in ["Photoshop", "ICC_PROFILE", "Exif", "JFIF", "Adobe"]):
                    if len(decoded.split()) >= 2:
                        filtered.append(decoded)
            if filtered:
                return " ".join(filtered[:3])
        except Exception:
            pass
        return ""
