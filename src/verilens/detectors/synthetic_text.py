"""
Synthetic Text Detector.
Analyzes linguistic burstiness, lexical density, repetition rates, and stylistic markers.
Supports NOT_RUN, UNCERTAIN, ERROR, and SUCCESS statuses.
"""

import re
import math
from typing import List, Tuple
from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)

class SyntheticTextDetector(BaseDetector):
    detector_id = "synthetic_text"
    detector_name = "Synthetic Text Detector"
    description = "Evaluates statistical burstiness, lexical diversity, and repetitive structure indicative of LLM generation."
    required_modalities = ["text"]

    def _execute(self, input_data: DetectorInput) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        text = (input_data.text or "").strip()
        
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        
        # Operational boundary check: if text is too short, return UNCERTAIN
        if word_count < 15:
            finding = DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.35,
                summary=f"Input length ({word_count} words) is below the threshold (15 words) for statistical burstiness analysis.",
                metrics={"word_count": word_count, "min_recommended_words": 15},
            )
            return finding, []

        # Split into sentences for burstiness (sentence length variance)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            sentences = [text]

        sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        mean_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        variance = (
            sum((l - mean_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            if len(sentence_lengths) > 1 else 0.0
        )
        std_dev = math.sqrt(variance)
        burstiness_score = std_dev / (mean_len + 1e-5)  # Human writing is usually bursty (> 0.5)

        # Lexical diversity (Type-Token Ratio)
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / word_count

        # Repetitive bigram count
        bigrams = [f"{words[i].lower()}_{words[i+1].lower()}" for i in range(len(words) - 1)]
        bigram_repetition = (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams))

        # Check for typical synthetic hedging patterns
        synthetic_markers = [
            r"\bin conclusion\b",
            r"\bit is important to remember\b",
            r"\bdelve into\b",
            r"\btapestry of\b",
            r"\btestament to\b",
            r"\bmoreover\b",
            r"\bfurthermore\b",
            r"\bplays a crucial role\b",
        ]
        marker_hits = sum(1 for p in synthetic_markers if re.search(p, text, re.IGNORECASE))

        # Synthetic likelihood calculation based on lower burstiness + uniform sentence lengths + marker hits
        # Lower burstiness (< 0.35) and low lexical diversity correlate with synthetic text
        synthetic_signal = 0.0
        if burstiness_score < 0.35 and len(sentences) >= 3:
            synthetic_signal += 0.35
        elif burstiness_score < 0.5:
            synthetic_signal += 0.15

        if marker_hits >= 2:
            synthetic_signal += 0.30
        elif marker_hits == 1:
            synthetic_signal += 0.15

        if bigram_repetition > 0.25:
            synthetic_signal += 0.20

        # Normalization
        manipulation_score = min(1.0, max(0.0, round(synthetic_signal, 2)))
        confidence = round(min(0.92, 0.50 + (word_count / 150.0) * 0.40), 2)

        evidence: List[EvidenceItem] = []
        if marker_hits > 0:
            evidence.append(
                EvidenceItem(
                    id="ev_text_markers",
                    detector_id=self.detector_id,
                    evidence_type="TEXT_STATISTIC",
                    description=f"Identified {marker_hits} common generative transition marker(s) in body text.",
                    severity=EvidenceSeverity.MEDIUM if marker_hits >= 2 else EvidenceSeverity.LOW,
                    metadata={"marker_hits": marker_hits},
                )
            )

        if burstiness_score < 0.35 and len(sentences) >= 3:
            evidence.append(
                EvidenceItem(
                    id="ev_text_burstiness",
                    detector_id=self.detector_id,
                    evidence_type="TEXT_STATISTIC",
                    description=f"Low sentence-length variation (burstiness coefficient: {burstiness_score:.2f}) indicates potential machine cadence.",
                    severity=EvidenceSeverity.MEDIUM,
                    metadata={"burstiness_score": round(burstiness_score, 2), "sentence_count": len(sentences)},
                )
            )

        summary = (
            f"Evaluated {word_count} words across {len(sentences)} sentences. "
            f"Burstiness index: {burstiness_score:.2f}, Lexical diversity: {ttr:.2f}."
        )

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=manipulation_score,
            confidence=confidence,
            summary=summary,
            metrics={
                "word_count": word_count,
                "sentence_count": len(sentences),
                "burstiness_index": round(burstiness_score, 3),
                "type_token_ratio": round(ttr, 3),
                "marker_hits": marker_hits,
            },
        )
        return finding, evidence
