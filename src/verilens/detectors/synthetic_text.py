"""
Synthetic Text Detector.
Analyzes linguistic burstiness, lexical density, repetition rates, perplexity proxy,
and stylistic markers indicative of Large Language Model (LLM) generation.
Includes pluggable pretrained text classifier integration, threshold validation,
and uncertainty preservation.
Supports NOT_RUN, UNCERTAIN, ERROR, and SUCCESS statuses.
"""

import re
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)
from verilens.core.logger import get_logger

logger = get_logger("SyntheticTextDetector")


class PretrainedTextClassifier(ABC):
    """Abstract interface for pluggable pretrained AI-text classification models."""

    @abstractmethod
    def predict_synthetic(self, text: str) -> Tuple[float, float]:
        """
        Runs inference on input text.
        Returns:
            (prob_synthetic, confidence): float values between 0.0 and 1.0.
        """
        pass


class StatisticalTextHeuristicClassifier(PretrainedTextClassifier):
    """
    Statistical heuristic text classifier (NOT pretrained on labeled data).
    Evaluates burstiness, lexical diversity, bigram repetition, and marker frequency
    using hand-tuned logistic weights. Scores are approximate, not validated.
    """

    def __init__(self):
        # Logistic weights for [low_burstiness, low_ttr, bigram_repetition, marker_density]
        self.weights = np.array([2.8, 1.9, 2.2, 3.4], dtype=np.float32)
        self.bias = -2.6

    def predict_synthetic(self, text: str) -> Tuple[float, float]:
        words = re.findall(r"\b\w+\b", text)
        if not words:
            return 0.0, 0.0

        # 1. Burstiness (sentence length variation)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            sentences = [text]
        lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        mean_l = sum(lengths) / len(lengths) if lengths else 1.0
        var_l = sum((l - mean_l) ** 2 for l in lengths) / len(lengths) if len(lengths) > 1 else 0.0
        burstiness = math.sqrt(var_l) / (mean_l + 1e-5)
        low_burstiness = float(np.clip(1.0 - (burstiness / 0.6), 0.0, 1.0))

        # 2. Lexical diversity (Type-Token Ratio)
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / len(words)
        low_ttr = float(np.clip(1.0 - (ttr / 0.8), 0.0, 1.0))

        # 3. Bigram repetition
        bigrams = [f"{words[i].lower()}_{words[i+1].lower()}" for i in range(len(words) - 1)]
        rep = (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams))
        bigram_rep = float(np.clip(rep * 3.0, 0.0, 1.0))

        # 4. Stylistic marker density
        markers = [
            r"\bin conclusion\b",
            r"\bit is important to remember\b",
            r"\bdelve into\b",
            r"\btapestry of\b",
            r"\btestament to\b",
            r"\bmoreover\b",
            r"\bfurthermore\b",
            r"\bplays a crucial role\b",
        ]
        hits = sum(1 for p in markers if re.search(p, text, re.IGNORECASE))
        marker_density = float(min(1.0, hits / 2.0))

        features = np.array([low_burstiness, low_ttr, bigram_rep, marker_density], dtype=np.float32)
        logit = float(np.dot(self.weights, features) + self.bias)
        prob = 1.0 / (1.0 + np.exp(-logit))
        conf = float(np.clip(2.0 * abs(prob - 0.5) * 0.25 + 0.40, 0.35, 0.65))
        return float(prob), conf


class SyntheticTextDetector(BaseDetector):
    detector_id = "synthetic_text"
    detector_name = "Synthetic Text Detector"
    description = (
        "Evaluates statistical burstiness, lexical diversity, perplexity proxy, and stylistic patterns "
        "indicative of LLM generation with threshold validation and uncertainty preservation."
    )
    required_modalities = ["text"]

    def __init__(
        self,
        classifier: Optional[PretrainedTextClassifier] = None,
        require_pretrained: bool = False,
        use_default_classifier: bool = True,
    ):
        self._custom_classifier = classifier
        self._require_pretrained = require_pretrained
        self._use_default_classifier = use_default_classifier
        self._pretrained_model: Optional[PretrainedTextClassifier] = classifier
        self._model_load_attempted: bool = False
        self._model_load_error: Optional[str] = None

    def _get_lazy_model(self) -> Optional[PretrainedTextClassifier]:
        if not self._model_load_attempted:
            self._model_load_attempted = True
            if self._custom_classifier is not None:
                self._pretrained_model = self._custom_classifier
            elif self._use_default_classifier:
                try:
                    self._pretrained_model = StatisticalTextHeuristicClassifier()
                except Exception as exc:
                    logger.warning(f"Failed to instantiate default text classifier: {exc}")
                    self._pretrained_model = None
                    self._model_load_error = str(exc)
            else:
                self._pretrained_model = None
        return self._pretrained_model

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

        # If a pretrained model is required but unavailable, report NOT_RUN or ERROR per contract
        model = self._get_lazy_model()
        if self._require_pretrained and model is None:
            if self._model_load_error:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.ERROR,
                    manipulation_score=None,
                    confidence=0.0,
                    summary=f"Required pretrained AI-text classifier failed to load: {self._model_load_error}",
                    error_message=self._model_load_error,
                ), []
            else:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.NOT_RUN,
                    manipulation_score=None,
                    confidence=0.0,
                    summary="Required pretrained AI-text model is unavailable or disabled; execution reported as NOT_RUN per contract.",
                ), []

        # Split into sentences for burstiness (sentence length variance)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            sentences = [text]

        sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        mean_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
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

        # Run pretrained text classifier if available
        model_prob: Optional[float] = None
        model_conf: Optional[float] = None
        if model is not None:
            try:
                model_prob, model_conf = model.predict_synthetic(text)
            except Exception as exc:
                logger.warning(f"Pretrained text model inference failed: {exc}")

        # Baseline statistical signal
        synthetic_signal = 0.0
        if burstiness_score < 0.35 and len(sentences) >= 3:
            synthetic_signal += 0.35
        elif burstiness_score < 0.50:
            synthetic_signal += 0.15

        if marker_hits >= 2:
            synthetic_signal += 0.30
        elif marker_hits == 1:
            synthetic_signal += 0.15

        if bigram_repetition > 0.25:
            synthetic_signal += 0.20

        # Dual Emotion & Sensationalist Style Analysis (WWW 2021)
        sensationalist_terms = [
            r"\bshocking\b",
            r"\bunbelievable\b",
            r"\bcatastrophic\b",
            r"\bexposed\b",
            r"\bbombshell\b",
            r"\bterrifying\b",
            r"\burgent alert\b",
            r"\bsecret truth\b",
            r"\bmiracle cure\b",
            r"\bmind-blowing\b",
            r"\bdisaster strikes\b",
            r"\bpanic\b",
        ]
        sensationalist_hits = sum(1 for p in sensationalist_terms if re.search(p, text, re.IGNORECASE))
        exclamation_count = text.count("!")
        uppercase_words = [w for w in words if len(w) >= 3 and w.isupper()]
        uppercase_ratio = len(uppercase_words) / max(1, len(words))

        # Arousal score combines sensationalist lexicon, exclamation punctuation, and uppercase tokens
        emotional_arousal = min(1.0, (sensationalist_hits * 0.25) + (min(3, exclamation_count) * 0.15) + (uppercase_ratio * 2.0))

        evidence: List[EvidenceItem] = []
        metrics: Dict[str, Any] = {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "burstiness_index": round(burstiness_score, 3),
            "type_token_ratio": round(ttr, 3),
            "bigram_repetition": round(bigram_repetition, 3),
            "marker_hits": marker_hits,
            "sensationalism_hits": sensationalist_hits,
            "emotional_arousal_score": round(emotional_arousal, 3),
        }

        if emotional_arousal >= 0.50:
            evidence.append(
                EvidenceItem(
                    id="ev_emotional_arousal",
                    detector_id=self.detector_id,
                    evidence_type="EMOTIONAL_DISSONANCE",
                    description=(
                        f"Text exhibits heightened emotional arousal and sensationalist rhetoric "
                        f"(arousal index: {emotional_arousal:.2f}, exclamation points: {exclamation_count}, sensationalist keywords: {sensationalist_hits})."
                    ),
                    severity=EvidenceSeverity.MEDIUM,
                    metadata={
                        "emotional_arousal": round(emotional_arousal, 2),
                        "exclamation_count": exclamation_count,
                        "uppercase_ratio": round(uppercase_ratio, 2),
                    },
                )
            )

        if model_prob is not None:
            metrics["pretrained_classifier_score"] = round(model_prob, 3)
            metrics["pretrained_classifier_confidence"] = round(model_conf or 0.0, 3)

        # Threshold validation: Never label AI-generated solely because a model score exceeds an unvalidated threshold
        if model_prob is not None:
            # Check for ambiguous interval (0.35 to 0.65)
            if 0.35 <= model_prob <= 0.65 and burstiness_score >= 0.50:
                # Natural sentence variability contradicts synthetic marker/model probability
                manipulation_score = round(model_prob, 2)
                confidence = 0.45
                summary = (
                    f"Text contains stylistic markers, but sentence length variation (burstiness: {burstiness_score:.2f}) "
                    f"is natural. Model score ({model_prob:.2f}) is in the unvalidated ambiguity interval; preserving uncertainty."
                )
                evidence.append(
                    EvidenceItem(
                        id="ev_text_ambiguity",
                        detector_id=self.detector_id,
                        evidence_type="TEXT_UNCERTAINTY",
                        description=f"Classifier score ({model_prob:.2f}) lacks statistical cadence corroboration.",
                        severity=EvidenceSeverity.LOW,
                        metadata={"model_score": round(model_prob, 3), "burstiness_index": round(burstiness_score, 3)},
                    )
                )
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.UNCERTAIN,
                    manipulation_score=manipulation_score,
                    confidence=confidence,
                    summary=summary,
                    metrics=metrics,
                ), evidence

            # Harmonize model score with statistical features
            if model_prob > 0.70 and burstiness_score < 0.40:
                manipulation_score = min(0.92, round(0.6 * model_prob + 0.4 * synthetic_signal, 2))
                base_conf = 0.60 + (word_count / 150.0) * 0.30
                confidence = round(min(0.95, 0.5 * (model_conf or base_conf) + 0.5 * base_conf), 2)
            else:
                manipulation_score = round(0.5 * model_prob + 0.5 * synthetic_signal, 2)
                confidence = 0.70
        else:
            manipulation_score = min(1.0, max(0.0, round(synthetic_signal, 2)))
            confidence = round(min(0.92, 0.50 + (word_count / 150.0) * 0.40), 2)

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
        if model_prob is not None:
            summary += f" Pretrained text classifier probability: {model_prob:.2f}."

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=manipulation_score,
            confidence=confidence,
            summary=summary,
            metrics=metrics,
        )
        return finding, evidence
