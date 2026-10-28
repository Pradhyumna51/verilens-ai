"""
Multimodal Pipeline Orchestrator for VeriLens AI.
Coordinates detector execution, aggregates findings, synthesizes evidence, 
and generates natural language explanations with transparent limitations.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from verilens.detectors.base import DetectorInput
from verilens.detectors.registry import registry
from verilens.pipeline.reasoner import reasoner, MultimodalReasoner
from verilens.pipeline.fact_check import fact_checker, FactChecker
from verilens.schemas.response import (
    DetectorStatus,
    Verdict,
    RiskLevel,
    EvidenceSeverity,
    EvidenceItem,
    DetectorFinding,
    OverallAssessment,
    Explanation,
    Limitations,
    AnalysisResponse,
)
from verilens.core.logger import get_logger

logger = get_logger("Orchestrator")

class AnalysisPipeline:
    def __init__(
        self,
        custom_reasoner: Optional[MultimodalReasoner] = None,
        custom_fact_checker: Optional[FactChecker] = None,
    ):
        self.registry = registry
        self.reasoner = custom_reasoner or reasoner
        self.fact_checker = custom_fact_checker or fact_checker

    def run(
        self,
        text: Optional[str] = None,
        media_bytes: Optional[bytes] = None,
        media_filename: Optional[str] = None,
        media_content_type: Optional[str] = None,
        enabled_detectors: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        api_key: Optional[str] = None,
        fact_check_api_key: Optional[str] = None,
    ) -> AnalysisResponse:
        start_time = time.perf_counter()
        req_id = request_id or f"req_{uuid.uuid4().hex[:8]}"
        iso_now = datetime.now(timezone.utc).isoformat()

        # Build input
        detector_input = DetectorInput(
            text=text,
            media_bytes=media_bytes,
            media_filename=media_filename,
            media_content_type=media_content_type,
        )

        # Detect modalities received
        modalities_received: List[str] = []
        if detector_input.has_text:
            modalities_received.append("text")
        if detector_input.has_image:
            modalities_received.append("image")
        if detector_input.has_video:
            modalities_received.append("video")

        # Determine active detectors
        all_detectors = self.registry.list_all()
        target_detectors = []
        for d in all_detectors:
            if enabled_detectors is None or d.detector_id in enabled_detectors:
                target_detectors.append(d)

        # Execute detectors
        findings: Dict[str, DetectorFinding] = {}
        all_evidence: List[EvidenceItem] = []
        untested_modalities: List[str] = []

        for detector in target_detectors:
            finding, evidence = detector.run(detector_input)
            findings[detector.detector_id] = finding
            all_evidence.extend(evidence)

            if finding.status == DetectorStatus.NOT_RUN:
                untested_modalities.append(f"{detector.detector_name}: {finding.summary}")

        # External Fact-Checking Intelligence (Google Fact Check Tools API)
        fc_records = []
        fc_dicts = []
        if text and len(text.strip()) >= 5 and self.fact_checker.is_available(override_key=fact_check_api_key):
            try:
                fc_records = self.fact_checker.search(text, api_key_override=fact_check_api_key)
                if fc_records:
                    fc_evidence = self.fact_checker.to_evidence(fc_records)
                    all_evidence.extend(fc_evidence)
                    fc_dicts = [
                        {
                            "publisher": r.publisher,
                            "rating": r.rating,
                            "claim": r.claim,
                            "url": r.url,
                            "title": r.title,
                        }
                        for r in fc_records
                    ]
            except Exception as exc:
                logger.warning(f"Fact check search failed: {exc}")

        # Signal Fusion and Assessment
        overall_assessment = self._fuse_assessment(findings, fact_checks=fc_records)
        explanation = self._generate_explanation(findings, all_evidence, overall_assessment)
        limitations = self._derive_limitations(findings, untested_modalities)

        if fc_records:
            limitations.model_assumptions.append(
                f"Cross-referenced with {len(fc_records)} verified ClaimReview fact-check publisher records."
            )

        # Multimodal AI Cognitive Reasoning Layer (if API key available)
        if self.reasoner.is_available(override_key=api_key):
            ai_result = self.reasoner.reason(
                text=text,
                media_bytes=media_bytes,
                media_content_type=media_content_type,
                local_findings=findings,
                local_evidence=all_evidence,
                api_key_override=api_key,
                fact_checks=fc_dicts if fc_dicts else None,
            )
            if ai_result:
                ai_assessment, ai_explanation, extra_evidence = ai_result
                overall_assessment = ai_assessment
                explanation = ai_explanation
                all_evidence.extend(extra_evidence)
                limitations.model_assumptions.append(
                    f"Multimodal reasoning enhanced via Google {self.reasoner.model}."
                )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return AnalysisResponse(
            request_id=req_id,
            timestamp=iso_now,
            processing_time_ms=elapsed_ms,
            modalities_received=modalities_received,
            overall_assessment=overall_assessment,
            findings=findings,
            evidence=all_evidence,
            explanation=explanation,
            limitations=limitations,
        )

    def _fuse_assessment(
        self,
        findings: Dict[str, DetectorFinding],
        fact_checks: Optional[List[Any]] = None,
    ) -> OverallAssessment:
        scored_findings = [
            f for f in findings.values()
            if f.status == DetectorStatus.SUCCESS and f.manipulation_score is not None
        ]

        primary_concerns: List[str] = []
        high_severity_fcs = [
            r for r in (fact_checks or [])
            if getattr(r, "severity", None) == EvidenceSeverity.HIGH
        ]

        if not scored_findings:
            if high_severity_fcs:
                top_fc = high_severity_fcs[0]
                primary_concerns.append(
                    f"Verified Fact-Check ({top_fc.publisher}): Rated '{top_fc.rating}'."
                )
                return OverallAssessment(
                    verdict=Verdict.LIKELY_MANIPULATED,
                    confidence=0.92,
                    risk_level=RiskLevel.HIGH,
                    summary=(
                        f"Claim matches verified debunk in ClaimReview archive by "
                        f"{top_fc.publisher} (Rating: '{top_fc.rating}')."
                    ),
                    primary_concerns=primary_concerns,
                )

            # If no detector could produce a conclusive score
            uncertain_findings = [f for f in findings.values() if f.status == DetectorStatus.UNCERTAIN]
            if uncertain_findings:
                summary = "Analysis inconclusive: Provided inputs yielded uncertain signals across active detectors."
                primary_concerns.append("Signals below operational thresholds (e.g. short text, low resolution, or absent landmarks).")
            else:
                summary = "Analysis incomplete: All active detectors were either skipped or encountered errors."
                primary_concerns.append("Required modalities were missing or encountered processing issues.")

            return OverallAssessment(
                verdict=Verdict.UNCERTAIN,
                confidence=0.30,
                risk_level=RiskLevel.MEDIUM,
                summary=summary,
                primary_concerns=primary_concerns,
            )

        # Weighted calculation
        total_weight = 0.0
        weighted_score_sum = 0.0
        confidence_sum = 0.0

        for f in scored_findings:
            w = max(0.2, f.confidence)
            total_weight += w
            weighted_score_sum += (f.manipulation_score or 0.0) * w
            confidence_sum += f.confidence

            if (f.manipulation_score or 0.0) >= 0.50:
                primary_concerns.append(f"{f.detector_name}: {f.summary}")

        if high_severity_fcs:
            top_fc = high_severity_fcs[0]
            primary_concerns.insert(
                0, f"Verified Fact-Check ({top_fc.publisher}): Rated '{top_fc.rating}'."
            )
            weighted_score_sum += 0.95 * 2.0
            total_weight += 2.0

        agg_score = weighted_score_sum / max(1e-5, total_weight)
        agg_confidence = max(confidence_sum / len(scored_findings), 0.88 if high_severity_fcs else 0.0)

        # Determine Verdict
        if agg_score >= 0.75:
            verdict = Verdict.LIKELY_MANIPULATED
        elif agg_score >= 0.50:
            verdict = Verdict.SUSPICIOUS
        elif agg_score < 0.25:
            verdict = Verdict.LIKELY_REAL
        else:
            verdict = Verdict.UNCERTAIN

        # Determine Risk Level
        if agg_score >= 0.85:
            risk = RiskLevel.CRITICAL
        elif agg_score >= 0.60:
            risk = RiskLevel.HIGH
        elif agg_score >= 0.30:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        summary_parts = []
        if verdict == Verdict.LIKELY_MANIPULATED:
            summary_parts.append("Content demonstrates strong indications of synthetic generation, visual tampering, or factual discrepancy.")
        elif verdict == Verdict.SUSPICIOUS:
            summary_parts.append("Notable inconsistencies or synthetic artifacts observed across multimodal evaluation.")
        elif verdict == Verdict.LIKELY_REAL:
            summary_parts.append("Content exhibits natural, consistent characteristics across examined modalities.")
        else:
            summary_parts.append("Mixed or borderline signals prevent a definitive authenticity classification.")

        return OverallAssessment(
            verdict=verdict,
            confidence=round(agg_confidence, 2),
            risk_level=risk,
            summary=" ".join(summary_parts),
            primary_concerns=primary_concerns,
        )

    def _generate_explanation(
        self,
        findings: Dict[str, DetectorFinding],
        evidence: List[EvidenceItem],
        assessment: OverallAssessment,
    ) -> Explanation:
        headline = f"Assessment: {assessment.verdict.value} (Risk: {assessment.risk_level.value})"
        
        supporting_points: List[str] = []
        counter_evidence: List[str] = []

        for ev in evidence:
            if ev.severity in (EvidenceSeverity.HIGH, EvidenceSeverity.MEDIUM):
                supporting_points.append(f"{ev.evidence_type}: {ev.description}")
            else:
                counter_evidence.append(f"{ev.evidence_type}: {ev.description}")

        for f in findings.values():
            if f.status == DetectorStatus.SUCCESS and f.manipulation_score is not None:
                if f.manipulation_score >= 0.50:
                    supporting_points.append(f"{f.detector_name} flagged anomalous metric ({f.manipulation_score:.2f}).")
                else:
                    counter_evidence.append(f"{f.detector_name} observed low manipulation probability ({f.manipulation_score:.2f}).")

        rationale = (
            f"{assessment.summary} The system aggregated {len(findings)} detection channels, "
            f"identifying {len(supporting_points)} potential points of concern."
        )

        return Explanation(
            headline=headline,
            rationale=rationale,
            supporting_points=supporting_points[:6],
            counter_evidence=counter_evidence[:4],
        )

    def _derive_limitations(
        self,
        findings: Dict[str, DetectorFinding],
        untested_modalities: List[str],
    ) -> Limitations:
        disclaimers = [
            "VeriLens AI is a hackathon-grade decision support system intended to augment, not replace, human fact-checking.",
            "Heuristic and statistical indicators are probabilistic and can be affected by aggressive compression or editing.",
        ]

        model_assumptions = [
            "Frequency analysis assumes standard non-compressed or mildly compressed media formats.",
            "Text statistical burstiness assumes standard English orthography and sentence syntax.",
        ]

        return Limitations(
            disclaimers=disclaimers,
            untested_modalities=untested_modalities,
            model_assumptions=model_assumptions,
            human_verification_recommended=True,
        )

# Global pipeline instance
pipeline = AnalysisPipeline()
