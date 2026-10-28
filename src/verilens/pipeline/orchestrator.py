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
from verilens.pipeline.web_knowledge import (
    web_knowledge_verifier,
    WebKnowledgeVerifier,
    KnowledgeVerificationResult,
)
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
        custom_web_verifier: Optional[WebKnowledgeVerifier] = None,
    ):
        self.registry = registry
        self.reasoner = custom_reasoner or reasoner
        self.fact_checker = custom_fact_checker or fact_checker
        self.web_verifier = custom_web_verifier or web_knowledge_verifier

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

        # Extract visual text from OCR detector if available
        ocr_extracted_text = ""
        ocr_finding = findings.get("ocr_detector")
        if ocr_finding and ocr_finding.metrics:
            ocr_extracted_text = str(ocr_finding.metrics.get("extracted_text", "")).strip()

        # Primary claim text: explicit input text OR extracted visual OCR text
        effective_claim_text = (text or "").strip()
        used_ocr_as_claim = False
        if not effective_claim_text and ocr_extracted_text and len(ocr_extracted_text) >= 5:
            effective_claim_text = ocr_extracted_text
            used_ocr_as_claim = True
            logger.info(f"Orchestrator: No headline provided. Using OCR text as claim: '{effective_claim_text[:80]}...'")

        # External Fact-Checking Intelligence (Google Fact Check Tools API)
        fc_records = []
        fc_dicts = []
        if effective_claim_text and len(effective_claim_text) >= 5 and self.fact_checker.is_available(override_key=fact_check_api_key):
            try:
                fc_records = self.fact_checker.search(effective_claim_text, api_key_override=fact_check_api_key)
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

        # Real-Time Web & Encyclopedia Knowledge Verification (Wikipedia REST, DuckDuckGo, Google News)
        knowledge_res: Optional[KnowledgeVerificationResult] = None
        if effective_claim_text and len(effective_claim_text) >= 5:
            try:
                knowledge_res = self.web_verifier.verify(effective_claim_text)
                if knowledge_res:
                    ev_severity = EvidenceSeverity.HIGH if knowledge_res.status == "CONTRADICTION" else EvidenceSeverity.LOW
                    all_evidence.append(
                        EvidenceItem(
                            id="ev_web_knowledge_1",
                            detector_id="web_knowledge",
                            evidence_type="WEB_KNOWLEDGE_CONTRADICTION" if knowledge_res.status == "CONTRADICTION" else "WEB_KNOWLEDGE_CORROBORATION",
                            description=knowledge_res.summary,
                            severity=ev_severity,
                            metadata={
                                "source_title": knowledge_res.source_title,
                                "source_url": knowledge_res.source_url,
                                "snippet": knowledge_res.snippet,
                                "status": knowledge_res.status,
                            },
                        )
                    )
            except Exception as exc:
                logger.warning(f"Web knowledge verification failed: {exc}")

        # Signal Fusion and Assessment
        overall_assessment = self._fuse_assessment(
            findings,
            fact_checks=fc_records,
            web_knowledge=knowledge_res,
            ocr_claimed_text=ocr_extracted_text if used_ocr_as_claim else None,
        )
        explanation = self._generate_explanation(findings, all_evidence, overall_assessment)
        limitations = self._derive_limitations(findings, untested_modalities)

        if fc_records:
            limitations.model_assumptions.append(
                f"Cross-referenced with {len(fc_records)} verified ClaimReview fact-check publisher records."
            )
        if knowledge_res:
            limitations.model_assumptions.append(
                f"Cross-referenced with verified encyclopedia knowledge source: {knowledge_res.source_title}."
            )

        # Multimodal AI Cognitive Reasoning Layer (advisory — never overwrites local evidence)
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
                # Merge AI advisory with local assessment — never wholesale replace
                overall_assessment = self._merge_ai_advisory(
                    local_assessment=overall_assessment,
                    ai_assessment=ai_assessment,
                    local_explanation=explanation,
                    ai_explanation=ai_explanation,
                )
                # Preserve local explanation, adopt AI headline and append AI rationale as supporting context
                if ai_explanation.headline:
                    explanation.headline = ai_explanation.headline
                if ai_explanation.rationale:
                    explanation.supporting_points.append(
                        f"AI Advisory ({self.reasoner.model}): {ai_explanation.rationale[:200]}"
                    )
                if ai_explanation.supporting_points:
                    explanation.supporting_points.extend(ai_explanation.supporting_points[:3])
                all_evidence.extend(extra_evidence)
                limitations.model_assumptions.append(
                    f"Multimodal reasoning advisory via Google {self.reasoner.model} "
                    f"(merged with local forensic signals, not used as sole authority)."
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
        web_knowledge: Optional[KnowledgeVerificationResult] = None,
        ocr_claimed_text: Optional[str] = None,
    ) -> OverallAssessment:
        scored_findings = [
            f for f in findings.values()
            if f.status == DetectorStatus.SUCCESS and f.manipulation_score is not None
        ]

        primary_concerns: List[str] = []
        basis_sources: List[str] = []

        if ocr_claimed_text:
            basis_sources.append(
                f"Visual Content Analyzed (Native OCR): \"{ocr_claimed_text[:110]}...\""
            )

        # Only DIRECT MATCH debunks can be treated as a debunk of the user's input!
        direct_debunks = [
            r for r in (fact_checks or [])
            if getattr(r, "severity", None) == EvidenceSeverity.HIGH
            and getattr(r, "is_direct_match", False)
        ]
        related_fact_checks = [
            r for r in (fact_checks or [])
            if not getattr(r, "is_direct_match", False)
        ]

        # 1. High priority: Direct Web Knowledge Contradiction (e.g. RJ Mitte alive -> deceased in 2026)
        if web_knowledge and web_knowledge.status == "CONTRADICTION":
            primary_concerns.append(
                f"Web Reference Contradiction ({web_knowledge.source_title}): {web_knowledge.summary}"
            )
            basis_sources.append(
                f"{web_knowledge.source_title}: {web_knowledge.snippet} Source: {web_knowledge.source_url}"
            )
            return OverallAssessment(
                verdict=Verdict.LIKELY_MANIPULATED,
                confidence=web_knowledge.confidence,
                risk_level=RiskLevel.HIGH,
                summary=web_knowledge.summary,
                primary_concerns=primary_concerns,
                basis_sources=basis_sources,
            )

        # 2. Direct Fact-Check Debunk
        if direct_debunks:
            top_fc = direct_debunks[0]
            primary_concerns.append(
                f"Verified Fact-Check ({top_fc.publisher}): Rated '{top_fc.rating}'."
            )
            basis_sources.append(
                f"ClaimReview Archive ({top_fc.publisher}): Directly debunks claim ('{top_fc.claim[:80]}...'). Source: {top_fc.url}"
            )
            return OverallAssessment(
                verdict=Verdict.LIKELY_MANIPULATED,
                confidence=0.92,
                risk_level=RiskLevel.HIGH,
                summary=(
                    f"Claim directly matches verified debunk in ClaimReview archive by "
                    f"{top_fc.publisher} (Rating: '{top_fc.rating}')."
                ),
                primary_concerns=primary_concerns,
                basis_sources=basis_sources,
            )

        # 3. Web Knowledge Confirmation (e.g. verified news death, historical events)
        if web_knowledge and web_knowledge.status == "CONFIRMATION":
            deepfake_flag = any(f.detector_id == "deepfake_detector" and (f.manipulation_score or 0) >= 0.75 for f in scored_findings)
            if not deepfake_flag:
                basis_sources.append(
                    f"{web_knowledge.source_title}: {web_knowledge.snippet} Source: {web_knowledge.source_url}"
                )
                return OverallAssessment(
                    verdict=Verdict.LIKELY_REAL,
                    confidence=web_knowledge.confidence,
                    risk_level=RiskLevel.LOW,
                    summary=web_knowledge.summary,
                    primary_concerns=[],
                    basis_sources=basis_sources,
                )

        if not scored_findings:
            # If no detector could produce a conclusive score
            uncertain_findings = [f for f in findings.values() if f.status == DetectorStatus.UNCERTAIN]
            if uncertain_findings:
                for uf in uncertain_findings:
                    basis_sources.append(f"{uf.detector_name}: {uf.summary}")
                
                if related_fact_checks:
                    basis_sources.append(
                        f"ClaimReview Search: Found {len(related_fact_checks)} related background debunks, "
                        f"but none match your input statement (e.g. '{related_fact_checks[0].claim[:60]}...')."
                    )
                else:
                    basis_sources.append("ClaimReview Search: No verified debunks found for this claim in Reuters/AP/Snopes archives.")

                summary = (
                    "Analysis inconclusive: Input length or media operational boundary is below the threshold "
                    "for deep statistical modeling. No anomalous forensic indicators detected."
                )
                # No manipulation detected on clean short text -> Risk is LOW, not MEDIUM
                risk_level = RiskLevel.LOW
            else:
                summary = "Analysis incomplete: All active detectors were either skipped or encountered errors."
                primary_concerns.append("Required modalities were missing or encountered processing issues.")
                basis_sources.append("System Telemetry: Missing modalities or processing errors across active channels.")
                risk_level = RiskLevel.MEDIUM

            return OverallAssessment(
                verdict=Verdict.UNCERTAIN,
                confidence=0.35,
                risk_level=risk_level,
                summary=summary,
                primary_concerns=primary_concerns,
                basis_sources=basis_sources,
            )

        # Quality-weighted fusion with detector disagreement detection
        total_weight = 0.0
        weighted_score_sum = 0.0
        confidence_sum = 0.0
        scores_list: List[float] = []

        for f in scored_findings:
            base_w = max(0.2, f.confidence)
            quality_factor = 1.0

            # Dynamic modality quality scaling
            if f.detector_id == "synthetic_text":
                wc = (f.metrics or {}).get("word_count", 30)
                quality_factor = min(1.0, max(0.5, wc / 30.0))

            w = base_w * quality_factor
            total_weight += w
            score = f.manipulation_score or 0.0
            weighted_score_sum += score * w
            confidence_sum += f.confidence
            scores_list.append(score)

            if score >= 0.50:
                primary_concerns.append(f"{f.detector_name}: {f.summary}")
                basis_sources.append(
                    f"{f.detector_name} (Anomalous, score: {score:.2f}): {f.summary}"
                )
            else:
                basis_sources.append(
                    f"{f.detector_name} (Clean, score: {score:.2f}): {f.summary}"
                )

        if direct_debunks:
            top_fc = direct_debunks[0]
            primary_concerns.insert(
                0, f"Verified Fact-Check ({top_fc.publisher}): Rated '{top_fc.rating}'."
            )
            basis_sources.insert(
                0,
                f"ClaimReview Archive ({top_fc.publisher}): Rated '{top_fc.rating}'. Source: {top_fc.url}"
            )
            weighted_score_sum += 0.95 * 2.0
            total_weight += 2.0

        agg_score = weighted_score_sum / max(1e-5, total_weight)
        agg_confidence = confidence_sum / len(scored_findings)
        if direct_debunks:
            agg_confidence = max(agg_confidence, 0.88)

        # Detector disagreement detection: if any two detectors differ by >0.40, cap confidence
        has_conflict = False
        if len(scores_list) >= 2:
            score_range = max(scores_list) - min(scores_list)
            if score_range > 0.40:
                has_conflict = True
                agg_confidence = min(agg_confidence, 0.50)
                primary_concerns.append(
                    f"Detector signals conflict (score range: {score_range:.2f}). "
                    f"Confidence capped — human review recommended."
                )
                basis_sources.append(
                    f"Conflict Telemetry: Disagreement between channels (range {score_range:.2f}) reduces confidence."
                )

        # Determine Verdict
        if agg_score >= 0.75:
            verdict = Verdict.LIKELY_MANIPULATED
        elif agg_score >= 0.50:
            verdict = Verdict.SUSPICIOUS
        elif agg_score < 0.25:
            verdict = Verdict.LIKELY_REAL
        else:
            verdict = Verdict.UNCERTAIN

        # If detectors conflict, downgrade strong verdicts to SUSPICIOUS/UNCERTAIN
        if has_conflict and verdict == Verdict.LIKELY_MANIPULATED:
            verdict = Verdict.SUSPICIOUS
        elif has_conflict and verdict == Verdict.LIKELY_REAL:
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

        if has_conflict:
            summary_parts.append("Note: Detector signals show significant disagreement; this assessment has reduced certainty.")

        if direct_debunks:
            summary_parts.append(f"Basis: Verified debunk by {direct_debunks[0].publisher} ('{direct_debunks[0].rating}').")
        elif primary_concerns:
            summary_parts.append(f"Basis: Anomalies flagged by {len(primary_concerns)} detection channel(s).")
        else:
            summary_parts.append("Basis: No forensic anomalies detected across examined channels.")

        return OverallAssessment(
            verdict=verdict,
            confidence=round(agg_confidence, 2),
            risk_level=risk,
            summary=" ".join(summary_parts),
            primary_concerns=primary_concerns,
            basis_sources=basis_sources,
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

    def _merge_ai_advisory(
        self,
        local_assessment: OverallAssessment,
        ai_assessment: OverallAssessment,
        local_explanation: Explanation,
        ai_explanation: Explanation,
    ) -> OverallAssessment:
        """
        Merges AI advisory assessment with local forensic assessment.
        Local evidence anchors the verdict; AI can shift within one tier
        but never wholesale override.
        """
        local_conf = local_assessment.confidence
        ai_conf = ai_assessment.confidence

        # Weight local signals at 0.65, AI advisory at 0.35
        # ponytail: simple weighted merge, no elaborate ensemble
        VERDICT_ORDER = {
            Verdict.LIKELY_REAL: 0,
            Verdict.UNCERTAIN: 1,
            Verdict.SUSPICIOUS: 2,
            Verdict.LIKELY_MANIPULATED: 3,
        }
        ORDERED_VERDICTS = [Verdict.LIKELY_REAL, Verdict.UNCERTAIN, Verdict.SUSPICIOUS, Verdict.LIKELY_MANIPULATED]

        local_idx = VERDICT_ORDER.get(local_assessment.verdict, 1)
        ai_idx = VERDICT_ORDER.get(ai_assessment.verdict, 1)

        # AI can shift verdict by at most 1 tier from local
        if ai_idx > local_idx:
            merged_idx = min(local_idx + 1, ai_idx)
        elif ai_idx < local_idx:
            merged_idx = max(local_idx - 1, ai_idx)
        else:
            merged_idx = local_idx

        merged_verdict = ORDERED_VERDICTS[min(merged_idx, 3)]
        merged_confidence = round(0.65 * local_conf + 0.35 * ai_conf, 2)

        # Merge primary concerns
        merged_concerns = list(local_assessment.primary_concerns)
        for concern in ai_assessment.primary_concerns[:2]:
            prefixed = f"AI Advisory: {concern}"
            if prefixed not in merged_concerns:
                merged_concerns.append(prefixed)

        # Use local summary as base, append AI advisory note if verdict shifted
        summary = local_assessment.summary
        if merged_verdict != local_assessment.verdict:
            summary += (
                f" AI advisory shifted assessment from {local_assessment.verdict.value} "
                f"to {merged_verdict.value}."
            )

        # Merge basis sources
        merged_sources = list(local_assessment.basis_sources)
        if ai_explanation and ai_explanation.rationale:
            merged_sources.append(
                f"AI Multimodal Advisory ({self.reasoner.model}): {ai_explanation.rationale[:140]}..."
            )

        return OverallAssessment(
            verdict=merged_verdict,
            confidence=merged_confidence,
            risk_level=local_assessment.risk_level,  # Keep local risk level
            summary=summary,
            primary_concerns=merged_concerns,
            basis_sources=merged_sources,
        )

# Global pipeline instance
pipeline = AnalysisPipeline()
