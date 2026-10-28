"""
Multimodal Cognitive Reasoning & Explanation Engine.
Leverages Google Gemini Multimodal API (Free Tier on Google AI Studio) to provide
human-understandable forensic explanations, semantic cross-modal verification,
and deepfake/synthetic anomaly detection on arbitrary real-world inputs.
Falls back gracefully to local heuristic synthesis if API key is not configured or offline.
"""

import json
import base64
from typing import Optional, Dict, Any, List, Tuple
import httpx

from verilens.core.config import settings
from verilens.core.logger import get_logger
from verilens.schemas.response import (
    Verdict,
    RiskLevel,
    EvidenceSeverity,
    EvidenceItem,
    DetectorFinding,
    OverallAssessment,
    Explanation,
)

logger = get_logger("MultimodalReasoner")


class MultimodalReasoner:
    """
    Multimodal Cognitive Forensics Engine.
    Combines local forensic signals with Gemini Multimodal reasoning
    to produce accurate verdicts and human-interpretable explanations.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = settings.GEMINI_API_KEY if api_key is None else api_key
        self.model = model or settings.GEMINI_MODEL or "gemini-2.5-flash"
        self.api_base = settings.GEMINI_API_BASE

    def is_available(self, override_key: Optional[str] = None) -> bool:
        """Returns True if a Gemini API key is configured."""
        key = override_key or self.api_key
        return bool(key and key.strip())

    def reason(
        self,
        text: Optional[str],
        media_bytes: Optional[bytes],
        media_content_type: Optional[str],
        local_findings: Dict[str, DetectorFinding],
        local_evidence: List[EvidenceItem],
        api_key_override: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> Optional[Tuple[OverallAssessment, Explanation, List[EvidenceItem]]]:
        """
        Executes multimodal forensic reasoning using Gemini API.
        Returns (OverallAssessment, Explanation, extra_evidence_items) if successful,
        or None if API key is not configured or an error occurs.
        """
        active_key = (api_key_override or self.api_key or "").strip()
        if not active_key:
            return None

        # Prepare summary of local signals for model context
        local_signals_summary = []
        for det_id, finding in local_findings.items():
            if finding.status.value == "SUCCESS" and finding.manipulation_score is not None:
                local_signals_summary.append(
                    f"- {finding.detector_name}: score={finding.manipulation_score:.2f}, summary='{finding.summary}'"
                )
        signals_text = "\n".join(local_signals_summary) or "No local anomalies flagged."

        prompt_text = (
            "You are an expert multimodal digital forensics investigator and fact-checking intelligence engine. "
            "Evaluate the provided claim text and visual media asset for misinformation, generative synthesis, or out-of-context reuse.\n\n"
            f"CLAIM / CAPTION TEXT:\n\"\"\"{text or '[No text provided]'}\"\"\"\n\n"
            f"LOCAL FORENSIC TELEMETRY OBSERVED:\n{signals_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Cross-Modal Veracity: Does the visual evidence genuinely depict the claimed event, location, and entities? "
            "Are there discrepancies between what the caption asserts and what the media portrays?\n"
            "2. Generative Synthesis / Deepfake: Inspect for AI diffusion artifacts (unnatural anatomy, warped geometry, "
            "synthetic skin sheen, prompt incoherence, blended boundaries, unnatural lighting).\n"
            "3. Out-of-Context Reuse: Is this media likely recycled historical footage or photos misrepresented as breaking news?\n"
            "4. Explanation: Provide a human-understandable, objective, journalistic explanation that clearly explains the verdict.\n\n"
            "Output MUST be strict JSON matching this exact structure:\n"
            "{\n"
            "  \"verdict\": \"LIKELY_REAL\" | \"SUSPICIOUS\" | \"LIKELY_MANIPULATED\" | \"UNCERTAIN\",\n"
            "  \"manipulation_score\": <float between 0.0 and 1.0>,\n"
            "  \"confidence\": <float between 0.0 and 1.0>,\n"
            "  \"risk_level\": \"LOW\" | \"MEDIUM\" | \"HIGH\" | \"CRITICAL\",\n"
            "  \"summary\": \"<Concise executive assessment summary>\",\n"
            "  \"primary_concerns\": [\"<concern 1>\", \"<concern 2>\"],\n"
            "  \"headline\": \"<Short, impactful headline explaining verdict>\",\n"
            "  \"rationale\": \"<Detailed 2-3 sentence forensic rationale explaining why this verdict was reached>\",\n"
            "  \"supporting_points\": [\"<evidence point 1>\", \"<evidence point 2>\"],\n"
            "  \"counter_evidence\": [\"<mitigating observation or context>\"],\n"
            "  \"evidence_items\": [\n"
            "    {\n"
            "      \"evidence_type\": \"<e.g. AI_SYNTHETIC_CUES | CONFLICTING_TEMPORAL_CUES | CROSS_MODAL_MISMATCH>\",\n"
            "      \"description\": \"<observed discrepancy>\",\n"
            "      \"severity\": \"LOW\" | \"MEDIUM\" | \"HIGH\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        parts: List[Dict[str, Any]] = [{"text": prompt_text}]

        # Attach image if provided
        if media_bytes and media_content_type and media_content_type.startswith("image/"):
            b64_img = base64.b64encode(media_bytes).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": media_content_type,
                    "data": b64_img,
                }
            })

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }

        url = f"{self.api_base}/{self.model}:generateContent?key={active_key}"

        try:
            logger.info(f"Invoking Gemini Multimodal Reasoner ({self.model})...")
            with httpx.Client(timeout=timeout_seconds) as client:
                res = client.post(url, json=payload)

            if res.status_code != 200:
                logger.warning(
                    f"Gemini API returned status {res.status_code}: {res.text[:200]}. Falling back to local ensemble."
                )
                return None

            data = res.json()
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini returned empty candidate response.")
                return None

            raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
            result_json = json.loads(raw_text)

            # Map into Pydantic models
            verdict_str = result_json.get("verdict", "UNCERTAIN").upper()
            verdict = getattr(Verdict, verdict_str, Verdict.UNCERTAIN)

            risk_str = result_json.get("risk_level", "MEDIUM").upper()
            risk_level = getattr(RiskLevel, risk_str, RiskLevel.MEDIUM)

            confidence = float(min(1.0, max(0.0, result_json.get("confidence", 0.85))))
            manip_score = float(min(1.0, max(0.0, result_json.get("manipulation_score", 0.50))))

            assessment = OverallAssessment(
                verdict=verdict,
                confidence=round(confidence, 2),
                risk_level=risk_level,
                summary=result_json.get("summary", "Forensic analysis completed via multimodal reasoning."),
                primary_concerns=result_json.get("primary_concerns", []),
            )

            explanation = Explanation(
                headline=result_json.get("headline", f"Assessment: {verdict.value}"),
                rationale=result_json.get("rationale", assessment.summary),
                supporting_points=result_json.get("supporting_points", []),
                counter_evidence=result_json.get("counter_evidence", []),
            )

            extra_evidence: List[EvidenceItem] = []
            for i, ev in enumerate(result_json.get("evidence_items", [])):
                sev_str = ev.get("severity", "MEDIUM").upper()
                sev = getattr(EvidenceSeverity, sev_str, EvidenceSeverity.MEDIUM)
                extra_evidence.append(
                    EvidenceItem(
                        id=f"ev_ai_reasoning_{i+1}",
                        detector_id="multimodal_reasoner",
                        evidence_type=ev.get("evidence_type", "MULTIMODAL_REASONING_FINDING"),
                        description=ev.get("description", "Multimodal observation recorded."),
                        severity=sev,
                        metadata={"source": "gemini_multimodal_engine"},
                    )
                )

            logger.info(f"Gemini reasoning successful: verdict={verdict.value}, score={manip_score}")
            return assessment, explanation, extra_evidence

        except Exception as exc:
            logger.warning(f"Exception during Gemini Multimodal Reasoning: {exc}. Falling back to local ensemble.")
            return None


# Global singleton reasoner
reasoner = MultimodalReasoner()
