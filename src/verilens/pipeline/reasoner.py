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

from verilens.core.logger import get_logger
from verilens.core.video import extract_video_frames, frame_to_jpeg_bytes

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
        fact_checks: Optional[List[Dict[str, Any]]] = None,
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

        fc_text = ""
        if fact_checks:
            fc_lines = [
                f"- {fc.get('publisher')}: Rated '{fc.get('rating')}'. Claim reviewed: '{fc.get('claim')}' (Article: {fc.get('url')})"
                for fc in fact_checks
            ]
            fc_text = "\nVERIFIED EXTERNAL FACT-CHECKS (CLAIMREVIEW ARCHIVES):\n" + "\n".join(fc_lines) + "\n"

        is_video = bool(media_bytes and media_content_type and media_content_type.startswith("video/"))
        is_image = bool(media_bytes and media_content_type and media_content_type.startswith("image/"))
        has_media = is_image or is_video

        if is_video:
            media_instruction = (
                "Video frames from the uploaded media are attached in chronological sequence. "
                "Carefully inspect the facial anatomy, eyes, mouth movement, object interactions, "
                "temporal consistency, and any visual morphing, melting, unnatural blurring, or generative AI "
                "video synthesis artifacts (e.g. ModelScope, Sora, Runway, Gen-2, or benchmark memes like the Will Smith eating spaghetti test)."
            )
        elif is_image:
            media_instruction = (
                "An image is attached. You may reference visual observations from the attached image directly."
            )
        else:
            media_instruction = (
                "NO image or video is attached. Base your analysis ONLY on the text claim and the local forensic telemetry below."
            )

        prompt_text = (
            "You are a multimodal forensic verification expert. Your role is to synthesize visual media observations "
            "with local algorithmic telemetry to produce an objective, accurate verdict.\n\n"
            "CRITICAL RULES:\n"
            "- Cross-reference the LOCAL FORENSIC TELEMETRY with direct visual observations from the attached media.\n"
            "- Carefully inspect attached visual media for generative AI artifacts (e.g. morphing facial features, distorted anatomy, melting food/objects, impossible physical physics, or famous AI video benchmarks like the Will Smith eating spaghetti test).\n"
            "- If the media clearly depicts an AI-generated video or synthetic image, flag it as SUSPICIOUS or LIKELY_MANIPULATED with high confidence, even if individual algorithmic detectors report low scores.\n"
            "- Do NOT invent claims not in the text, but accurately describe observed visual distortions.\n"
            "- If you are genuinely uncertain, return UNCERTAIN.\n"
            "- When local detectors report conflicting signals, acknowledge the conflict.\n\n"
            f"MEDIA STATUS: {media_instruction}\n\n"
            f"CLAIM / CAPTION TEXT:\n\"\"\"{text or '[No text provided]'}\"\"\"\n\n"
            f"LOCAL FORENSIC TELEMETRY (from algorithmic detectors):\n{signals_text}\n"
            f"{fc_text}\n"
            "ANALYSIS TASKS:\n"
            "1. Evaluate whether the visual media exhibits generative AI synthesis, morphing, face replacement, or physical anomalies.\n"
            "2. Cross-reference visual observations with the local algorithmic detector scores below.\n"
            "3. If verified external fact-checks or known AI memes/benchmarks match the claim, cite them explicitly.\n"
            "4. Provide a balanced, objective assessment with decisive evidence.\n\n"
            "Output MUST be strict JSON matching this exact structure:\n"
            "{\n"
            "  \"verdict\": \"LIKELY_REAL\" | \"SUSPICIOUS\" | \"LIKELY_MANIPULATED\" | \"UNCERTAIN\",\n"
            "  \"manipulation_score\": <float between 0.0 and 1.0>,\n"
            "  \"confidence\": <float between 0.0 and 1.0>,\n"
            "  \"risk_level\": \"LOW\" | \"MEDIUM\" | \"HIGH\" | \"CRITICAL\",\n"
            "  \"summary\": \"<Concise assessment grounded in provided signals>\",\n"
            "  \"primary_concerns\": [\"<concern backed by telemetry>\"],\n"
            "  \"headline\": \"<Short headline>\",\n"
            "  \"rationale\": \"<2-3 sentence rationale referencing specific local signals>\",\n"
            "  \"supporting_points\": [\"<evidence point citing local data>\"],\n"
            "  \"counter_evidence\": [\"<mitigating observation>\"],\n"
            "  \"evidence_items\": [\n"
            "    {\n"
            "      \"evidence_type\": \"<e.g. CROSS_MODAL_MISMATCH | SIGNAL_CORROBORATION>\",\n"
            "      \"description\": \"<observation grounded in provided data>\",\n"
            "      \"severity\": \"LOW\" | \"MEDIUM\" | \"HIGH\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        parts: List[Dict[str, Any]] = [{"text": prompt_text}]

        # Attach image or video frames if provided
        if is_image and media_bytes:
            b64_img = base64.b64encode(media_bytes).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": media_content_type,
                    "data": b64_img,
                }
            })
        elif is_video and media_bytes:
            try:
                frames = extract_video_frames(media_bytes, max_frames=4, max_dimension=720)
                for f in frames:
                    jpeg_bytes = frame_to_jpeg_bytes(f, quality=80)
                    b64_f = base64.b64encode(jpeg_bytes).decode("utf-8")
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_f,
                        }
                    })
            except Exception as exc:
                logger.warning(f"Failed to attach video frames to Gemini reasoner: {exc}")

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
