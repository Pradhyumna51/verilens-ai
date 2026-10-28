"""
Unit test suite for Multimodal Cognitive Reasoner & Explanation Engine.
Tests offline fallback, mocked Gemini API responses, graceful error handling,
and integration into the AnalysisPipeline.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.pipeline.reasoner import MultimodalReasoner
from verilens.pipeline.orchestrator import AnalysisPipeline
from verilens.schemas.response import (
    Verdict,
    RiskLevel,
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
    OverallAssessment,
    Explanation,
)


class TestMultimodalReasoner:
    @pytest.fixture
    def reasoner(self):
        return MultimodalReasoner(api_key="")

    def test_offline_when_no_api_key(self, reasoner):
        assert reasoner.is_available() is False
        res = reasoner.reason(
            text="A test claim",
            media_bytes=None,
            media_content_type=None,
            local_findings={},
            local_evidence=[],
        )
        assert res is None

    def test_mocked_gemini_success(self, reasoner):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{\n'
                                    '  "verdict": "LIKELY_MANIPULATED",\n'
                                    '  "manipulation_score": 0.88,\n'
                                    '  "confidence": 0.94,\n'
                                    '  "risk_level": "HIGH",\n'
                                    '  "summary": "AI diffusion artifacts detected across background architecture.",\n'
                                    '  "primary_concerns": ["Synthetic geometry", "Inconsistent shadow angles"],\n'
                                    '  "headline": "Fabricated Scene with AI Diffusion Signatures",\n'
                                    '  "rationale": "The structural lines in the background exhibit wave distortion typical of image generators.",\n'
                                    '  "supporting_points": ["Warped pillar geometry", "Unnatural specular highlights"],\n'
                                    '  "counter_evidence": ["No known reverse image matches found"],\n'
                                    '  "evidence_items": [\n'
                                    '    {\n'
                                    '      "evidence_type": "AI_SYNTHETIC_CUES",\n'
                                    '      "description": "Geometric warping in structural pillars",\n'
                                    '      "severity": "HIGH"\n'
                                    '    }\n'
                                    '  ]\n'
                                    '}'
                                )
                            }
                        ]
                    }
                }
            ]
        }

        with patch("httpx.Client.post", return_value=mock_response):
            res = reasoner.reason(
                text="Breaking news claim.",
                media_bytes=b"dummy_bytes",
                media_content_type="image/png",
                local_findings={},
                local_evidence=[],
                api_key_override="test_mock_gemini_key",
            )

        assert res is not None
        assessment, explanation, extra_evidence = res
        assert assessment.verdict == Verdict.LIKELY_MANIPULATED
        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.confidence == 0.94
        assert "AI diffusion artifacts" in assessment.summary
        assert len(assessment.primary_concerns) == 2
        assert "Fabricated Scene" in explanation.headline
        assert len(extra_evidence) == 1
        assert extra_evidence[0].evidence_type == "AI_SYNTHETIC_CUES"

    def test_mocked_gemini_handles_api_error_gracefully(self, reasoner):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        with patch("httpx.Client.post", return_value=mock_response):
            res = reasoner.reason(
                text="A test claim",
                media_bytes=None,
                media_content_type=None,
                local_findings={},
                local_evidence=[],
                api_key_override="test_key",
            )
        assert res is None

    def test_pipeline_integrates_mocked_reasoner(self):
        mock_reasoner = MagicMock(spec=MultimodalReasoner)
        mock_reasoner.is_available.return_value = True

        mock_assessment = OverallAssessment(
            verdict=Verdict.LIKELY_REAL,
            confidence=0.90,
            risk_level=RiskLevel.LOW,
            summary="Multimodal check confirms authenticity.",
            primary_concerns=[],
        )

        mock_explanation = Explanation(
            headline="Authentic Media Verified",
            rationale="Claim and visual metadata match documented records.",
            supporting_points=[],
            counter_evidence=[],
        )

        mock_evidence = [
            EvidenceItem(
                id="ev_ai_01",
                detector_id="multimodal_reasoner",
                evidence_type="VERIFIED_CORRESPONDENCE",
                description="Visual scene conforms to verified geographical archives.",
                severity=EvidenceSeverity.LOW,
            )
        ]

        mock_reasoner.reason.return_value = (mock_assessment, mock_explanation, mock_evidence)
        mock_reasoner.model = "gemini-2.0-flash"

        custom_pipeline = AnalysisPipeline(custom_reasoner=mock_reasoner)
        response = custom_pipeline.run(
            text="Valid journalistic report about clear skies.",
            api_key="mock_key",
        )

        assert response.overall_assessment.verdict == Verdict.LIKELY_REAL
        assert response.explanation.headline == "Authentic Media Verified"
        assert any(e.evidence_type == "VERIFIED_CORRESPONDENCE" for e in response.evidence)
