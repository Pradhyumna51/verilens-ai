"""
Unit tests for WebKnowledgeVerifier & Encyclopedia Reference Grounding.
Tests life status contradiction detection, death hoaxes, and historical confirmations.
"""

from unittest.mock import patch, MagicMock
import pytest

from verilens.pipeline.web_knowledge import WebKnowledgeVerifier, KnowledgeVerificationResult
from verilens.pipeline.orchestrator import AnalysisPipeline
from verilens.schemas.response import Verdict, RiskLevel


@pytest.fixture
def verifier():
    return WebKnowledgeVerifier()


def test_life_status_false_alive_contradiction(verifier):
    mock_knowledge = {
        "title": "RJ Mitte",
        "description": "American actor and activist (1992 - 2026)",
        "extract": "Roy Frank 'RJ' Mitte III was an American actor and disability rights activist...",
        "url": "https://en.wikipedia.org/wiki/RJ_Mitte",
    }
    with patch.object(verifier, "_fetch_entity_knowledge", return_value=mock_knowledge):
        res = verifier.verify("RJ Mitte is alive")
        assert res is not None
        assert res.status == "CONTRADICTION"
        assert res.contradiction_type == "LIFE_STATUS_CONTRADICTION"
        assert res.confidence >= 0.90
        assert "passed away" in res.summary
        assert "Wikipedia" in res.source_title


def test_life_status_death_hoax_contradiction(verifier):
    mock_knowledge = {
        "title": "Bryan Cranston",
        "description": "American actor (born 1956)",
        "extract": "Bryan Lee Cranston is an American actor, producer, director, and screenwriter...",
        "url": "https://en.wikipedia.org/wiki/Bryan_Cranston",
    }
    with patch.object(verifier, "_fetch_entity_knowledge", return_value=mock_knowledge):
        res = verifier.verify("Bryan Cranston is dead")
        assert res is not None
        assert res.status == "CONTRADICTION"
        assert res.contradiction_type == "DEATH_HOAX"
        assert "alive" in res.summary


def test_life_status_confirmed_death(verifier):
    mock_knowledge = {
        "title": "Robin Williams",
        "description": "American actor and comedian (1951 - 2014)",
        "extract": "Robin McLaurin Williams was an American actor and comedian...",
        "url": "https://en.wikipedia.org/wiki/Robin_Williams",
    }
    with patch.object(verifier, "_fetch_entity_knowledge", return_value=mock_knowledge):
        res = verifier.verify("Robin Williams passed away")
        assert res is not None
        assert res.status == "CONFIRMATION"
        assert "passed away" in res.summary


def test_life_status_confirmed_living(verifier):
    mock_knowledge = {
        "title": "Bryan Cranston",
        "description": "American actor (born 1956)",
        "extract": "Bryan Lee Cranston is an American actor, producer, director...",
        "url": "https://en.wikipedia.org/wiki/Bryan_Cranston",
    }
    with patch.object(verifier, "_fetch_entity_knowledge", return_value=mock_knowledge):
        res = verifier.verify("Bryan Cranston is alive")
        assert res is not None
        assert res.status == "CONFIRMATION"
        assert "living public figure" in res.summary


def test_historical_fact_confirmation(verifier):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "query": {
            "search": [
                {
                    "title": "Indian Independence Act 1947",
                    "snippet": "The Indian Independence Act was passed by the Parliament of the United Kingdom in 1947.",
                }
            ]
        }
    }
    mock_summary_res = MagicMock()
    mock_summary_res.status_code = 200
    mock_summary_res.json.return_value = {
        "title": "Indian Independence Act 1947",
        "extract": "The 1947 Indian Independence Act partitioned British India...",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Indian_Independence_Act_1947"}},
    }

    with patch("httpx.Client.get") as mock_get:
        mock_get.side_effect = [mock_response, mock_summary_res]
        res = verifier.verify("India gained independence in 1947")
        assert res is not None
        assert res.status == "CONFIRMATION"
        assert "1947" in res.summary


def test_orchestrator_contradiction_yields_high_risk():
    pipe = AnalysisPipeline()
    mock_contradiction = KnowledgeVerificationResult(
        status="CONTRADICTION",
        subject="RJ Mitte",
        confidence=0.92,
        summary="Claim asserts 'RJ Mitte is alive', but verified records prove deceased (1992 - 2026).",
        source_title="Wikipedia / Reference Archive (RJ Mitte)",
        source_url="https://en.wikipedia.org/wiki/RJ_Mitte",
        snippet="RJ Mitte passed away in 2026.",
        contradiction_type="LIFE_STATUS_CONTRADICTION",
    )
    with patch.object(pipe.web_verifier, "verify", return_value=mock_contradiction):
        result = pipe.run(text="RJ Mitte is alive")
        assert result.overall_assessment.verdict == Verdict.LIKELY_MANIPULATED
        assert result.overall_assessment.risk_level == RiskLevel.HIGH
        assert result.overall_assessment.confidence >= 0.90
        assert len(result.overall_assessment.basis_sources) >= 1
        assert "Wikipedia" in result.overall_assessment.basis_sources[0]
