"""
Unit test suite for Google Fact Check Tools API Client and ClaimReview integration.
Tests offline fallback, mocked API responses, severity classification, and pipeline fusion.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.pipeline.fact_check import FactChecker, FactCheckRecord
from verilens.pipeline.orchestrator import AnalysisPipeline
from verilens.schemas.response import EvidenceSeverity, EvidenceItem


class TestFactChecker:
    @pytest.fixture
    def checker(self):
        return FactChecker(api_key="")

    def test_offline_when_no_api_key(self, checker):
        assert checker.is_available() is False
        records = checker.search("Shark swimming on flooded freeway highway")
        assert records == []

    def test_short_query_returns_empty(self, checker):
        records = checker.search("tiny", api_key_override="mock_key")
        assert records == []

    def test_mocked_search_success(self, checker):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "claims": [
                {
                    "text": "Shark was seen swimming on flooded freeway during Hurricane Ian",
                    "claimant": "Social Media Users",
                    "claimDate": "2022-09-29T00:00:00Z",
                    "claimReview": [
                        {
                            "publisher": {
                                "name": "USA TODAY",
                                "site": "usatoday.com",
                            },
                            "url": "https://www.usatoday.com/story/news/factcheck/2022/09/29/fact-check-shark-swimming-highway-photo-altered/10457635002/",
                            "title": "Fact check: Viral shark swimming on highway photo altered",
                            "reviewDate": "2022-09-29T19:30:00Z",
                            "textualRating": "Altered",
                            "languageCode": "en",
                        }
                    ],
                },
                {
                    "text": "Another claim partially true",
                    "claimant": "Anonymous",
                    "claimReview": [
                        {
                            "publisher": {"name": "Snopes"},
                            "url": "https://www.snopes.com/fact-check/shark-highway/",
                            "title": "Shark on Highway Debunk",
                            "textualRating": "Misleading",
                        }
                    ],
                },
            ]
        }

        with patch("httpx.Client.get", return_value=mock_response):
            records = checker.search(
                query="shark flooded highway",
                api_key_override="test_fact_check_key",
            )

        assert len(records) == 2
        assert records[0].publisher == "USA TODAY"
        assert records[0].rating == "Altered"
        assert records[0].severity == EvidenceSeverity.HIGH
        assert "usatoday.com" in records[0].url

        assert records[1].publisher == "Snopes"
        assert records[1].rating == "Misleading"
        assert records[1].severity == EvidenceSeverity.MEDIUM

    def test_to_evidence_conversion(self, checker):
        sample_record = FactCheckRecord(
            claim="Viral shark image is from hurricane",
            publisher="Reuters Fact Check",
            rating="False",
            url="https://www.reuters.com/fact-check/sample",
            title="Reuters Debunk",
            severity=EvidenceSeverity.HIGH,
        )

        evidence = checker.to_evidence([sample_record])
        assert len(evidence) == 1
        ev = evidence[0]
        assert ev.evidence_type == "VERIFIED_FACT_CHECK"
        assert ev.detector_id == "fact_checker"
        assert ev.severity == EvidenceSeverity.HIGH
        assert "Reuters Fact Check" in ev.description
        assert ev.metadata["publisher"] == "Reuters Fact Check"
        assert ev.metadata["rating"] == "False"
        assert ev.metadata["url"] == "https://www.reuters.com/fact-check/sample"

    def test_api_http_error_graceful(self, checker):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("httpx.Client.get", return_value=mock_response):
            records = checker.search("some viral claim", api_key_override="invalid_key")
        assert records == []

    def test_pipeline_integration_with_fact_check(self):
        mock_checker = MagicMock()
        mock_checker.is_available.return_value = True
        mock_checker.search.return_value = [
            FactCheckRecord(
                claim="Shark seen on freeway",
                publisher="AP News",
                rating="False",
                url="https://apnews.com/article/fact-check-shark",
                title="AP Fact Check",
                severity=EvidenceSeverity.HIGH,
            )
        ]
        mock_checker.to_evidence.return_value = [
            EvidenceItem(
                id="ev_factcheck_1",
                detector_id="fact_checker",
                evidence_type="VERIFIED_FACT_CHECK",
                description="Verified check by AP News: Rated 'False'.",
                severity=EvidenceSeverity.HIGH,
                metadata={"publisher": "AP News", "rating": "False"},
            )
        ]

        pipe = AnalysisPipeline(custom_fact_checker=mock_checker)
        response = pipe.run(text="Shark seen on freeway after hurricane disaster")

        assert response is not None
        fc_evidence = [e for e in response.evidence if e.evidence_type == "VERIFIED_FACT_CHECK"]
        assert len(fc_evidence) == 1
        assert fc_evidence[0].metadata["publisher"] == "AP News"
        assert any("ClaimReview" in a for a in response.limitations.model_assumptions)
