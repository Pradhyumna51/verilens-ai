"""
Google Fact Check Tools API Client.
Queries verified global fact-checking publishers (Reuters, AP, Snopes, PolitiFact, etc.)
using ClaimReview schema to cross-reference claims against published debunks.
100% free via Google News Initiative.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import httpx

from verilens.core.config import settings
from verilens.core.logger import get_logger
from verilens.schemas.response import EvidenceItem, EvidenceSeverity

logger = get_logger("FactChecker")


@dataclass
class FactCheckRecord:
    claim: str
    publisher: str
    rating: str
    url: str
    title: str = ""
    claimant: str = ""
    review_date: str = ""
    severity: EvidenceSeverity = EvidenceSeverity.MEDIUM


class FactChecker:
    """
    Client for Google Fact Check Tools API.
    Searches published ClaimReview records for matching public claims.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = settings.FACT_CHECK_API_KEY if api_key is None else api_key
        self.base_url = settings.FACT_CHECK_API_BASE

    def is_available(self, override_key: Optional[str] = None) -> bool:
        key = self.api_key if override_key is None else override_key
        return bool(key and key.strip())

    def search(
        self,
        query: str,
        api_key_override: Optional[str] = None,
        max_results: int = 3,
        timeout_seconds: float = 6.0,
    ) -> List[FactCheckRecord]:
        """
        Queries Google Fact Check Tools API for claims matching the query text.
        Returns a list of FactCheckRecord instances.
        """
        active_key = (self.api_key if api_key_override is None else api_key_override).strip()
        if not active_key or not query or len(query.strip()) < 5:
            return []

        # Clean query: truncate to first 120 chars for search relevance
        cleaned_query = " ".join(query.strip().split()[:16])

        params = {
            "query": cleaned_query,
            "key": active_key,
            "pageSize": max_results,
            "languageCode": "en",
        }

        url = f"{self.base_url}/claims:search"

        try:
            logger.info(f"Querying Google Fact Check Tools API for: '{cleaned_query[:40]}...'")
            with httpx.Client(timeout=timeout_seconds) as client:
                res = client.get(url, params=params)

            if res.status_code != 200:
                logger.warning(
                    f"Google Fact Check API returned {res.status_code}: {res.text[:150]}"
                )
                return []

            data = res.json()
            raw_claims = data.get("claims", [])
            records: List[FactCheckRecord] = []

            for claim in raw_claims:
                claim_text = claim.get("text", "")
                claimant = claim.get("claimant", "")
                
                for review in claim.get("claimReview", []):
                    publisher = review.get("publisher", {}).get("name", "Verified Fact-Checker")
                    rating = review.get("textualRating", "Unrated")
                    review_url = review.get("url", "")
                    title = review.get("title", "")
                    review_date = review.get("reviewDate", "")

                    # Determine evidence severity based on rating keyword
                    r_lower = rating.lower()
                    if any(w in r_lower for w in ["false", "altered", "manipulated", "fake", "hoax", "incorrect"]):
                        severity = EvidenceSeverity.HIGH
                    elif any(w in r_lower for w in ["misleading", "partly", "missing", "unproven", "distorted"]):
                        severity = EvidenceSeverity.MEDIUM
                    else:
                        severity = EvidenceSeverity.LOW

                    records.append(
                        FactCheckRecord(
                            claim=claim_text,
                            publisher=publisher,
                            rating=rating,
                            url=review_url,
                            title=title,
                            claimant=claimant,
                            review_date=review_date,
                            severity=severity,
                        )
                    )

            logger.info(f"Fact Check search found {len(records)} verified matches.")
            return records[:max_results]

        except Exception as exc:
            logger.warning(f"Error querying Google Fact Check API: {exc}")
            return []

    def to_evidence(self, records: List[FactCheckRecord]) -> List[EvidenceItem]:
        """Converts FactCheckRecord instances into EvidenceItems."""
        items: List[EvidenceItem] = []
        for i, rec in enumerate(records):
            items.append(
                EvidenceItem(
                    id=f"ev_factcheck_{i+1}",
                    detector_id="fact_checker",
                    evidence_type="VERIFIED_FACT_CHECK",
                    description=(
                        f"Verified check by {rec.publisher}: Rated '{rec.rating}'. "
                        f"Claim: '{rec.claim[:90]}...'"
                    ),
                    severity=rec.severity,
                    metadata={
                        "publisher": rec.publisher,
                        "rating": rec.rating,
                        "url": rec.url,
                        "title": rec.title,
                        "claimant": rec.claimant,
                        "review_date": rec.review_date,
                    },
                )
            )
        return items


# Singleton instance
fact_checker = FactChecker()
