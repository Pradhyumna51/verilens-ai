"""
Google Fact Check Tools API Client.
Queries verified global fact-checking publishers (Reuters, AP, Snopes, PolitiFact, etc.)
using ClaimReview schema to cross-reference claims against published debunks.
100% free via Google News Initiative.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import httpx

from verilens.core.config import settings
from verilens.core.logger import get_logger
from verilens.schemas.response import EvidenceItem, EvidenceSeverity

logger = get_logger("FactChecker")


def compute_claim_similarity(query: str, claim: str) -> tuple[float, bool]:
    """
    Computes semantic-lexical alignment between the user's input query
    and an external ClaimReview claim statement.
    Returns (similarity_score: float, is_direct_match: bool).
    """
    stopwords = {
        'the', 'a', 'an', 'in', 'on', 'at', 'from', 'to', 'of', 'for', 'is', 'was',
        'were', 'that', 'this', 'with', 'by', 'and', 'or', 'as', 'be', 'are', 'it',
        'its', 'has', 'have', 'had', 'been', 'out', 'between', 'during', 'over', 'into',
        'about', 'above', 'after', 'against', 'all', 'also', 'any', 'because', 'before',
        'below', 'both', 'but', 'can', 'did', 'do', 'does', 'doing', 'down', 'each',
        'few', 'further', 'here', 'how', 'just', 'more', 'most', 'no', 'nor', 'not',
        'only', 'other', 'our', 'same', 'should', 'so', 'some', 'such', 'than', 'too',
        'under', 'until', 'up', 'very', 'what', 'when', 'where', 'which', 'while', 'who'
    }
    words_q = re.findall(r'[a-zA-Z0-9]+', query.lower())
    words_c = re.findall(r'[a-zA-Z0-9]+', claim.lower())
    q_toks = set(w for w in words_q if w not in stopwords and len(w) > 2)
    c_toks = set(w for w in words_c if w not in stopwords and len(w) > 2)
    if not q_toks or not c_toks:
        return 0.0, False

    common = q_toks & c_toks
    if not common:
        return 0.0, False

    jaccard = len(common) / len(q_toks | c_toks)
    q_cov = len(common) / len(q_toks)
    c_cov = len(common) / len(c_toks)
    seq = difflib.SequenceMatcher(None, query.lower().strip(), claim.lower().strip()).ratio()

    harmonic_cov = 2 * (q_cov * c_cov) / (q_cov + c_cov) if (q_cov + c_cov) > 0 else 0.0
    score = (jaccard * 0.30) + (harmonic_cov * 0.40) + (seq * 0.30)
    if c_cov >= 0.70:
        score = max(score, c_cov * 0.85)

    is_direct = False
    if score >= 0.45:
        is_direct = True
    elif q_cov >= 0.65 and c_cov >= 0.35 and len(common) >= 3:
        is_direct = True

    return round(score, 3), is_direct


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
    similarity_score: float = 0.0
    is_direct_match: bool = True


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
        Returns a list of FactCheckRecord instances with semantic alignment checking.
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
                sim_score, is_direct = compute_claim_similarity(query, claim_text)
                
                for review in claim.get("claimReview", []):
                    publisher = review.get("publisher", {}).get("name", "Verified Fact-Checker")
                    rating = review.get("textualRating", "Unrated")
                    review_url = review.get("url", "")
                    title = review.get("title", "")
                    review_date = review.get("reviewDate", "")

                    # Determine rating severity
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
                            similarity_score=sim_score,
                            is_direct_match=is_direct,
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
            if rec.is_direct_match:
                desc = (
                    f"Verified direct check by {rec.publisher}: Rated '{rec.rating}'. "
                    f"Claim: '{rec.claim[:90]}...'"
                )
                ev_severity = rec.severity
            else:
                desc = (
                    f"Related fact-check topic by {rec.publisher} (rated '{rec.rating}', does not match query). "
                    f"Claim: '{rec.claim[:90]}...'"
                )
                ev_severity = EvidenceSeverity.LOW
            items.append(
                EvidenceItem(
                    id=f"ev_factcheck_{i+1}",
                    detector_id="fact_checker",
                    evidence_type="VERIFIED_FACT_CHECK",
                    description=desc,
                    severity=ev_severity,
                    metadata={
                        "publisher": rec.publisher,
                        "rating": rec.rating,
                        "url": rec.url,
                        "title": rec.title,
                        "claim": rec.claim,
                        "claimant": rec.claimant,
                        "review_date": rec.review_date,
                        "is_direct_match": rec.is_direct_match,
                        "similarity_score": rec.similarity_score,
                    },
                )
            )
        return items


# Singleton instance
fact_checker = FactChecker()
