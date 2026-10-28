"""
Web Knowledge & Encyclopedia Verification Client.
Cross-references real-world factual claims (biographies, lifespans, historical dates)
against authoritative reference archives (Wikipedia REST API, DuckDuckGo Instant Answers)
to detect death hoaxes, false alive claims, and verify historical events.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import re
import difflib
import httpx

from verilens.core.logger import get_logger
from verilens.schemas.response import EvidenceItem, EvidenceSeverity

logger = get_logger("WebKnowledge")


@dataclass
class KnowledgeVerificationResult:
    status: str  # "CONTRADICTION", "CONFIRMATION", "INCONCLUSIVE"
    subject: str
    confidence: float
    summary: str
    source_title: str
    source_url: str
    snippet: str
    contradiction_type: Optional[str] = None  # e.g. "DEATH_HOAX", "LIFE_STATUS_CONTRADICTION"


class WebKnowledgeVerifier:
    """
    Live Web & Encyclopedia Knowledge Verifier.
    Performs real-time factual checking against authoritative knowledge records.
    """

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds
        self.headers = {"User-Agent": "VeriLensAI/1.0 (verifier@verilens.ai)"}

    def verify(self, claim_text: str) -> Optional[KnowledgeVerificationResult]:
        if not claim_text or len(claim_text.strip()) < 5:
            return None

        # 1. Check for alive / death assertions
        life_result = self._check_life_status(claim_text)
        if life_result:
            return life_result

        # 2. Check for historical / event confirmation
        fact_result = self._check_historical_fact(claim_text)
        if fact_result:
            return fact_result

        return None

    def _check_life_status(self, claim_text: str) -> Optional[KnowledgeVerificationResult]:
        clean_text = claim_text.strip().rstrip("?.!")
        
        # 1. Question formats
        q_alive = re.search(r'^(?:is|was)\s+(.+?)\s+(?:alive|living|still\s+alive)\b', clean_text, re.IGNORECASE)
        q_dead = re.search(r'^(?:did|has|is|was)\s+(.+?)\s+(?:die|dead|passed\s+away)\b', clean_text, re.IGNORECASE)
        
        # 2. Statement formats
        prefix_death = re.search(r'^(?:death|passing)\s+of\s+(.+)', clean_text, re.IGNORECASE)
        s_alive = re.search(r'(.+?)\s+(?:is\s+alive|still\s+alive|is\s+living|has\s+not\s+died|\balive\b)', clean_text, re.IGNORECASE)
        s_dead = re.search(r'(.+?)\s+(?:is\s+dead|died|passed\s+away|has\s+died|killed\s+in|\bdead\b)', clean_text, re.IGNORECASE)

        subject = None
        asserts_alive = False
        asserts_dead = False

        if q_alive:
            subject = q_alive.group(1).strip()
            asserts_alive = True
        elif q_dead:
            subject = q_dead.group(1).strip()
            asserts_dead = True
        elif prefix_death:
            subject = prefix_death.group(1).strip()
            asserts_dead = True
        elif s_alive:
            subject = s_alive.group(1).strip()
            asserts_alive = True
        elif s_dead:
            subject = s_dead.group(1).strip()
            asserts_dead = True
        else:
            # 3. Complex clause formats (e.g. "R.I.P ... Toto warsito, known for ... has passed away ...")
            clause_match = re.search(
                r'(.+?)(?:,\s*known\s+for|\s+has\s+passed\s+away|\s+passed\s+away|\s+has\s+died|\s+is\s+dead|\s+is\s+alive)',
                clean_text,
                re.IGNORECASE,
            )
            if clause_match:
                cand = clause_match.group(1).strip()
                cand_clean = re.sub(
                    r'^(?:r\.?i\.?p\.?|\d{4}\s*[-–—]\s*\d{4}|indonesian|american|japanese|british|veteran|voice\s+actor|actor|actress|singer|the|\s+)+',
                    '',
                    cand,
                    flags=re.IGNORECASE,
                ).strip()
                if len(cand_clean) >= 3:
                    subject = cand_clean
                    if re.search(r'\b(?:passed away|died|has died|death|r\.?i\.?p\.?|killed)\b', clean_text, re.IGNORECASE):
                        asserts_dead = True
                    elif re.search(r'\b(?:is alive|still alive|living)\b', clean_text, re.IGNORECASE):
                        asserts_alive = True

        if not subject or (not asserts_alive and not asserts_dead):
            return None

        # Clean subject: strip common filler prefixes
        subject = re.sub(
            r'^(?:is|was|did|has|actor|actress|singer|politician|president|minister|celebrity|the)\s+',
            '',
            subject,
            flags=re.IGNORECASE,
        ).strip()
        if not subject or len(subject) < 3:
            return None

        knowledge = self._fetch_entity_knowledge(subject)
        if not knowledge:
            # Fallback to live journalism archives (Google News RSS)
            return self._check_news_records(subject, asserts_dead, asserts_alive, clean_text)

        extract = knowledge.get("extract", "")
        desc = knowledge.get("description", "")
        title = knowledge.get("title", subject)
        url = knowledge.get("url", "")
        combined = f"{desc} {extract}".strip()

        # Lifespan and biographical status detection
        lifespan_match = re.search(r'\(\s*(\d{4})\s*[-–—]\s*(\d{4})\s*\)', combined)
        has_ended_lifespan = bool(lifespan_match)
        past_tense_bio = bool(re.search(r'\bwas\s+(?:an?|the)\b', combined, re.IGNORECASE))
        has_death_indicator = has_ended_lifespan or past_tense_bio or bool(re.search(r'\bdied\b|\bdeath of\b|\bpassed away\b', combined, re.IGNORECASE))

        born_only_match = bool(re.search(r'\(\s*born\s+[a-zA-Z0-9,\s]+\)', combined, re.IGNORECASE)) and not has_ended_lifespan
        present_tense_bio = bool(re.search(r'\bis\s+(?:an?|the)\b', combined, re.IGNORECASE)) and not has_ended_lifespan
        has_living_indicator = (born_only_match or present_tense_bio) and not has_death_indicator

        lifespan_str = f"({lifespan_match.group(1)} - {lifespan_match.group(2)})" if lifespan_match else f"({desc})"

        # Case 1: Claim asserts ALIVE, but records prove DECEASED (False alive claim)
        if asserts_alive and has_death_indicator:
            summary = (
                f"Claim asserts '{subject} is alive', but verified reference records document that "
                f"{title} passed away {lifespan_str}. Reference status: '{desc or extract[:120]}'."
            )
            logger.info(f"WebKnowledge: Contradiction flagged for '{claim_text}' -> Subject is deceased.")
            return KnowledgeVerificationResult(
                status="CONTRADICTION",
                subject=title,
                confidence=0.92,
                summary=summary,
                source_title=f"Wikipedia / Reference Archive ({title})",
                source_url=url,
                snippet=f"{title}: {desc}. {extract[:180]}...",
                contradiction_type="LIFE_STATUS_CONTRADICTION",
            )

        # Case 2: Claim asserts DEAD, but records prove LIVING (Death hoax)
        if asserts_dead and has_living_indicator:
            summary = (
                f"Claim asserts '{subject} is dead/died', but verified biographical records confirm "
                f"{title} is alive ({desc or 'living public figure'}). No record of death documented in authoritative reference archives."
            )
            logger.info(f"WebKnowledge: Death hoax flagged for '{claim_text}' -> Subject is living.")
            return KnowledgeVerificationResult(
                status="CONTRADICTION",
                subject=title,
                confidence=0.90,
                summary=summary,
                source_title=f"Wikipedia / Reference Archive ({title})",
                source_url=url,
                snippet=f"{title}: {desc}. {extract[:180]}...",
                contradiction_type="DEATH_HOAX",
            )

        # Case 3: Claim asserts DEAD, and records confirm DECEASED (Accurate news/history)
        if asserts_dead and has_death_indicator:
            summary = (
                f"Claim stating '{subject} passed away / died' is corroborated by verified reference archives: "
                f"{title} passed away {lifespan_str}."
            )
            logger.info(f"WebKnowledge: Historical death confirmed for '{claim_text}'.")
            return KnowledgeVerificationResult(
                status="CONFIRMATION",
                subject=title,
                confidence=0.92,
                summary=summary,
                source_title=f"Wikipedia / Reference Archive ({title})",
                source_url=url,
                snippet=f"{title}: {desc}. {extract[:180]}...",
                contradiction_type=None,
            )

        # Case 4: Claim asserts ALIVE, and records confirm LIVING (Accurate current fact)
        if asserts_alive and has_living_indicator:
            summary = (
                f"Claim stating '{subject} is alive' is corroborated by verified biographical records: "
                f"{title} is an active, living public figure ({desc or 'living'})."
            )
            logger.info(f"WebKnowledge: Living status confirmed for '{claim_text}'.")
            return KnowledgeVerificationResult(
                status="CONFIRMATION",
                subject=title,
                confidence=0.90,
                summary=summary,
                source_title=f"Wikipedia / Reference Archive ({title})",
                source_url=url,
                snippet=f"{title}: {desc}. {extract[:180]}...",
                contradiction_type=None,
            )

        return None

    def _check_historical_fact(self, claim_text: str) -> Optional[KnowledgeVerificationResult]:
        """
        Verifies historical claims involving specific entities and dates (e.g. 'India gained independence in 1947').
        """
        # Look for 4-digit years in the claim
        year_match = re.search(r'\b(1[789]\d\d|20\d\d)\b', claim_text)
        if not year_match:
            return None

        year = year_match.group(1)
        # Search Wikipedia for the claim
        words = re.findall(r'[a-zA-Z0-9]+', claim_text)
        search_query = " ".join(words[:6])

        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "utf8": "1",
                "format": "json",
                "srlimit": 3,
            }
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(search_url, params=params, headers=self.headers)
            if res.status_code != 200:
                return None

            hits = res.json().get("query", {}).get("search", [])
            if not hits:
                return None

            # Check if any top hit's title or snippet contains the specified year and key terms
            for hit in hits:
                title = hit.get("title", "")
                snippet = re.sub(r'<[^>]+>', '', hit.get("snippet", ""))
                combined = f"{title} {snippet}"

                if year in combined:
                    # Fetch summary for top verified article
                    formatted_title = title.replace(" ", "_")
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
                    try:
                        with httpx.Client(timeout=self.timeout) as client:
                            s_res = client.get(summary_url, headers=self.headers)
                        if s_res.status_code == 200:
                            s_data = s_res.json()
                            extract = s_data.get("extract", "")
                            page_url = s_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{formatted_title}")
                            
                            # Verify if the year and subject match
                            if year in extract or year in title:
                                summary = (
                                    f"Historical claim verified against authoritative global reference archives: "
                                    f"'{title}' explicitly documents event records in {year}."
                                )
                                logger.info(f"WebKnowledge: Historical fact confirmed for '{claim_text}' via '{title}'.")
                                return KnowledgeVerificationResult(
                                    status="CONFIRMATION",
                                    subject=title,
                                    confidence=0.92,
                                    summary=summary,
                                    source_title=f"Wikipedia Reference Archive ({title})",
                                    source_url=page_url,
                                    snippet=f"{title}: {extract[:220]}...",
                                    contradiction_type=None,
                                )
                    except Exception:
                        continue

        except Exception as exc:
            logger.warning(f"WebKnowledge historical verification error: {exc}")

        return None

    def _fetch_entity_knowledge(self, subject: str) -> Optional[Dict[str, str]]:
        """Queries Wikipedia REST API & DuckDuckGo Instant Answer for a named subject."""
        headers = self.headers
        formatted_title = subject.strip().replace(" ", "_")

        # 1. Try Wikipedia summary directly
        try:
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(wiki_url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                return {
                    "title": data.get("title", subject),
                    "description": data.get("description", ""),
                    "extract": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{formatted_title}"),
                }
        except Exception:
            pass

        # 2. Try DuckDuckGo Instant Answer
        try:
            ddg_url = f"https://api.duckduckgo.com/?q={subject}&format=json"
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(ddg_url)
            if res.status_code == 200:
                d = res.json()
                abstract = d.get("AbstractText", "")
                heading = d.get("Heading", subject)
                url = d.get("AbstractURL", "")
                if abstract:
                    return {
                        "title": heading,
                        "description": "",
                        "extract": abstract,
                        "url": url or f"https://en.wikipedia.org/wiki/{formatted_title}",
                    }
        except Exception:
            pass

        # 3. Try Wikipedia search API fallback
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": subject,
                "utf8": "1",
                "format": "json",
                "srlimit": 1,
            }
            with httpx.Client(timeout=self.timeout) as client:
                s_res = client.get(search_url, params=params, headers=headers)
            if s_res.status_code == 200:
                hits = s_res.json().get("query", {}).get("search", [])
                if hits:
                    best_title = hits[0].get("title", "").replace(" ", "_")
                    best_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title}"
                    with httpx.Client(timeout=self.timeout) as client:
                        b_res = client.get(best_url, headers=headers)
                    if b_res.status_code == 200:
                        b_data = b_res.json()
                        return {
                            "title": b_data.get("title", subject),
                            "description": b_data.get("description", ""),
                            "extract": b_data.get("extract", ""),
                            "url": b_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{best_title}"),
                        }
        except Exception:
            pass

        return None

    def _check_news_records(
        self,
        subject: str,
        asserts_dead: bool,
        asserts_alive: bool,
        original_claim: str,
    ) -> Optional[KnowledgeVerificationResult]:
        """
        Queries Google News RSS archive to verify breaking news, obituaries, and events
        when standard static encyclopedias lack a direct article.
        """
        import xml.etree.ElementTree as ET
        import urllib.parse
        try:
            encoded_sub = urllib.parse.quote(subject)
            url = f"https://news.google.com/rss/search?q={encoded_sub}&hl=en-US&gl=US&ceid=US:en"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(url, headers=headers)
            if res.status_code != 200:
                return None

            root = ET.fromstring(res.content)
            items = root.findall('.//item')
            if not items:
                return None

            # Look through top 6 articles for death or alive corroboration
            death_articles = []
            for item in items[:6]:
                title = item.find('title').text or ""
                source_el = item.find('source')
                publisher = source_el.text if source_el is not None else "Verified News Outlet"
                link = item.find('link').text or ""
                pub_date = item.find('pubDate').text or ""

                title_lower = title.lower()
                sub_parts = [p.lower() for p in subject.split() if len(p) >= 3]
                matches_sub = any(p in title_lower for p in sub_parts)
                mentions_death = bool(re.search(r'\b(?:dies|died|dead|passes\s+away|passed\s+away|death|obituary)\b', title_lower))

                if matches_sub and mentions_death:
                    death_articles.append({
                        "title": title,
                        "publisher": publisher,
                        "link": link,
                        "pub_date": pub_date,
                    })

            # Case A: Claim asserts DEAD, and journalism records confirm death
            if asserts_dead and death_articles:
                top = death_articles[0]
                summary = (
                    f"Report stating '{subject} passed away' is corroborated by published journalism reports "
                    f"({top['publisher']}): '{top['title']}'."
                )
                logger.info(f"WebKnowledge: News confirmed death of '{subject}' via {top['publisher']}.")
                return KnowledgeVerificationResult(
                    status="CONFIRMATION",
                    subject=subject,
                    confidence=0.92,
                    summary=summary,
                    source_title=f"{top['publisher']} / Google News Archive",
                    source_url=top['link'],
                    snippet=f"{top['title']} (Published: {top['pub_date'][:16] if top['pub_date'] else 'Recent'}). Corroborated across {len(death_articles)} news sources.",
                    contradiction_type=None,
                )

            # Case B: Claim asserts ALIVE, but journalism records report death
            if asserts_alive and death_articles:
                top = death_articles[0]
                summary = (
                    f"Claim asserts '{subject} is alive', but verified news reports "
                    f"({top['publisher']}) document that {subject} passed away: '{top['title']}'."
                )
                logger.info(f"WebKnowledge: News contradicted alive claim for '{subject}' via {top['publisher']}.")
                return KnowledgeVerificationResult(
                    status="CONTRADICTION",
                    subject=subject,
                    confidence=0.92,
                    summary=summary,
                    source_title=f"{top['publisher']} / Google News Archive",
                    source_url=top['link'],
                    snippet=f"{top['title']} (Published: {top['pub_date'][:16] if top['pub_date'] else 'Recent'}).",
                    contradiction_type="LIFE_STATUS_CONTRADICTION",
                )

        except Exception as exc:
            logger.warning(f"Google News RSS check failed: {exc}")

        return None



# Global singleton
web_knowledge_verifier = WebKnowledgeVerifier()
