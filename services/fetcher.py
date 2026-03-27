"""
Paper fetcher: arXiv (feedparser) + OpenAlex, with deduplication and orchestration.
"""

import asyncio
import logging
import re
from datetime import date, datetime
from typing import Optional

import feedparser
import httpx
from Levenshtein import distance as levenshtein_distance

from db.database import Paper
from services.openalex_client import fetch_works_by_journals_async, openalex_work_to_paper

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
MAX_RESULTS_PER_REQUEST = 200
REQUEST_DELAY = 1.0  # seconds between paginated requests
MAX_RETRIES = 5
INITIAL_BACKOFF = 3.0  # seconds; doubles on each retry

_TITLE_DISTANCE_THRESHOLD = 5


# ── arXiv helpers ─────────────────────────────────────────────────────────────


def _extract_arxiv_id(entry_id: str) -> str:
    """Strip version suffix and URL prefix, e.g. http://arxiv.org/abs/2401.12345v2 → 2401.12345"""
    match = re.search(r"arxiv\.org/abs/([^v]+)", entry_id)
    if match:
        return match.group(1)
    return entry_id.split("/")[-1]


def _normalize_doi(raw: Optional[str]) -> Optional[str]:
    """Lowercase and strip the https://doi.org/ prefix if present."""
    if not raw:
        return None
    doi = raw.lower().strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None


def _parse_authors(entry) -> list:
    return [{"name": a.get("name", "").strip()} for a in getattr(entry, "authors", [])]


def _build_query(categories: list[str], date_from: date, date_to: date) -> str:
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    date_filter = (
        f"submittedDate:[{date_from.strftime('%Y%m%d')}000000 "
        f"TO {date_to.strftime('%Y%m%d')}235959]"
    )
    return f"({cat_query}) AND {date_filter}"


# ── arXiv page fetch ──────────────────────────────────────────────────────────


async def _fetch_page(
    client: httpx.AsyncClient,
    query: str,
    start: int,
    max_results: int,
) -> feedparser.FeedParserDict:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(ARXIV_API, params=params, timeout=30.0)
        except httpx.TimeoutException as exc:
            last_exc = exc
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            logger.warning(
                "arXiv timeout — retrying in %.0fs (attempt %d/%d)",
                backoff, attempt + 1, MAX_RETRIES,
            )
            await asyncio.sleep(backoff)
            continue
        if response.status_code in (429, 500, 502, 503):
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            logger.warning(
                "arXiv %d — retrying in %.0fs (attempt %d/%d)",
                response.status_code, backoff, attempt + 1, MAX_RETRIES,
            )
            await asyncio.sleep(backoff)
            continue
        response.raise_for_status()
        return feedparser.parse(response.text)
    if last_exc is not None:
        raise last_exc
    response.raise_for_status()  # raise after final retry


# ── Source fetchers ────────────────────────────────────────────────────────────


async def fetch_arxiv_papers(
    categories: list[str],
    from_date: date,
    to_date: date,
) -> list[Paper]:
    query = _build_query(categories, from_date, to_date)
    papers: list[Paper] = []

    async with httpx.AsyncClient() as client:
        start = 0
        while True:
            try:
                feed = await _fetch_page(client, query, start, MAX_RESULTS_PER_REQUEST)
            except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                logger.warning(
                    "arXiv fetch failed after retries: %s. Returning %d papers so far.",
                    exc, len(papers),
                )
                break

            entries = feed.get("entries", [])
            if not entries:
                break

            for entry in entries:
                arxiv_id = _extract_arxiv_id(entry.get("id", ""))
                title = re.sub(r"\s+", " ", entry.get("title", "")).strip()
                abstract = entry.get("summary", "").strip() or None
                doi = _normalize_doi(entry.get("arxiv_doi") or entry.get("doi"))
                journal = (entry.get("arxiv_journal_ref") or "").strip() or None
                authors = _parse_authors(entry)

                published_str = entry.get("published", "")
                try:
                    published_date = datetime.strptime(published_str[:10], "%Y-%m-%d").date()
                except ValueError:
                    published_date = to_date

                paper_id = doi if doi else f"arxiv:{arxiv_id}"
                papers.append(Paper(
                    id=paper_id,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    openalex_id=None,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal=journal,
                    published_date=published_date,
                    source="arxiv",
                ))

            total = int(feed.feed.get("opensearch_totalresults", 0))
            start += len(entries)
            if start >= total or len(entries) < MAX_RESULTS_PER_REQUEST:
                break

            await asyncio.sleep(REQUEST_DELAY)

    logger.info("Fetched %d papers from arXiv", len(papers))
    return papers


async def fetch_openalex_papers(
    issns: list[str],
    from_date: date,
    to_date: date,
) -> list[Paper]:
    logger.info(
        "Fetching OpenAlex papers for %d ISSNs (%s to %s)",
        len(issns), from_date, to_date,
    )
    works = await fetch_works_by_journals_async(issns, from_date, to_date)
    papers: list[Paper] = []
    for work in works:
        try:
            paper = openalex_work_to_paper(work)
            if paper.id:
                papers.append(paper)
        except Exception:
            logger.warning("Failed to convert work %s", work.get("id"), exc_info=True)
    logger.info("Received %d papers from OpenAlex", len(papers))
    return papers


# ── Deduplication ─────────────────────────────────────────────────────────────


def _normalise_title(title: str) -> str:
    return title.lower().strip()


def _merge(existing: Paper, incoming: Paper) -> None:
    """Fill missing fields on existing from incoming, preferring non-None values."""
    if existing.abstract is None and incoming.abstract is not None:
        existing.abstract = incoming.abstract
    if existing.doi is None and incoming.doi is not None:
        existing.doi = incoming.doi
        existing.id = incoming.doi  # DOI is preferred PK
    if existing.arxiv_id is None and incoming.arxiv_id is not None:
        existing.arxiv_id = incoming.arxiv_id
    if existing.openalex_id is None and incoming.openalex_id is not None:
        existing.openalex_id = incoming.openalex_id
    if existing.journal is None and incoming.journal is not None:
        existing.journal = incoming.journal


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """
    Deduplicate by DOI → arXiv ID → fuzzy title match (Levenshtein ≤ 5).
    On conflict, merge records preserving non-None fields.
    """
    seen_dois: dict[str, Paper] = {}
    seen_arxiv_ids: dict[str, Paper] = {}
    result: list[Paper] = []

    for paper in papers:
        # 1. DOI match
        if paper.doi and paper.doi in seen_dois:
            _merge(seen_dois[paper.doi], paper)
            continue

        # 2. arXiv ID match
        if paper.arxiv_id and paper.arxiv_id in seen_arxiv_ids:
            _merge(seen_arxiv_ids[paper.arxiv_id], paper)
            continue

        # 3. Fuzzy title match against already-accepted papers
        norm_title = _normalise_title(paper.title)
        duplicate = False
        for existing in result:
            if levenshtein_distance(norm_title, _normalise_title(existing.title)) <= _TITLE_DISTANCE_THRESHOLD:
                _merge(existing, paper)
                duplicate = True
                break
        if duplicate:
            continue

        # Not a duplicate — accept and register
        if paper.doi:
            seen_dois[paper.doi] = paper
        if paper.arxiv_id:
            seen_arxiv_ids[paper.arxiv_id] = paper
        result.append(paper)

    return result


# ── Backward-compatible arXiv fetch (returns shared.types.Paper) ──────────────


async def fetch_arxiv(
    categories: list[str],
    date_from: date,
    date_to: date,
) -> list:
    """
    Fetch arXiv papers and return shared.types.Paper objects.
    Used by run_digest.py and the scorer pipeline.
    """
    from shared.types import Author
    from shared.types import Paper as SharedPaper

    db_papers = await fetch_arxiv_papers(categories, date_from, date_to)
    result = []
    for p in db_papers:
        authors = [Author(name=a.get("name", "")) for a in (p.authors or [])]
        result.append(SharedPaper(
            doi=p.doi,
            arxiv_id=p.arxiv_id,
            openalex_id=p.openalex_id,
            title=p.title,
            abstract=p.abstract,
            authors=authors,
            journal=p.journal,
            journal_issn=None,
            published_date=p.published_date,
            source=p.source,
            url=f"https://arxiv.org/abs/{p.arxiv_id}" if p.arxiv_id else None,
        ))
    return result


# ── Orchestrator ───────────────────────────────────────────────────────────────


async def run_fetch(
    issns: list[str],
    arxiv_categories: list[str],
    from_date: date,
    to_date: date,
) -> list[Paper]:
    """Fetch from all sources, deduplicate, and return the merged paper list."""
    openalex_papers, arxiv_papers = await asyncio.gather(
        fetch_openalex_papers(issns, from_date, to_date),
        fetch_arxiv_papers(arxiv_categories, from_date, to_date),
    )
    all_papers = openalex_papers + arxiv_papers
    deduped = deduplicate_papers(all_papers)
    logger.info(
        "Fetch complete: %d unique papers (from %d total)",
        len(deduped), len(all_papers),
    )
    return deduped
