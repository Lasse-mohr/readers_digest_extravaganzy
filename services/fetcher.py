"""
arXiv paper fetcher.

Contract (from synthesis_model_spec.md):

    def fetch_arxiv(
        categories: list[str],
        date_from: date,
        date_to: date,
    ) -> list[Paper]: ...

Returns raw, un-deduplicated results. Deduplication is the synthesis model's
responsibility. The fetcher's job is to retrieve every available paper and
return it in the shared Paper format.
"""

import asyncio
import re
from datetime import date, datetime
from typing import Optional

import feedparser
import httpx

from shared.types import Author, Paper

ARXIV_API = "https://export.arxiv.org/api/query"
MAX_RESULTS_PER_REQUEST = 200
REQUEST_DELAY = 1.0  # seconds between paginated requests


# ── Helpers ────────────────────────────────────────────────────────────────


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


def _parse_authors(entry) -> list[Author]:
    return [
        Author(name=a.get("name", "").strip())
        for a in getattr(entry, "authors", [])
    ]


def _build_query(categories: list[str], date_from: date, date_to: date) -> str:
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    date_filter = (
        f"submittedDate:[{date_from.strftime('%Y%m%d')}000000 "
        f"TO {date_to.strftime('%Y%m%d')}235959]"
    )
    return f"({cat_query}) AND {date_filter}"


# ── Core fetch ─────────────────────────────────────────────────────────────


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
    response = await client.get(ARXIV_API, params=params, timeout=30.0)
    response.raise_for_status()
    return feedparser.parse(response.text)


async def fetch_arxiv(
    categories: list[str],
    date_from: date,
    date_to: date,
) -> list[Paper]:
    """
    Fetch papers from arXiv for the given categories and date window.

    Returns raw results — no deduplication. The synthesis model deduplicates
    across fetcher outputs.
    """
    query = _build_query(categories, date_from, date_to)
    papers: list[Paper] = []

    async with httpx.AsyncClient() as client:
        start = 0
        while True:
            feed = await _fetch_page(client, query, start, MAX_RESULTS_PER_REQUEST)

            entries = feed.get("entries", [])
            if not entries:
                break

            for entry in entries:
                arxiv_id = _extract_arxiv_id(entry.get("id", ""))
                title = re.sub(r"\s+", " ", entry.get("title", "")).strip()

                abstract = entry.get("summary", "").strip() or None

                # entry.arxiv_doi is set when the paper has a published DOI
                doi = _normalize_doi(entry.get("arxiv_doi") or entry.get("doi"))

                # journal_ref is author-supplied; use verbatim per spec
                journal = entry.get("arxiv_journal_ref") or None
                if journal:
                    journal = journal.strip()

                authors = _parse_authors(entry)

                published_str = entry.get("published", "")
                try:
                    published_date = datetime.strptime(published_str[:10], "%Y-%m-%d").date()
                except ValueError:
                    published_date = date_to

                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

                papers.append(
                    Paper(
                        doi=doi,
                        arxiv_id=arxiv_id,
                        openalex_id=None,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal,
                        journal_issn=None,  # arXiv papers don't carry ISSNs
                        published_date=published_date,
                        source="arxiv",
                        url=arxiv_url,
                    )
                )

            total = int(feed.feed.get("opensearch_totalresults", 0))
            start += len(entries)
            if start >= total or len(entries) < MAX_RESULTS_PER_REQUEST:
                break

            await asyncio.sleep(REQUEST_DELAY)

    return papers


# ── Entry point for manual testing ────────────────────────────────────────


if __name__ == "__main__":
    from datetime import timedelta
    from pathlib import Path

    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    categories = config.get("arxiv_categories", ["q-bio.GN"])
    lookback = config.get("digest", {}).get("lookback_days", 7)
    date_to = date.today()
    date_from = date_to - timedelta(days=lookback)

    print(f"Fetching arXiv papers for: {categories}")
    print(f"Window: {date_from} → {date_to}\n")

    papers = asyncio.run(fetch_arxiv(categories, date_from, date_to))
    print(f"Fetched {len(papers)} papers.\n")

    with_doi = sum(1 for p in papers if p.doi)
    with_abstract = sum(1 for p in papers if p.abstract)
    print(f"  with DOI:      {with_doi}/{len(papers)}")
    print(f"  with abstract: {with_abstract}/{len(papers)}\n")

    for p in papers[:5]:
        print(f"  [{p.arxiv_id}] {p.title[:80]}")
        print(f"    doi:      {p.doi}")
        print(f"    authors:  {', '.join(a.name for a in p.authors[:3])}")
        print(f"    journal:  {p.journal}")
        print(f"    date:     {p.published_date}")
        print(f"    abstract: {(p.abstract or '')[:120]}...")
        print()
