import asyncio
import logging
from datetime import date, datetime
from functools import partial
from typing import Optional

import pyalex
from pyalex import Works

from db.database import Paper

logger = logging.getLogger(__name__)


def init_openalex(api_key: str):
    """Call once at app startup."""
    pyalex.config.api_key = api_key
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 503]


# ── Journal fetch ──────────────────────────────────────────────────────────────

def fetch_works_by_journals(
    issns: list[str],
    from_date: date,
    to_date: date,
) -> list[dict]:
    """
    Fetch works published in the given journals during the date window.

    Uses OR-filter on ISSNs to minimise API calls (one request set for all journals).
    OpenAlex allows up to 50 values in an OR filter; split into chunks if needed.
    """
    all_works: list[dict] = []

    # OR-filter: "0028-0836|0036-8075|..."  (max 50 per request)
    chunks = [issns[i:i + 50] for i in range(0, len(issns), 50)]

    for chunk in chunks:
        issn_filter = "|".join(chunk)
        pager = (
            Works()
            .filter(
                primary_location={"source": {"issn": issn_filter}},
                from_publication_date=from_date.isoformat(),
                to_publication_date=to_date.isoformat(),
            )
            .select([
                "id", "doi", "title", "abstract_inverted_index",
                "authorships", "primary_location", "publication_date",
            ])
            .paginate(per_page=200, n_max=None)
        )
        for page in pager:
            all_works.extend(page)
            logger.debug("Page received: %d works (total so far: %d)", len(page), len(all_works))

    return all_works


async def fetch_works_by_journals_async(
    issns: list[str],
    from_date: date,
    to_date: date,
) -> list[dict]:
    """Async wrapper — runs the synchronous pyalex call in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(fetch_works_by_journals, issns, from_date, to_date),
    )


# ── Abstract reconstruction ────────────────────────────────────────────────────

def reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """
    Reconstruct plaintext abstract from OpenAlex's inverted index format.

    pyalex does this automatically when you access work["abstract"], but this
    utility is available as a fallback for raw dict handling.
    """
    if not inverted_index:
        return None
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


# ── DOI lookup ────────────────────────────────────────────────────────────────


def fetch_abstract_by_doi(doi: str) -> Optional[str]:
    """
    Fetch the abstract for a single paper by DOI.

    Returns None if the work is not found in OpenAlex or has no abstract.
    The DOI can be bare (10.1038/...) or include the https://doi.org/ prefix.
    """
    if not doi.startswith("https://doi.org/"):
        doi = f"https://doi.org/{doi}"
    try:
        work = Works()[doi]
    except Exception:
        logger.warning("OpenAlex lookup failed for DOI %s", doi, exc_info=True)
        return None
    return work.get("abstract") or reconstruct_abstract(work.get("abstract_inverted_index"))


async def fetch_abstract_by_doi_async(doi: str) -> Optional[str]:
    """Async wrapper for fetch_abstract_by_doi."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fetch_abstract_by_doi, doi))


# ── Data normalisation ─────────────────────────────────────────────────────────

def openalex_work_to_paper(work: dict) -> Paper:
    """Map an OpenAlex work dict to our Paper model."""
    # DOI: strip prefix, lowercase
    doi = work.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    if doi:
        doi = doi.lower()

    # Authors (truncated at 100 by OpenAlex for list queries — acceptable for our use case)
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author") or {}
        authors.append({
            "name": author.get("display_name"),
            "openalex_id": author.get("id"),
            "orcid": author.get("orcid"),
        })

    # Journal from primary_location
    journal: Optional[str] = None
    primary_loc = work.get("primary_location") or {}
    source_obj = primary_loc.get("source") or {}
    if source_obj:
        journal = source_obj.get("display_name")

    # Abstract: pyalex auto-reconstructs when accessing work["abstract"];
    # fall back to manual reconstruction from the raw inverted index.
    abstract = work.get("abstract") or reconstruct_abstract(
        work.get("abstract_inverted_index")
    )

    # Publication date
    pub_date: Optional[date] = None
    raw_date = work.get("publication_date")
    if isinstance(raw_date, str):
        try:
            pub_date = date.fromisoformat(raw_date)
        except ValueError:
            logger.warning("Unparseable publication_date %r for work %s", raw_date, work.get("id"))

    paper_id = doi or work.get("id")

    return Paper(
        id=paper_id,
        title=work.get("title") or work.get("display_name") or "",
        abstract=abstract,
        authors=authors,
        journal=journal,
        published_date=pub_date,
        doi=doi,
        arxiv_id=None,
        openalex_id=work.get("id"),
        source="openalex",
        fetched_at=datetime.utcnow(),
    )
