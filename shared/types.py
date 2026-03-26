from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Author:
    """
    A single author on a paper.
    All fields except `name` are optional — not every source provides them.
    At least one of openalex_id or orcid should be populated when available,
    as these are used by the scorer to match against the priority author list.
    """
    name: str
    openalex_id: Optional[str] = None
    orcid: Optional[str] = None


@dataclass
class Paper:
    """
    A single paper record as returned by a fetcher.

    Fetchers must populate every field they have data for.
    Fields marked Optional may be None if genuinely unavailable —
    do not substitute empty strings for None.

    Identity fields (doi, arxiv_id, openalex_id):
      - At least one must be non-None.
      - doi is preferred as the canonical dedup key.
      - Normalise DOIs to lowercase, strip leading "https://doi.org/".
    """
    # Identity
    doi: Optional[str]
    arxiv_id: Optional[str]
    openalex_id: Optional[str]

    # Content
    title: str
    abstract: Optional[str]
    authors: list[Author]

    # Provenance
    journal: Optional[str]
    journal_issn: Optional[str]
    published_date: date
    source: str

    # URLs
    url: Optional[str]


@dataclass
class BlueskySighting:
    """
    A record of one paper being linked by one Bluesky account.
    One paper linked by three accounts produces three BlueskySightings.
    """
    doi: Optional[str]
    arxiv_id: Optional[str]
    handle: str
    post_url: str
    posted_at: date
    post_text: Optional[str]
