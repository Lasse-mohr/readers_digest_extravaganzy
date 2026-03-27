"""
Shared data types for all fetcher modules and the synthesis model.

All fetchers return list[Paper]. The synthesis model consumes list[Paper]
from each fetcher plus list[BlueskySighting] from the Bluesky fetcher.

Do NOT redefine these types per-module — import from here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Author:
    """
    A single author on a paper.
    At least one of openalex_id or orcid should be populated when available —
    these are used by the scorer to match against the priority author list.
    """
    name: str
    openalex_id: Optional[str] = None
    orcid: Optional[str] = None


@dataclass
class Paper:
    """
    A single paper record as returned by a fetcher.

    Identity fields (doi, arxiv_id, openalex_id):
      - At least one must be non-None.
      - doi is the preferred canonical dedup key.
      - DOIs are normalised: lowercase, no leading "https://doi.org/".
    """
    # ── Identity ──────────────────────────────────────────────────
    doi: Optional[str]
    arxiv_id: Optional[str]
    openalex_id: Optional[str]

    # ── Content ───────────────────────────────────────────────────
    title: str
    abstract: Optional[str]
    authors: list[Author]

    # ── Provenance ────────────────────────────────────────────────
    journal: Optional[str]
    journal_issn: Optional[str]
    published_date: date
    source: str  # "openalex" | "arxiv"

    # ── URLs ──────────────────────────────────────────────────────
    url: Optional[str]


@dataclass
class BlueskyEngagement:
    """Engagement metrics for a Bluesky post."""
    like_count: int
    reply_count: int
    repost_count: int
    quote_count: int

    @property
    def total(self) -> int:
        return self.like_count + self.reply_count + self.repost_count + self.quote_count


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
    post_text: Optional[str] = None
    engagement: Optional[BlueskyEngagement] = None
    commentary: Optional[str] = None
    has_commentary: bool = False

    @property
    def engagement_score(self) -> float:
        """Normalised 0-1 engagement signal on a log scale.

        Roughly: 0 engagement -> 0.0, 10 -> 0.5, 100 -> 1.0, 1000+ -> capped at 1.0.
        """
        if self.engagement is None:
            return 0.0
        return min(1.0, math.log1p(self.engagement.total) / math.log1p(100))


# ── Synthesis model types ──────────────────────────────────────────────────


@dataclass
class JournalConfig:
    name: str
    issn: str
    tier: int  # 1 or 2


@dataclass
class ScoringConfig:
    semantic_similarity_weight: float
    journal_tier1_bonus: float
    journal_tier2_bonus: float
    priority_author_bonus: float
    bluesky_mention_bonus: float


@dataclass
class DigestConfig:
    research_profile: str
    journals: list[JournalConfig]
    priority_author_ids: set[str]
    bluesky_handles: list[str]
    scoring: ScoringConfig
    max_papers_in_digest: int
    min_similarity_threshold: float
    ollama_model: str
    ollama_host: str


@dataclass
class SynthesisInput:
    papers_openalex: list[Paper]
    papers_arxiv: list[Paper]
    bluesky_sightings: list[BlueskySighting]
    config: DigestConfig
    profile_embedding: np.ndarray


@dataclass
class ScoredPaperRecord:
    paper: Paper
    similarity_score: float
    final_score: float
    priority_author_match: bool
    bluesky_sightings: list[BlueskySighting]
    trending: bool
    summary: Optional[str]
    relevance: Optional[str]
    digest_section: str  # "following" | "field" | "notable"
    included_in_digest: bool


@dataclass
class DigestResult:
    run_id: int
    created_at: datetime
    window_start: date
    window_end: date
    papers_fetched: int
    papers_after_dedup: int
    papers_included: int
    papers_dropped_no_abstract: int
    markdown_path: str
    markdown: str
    scored_papers: list[ScoredPaperRecord]
    linkedin_post: Optional[str] = None
