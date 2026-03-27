from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from Levenshtein import distance as levenshtein_distance

from shared.types import BlueskySighting, Paper


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class ScoredPaper:
    paper: Paper
    similarity_score: float
    final_score: float
    priority_author_match: bool
    bluesky_sightings: list[BlueskySighting] = field(default_factory=list)
    trending: bool = False
    digest_section: Optional[str] = None  # "following" | "field" | "notable"
    included_in_digest: bool = False


@dataclass
class ScoringResult:
    papers_fetched: int
    papers_after_dedup: int
    papers_included: int
    sections: dict[str, list[ScoredPaper]]  # section name -> papers
    dropped: list[ScoredPaper]


# ---------------------------------------------------------------------------
# Config types expected by the scorer
# ---------------------------------------------------------------------------

@dataclass
class JournalConfig:
    name: str
    issn: str
    tier: int


@dataclass
class ScoringWeights:
    semantic_similarity: float
    journal_tier1_bonus: float
    journal_tier2_bonus: float
    priority_author_bonus: float
    bluesky_mention_bonus: float


@dataclass
class ScorerConfig:
    journals: list[JournalConfig]
    priority_author_ids: set[str]  # openalex_ids from config + coauthor cache
    scoring: ScoringWeights
    max_papers_in_digest: int
    min_similarity_threshold: float


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("allenai/specter2_base")
    return _model


def encode_profile(research_profile: str) -> np.ndarray:
    return _get_model().encode(
        "[SEP] " + research_profile, normalize_embeddings=True
    )


def encode_paper(title: str, abstract: str) -> np.ndarray:
    return _get_model().encode(
        title + " [SEP] " + abstract, normalize_embeddings=True
    )


# ---------------------------------------------------------------------------
# Merge & dedup
# ---------------------------------------------------------------------------

def _normalise_title(title: str) -> str:
    t = title.lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", t).strip()


def _merge_paper(existing: Paper, incoming: Paper) -> Paper:
    """Merge two records for the same paper, preferring non-None fields."""
    return Paper(
        doi=existing.doi or incoming.doi,
        arxiv_id=existing.arxiv_id or incoming.arxiv_id,
        openalex_id=existing.openalex_id or incoming.openalex_id,
        title=existing.title if existing.abstract else incoming.title,
        abstract=existing.abstract or incoming.abstract,
        authors=existing.authors if existing.authors else incoming.authors,
        journal=existing.journal or incoming.journal,
        journal_issn=existing.journal_issn or incoming.journal_issn,
        published_date=existing.published_date,
        source=existing.source,
        url=existing.url or incoming.url,
    )


def merge_and_dedup(
    papers_openalex: list[Paper], papers_arxiv: list[Paper]
) -> list[Paper]:
    by_doi: dict[str, Paper] = {}
    by_arxiv: dict[str, Paper] = {}
    merged: list[Paper] = []

    def _add(paper: Paper) -> None:
        # Try DOI match first
        if paper.doi and paper.doi in by_doi:
            existing = by_doi[paper.doi]
            merged_paper = _merge_paper(existing, paper)
            idx = merged.index(existing)
            merged[idx] = merged_paper
            by_doi[paper.doi] = merged_paper
            if merged_paper.arxiv_id:
                by_arxiv[merged_paper.arxiv_id] = merged_paper
            return

        # Try arXiv ID match
        if paper.arxiv_id and paper.arxiv_id in by_arxiv:
            existing = by_arxiv[paper.arxiv_id]
            merged_paper = _merge_paper(existing, paper)
            idx = merged.index(existing)
            merged[idx] = merged_paper
            by_arxiv[paper.arxiv_id] = merged_paper
            if merged_paper.doi:
                by_doi[merged_paper.doi] = merged_paper
            return

        # Try fuzzy title match
        norm_title = _normalise_title(paper.title)
        for i, existing in enumerate(merged):
            if levenshtein_distance(norm_title, _normalise_title(existing.title)) <= 5:
                merged_paper = _merge_paper(existing, paper)
                merged[i] = merged_paper
                if merged_paper.doi:
                    by_doi[merged_paper.doi] = merged_paper
                if merged_paper.arxiv_id:
                    by_arxiv[merged_paper.arxiv_id] = merged_paper
                return

        # No match — new paper
        merged.append(paper)
        if paper.doi:
            by_doi[paper.doi] = paper
        if paper.arxiv_id:
            by_arxiv[paper.arxiv_id] = paper

    for p in papers_openalex:
        _add(p)
    for p in papers_arxiv:
        _add(p)

    return merged


# ---------------------------------------------------------------------------
# Attach Bluesky sightings
# ---------------------------------------------------------------------------

def _attach_bluesky(
    papers: list[Paper], sightings: list[BlueskySighting]
) -> dict[int, list[BlueskySighting]]:
    """Returns a mapping of paper list-index -> sightings."""
    doi_index: dict[str, int] = {}
    arxiv_index: dict[str, int] = {}
    for i, p in enumerate(papers):
        if p.doi:
            doi_index[p.doi] = i
        if p.arxiv_id:
            arxiv_index[p.arxiv_id] = i

    result: dict[int, list[BlueskySighting]] = {}
    for s in sightings:
        idx = None
        if s.doi and s.doi in doi_index:
            idx = doi_index[s.doi]
        elif s.arxiv_id and s.arxiv_id in arxiv_index:
            idx = arxiv_index[s.arxiv_id]
        if idx is not None:
            result.setdefault(idx, []).append(s)
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _build_issn_tier_map(journals: list[JournalConfig]) -> dict[str, int]:
    return {j.issn: j.tier for j in journals}


def score_papers(
    papers: list[Paper],
    bluesky_sightings: list[BlueskySighting],
    profile_embedding: np.ndarray,
    config: ScorerConfig,
) -> ScoringResult:
    papers_fetched = len(papers)

    sightings_map = _attach_bluesky(papers, bluesky_sightings)
    issn_tiers = _build_issn_tier_map(config.journals)
    w = config.scoring

    scored: list[ScoredPaper] = []

    for i, paper in enumerate(papers):
        paper_sightings = sightings_map.get(i, [])
        has_bluesky = len(paper_sightings) > 0

        # Semantic similarity
        if paper.abstract:
            emb = encode_paper(paper.title, paper.abstract)
            sim = float(np.dot(profile_embedding, emb))
        else:
            sim = 0.0

        # Journal bonus
        tier = issn_tiers.get(paper.journal_issn, 0) if paper.journal_issn else 0
        if tier == 1:
            journal_bonus = w.journal_tier1_bonus
        elif tier == 2:
            journal_bonus = w.journal_tier2_bonus
        else:
            journal_bonus = 0.0

        # Author bonus
        author_match = any(
            a.openalex_id and a.openalex_id in config.priority_author_ids
            for a in paper.authors
        )
        author_bonus = w.priority_author_bonus if author_match else 0.0

        # Bluesky bonus — scaled by engagement (floor at 50% of bonus)
        if has_bluesky:
            max_eng = max(s.engagement_score for s in paper_sightings)
            bluesky_bonus = w.bluesky_mention_bonus * (0.5 + 0.5 * max_eng)
        else:
            bluesky_bonus = 0.0

        final = (sim * w.semantic_similarity) + journal_bonus + author_bonus + bluesky_bonus

        # Trending: >= 2 distinct handles
        distinct_handles = {s.handle for s in paper_sightings}
        trending = len(distinct_handles) >= 2

        scored.append(ScoredPaper(
            paper=paper,
            similarity_score=sim,
            final_score=final,
            priority_author_match=author_match,
            bluesky_sightings=paper_sightings,
            trending=trending,
        ))

    # Filter: drop papers below threshold unless they have bluesky or author signal
    kept: list[ScoredPaper] = []
    dropped: list[ScoredPaper] = []
    for sp in scored:
        above_threshold = sp.similarity_score >= config.min_similarity_threshold
        has_signal = sp.bluesky_sightings or sp.priority_author_match
        if above_threshold or has_signal:
            kept.append(sp)
        else:
            dropped.append(sp)

    # Sort by final_score descending, cap at max
    kept.sort(key=lambda sp: sp.final_score, reverse=True)
    kept = kept[: config.max_papers_in_digest]

    # Classify into sections (mutually exclusive, priority order)
    sections: dict[str, list[ScoredPaper]] = {
        "following": [],
        "field": [],
        "notable": [],
    }
    tier1_issns = {j.issn for j in config.journals if j.tier == 1}

    for sp in kept:
        if sp.bluesky_sightings:
            sp.digest_section = "following"
        elif sp.priority_author_match or (
            sp.paper.journal_issn in tier1_issns
            and sp.similarity_score >= config.min_similarity_threshold
        ):
            sp.digest_section = "field"
        else:
            sp.digest_section = "notable"
        sp.included_in_digest = True
        sections[sp.digest_section].append(sp)

    return ScoringResult(
        papers_fetched=papers_fetched,
        papers_after_dedup=len(papers),
        papers_included=len(kept),
        sections=sections,
        dropped=dropped,
    )
