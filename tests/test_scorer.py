from datetime import date
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

from shared.types import Author, BlueskyEngagement, BlueskySighting, Paper
from services.scorer import (
    ScorerConfig,
    ScoringWeights,
    JournalConfig,
    merge_and_dedup,
    score_papers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper(
    doi: Optional[str] = "10.1234/test",
    arxiv_id: Optional[str] = None,
    title: str = "Test Paper",
    abstract: Optional[str] = "This is a test abstract about genomics.",
    source: str = "openalex",
    journal: Optional[str] = None,
    journal_issn: Optional[str] = None,
    authors: Optional[list[Author]] = None,
    openalex_id: Optional[str] = None,
) -> Paper:
    return Paper(
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        title=title,
        abstract=abstract,
        authors=authors or [],
        journal=journal,
        journal_issn=journal_issn,
        published_date=date(2026, 3, 20),
        source=source,
        url=None,
    )


def _sighting(doi=None, arxiv_id=None, handle="user.bsky.social", total_engagement=0) -> BlueskySighting:
    engagement = BlueskyEngagement(
        like_count=total_engagement, reply_count=0, repost_count=0, quote_count=0,
    ) if total_engagement > 0 else None
    return BlueskySighting(
        doi=doi,
        arxiv_id=arxiv_id,
        handle=handle,
        post_url="https://bsky.app/post/123",
        posted_at=date(2026, 3, 21),
        engagement=engagement,
    )


def _default_config(**overrides) -> ScorerConfig:
    defaults = dict(
        journals=[
            JournalConfig(name="Nature", issn="0028-0836", tier=1),
            JournalConfig(name="eLife", issn="2050-084X", tier=2),
        ],
        priority_author_ids=set(),
        scoring=ScoringWeights(
            semantic_similarity=0.60,
            journal_tier1_bonus=0.15,
            journal_tier2_bonus=0.05,
            priority_author_bonus=0.20,
            bluesky_mention_bonus=0.10,
        ),
        max_papers_in_digest=25,
        min_similarity_threshold=0.20,
    )
    defaults.update(overrides)
    return ScorerConfig(**defaults)


def _fake_embedding(*_args, **_kwargs):
    """Return a deterministic normalised vector."""
    v = np.ones(768, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Merge & dedup tests
# ---------------------------------------------------------------------------

class TestMergeAndDedup:
    def test_dedup_by_doi(self):
        p1 = _paper(doi="10.1234/a", abstract="abstract one", source="openalex")
        p2 = _paper(doi="10.1234/a", abstract=None, source="arxiv")
        result = merge_and_dedup([p1], [p2])
        assert len(result) == 1
        assert result[0].abstract == "abstract one"

    def test_dedup_by_arxiv_id(self):
        p1 = _paper(doi=None, arxiv_id="2401.00001", source="openalex")
        p2 = _paper(doi=None, arxiv_id="2401.00001", source="arxiv")
        result = merge_and_dedup([p1], [p2])
        assert len(result) == 1

    def test_dedup_by_fuzzy_title(self):
        p1 = _paper(doi="10.1234/a", title="A Novel Method for Genomics")
        p2 = _paper(doi=None, arxiv_id="2401.99999", title="A Novel Method for Genomics!")
        result = merge_and_dedup([p1], [p2])
        assert len(result) == 1
        assert result[0].doi == "10.1234/a"
        assert result[0].arxiv_id == "2401.99999"

    def test_no_dedup_different_papers(self):
        p1 = _paper(doi="10.1234/a", title="Paper A")
        p2 = _paper(doi="10.1234/b", title="Completely Different Paper B")
        result = merge_and_dedup([p1], [p2])
        assert len(result) == 2

    def test_merge_prefers_non_none_abstract(self):
        p1 = _paper(doi="10.1234/a", abstract=None)
        p2 = _paper(doi="10.1234/a", abstract="Real abstract")
        result = merge_and_dedup([p1], [p2])
        assert result[0].abstract == "Real abstract"

    def test_empty_inputs(self):
        assert merge_and_dedup([], []) == []


# ---------------------------------------------------------------------------
# Scoring tests — mock SPECTER2 so no model download needed
# ---------------------------------------------------------------------------

@patch("services.scorer.encode_paper", side_effect=_fake_embedding)
@patch("services.scorer.encode_profile", side_effect=_fake_embedding)
class TestScoring:
    def test_basic_scoring(self, _mock_profile, _mock_paper):
        papers = [_paper()]
        profile_emb = _fake_embedding()
        config = _default_config()
        result = score_papers(papers, [], profile_emb, config)

        assert result.papers_included == 1
        sp = result.sections["notable"][0]
        # cosine sim of identical normalised vectors = 1.0
        assert sp.similarity_score == pytest.approx(1.0, abs=0.01)
        expected_final = 1.0 * 0.60  # sim * weight, no bonuses
        assert sp.final_score == pytest.approx(expected_final, abs=0.01)

    def test_journal_tier1_bonus(self, _mock_profile, _mock_paper):
        papers = [_paper(journal="Nature", journal_issn="0028-0836")]
        result = score_papers(papers, [], _fake_embedding(), _default_config())
        sp = result.sections["field"][0]  # tier-1 journal -> "field" section
        assert sp.final_score == pytest.approx(0.60 + 0.15, abs=0.01)

    def test_journal_tier2_bonus(self, _mock_profile, _mock_paper):
        papers = [_paper(journal="eLife", journal_issn="2050-084X")]
        result = score_papers(papers, [], _fake_embedding(), _default_config())
        sp = result.sections["notable"][0]
        assert sp.final_score == pytest.approx(0.60 + 0.05, abs=0.01)

    def test_priority_author_bonus(self, _mock_profile, _mock_paper):
        papers = [_paper(authors=[Author(name="Jane", openalex_id="A123")])]
        config = _default_config(priority_author_ids={"A123"})
        result = score_papers(papers, [], _fake_embedding(), config)
        sp = result.sections["field"][0]
        assert sp.priority_author_match is True
        assert sp.final_score == pytest.approx(0.60 + 0.20, abs=0.01)

    def test_bluesky_bonus_no_engagement(self, _mock_profile, _mock_paper):
        """Sighting with no engagement gets 50% of bluesky bonus."""
        papers = [_paper(doi="10.1234/a")]
        sightings = [_sighting(doi="10.1234/a")]
        result = score_papers(papers, sightings, _fake_embedding(), _default_config())
        sp = result.sections["following"][0]
        # 0.10 * (0.5 + 0.5 * 0.0) = 0.05
        assert sp.final_score == pytest.approx(0.60 + 0.05, abs=0.01)
        assert len(sp.bluesky_sightings) == 1

    def test_bluesky_bonus_high_engagement(self, _mock_profile, _mock_paper):
        """Sighting with 100+ engagement gets full bluesky bonus."""
        papers = [_paper(doi="10.1234/a")]
        sightings = [_sighting(doi="10.1234/a", total_engagement=100)]
        result = score_papers(papers, sightings, _fake_embedding(), _default_config())
        sp = result.sections["following"][0]
        # 0.10 * (0.5 + 0.5 * 1.0) = 0.10
        assert sp.final_score == pytest.approx(0.60 + 0.10, abs=0.01)

    def test_bluesky_bonus_mid_engagement(self, _mock_profile, _mock_paper):
        """Sighting with ~10 engagement gets roughly 75% of bluesky bonus."""
        papers = [_paper(doi="10.1234/a")]
        sightings = [_sighting(doi="10.1234/a", total_engagement=10)]
        result = score_papers(papers, sightings, _fake_embedding(), _default_config())
        sp = result.sections["following"][0]
        # engagement_score(10) ≈ 0.52, bonus = 0.10 * (0.5 + 0.5 * 0.52) ≈ 0.076
        assert 0.65 < sp.final_score < 0.70

    def test_trending_flag(self, _mock_profile, _mock_paper):
        papers = [_paper(doi="10.1234/a")]
        sightings = [
            _sighting(doi="10.1234/a", handle="alice.bsky.social"),
            _sighting(doi="10.1234/a", handle="bob.bsky.social"),
        ]
        result = score_papers(papers, sightings, _fake_embedding(), _default_config())
        sp = result.sections["following"][0]
        assert sp.trending is True

    def test_not_trending_single_handle(self, _mock_profile, _mock_paper):
        papers = [_paper(doi="10.1234/a")]
        sightings = [
            _sighting(doi="10.1234/a", handle="alice.bsky.social"),
            _sighting(doi="10.1234/a", handle="alice.bsky.social"),
        ]
        result = score_papers(papers, sightings, _fake_embedding(), _default_config())
        sp = result.sections["following"][0]
        assert sp.trending is False

    def test_no_abstract_dropped_without_signal(self, _mock_profile, _mock_paper):
        papers = [_paper(abstract=None)]
        result = score_papers(papers, [], _fake_embedding(), _default_config())
        assert result.papers_included == 0
        assert len(result.dropped) == 1

    def test_no_abstract_kept_with_bluesky(self, _mock_profile, _mock_paper):
        papers = [_paper(doi="10.1234/a", abstract=None)]
        sightings = [_sighting(doi="10.1234/a")]
        result = score_papers(papers, sightings, _fake_embedding(), _default_config())
        assert result.papers_included == 1

    def test_no_abstract_kept_with_author_match(self, _mock_profile, _mock_paper):
        papers = [_paper(abstract=None, authors=[Author(name="Jane", openalex_id="A1")])]
        config = _default_config(priority_author_ids={"A1"})
        result = score_papers(papers, [], _fake_embedding(), config)
        assert result.papers_included == 1

    def test_max_papers_cap(self, _mock_profile, _mock_paper):
        papers = [_paper(doi=f"10.1234/{i}", title=f"Paper {i}") for i in range(30)]
        config = _default_config(max_papers_in_digest=5)
        result = score_papers(papers, [], _fake_embedding(), config)
        assert result.papers_included == 5

    def test_unmatched_sighting_discarded(self, _mock_profile, _mock_paper):
        papers = [_paper(doi="10.1234/a")]
        sightings = [_sighting(doi="10.9999/no-match")]
        result = score_papers(papers, sightings, _fake_embedding(), _default_config())
        sp = result.sections["notable"][0]
        assert len(sp.bluesky_sightings) == 0

    def test_section_priority_bluesky_over_author(self, _mock_profile, _mock_paper):
        """A paper with both bluesky and author signal goes to 'following'."""
        papers = [_paper(doi="10.1234/a", authors=[Author(name="J", openalex_id="A1")])]
        sightings = [_sighting(doi="10.1234/a")]
        config = _default_config(priority_author_ids={"A1"})
        result = score_papers(papers, sightings, _fake_embedding(), config)
        assert result.sections["following"][0].priority_author_match is True
        assert len(result.sections["field"]) == 0
