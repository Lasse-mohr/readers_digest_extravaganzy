"""Tests for services.bluesky — Bluesky academic paper sighting extraction."""

import asyncio
import unittest
from datetime import date, timedelta

from shared.types import BlueskyEngagement, BlueskySighting, Paper
from services.bluesky import (
    extract_paper_id,
    extract_urls_from_post,
    fetch_bluesky,
    promote_trending_sightings,
    _parse_post_date,
    _get_post_url,
    _get_engagement,
    _make_commentary,
    _make_sighting,
)


class TestExtractPaperId(unittest.TestCase):
    """Unit tests for extract_paper_id."""

    def test_doi_org(self) -> None:
        result = extract_paper_id("https://doi.org/10.1038/s41586-024-00001-1")
        assert result == ("doi", "10.1038/s41586-024-00001-1")

    def test_arxiv(self) -> None:
        result = extract_paper_id("https://arxiv.org/abs/2401.12345")
        assert result == ("arxiv", "2401.12345")

    def test_nature(self) -> None:
        result = extract_paper_id("https://www.nature.com/articles/s41586-024-00001-1")
        assert result is not None
        assert result[0] == "doi"
        assert "10.1038" in result[1]

    def test_biorxiv(self) -> None:
        result = extract_paper_id(
            "https://www.biorxiv.org/content/10.1101/2024.01.01.123456"
        )
        assert result is not None
        assert result[0] == "doi"
        assert result[1].startswith("10.1101/")

    def test_medrxiv(self) -> None:
        result = extract_paper_id(
            "https://www.medrxiv.org/content/10.1101/2024.02.02.654321"
        )
        assert result is not None
        assert result[0] == "doi"

    def test_science(self) -> None:
        result = extract_paper_id(
            "https://www.science.org/doi/10.1126/science.abcdefg"
        )
        assert result == ("doi", "10.1126/science.abcdefg")

    def test_pubmed(self) -> None:
        result = extract_paper_id("https://pubmed.ncbi.nlm.nih.gov/12345678")
        assert result == ("pmid", "12345678")

    def test_unknown_url(self) -> None:
        assert extract_paper_id("https://example.com/some/page") is None

    def test_doi_lowercase(self) -> None:
        result = extract_paper_id("https://doi.org/10.1038/S41586-024-00001-1")
        assert result is not None
        assert result[1] == "10.1038/s41586-024-00001-1"


class TestExtractUrlsFromPost(unittest.TestCase):
    """Unit tests for extract_urls_from_post."""

    def test_facet_links(self) -> None:
        post = {
            "post": {
                "record": {
                    "text": "Check out this paper",
                    "facets": [
                        {
                            "features": [
                                {
                                    "$type": "app.bsky.richtext.facet#link",
                                    "uri": "https://doi.org/10.1038/test",
                                }
                            ]
                        }
                    ],
                }
            }
        }
        urls = extract_urls_from_post(post)
        assert urls == ["https://doi.org/10.1038/test"]

    def test_text_fallback(self) -> None:
        post = {
            "post": {
                "record": {
                    "text": "Check https://arxiv.org/abs/2401.12345 cool paper",
                }
            }
        }
        urls = extract_urls_from_post(post)
        assert "https://arxiv.org/abs/2401.12345" in urls

    def test_empty_post(self) -> None:
        post = {"post": {"record": {"text": "No links here"}}}
        assert extract_urls_from_post(post) == []


class TestParsePostDate(unittest.TestCase):
    """Unit tests for _parse_post_date."""

    def test_valid_iso_date(self) -> None:
        post = {"post": {"record": {"createdAt": "2025-03-20T14:30:00Z"}}}
        assert _parse_post_date(post) == date(2025, 3, 20)

    def test_missing_date(self) -> None:
        post = {"post": {"record": {}}}
        assert _parse_post_date(post) is None

    def test_with_timezone_offset(self) -> None:
        post = {"post": {"record": {"createdAt": "2025-03-20T14:30:00+02:00"}}}
        assert _parse_post_date(post) == date(2025, 3, 20)


class TestGetPostUrl(unittest.TestCase):
    """Unit tests for _get_post_url."""

    def test_constructs_url(self) -> None:
        post = {
            "post": {
                "uri": "at://did:plc:abc123/app.bsky.feed.post/xyz789",
                "author": {"handle": "user.bsky.social"},
            }
        }
        url = _get_post_url(post)
        assert url == "https://bsky.app/profile/user.bsky.social/post/xyz789"


class TestGetEngagement(unittest.TestCase):
    """Unit tests for _get_engagement."""

    def test_all_counts(self) -> None:
        post = {
            "post": {
                "likeCount": 42,
                "replyCount": 5,
                "repostCount": 10,
                "quoteCount": 3,
            }
        }
        eng = _get_engagement(post)
        assert isinstance(eng, BlueskyEngagement)
        assert eng.like_count == 42
        assert eng.reply_count == 5
        assert eng.repost_count == 10
        assert eng.quote_count == 3
        assert eng.total == 60

    def test_missing_counts_default_zero(self) -> None:
        post = {"post": {"likeCount": 7}}
        eng = _get_engagement(post)
        assert eng.like_count == 7
        assert eng.reply_count == 0
        assert eng.repost_count == 0
        assert eng.quote_count == 0
        assert eng.total == 7

    def test_empty_post(self) -> None:
        eng = _get_engagement({"post": {}})
        assert eng.total == 0


class TestMakeCommentary(unittest.TestCase):
    """Unit tests for _make_commentary."""

    def test_strips_urls(self) -> None:
        text = "Great paper! https://doi.org/10.1038/test Check it out"
        commentary, has = _make_commentary(text)
        assert commentary == "Great paper!  Check it out"
        assert has is False  # under 80 chars

    def test_long_commentary(self) -> None:
        text = (
            "This is a really insightful paper that changes how we think about "
            "protein folding and its implications for drug design https://example.com/paper"
        )
        commentary, has = _make_commentary(text)
        assert "https" not in commentary
        assert has is True

    def test_url_only(self) -> None:
        commentary, has = _make_commentary("https://doi.org/10.1038/test")
        assert commentary is None
        assert has is False

    def test_none_text(self) -> None:
        commentary, has = _make_commentary(None)
        assert commentary is None
        assert has is False


class TestEngagementScore(unittest.TestCase):
    """Unit tests for BlueskySighting.engagement_score."""

    def _sighting(self, total: int) -> BlueskySighting:
        """Helper to build a sighting with a given total engagement."""
        return BlueskySighting(
            doi="10.1038/test", arxiv_id=None, handle="u.bsky.social",
            post_url="https://bsky.app/profile/u.bsky.social/post/x",
            posted_at=date(2025, 3, 20),
            engagement=BlueskyEngagement(
                like_count=total, reply_count=0, repost_count=0, quote_count=0,
            ),
        )

    def test_zero_engagement(self) -> None:
        s = self._sighting(0)
        assert s.engagement_score == 0.0

    def test_ten_engagement(self) -> None:
        score = self._sighting(10).engagement_score
        assert 0.45 < score < 0.55, f"Expected ~0.5, got {score}"

    def test_hundred_engagement(self) -> None:
        score = self._sighting(100).engagement_score
        assert score == 1.0

    def test_thousand_capped(self) -> None:
        score = self._sighting(1000).engagement_score
        assert score == 1.0

    def test_no_engagement_object(self) -> None:
        s = BlueskySighting(
            doi="10.1038/test", arxiv_id=None, handle="u.bsky.social",
            post_url="https://bsky.app/profile/u.bsky.social/post/x",
            posted_at=date(2025, 3, 20),
        )
        assert s.engagement_score == 0.0

    def test_monotonically_increasing(self) -> None:
        scores = [self._sighting(n).engagement_score for n in [0, 1, 5, 10, 50, 100]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], f"Not monotonic at {i}: {scores}"


class TestMakeSighting(unittest.TestCase):
    """Unit tests for _make_sighting."""

    def test_doi_sighting(self) -> None:
        eng = BlueskyEngagement(like_count=10, reply_count=2, repost_count=3, quote_count=1)
        s = _make_sighting(
            "doi", "10.1038/test", "user.bsky.social",
            "https://bsky.app/profile/user.bsky.social/post/abc",
            date(2025, 3, 20), "Great paper!",
            eng, "Great paper!", False,
        )
        assert s is not None
        assert isinstance(s, BlueskySighting)
        assert s.doi == "10.1038/test"
        assert s.arxiv_id is None
        assert s.handle == "user.bsky.social"
        assert s.post_text == "Great paper!"
        assert s.engagement is not None
        assert s.engagement.total == 16
        assert s.engagement_score > 0.0
        assert s.commentary == "Great paper!"
        assert s.has_commentary is False

    def test_arxiv_sighting(self) -> None:
        s = _make_sighting(
            "arxiv", "2401.12345", "user.bsky.social",
            "https://bsky.app/profile/user.bsky.social/post/abc",
            date(2025, 3, 20), None,
        )
        assert s is not None
        assert s.doi is None
        assert s.arxiv_id == "2401.12345"
        assert s.engagement is None
        assert s.engagement_score == 0.0
        assert s.commentary is None
        assert s.has_commentary is False

    def test_pmid_discarded(self) -> None:
        s = _make_sighting(
            "pmid", "12345678", "user.bsky.social",
            "https://bsky.app/profile/user.bsky.social/post/abc",
            date(2025, 3, 20), None,
        )
        assert s is None


class TestPromoteTrendingSightings(unittest.TestCase):
    """Unit tests for promote_trending_sightings."""

    def _sighting(self, doi=None, arxiv_id=None, handle="a.bsky.social") -> BlueskySighting:
        return BlueskySighting(
            doi=doi, arxiv_id=arxiv_id, handle=handle,
            post_url="https://bsky.app/profile/x/post/1",
            posted_at=date(2025, 3, 20),
        )

    def test_trending_doi_promoted(self) -> None:
        sightings = [
            self._sighting(doi="10.1038/new", handle="alice.bsky.social"),
            self._sighting(doi="10.1038/new", handle="bob.bsky.social"),
        ]
        result = asyncio.run(promote_trending_sightings(sightings, []))
        assert len(result) == 1
        assert isinstance(result[0], Paper)
        assert result[0].doi == "10.1038/new"
        assert result[0].source == "bluesky"

    def test_trending_arxiv_promoted(self) -> None:
        sightings = [
            self._sighting(arxiv_id="2401.99999", handle="alice.bsky.social"),
            self._sighting(arxiv_id="2401.99999", handle="bob.bsky.social"),
        ]
        result = asyncio.run(promote_trending_sightings(sightings, []))
        assert len(result) == 1
        assert result[0].arxiv_id == "2401.99999"

    def test_single_handle_not_promoted(self) -> None:
        sightings = [
            self._sighting(doi="10.1038/solo", handle="alice.bsky.social"),
            self._sighting(doi="10.1038/solo", handle="alice.bsky.social"),
        ]
        result = asyncio.run(promote_trending_sightings(sightings, []))
        assert result == []

    def test_already_in_corpus_not_promoted(self) -> None:
        sightings = [
            self._sighting(doi="10.1038/exists", handle="alice.bsky.social"),
            self._sighting(doi="10.1038/exists", handle="bob.bsky.social"),
        ]
        existing = [Paper(
            doi="10.1038/exists", arxiv_id=None, openalex_id=None,
            title="Existing", abstract=None, authors=[],
            journal=None, journal_issn=None, published_date=date(2025, 3, 20),
            source="arxiv", url=None,
        )]
        result = asyncio.run(promote_trending_sightings(sightings, existing))
        assert result == []

    def test_empty_sightings(self) -> None:
        result = asyncio.run(promote_trending_sightings([], []))
        assert result == []


class TestFetchBlueskyIntegration(unittest.TestCase):
    """Integration test — calls the real Bluesky API."""

    def test_real_handles(self) -> None:
        today = date.today()
        results = asyncio.run(
            fetch_bluesky(
                ["atproto.bsky.social"],
                today - timedelta(days=7),
                today,
            )
        )

        print(f"\nFound {len(results)} sightings:")
        for s in results:
            paper_id = s.doi or s.arxiv_id
            eng = s.engagement
            eng_str = f"[{eng.like_count}L {eng.repost_count}R {eng.reply_count}C {eng.quote_count}Q]" if eng else "[no engagement]"
            print(f"  {paper_id} — {s.handle} on {s.posted_at} {eng_str}")

        assert isinstance(results, list)
        for s in results:
            assert isinstance(s, BlueskySighting)
            assert isinstance(s.handle, str)
            assert isinstance(s.post_url, str)
            assert isinstance(s.posted_at, date)
            assert s.doi is not None or s.arxiv_id is not None
            assert isinstance(s.engagement, BlueskyEngagement)
            assert isinstance(s.has_commentary, bool)

    def test_empty_handles(self) -> None:
        today = date.today()
        results = asyncio.run(
            fetch_bluesky([], today - timedelta(days=7), today)
        )
        assert results == []

    def test_invalid_handle(self) -> None:
        today = date.today()
        results = asyncio.run(
            fetch_bluesky(
                ["this-handle-does-not-exist-xyz123.bsky.social"],
                today - timedelta(days=7),
                today,
            )
        )
        assert isinstance(results, list)

    def test_date_filtering(self) -> None:
        results = asyncio.run(
            fetch_bluesky(
                ["atproto.bsky.social"],
                date(2020, 1, 1),
                date(2020, 1, 7),
            )
        )
        assert results == []


if __name__ == "__main__":
    unittest.main()
