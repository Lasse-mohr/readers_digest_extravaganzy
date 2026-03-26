"""Tests for services.bluesky — Bluesky academic paper sighting extraction."""

import asyncio
import unittest
from datetime import date, timedelta

from shared.types import BlueskySighting
from services.bluesky import (
    extract_paper_id,
    extract_urls_from_post,
    fetch_bluesky,
    _parse_post_date,
    _get_post_url,
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


class TestMakeSighting(unittest.TestCase):
    """Unit tests for _make_sighting."""

    def test_doi_sighting(self) -> None:
        s = _make_sighting(
            "doi", "10.1038/test", "user.bsky.social",
            "https://bsky.app/profile/user.bsky.social/post/abc",
            date(2025, 3, 20), "Great paper!",
        )
        assert s is not None
        assert isinstance(s, BlueskySighting)
        assert s.doi == "10.1038/test"
        assert s.arxiv_id is None
        assert s.handle == "user.bsky.social"
        assert s.post_text == "Great paper!"

    def test_arxiv_sighting(self) -> None:
        s = _make_sighting(
            "arxiv", "2401.12345", "user.bsky.social",
            "https://bsky.app/profile/user.bsky.social/post/abc",
            date(2025, 3, 20), None,
        )
        assert s is not None
        assert s.doi is None
        assert s.arxiv_id == "2401.12345"

    def test_pmid_discarded(self) -> None:
        s = _make_sighting(
            "pmid", "12345678", "user.bsky.social",
            "https://bsky.app/profile/user.bsky.social/post/abc",
            date(2025, 3, 20), None,
        )
        assert s is None


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
            print(f"  {paper_id} — {s.handle} on {s.posted_at}")

        # Verify return type
        assert isinstance(results, list)
        for s in results:
            assert isinstance(s, BlueskySighting)
            assert isinstance(s.handle, str)
            assert isinstance(s.post_url, str)
            assert isinstance(s.posted_at, date)
            assert s.doi is not None or s.arxiv_id is not None

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
        # Very old date range — should return nothing from a recent feed
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
