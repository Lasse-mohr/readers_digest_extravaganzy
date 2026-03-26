"""
Unit tests for the arXiv fetcher.

Tests cover:
  - Helper functions (_extract_arxiv_id, _normalize_doi, _build_query)
  - Full fetch_arxiv with mocked HTTP responses
  - Edge cases: missing fields, empty results, pagination
"""

import asyncio
from datetime import date
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from services.fetcher import (
    _build_query,
    _extract_arxiv_id,
    _normalize_doi,
    fetch_arxiv,
)
from shared.types import Paper


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_ATOM_SINGLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>200</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title>  A Test Paper With   Extra Whitespace  </title>
    <summary>This is the abstract of a test paper.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">10.1038/s41586-024-00001-1</arxiv:doi>
    <arxiv:journal_ref xmlns:arxiv="http://arxiv.org/schemas/atom">Nature 625, 123-130 (2024)</arxiv:journal_ref>
  </entry>
</feed>"""

SAMPLE_ATOM_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>0</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>200</opensearch:itemsPerPage>
</feed>"""

SAMPLE_ATOM_NO_OPTIONAL = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2403.99999v1</id>
    <title>Minimal Paper</title>
    <summary>Short abstract.</summary>
    <published>2024-03-01T00:00:00Z</published>
    <author><name>Solo Author</name></author>
  </entry>
</feed>"""


# ── Helper tests ──────────────────────────────────────────────────────────


class TestExtractArxivId:
    def test_standard_url_with_version(self):
        assert _extract_arxiv_id("http://arxiv.org/abs/2401.12345v2") == "2401.12345"

    def test_standard_url_without_version(self):
        assert _extract_arxiv_id("http://arxiv.org/abs/2401.12345") == "2401.12345"

    def test_old_style_id(self):
        assert _extract_arxiv_id("http://arxiv.org/abs/hep-th/9901001v1") == "hep-th/9901001"

    def test_bare_id_fallback(self):
        assert _extract_arxiv_id("2401.12345") == "2401.12345"


class TestNormalizeDoi:
    def test_none_returns_none(self):
        assert _normalize_doi(None) is None

    def test_empty_returns_none(self):
        assert _normalize_doi("") is None

    def test_strips_https_prefix(self):
        assert _normalize_doi("https://doi.org/10.1038/S41586-024-00001-1") == "10.1038/s41586-024-00001-1"

    def test_strips_http_prefix(self):
        assert _normalize_doi("http://doi.org/10.1000/TEST") == "10.1000/test"

    def test_strips_bare_prefix(self):
        assert _normalize_doi("doi.org/10.1000/TEST") == "10.1000/test"

    def test_plain_doi_lowercased(self):
        assert _normalize_doi("10.1038/S41586-024-00001-1") == "10.1038/s41586-024-00001-1"

    def test_whitespace_stripped(self):
        assert _normalize_doi("  10.1000/test  ") == "10.1000/test"


class TestBuildQuery:
    def test_single_category(self):
        q = _build_query(["q-bio.GN"], date(2024, 1, 1), date(2024, 1, 7))
        assert "cat:q-bio.GN" in q
        assert "20240101" in q
        assert "20240107" in q

    def test_multiple_categories(self):
        q = _build_query(["q-bio.GN", "cs.LG"], date(2024, 1, 1), date(2024, 1, 7))
        assert "cat:q-bio.GN OR cat:cs.LG" in q

    def test_date_format(self):
        q = _build_query(["cs.AI"], date(2024, 3, 15), date(2024, 3, 22))
        assert "submittedDate:[20240315000000 TO 20240322235959]" in q


# ── Integration tests with mocked HTTP ────────────────────────────────────


def _mock_response(xml_body: str) -> httpx.Response:
    return httpx.Response(200, text=xml_body, request=httpx.Request("GET", "https://test"))


@pytest.mark.asyncio
async def test_fetch_single_paper():
    with patch("services.fetcher.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(SAMPLE_ATOM_SINGLE)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        papers = await fetch_arxiv(["q-bio.GN"], date(2024, 1, 1), date(2024, 1, 31))

    assert len(papers) == 1
    p = papers[0]

    assert isinstance(p, Paper)
    assert p.arxiv_id == "2401.12345"
    assert p.doi == "10.1038/s41586-024-00001-1"
    assert p.title == "A Test Paper With Extra Whitespace"
    assert p.abstract == "This is the abstract of a test paper."
    assert p.journal == "Nature 625, 123-130 (2024)"
    assert p.published_date == date(2024, 1, 15)
    assert p.source == "arxiv"
    assert p.openalex_id is None
    assert p.journal_issn is None
    assert p.url == "https://arxiv.org/abs/2401.12345"

    assert len(p.authors) == 2
    assert p.authors[0].name == "Alice Smith"
    assert p.authors[1].name == "Bob Jones"
    assert p.authors[0].openalex_id is None
    assert p.authors[0].orcid is None


@pytest.mark.asyncio
async def test_fetch_empty_results():
    with patch("services.fetcher.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(SAMPLE_ATOM_EMPTY)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        papers = await fetch_arxiv(["q-bio.GN"], date(2024, 1, 1), date(2024, 1, 7))

    assert papers == []


@pytest.mark.asyncio
async def test_fetch_paper_without_doi_or_journal():
    with patch("services.fetcher.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(SAMPLE_ATOM_NO_OPTIONAL)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        papers = await fetch_arxiv(["cs.LG"], date(2024, 3, 1), date(2024, 3, 7))

    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2403.99999"
    assert p.doi is None
    assert p.journal is None
    assert len(p.authors) == 1
    assert p.authors[0].name == "Solo Author"


@pytest.mark.asyncio
async def test_fetch_builds_correct_query():
    """Verify the HTTP request is made with the right query parameters."""
    with patch("services.fetcher.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(SAMPLE_ATOM_EMPTY)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        await fetch_arxiv(["q-bio.GN", "cs.LG"], date(2024, 6, 1), date(2024, 6, 7))

    call_args = mock_client.get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params")
    query = params["search_query"]

    assert "cat:q-bio.GN OR cat:cs.LG" in query
    assert "20240601" in query
    assert "20240607" in query
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"
    assert params["max_results"] == 200
