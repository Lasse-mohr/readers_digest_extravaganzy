"""Tests for services.linkedin — LinkedIn post generation."""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.types import Author, BlueskySighting, DigestResult, Paper, ScoredPaperRecord
from services.linkedin import (
    _build_prompt,
    _default_persona,
    _select_top_papers,
    build_linkedin_post,
)


def _make_record(
    title: str = "Test Paper",
    score: float = 0.8,
    included: bool = True,
    summary: Optional[str] = "A great finding.",
    relevance: Optional[str] = "Relevant to business.",
) -> ScoredPaperRecord:
    return ScoredPaperRecord(
        paper=Paper(
            doi="10.1234/test",
            arxiv_id=None,
            openalex_id=None,
            title=title,
            abstract="An abstract.",
            authors=[Author(name="Alice"), Author(name="Bob")],
            journal="Nature",
            journal_issn="0028-0836",
            published_date=date(2026, 3, 27),
            source="arxiv",
            url=None,
        ),
        similarity_score=score,
        final_score=score,
        priority_author_match=False,
        bluesky_sightings=[],
        trending=False,
        summary=summary,
        relevance=relevance,
        digest_section="field",
        included_in_digest=included,
    )


def _make_digest_result(records: Optional[list[ScoredPaperRecord]] = None) -> DigestResult:
    if records is None:
        records = [
            _make_record("High Score Paper", score=0.95),
            _make_record("Medium Score Paper", score=0.70),
            _make_record("Low Score Paper", score=0.50),
            _make_record("Excluded Paper", score=0.90, included=False),
            _make_record("No Summary Paper", score=0.85, summary=None),
        ]
    return DigestResult(
        run_id=1,
        created_at=datetime(2026, 3, 27, 10, 0),
        window_start=date(2026, 3, 20),
        window_end=date(2026, 3, 27),
        papers_fetched=100,
        papers_after_dedup=80,
        papers_included=len([r for r in records if r.included_in_digest]),
        papers_dropped_no_abstract=0,
        markdown_path="digests/digest_2026-03-27.md",
        markdown="# Digest",
        scored_papers=records,
    )


class TestSelectTopPapers:
    def test_filters_excluded_and_no_summary(self):
        result = _make_digest_result()
        top = _select_top_papers(result.scored_papers, max_papers=10)
        titles = [sp.paper.title for sp in top]
        assert "Excluded Paper" not in titles
        assert "No Summary Paper" not in titles

    def test_respects_max_papers(self):
        result = _make_digest_result()
        top = _select_top_papers(result.scored_papers, max_papers=2)
        assert len(top) == 2

    def test_sorted_by_score_descending(self):
        result = _make_digest_result()
        top = _select_top_papers(result.scored_papers, max_papers=10)
        scores = [sp.final_score for sp in top]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        top = _select_top_papers([], max_papers=3)
        assert top == []


class TestBuildPrompt:
    def test_contains_paper_titles(self):
        records = [_make_record("Neural Synchrony in Teams")]
        prompt = _build_prompt(records, date(2026, 3, 20), date(2026, 3, 27))
        assert "Neural Synchrony in Teams" in prompt

    def test_contains_good_and_bad_examples(self):
        records = [_make_record()]
        prompt = _build_prompt(records, date(2026, 3, 20), date(2026, 3, 27))
        assert "GOOD EXAMPLE" in prompt
        assert "BAD EXAMPLE" in prompt

    def test_contains_structure_guidance(self):
        records = [_make_record()]
        prompt = _build_prompt(records, date(2026, 3, 20), date(2026, 3, 27))
        assert "STRUCTURE" in prompt
        assert "Plain text only" in prompt


class TestBuildLinkedinPost:
    @pytest.mark.asyncio
    async def test_generates_and_saves_post(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_response = (
            "Most people think leadership is about authority.\n\n"
            "They're wrong.\n\n#Leadership #Research"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": fake_response}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.linkedin.httpx.AsyncClient", return_value=mock_client):
            result = _make_digest_result()
            post = await build_linkedin_post(
                digest_result=result,
                research_profile="I study social neuroscience.",
                ollama_model="qwen2.5:7b",
                ollama_host="http://localhost:11434",
                linkedin_config={"max_papers": 2},
            )

        assert "leadership" in post.lower()
        saved = (tmp_path / "digests" / "linkedin_2026-03-27.txt").read_text()
        assert saved == post

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_papers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _make_digest_result(records=[])
        post = await build_linkedin_post(
            digest_result=result,
            research_profile="I study things.",
            ollama_model="qwen2.5:7b",
            ollama_host="http://localhost:11434",
            linkedin_config={},
        )
        assert post == ""

    @pytest.mark.asyncio
    async def test_uses_custom_persona(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_response = "Custom persona post content."

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": fake_response}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("services.linkedin.httpx.AsyncClient", return_value=mock_client) as _:
            result = _make_digest_result()
            await build_linkedin_post(
                digest_result=result,
                research_profile="I study things.",
                ollama_model="qwen2.5:7b",
                ollama_host="http://localhost:11434",
                linkedin_config={
                    "max_papers": 1,
                    "persona": "You are a tech CEO who loves buzzwords.",
                },
            )

        call_kwargs = mock_client.post.call_args
        sent_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json") or call_kwargs[0][1]
        assert "tech CEO" in sent_payload["system"]
