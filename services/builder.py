"""
Digest builder — summarises scored papers via Ollama and assembles the
final markdown digest.

Steps (per synthesis_model_spec.md):
  5. Summarise each paper (one Ollama call per paper, up to 5 concurrent)
  6. Write section intros (one Ollama call per non-empty section)
  7. Assemble markdown and save to digests/
"""
from __future__ import annotations

import asyncio
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import httpx

from services.scorer import ScoredPaper, ScoringResult
from shared.types import DigestResult, ScoredPaperRecord

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

_OLLAMA_TIMEOUT = 120.0  # seconds per call
_MAX_CONCURRENT = 5
_RETRY_DELAY = 5.0


async def _ollama_generate(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    system: str,
    prompt: str,
) -> str:
    """Call Ollama /api/generate and return the full response text."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
    }
    resp = await client.post(url, json=payload, timeout=_OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_summary_response(raw: str) -> tuple[str, Optional[str]]:
    """Parse SUMMARY: / RELEVANCE: from Ollama output.

    Returns (summary, relevance). If parsing fails, returns (raw, None).
    """
    # Try to split on RELEVANCE: (with optional markdown bold/italic)
    parts = re.split(r"\*{0,2}RELEVANCE\*{0,2}\s*:", raw, maxsplit=1)
    if len(parts) == 2:
        summary_part = parts[0]
        relevance = parts[1].strip()
        # Strip the SUMMARY: prefix if present (with optional markdown bold/italic)
        summary = re.sub(r"^\*{0,2}SUMMARY\*{0,2}\s*:\s*", "", summary_part, count=1).strip()
        return summary, relevance
    return raw.strip(), None


# ---------------------------------------------------------------------------
# Paper summarisation (Step 5)
# ---------------------------------------------------------------------------

def _build_system_prompt(research_profile: str) -> str:
    return (
        "You are a research assistant writing a weekly digest for a scientist.\n"
        "Write clearly and precisely. Do not pad. Do not use filler phrases like\n"
        '"the authors demonstrate" or "this study shows" — get to the finding directly.\n'
        "The scientist's current focus:\n\n"
        f"{research_profile}"
    )


def _build_paper_prompt(sp: ScoredPaper) -> str:
    paper = sp.paper
    authors_str = ", ".join(a.name for a in paper.authors[:5])
    if len(paper.authors) > 5:
        authors_str += f" et al. ({len(paper.authors)} authors)"
    venue = paper.journal or "arXiv"

    lines = [
        f"Title: {paper.title}",
        f"Authors: {authors_str}",
        f"Published: {paper.published_date} in {venue}",
    ]
    if sp.bluesky_sightings:
        handles = ", ".join(s.handle for s in sp.bluesky_sightings)
        lines.append(f"Mentioned on Bluesky by: {handles}")

    lines.append("")
    lines.append("Abstract:")
    lines.append(paper.abstract or "(no abstract available)")
    lines.append("")
    lines.append("Write:")
    lines.append("SUMMARY: 2-3 sentences. What was done, what was found. Plain language.")
    lines.append(
        "RELEVANCE: 1 sentence. Is this relevant to the researcher's focus, and how?\n"
        "           If it is not relevant, say so plainly — that is also useful."
    )
    return "\n".join(lines)


async def _summarise_paper(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    system: str,
    sp: ScoredPaper,
    semaphore: asyncio.Semaphore,
) -> tuple[str, Optional[str]]:
    """Summarise one paper. Returns (summary, relevance)."""
    prompt = _build_paper_prompt(sp)
    async with semaphore:
        try:
            raw = await _ollama_generate(client, host, model, system, prompt)
        except (httpx.HTTPError, KeyError):
            # Retry once after delay
            await asyncio.sleep(_RETRY_DELAY)
            try:
                raw = await _ollama_generate(client, host, model, system, prompt)
            except (httpx.HTTPError, KeyError):
                return "*(summary unavailable)*", None

    summary, relevance = _parse_summary_response(raw)
    if relevance is None:
        # Retry once on parse failure
        async with semaphore:
            try:
                raw = await _ollama_generate(client, host, model, system, prompt)
                s, r = _parse_summary_response(raw)
                if r is not None:
                    summary, relevance = s, r
            except (httpx.HTTPError, KeyError):
                pass
    return summary, relevance


# ---------------------------------------------------------------------------
# Section intros (Step 6)
# ---------------------------------------------------------------------------

_SECTION_DISPLAY_NAMES = {
    "following": "People you follow are talking about",
    "field": "From your field",
    "notable": "Worth noticing",
}


def _build_section_intro_prompt(
    section_key: str, papers: list[ScoredPaper], research_profile: str,
) -> str:
    display_name = _SECTION_DISPLAY_NAMES[section_key]
    paper_list = "\n".join(
        f"- {sp.paper.title} ({', '.join(a.name for a in sp.paper.authors[:3])})"
        for sp in papers
    )
    return (
        "You are writing a brief intro paragraph for a section of a research digest.\n"
        f"The scientist's focus: {research_profile}\n\n"
        f'This section is called "{display_name}" and contains these papers:\n'
        f"{paper_list}\n\n"
        "Write 2-3 sentences that characterise what is happening in this group of papers\n"
        "this week. Note any theme, tension, or pattern across them if one exists.\n"
        "Do not list the papers — they follow immediately after this paragraph.\n"
        "If no clear pattern exists, say so in one sentence and move on."
    )


async def _generate_section_intros(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    research_profile: str,
    sections: dict[str, list[ScoredPaper]],
) -> dict[str, str]:
    """Generate an intro paragraph for each non-empty section."""
    intros: dict[str, str] = {}
    for section_key, papers in sections.items():
        if not papers:
            continue
        prompt = _build_section_intro_prompt(section_key, papers, research_profile)
        try:
            raw = await _ollama_generate(
                client, host, model,
                system="You are a research digest writer. Be concise.",
                prompt=prompt,
            )
            intros[section_key] = raw.strip()
        except (httpx.HTTPError, KeyError):
            intros[section_key] = ""
    return intros


# ---------------------------------------------------------------------------
# Markdown assembly (Step 7)
# ---------------------------------------------------------------------------

def _format_paper_link(sp: ScoredPaper) -> str:
    paper = sp.paper
    if paper.doi:
        return f"[doi:{paper.doi}](https://doi.org/{paper.doi})"
    if paper.arxiv_id:
        return f"[arXiv:{paper.arxiv_id}](https://arxiv.org/abs/{paper.arxiv_id})"
    if paper.url:
        return f"[link]({paper.url})"
    return ""


def _format_paper_block(sp: ScoredPaper, summary: Optional[str], relevance: Optional[str]) -> str:
    paper = sp.paper
    authors_str = ", ".join(a.name for a in paper.authors[:5])
    if len(paper.authors) > 5:
        authors_str += " et al."
    venue = paper.journal or "arXiv"
    link = _format_paper_link(sp)

    lines = [
        f"### {paper.title}",
        f"**{authors_str}** · {venue} · {paper.published_date}  ",
        f"{link}  ",
    ]

    if sp.trending:
        n_handles = len({s.handle for s in sp.bluesky_sightings})
        lines.append(f"*Trending — mentioned by {n_handles} people you follow*  ")

    if sp.bluesky_sightings:
        shared = ", ".join(
            f"[{s.handle}]({s.post_url})" for s in sp.bluesky_sightings
        )
        lines.append(f"*Shared by: {shared}*  ")

        # Include informative post text (>80 chars after URL removal)
        for s in sp.bluesky_sightings:
            if s.post_text:
                clean = re.sub(r"https?://\S+", "", s.post_text).strip()
                if len(clean) > 80:
                    excerpt = clean[:300]
                    lines.append(f'> "{excerpt}"  ')
                    break  # only include one excerpt

    lines.append("")
    if summary:
        lines.append(summary)
    else:
        lines.append("*(summary unavailable)*")
    lines.append("")
    if relevance:
        lines.append(f"*{relevance}*")
    lines.append("")
    return "\n".join(lines)


def _assemble_markdown(
    sections: dict[str, list[ScoredPaper]],
    summaries: dict[int, tuple[Optional[str], Optional[str]]],
    intros: dict[str, str],
    scoring_result: ScoringResult,
    window_start: date,
    window_end: date,
    created_at: datetime,
) -> str:
    parts = [
        f"# Research digest · {window_start} – {window_end}",
        "",
        f"{scoring_result.papers_included} papers · fetched {created_at.strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]

    section_order = ["following", "field", "notable"]
    for section_key in section_order:
        papers = sections.get(section_key, [])
        if not papers:
            continue

        display_name = _SECTION_DISPLAY_NAMES[section_key]
        parts.append(f"## {display_name}")
        parts.append("")

        intro = intros.get(section_key, "")
        if intro:
            parts.append(intro)
            parts.append("")

        for sp in papers:
            paper_id = id(sp)
            summary, relevance = summaries.get(paper_id, (None, None))
            parts.append(_format_paper_block(sp, summary, relevance))
        parts.append("---")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_digest(
    scoring_result: ScoringResult,
    research_profile: str,
    ollama_model: str,
    ollama_host: str,
    window_start: date,
    window_end: date,
    run_id: int = 0,
) -> DigestResult:
    """Run Steps 5-7: summarise papers, generate section intros, assemble markdown.

    Parameters
    ----------
    scoring_result : ScoringResult from scorer.score_papers()
    research_profile : The user's research profile text
    ollama_model : Ollama model name, e.g. "qwen2.5:7b"
    ollama_host : Ollama base URL, e.g. "http://localhost:11434"
    window_start / window_end : Date range for this digest
    run_id : Digest run identifier
    """
    created_at = datetime.now()
    system = _build_system_prompt(research_profile)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    # Collect all included papers across sections
    all_papers: list[ScoredPaper] = []
    for section_key in ["following", "field", "notable"]:
        all_papers.extend(scoring_result.sections.get(section_key, []))

    # Step 5 — summarise each paper concurrently
    summaries: dict[int, tuple[Optional[str], Optional[str]]] = {}
    async with httpx.AsyncClient() as client:
        tasks = []
        paper_ids = []
        for sp in all_papers:
            paper_ids.append(id(sp))
            tasks.append(
                _summarise_paper(client, ollama_host, ollama_model, system, sp, semaphore)
            )
        results = await asyncio.gather(*tasks)
        for pid, (summary, relevance) in zip(paper_ids, results):
            summaries[pid] = (summary, relevance)

        # Step 6 — section intros
        intros = await _generate_section_intros(
            client, ollama_host, ollama_model, research_profile,
            scoring_result.sections,
        )

    # Step 7 — assemble markdown
    markdown = _assemble_markdown(
        scoring_result.sections, summaries, intros,
        scoring_result, window_start, window_end, created_at,
    )

    # Save to file
    digests_dir = Path("digests")
    digests_dir.mkdir(exist_ok=True)
    filename = f"digest_{window_end.strftime('%Y-%m-%d')}.md"
    filepath = digests_dir / filename
    filepath.write_text(markdown, encoding="utf-8")

    # Build ScoredPaperRecords for DB persistence
    scored_records: list[ScoredPaperRecord] = []
    no_abstract_count = 0
    for sp in all_papers:
        summary, relevance = summaries.get(id(sp), (None, None))
        if sp.paper.abstract is None:
            no_abstract_count += 1
        scored_records.append(ScoredPaperRecord(
            paper=sp.paper,
            similarity_score=sp.similarity_score,
            final_score=sp.final_score,
            priority_author_match=sp.priority_author_match,
            bluesky_sightings=sp.bluesky_sightings,
            trending=sp.trending,
            summary=summary,
            relevance=relevance,
            digest_section=sp.digest_section,
            included_in_digest=True,
        ))
    # Also include dropped papers
    for sp in scoring_result.dropped:
        if sp.paper.abstract is None:
            no_abstract_count += 1
        scored_records.append(ScoredPaperRecord(
            paper=sp.paper,
            similarity_score=sp.similarity_score,
            final_score=sp.final_score,
            priority_author_match=sp.priority_author_match,
            bluesky_sightings=sp.bluesky_sightings,
            trending=sp.trending,
            summary=None,
            relevance=None,
            digest_section=sp.digest_section if sp.digest_section else "dropped",
            included_in_digest=False,
        ))

    return DigestResult(
        run_id=run_id,
        created_at=created_at,
        window_start=window_start,
        window_end=window_end,
        papers_fetched=scoring_result.papers_fetched,
        papers_after_dedup=scoring_result.papers_after_dedup,
        papers_included=scoring_result.papers_included,
        papers_dropped_no_abstract=no_abstract_count,
        markdown_path=str(filepath.resolve()),
        markdown=markdown,
        scored_papers=scored_records,
    )
