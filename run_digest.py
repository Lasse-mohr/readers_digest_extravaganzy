"""
Run the full digest pipeline: fetch → score → summarise → markdown.

Usage:
    python -m run_digest
"""

import asyncio
import sys
from datetime import date, timedelta

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def main():
    cfg = load_config()
    lookback = cfg["digest"]["lookback_days"]
    date_to = date.today()
    date_from = date_to - timedelta(days=lookback)

    print(f"Date window: {date_from} → {date_to}")
    print()

    # ── 1. Fetch arXiv ─────────────────────────────────────────────
    from services.fetcher import fetch_arxiv

    print("Fetching from arXiv...")
    arxiv_papers = await fetch_arxiv(
        categories=cfg["arxiv_categories"],
        date_from=date_from,
        date_to=date_to,
    )
    print(f"  {len(arxiv_papers)} papers from arXiv")

    # ── 2. Fetch Bluesky ───────────────────────────────────────────
    from services.bluesky import fetch_bluesky

    print("Fetching from Bluesky...")
    bluesky_sightings = await fetch_bluesky(
        handles=cfg["bluesky_handles"],
        date_from=date_from,
        date_to=date_to,
    )
    print(f"  {len(bluesky_sightings)} Bluesky sightings")

    # ── 3. Merge, dedup, score ─────────────────────────────────────
    from services.scorer import (
        JournalConfig,
        ScorerConfig,
        ScoringWeights,
        encode_profile,
        merge_and_dedup,
        score_papers,
    )

    merged = merge_and_dedup(papers_openalex=[], papers_arxiv=arxiv_papers)
    print(f"  {len(merged)} papers after dedup")

    # ── 2b. Promote trending Bluesky sightings to papers ─────────
    from services.bluesky import promote_trending_sightings

    promoted = await promote_trending_sightings(bluesky_sightings, merged)
    if promoted:
        merged.extend(promoted)
        print(f"  {len(promoted)} trending Bluesky papers promoted (not in arXiv corpus)")

    print("Loading SPECTER2 & scoring...")
    profile_emb = encode_profile(cfg["research_profile"])

    scorer_config = ScorerConfig(
        journals=[
            JournalConfig(name=j["name"], issn=j["issn"], tier=j["tier"])
            for j in cfg["journals"]
        ],
        priority_author_ids={
            a["openalex_id"]
            for a in cfg["priority_authors"]
            if a.get("openalex_id")
        },
        scoring=ScoringWeights(
            semantic_similarity=cfg["scoring"]["semantic_similarity"],
            journal_tier1_bonus=cfg["scoring"]["journal_tier_bonus"]["tier1"],
            journal_tier2_bonus=cfg["scoring"]["journal_tier_bonus"]["tier2"],
            priority_author_bonus=cfg["scoring"]["priority_author_bonus"],
            bluesky_mention_bonus=cfg["scoring"]["bluesky_mention_bonus"],
        ),
        max_papers_in_digest=cfg["digest"]["max_papers_in_digest"],
        min_similarity_threshold=cfg["digest"]["min_similarity_threshold"],
    )

    scoring_result = score_papers(
        papers=merged,
        bluesky_sightings=bluesky_sightings,
        profile_embedding=profile_emb,
        config=scorer_config,
    )

    for section, papers in scoring_result.sections.items():
        print(f"  Section '{section}': {len(papers)} papers")
    print(f"  Dropped: {len(scoring_result.dropped)}")

    if scoring_result.papers_included == 0:
        print("\nNo papers to include — digest would be empty.")
        return

    # ── 4. Build digest (summarise + markdown) ─────────────────────
    from services.builder import build_digest

    print(f"\nSummarising {scoring_result.papers_included} papers with Ollama ({cfg['digest']['ollama_model']})...")
    result = await build_digest(
        scoring_result=scoring_result,
        research_profile=cfg["research_profile"],
        ollama_model=cfg["digest"]["ollama_model"],
        ollama_host=cfg["digest"]["ollama_host"],
        window_start=date_from,
        window_end=date_to,
    )

    print(f"\nDigest saved to: {result.markdown_path}")
    print(f"  {result.papers_included} papers included")
    print(f"  {result.papers_dropped_no_abstract} had no abstract")


if __name__ == "__main__":
    asyncio.run(main())
