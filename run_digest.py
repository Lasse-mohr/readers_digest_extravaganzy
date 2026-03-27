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

    # ── 1. Fetch arXiv + OpenAlex ──────────────────────────────────
    from services.fetcher import run_fetch
    from services.openalex_client import init_openalex
    from shared.types import Author
    from shared.types import Paper as SharedPaper
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        print("ERROR: OPENALEX_API_KEY not set in .env — aborting.")
        sys.exit(1)
    init_openalex(api_key)

    issns = [j["issn"] for j in cfg["journals"]]
    print(f"Fetching from arXiv ({len(cfg['arxiv_categories'])} categories) "
          f"and OpenAlex ({len(issns)} journals)...")
    db_papers = await run_fetch(
        issns=issns,
        arxiv_categories=cfg["arxiv_categories"],
        from_date=date_from,
        to_date=date_to,
    )
    print(f"  {len(db_papers)} papers after fetch and dedup")

    # Convert db.database.Paper → shared.types.Paper for the scorer
    def _to_shared(p) -> SharedPaper:
        authors = [
            Author(name=a.get("name", ""), openalex_id=a.get("openalex_id"))
            for a in (p.authors or [])
        ]
        return SharedPaper(
            doi=p.doi,
            arxiv_id=p.arxiv_id,
            openalex_id=p.openalex_id,
            title=p.title,
            abstract=p.abstract,
            authors=authors,
            journal=p.journal,
            journal_issn=None,
            published_date=p.published_date,
            source=p.source,
            url=None,
        )

    all_papers = [_to_shared(p) for p in db_papers]

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

    merged = merge_and_dedup(papers_openalex=[], papers_arxiv=all_papers)
    print(f"  {len(merged)} papers after scorer dedup")

    # ── 2b. Promote trending Bluesky sightings to papers ─────────
    from services.bluesky import promote_trending_sightings

    promoted = await promote_trending_sightings(bluesky_sightings, merged)
    if promoted:
        merged.extend(promoted)
        print(f"  {len(promoted)} trending Bluesky papers promoted (not in arXiv corpus)")

    # ── 3a. Coauthor expansion ─────────────────────────────────────
    from services.coauthors import get_coauthor_ids, is_cache_fresh, refresh_coauthors
    from db.database import create_db_and_tables, get_session

    create_db_and_tables()

    expand_authors = [
        a for a in cfg["priority_authors"]
        if a.get("openalex_id") and a.get("expand_coauthors")
    ]

    coauthor_ids: set[str] = set()
    if expand_authors:
        print(f"Expanding coauthors for {len(expand_authors)} priority author(s)...")
        session = get_session()
        for author in expand_authors:
            oid = author["openalex_id"]
            if not is_cache_fresh(oid, session):
                print(f"  Refreshing coauthor cache for {author['name']}...")
                await refresh_coauthors(oid, session)
            ids = get_coauthor_ids(oid, session)
            print(f"  {author['name']}: {len(ids)} coauthors")
            coauthor_ids.update(ids)
        session.close()
        print(f"  {len(coauthor_ids)} unique coauthor IDs added to priority set")

    priority_author_ids = {
        a["openalex_id"]
        for a in cfg["priority_authors"]
        if a.get("openalex_id")
    } | coauthor_ids

    print("Loading SPECTER2 & scoring...")
    profile_emb = encode_profile(cfg["research_profile"])

    scorer_config = ScorerConfig(
        journals=[
            JournalConfig(name=j["name"], issn=j["issn"], tier=j["tier"])
            for j in cfg["journals"]
        ],
        priority_author_ids=priority_author_ids,
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
