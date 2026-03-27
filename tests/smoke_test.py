"""
End-to-end smoke test: fetch from arXiv + Bluesky, score results.

Hits real APIs (arXiv, Bluesky) — needs network access.
SPECTER2 model is downloaded on first run (~500 MB).

Usage:
    python -m tests.smoke_test
"""

import asyncio
import sys
from datetime import date, timedelta

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_smoke_test():
    # ── 1. Load config ────────────────────────────────────────────
    cfg = load_config()
    lookback = cfg["digest"]["lookback_days"]
    date_to = date.today()
    date_from = date_to - timedelta(days=lookback)

    print(f"Date window: {date_from} → {date_to}")
    print(f"arXiv categories: {cfg['arxiv_categories']}")
    print(f"Bluesky handles: {cfg['bluesky_handles']}")
    print()

    # ── 2. Fetch from arXiv ───────────────────────────────────────
    from services.fetcher import fetch_arxiv

    print("Fetching from arXiv...")
    arxiv_papers = await fetch_arxiv(
        categories=cfg["arxiv_categories"],
        date_from=date_from,
        date_to=date_to,
    )
    if arxiv_papers:
        print(f"  ✓ {len(arxiv_papers)} papers from arXiv")
    else:
        print("  ⚠ 0 papers from arXiv (API may be unavailable or date range empty)")

    # ── 3. Fetch from Bluesky ────────────────────────────────────
    from services.bluesky import fetch_bluesky

    print("Fetching from Bluesky...")
    bluesky_sightings = await fetch_bluesky(
        handles=cfg["bluesky_handles"],
        date_from=date_from,
        date_to=date_to,
    )
    print(f"  ✓ {len(bluesky_sightings)} Bluesky sightings")

    # ── 4. Merge & dedup ─────────────────────────────────────────
    from services.scorer import merge_and_dedup

    # No OpenAlex fetcher yet — pass empty list
    merged = merge_and_dedup(papers_openalex=[], papers_arxiv=arxiv_papers)
    print(f"  ✓ {len(merged)} papers after dedup")

    # ── 5. Encode profile & score ────────────────────────────────
    from services.scorer import (
        JournalConfig,
        ScorerConfig,
        ScoringWeights,
        encode_profile,
        score_papers,
    )

    print("Loading SPECTER2 model & encoding profile...")
    profile_emb = encode_profile(cfg["research_profile"])
    print(f"  ✓ Profile embedding shape: {profile_emb.shape}")

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

    print("Scoring papers...")
    result = score_papers(
        papers=merged,
        bluesky_sightings=bluesky_sightings,
        profile_embedding=profile_emb,
        config=scorer_config,
    )

    # ── 6. Report ────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("SMOKE TEST RESULTS")
    print("═" * 60)
    print(f"Papers fetched:    {result.papers_fetched}")
    print(f"After dedup:       {result.papers_after_dedup}")
    print(f"Included in digest:{result.papers_included}")
    print(f"Dropped:           {len(result.dropped)}")
    print()

    for section, papers in result.sections.items():
        print(f"Section '{section}': {len(papers)} papers")
        for sp in papers[:3]:  # show top 3 per section
            print(f"  [{sp.final_score:.3f}] {sp.paper.title[:80]}")
            if sp.bluesky_sightings:
                handles = {s.handle for s in sp.bluesky_sightings}
                print(f"         mentioned by: {', '.join(handles)}")
        if len(papers) > 3:
            print(f"  ... and {len(papers) - 3} more")
    print()

    # ── 7. Basic assertions ──────────────────────────────────────
    errors = []

    if result.papers_fetched == 0:
        print("  ⚠ No papers fetched from arXiv (API may be unavailable or date range empty)")

    for sp in result.sections.get("following", []):
        if not sp.bluesky_sightings:
            errors.append(f"Paper in 'following' has no bluesky sightings: {sp.paper.title[:50]}")

    for section, papers in result.sections.items():
        if section not in ("following", "field", "notable"):
            errors.append(f"Unexpected section: {section}")
        for sp in papers:
            if sp.digest_section != section:
                errors.append(f"Section mismatch: {sp.digest_section} != {section}")

    for sp in result.dropped:
        if sp.included_in_digest:
            errors.append("Dropped paper marked as included")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        return False

    print("✓ All checks passed!")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run_smoke_test())
    sys.exit(0 if ok else 1)
