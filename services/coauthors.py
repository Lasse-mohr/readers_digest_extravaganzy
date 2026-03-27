import asyncio
import logging
from collections import Counter
from datetime import datetime, timedelta
from functools import partial

from pyalex import Works
from sqlmodel import Session, select

from db.database import CoauthorCache

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 30
DEFAULT_MIN_COLLABORATIONS = 3


# ── Sync fetch (runs in thread pool) ──────────────────────────────────────────

def _fetch_coauthors_sync(
    anchor_openalex_id: str,
    min_collaborations: int,
) -> list[dict]:
    coauthor_counter: Counter = Counter()
    coauthor_names: dict[str, str] = {}

    pager = (
        Works()
        .filter(authorships={"author": {"id": anchor_openalex_id}})
        .select(["id", "authorships"])
        .paginate(per_page=200, n_max=None)
    )

    for page in pager:
        for work in page:
            for authorship in work.get("authorships", []):
                author = authorship.get("author") or {}
                author_id = author.get("id")
                if author_id and author_id != anchor_openalex_id:
                    coauthor_counter[author_id] += 1
                    if author_id not in coauthor_names:
                        coauthor_names[author_id] = author.get("display_name", "")

    return [
        {
            "openalex_id": aid,
            "name": coauthor_names[aid],
            "collaboration_count": count,
        }
        for aid, count in coauthor_counter.items()
        if count >= min_collaborations
    ]


# ── Public async interface ─────────────────────────────────────────────────────

async def refresh_coauthors(
    anchor_openalex_id: str,
    session: Session,
    min_collaborations: int = DEFAULT_MIN_COLLABORATIONS,
) -> int:
    """
    Fetch coauthors from OpenAlex and update the cache for this anchor author.

    Should be called as a background task on config save, not during digest generation.
    Returns the number of coauthors cached.
    """
    logger.info("Fetching coauthors for %s", anchor_openalex_id)
    loop = asyncio.get_event_loop()
    coauthors = await loop.run_in_executor(
        None,
        partial(_fetch_coauthors_sync, anchor_openalex_id, min_collaborations),
    )

    # Replace stale cache entries for this anchor
    existing = session.exec(
        select(CoauthorCache).where(CoauthorCache.anchor_openalex_id == anchor_openalex_id)
    ).all()
    for entry in existing:
        session.delete(entry)

    now = datetime.utcnow()
    for c in coauthors:
        session.add(CoauthorCache(
            anchor_openalex_id=anchor_openalex_id,
            coauthor_openalex_id=c["openalex_id"],
            coauthor_name=c["name"],
            collaboration_count=c["collaboration_count"],
            cached_at=now,
        ))
    session.commit()
    logger.info("Cached %d coauthors for %s", len(coauthors), anchor_openalex_id)
    return len(coauthors)


def is_cache_fresh(anchor_openalex_id: str, session: Session) -> bool:
    """Return True if the coauthor cache for this anchor is within TTL."""
    cutoff = datetime.utcnow() - timedelta(days=CACHE_TTL_DAYS)
    row = session.exec(
        select(CoauthorCache).where(
            CoauthorCache.anchor_openalex_id == anchor_openalex_id,
            CoauthorCache.cached_at >= cutoff,
        )
    ).first()
    return row is not None


def get_coauthor_ids(anchor_openalex_id: str, session: Session) -> set[str]:
    """
    Return the set of cached coauthor OpenAlex IDs for this anchor.

    Returns an empty set if the cache is stale — caller should trigger a refresh.
    """
    cutoff = datetime.utcnow() - timedelta(days=CACHE_TTL_DAYS)
    rows = session.exec(
        select(CoauthorCache).where(
            CoauthorCache.anchor_openalex_id == anchor_openalex_id,
            CoauthorCache.cached_at >= cutoff,
        )
    ).all()
    return {row.coauthor_openalex_id for row in rows}
