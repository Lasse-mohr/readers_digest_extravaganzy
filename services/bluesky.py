"""Bluesky integration — fetch academic paper mentions from Bluesky posts."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime
from urllib.parse import unquote

import httpx

from shared.types import BlueskySighting

logger = logging.getLogger("bluesky")

BLUESKY_API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"

# Regex for extracting URLs from plain text
URL_REGEX = re.compile(r"https?://[^\s<>\"\)]+")

# Patterns for identifying academic paper URLs
DOI_ORG_REGEX = re.compile(r"doi\.org/(10\.\d{4,}/[^\s&?#]+)", re.IGNORECASE)
NATURE_REGEX = re.compile(r"nature\.com/articles/(s\d+-\d+-\d+-\w+)", re.IGNORECASE)
SCIENCE_REGEX = re.compile(r"science\.org/doi/(10\.\d{4,}/[^\s&?#]+)", re.IGNORECASE)
ARXIV_REGEX = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,})", re.IGNORECASE)
BIORXIV_REGEX = re.compile(
    r"biorxiv\.org/content/(10\.\d{4,}/[^\s&?#v]+)", re.IGNORECASE
)
MEDRXIV_REGEX = re.compile(
    r"medrxiv\.org/content/(10\.\d{4,}/[^\s&?#v]+)", re.IGNORECASE
)
PUBMED_REGEX = re.compile(
    r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE
)


def extract_urls_from_post(post: dict) -> list[str]:
    """Extract URLs from a Bluesky post.

    Checks facets first (structured links), then falls back to regex on text.
    """
    urls: list[str] = []

    record = post.get("post", {}).get("record", {})

    # Primary: facets
    facets = record.get("facets", [])
    for facet in facets:
        for feature in facet.get("features", []):
            if feature.get("$type") == "app.bsky.richtext.facet#link":
                uri = feature.get("uri")
                if uri:
                    urls.append(uri)

    # Fallback: regex on text (only if no facet URLs found)
    if not urls:
        text = record.get("text", "")
        urls = URL_REGEX.findall(text)

    return urls


async def resolve_url(
    client: httpx.AsyncClient, url: str, cache: dict[str, str | None]
) -> str | None:
    """Resolve a URL via HEAD request, following redirects. Returns final URL or None."""
    if url in cache:
        return cache[url]

    try:
        resp = await client.head(url, follow_redirects=True, timeout=10.0)
        final_url = str(resp.url)
        cache[url] = final_url
        await asyncio.sleep(1)  # rate limit
        return final_url
    except Exception as e:
        logger.debug("Failed to resolve %s: %s", url, e)
        cache[url] = None
        return None


def extract_paper_id(url: str) -> tuple[str, str] | None:
    """Extract a paper identifier from a resolved URL.

    Returns:
        (id_type, id_value) e.g. ("doi", "10.1038/...") or ("arxiv", "2401.12345")
        or None if the URL doesn't match a known pattern.
    """
    url = unquote(url).rstrip("/")

    # arXiv (check before DOI since arxiv URLs don't have DOIs)
    m = ARXIV_REGEX.search(url)
    if m:
        return ("arxiv", m.group(1))

    # doi.org
    m = DOI_ORG_REGEX.search(url)
    if m:
        return ("doi", m.group(1).lower())

    # nature.com — DOI is 10.1038/{article_id}
    m = NATURE_REGEX.search(url)
    if m:
        return ("doi", f"10.1038/{m.group(1)}".lower())

    # science.org
    m = SCIENCE_REGEX.search(url)
    if m:
        return ("doi", m.group(1).lower())

    # biorxiv
    m = BIORXIV_REGEX.search(url)
    if m:
        return ("doi", m.group(1).lower())

    # medrxiv
    m = MEDRXIV_REGEX.search(url)
    if m:
        return ("doi", m.group(1).lower())

    # pubmed
    m = PUBMED_REGEX.search(url)
    if m:
        return ("pmid", m.group(1))

    return None


def _parse_post_date(post: dict) -> date | None:
    """Extract the post date from a Bluesky post record."""
    record = post.get("post", {}).get("record", {})
    created_at = record.get("createdAt")
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


def _get_post_url(post: dict) -> str:
    """Construct the public URL for a Bluesky post."""
    post_data = post.get("post", {})
    uri = post_data.get("uri", "")
    author_handle = post_data.get("author", {}).get("handle", "")
    # URI format: at://did:plc:xxx/app.bsky.feed.post/rkey
    parts = uri.rsplit("/", 1)
    rkey = parts[-1] if len(parts) == 2 else ""
    if author_handle and rkey:
        return f"https://bsky.app/profile/{author_handle}/post/{rkey}"
    return uri


def _get_post_text(post: dict) -> str | None:
    """Extract the raw text from a Bluesky post."""
    text = post.get("post", {}).get("record", {}).get("text")
    return text if text else None


async def _fetch_author_feed(
    client: httpx.AsyncClient, handle: str
) -> list[dict]:
    """Fetch recent posts for a Bluesky handle."""
    try:
        resp = await client.get(
            BLUESKY_API,
            params={"actor": handle, "limit": 100},
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json().get("feed", [])
    except Exception as e:
        logger.warning("Failed to fetch feed for %s: %s", handle, e)
        return []


def _make_sighting(
    id_type: str,
    id_value: str,
    handle: str,
    post_url: str,
    posted_at: date,
    post_text: str | None,
) -> BlueskySighting | None:
    """Create a BlueskySighting from extracted paper ID fields."""
    if id_type == "doi":
        return BlueskySighting(
            doi=id_value, arxiv_id=None, handle=handle,
            post_url=post_url, posted_at=posted_at, post_text=post_text,
        )
    elif id_type == "arxiv":
        return BlueskySighting(
            doi=None, arxiv_id=id_value, handle=handle,
            post_url=post_url, posted_at=posted_at, post_text=post_text,
        )
    # Discard pmid and other types we can't match on
    return None


async def fetch_bluesky(
    handles: list[str],
    date_from: date,
    date_to: date,
) -> list[BlueskySighting]:
    """
    For each Bluesky handle, fetch recent public posts, extract URLs,
    resolve them, and identify academic paper links.

    Args:
        handles: List of Bluesky handles (e.g. ["researcher1.bsky.social"])
        date_from: Start of the date window (inclusive).
        date_to: End of the date window (inclusive).

    Returns:
        List of BlueskySighting objects — one per resolved paper link per post.
    """
    sightings: list[BlueskySighting] = []
    resolve_cache: dict[str, str | None] = {}

    async with httpx.AsyncClient() as client:
        for handle in handles:
            logger.info("Fetching feed for %s", handle)
            posts = await _fetch_author_feed(client, handle)
            logger.info("Got %d posts for %s", len(posts), handle)

            for post in posts:
                posted_at = _parse_post_date(post)
                if posted_at is None:
                    continue

                # Filter by date window
                if posted_at < date_from or posted_at > date_to:
                    continue

                post_url = _get_post_url(post)
                post_text = _get_post_text(post)
                urls = extract_urls_from_post(post)

                for url in urls:
                    # Try direct pattern match first
                    paper = extract_paper_id(url)
                    if paper:
                        s = _make_sighting(
                            paper[0], paper[1], handle,
                            post_url, posted_at, post_text,
                        )
                        if s:
                            sightings.append(s)
                        continue

                    # Resolve and try again
                    resolved = await resolve_url(client, url, resolve_cache)
                    if resolved:
                        paper = extract_paper_id(resolved)
                        if paper:
                            s = _make_sighting(
                                paper[0], paper[1], handle,
                                post_url, posted_at, post_text,
                            )
                            if s:
                                sightings.append(s)

    logger.info(
        "Found %d sightings across %d handles", len(sightings), len(handles)
    )
    return sightings


if __name__ == "__main__":
    import sys
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO)

    test_handles = ["atproto.bsky.social"]
    if len(sys.argv) > 1:
        test_handles = sys.argv[1:]

    today = date.today()
    results = asyncio.run(
        fetch_bluesky(test_handles, today - timedelta(days=7), today)
    )
    print(f"\nFound {len(results)} sightings:")
    for s in results:
        paper_id = s.doi or s.arxiv_id
        print(f"  {paper_id} — {s.handle} on {s.posted_at}")
