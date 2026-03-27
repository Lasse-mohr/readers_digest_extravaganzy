"""
Quick smoke test for OpenAlex integration.

Usage:
    python3 test_openalex.py "Author Name"
    python3 test_openalex.py  # defaults to "Karl Deisseroth"
"""
import sys
from dotenv import load_dotenv
import os

load_dotenv()

import pyalex
from pyalex import Authors, Works

api_key = os.environ.get("OPENALEX_API_KEY")
if not api_key:
    print("ERROR: OPENALEX_API_KEY not set in .env")
    sys.exit(1)

pyalex.config.api_key = api_key
pyalex.config.max_retries = 3
pyalex.config.retry_backoff_factor = 0.5
pyalex.config.retry_http_codes = [429, 500, 503]

author_name = sys.argv[1] if len(sys.argv) > 1 else "Karl Deisseroth"

print(f"Searching for author: {author_name!r}")
results = Authors().search(author_name).get()

if not results:
    print("No authors found.")
    sys.exit(0)

# Show top matches
print(f"\nTop {min(3, len(results))} matches:")
for a in results[:3]:
    print(f"  {a['display_name']}  |  ID: {a['id']}  |  works: {a['works_count']}  |  citations: {a['cited_by_count']}")

# Use the top match to fetch recent works
top = results[0]
print(f"\nFetching recent works for: {top['display_name']} ({top['id']})")

works = (
    Works()
    .filter(authorships={"author": {"id": top["id"]}})
    .select(["id", "doi", "title", "publication_date", "primary_location"])
    .sort(publication_date="desc")
    .get(per_page=5)
)

print(f"\nMost recent {len(works)} works:")
for w in works:
    journal = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "—")
    print(f"  [{w.get('publication_date')}] {w.get('title', '(no title)')[:80]}")
    print(f"    Journal: {journal}  |  DOI: {w.get('doi', '—')}")
