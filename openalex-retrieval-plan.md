# OpenAlex Retrieval Layer — Implementation Plan

**For:** readers_digest_extravaganzy  
**Scope:** `services/fetcher.py` (OpenAlex parts) + `services/coauthors.py`  
**Date:** 2026-03-26

---

## 0. Critical corrections to the master plan

The master plan was written before several breaking changes to the OpenAlex API that
took effect in February 2026. These must be addressed before any code is written.

### 0.1 Authentication: API keys required, polite pool is dead

The master plan says to use `mailto=` for the polite pool. **This no longer works.**
As of February 13, 2026, all OpenAlex API requests require a free API key. The
`mailto=` parameter is deprecated and ignored.

**What to change:**

- In `.env`, replace `OPENALEX_EMAIL` with `OPENALEX_API_KEY`.
- Every request must include `?api_key=YOUR_KEY` as a query parameter.
- Get a free key at <https://openalex.org/settings/api>.

### 0.2 Rate limits are now credit-based, not call-based

The master plan says "stay under 10 req/s; add `asyncio.sleep(0.1)`". The actual
limits are now:

| Tier | Daily budget | List query cost | Singleton cost |
|------|-------------|-----------------|----------------|
| No key | 100 credits (testing only) | 1 credit | 0 credits |
| Free key | 100,000 credits/day | 1 credit | 0 credits |
| Max concurrency | 100 req/s hard cap (all tiers) | — | — |

At 1 credit per list request, a weekly digest fetching papers from ~6 journals for
7 days will use well under 100 credits. **Budget is not a concern for this use case**,
but the code should still track credits via response headers and log a warning if
daily usage exceeds 80%.

The `asyncio.sleep(0.1)` between pages is still a reasonable courtesy, but the
hard limit is 100 req/s, not 10.

### 0.3 New developer docs

OpenAlex has rebuilt their documentation. The old docs at `docs.openalex.org`
still work but with a restructured layout, plus a new `developers.openalex.org`
site. When in doubt, check both.

### 0.4 XPAC works are excluded by default

Since November 2025, OpenAlex added 190M+ lower-quality works (from DataCite and
repositories). These are excluded from API results by default, which is the
behavior we want. No action needed, but be aware that adding `include_xpac=true`
would flood results with noise.

### 0.5 The `select` parameter is root-level only

The master plan asks for fields like `authorships` and `primary_location`, which
are root-level — that's fine. But you cannot select nested fields like
`primary_location.source.issn` individually. You get the entire `primary_location`
object or nothing.

---

## 1. Library decision: pyalex vs. raw httpx

### Option A: Use `pyalex` (recommended for MVP)

[pyalex](https://github.com/J535D165/pyalex) is a thin, well-maintained Python
wrapper around the OpenAlex API. It handles:

- **Abstract reconstruction** from inverted index → plaintext (on the fly)
- **Cursor pagination** with `.paginate()` — no manual cursor management
- **Filtering** with a Pythonic chained API
- **API key auth** via `pyalex.config.api_key`
- **Retries and backoff** (configurable `max_retries`, `retry_backoff_factor`)

**Downside:** pyalex is synchronous (uses `requests`). The project uses FastAPI
(async). This means OpenAlex fetches should be run in a thread pool:

```python
from functools import partial
import asyncio
import pyalex

# In your async fetch function:
loop = asyncio.get_event_loop()
results = await loop.run_in_executor(None, partial(sync_fetch_function))
```

This is an acceptable trade-off for an MVP of a localhost tool that fetches papers
once a week. pyalex saves you from writing cursor pagination, abstract
reconstruction, retry logic, and ISSN-filter construction yourself.

**Caveat to verify before committing:** There is an open issue
(J535D165/pyalex#91) about whether pyalex passes the API key as a URL parameter
vs. header. As of the latest pyalex release, `config.api_key` works, but test it
against the live API before building on it. If it breaks, the fix is a one-line
monkey-patch or a small PR.

### Option B: Raw httpx (if you need full async control)

Use `httpx.AsyncClient` and build the query construction, pagination, and abstract
reconstruction yourself. More code, more control, fully async. Reasonable if the
project grows beyond a weekly localhost tool.

**Recommendation:** Start with pyalex. Switch to raw httpx later only if the
synchronous calls become a bottleneck (unlikely for weekly batch runs).

---

## 2. Module structure

The master plan puts all fetch logic in `services/fetcher.py`. I recommend splitting
the OpenAlex-specific code into its own internal module for testability:

```
services/
├── fetcher.py              # Orchestrator: calls openalex + arxiv, deduplicates
├── openalex_client.py      # All OpenAlex API interaction (this plan)
└── coauthors.py            # Coauthor expansion (also OpenAlex, but separate concern)
```

`openalex_client.py` owns all interaction with the OpenAlex API. `fetcher.py`
calls it and also calls the arXiv fetcher, then deduplicates.

---

## 3. Implementation plan for `openalex_client.py`

### 3.1 Configuration and initialization

```python
import pyalex
from pyalex import Works

def init_openalex(api_key: str):
    """Call once at app startup."""
    pyalex.config.api_key = api_key
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.retry_http_codes = [429, 500, 503]
```

The API key comes from `.env` → `config.yaml` loader → passed here.

### 3.2 Fetching works by journal ISSNs

The master plan correctly identifies that the filter is
`primary_location.source.issn`. Multiple ISSNs can be OR'd with the pipe `|`
operator in a single request, which is much more efficient than one request per
journal.

```python
from pyalex import Works
from datetime import date

def fetch_works_by_journals(
    issns: list[str],
    from_date: date,
    to_date: date,
) -> list[dict]:
    """
    Fetch works published in the given journals during the date window.

    Uses OR-filter on ISSNs to minimize API calls.
    OpenAlex allows up to 50 values in an OR filter.
    """
    # Build OR filter: "0028-0836|0036-8075|0092-8674"
    issn_filter = "|".join(issns)

    all_works = []
    pager = (
        Works()
        .filter(
            primary_location={"source": {"issn": issn_filter}},
            from_publication_date=from_date.isoformat(),
            to_publication_date=to_date.isoformat(),
        )
        .select(["id", "doi", "title", "abstract_inverted_index",
                 "authorships", "primary_location", "publication_date"])
        .paginate(per_page=200, n_max=None)
    )

    for page in pager:
        all_works.extend(page)

    return all_works
```

**Key points:**

- The `select` parameter limits returned fields, reducing bandwidth. Only
  root-level fields are selectable.
- `paginate(n_max=None)` uses cursor paging under the hood and fetches all
  matching results. For a 7-day window across ~6 journals, this is typically a
  few hundred papers at most — well within budget.
- pyalex reconstructs `abstract_inverted_index` into plaintext automatically
  when you access `work["abstract"]` on the returned dict.

### 3.3 Abstract reconstruction (if not using pyalex)

If you go the raw httpx route, you need to reconstruct abstracts yourself.
The master plan correctly notes this. Here is the utility function:

```python
def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """
    Reconstruct plaintext abstract from OpenAlex's inverted index format.

    The inverted index maps each word to its position(s) in the abstract.
    Example: {"Despite": [0], "growing": [1], "interest": [2], "in": [3, 57]}
    """
    if not inverted_index:
        return None

    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)
```

### 3.4 Normalizing work records into the Paper model

The fetched OpenAlex data needs to be normalized into the `Paper` SQLModel:

```python
from db.database import Paper
from datetime import datetime

def openalex_work_to_paper(work: dict) -> Paper:
    """Map an OpenAlex work dict to our Paper model."""
    # Extract DOI, strip prefix
    doi = work.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    # Extract authors
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        authors.append({
            "name": author.get("display_name"),
            "openalex_id": author.get("id"),
            "orcid": author.get("orcid"),
        })

    # Extract journal name from primary_location
    journal = None
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    if source:
        journal = source.get("display_name")

    return Paper(
        id=doi or work.get("id"),  # DOI preferred, fallback to OpenAlex ID
        title=work.get("title") or work.get("display_name"),
        abstract=work.get("abstract"),  # pyalex auto-reconstructs this
        authors=authors,
        journal=journal,
        published_date=work.get("publication_date"),
        doi=doi,
        arxiv_id=None,
        openalex_id=work.get("id"),
        source="openalex",
        fetched_at=datetime.utcnow(),
    )
```

### 3.5 Handling the authorships truncation at 100

OpenAlex truncates the authorships list at 100 for list queries. A boolean
`is_authors_truncated` is set to `true` when this happens. For this project,
100 authors is more than enough — the scorer only needs to check if a priority
author appears in the list. But be aware that for mega-authored papers (e.g.,
CERN physics), a priority author at position 101+ will be missed.

If this becomes an issue, do a singleton lookup for flagged papers:

```python
# Singleton lookups are free (0 credits)
full_work = Works()[work["id"]]
```

---

## 4. Implementation plan for `services/coauthors.py`

The master plan's approach is sound. Query OpenAlex for all works by the anchor
author, collect co-occurring author IDs, filter to those with ≥3 collaborations.

### 4.1 Fetching works by author

```python
from pyalex import Works
from collections import Counter

def fetch_coauthors(
    anchor_openalex_id: str,
    min_collaborations: int = 3,
) -> list[dict]:
    """
    Find frequent coauthors of the given author.

    Returns list of {openalex_id, name, collaboration_count}.
    """
    coauthor_counter = Counter()
    coauthor_names = {}

    pager = (
        Works()
        .filter(authorships={"author": {"id": anchor_openalex_id}})
        .select(["id", "authorships"])
        .paginate(per_page=200, n_max=None)
    )

    for page in pager:
        for work in page:
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                author_id = author.get("id")
                if author_id and author_id != anchor_openalex_id:
                    coauthor_counter[author_id] += 1
                    coauthor_names[author_id] = author.get("display_name")

    return [
        {
            "openalex_id": aid,
            "name": coauthor_names[aid],
            "collaboration_count": count,
        }
        for aid, count in coauthor_counter.items()
        if count >= min_collaborations
    ]
```

### 4.2 Performance warning for prolific authors

The master plan correctly flags this. An author with 500 publications means
~3 paginated requests (at 200/page). An author with 5,000 publications means
~25 requests. At 1 credit each, this is trivially within budget, but it takes
time. **Run coauthor expansion as a background task on config save**, not inline
during digest generation.

### 4.3 Caching in CoauthorCache

Cache results for 30 days as specified in the master plan. On each digest run,
check `cached_at` and skip re-fetching if fresh enough.

---

## 5. Build order

This is the recommended implementation sequence for the OpenAlex retrieval code
specifically (not the full project):

| Step | Task | Validates |
|------|------|-----------|
| 1 | Get a free API key from openalex.org | Account works |
| 2 | Install pyalex, set `config.api_key`, make one test query | Auth works, pyalex issue #91 not blocking |
| 3 | Implement `init_openalex()` | Startup wiring |
| 4 | Implement `fetch_works_by_journals()` with one real ISSN and a 7-day window | Filtering, pagination, abstract reconstruction |
| 5 | Implement `openalex_work_to_paper()` mapping | Data flows into Paper model correctly |
| 6 | Implement `fetch_coauthors()` for one test author | Coauthor expansion works |
| 7 | Wire coauthor results into CoauthorCache with 30-day TTL | Cache lifecycle works |
| 8 | Add credit-usage logging (read response headers) | Operational visibility |
| 9 | Wire into `fetcher.py` orchestrator alongside arXiv fetch + dedup | Full fetch pipeline |

### Step 4 is the critical gate

If step 4 works — you can filter by ISSN, paginate, and get reconstructed
abstracts — the rest is largely plumbing. Test it manually against a real journal
(e.g., ISSN `0028-0836` for Nature, last 7 days) before proceeding.

---

## 6. Things I verified against the current API docs

| Claim in master plan | Status | Notes |
|---|---|---|
| Endpoint is `https://api.openalex.org/works` | ✅ Correct | |
| Filter by `primary_location.source.issn` | ✅ Correct | Use pipe `\|` for OR |
| Filter by `publication_date` | ✅ Correct | Use `from_publication_date` / `to_publication_date` |
| Request specific fields with `select` | ✅ Correct | Root-level fields only |
| Abstracts come as inverted index | ✅ Correct | pyalex auto-reconstructs |
| Paginate with `cursor=*` | ✅ Correct | pyalex handles via `.paginate()` |
| Use `mailto=` param for polite pool | ❌ **Deprecated** | Use `api_key=` instead |
| Rate limit: stay under 10 req/s | ❌ **Outdated** | Hard limit is 100 req/s; credit-based now |
| Filter author works by `authorships.author.id` | ✅ Correct | |
| Authorships truncated at 100 in list queries | ✅ Correct | Singleton lookups return full list |
| `host_venue` property | ❌ **Removed** | Use `primary_location` (master plan already does) |

---

## 7. Risks and open questions

**pyalex API key delivery method.** There is an open GitHub issue (#91) about
whether pyalex passes the key correctly (header vs. URL param). Test this in
step 2 before building on it. Fallback: pass `api_key` as a query param
manually using httpx.

**Credit cost may change.** OpenAlex's blog post from February 2026 says pricing
is still being calibrated. The free daily $1 allowance covers this project's
needs by a large margin, but monitor the `x-credits-remaining` response header
in production.

**`.search` filters are deprecated.** The old syntax
`filter=title.search:keyword` is being deprecated in favor of the `?search=`
parameter. This project doesn't use search filters (it filters by ISSN and date),
so no impact — but worth knowing if the scope expands.

**Semantic search exists now.** OpenAlex launched semantic search (embedding-based)
in February 2026. This could potentially replace or supplement the SPECTER2
scoring in `services/scorer.py` — but that is out of scope for the retrieval
layer.

---

## 8. Dependencies to add

```
# In requirements.txt, add:
pyalex           # OpenAlex Python client (handles pagination, abstracts, retries)

# httpx is still needed for arXiv + Bluesky, but not for OpenAlex if using pyalex
```

pyalex depends on `requests` (synchronous HTTP). Since the project already uses
`httpx` for arXiv and Bluesky, both will be in the dependency tree. That is fine.
