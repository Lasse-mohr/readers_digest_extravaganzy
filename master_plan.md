# Research digest app — project plan

## Overview

A locally-hosted FastAPI web app that fetches papers from OpenAlex and arXiv on a weekly schedule, scores them by semantic similarity to a user-defined research profile, annotates them with signal from a curated Bluesky follow list, and generates a structured markdown digest using a local LLM served by Ollama.

No external paid APIs. No cloud dependencies. Runs on localhost.

---

## Repository layout

```
research-digest/
├── main.py                  # FastAPI app, routes, scheduler startup
├── config.yaml              # All user configuration (journals, people, Bluesky, profile)
├── requirements.txt
├── .env                     # OPENALEX_EMAIL, OLLAMA_HOST (optional overrides)
│
├── db/
│   └── database.py          # SQLite setup via SQLModel; all table definitions
│
├── services/
│   ├── fetcher.py           # OpenAlex + arXiv fetch logic
│   ├── bluesky.py           # Bluesky follow-list scraper
│   ├── coauthors.py         # OpenAlex coauthor expansion
│   ├── scorer.py            # SPECTER2 embedding + cosine ranking
│   └── builder.py           # Digest assembly + Ollama summarisation
│
├── scheduler.py             # APScheduler: weekly job wiring
│
└── templates/
    ├── base.html
    ├── index.html           # Digest list view
    ├── digest.html          # Single digest viewer (rendered markdown)
    └── config.html          # Config editor UI
```

---

## Configuration — `config.yaml`

This is the single file the user edits. The web UI reads and writes it directly.

```yaml
# ── Research profile ──────────────────────────────────────────────
# Free-text description of your current work and interests.
# This is encoded by SPECTER2 to produce your "profile embedding".
# Update it whenever your focus shifts.
research_profile: |
  I am working on ...

# ── Journals ──────────────────────────────────────────────────────
# tier 1 papers get a score bonus; tier 2 are included but ranked lower.
# ISSN is the canonical identifier used by OpenAlex filtering.
journals:
  - name: Nature
    issn: "0028-0836"
    tier: 1
  - name: Science
    issn: "0036-8075"
    tier: 1
  - name: Cell
    issn: "0092-8674"
    tier: 1
  - name: Nature Methods
    issn: "1548-7091"
    tier: 2
  - name: Nature Biotechnology
    issn: "1087-0156"
    tier: 2
  - name: eLife
    issn: "2050-084X"
    tier: 2
  # Add/remove as needed.

# ── arXiv categories ──────────────────────────────────────────────
# Papers matching these categories are fetched from arXiv.
# See arxiv.org/category_taxonomy for the full list.
arxiv_categories:
  - q-bio.GN
  - q-bio.CB
  - cs.LG
  # Add/remove as needed.

# ── Priority authors ──────────────────────────────────────────────
# Papers authored or co-authored by these people receive a score boost.
# openalex_id is the most reliable anchor; orcid is a good alternative.
# If you only have a name, set the id fields to null and accept some
# ambiguity in matching — OpenAlex name search is imperfect.
priority_authors:
  - name: Jane Smith
    openalex_id: "A12345678"   # from openalex.org/authors/A12345678
    orcid: "0000-0001-2345-6789"
    expand_coauthors: true     # if true, fetch their frequent coauthors too
  - name: Bob Jones
    openalex_id: "A87654321"
    orcid: null
    expand_coauthors: false

# ── Bluesky follow list ───────────────────────────────────────────
# The app fetches recent posts from these handles and extracts any
# URLs that resolve to papers (doi.org links, publisher URLs, arXiv links).
# A paper mentioned by someone on this list gets an annotation in the digest.
# This is intentionally a short, curated list — quality over quantity.
bluesky_handles:
  - researcher1.bsky.social
  - labgroup.bsky.social
  # Add handles of people whose paper-sharing you trust.

# ── Scoring weights ───────────────────────────────────────────────
# These are relative weights; they are normalised internally.
# semantic_similarity is the dominant signal — keep it highest.
scoring:
  semantic_similarity: 0.60   # cosine sim between abstract and research_profile
  journal_tier_bonus:          # additive flat bonuses by tier
    tier1: 0.15
    tier2: 0.05
  priority_author_bonus: 0.20  # if any author is in priority list or is an expanded coauthor
  bluesky_mention_bonus: 0.10  # if any handle in bluesky_handles linked this paper

# ── Digest settings ───────────────────────────────────────────────
digest:
  lookback_days: 7             # how far back to search
  max_papers_in_digest: 25     # hard cap on papers included
  min_similarity_threshold: 0.20  # papers below this are dropped before scoring
  ollama_model: "qwen2.5:7b"   # must match a model you have pulled in Ollama
  ollama_host: "http://localhost:11434"
```

---

## Database schema — `db/database.py`

Use **SQLModel** (wraps SQLAlchemy + Pydantic). Four tables.

### `Paper`

| column | type | notes |
|---|---|---|
| `id` | str PK | DOI if available, else arXiv ID |
| `title` | str | |
| `abstract` | str \| None | null if unavailable |
| `authors` | JSON | list of `{name, openalex_id, orcid}` |
| `journal` | str \| None | |
| `published_date` | date | |
| `doi` | str \| None | |
| `arxiv_id` | str \| None | |
| `openalex_id` | str \| None | |
| `source` | str | `"openalex"` \| `"arxiv"` |
| `fetched_at` | datetime | |

### `ScoredPaper`

| column | type | notes |
|---|---|---|
| `id` | int PK | |
| `paper_id` | str FK → Paper | |
| `digest_run_id` | int FK → DigestRun | |
| `similarity_score` | float | raw SPECTER2 cosine score |
| `final_score` | float | after bonuses |
| `priority_author_match` | bool | |
| `bluesky_mentions` | JSON | list of handles that mentioned this paper |
| `summary` | str \| None | Ollama-generated summary |
| `included_in_digest` | bool | |

### `DigestRun`

| column | type | notes |
|---|---|---|
| `id` | int PK | |
| `created_at` | datetime | |
| `window_start` | date | |
| `window_end` | date | |
| `papers_fetched` | int | |
| `papers_included` | int | |
| `markdown_path` | str | path to saved .md file |
| `status` | str | `"running"` \| `"complete"` \| `"error"` |

### `CoauthorCache`

| column | type | notes |
|---|---|---|
| `anchor_openalex_id` | str | the priority author |
| `coauthor_openalex_id` | str | |
| `coauthor_name` | str | |
| `collaboration_count` | int | how many shared papers |
| `cached_at` | datetime | refresh if older than 30 days |

---

## Services

### `services/fetcher.py`

**OpenAlex fetch:**

- Endpoint: `https://api.openalex.org/works`
- Filter by `primary_location.source.issn` for each ISSN in the journal list (OR query).
- Filter by `publication_date` for the lookback window.
- Request fields: `id, doi, title, abstract_inverted_index, authorships, primary_location, publication_date`.
- Abstracts come back as an inverted index (word → list of positions) — reconstruct into plain text with a small utility function.
- Always pass `mailto=` param (polite pool; faster rate limits).
- Paginate with `cursor=*` until exhausted.
- Rate limit: stay under 10 req/s; add `asyncio.sleep(0.1)` between pages.

**arXiv fetch:**

- Endpoint: `http://export.arxiv.org/api/query`
- Build query as `cat:q-bio.GN OR cat:q-bio.CB` etc., filtered by `submittedDate`.
- Parse Atom XML response with `feedparser`.
- Extract: `entry.id` (arXiv ID), `entry.title`, `entry.summary` (full abstract — no reconstruction needed), `entry.authors`, `entry.published`, `entry.arxiv_journal_ref` (set if published in a journal).
- Use `journal_ref` to attempt DOI lookup via CrossRef if needed for dedup.
- Fetch up to 200 results per category; paginate with `start=` param.

**Deduplication:**

- Primary key: DOI (normalised lowercase, stripped of `https://doi.org/` prefix).
- Secondary: arXiv ID.
- Tertiary: fuzzy title match (Levenshtein distance < 5 on normalised title) — catches the same paper appearing from both sources without a shared DOI yet.
- On conflict: merge records, preferring the source with a non-null abstract.

---

### `services/coauthors.py`

Called once per priority author marked `expand_coauthors: true`, with results cached in `CoauthorCache` for 30 days.

- Query OpenAlex for all works by the anchor author: `https://api.openalex.org/works?filter=authorships.author.id:{openalex_id}`
- Collect all co-appearing author IDs across those works.
- Count co-occurrences; keep those with `collaboration_count >= 3` (configurable).
- Store in `CoauthorCache`.
- At scoring time, a paper gets the `priority_author_bonus` if any of its authors matches a priority author ID **or** any coauthor ID in the cache.

---

### `services/bluesky.py`

- For each handle in `bluesky_handles`, call `GET https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={handle}&limit=100`.
- No authentication needed for public profiles.
- For each post, extract all URLs from:
  - `post.record.facets` (typed link facets — most reliable)
  - Plain URL patterns in `post.record.text` as fallback regex
- Resolve each URL with a HEAD request (follow redirects) to get the final URL.
- Check final URL against known paper patterns:
  - `doi.org/` → extract DOI directly
  - `nature.com/articles/` → extract DOI from path
  - `science.org/doi/` → extract DOI
  - `arxiv.org/abs/` → extract arXiv ID
  - `biorxiv.org/content/` → extract DOI
  - `pubmed.ncbi.nlm.nih.gov/` → use PubMed ID to look up DOI via E-utils
- Return a dict `{doi_or_arxiv_id: [list of handles that mentioned it]}`.
- Cache results per run; don't re-fetch within the same digest run.
- Be conservative with rate limits: 1 req/s on URL resolution.

---

### `services/scorer.py`

**Setup (run once at app startup):**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("allenai-specter2")

def encode_profile(research_profile: str) -> np.ndarray:
    # SPECTER2 expects: title + [SEP] + abstract
    # For the profile, treat the whole text as the abstract with an empty title
    return model.encode("[SEP] " + research_profile, normalize_embeddings=True)

def encode_paper(title: str, abstract: str) -> np.ndarray:
    return model.encode(title + " [SEP] " + abstract, normalize_embeddings=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors are already normalised
```

**Scoring pipeline:**

1. Encode profile once; cache in memory for the run.
2. For each paper with a non-null abstract, encode and compute cosine similarity.
3. Drop papers below `min_similarity_threshold`.
4. Apply bonuses from config (journal tier, priority author, Bluesky mention).
5. Sort descending by `final_score`; take top `max_papers_in_digest`.
6. Papers with null abstracts: keep if they have a Bluesky mention or priority author match; otherwise drop.

---

### `services/builder.py`

**Ollama call** (one paper at a time; run concurrently with `asyncio.gather` up to 5 at once to avoid overwhelming Ollama):

```
System prompt:
  You are a research assistant helping a scientist stay current with the literature.
  The scientist's current focus: {research_profile}
  Be concise. Use plain language. Avoid restating the title.

User prompt per paper:
  Title: {title}
  Authors: {author_string}
  Journal: {journal} ({published_date})
  
  Abstract:
  {abstract}
  
  Write exactly two things:
  1. A 2-3 sentence lay summary of what this paper does and finds.
  2. One sentence on its relevance (or lack thereof) to the researcher's focus above.
  
  Format:
  SUMMARY: <text>
  RELEVANCE: <text>
```

**Digest markdown structure:**

```markdown
# Research digest — {window_start} to {window_end}

Generated {created_at} · {papers_included} papers from {papers_fetched} fetched

---

## Highlighted by people you follow

<!-- Papers with bluesky_mentions first, sorted by final_score -->

### {title}
**{authors}** · {journal} · {date} · [DOI]({doi})
*Mentioned by: {handles}*

{SUMMARY}

{RELEVANCE}

---

## Top papers by relevance

<!-- Remaining papers sorted by final_score -->

### {title}
...

---

## Priority author papers

<!-- Papers matching priority authors not already in top sections -->

### {title}
...
```

Save markdown to `digests/digest_{run_id}_{date}.md`.

---

## FastAPI routes — `main.py`

| method | path | description |
|---|---|---|
| GET | `/` | redirect to `/digests` |
| GET | `/digests` | list all digest runs; HTMX-rendered table |
| GET | `/digests/{run_id}` | render a single digest (markdown → HTML via `python-markdown`) |
| GET | `/digests/{run_id}/raw` | serve the raw `.md` file for download |
| POST | `/run` | trigger a digest run manually; returns a progress SSE stream |
| GET | `/config` | config editor page |
| POST | `/config` | save updated config.yaml; re-encode profile embedding |
| GET | `/status` | JSON status of current run (for polling) |

Use **Server-Sent Events** (SSE) on `/run` to stream progress back to the browser: `fetching papers...`, `scoring...`, `summarising 1/20...`, `done`. HTMX handles this natively with `hx-ext="sse"`.

---

## Scheduler — `scheduler.py`

Use `APScheduler` with `AsyncIOScheduler`. Wire up on app startup via FastAPI lifespan.

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app):
    scheduler.add_job(run_digest_job, "cron", day_of_week="mon", hour=7)
    scheduler.start()
    yield
    scheduler.shutdown()
```

The job calls the same function as the manual `/run` endpoint.

---

## Dependencies — `requirements.txt`

```
fastapi
uvicorn[standard]
sqlmodel
apscheduler
httpx
feedparser
python-markdown
sentence-transformers
torch          # CPU-only is fine: pip install torch --index-url https://download.pytorch.org/whl/cpu
numpy
python-levenshtein
pyyaml
jinja2
```

Ollama is a separate binary install (`curl -fsSL https://ollama.com/install.sh | sh`), not a Python package. The app talks to it over HTTP.

---

## Known limitations and deferred work

- **Abstract reconstruction from OpenAlex inverted index** is reliable but verbose; a utility function is needed. See OpenAlex docs on `abstract_inverted_index`.
- **arXiv ↔ published paper matching**: `journal_ref` field is author-supplied and inconsistently formatted. A fallback title-based CrossRef lookup is advisable but not required for MVP.
- **Bluesky URL resolution** will hit some dead links and paywalled pages gracefully — always use HEAD not GET, catch timeouts, move on.
- **SPECTER2 first load** downloads ~500 MB of model weights. This happens once and is cached by `sentence-transformers` in `~/.cache`.
- **Coauthor expansion** can be slow on first run for prolific authors (many pages of works to fetch). Run it as a background task on config save, not inline during digest generation.
- **No authentication** on the web UI — this is intentional for a localhost-only personal tool. Do not expose it publicly without adding auth.

---

## Build order recommendation for Claude Code

1. `db/database.py` — define all tables with SQLModel; run `create_all()`.
2. `config.yaml` — populate with placeholder values; write a `load_config()` utility.
3. `services/fetcher.py` — OpenAlex first, then arXiv. Test with a real ISSN and date range before moving on.
4. `services/bluesky.py` — fetch one handle, print extracted URLs. Verify resolution logic.
5. `services/coauthors.py` — fetch and cache coauthors for one test author.
6. `services/scorer.py` — encode a test profile and a few abstracts; print similarity scores to sanity-check.
7. `services/builder.py` — wire Ollama call; test with a single paper before batching.
8. `main.py` — FastAPI shell with all routes stubbed; add real logic progressively.
9. `templates/` — base layout, digest viewer, config editor.
10. `scheduler.py` — add last once everything else works manually.
