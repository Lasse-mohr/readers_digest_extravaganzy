# Synthesis model — interface specification

**Version:** 0.1  
**Owner:** Digest model team  
**Status:** Draft — for review by OpenAlex, arXiv, and Bluesky sub-teams

---

## Purpose

This document defines the data contracts between the three API fetcher modules
(OpenAlex, arXiv, Bluesky) and the synthesis model. It specifies:

- The exact Python types each fetcher must return
- What the synthesis model does with those inputs
- What the synthesis model produces as output

If you are working on a fetcher, this document tells you what shape your output
must take. If you are working on the synthesis model, this document is your
contract with the fetchers.

---

## Shared types

These types are used across all three fetcher outputs. Define them in a shared
module, e.g. `shared/types.py`, and import from there. Do not redefine them
per-module.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Author:
    """
    A single author on a paper.
    All fields except `name` are optional — not every source provides them.
    At least one of openalex_id or orcid should be populated when available,
    as these are used by the scorer to match against the priority author list.
    """
    name: str                          # Display name, e.g. "Jane Smith"
    openalex_id: Optional[str] = None  # e.g. "A12345678"
    orcid: Optional[str] = None        # e.g. "0000-0001-2345-6789"


@dataclass
class Paper:
    """
    A single paper record as returned by a fetcher.

    Fetchers must populate every field they have data for.
    Fields marked Optional may be None if genuinely unavailable —
    do not substitute empty strings for None.

    Identity fields (doi, arxiv_id, openalex_id):
      - At least one must be non-None.
      - doi is preferred as the canonical dedup key.
      - Normalise DOIs to lowercase, strip leading "https://doi.org/".
    """
    # ── Identity ──────────────────────────────────────────────────
    doi: Optional[str]                 # e.g. "10.1038/s41586-024-00001-1"
    arxiv_id: Optional[str]            # e.g. "2401.12345"
    openalex_id: Optional[str]         # e.g. "W12345678"

    # ── Content ───────────────────────────────────────────────────
    title: str
    abstract: Optional[str]            # Plain text. None if unavailable.
    authors: list[Author]              # Ordered as they appear on the paper.

    # ── Provenance ────────────────────────────────────────────────
    journal: Optional[str]             # Journal/venue display name.
    journal_issn: Optional[str]        # ISSN, used to match config journal list.
    published_date: date               # Date of first public availability.
    source: str                        # "openalex" | "arxiv"

    # ── URLs ──────────────────────────────────────────────────────
    url: Optional[str]                 # Canonical landing page for this paper.
```

---

## Fetcher output contracts

Each fetcher returns a single value: `list[Paper]`. Nothing else.

The synthesis model calls all three fetchers and merges their outputs.
Fetchers are not responsible for deduplication — that happens in the model.

### OpenAlex fetcher

```python
def fetch_openalex(
    issnl_list: list[str],
    date_from: date,
    date_to: date,
) -> list[Paper]:
    ...
```

**Notes for the OpenAlex team:**

- `issnl_list` is the list of ISSNs from `config.yaml`.
- Abstract comes back from OpenAlex as an inverted index
  (`{"word": [position, ...], ...}`). Reconstruct to plain text before
  populating `Paper.abstract`. If reconstruction yields an empty string, set
  `abstract=None`.
- `published_date` should be the `publication_date` field from OpenAlex
  (earliest of online-first and issue date).
- Populate `openalex_id` from the work's `id` field (strip the URL prefix,
  keep just the `W...` identifier).
- `journal` and `journal_issn` come from `primary_location.source.display_name`
  and `primary_location.source.issn_l` respectively.

### arXiv fetcher

```python
def fetch_arxiv(
    categories: list[str],
    date_from: date,
    date_to: date,
) -> list[Paper]:
    ...
```

**Notes for the arXiv team:**

- `categories` is the list from `config.yaml`, e.g. `["q-bio.GN", "cs.LG"]`.
- arXiv abstracts are plain text — populate `Paper.abstract` directly.
- `arxiv_id` should be the short form only: `"2401.12345"`, not the full URL.
- If `entry.arxiv_journal_ref` is set, populate `journal` with that string
  verbatim. It is author-supplied and inconsistently formatted — do not try
  to parse it.
- `doi` should be populated from `entry.arxiv_doi` if present.
- Set `source="arxiv"`.

### Bluesky fetcher

The Bluesky fetcher has a different contract because it does not fetch full
paper records — it fetches *signals* (links shared by watched accounts).
The synthesis model uses these signals to annotate papers fetched by the
other two fetchers.

```python
@dataclass
class BlueskySighting:
    """
    A record of one paper being linked by one Bluesky account.
    One paper linked by three accounts produces three BlueskySightings.
    """
    doi: Optional[str]         # Resolved DOI, normalised. Preferred match key.
    arxiv_id: Optional[str]    # Resolved arXiv ID. Used if doi is None.
    handle: str                # The Bluesky handle that posted the link.
    post_url: str              # Full URL of the Bluesky post itself.
    posted_at: date            # Date of the post.
    post_text: Optional[str]   # Full text of the post. May be useful for
                               # extracting commentary to include in digest.


def fetch_bluesky(
    handles: list[str],
    date_from: date,
    date_to: date,
) -> list[BlueskySighting]:
    ...
```

**Notes for the Bluesky team:**

- One call to `fetch_bluesky` returns all sightings across all handles.
- A single post may yield zero, one, or multiple sightings (if multiple paper
  links appear in one post). Emit one `BlueskySighting` per resolved paper
  link, not per post.
- URL resolution: follow redirects on every extracted URL. Classify the
  final URL against known paper domains to extract DOI or arXiv ID:
  - `doi.org/{doi}` → doi
  - `nature.com/articles/{suffix}` → doi = `10.1038/{suffix}`
  - `science.org/doi/{doi}` → doi
  - `cell.com/...` → doi (from URL or page metadata)
  - `arxiv.org/abs/{id}` → arxiv_id
  - `biorxiv.org/content/{doi}` → doi
  - If resolution fails or times out, discard the sighting silently.
- `post_text` should be the raw post text with no truncation.
- Do not filter by topic or try to determine whether the link is paper-related
  before resolution — the model handles that.

---

## Coauthor expansion input

The coauthor expansion module runs separately (see `services/coauthors.py` in
the project plan) and caches results to the database. The synthesis model reads
from that cache at runtime. The coauthor module is not a fetcher and does not
need to return `Paper` objects.

The synthesis model will query the `CoauthorCache` table directly.
No special contract is needed here beyond what is defined in the database schema.

---

## Synthesis model inputs

The synthesis model receives the following at runtime:

```python
@dataclass
class SynthesisInput:
    papers_openalex: list[Paper]
    papers_arxiv: list[Paper]
    bluesky_sightings: list[BlueskySighting]
    config: DigestConfig          # Parsed from config.yaml (see below)
    profile_embedding: np.ndarray # Pre-computed SPECTER2 embedding of
                                  # research_profile. Computed at app startup
                                  # or on config save; passed in, not computed
                                  # inside the model.


@dataclass
class DigestConfig:
    research_profile: str
    journals: list[JournalConfig]
    priority_author_ids: set[str]      # openalex_ids from config + coauthor cache
    bluesky_handles: list[str]
    scoring: ScoringConfig
    max_papers_in_digest: int
    min_similarity_threshold: float
    ollama_model: str
    ollama_host: str


@dataclass
class JournalConfig:
    name: str
    issn: str
    tier: int           # 1 or 2


@dataclass
class ScoringConfig:
    semantic_similarity_weight: float
    journal_tier1_bonus: float
    journal_tier2_bonus: float
    priority_author_bonus: float
    bluesky_mention_bonus: float
```

---

## Synthesis model internal pipeline

This section is for the model team only. It describes what the model does
with the inputs above. The fetcher teams do not need to implement any of this.

### Step 1 — Merge and deduplicate

Combine `papers_openalex` and `papers_arxiv` into a single list.

Dedup key priority:
1. If both records share a DOI → same paper. Merge: prefer non-None abstract;
   keep all author identity fields from both records.
2. If one has an arXiv ID that matches the other → same paper. Merge as above.
3. Fuzzy title match as last resort: normalise both titles (lowercase, strip
   punctuation), compute Levenshtein distance. If distance ≤ 5 → same paper.

After merging, the canonical record should have as many non-None fields as
possible. Log any merges for debugging.

### Step 2 — Attach Bluesky sightings

For each `BlueskySighting`, find the matching merged paper by DOI (preferred)
or arXiv ID. Attach the sighting to that paper's record. If no match is found,
discard the sighting — do not create a new paper record from a Bluesky link
alone.

Each paper now carries: `bluesky_sightings: list[BlueskySighting]`.

### Step 3 — Score each paper

For each paper:

```
base_score = cosine_similarity(paper_embedding, profile_embedding)
           # paper_embedding = SPECTER2.encode(title + " [SEP] " + abstract)
           # Skip paper if abstract is None (score = 0.0, may still include
           # if Bluesky-mentioned or priority-author match)

journal_bonus = tier1_bonus if journal_issn matches a tier-1 journal
              else tier2_bonus if tier-2
              else 0.0

author_bonus = priority_author_bonus
               if any author.openalex_id is in priority_author_ids
               else 0.0

bluesky_bonus = bluesky_mention_bonus if len(paper.bluesky_sightings) > 0
               else 0.0

final_score = (base_score * semantic_similarity_weight)
            + journal_bonus + author_bonus + bluesky_bonus
```

Drop papers where `base_score < min_similarity_threshold` AND
`bluesky_sightings == []` AND `author_bonus == 0.0`.
(Papers with no abstract that have a Bluesky mention or author match are
kept with `base_score = 0.0`.)

### Step 4 — Classify papers into digest sections

Before summarising, classify the scored papers into three non-overlapping
sections. A paper appears in exactly one section, in priority order:

| Section | Condition |
|---|---|
| **People you follow** | `len(bluesky_sightings) > 0` |
| **From your field** | `author_bonus > 0` OR `journal_issn` in tier-1 list AND `base_score >= threshold` |
| **Worth noticing** | Everything else above the score threshold |

Sort within each section by `final_score` descending.
Cap total papers across all sections at `max_papers_in_digest`.

Additionally, compute a **trending signal** across the full paper list:
a paper is flagged as trending if it appears in `bluesky_sightings` from
≥ 2 distinct handles. Trending papers are annotated in the digest regardless
of which section they fall into.

### Step 5 — Summarise each paper

Call Ollama once per paper. Run up to 5 concurrent async calls.

**System prompt (constant across all calls):**

```
You are a research assistant writing a weekly digest for a scientist.
Write clearly and precisely. Do not pad. Do not use filler phrases like
"the authors demonstrate" or "this study shows" — get to the finding directly.
The scientist's current focus:

{research_profile}
```

**User prompt per paper:**

```
Title: {title}
Authors: {author_names_joined}
Published: {published_date} in {journal or "arXiv"}
{if bluesky_sightings}: Mentioned on Bluesky by: {handles_joined}

Abstract:
{abstract}

Write:
SUMMARY: 2-3 sentences. What was done, what was found. Plain language.
RELEVANCE: 1 sentence. Is this relevant to the researcher's focus, and how?
           If it is not relevant, say so plainly — that is also useful.
```

Parse the response by splitting on `RELEVANCE:`. If parsing fails, retry once.
If the second attempt also fails, set `summary = raw_response` and
`relevance = None`.

### Step 6 — Write the section intros

After all papers are summarised, make one additional Ollama call per section
to write a 2-3 sentence intro paragraph for that section. This is what makes
the digest read like a newsletter rather than a list.

**Prompt for section intro:**

```
You are writing a brief intro paragraph for a section of a research digest.
The scientist's focus: {research_profile}

This section is called "{section_name}" and contains these papers:
{for each paper: "- {title} ({authors_short})"}

Write 2-3 sentences that characterise what is happening in this group of papers
this week. Note any theme, tension, or pattern across them if one exists.
Do not list the papers — they follow immediately after this paragraph.
If no clear pattern exists, say so in one sentence and move on.
```

### Step 7 — Assemble the markdown digest

Output structure:

```markdown
# Research digest · {date_from} – {date_to}

{total_papers_included} papers · fetched {created_at}

---

## People you follow are talking about

{section_intro}

### {title}
**{authors}** · {journal} · {date}  
{doi_link or arxiv_link}  
{if trending}: *Trending — mentioned by {N} people you follow*  
{if bluesky_sightings}: *Shared by: {handles with post links}*  
{if post_text is informative}: > "{post_text excerpt}"  

{SUMMARY}

*{RELEVANCE}*

---

## From your field

{section_intro}

### {title}
...

---

## Worth noticing

{section_intro}

### {title}
...
```

Save as `digests/digest_{YYYY-MM-DD}.md`.

---

## Synthesis model output

```python
@dataclass
class DigestResult:
    run_id: int
    created_at: datetime
    window_start: date
    window_end: date
    papers_fetched: int          # total before dedup
    papers_after_dedup: int
    papers_included: int         # in the final digest
    papers_dropped_no_abstract: int
    markdown_path: str           # absolute path to saved .md file
    markdown: str                # full markdown content (also returned in-memory
                                 # for the SSE progress stream)
    scored_papers: list[ScoredPaperRecord]  # for writing to DB


@dataclass
class ScoredPaperRecord:
    """Written to the ScoredPaper table in the database."""
    paper: Paper                         # the merged canonical record
    similarity_score: float
    final_score: float
    priority_author_match: bool
    bluesky_sightings: list[BlueskySighting]
    trending: bool
    summary: Optional[str]
    relevance: Optional[str]
    digest_section: str                  # "following" | "field" | "notable"
    included_in_digest: bool
```

---

## Error handling expectations

The synthesis model should be defensive about upstream data quality:

| Condition | Handling |
|---|---|
| `abstract` is None | Score as 0.0; include only if author/Bluesky match |
| `authors` is empty list | Score normally; no author bonus |
| `published_date` outside window | Log and discard — fetcher bug |
| Ollama call times out | Retry once after 5s; on second failure set summary to None, mark paper as included without summary |
| Ollama returns unparseable output | Use raw output as summary; relevance = None |
| No papers survive scoring | Emit a digest with a single section noting the digest is empty this week |
| Bluesky sighting with no matching paper | Discard silently |

---

## What the model does NOT do

To keep scope clear:

- Does not call the OpenAlex, arXiv, or Bluesky APIs directly.
- Does not read or write `config.yaml`.
- Does not manage the database — it returns `DigestResult` and the caller
  (FastAPI route handler) writes to the DB.
- Does not compute the `profile_embedding` — this is passed in as a
  pre-computed `np.ndarray`.
- Does not handle scheduling.

---

## Open questions for the team

1. **Bluesky `post_text` in digest**: Should we include a quoted excerpt of the
   post text if it contains commentary beyond just a link? Needs a heuristic
   for "is this post text informative" (e.g. length > 80 chars after removing
   the URL).

2. **Trending threshold**: Currently set to ≥ 2 handles. With a short follow
   list this may never trigger. Should it be configurable in `config.yaml`?

3. **Section size caps**: Should sections have individual caps (e.g. max 5 in
   "People you follow", max 15 in "From your field") or just a global cap on
   total papers? Global cap is simpler but may let one section dominate.

4. **Papers with no abstract and no Bluesky/author signal**: Currently dropped.
   Should there be a "titles only" appendix section for these so the user can
   decide manually whether to look them up?

5. **arXiv preprint vs published version**: If the same paper appears as both
   an arXiv preprint and a published journal paper, the merged record should
   prefer the journal metadata. Confirm with arXiv team whether `journal_ref`
   is reliably set when a paper has been published.
