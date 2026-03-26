# Scoring branch — work in progress

## Branch
`feat/scoring-and-summarizer`

## What's been done

### shared/types.py (new file)
- `Author`, `Paper`, `BlueskySighting` dataclasses per the synthesis_model_spec.md
- Minimal — just what the scorer needs

### services/scorer.py (implemented)
- **Merge & dedup**: combines OpenAlex + arXiv paper lists. Dedup by DOI → arXiv ID → fuzzy title (Levenshtein ≤ 5). Merged records keep all non-None fields.
- **Attach Bluesky sightings**: match sightings to papers by DOI or arXiv ID. Unmatched sightings discarded.
- **Score**: `final_score = (cosine_sim × semantic_similarity_weight) + journal_bonus + author_bonus + bluesky_bonus`
  - No-abstract papers score 0.0 similarity but kept if they have Bluesky or priority author signal
  - ISSN exact match for journal tier bonuses
- **Classify into sections** (mutually exclusive, priority order): "following" (has Bluesky) → "field" (author match or tier-1 journal above threshold) → "notable" (rest)
- **Trending flag**: ≥ 2 distinct Bluesky handles
- **Cap**: sorted by final_score desc, capped at max_papers_in_digest
- Embedding: loads SPECTER2 lazily via `_get_model()`, exposes `encode_profile()` and `encode_paper()`
- Output types: `ScoredPaper`, `ScoringResult`, config types `ScorerConfig`, `ScoringWeights`, `JournalConfig`

### tests/test_scorer.py (written, not yet run)
- Mocks SPECTER2 with fake embeddings so tests run without model download
- Covers: dedup (DOI, arXiv ID, fuzzy title, merge preference), scoring math (each bonus type), section classification priority, trending flag, no-abstract handling, max cap, unmatched sightings

## What's left
1. **Pull main** to get the newly merged feature (SSH blocked in this session)
2. **Rebase** `feat/scoring-and-summarizer` onto updated main
3. **Set up conda env** `readers-digest` (was created with python=3.11 but deps not installed yet)
4. **Install deps**: `pip install numpy python-Levenshtein sentence-transformers pytest`
5. **Run tests**: `pytest tests/test_scorer.py -v` from repo root
6. Fix any test failures

## Scope decisions
- **No summarization** — builder.py / Ollama calls deferred to a later branch
- **No digest markdown assembly** — deferred
- Scorer does NOT call APIs, read config, or touch the DB — it receives everything as arguments
