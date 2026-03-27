# Readers Digest Extravaganzy

A locally-hosted research digest tool that fetches papers from arXiv and OpenAlex, scores them against your research profile using SPECTER2 embeddings, annotates with Bluesky social signals, and generates a summarised markdown digest using a local LLM via Ollama.

## Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/) (Anaconda or Miniconda)
- [Ollama](https://ollama.com/) for local LLM summarisation

## Setup

### 1. Get an OpenAlex API key

All OpenAlex API requests require a free API key (required as of February 2026 — the old `mailto=` polite pool no longer works).

1. Go to [openalex.org/settings/api](https://openalex.org/settings/api)
2. Create a free account and generate an API key

### 2. Create your `.env` file

The API key must **not** go in `config.yaml` or anywhere else tracked by git, as this is a public repository.

Copy the example file and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENALEX_API_KEY=your_actual_key_here
OLLAMA_HOST=http://localhost:11434
```

The `.env` file is listed in `.gitignore` and will never be committed.

### 3. Create the conda environment

```bash
conda create -n readers-digest python=3.11 -y
conda activate readers-digest
pip install -r requirements.txt
```

### 4. Install and start Ollama

```bash
brew install ollama          # macOS
brew services start ollama
ollama pull qwen2.5:7b       # ~4.7 GB download
```

On first run the SPECTER2 model (~500 MB) is also downloaded automatically and cached in `~/.cache/huggingface/`.

### 5. Configure

Copy the example config and edit it with your own details:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to set:

- **research_profile** -- free-text description of your current work (used for SPECTER2 similarity scoring)
- **arxiv_categories** -- arXiv categories to fetch papers from
- **journals** -- journal ISSNs and tiers for OpenAlex fetching and score bonuses
- **bluesky_handles** -- Bluesky accounts whose shared papers you want highlighted
- **priority_authors** -- authors whose papers get a score boost
- **scoring** -- weights for similarity, journal, author, and Bluesky bonuses
- **digest settings** -- lookback window, max papers, similarity threshold, Ollama model

This file **is** committed — do not put secrets in it.

## Running the digest pipeline

```bash
conda activate readers-digest
python -m run_digest
```

This runs the full pipeline:

1. Fetches papers from arXiv and OpenAlex for the configured categories/journals and lookback window
2. Fetches Bluesky sightings from your follow list
3. Merges and deduplicates papers
4. Scores papers against your research profile (SPECTER2 cosine similarity + bonuses)
5. Classifies papers into sections: "People you follow are talking about", "From your field", "Worth noticing"
6. Summarises each paper using Ollama (locally, no API keys needed)
7. Saves the digest to `digests/digest_{date}.md`

## Running tests

```bash
# Unit tests (mocked, no network needed)
pytest tests/test_scorer.py

# Smoke test (hits real APIs, needs network + SPECTER2 model)
python -m tests.smoke_test
```

## Project structure

```
services/
    fetcher.py          arXiv + OpenAlex fetcher, deduplication, orchestrator
    openalex_client.py  OpenAlex API client (pyalex wrapper)
    bluesky.py          Bluesky sighting extractor
    scorer.py           SPECTER2 scoring, merge/dedup, section classification
    builder.py          Ollama summarisation and markdown assembly
    coauthors.py        Coauthor expansion
shared/
    types.py            Shared dataclasses (Paper, BlueskySighting, DigestResult, etc.)
db/
    database.py         SQLModel schema
tests/                  Unit and integration tests
digests/                Generated markdown digests (gitignored)
config.yaml             Your personal config (gitignored)
run_digest.py           CLI entry point for the full pipeline
```
