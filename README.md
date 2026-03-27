# Readers Digest Extravaganzy

A locally-hosted research digest tool that fetches papers from arXiv (and OpenAlex, planned), scores them against your research profile using SPECTER2 embeddings, annotates with Bluesky social signals, and generates a summarised markdown digest using a local LLM via Ollama.

## Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/) (Anaconda or Miniconda)
- [Ollama](https://ollama.com/) for local LLM summarisation

## Setup

### 1. Create the conda environment

```bash
conda create -n readers-digest python=3.11 -y
conda activate readers-digest
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
brew install ollama          # macOS
brew services start ollama
ollama pull qwen2.5:7b       # ~4.7 GB download
```

On first run the SPECTER2 model (~500 MB) is also downloaded automatically and cached in `~/.cache/huggingface/`.

### 3. Configure

Copy the example config and edit it with your own details:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to set:

- **research_profile** -- free-text description of your current work (used for SPECTER2 similarity scoring)
- **arxiv_categories** -- arXiv categories to fetch papers from
- **bluesky_handles** -- Bluesky accounts whose shared papers you want highlighted
- **priority_authors** -- authors whose papers get a score boost
- **journals** -- journal tiers for score bonuses
- **scoring** -- weights for similarity, journal, author, and Bluesky bonuses
- **digest settings** -- lookback window, max papers, similarity threshold, Ollama model

## Running the digest pipeline

```bash
conda activate readers-digest
python -m run_digest
```

This runs the full pipeline:

1. Fetches papers from arXiv for the configured categories and lookback window
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
    fetcher.py      arXiv paper fetcher
    bluesky.py      Bluesky sighting extractor
    scorer.py       SPECTER2 scoring, merge/dedup, section classification
    builder.py      Ollama summarisation and markdown assembly
    coauthors.py    Coauthor expansion (planned)
shared/
    types.py        Shared dataclasses (Paper, BlueskySighting, DigestResult, etc.)
db/
    database.py     SQLModel schema
tests/              Unit and integration tests
digests/            Generated markdown digests (gitignored)
config.yaml         Your personal config (gitignored)
run_digest.py       CLI entry point for the full pipeline
```
