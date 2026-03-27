"""
Smoke test: fetch abstract from OpenAlex given a DOI.

Usage:
    python test_abstract_by_doi.py 10.1038/s41586-024-07487-w
    python test_abstract_by_doi.py  # uses a default DOI
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from services.openalex_client import fetch_abstract_by_doi, init_openalex

api_key = os.environ.get("OPENALEX_API_KEY")
if not api_key:
    print("ERROR: OPENALEX_API_KEY not set in .env")
    sys.exit(1)

init_openalex(api_key)

doi = sys.argv[1] if len(sys.argv) > 1 else "10.1038/s41586-024-07487-w"

print(f"Looking up DOI: {doi}")
abstract = fetch_abstract_by_doi(doi)

if abstract:
    print(f"\nAbstract ({len(abstract)} chars):\n")
    print(abstract)
else:
    print("\nNo abstract found (paper may not be in OpenAlex, or has no abstract).")
