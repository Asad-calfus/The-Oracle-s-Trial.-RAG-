"""
Step E1 of PLANNING.md Section 17 — first standalone cognee test.

Does NOT touch backend/ or any existing module. Purpose: find out how long
cognee.cognify() takes and roughly how many LLM calls it makes, on ONE real
project document, before testing at any bigger scale.

Run from the project root:
    python experiments/cognee_e1.py
"""

import asyncio
import time

import pdfplumber
from dotenv import load_dotenv
import os

load_dotenv()

# cognee reads its own env var name for the LLM key — reuse the same key
# already configured for this project rather than asking for a second one.
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

import cognee

# Pick one real, already-uploaded document to test against — kept small on
# purpose for this first run, since cognify() cost/time scales with content.
TEST_PDF = "data/documents/9/somatosensory-2.pdf"


def load_text(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


async def main():
    text = load_text(TEST_PDF)
    print(f"Loaded {len(text)} characters from {TEST_PDF}")

    # Clean slate so repeated runs of this script don't accumulate old data.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print("Adding document...")
    await cognee.add(text)

    print("Running cognify() — this is the expensive step, timing it...")
    start = time.time()
    await cognee.cognify()
    elapsed = time.time() - start
    print(f"cognify() took {elapsed:.1f} seconds")

    print("\nSanity-check search:")
    results = await cognee.search("What is this document about?")
    for r in results:
        print("-", r)


if __name__ == "__main__":
    asyncio.run(main())
