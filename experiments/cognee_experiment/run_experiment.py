"""
Standalone cognee experiment — NOT wired into backend/.

Ingests one existing test PDF through cognee.add() + cognee.cognify(),
then renders an interactive HTML graph visualization so we can look at
the entities/relationships cognee extracted.

Usage:
    source .venv/bin/activate
    pip install pymupdf  # if not already installed, for text extraction
    python experiments/cognee_experiment/run_experiment.py
"""

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# cognee reads its own LLM_API_KEY env var, separate from OPENAI_API_KEY
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_MODEL", "gpt-4o-mini")

import cognee
import pymupdf

TEST_PDF = REPO_ROOT / "data" / "documents" / "5" / "worldhealthstatistics_2022.pdf"
OUTPUT_HTML = Path(__file__).parent / "graph_visualization.html"


def extract_text(pdf_path: Path, max_pages: int = 15) -> str:
    with pymupdf.open(pdf_path) as doc:
        pages = doc[:max_pages] if len(doc) > max_pages else doc
        return "\n".join(page.get_text() for page in pages)


async def main():
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    print(f"Extracting text from {TEST_PDF.name} ...")
    text = extract_text(TEST_PDF)
    print(f"  {len(text):,} characters extracted (first 15 pages)")

    # Start clean so repeated runs don't accumulate duplicate graph data
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print("Running cognee.add() ...")
    await cognee.add(text, dataset_name="who_stats_experiment")

    print("Running cognee.cognify() — this is the expensive step (multiple LLM calls) ...")
    t0 = time.time()
    await cognee.cognify(datasets=["who_stats_experiment"])
    elapsed = time.time() - t0
    print(f"  cognify() finished in {elapsed:.1f}s")

    print("Rendering graph visualization ...")
    await cognee.visualize_graph(str(OUTPUT_HTML))
    print(f"  Graph saved to: {OUTPUT_HTML}")

    print("\nRunning sample searches for comparison ...")
    question = "What factors are associated with life expectancy trends discussed in this report?"

    for search_type in (cognee.SearchType.GRAPH_COMPLETION, cognee.SearchType.RAG_COMPLETION):
        print(f"\n--- {search_type} ---")
        result = await cognee.search(query_text=question, query_type=search_type)
        print(result)

    print(f"\nDone. Open {OUTPUT_HTML} in a browser to view the graph.")


if __name__ == "__main__":
    asyncio.run(main())
