"""
Step E2 of PLANNING.md Section 17 — compare cognee's search types on the
same question, and check whether any of them carry a traceable citation
the way this project's own /query endpoint guarantees (see PLANNING.md
17.3 — this is the real open question, not just answer quality).

Reuses the graph already built by experiments/cognee_e1.py (cognee keeps
its data between runs unless pruned) — no re-ingestion here.

Run from the project root, AFTER cognee_e1.py has run at least once:
    python experiments/cognee_e2.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

from cognee import SearchType, search

QUESTION = "How are muscle spindles and joint receptors related to proprioceptive feedback?"

SEARCH_TYPES = {
    "GRAPH_COMPLETION": SearchType.GRAPH_COMPLETION,
    "RAG_COMPLETION": SearchType.RAG_COMPLETION,
    "CHUNKS": SearchType.CHUNKS,  # closest thing to "trace the source"
}


async def main():
    for label, search_type in SEARCH_TYPES.items():
        print(f"\n=== {label} ===")
        results = await search(query_text=QUESTION, query_type=search_type)
        for r in results:
            print(r)


if __name__ == "__main__":
    asyncio.run(main())
