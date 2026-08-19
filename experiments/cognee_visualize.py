"""
Knowledge-graph visualization experiment — standalone, NOT wired into the
main app. Generates an interactive HTML file showing the entities/
relationships cognee extracted (from experiments/cognee_e1.py's graph).

This is purely exploratory (PLANNING.md Section 17) — no citation claims,
just a visual look at what the graph looks like.

Run from the project root, AFTER cognee_e1.py has run at least once:
    python experiments/cognee_visualize.py

Then open experiments/graph_visualization.html in a browser.
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

from cognee.api.v1.visualize.visualize import visualize_graph

OUTPUT_PATH = "experiments/graph_visualization.html"


async def main():
    await visualize_graph(OUTPUT_PATH)
    print(f"Graph visualization written to {OUTPUT_PATH}")
    print("Open it in a browser to explore the entities/relationships.")


if __name__ == "__main__":
    asyncio.run(main())
