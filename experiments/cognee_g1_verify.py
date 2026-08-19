"""
Step G1 verification — does scoping by dataset_name actually keep two
documents' graphs isolated? cognee's own docs/issue tracker mention this
isn't always guaranteed, so this is tested directly rather than assumed.

Uses two tiny, clearly unrelated texts (cheap and fast) instead of real
PDFs — isolation either works or it doesn't, content size doesn't matter
for this check.

Run from the project root:
    python experiments/cognee_g1_verify.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

import cognee
from cognee import SearchType, search

from backend.knowledge_graph import dataset_name_for

DOC_A_TEXT = "Penguins live in Antarctica and eat fish."
DOC_B_TEXT = "The Eiffel Tower is in Paris and was built in 1889."

DATASET_A = dataset_name_for(thread_id=999, filename="penguins.txt")
DATASET_B = dataset_name_for(thread_id=999, filename="eiffel_tower.txt")


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print(f"Adding doc A to dataset '{DATASET_A}'...")
    await cognee.add(DOC_A_TEXT, dataset_name=DATASET_A)
    print(f"Adding doc B to dataset '{DATASET_B}'...")
    await cognee.add(DOC_B_TEXT, dataset_name=DATASET_B)

    print("Running cognify() on both datasets...")
    await cognee.cognify(datasets=[DATASET_A, DATASET_B])

    # The real test: ask about doc A's content, but SCOPE the search to
    # doc B's dataset only. If isolation works, this should find nothing
    # about penguins. If it leaks, the answer will mention penguins anyway.
    print(f"\nSearching for penguin content, scoped ONLY to dataset '{DATASET_B}' (should find NOTHING):")
    leaked = await search(
        query_text="What does the document say about penguins?",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET_B],
    )
    print(leaked)

    print(f"\nSanity check — same question, scoped to the CORRECT dataset '{DATASET_A}' (should find penguins):")
    correct = await search(
        query_text="What does the document say about penguins?",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET_A],
    )
    print(correct)


if __name__ == "__main__":
    asyncio.run(main())
