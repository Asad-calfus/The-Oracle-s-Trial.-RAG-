from backend.config import RETRIEVAL_TOP_K
from backend.llm import get_llm

# The LLM is asked to return ONLY passage numbers (never the passage text
# itself) — same principle as citations: never let the LLM retype content,
# only let it choose from what we already have.
RERANK_PROMPT_TEMPLATE = """You will be given a question and a numbered list of text passages.

Rank the passages by how relevant they are to answering the question, from most relevant to least relevant.

Return ONLY a comma-separated list of the passage numbers in that order, for example: 3,1,4,2
Do not include any other text, explanation, or the passage contents themselves.

Question:
{question}

Passages:
{passages}

Ranked passage numbers:"""


def build_rerank_prompt(question: str, chunks) -> str:
    """Number the candidate chunks and fill the reranking prompt template."""
    passages = "\n\n".join(
        f"[{i + 1}] {chunk.page_content}" for i, chunk in enumerate(chunks)
    )
    return RERANK_PROMPT_TEMPLATE.format(question=question, passages=passages)


def rerank_chunks(question: str, chunks):
    """Reorder candidate chunks by asking the LLM to judge relevance directly.

    Returns the top RETRIEVAL_TOP_K chunks, in relevance order. Falls back to
    the original (embedding-similarity) order if the LLM's reply can't be
    parsed into valid passage numbers — reranking should never be able to
    break the pipeline, only improve it.
    """
    if not chunks:
        return chunks

    prompt = build_rerank_prompt(question, chunks)
    llm = get_llm()
    response = llm.invoke(prompt)

    ranked_chunks = []
    seen_indices = set()
    for token in response.content.split(","):
        token = token.strip()
        if not token.isdigit():
            continue
        index = int(token) - 1
        if 0 <= index < len(chunks) and index not in seen_indices:
            seen_indices.add(index)
            ranked_chunks.append(chunks[index])

    if not ranked_chunks:
        return chunks[:RETRIEVAL_TOP_K]

    return ranked_chunks[:RETRIEVAL_TOP_K]
