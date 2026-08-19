import logging

from backend.config import RETRIEVAL_TOP_K
from backend.llm import get_llm

logger = logging.getLogger(__name__)

# The LLM is asked to return ONLY passage numbers (never the passage text
# itself) — same principle as citations: never let the LLM retype content,
# only let it choose from what we already have.
RERANK_PROMPT_TEMPLATE = """You will be given a question and a numbered list of text passages.

Select the passages that actually help answer the question, and list them from most relevant to least relevant.

LEAVE OUT any passage that does not help answer the question — do not include it just to fill the list.
If none of the passages help, return nothing at all.

Return ONLY a comma-separated list of the selected passage numbers, for example: 3,1,4
Do not include any other text, explanation, or the passage contents themselves.

Question:
{question}

Passages:
{passages}

Selected passage numbers:"""


def build_rerank_prompt(question: str, chunks) -> str:
    """Number the candidate chunks and fill the reranking prompt template."""
    passages = "\n\n".join(
        f"[{i + 1}] {chunk.page_content}" for i, chunk in enumerate(chunks)
    )
    return RERANK_PROMPT_TEMPLATE.format(question=question, passages=passages)


def rerank_chunks(question: str, chunks, top_k: int = RETRIEVAL_TOP_K):
    """Pick the chunks that actually help answer the question, best first.

    Returns at most top_k chunks (defaults to the config constant, but a
    thread's own resolved setting — Section 21 — can override it), and
    possibly none — irrelevant candidates are dropped rather than kept to
    pad the list, so they can't end up cited as evidence for an answer
    they had no part in.

    Falls back to the original (embedding-similarity) order if the LLM's
    reply can't be parsed into valid passage numbers — reranking should
    never be able to break the pipeline, only improve it.
    """
    if not chunks:
        return chunks

    prompt = build_rerank_prompt(question, chunks)
    llm = get_llm()
    response = llm.invoke(prompt)
    reply = response.content.strip()
    logger.debug("Reranker LLM reply for %d candidates: %r", len(chunks), reply)

    # An empty reply is the model deliberately saying "none of these help",
    # which is a real answer — not a parsing failure to fall back from.
    if not reply:
        logger.info("Reranker selected 0 of %d candidates", len(chunks))
        return []

    ranked_chunks = []
    seen_indices = set()
    for token in reply.split(","):
        token = token.strip()
        if not token.isdigit():
            continue
        index = int(token) - 1
        if 0 <= index < len(chunks) and index not in seen_indices:
            seen_indices.add(index)
            ranked_chunks.append(chunks[index])

    # Non-empty reply that yielded no usable numbers => we failed to read it.
    if not ranked_chunks:
        logger.warning(
            "Could not parse any passage numbers from reranker reply %r — "
            "falling back to embedding-similarity order",
            reply,
        )
        return chunks[:top_k]

    logger.info("Reranker selected %d of %d candidates", len(ranked_chunks), len(chunks))
    return ranked_chunks[:RETRIEVAL_TOP_K]
