import os

from backend.config import (
    RERANK_CANDIDATE_K,
    RETRIEVAL_TOP_K,
    SIMILARITY_SCORE_THRESHOLD,
)
from backend.llm import get_llm
from backend.reranker import rerank_chunks
from backend.vectorstore import get_vectorstore


def retrieve_documents(question: str):
    """Return the top-K most relevant chunks for a question via similarity search."""
    store = get_vectorstore()
    return store.similarity_search(question, k=RETRIEVAL_TOP_K)


def retrieve_with_scores(question: str, k: int = RETRIEVAL_TOP_K):
    """Same as retrieve_documents, but also return each chunk's similarity score."""
    store = get_vectorstore()
    return store.similarity_search_with_score(question, k=k)


# {context} and {question} are placeholders filled in by build_prompt() below.
# The wording here is the entire hallucination-protection mechanism (Rule 16) —
# there is no special API flag for "don't make things up", just this instruction.
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the context provided below.

If the answer is not contained in the context, respond exactly with:
"I don't know based on the uploaded documents."

Do not use any outside knowledge. Do not guess.

Context:
{context}

Question:
{question}

Answer:"""


def build_prompt(question: str, chunks) -> str:
    """Combine retrieved chunks into one context block and fill the prompt template."""
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def get_sources(chunks) -> list[dict]:
    """Build a de-duplicated list of {filename, page} from chunk metadata.

    Citation info always comes from metadata attached during ingestion —
    never from the LLM, which could otherwise invent a wrong filename/page.
    """
    seen = set()
    sources = []
    for chunk in chunks:
        filename = os.path.basename(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page")
        # PyPDFLoader pages are 0-indexed internally; show 1-indexed page
        # numbers since that's how humans actually count pages in a PDF.
        page_display = page + 1 if isinstance(page, int) else None

        key = (filename, page_display)
        if key not in seen:
            seen.add(key)
            sources.append({"filename": filename, "page": page_display})

    return sources


def generate_answer(question: str) -> dict:
    """Run the full query pipeline: retrieve -> filter weak matches -> rerank -> LLM.

    Returns {"answer": str, "sources": list[dict]} so the answer and its
    citations travel together as one structured result.
    """
    # Cast a WIDER net than before (RERANK_CANDIDATE_K, not RETRIEVAL_TOP_K)
    # so a genuinely relevant chunk has a real chance of surviving even when
    # other documents in the store are competing for the same top spots.
    results = retrieve_with_scores(question, k=RERANK_CANDIDATE_K)

    # Drop chunks that aren't actually similar enough to be useful — our
    # "weak evidence" check (Rule 16), now applied to that wider pool. This
    # also covers the "zero chunks" case for free: an empty results list
    # just produces an empty good_chunks.
    good_chunks = [
        chunk for chunk, score in results if score <= SIMILARITY_SCORE_THRESHOLD
    ]

    if not good_chunks:
        return {
            "answer": "I don't know based on the uploaded documents.",
            "sources": [],
        }

    # Narrow the wide pool back down to the true best few, by relevance
    # rather than raw embedding distance.
    top_chunks = rerank_chunks(question, good_chunks)

    prompt = build_prompt(question, top_chunks)
    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": get_sources(top_chunks),
    }
