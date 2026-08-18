import logging
import os
from typing import Optional

from backend.config import (
    DOCUMENTS_DIR,
    RERANK_CANDIDATE_K,
    RETRIEVAL_TOP_K,
    SIMILARITY_SCORE_THRESHOLD,
)
from backend.llm import get_llm
from backend.reranker import rerank_chunks
from backend.rewriter import rewrite_question
from backend.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def build_source_filter(
    thread_id: Optional[int],
    filenames: Optional[list[str]] = None,
) -> Optional[dict]:
    """Build a Chroma metadata filter scoping search to one thread's documents.

    thread_id is the mandatory scope — it's what keeps one chat's uploads
    invisible to every other chat's questions. filenames optionally narrows
    further, to specific files within that same thread (Phase U4's "search
    only in" selector). The two combine with $and when both are present.
    """
    conditions = []

    if thread_id is not None:
        conditions.append({"thread_id": thread_id})

        if filenames:
            # Uploads land in DOCUMENTS_DIR/{thread_id}/{filename} (Step T2),
            # so the full path is only reconstructible when thread_id is known.
            paths = [
                os.path.join(DOCUMENTS_DIR, str(thread_id), filename)
                for filename in filenames
            ]
            conditions.append({"source": {"$in": paths}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_documents(question: str):
    """Return the top-K most relevant chunks for a question via similarity search."""
    store = get_vectorstore()
    return store.similarity_search(question, k=RETRIEVAL_TOP_K)


def retrieve_with_scores(
    question: str,
    k: int = RETRIEVAL_TOP_K,
    source_filter: Optional[dict] = None,
):
    """Same as retrieve_documents, but also return each chunk's similarity score.

    source_filter, when given, restricts the search to specific documents
    instead of every chunk in the store.
    """
    store = get_vectorstore()
    return store.similarity_search_with_score(question, k=k, filter=source_filter)


NO_ANSWER_RESPONSE = "I don't know based on the uploaded documents."


def is_no_answer(answer: str) -> bool:
    """True when the LLM declined to answer.

    A prefix check rather than an exact match: the model reproduces the
    sentence closely but not always character-for-character.
    """
    return answer.strip().lower().startswith("i don't know")


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


def generate_answer(
    question: str,
    thread_id: Optional[int] = None,
    sources: Optional[list[str]] = None,
    history: Optional[list[dict]] = None,
) -> dict:
    """Run the full query pipeline: rewrite -> retrieve -> filter -> rerank -> LLM.

    `thread_id` scopes the search to one chat's own uploaded documents —
    leaving it out searches every chunk in the store, regardless of thread
    (mainly useful for ad-hoc/standalone testing). `sources` optionally
    narrows further to specific filenames within that thread. `history` is
    the thread's earlier messages, used only to resolve what a follow-up
    question refers to.

    Returns {"answer", "sources", "rewritten_question"} so the answer and its
    citations travel together as one structured result.
    """
    logger.info("Question for thread_id=%s: %r (sources=%s)", thread_id, question, sources)

    # Resolve "he" / "it" / "that document" against earlier turns BEFORE
    # searching — a raw follow-up like "what about his education?" contains
    # no name to match on, so retrieval on it alone finds nothing useful.
    search_question = rewrite_question(question, history or [])

    # Cast a WIDER net than before (RERANK_CANDIDATE_K, not RETRIEVAL_TOP_K)
    # so a genuinely relevant chunk has a real chance of surviving even when
    # other documents in the store are competing for the same top spots.
    results = retrieve_with_scores(
        search_question,
        k=RERANK_CANDIDATE_K,
        source_filter=build_source_filter(thread_id, sources),
    )
    logger.debug(
        "Retrieved %d candidates, scores=%s",
        len(results),
        [round(score, 3) for _, score in results],
    )

    # Drop chunks that aren't actually similar enough to be useful — our
    # "weak evidence" check (Rule 16), now applied to that wider pool. This
    # also covers the "zero chunks" case for free: an empty results list
    # just produces an empty good_chunks.
    good_chunks = [
        chunk for chunk, score in results if score <= SIMILARITY_SCORE_THRESHOLD
    ]

    if not good_chunks:
        logger.info("No candidates passed the score threshold — answering 'I don't know'")
        return {
            "answer": NO_ANSWER_RESPONSE,
            "sources": [],
            "rewritten_question": search_question,
        }

    # Narrow the wide pool back down to the true best few, by relevance
    # rather than raw embedding distance. Can come back empty if the reranker
    # judges that none of the candidates are actually relevant.
    top_chunks = rerank_chunks(search_question, good_chunks)

    if not top_chunks:
        logger.info("Reranker kept 0 chunks — answering 'I don't know'")
        return {
            "answer": NO_ANSWER_RESPONSE,
            "sources": [],
            "rewritten_question": search_question,
        }

    # Answer the REWRITTEN question, not the original: the raw follow-up
    # still says "his", and the LLM has no history here to work that out.
    prompt = build_prompt(search_question, top_chunks)
    llm = get_llm()
    response = llm.invoke(prompt)
    answer = response.content
    logger.info("Answered using %d chunks (declined=%s)", len(top_chunks), is_no_answer(answer))

    return {
        "answer": answer,
        # Citations mean "here is the evidence for this answer". When there is
        # no answer, showing them anyway implies support that doesn't exist.
        "sources": [] if is_no_answer(answer) else get_sources(top_chunks),
        "rewritten_question": search_question,
    }

