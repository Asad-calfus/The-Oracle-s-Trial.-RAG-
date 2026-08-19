import logging
import os
from typing import Optional

from backend.config import DOCUMENTS_DIR, LLM_MODEL, RETRIEVAL_TOP_K
from backend.llm import get_llm
from backend.reranker import rerank_chunks
from backend.rewriter import rewrite_question
from backend.settings import resolve_settings
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

You may summarize, synthesize, compare, or reorganize information from the context to answer the question — you are not limited to quoting a single fact verbatim.

Only respond with "I don't know based on the uploaded documents." if the context contains nothing relevant to the question at all.

Do not use any outside knowledge. Do not invent facts not present in the context.

Context:
{context}

Question:
{question}

Answer:"""


def build_prompt(question: str, chunks) -> str:
    """Combine retrieved chunks into one context block and fill the prompt template."""
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


EXCERPT_LENGTH = 150


def build_excerpt(text: str, length: int = EXCERPT_LENGTH) -> str:
    """Shorten a chunk's text into a quotable excerpt for a citation.

    Collapses whitespace/newlines first so the excerpt reads as one clean
    line instead of breaking mid-sentence across the chunk's original
    line breaks.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= length:
        return collapsed
    return collapsed[:length].rstrip() + "..."


def get_sources(chunks) -> list[dict]:
    """Build a de-duplicated list of {filename, page, excerpt} from chunk metadata.

    Citation info always comes from metadata attached during ingestion —
    never from the LLM, which could otherwise invent a wrong filename/page.
    The excerpt is a short quote of the actual chunk that was used, so the
    user can jump straight to the relevant part of the page instead of
    reading it end to end (paragraph-level citation, Tier 1 — see
    PLANNING.md Section 16.1).
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
            sources.append({
                "filename": filename,
                "page": page_display,
                "excerpt": build_excerpt(chunk.page_content),
            })

    return sources


def build_thinking(
    question: str,
    search_question: str,
    results,
    good_chunks,
    settings: dict,
    top_chunks=None,
    thread_id: Optional[int] = None,
    filenames: Optional[list[str]] = None,
    usage=None,
) -> dict:
    """Package the pipeline's own intermediate state for the "model
    thinking" panel (PLANNING.md 19.2) — nothing here is newly computed
    beyond simple lookups; it's all data generate_answer()/
    generate_answer_stream() already has mid-function, just returned
    instead of discarded.

    settings is the ALREADY-RESOLVED dict (Section 21's resolve_settings())
    used for this specific answer — showing it here means "🧠 Show
    thinking" always reflects what actually ran, whether that was the
    config defaults or a thread's own overrides.

    rewritten_question is only included when it actually differs from the
    original — showing it unchanged would just be noise. usage is the
    final answer LLM call's token count (None for the two "declined to
    answer" cases below, since no answer-generation call happens there —
    rewrite/rerank calls are deliberately not counted here, to keep this
    number meaning "what the actual answer cost").
    """
    kept_chunks = top_chunks or []

    # Per-chunk breakdown: which specific chunks were retrieved, their exact
    # score, and what happened to each — not just aggregate counts.
    chunk_breakdown = []
    for chunk, score in results:
        filename = os.path.basename(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page")
        page_display = page + 1 if isinstance(page, int) else None

        if chunk in kept_chunks:
            status = "kept"
        elif chunk in good_chunks:
            status = "passed_threshold"
        else:
            status = "rejected"

        chunk_breakdown.append({
            "filename": filename,
            "page": page_display,
            "score": round(score, 3),
            "status": status,
        })

    return {
        "rewritten_question": search_question if search_question != question else None,
        "retrieved_count": len(results),
        "passed_threshold_count": len(good_chunks),
        "kept_after_rerank_count": len(kept_chunks),
        "chunk_breakdown": chunk_breakdown,
        "source_filter": {"thread_id": thread_id, "filenames": filenames or None},
        "model": LLM_MODEL,
        "settings_used": settings,
        "token_usage": dict(usage) if usage else None,
    }


def generate_answer(
    question: str,
    thread_id: Optional[int] = None,
    sources: Optional[list[str]] = None,
    history: Optional[list[dict]] = None,
    settings: Optional[dict] = None,
) -> dict:
    """Run the full query pipeline: rewrite -> retrieve -> filter -> rerank -> LLM.

    `thread_id` scopes the search to one chat's own uploaded documents —
    leaving it out searches every chunk in the store, regardless of thread
    (mainly useful for ad-hoc/standalone testing). `sources` optionally
    narrows further to specific filenames within that thread. `history` is
    the thread's earlier messages, used only to resolve what a follow-up
    question refers to. `settings` is a thread's raw overrides (or None) —
    resolve_settings() (Section 21) fills in config defaults for anything
    not overridden, so every call below always has a complete set of 5
    values regardless of what the caller passed.

    Returns {"answer", "sources", "rewritten_question"} so the answer and its
    citations travel together as one structured result.
    """
    resolved = resolve_settings(settings or {})
    logger.info(
        "Question for thread_id=%s: %r (sources=%s, settings=%s)",
        thread_id, question, sources, resolved,
    )

    # Resolve "he" / "it" / "that document" against earlier turns BEFORE
    # searching — a raw follow-up like "what about his education?" contains
    # no name to match on, so retrieval on it alone finds nothing useful.
    search_question = rewrite_question(
        question, history or [], history_limit=resolved["rewrite_history_messages"]
    )

    # Cast a WIDER net than before (rerank_candidate_k, not retrieval_top_k)
    # so a genuinely relevant chunk has a real chance of surviving even when
    # other documents in the store are competing for the same top spots.
    results = retrieve_with_scores(
        search_question,
        k=resolved["rerank_candidate_k"],
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
        chunk for chunk, score in results if score <= resolved["similarity_threshold"]
    ]

    if not good_chunks:
        logger.info("No candidates passed the score threshold — answering 'I don't know'")
        return {
            "answer": NO_ANSWER_RESPONSE,
            "sources": [],
            "rewritten_question": search_question,
            "thinking": build_thinking(
                question, search_question, results, good_chunks, resolved,
                thread_id=thread_id, filenames=sources,
            ),
        }

    # Narrow the wide pool back down to the true best few, by relevance
    # rather than raw embedding distance. Can come back empty if the reranker
    # judges that none of the candidates are actually relevant.
    top_chunks = rerank_chunks(search_question, good_chunks, top_k=resolved["retrieval_top_k"])

    if not top_chunks:
        logger.info("Reranker kept 0 chunks — answering 'I don't know'")
        return {
            "answer": NO_ANSWER_RESPONSE,
            "sources": [],
            "rewritten_question": search_question,
            "thinking": build_thinking(
                question, search_question, results, good_chunks, resolved, top_chunks,
                thread_id=thread_id, filenames=sources,
            ),
        }

    # Answer the REWRITTEN question, not the original: the raw follow-up
    # still says "his", and the LLM has no history here to work that out.
    prompt = build_prompt(search_question, top_chunks)
    llm = get_llm(temperature=resolved["llm_temperature"])
    response = llm.invoke(prompt)
    answer = response.content
    logger.info("Answered using %d chunks (declined=%s)", len(top_chunks), is_no_answer(answer))

    return {
        "answer": answer,
        # Citations mean "here is the evidence for this answer". When there is
        # no answer, showing them anyway implies support that doesn't exist.
        "sources": [] if is_no_answer(answer) else get_sources(top_chunks),
        "rewritten_question": search_question,
        "thinking": build_thinking(
            question, search_question, results, good_chunks, resolved, top_chunks,
            thread_id=thread_id, filenames=sources, usage=response.usage_metadata,
        ),
    }


def generate_answer_stream(
    question: str,
    thread_id: Optional[int] = None,
    sources: Optional[list[str]] = None,
    history: Optional[list[dict]] = None,
    settings: Optional[dict] = None,
):
    """Same pipeline as generate_answer(), but streams the final LLM call.

    Retrieval, filtering, and reranking all happen up front exactly as in
    generate_answer() — they're fast and don't need streaming. Only the
    answer-generation LLM call streams token by token. `settings` behaves
    exactly as in generate_answer() (Section 21).

    Yields ("token", text) for each piece of the answer as it arrives, then
    exactly one final ("done", {...}) with the same shape generate_answer()
    returns — sources can only be computed once the full answer is known
    (is_no_answer() needs the complete text), so that check happens after
    the loop instead of before the response starts.
    """
    resolved = resolve_settings(settings or {})
    logger.info(
        "Question for thread_id=%s: %r (sources=%s, settings=%s) [stream]",
        thread_id, question, sources, resolved,
    )

    search_question = rewrite_question(
        question, history or [], history_limit=resolved["rewrite_history_messages"]
    )

    results = retrieve_with_scores(
        search_question,
        k=resolved["rerank_candidate_k"],
        source_filter=build_source_filter(thread_id, sources),
    )
    good_chunks = [
        chunk for chunk, score in results if score <= resolved["similarity_threshold"]
    ]

    if not good_chunks:
        logger.info("No candidates passed the score threshold — answering 'I don't know' [stream]")
        yield ("token", NO_ANSWER_RESPONSE)
        yield ("done", {
            "answer": NO_ANSWER_RESPONSE,
            "sources": [],
            "rewritten_question": search_question,
            "thinking": build_thinking(
                question, search_question, results, good_chunks, resolved,
                thread_id=thread_id, filenames=sources,
            ),
        })
        return

    top_chunks = rerank_chunks(search_question, good_chunks, top_k=resolved["retrieval_top_k"])

    if not top_chunks:
        logger.info("Reranker kept 0 chunks — answering 'I don't know' [stream]")
        yield ("token", NO_ANSWER_RESPONSE)
        yield ("done", {
            "answer": NO_ANSWER_RESPONSE,
            "sources": [],
            "rewritten_question": search_question,
            "thinking": build_thinking(
                question, search_question, results, good_chunks, resolved, top_chunks,
                thread_id=thread_id, filenames=sources,
            ),
        })
        return

    prompt = build_prompt(search_question, top_chunks)
    llm = get_llm(temperature=resolved["llm_temperature"])

    full_answer = ""
    usage = None
    for piece in llm.stream(prompt):
        token = piece.content
        if token:
            full_answer += token
            yield ("token", token)
        if piece.usage_metadata:
            # OpenAI's streaming API reports usage only on (typically) the
            # final chunk — stream_usage=True (llm.py) is what makes this
            # populated at all instead of always being None.
            usage = piece.usage_metadata

    logger.info(
        "Answered using %d chunks (declined=%s) [stream]",
        len(top_chunks), is_no_answer(full_answer),
    )
    yield ("done", {
        "answer": full_answer,
        "sources": [] if is_no_answer(full_answer) else get_sources(top_chunks),
        "rewritten_question": search_question,
        "thinking": build_thinking(
            question, search_question, results, good_chunks, resolved, top_chunks,
            thread_id=thread_id, filenames=sources, usage=usage,
        ),
    })

