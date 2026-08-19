from typing import Optional

from langchain_openai import ChatOpenAI

from backend.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY


def get_llm(temperature: Optional[float] = None):
    """Return the chat model used for answer generation, reranking, and rewriting.

    temperature defaults to LLM_TEMPERATURE (0) when not given — reranker.py
    and rewriter.py both call get_llm() with no argument on purpose, so
    their pass/fail and rewrite-vs-leave-unchanged decisions stay
    deterministic regardless of what a user sets the ANSWER temperature to
    (Section 21) — only rag.py's final answer call passes a resolved,
    possibly user-overridden value.

    stream_usage=True: without it, a streamed response (llm.stream()) never
    reports token counts — only a plain invoke() would. Needed so the
    "model thinking" panel can show token usage on streamed answers too.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        stream_usage=True,
    )
