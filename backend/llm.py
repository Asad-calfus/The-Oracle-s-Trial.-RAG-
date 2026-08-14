from langchain_openai import ChatOpenAI

from backend.config import LLM_MODEL, OPENAI_API_KEY


def get_llm():
    """Return the chat model used for both answer generation and reranking."""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
    )
