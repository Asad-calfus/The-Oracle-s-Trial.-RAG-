from backend.config import (
    LLM_TEMPERATURE,
    RERANK_CANDIDATE_K,
    RETRIEVAL_TOP_K,
    REWRITE_HISTORY_MESSAGES,
    SIMILARITY_SCORE_THRESHOLD,
)

# The 5 knobs a thread can override (Section 21). Keys here are the only
# settings resolve_settings() knows about — anything else in a thread's
# stored `settings` dict is ignored rather than silently trusted.
DEFAULT_SETTINGS = {
    "similarity_threshold": SIMILARITY_SCORE_THRESHOLD,
    "retrieval_top_k": RETRIEVAL_TOP_K,
    "rerank_candidate_k": RERANK_CANDIDATE_K,
    "rewrite_history_messages": REWRITE_HISTORY_MESSAGES,
    "llm_temperature": LLM_TEMPERATURE,
}


def resolve_settings(thread_settings: dict) -> dict:
    """Merge a thread's stored overrides on top of the config defaults.

    A thread with no overrides at all (thread_settings == {}) resolves to
    exactly today's hardcoded constants — nothing changes for a thread
    nobody has ever touched the settings panel on.
    """
    return {**DEFAULT_SETTINGS, **(thread_settings or {})}
