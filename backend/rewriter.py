import logging

from backend.config import REWRITE_HISTORY_MESSAGES
from backend.llm import get_llm

logger = logging.getLogger(__name__)

# The LLM rewrites the QUESTION only — it is never shown the documents here,
# and never asked for facts. That keeps chat history out of the answer itself:
# history resolves what the question means, documents supply what it answers.
REWRITE_PROMPT_TEMPLATE = """Given the conversation below and a new message, decide whether the message is a genuine follow-up question that relies on the conversation to make sense (for example, it uses a pronoun or vague reference like "he", "it", "that document", or omits context that was just discussed).

- If it IS a genuine follow-up, rewrite it as a standalone question, replacing pronouns/vague references with what they actually refer to.
- If it already makes sense on its own, OR it is gibberish, random characters, or unrelated to the conversation, return it EXACTLY UNCHANGED. Do not invent a new question from the conversation history — only resolve references that are actually present in the message.

Return ONLY the resulting question — no explanation, no quotes, nothing else.

Conversation:
{history}

New message:
{question}

Result:"""


def build_history_text(history, history_limit: int) -> str:
    """Format recent messages as plain "role: text" lines for the prompt."""
    recent = history[-history_limit:]
    return "\n".join(f"{message['role']}: {message['content']}" for message in recent)


def rewrite_question(question: str, history, history_limit: int = REWRITE_HISTORY_MESSAGES) -> str:
    """Turn a follow-up question into one that stands on its own.

    Returns the question unchanged when there's no history to resolve
    against (the first question of a chat) OR when history_limit <= 0 —
    the latter is how a thread's own setting (Section 21) turns
    conversational memory off entirely. This needs an explicit check:
    Python slicing treats history[-0:] as history[0:] (the WHOLE list,
    since -0 == 0), not an empty slice, so history_limit=0 would silently
    do the opposite of "disabled" without this guard.

    Falls back to the original if the LLM returns nothing usable.
    """
    if not history or history_limit <= 0:
        logger.debug("No history (or history_limit<=0) — skipping rewrite for: %r", question)
        return question

    prompt = REWRITE_PROMPT_TEMPLATE.format(
        history=build_history_text(history, history_limit),
        question=question,
    )
    response = get_llm().invoke(prompt)
    rewritten = response.content.strip() or question

    if rewritten != question:
        logger.info("Rewrote question %r -> %r", question, rewritten)
    else:
        logger.debug("Question already standalone: %r", question)

    return rewritten
