import logging

from backend.config import REWRITE_HISTORY_MESSAGES
from backend.llm import get_llm

logger = logging.getLogger(__name__)

# The LLM rewrites the QUESTION only — it is never shown the documents here,
# and never asked for facts. That keeps chat history out of the answer itself:
# history resolves what the question means, documents supply what it answers.
REWRITE_PROMPT_TEMPLATE = """Given the conversation below and a follow-up question, rewrite the follow-up as a standalone question that makes sense on its own.

Replace pronouns and vague references ("he", "she", "it", "that document") with what they actually refer to, based on the conversation.

If the question already makes sense on its own, return it completely unchanged.

Return ONLY the rewritten question — no explanation, no quotes, nothing else.

Conversation:
{history}

Follow-up question:
{question}

Standalone question:"""


def build_history_text(history) -> str:
    """Format recent messages as plain "role: text" lines for the prompt."""
    recent = history[-REWRITE_HISTORY_MESSAGES:]
    return "\n".join(f"{message['role']}: {message['content']}" for message in recent)


def rewrite_question(question: str, history) -> str:
    """Turn a follow-up question into one that stands on its own.

    Returns the question unchanged when there's no history to resolve against
    (the first question of a chat), which also avoids a pointless LLM call.
    Falls back to the original if the LLM returns nothing usable.
    """
    if not history:
        logger.debug("No history — skipping rewrite for: %r", question)
        return question

    prompt = REWRITE_PROMPT_TEMPLATE.format(
        history=build_history_text(history),
        question=question,
    )
    response = get_llm().invoke(prompt)
    rewritten = response.content.strip() or question

    if rewritten != question:
        logger.info("Rewrote question %r -> %r", question, rewritten)
    else:
        logger.debug("Question already standalone: %r", question)

    return rewritten
