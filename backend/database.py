import logging
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from backend.config import DATABASE_URL

logger = logging.getLogger(__name__)


def get_connection():
    """Open a PostgreSQL connection.

    Meant to be used with `with`, which commits when the block finishes and
    rolls back if it raises — so no call site has to remember to commit.

    row_factory=dict_row makes queries return {"column": value} dicts instead
    of positional tuples, so results can be read by name and handed straight
    back to FastAPI as JSON.
    """
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


# IF NOT EXISTS makes these safe to run over and over: they create the table
# the first time and quietly do nothing on every run after that.
CREATE_THREADS_TABLE = """
CREATE TABLE IF NOT EXISTS threads (
    id         SERIAL PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

# ON DELETE CASCADE: removing a thread removes its messages too, instead of
# leaving rows behind that point at a thread that no longer exists.
CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id         SERIAL PRIMARY KEY,
    thread_id  INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    sources    JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

# ADD COLUMN IF NOT EXISTS: added after messages already existed in earlier
# deployments — CREATE TABLE IF NOT EXISTS above only creates the column on
# a BRAND NEW table, so an existing "messages" table needs this to actually
# gain the column (Section 19.2's "model thinking" panel).
ADD_THINKING_COLUMN = """
ALTER TABLE messages ADD COLUMN IF NOT EXISTS thinking JSONB
"""

# One JSONB column for all 5 tunable knobs (Section 21) instead of 5
# separate columns — NULL/a missing key means "use the config default"
# for that particular setting; resolve_settings() (backend/settings.py)
# is what actually applies that fallback.
ADD_SETTINGS_COLUMN = """
ALTER TABLE threads ADD COLUMN IF NOT EXISTS settings JSONB
"""


def init_db():
    """Create the chat tables if they don't already exist.

    threads is created first because messages references it — the foreign
    key can't point at a table that doesn't exist yet.
    """
    with get_connection() as conn:
        conn.execute(CREATE_THREADS_TABLE)
        conn.execute(CREATE_MESSAGES_TABLE)
        conn.execute(ADD_THINKING_COLUMN)
        conn.execute(ADD_SETTINGS_COLUMN)


def create_thread(title: str) -> dict:
    """Insert a new chat thread and return it as {id, title}.

    RETURNING hands back the row Postgres just created — including the id it
    assigned — so we don't need a second query to find out what that id was.
    """
    with get_connection() as conn:
        thread = conn.execute(
            "INSERT INTO threads (title) VALUES (%s) RETURNING id, title",
            (title,),
        ).fetchone()
    logger.info("Created thread id=%s title=%r", thread["id"], thread["title"])
    return thread


def list_threads() -> list[dict]:
    """Return every thread as {id, title, settings}, newest first.

    settings comes back as {} rather than None for a thread that's never
    had one customized — callers can merge it straight into defaults
    without a None-check first.
    """
    with get_connection() as conn:
        threads = conn.execute(
            "SELECT id, title, settings FROM threads ORDER BY created_at DESC"
        ).fetchall()
    for thread in threads:
        thread["settings"] = thread["settings"] or {}
    return threads


def get_thread_settings(thread_id: int) -> dict:
    """Return one thread's setting overrides — {} if it has none.

    Only ever contains the keys the user has actually changed; anything
    missing is meant to fall back to a config default (resolve_settings()
    in backend/settings.py does that merge).
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT settings FROM threads WHERE id = %s", (thread_id,)
        ).fetchone()
    return (row["settings"] or {}) if row else {}


def update_thread_settings(thread_id: int, partial: dict) -> dict:
    """Merge new values into a thread's settings, keeping any others as-is.

    The `||` jsonb operator merges partial into whatever's already stored
    (or an empty object, via COALESCE, the first time) — the caller only
    ever needs to send the ONE key that changed, not all 5.
    """
    with get_connection() as conn:
        thread = conn.execute(
            """
            UPDATE threads
            SET settings = COALESCE(settings, '{}'::jsonb) || %s::jsonb
            WHERE id = %s
            RETURNING id, title, settings
            """,
            (Jsonb(partial), thread_id),
        ).fetchone()
    logger.info("Updated settings for thread_id=%s: %s", thread_id, partial)
    return thread


def reset_thread_settings(thread_id: int) -> dict:
    """Clear ALL of a thread's overrides — back to every config default.

    A full overwrite to '{}', not a merge like update_thread_settings() —
    resetting means forgetting every override, not keeping some of them.
    """
    with get_connection() as conn:
        thread = conn.execute(
            """
            UPDATE threads
            SET settings = '{}'::jsonb
            WHERE id = %s
            RETURNING id, title, settings
            """,
            (thread_id,),
        ).fetchone()
    logger.info("Reset settings for thread_id=%s", thread_id)
    return thread


def add_message(
    thread_id: int,
    role: str,
    content: str,
    sources: Optional[list[dict]] = None,
    thinking: Optional[dict] = None,
) -> dict:
    """Save one message (a question or an answer) against a thread.

    Jsonb() tells psycopg to store the Python list/dict in the JSONB
    column — plain Python values would be ambiguous, so the wrapper is
    required. Reading it back converts it into a Python object again
    automatically. thinking is only ever set on assistant messages (a
    user message has no retrieval pipeline behind it).
    """
    with get_connection() as conn:
        message = conn.execute(
            """
            INSERT INTO messages (thread_id, role, content, sources, thinking)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, role, content, sources, thinking
            """,
            (
                thread_id, role, content,
                Jsonb(sources) if sources else None,
                Jsonb(thinking) if thinking else None,
            ),
        ).fetchone()
    logger.debug("Saved %s message id=%s to thread_id=%s", role, message["id"], thread_id)
    return message


def get_messages(thread_id: int) -> list[dict]:
    """Return one thread's messages, oldest first.

    Ordered by id as well as created_at: two messages saved in the same
    instant would otherwise come back in an unpredictable order, and a
    question showing up after its own answer would be confusing.
    """
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT role, content, sources, thinking
            FROM messages
            WHERE thread_id = %s
            ORDER BY created_at, id
            """,
            (thread_id,),
        ).fetchall()
