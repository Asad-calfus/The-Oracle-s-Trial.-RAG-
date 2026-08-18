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


def init_db():
    """Create the chat tables if they don't already exist.

    threads is created first because messages references it — the foreign
    key can't point at a table that doesn't exist yet.
    """
    with get_connection() as conn:
        conn.execute(CREATE_THREADS_TABLE)
        conn.execute(CREATE_MESSAGES_TABLE)


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
    """Return every thread as {id, title}, newest first."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, title FROM threads ORDER BY created_at DESC"
        ).fetchall()


def add_message(
    thread_id: int,
    role: str,
    content: str,
    sources: Optional[list[dict]] = None,
) -> dict:
    """Save one message (a question or an answer) against a thread.

    Jsonb() tells psycopg to store the Python list in the JSONB column —
    a plain list would be ambiguous, so the wrapper is required. Reading it
    back converts it into a Python list again automatically.
    """
    with get_connection() as conn:
        message = conn.execute(
            """
            INSERT INTO messages (thread_id, role, content, sources)
            VALUES (%s, %s, %s, %s)
            RETURNING id, role, content, sources
            """,
            (thread_id, role, content, Jsonb(sources) if sources else None),
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
            SELECT role, content, sources
            FROM messages
            WHERE thread_id = %s
            ORDER BY created_at, id
            """,
            (thread_id,),
        ).fetchall()
