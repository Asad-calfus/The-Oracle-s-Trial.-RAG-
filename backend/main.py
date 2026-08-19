import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.config import DOCUMENTS_DIR
from backend.database import (
    add_message,
    create_thread,
    get_messages,
    get_thread_settings,
    init_db,
    list_threads,
    reset_thread_settings,
    update_thread_settings,
)
from backend.ingest import SUPPORTED_EXTENSIONS, ingest_document
from backend.knowledge_graph import GraphTooLargeError, build_graph
from backend.rag import generate_answer, generate_answer_stream
from backend.settings import DEFAULT_SETTINGS
from backend.vectorstore import list_documents

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once when the server starts, before it accepts any request."""
    # Safe to call on every boot: the tables are only created the first time.
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def log_unhandled_exceptions(request: Request, exc: Exception):
    """Log the real traceback for ANY unhandled exception before returning
    a generic 500.

    Without this, FastAPI/Starlette's own default handler deals with the
    exception using uvicorn's logging setup — a separate logger tree from
    this project's own (logging_config.py) — so it shows in the terminal
    but never reaches app.log. This is very likely why an earlier `/upload`
    500 was never root-caused: app.log genuinely had nothing to show.
    HTTPException is intentionally NOT caught here — those are deliberate,
    already-logged-if-relevant responses (like GraphTooLargeError's 400),
    not unexpected failures.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check the server logs."},
    )


class QueryRequest(BaseModel):
    question: str
    # Optional: restrict the search to these uploaded filenames.
    # Left out (or empty) means "search every document".
    sources: Optional[list[str]] = None
    # Optional: save this exchange into a chat thread.
    # Left out means answer normally but persist nothing.
    thread_id: Optional[int] = None


class ThreadRequest(BaseModel):
    title: str


class ThreadSettingsRequest(BaseModel):
    """A PARTIAL update — every field is optional, and only the ones the
    client actually sends get applied (Section 21.2's jsonb merge)."""
    similarity_threshold: Optional[float] = None
    retrieval_top_k: Optional[int] = None
    rerank_candidate_k: Optional[int] = None
    rewrite_history_messages: Optional[int] = None
    llm_temperature: Optional[float] = None


def safe_filename(filename: Optional[str]) -> str:
    """Reduce a client-supplied filename to a bare name, with no directory parts.

    The uploader controls this string completely, so a name like
    "../../secrets.pdf" would otherwise make os.path.join() write outside
    DOCUMENTS_DIR entirely. basename() drops every directory component and
    keeps only the last piece, so the file can only ever land where we intend.
    """
    name = os.path.basename(filename or "").strip()

    # basename(".."), basename("") and friends survive the strip above but
    # aren't usable filenames.
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # Reject unsupported types here, before a file is ever written to disk
    # or handed to ingest_document() — same "validate before trusting"
    # spirit as the checks above (Section 20.4/F4).
    extension = os.path.splitext(name)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {extension!r}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}.",
        )

    return name


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config/defaults")
async def config_defaults():
    """The config-level default for each of the 5 tunable settings
    (Section 21) — the frontend's single source of truth, instead of
    hardcoding the same numbers a second time in a different process.
    """
    return DEFAULT_SETTINGS


@app.patch("/threads/{thread_id}/settings")
async def patch_thread_settings(thread_id: int, request: ThreadSettingsRequest):
    partial = request.model_dump(exclude_none=True)
    logger.debug("PATCH /threads/%s/settings %s", thread_id, partial)
    return update_thread_settings(thread_id, partial)


@app.delete("/threads/{thread_id}/settings")
async def delete_thread_settings(thread_id: int):
    """Clear every override for this thread — back to all 5 config defaults."""
    logger.info("DELETE /threads/%s/settings (reset to defaults)", thread_id)
    return reset_thread_settings(thread_id)


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    thread_id: int = Form(...),
    include_images: bool = Form(False),
):
    logger.info(
        "POST /upload thread_id=%s filename=%r include_images=%s",
        thread_id, file.filename, include_images,
    )
    filename = safe_filename(file.filename)

    # Each thread gets its own subfolder, so two chats uploading a
    # same-named file (e.g. "resume.pdf") never collide on disk — and the
    # full path (unique per thread) becomes the chunk's `source` metadata.
    thread_dir = os.path.join(DOCUMENTS_DIR, str(thread_id))
    os.makedirs(thread_dir, exist_ok=True)
    file_path = os.path.join(thread_dir, filename)

    # Save the uploaded PDF to disk before ingesting it.
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_document(file_path, thread_id, include_images=include_images)

    return {
        # Return the sanitized name, not the raw one — this is the name the
        # file was actually saved under, and what /documents will list.
        "filename": filename,
        "chunks": result["chunks"],
        "replaced": result["replaced"],
        "status": "success",
    }


@app.get("/documents")
async def documents(thread_id: int):
    logger.debug("GET /documents thread_id=%s", thread_id)
    return list_documents(thread_id)


@app.post("/documents/graph")
async def document_graph(filename: str, thread_id: int):
    """Generate (or reuse) a knowledge-graph visualization for one document.

    filename/thread_id are query params (not a path segment) so filenames
    with spaces or special characters don't need manual URL-encoding.

    Opt-in and per-document by design (Section 18) — this does real LLM
    work the first time it's called for a given document, so it only runs
    when a user explicitly asks for it, never automatically on upload.
    Returns the raw HTML so the frontend can embed it inline.
    """
    logger.info("POST /documents/graph filename=%r thread_id=%s", filename, thread_id)

    file_path = os.path.join(DOCUMENTS_DIR, str(thread_id), filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        output_path = await build_graph(file_path, thread_id, filename)
    except GraphTooLargeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()

    return {"html": html}


@app.post("/threads")
async def new_thread(request: ThreadRequest):
    logger.debug("POST /threads title=%r", request.title)
    return create_thread(request.title)


@app.get("/threads")
async def threads():
    logger.debug("GET /threads")
    return list_threads()


@app.get("/threads/{thread_id}/messages")
async def thread_messages(thread_id: int):
    logger.debug("GET /threads/%s/messages", thread_id)
    return get_messages(thread_id)


@app.post("/query")
async def query(request: QueryRequest):
    logger.debug("POST /query thread_id=%s", request.thread_id)

    # Earlier turns only — this exchange is saved further down, after the
    # answer exists, so the question being asked can't leak into its own
    # history.
    history = get_messages(request.thread_id) if request.thread_id else [] # for rewriter
    # A thread's own overrides (Section 21) — {} for a thread that's never
    # touched the settings panel, which resolve_settings() (inside
    # generate_answer()) turns into the plain config defaults.
    settings = get_thread_settings(request.thread_id) if request.thread_id else {}

    result = generate_answer(
        request.question,
        thread_id=request.thread_id,
        sources=request.sources,
        history=history,
        settings=settings,
    )

    # Persisting is opt-in: a query without a thread_id still behaves exactly
    # as it did before this endpoint knew anything about chat history.
    if request.thread_id is not None:
        add_message(request.thread_id, "user", request.question)
        add_message(
            request.thread_id, "assistant", result["answer"], result["sources"],
            thinking=result.get("thinking"),
        )

    return result


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Same as /query, but the answer streams as it's generated.

    Each line of the response body is one JSON object: {"type": "token",
    "text": "..."} while the answer is being generated, then exactly one
    {"type": "done", "answer", "sources", "rewritten_question"} once the
    full answer (and therefore its citations) is known. Persisting to
    Postgres happens after the stream ends, same as /query does before
    returning — it just needs the complete answer text first.
    """
    logger.debug("POST /query/stream thread_id=%s", request.thread_id)

    history = get_messages(request.thread_id) if request.thread_id else []
    settings = get_thread_settings(request.thread_id) if request.thread_id else {}

    def event_stream():
        final = None
        for kind, payload in generate_answer_stream(
            request.question,
            thread_id=request.thread_id,
            sources=request.sources,
            history=history,
            settings=settings,
        ):
            if kind == "token":
                yield json.dumps({"type": "token", "text": payload}) + "\n"
            else:
                final = payload
                yield json.dumps({"type": "done", **payload}) + "\n"

        if request.thread_id is not None and final is not None:
            add_message(request.thread_id, "user", request.question)
            add_message(
                request.thread_id, "assistant", final["answer"], final["sources"],
                thinking=final.get("thinking"),
            )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
