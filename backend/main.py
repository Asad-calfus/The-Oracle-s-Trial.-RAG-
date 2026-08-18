import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.config import DOCUMENTS_DIR
from backend.database import (
    add_message,
    create_thread,
    get_messages,
    init_db,
    list_threads,
)
from backend.ingest import ingest_pdf
from backend.rag import generate_answer
from backend.vectorstore import list_documents

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once when the server starts, before it accepts any request."""
    # Safe to call on every boot: the tables are only created the first time.
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


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

    return name


@app.get("/health")
def health():
    return {"status": "ok"}


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

    result = ingest_pdf(file_path, thread_id, include_images=include_images)

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

    result = generate_answer(
        request.question,
        thread_id=request.thread_id,
        sources=request.sources,
        history=history,
    )

    # Persisting is opt-in: a query without a thread_id still behaves exactly
    # as it did before this endpoint knew anything about chat history.
    if request.thread_id is not None:
        add_message(request.thread_id, "user", request.question)
        add_message(
            request.thread_id, "assistant", result["answer"], result["sources"]
        )

    return result
