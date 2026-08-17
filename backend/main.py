import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)

    # Save the uploaded PDF to disk before ingesting it.
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_pdf(file_path)

    return {
        "filename": file.filename,
        "chunks": result["chunks"],
        "replaced": result["replaced"],
        "status": "success",
    }


@app.get("/documents")
async def documents():
    return list_documents()


@app.post("/threads")
async def new_thread(request: ThreadRequest):
    return create_thread(request.title)


@app.get("/threads")
async def threads():
    return list_threads()


@app.get("/threads/{thread_id}/messages")
async def thread_messages(thread_id: int):
    return get_messages(thread_id)


@app.post("/query")
async def query(request: QueryRequest):
    # Earlier turns only — this exchange is saved further down, after the
    # answer exists, so the question being asked can't leak into its own
    # history.
    history = get_messages(request.thread_id) if request.thread_id else [] # for rewriter 

    result = generate_answer(request.question, request.sources, history)

    # Persisting is opt-in: a query without a thread_id still behaves exactly
    # as it did before this endpoint knew anything about chat history.
    if request.thread_id is not None:
        add_message(request.thread_id, "user", request.question)
        add_message(
            request.thread_id, "assistant", result["answer"], result["sources"]
        )

    return result
