import os
import shutil
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from backend.config import DOCUMENTS_DIR
from backend.ingest import ingest_pdf
from backend.rag import generate_answer
from backend.vectorstore import list_documents

app = FastAPI()


class QueryRequest(BaseModel):
    question: str
    # Optional: restrict the search to these uploaded filenames.
    # Left out (or empty) means "search every document".
    sources: Optional[list[str]] = None


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


@app.post("/query")
async def query(request: QueryRequest):
    return generate_answer(request.question, request.sources)
