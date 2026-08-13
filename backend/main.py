import os
import shutil

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from backend.config import DOCUMENTS_DIR
from backend.ingest import ingest_pdf
from backend.rag import generate_answer

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)

    # Save the uploaded PDF to disk before ingesting it.
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    num_chunks = ingest_pdf(file_path)

    return {
        "filename": file.filename,
        "chunks": num_chunks,
        "status": "success",
    }


@app.post("/query")
async def query(request: QueryRequest):
    return generate_answer(request.question)
