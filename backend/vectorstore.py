import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from backend.config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY


def get_embedding_model():
    """Return the embedding model used for both storing chunks and embedding questions."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )


def get_vectorstore():
    """Open the persistent Chroma collection, creating it on disk on first use."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embedding_model(),
    )


def add_chunks_to_store(chunks):
    """Embed the given chunks and persist them to the Chroma collection."""
    store = get_vectorstore()
    store.add_documents(chunks)


def delete_document(source_path: str) -> int:
    """Remove every chunk that came from the given source file.

    Chroma has no "delete this document" operation — it only knows about
    chunks and their ids. So we first look up which ids carry this source,
    then delete those. Returns how many chunks were removed.
    """
    store = get_vectorstore()
    existing = store.get(where={"source": source_path})
    ids = existing["ids"]

    if ids:
        store.delete(ids=ids)

    return len(ids)


def list_documents() -> list[dict]:
    """Return every ingested document as {filename, chunks}, sorted by name.

    Built by counting the `source` metadata we attach during ingestion, since
    Chroma has no separate notion of "a document" — it only stores chunks.
    """
    store = get_vectorstore()
    # Pulls the whole collection into memory. Fine at this project's scale;
    # a large corpus would want a proper per-document index instead.
    data = store.get()

    counts = {}
    for metadata in data["metadatas"]:
        filename = os.path.basename(metadata.get("source", "unknown"))
        counts[filename] = counts.get(filename, 0) + 1

    return [
        {"filename": filename, "chunks": chunk_count}
        for filename, chunk_count in sorted(counts.items())
    ]
