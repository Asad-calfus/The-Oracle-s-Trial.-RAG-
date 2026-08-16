import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.vectorstore import add_chunks_to_store, delete_document


# loading the pdf using the pypdfloader lab
def load_pdf(file_path: str):
    """Load a PDF and return one Document per page (text + source/page metadata)."""
    # Normalize to an absolute path so the same file always gets the same
    # "source" metadata, whether it's called with a relative or absolute
    # path — otherwise the same PDF can end up stored twice under two
    # different source values.
    file_path = os.path.abspath(file_path)
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents):
    """Split page-level Documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # chunk_id gives every chunk a stable identifier, on top of the
    # source/page metadata it already inherited from the page it came from.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def ingest_pdf(file_path: str) -> dict:
    """Run the full ingestion pipeline for one PDF: load -> chunk -> embed -> store.

    Re-uploading a file REPLACES its existing chunks rather than adding a
    second copy beside them — otherwise the same document accumulates
    duplicates every time it's uploaded, inflating both the chunk count and
    the retrieval results.

    Returns {"chunks": int, "replaced": int} so the /upload endpoint can tell
    the user what actually happened.
    """
    file_path = os.path.abspath(file_path)

    replaced = delete_document(file_path)

    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    add_chunks_to_store(chunks)

    return {"chunks": len(chunks), "replaced": replaced}
