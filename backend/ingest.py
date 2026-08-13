import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.vectorstore import add_chunks_to_store


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


def ingest_pdf(file_path: str) -> int:
    """Run the full ingestion pipeline for one PDF: load -> chunk -> embed -> store.

    Returns the number of chunks created, so callers (e.g. the /upload
    endpoint) can report it back to the user.
    """
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    add_chunks_to_store(chunks)
    return len(chunks)
