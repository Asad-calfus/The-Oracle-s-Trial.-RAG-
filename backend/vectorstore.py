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
