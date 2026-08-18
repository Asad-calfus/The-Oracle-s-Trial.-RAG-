import logging
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.ocr import describe_page_image, extract_page_image, is_image_page, page_has_image
from backend.vectorstore import add_chunks_to_store, delete_document

logger = logging.getLogger(__name__)


# loading the pdf using the pypdfloader lab
def load_pdf(file_path: str, include_images: bool = False):
    """Load a PDF and return one Document per page (text + source/page metadata).

    include_images is an explicit opt-in — vision LLM calls cost money, so
    they only run when the caller (the user, via the upload UI) asks for
    them. When False, pages load exactly as before this feature existed.

    Two cases are handled for image content, both via a vision-LLM
    description:
    - Fully-scanned page (no text layer, has an image) — page_content is
      REPLACED with the description, since there's nothing else there.
    - Mixed page (real text AND a meaningful image/chart) — the description
      is APPENDED to the existing text, so a chart's content becomes
      searchable without discarding the real text already extracted.

    Everything downstream (chunking, embedding, retrieval, citations) then
    works exactly the same, unaware any of the text came from OCR.
    """
    # Normalize to an absolute path so the same file always gets the same
    # "source" metadata, whether it's called with a relative or absolute
    # path — otherwise the same PDF can end up stored twice under two
    # different source values.
    file_path = os.path.abspath(file_path)
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    logger.debug("Loaded %s (%d pages)", file_path, len(pages))

    if not include_images:
        return pages

    for page in pages:
        page_number = page.metadata.get("page", 0)
        if not page_has_image(file_path, page_number):
            continue

        scanned = is_image_page(file_path, page_number)
        logger.info(
            "Page %d of %s has image content (scanned=%s), running vision description",
            page_number, file_path, scanned,
        )
        try:
            image_bytes = extract_page_image(file_path, page_number)
            description = describe_page_image(image_bytes)
            if scanned:
                page.page_content = description
            else:
                page.page_content = f"{page.page_content}\n\n[Image content]\n{description}"
            logger.info("Vision description succeeded for page %d of %s", page_number, file_path)
        except Exception:
            logger.exception("Vision description failed for page %d of %s", page_number, file_path)

    return pages


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

    logger.debug("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def ingest_pdf(file_path: str, thread_id: int, include_images: bool = False) -> dict:
    """Run the full ingestion pipeline for one PDF: load -> chunk -> embed -> store.

    Every chunk is tagged with thread_id, which is what makes a chat's
    documents searchable only from that same chat — see rag.py's mandatory
    thread filter, which relies on this tag being present.

    Re-uploading a file REPLACES its existing chunks rather than adding a
    second copy beside them — otherwise the same document accumulates
    duplicates every time it's uploaded, inflating both the chunk count and
    the retrieval results.

    Returns {"chunks": int, "replaced": int} so the /upload endpoint can tell
    the user what actually happened.
    """
    file_path = os.path.abspath(file_path)
    logger.info(
        "Ingesting %s for thread_id=%s (include_images=%s)",
        file_path, thread_id, include_images,
    )

    replaced = delete_document(file_path)
    if replaced:
        logger.debug("Replaced %d existing chunks before re-ingesting", replaced)

    documents = load_pdf(file_path, include_images=include_images)
    chunks = split_documents(documents)

    for chunk in chunks:
        chunk.metadata["thread_id"] = thread_id

    add_chunks_to_store(chunks)
    logger.info(
        "Ingested %s: %d chunks tagged thread_id=%s (%d replaced)",
        os.path.basename(file_path), len(chunks), thread_id, replaced,
    )

    return {"chunks": len(chunks), "replaced": replaced}
