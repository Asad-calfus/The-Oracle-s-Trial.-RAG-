import base64
import io
import logging

import pdfplumber
from langchain_core.messages import HumanMessage

from backend.llm import get_llm

logger = logging.getLogger(__name__)


DESCRIBE_PROMPT = """You are transcribing a scanned document page for a search system.

Write out everything on this page as plain text:
- Transcribe all visible text verbatim, preserving reading order (headings, \
paragraphs, tables, labels, footnotes).
- For any non-text visual content (a chart, diagram, photo, stamp, signature, \
logo), add a short plain-language description of it in its place, e.g. \
"[Diagram: bar chart comparing X and Y]".

Do not summarize or skip anything. Do not add commentary about the task \
itself — output only the page's transcribed content."""


def is_image_page(file_path: str, page_number: int) -> bool:
    """
    Check whether a PDF page is image-only.

    A page is considered image-only when:
    1. It has no extractable text layer.
    2. It contains at least one displayed image.
    """
    with pdfplumber.open(file_path) as pdf:
        if page_number < 0 or page_number >= len(pdf.pages):
            raise IndexError(
                f"Page {page_number} is outside document range "
                f"(0-{len(pdf.pages) - 1})"
            )

        page = pdf.pages[page_number]

        # Check for actual text layer
        text = (page.extract_text() or "").strip()
        has_text = bool(text)

        # Check for images displayed on the page
        has_images = bool(page.images)

    is_image_only = not has_text and has_images

    logger.debug(
        "is_image_page(%s, page=%d): "
        "has_text=%s, has_images=%s -> %s",
        file_path,
        page_number,
        has_text,
        has_images,
        is_image_only,
    )

    return is_image_only


def page_has_image(file_path: str, page_number: int) -> bool:
    """Check whether a page contains any embedded image, regardless of text.

    Unlike is_image_page(), this doesn't care if the page also has a real
    text layer — it's used to catch "mixed" pages (real text plus a
    meaningful chart/diagram/photo) so that image content isn't silently
    dropped just because the page already has some extractable text.
    """
    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[page_number]
        has_images = bool(page.images)

    logger.debug("page_has_image(%s, page=%d): %s", file_path, page_number, has_images)
    return has_images


def extract_page_image(file_path: str, page_number: int, resolution: int = 144) -> bytes:
    """Render one PDF page to PNG bytes, ready to hand to a vision LLM.

    resolution=144 (2x pdfplumber's 72 DPI default) keeps small text
    (footnotes, stamps) legible to the vision model. If a transcription
    ever comes out garbled, raising this is the first fix to try, before
    touching the OCR prompt itself.
    """
    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[page_number]
        image = page.to_image(resolution=resolution).original

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    logger.debug(
        "extract_page_image(%s, page=%d): rendered %d bytes at resolution=%s",
        file_path, page_number, len(image_bytes), resolution,
    )
    return image_bytes


def describe_page_image(image_bytes: bytes) -> str:
    """Send a rendered page image to the vision LLM and return its transcription.

    Uses the same chat model as answers/reranking/rewriting (get_llm()) —
    gpt-4o-mini can read images directly, so no separate OCR model or
    account is needed.
    """
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": DESCRIBE_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ]
    )

    llm = get_llm()
    response = llm.invoke([message])
    description = response.content.strip()

    logger.debug("describe_page_image: got %d characters back", len(description))
    return description