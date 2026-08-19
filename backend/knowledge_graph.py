import logging
import os
import re
import time

from backend.config import GRAPH_MAX_CHARS, GRAPHS_DIR, LOG_LEVEL, LOGS_DIR, OPENAI_API_KEY
from backend.ingest import load_document
from backend.logging_config import setup_logging

# cognee reads its own env var name for the LLM key — reuse the project's
# existing key instead of requiring a second one to be configured.
os.environ.setdefault("LLM_API_KEY", OPENAI_API_KEY or "")

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph

# cognee configures its own logging on import, which replaces this
# project's handlers on the root logger — reassert ours immediately after,
# so the app's own log lines keep reaching app.log/the console.
setup_logging(LOGS_DIR, LOG_LEVEL)

logger = logging.getLogger(__name__)


class GraphTooLargeError(Exception):
    """Raised when a document exceeds GRAPH_MAX_CHARS — refused rather than
    silently running an expensive, slow cognify() job."""


def _safe_name(filename: str) -> str:
    """Turn a filename into a string safe to use as both a cognee dataset
    name and part of a file path — cognee dataset names in particular are
    picky about special characters.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^a-zA-Z0-9_]", "_", stem)


def dataset_name_for(thread_id: int, filename: str) -> str:
    """One cognee dataset per document, scoped to its thread.

    Isolating by dataset is what keeps two different documents' graphs
    (even across different chats) from being merged into one — without
    this, cognee would treat everything ever added as one big graph.
    """
    return f"thread_{thread_id}_{_safe_name(filename)}"


def graph_output_path(thread_id: int, filename: str) -> str:
    """Where this document's generated graph HTML lives on disk.

    Existence of this file IS the cache: build_graph() (Step G2) checks
    for it before doing anything expensive, so a document's graph is only
    ever generated once.
    """
    thread_dir = os.path.join(GRAPHS_DIR, str(thread_id))
    return os.path.join(thread_dir, f"{_safe_name(filename)}.html")


def _extract_text(file_path: str) -> str:
    """Extract a document's full text regardless of format.

    Reuses load_document() (Section 20's extension dispatcher) instead of
    assuming PDF — this used to call pdfplumber directly, which crashed
    with an unhandled exception on any non-PDF file (e.g. a .txt upload).
    Also means graph generation runs on the SAME PII-redacted text as
    retrieval does, not raw un-redacted content.
    """
    pages = load_document(file_path)
    return "\n\n".join(page.page_content for page in pages)


async def build_graph(file_path: str, thread_id: int, filename: str) -> str:
    """Generate (or reuse) a knowledge-graph visualization for one document.

    Cheap path first: if this document's graph was already generated, the
    file already exists and cognify() never runs again for it — cognify()
    is the expensive part (Section 17.6: ~110s and dozens of LLM calls for
    even a small document), so re-running it on every click would be a
    real, avoidable cost.

    Raises GraphTooLargeError instead of silently running an expensive job
    on a document whose extracted text exceeds GRAPH_MAX_CHARS.
    """
    output_path = graph_output_path(thread_id, filename)
    if os.path.exists(output_path):
        logger.info("Reusing cached graph for %s: %s", filename, output_path)
        return output_path

    text = _extract_text(file_path)
    if len(text) > GRAPH_MAX_CHARS:
        logger.info(
            "Refusing graph generation for %s: %d characters exceeds GRAPH_MAX_CHARS=%d",
            filename, len(text), GRAPH_MAX_CHARS,
        )
        raise GraphTooLargeError(
            f"Document has {len(text)} characters, which exceeds the "
            f"{GRAPH_MAX_CHARS}-character limit for graph visualization."
        )

    dataset_name = dataset_name_for(thread_id, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Building knowledge graph for %s (dataset=%s)", filename, dataset_name)
    start = time.time()

    await cognee.add(text, dataset_name=dataset_name)
    await cognee.cognify(datasets=[dataset_name])
    await visualize_graph(output_path, dataset=dataset_name)

    elapsed = time.time() - start
    logger.info("Knowledge graph for %s built in %.1fs -> %s", filename, elapsed, output_path)

    return output_path
