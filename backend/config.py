import os
from dotenv import load_dotenv

from backend.logging_config import setup_logging

# Load variables from .env into the process environment.
# Without this, os.getenv() below would not see anything from .env.
load_dotenv()

# --- Secrets ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Database ---
# PostgreSQL connection string for chat thread persistence. Read from .env
# rather than hardcoded, since it differs per machine (and can contain a
# password, which must never end up in source control).
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Paths ---
# Computed relative to this file so the project works no matter where it's cloned.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")
LOGS_DIR = os.path.join(BASE_DIR, "data", "logs")
GRAPHS_DIR = os.path.join(BASE_DIR, "data", "graphs")

# --- Logging ---
# DEBUG by default so every pipeline step is visible; override in .env
# (e.g. LOG_LEVEL=INFO) to quiet it down without touching code.
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# Runs once, the first time ANY module in backend/ is imported — config.py
# is imported by everything else, making it the natural bootstrap point
# (it already does the same thing for load_dotenv() above).
setup_logging(LOGS_DIR, LOG_LEVEL)

# --- Chunking (Rule 10: initial guess, not a tuned value) ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
# 0 so the same question with the same retrieved context makes the same
# answer/decline decision every time (llm.py). Also surfaced in the "model
# thinking" panel (Section 19.2) so a technical user can see what was used.
LLM_TEMPERATURE = 0

# --- Retrieval ---
# How many chunks to pull back per question. Raised 4 -> 6 so the real
# match has a better chance of even being in the candidate list when the
# question is worded differently from the document's own text.
RETRIEVAL_TOP_K = 6

# How many chunks to fetch BEFORE reranking — a wider, rougher pool for the
# reranker to choose from. Must be >= RETRIEVAL_TOP_K, since reranking only
# narrows this pool down, never grows it.
RERANK_CANDIDATE_K = 15

# --- Conversational memory ---
# How many past chat messages the question-rewriter gets to see. Enough to
# resolve "he"/"it"/"that document" from recent turns, small enough to keep
# the extra LLM call cheap.
REWRITE_HISTORY_MESSAGES = 6

# Chroma gives every retrieved chunk a "distance" score — LOWER means MORE
# similar. Real tests showed the "relevant" and "irrelevant" score ranges
# overlap (a genuinely relevant resume chunk scored 1.3-1.66 depending on
# question phrasing, while unrelated chunks scored 1.39-1.7) — so no single
# number can cleanly separate them. This threshold is now just a loose
# safety net for the "nothing even remotely close was found" case; the
# real relevance judgment is left to the strict LLM prompt below, which
# actually reads the chunk instead of just measuring distance.
SIMILARITY_SCORE_THRESHOLD = 1.8

# --- Knowledge graph visualization (opt-in, Section 18) ---
# A single small Wikibooks-chapter-sized document (~13K characters) already
# took ~110 seconds and dozens of LLM calls to cognify() (see PLANNING.md
# Section 17.6) — this cap keeps a single graph generation in roughly that
# same ballpark, refusing anything that would run noticeably longer/costlier
# instead of silently doing it.
GRAPH_MAX_CHARS = 30_000
