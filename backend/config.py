import os
from dotenv import load_dotenv

# Load variables from .env into the process environment.
# Without this, os.getenv() below would not see anything from .env.
load_dotenv()

# --- Secrets ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Paths ---
# Computed relative to this file so the project works no matter where it's cloned.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")

# --- Chunking (Rule 10: initial guess, not a tuned value) ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# --- Retrieval ---
# How many chunks to pull back per question. Raised 4 -> 6 so the real
# match has a better chance of even being in the candidate list when the
# question is worded differently from the document's own text.
RETRIEVAL_TOP_K = 6

# Chroma gives every retrieved chunk a "distance" score — LOWER means MORE
# similar. Real tests showed the "relevant" and "irrelevant" score ranges
# overlap (a genuinely relevant resume chunk scored 1.3-1.66 depending on
# question phrasing, while unrelated chunks scored 1.39-1.7) — so no single
# number can cleanly separate them. This threshold is now just a loose
# safety net for the "nothing even remotely close was found" case; the
# real relevance judgment is left to the strict LLM prompt below, which
# actually reads the chunk instead of just measuring distance.
SIMILARITY_SCORE_THRESHOLD = 1.8
