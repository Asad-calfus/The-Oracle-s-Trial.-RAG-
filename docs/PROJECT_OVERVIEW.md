# SmartDoc — Project Overview

One-page summary of what this project is, what's been built, and how it works — for anyone (including future-you) who needs the whole picture without reading every file.

## What this is

SmartDoc is a RAG (Retrieval-Augmented Generation) system: upload PDFs, ask questions in plain English, get answers grounded **only** in those documents, with a citation (filename + page) for every answer. If the documents don't contain the answer, it says so honestly instead of guessing.

Built as a learning project following a deliberate, step-by-step plan (see [`PLANNING.md`](../PLANNING.md)) — every major decision has a documented reason, not just a working result.

## Status snapshot

- **9 commits**, `2026-08-13` → `2026-08-16`, tracking cleanly with the plan's development order (scaffolding → ingestion → vector store → retrieval/citations → API → frontend → README → reranking → document management/UI overhaul).
- Both backend (FastAPI) and frontend (Streamlit) are functional and have been manually tested end-to-end.
- Currently well past the original "basic mandatory version" scope in `PLANNING.md` — reranking, multi-document management, and per-document search scoping were all added afterward.

| Date | Commit | What it added |
|---|---|---|
| 08-13 | `a801f0c` | Project planning doc and base scaffolding |
| 08-13 | `b8ef653` | Config and PDF ingestion pipeline |
| 08-13 | `38b7a51` | Chroma vector store integration |
| 08-13 | `2bf176b` | RAG retrieval, citations, hallucination protection |
| 08-13 | `d0b0650` | FastAPI backend endpoints |
| 08-13 | `7895ff9` | Streamlit frontend |
| 08-13 | `bac96a1` | README with architecture and known limitations |
| 08-14 | `facdc14` | LLM-based reranking added to retrieval pipeline |
| 08-16 | `5433fd6` | Document management (list/delete/replace) + chat-style multi-thread UI |

## Architecture

Two independent pipelines that meet at exactly one point — ChromaDB, which ingestion writes into and querying reads from (see `PLANNING.md` section 3 for why they're kept separate).

```
INGESTION (runs once, per uploaded PDF)
  PDF → PyPDFLoader (extract text/page) → RecursiveCharacterTextSplitter
      (chunk_size=700, overlap=100) → OpenAI embeddings (text-embedding-3-small)
      → ChromaDB  [metadata per chunk: source path, page, chunk_id]

  Re-uploading a file DELETES its old chunks first, then re-ingests — so the
  same PDF never accumulates duplicate chunks across repeated uploads.

QUERY (runs per user question, optionally scoped to chosen documents)
  Question → embed (same model) → similarity search
      → wide candidate pool (RERANK_CANDIDATE_K = 15, optionally filtered
        to selected `source` files)
      → drop chunks with distance > SIMILARITY_SCORE_THRESHOLD (1.8) —
        a loose safety net, not the primary relevance judge
      → LLM reranker: ask gpt-4o-mini to rank the survivors by true
        relevance, keep the top RETRIEVAL_TOP_K (6)
      → build strict prompt (context + question) → gpt-4o-mini (temp=0)
      → answer + citations (filename/page pulled from chunk metadata,
        never from the LLM itself)
```

System layout: Streamlit (frontend) and FastAPI (backend) are separate processes that only talk over HTTP. Streamlit never touches ChromaDB or the LLM directly.

## Technical specifications

| | |
|---|---|
| Language / runtime | Python 3.14, virtualenv at `.venv/` |
| Backend framework | FastAPI + uvicorn |
| Frontend framework | Streamlit |
| Orchestration | LangChain (`langchain`, `langchain-openai`, `langchain-community`, `langchain-chroma`) |
| Vector store | ChromaDB, persisted at `data/chroma/` (single collection, default name `"langchain"`) |
| Embedding model | OpenAI `text-embedding-3-small` |
| LLM (answers + reranking) | OpenAI `gpt-4o-mini` — temp `0` for answers, temp `0.7` for reranking |
| PDF parsing | `pypdf` via `PyPDFLoader` (text-layer only, no OCR) |
| Chunking | `RecursiveCharacterTextSplitter`, size 700 / overlap 100 characters |
| Retrieval | top-15 candidates → threshold filter (1.8) → LLM rerank → top-6 |
| Config | all thresholds/paths/model names centralized in `backend/config.py` |

### Module map

```
backend/
  config.py       all settings in one place (paths, models, thresholds)
  ingest.py       PDF -> text -> chunks -> vector store (delete-then-reinsert on re-upload)
  vectorstore.py  Chroma access: add / delete / list documents
  llm.py          shared LLM client (used by rag.py and reranker.py)
  reranker.py     LLM-based candidate reranking
  rag.py          retrieval + threshold filter + rerank + prompt + citations
  main.py         FastAPI app: /health, /upload, /documents, /query
frontend/
  app.py          Streamlit UI: multi-thread chat, upload, per-document search scoping
data/
  documents/      uploaded PDFs (gitignored — may contain personal data)
  chroma/         persistent vector DB (gitignored — regenerated at runtime)
```

## What's been built (by capability)

- **Ingestion**: PDF → page-level text → overlapping chunks → embeddings → ChromaDB, with `source`/`page`/`chunk_id` metadata on every chunk.
- **Hallucination protection, two independent layers**: a score-threshold filter before reranking, and a strict system prompt that requires the LLM to say `"I don't know based on the uploaded documents."` when the context doesn't answer the question. The prompt is the real safety net — the threshold alone can't cleanly separate relevant from irrelevant (see Limitations).
- **LLM-based reranking**: rather than trusting raw embedding-distance order, a wider candidate pool (15) is handed to the LLM directly, which is asked to rank passages by actual relevance to the question — see [`RERANKING_EXPERIMENT.md`](../RERANKING_EXPERIMENT.md) for a real measured comparison against no-rerank (methodology, latency cost, and when it does/doesn't change the final answer).
- **Citations**: filename + page always come from stored chunk metadata, never generated by the LLM — so a citation can't be hallucinated.
- **Document management**: `GET /documents` lists every ingested file with its chunk count (derived from `source` metadata, since Chroma itself has no native "document" concept); re-uploading a file replaces its old chunks instead of duplicating them.
- **Per-document search scoping**: the UI lets you restrict a question to specific uploaded files (via a Chroma metadata filter, `source: {$in: [...]}`) instead of always searching everything.
- **Multi-thread chat UI**: Streamlit sidebar supports multiple independent conversation threads (like a chat app), each with its own message history kept in `st.session_state`.

## Key design decisions (the "why", not just the "what")

- **One shared Chroma collection, not one per document.** Documents are distinguished purely by `source` metadata. This is intentional — it's what lets a question search across every uploaded document at once. Per-document isolation is available on demand via metadata filtering, not by physically separating storage.
- **Reranking is LLM-based, not a cross-encoder model.** Simpler (reuses the existing `gpt-4o-mini` client, no new model/dependency), at the cost of an extra LLM round-trip (~+1.2s/query measured — see the experiment doc).
- **The similarity-score threshold is deliberately loose.** Real testing showed relevant and irrelevant chunks' distance scores overlap (~1.3–1.7 in both cases), so no single cutoff cleanly separates them. The threshold only catches "nothing even remotely close was found"; the actual relevance judgment is left to the LLM (both in reranking and in the strict answer prompt).
- **Ingestion normalizes to an absolute file path before tagging `source`.** Otherwise the same PDF could be stored under two different `source` values (relative vs. absolute) and look like two different documents — this was a real bug found and fixed during testing (see Limitations note below).
- **Re-upload = delete + re-insert, not append.** Chroma has no upsert-by-source operation, so `ingest_pdf()` explicitly deletes any existing chunks for that path first.

## Known limitations

From `README.md`, still accurate except where noted:

- **Broad/summary questions** ("what is this document about?") often fail — no single chunk describes a whole document's topic. A limitation of chunk-based retrieval generally, not a bug.
- **Scanned/image-based PDFs won't work** — `PyPDFLoader` reads embedded text only, no OCR step exists.
- **The score threshold is a blunt instrument** (see design decisions above) — mitigated by the strict prompt, not eliminated.
- ~~Re-uploading the same PDF adds duplicate chunks~~ — **this has since been fixed** (delete-then-reinsert on re-upload, plus absolute-path normalization). `README.md`'s "Known limitations" section still lists this as open and should be updated to match.

## Suggested next steps

- Update `README.md`'s Known Limitations section to drop the now-fixed duplicate-chunk issue and mention document management / source-scoping / multi-thread chat, which the README currently doesn't describe at all.
- Per `PLANNING.md`'s original step list: Step 18 ("upload an unseen PDF, test end-to-end") and Step 19 (final README polish) are the remaining formal items, though the project has already grown beyond that plan's original scope.
- Consider whether the delete-on-reupload behavior should also expose an explicit `DELETE /documents/{filename}` endpoint, now that `vectorstore.delete_document()` already does the underlying work but is currently only called internally during re-ingestion.

## Where to look for more

- [`README.md`](../README.md) — setup/run instructions, architecture, tech stack.
- [`PLANNING.md`](../PLANNING.md) — the original step-by-step plan and reasoning behind the architecture.
- [`RERANKING_EXPERIMENT.md`](../RERANKING_EXPERIMENT.md) — real measured data comparing retrieval with vs. without LLM reranking.
- `git log --oneline` — commit-by-commit build history.
