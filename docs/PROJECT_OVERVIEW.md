# SmartDoc — Project Overview

A complete picture of what this project is, who it's for, what's actually been built, and how it works — written so someone with no prior context (including future-you) can read it once and understand the whole thing.

## What this is, in one paragraph

SmartDoc is a RAG (Retrieval-Augmented Generation) system. You upload PDF documents, ask questions about them in plain English, and it answers using **only** the content of those documents — never from the AI's general knowledge. Every answer comes with a citation (which file, which page) so you can verify it yourself. If the documents don't actually contain the answer, it honestly says "I don't know based on the uploaded documents." instead of guessing.

Built as a learning project following a deliberate, step-by-step plan (see [`PLANNING.md`](../PLANNING.md)) — every major decision has a documented reason, not just a working result.

## Who this is for / Use cases

SmartDoc isn't tied to one type of document — it works the same way for any PDF. Some concrete scenarios it's actually been tested against:

- **Company/internship FAQ assistant** — upload an HR or intern FAQ PDF, ask things like "What's the notice period if I leave early?" or "Is there a referral policy?", get a direct answer with the exact page it came from, instead of scrolling through a long document.
- **Resume Q&A / screening helper** — upload a candidate's resume and ask targeted questions ("What skills does this person have?", "What was their role at X?") rather than reading the whole document manually.
- **Studying a technical standard or spec** — upload something dense like an ISO standard and ask about specific clauses instead of searching through it page by page.
- **Multiple documents at once, with control** — upload several PDFs and either ask a question across all of them, or narrow it to just one specific file when you don't want other documents' content interfering with the answer.
- **Separate chats for separate topics** — each chat thread now has its own, completely private set of uploaded documents (see "Per-thread document isolation" below). Uploading a resume into one chat and an FAQ into another means neither can ever see the other's file.
- **An ongoing conversation, not just one-off questions** — ask a question, then a natural follow-up like "what about his education?" — the system figures out who "his" refers to from the conversation so far, instead of forcing every question to be fully self-contained.
- **Coming back later** — chats are saved, so you can close the app, come back tomorrow, and your previous conversations (and their answers/citations) are still there.

## Status snapshot

- **10 commits** pushed so far, `2026-08-13` → `2026-08-16`. A substantial amount of new work exists **locally, uncommitted**, from the most recent session (see below) — two full features built end-to-end and manually tested, but not yet pushed.
- Backend (FastAPI + PostgreSQL + ChromaDB) and frontend (Streamlit) are both functional and have been manually tested end-to-end, including multi-thread, multi-document, and follow-up-question scenarios.
- Well past the original "basic mandatory version" scope in `PLANNING.md` — reranking, document management, per-document search scoping, chat persistence, conversational memory, per-thread document isolation, and structured logging were all added afterward as deliberate extensions.
- The database and vector store were **deliberately wiped clean** partway through the most recent session (at the user's request, to get a truly fresh app) — any chats or documents from before that point no longer exist, by design, not by accident.

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
| 08-16 | `5433fd6` | Document management (list/delete-on-reupload) + chat-style multi-thread UI |
| 08-16 | `161f613` | PostgreSQL chat persistence + conversational memory (query rewriting) |

### Uncommitted as of this writing (not yet pushed)

Two complete features, both built step-by-step and manually verified, sitting locally uncommitted:

- **Per-thread document isolation** (`PLANNING.md` Section 13). Every chunk is now tagged with the `thread_id` of the chat it was uploaded into, uploads are saved to a per-thread folder (`data/documents/{thread_id}/{filename}`) instead of one shared folder, and every search is mandatorily scoped to the current thread's own documents. "New chat" now genuinely starts with zero documents until you upload into that specific chat. One accepted tradeoff: chunks ingested *before* this change have no `thread_id` and are now orphaned (unreachable) — accepted rather than writing migration/backfill code, since re-uploading is trivial.
- **Structured logging** (`PLANNING.md` Section 14). Every module now logs through Python's standard `logging` module (`backend/logging_config.py`), at `DEBUG` level by default (overridable via `LOG_LEVEL` in `.env`), writing to both the console and a rotating file at `data/logs/app.log`. Covers the full pipeline: question received → rewritten → candidates retrieved with scores → reranker's decision → final answer type, plus every API endpoint hit and every database write.
- **The previously-reported `/upload` `Internal Server Error`** (flagged as an open, unresolved bug at the end of the prior session — see [`ORACLES_TRIAL_READINESS.md`](ORACLES_TRIAL_READINESS.md)) **has not recurred** across extensive re-testing during this session's work (both features above involved uploading repeatedly through `/upload`). It was never formally root-caused, so treat this as "not currently reproducing" rather than "confirmed fixed" — the leading theory remains a stale `--reload` process rather than an actual code defect.
- **Filename sanitization fix** in `/upload` (`backend/main.py`) — closes a path-traversal security gap, manually verified to block the attack.
- **Reranker + citation correctness fixes** — the reranker now *drops* irrelevant candidates instead of just reordering them, and the API no longer returns citations alongside an "I don't know" answer.
- **A second frontend is still being prototyped in parallel**, in a separate session: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`, `frontend/serve.py` — a plain HTML/JS UI, alongside (not replacing) the existing Streamlit app (`frontend/app.py`). Still nothing decided about which one SmartDoc will actually ship with.
- `hello.py` at the project root is still an empty, unused file — safe to delete whenever.

## How it works

Two independent pipelines that meet at exactly one point — ChromaDB, which ingestion writes into and querying reads from (see `PLANNING.md` section 3 for why they're kept deliberately separate).

```
INGESTION — runs once, whenever a PDF is uploaded into a specific chat
  PDF → PyPDFLoader (extract text per page) → RecursiveCharacterTextSplitter
      (chunk_size=700, overlap=100) → OpenAI embeddings (text-embedding-3-small)
      → ChromaDB  [each chunk tagged with: source file path, page number,
                    chunk id, AND the thread_id it was uploaded into]

  Files are saved to data/documents/{thread_id}/{filename} — a per-thread
  folder, so two different chats can each have their own "resume.pdf"
  without colliding on disk or in the vector store.

  Uploading the same file again (within the same thread) first DELETES its
  old chunks, then re-inserts fresh ones — so a document never accumulates
  duplicate copies of itself.

QUERY — runs every time a question is asked
  Question
    → REWRITE: if this is a follow-up in an existing chat, ask the LLM to
      rewrite it as a standalone question using recent chat history
      (e.g. "what about his education?" → "What is the candidate's education?")
      — this step is skipped for the first question in a chat.
    → embed the (rewritten) question, same model as ingestion
    → similarity search: pull a WIDE pool of 15 candidate chunks, MANDATORILY
      scoped to the current chat's own thread_id, optionally narrowed
      further to specific files within that thread if the user picked some
    → drop any candidate whose distance score is above 1.8 — a loose safety
      net, not the real relevance judge
    → RERANK: hand the survivors to the LLM and ask it to pick out and order
      only the ones that actually help answer the question (it can select
      none at all if nothing is relevant) — keep the top 6
    → build a strict prompt (only the picked chunks + the question) → LLM
    → if the LLM says "I don't know...", return no citations at all
    → otherwise: answer + citations (filename/page pulled from chunk
      metadata — never generated by the LLM, so they can't be invented)

  Every step above now logs what it did (question, rewritten question,
  candidate scores, reranker's decision, final answer type) — see Logging.
```

System layout: Streamlit (frontend) and FastAPI (backend) are separate processes that only talk over HTTP. The frontend never touches ChromaDB, PostgreSQL, or the LLM directly — everything goes through the API.

## Feature tour (what's actually been built, in plain terms)

- **Turns a PDF into searchable knowledge.** Text is pulled out page by page, cut into overlapping chunks (so an answer near a chunk boundary doesn't get cut in half), and converted into embeddings — numeric representations that let the system search "by meaning" instead of exact keyword matching.

- **Won't make things up — two independent safety layers.** (1) A distance-score filter throws out chunks that are clearly unrelated before they're even considered. (2) A strict instruction to the LLM: answer *only* from the given text, and say "I don't know based on the uploaded documents." if it isn't there. The second layer is the one doing the real work — see Key Design Decisions for why.

- **Reranking — a second, smarter look at what was retrieved.** The first search step is fast but rough (it just measures mathematical distance). Before finalizing an answer, a wider set of candidates is handed directly to the LLM, which actually reads them against the question and picks the genuinely relevant ones — dropping any that don't help at all, not just shuffling their order. A real measured comparison of this against plain retrieval is documented in [`RERANKING_EXPERIMENT.md`](../RERANKING_EXPERIMENT.md).

- **Citations you can trust.** The filename and page number shown under every answer come from metadata attached when the document was first processed — never from the LLM guessing. And if the answer is "I don't know," no citations are shown at all.

- **Manage more than one document.** A sidebar lists every uploaded PDF (within the current chat) with its chunk count. Uploading the same file again replaces its old content instead of duplicating it.

- **Search just one document, or all of them (within a chat).** A selector lets you restrict a question to specific uploaded files belonging to the current chat.

- **Per-thread document isolation — each chat is its own sandbox.** Clicking "New chat" gives you a genuinely empty chat: no documents, an upload box front and center. Whatever you upload there is searchable *only* from that chat — another chat's questions can never retrieve it, even if the files happen to share the same name.

- **A real chat interface, not a single Q&A box.** Multiple separate conversation threads (like a chat app's sidebar), chat-bubble style messages, a "new chat" button, and threads are automatically named — after the first uploaded file if that happens first, or the first question otherwise.

- **Conversations survive a restart.** Chat threads and messages are stored in a PostgreSQL database, not just in the browser tab. Close the app, reopen it, and every past conversation and its answers are still there.

- **Follow-up questions actually work.** If you ask "What skills does the resume list?" and then "what about his education?", the system recognizes "his" refers to the person just discussed and rewrites the question accordingly before searching — without ever feeding that assumption into the answer itself as a "fact."

- **Upload safety.** Uploaded filenames are sanitized before being used to build a file path, closing off a path-traversal trick (a filename like `../../something.pdf`) that could otherwise write files outside the intended folder.

- **Structured logging.** Every meaningful step — a question arriving, a question being rewritten, retrieval scores, the reranker's decision, an endpoint being hit, a database write — is logged with a timestamp and the exact module it came from, visible live in the terminal and saved to a rotating log file. Makes it possible to see *why* the system answered the way it did without re-running a debug script by hand.

## Technical specifications

| | |
|---|---|
| Language / runtime | Python 3.14, virtualenv at `.venv/` |
| Backend framework | FastAPI + uvicorn |
| Frontend framework | Streamlit (a second, HTML/JS-based frontend is being prototyped separately — see Status snapshot) |
| Orchestration | LangChain (`langchain`, `langchain-openai`, `langchain-community`, `langchain-chroma`) |
| Vector store | ChromaDB, persisted at `data/chroma/` (one shared collection; every chunk is scoped to a chat via a `thread_id` metadata field, plus `source`/`page`/`chunk_id`) |
| Relational database | PostgreSQL, via `psycopg` — stores chat threads and messages |
| Embedding model | OpenAI `text-embedding-3-small` |
| LLM (answers, reranking, and question rewriting) | OpenAI `gpt-4o-mini`, temperature `0.7` for all three uses |
| PDF parsing | `pypdf` via `PyPDFLoader` (reads embedded text only — no OCR for scanned/image PDFs) |
| Chunking | `RecursiveCharacterTextSplitter`, size 700 / overlap 100 characters |
| Retrieval | top-15 candidates (mandatorily scoped to one thread) → distance filter (threshold 1.8) → LLM rerank (can shrink to fewer, even zero) → top-6 |
| Conversational memory | last 6 messages of a thread's history are shown to the LLM only to rewrite the current question, never used as answer content |
| File storage | uploaded PDFs saved to `data/documents/{thread_id}/{filename}` — per-thread folders, so identically-named files from different chats never collide |
| Logging | Python `logging`, level via `LOG_LEVEL` in `.env` (default `DEBUG`), console + rotating file at `data/logs/app.log` |
| Config | all thresholds/paths/model names centralized in `backend/config.py` |

### Module map

```
backend/
  config.py         all settings in one place (paths, models, thresholds); also
                     bootstraps logging on import
  logging_config.py setup_logging(): console + rotating file handler
  ingest.py         PDF -> text -> chunks -> vector store (delete-then-reinsert
                     on re-upload; tags every chunk with thread_id)
  vectorstore.py    Chroma access: add / delete / list documents (thread-scoped)
  database.py       PostgreSQL: chat threads and messages (create/list/read/write)
  llm.py            shared LLM client (used by rag.py, reranker.py, and rewriter.py)
  reranker.py       LLM-based candidate selection/reranking
  rewriter.py       rewrites a follow-up question into a standalone one, using chat history
  rag.py            rewrite -> retrieve (thread-scoped) -> threshold filter -> rerank -> prompt -> citations
  main.py           FastAPI app: /health, /upload, /documents, /threads,
                     /threads/{id}/messages, /query — all logged
frontend/
  app.py            Streamlit UI: multi-thread chat, thread-scoped upload/document
                     list/search selector
  index.html, app.js, styles.css, serve.py   a second, in-progress HTML/JS frontend (not yet integrated)
data/
  documents/{thread_id}/   uploaded PDFs, one folder per chat (gitignored)
  chroma/                  persistent vector DB (gitignored — regenerated at runtime)
  logs/                    rotating application log files (gitignored)
```

## Key design decisions (the "why", not just the "what")

- **One shared Chroma collection, scoped by metadata — not one collection per document or per thread.** Every chunk carries `source`, `page`, `chunk_id`, and now `thread_id`. This is intentional — it lets a question search across every document *in its own thread* at once, while a mandatory `thread_id` filter keeps chats fully isolated from each other, without the overhead of managing many separate collections.
- **Per-thread isolation reuses existing infrastructure rather than adding new tech.** The mandatory thread filter is just another Chroma metadata filter (the same mechanism built for per-document search scoping); per-thread file storage just adds one path segment. No new database, no new library.
- **A thread can now be created by either an upload or a question — whichever happens first.** Originally a thread only existed once the first question was asked. Since "New chat" needed to offer an upload box immediately, thread creation became a small shared helper (`ensure_thread()` in the frontend) callable from either place.
- **Old, un-tagged chunks are accepted as orphaned rather than migrated.** Once thread scoping became mandatory, chunks ingested before this feature (with no `thread_id`) can never be retrieved again. Written off as a one-time, easy-to-fix cost (just re-upload) rather than building backfill tooling for a project at this stage.
- **Logging is bootstrapped from `config.py`, not a separate call from `main.py`.** `config.py` is imported by every other module and already runs setup code (`load_dotenv()`) at import time — making it the natural single place to guarantee logging is configured before anything else runs, including ad-hoc test scripts that never touch `main.py` at all.
- **Reranking is LLM-based, not a separate cross-encoder model.** Simpler — it reuses the same `gpt-4o-mini` client already used for answers, no new model or dependency to install — at the cost of one extra LLM round-trip per question.
- **The similarity-score threshold is deliberately loose, on purpose.** Real testing showed relevant and irrelevant chunks' distance scores overlap significantly, so no single cutoff number can cleanly tell them apart. The threshold only exists to catch "nothing even remotely close was found"; the real relevance judgment is left to the LLM.
- **Conversational memory only rewrites the question — it never becomes "context" the LLM answers from.** Keeps the hallucination-protection guarantee airtight; the tradeoff is that purely conversational facts (e.g. "what's my name?") still correctly get "I don't know," since they were never in a document.
- **PostgreSQL over SQLite for chat storage**, chosen for the learning value of a real client/server database. The tradeoff: without a login system, every user of the same backend sees the same shared list of chat threads (though each thread's *documents* are now isolated — see per-thread isolation above).
- **Citations are withheld when the answer is "I don't know."** Showing a citation implies "here's the evidence," which would be misleading alongside a non-answer.

## Known limitations

- **Broad/summary questions** ("what is this document about?") often fail — no single chunk describes a whole document's topic. A limitation of chunk-based retrieval in general, not a bug specific to this project.
- **Scanned/image-based PDFs won't work** — text extraction only reads an embedded text layer; there's no OCR step.
- **The score threshold is a blunt instrument** by design — mitigated by the LLM-based reranking and strict prompt, not eliminated.
- **No login or per-user separation of chat threads.** Every chat thread's metadata (title, list membership) lives in one shared PostgreSQL database; anyone using the same backend sees the same thread list — even though each thread's *own documents* are now properly isolated from every other thread.
- **Conversational memory is intentionally limited.** Facts stated only in conversation (never in an uploaded document) are still correctly answered with "I don't know" — a deliberate hallucination-protection tradeoff, not an oversight.
- **Chunks ingested before per-thread isolation was added are orphaned** — they carry no `thread_id`, so no thread's search can ever retrieve them anymore. Accepted tradeoff, not a bug (see Key Design Decisions).
- **The `/upload` `Internal Server Error` reported at the end of the previous session has not recurred**, but was never formally root-caused — worth keeping an eye on rather than considering fully closed.

## Suggested next steps

- Commit and push the per-thread isolation and logging work — both are complete and manually tested but still local-only.
- Keep watching for the `/upload` error's recurrence now that logging is in place — if it happens again, the log file should finally show exactly what broke.
- Decide the fate of the second (HTML/JS) frontend prototype — merge, replace Streamlit, or discard — before it drifts further out of sync with the API.
- Update `README.md` — it still doesn't mention document management, per-document/per-thread search scoping, multi-thread chat, PostgreSQL persistence, conversational memory, or logging.
- Consider a per-user login system if the shared-thread-list limitation becomes a real problem.
- Delete the unused `hello.py`.

## Where to look for more

- [`README.md`](../README.md) — setup/run instructions, architecture, tech stack.
- [`PLANNING.md`](../PLANNING.md) — the original step-by-step plan, plus detailed implementation plans for reranking (Section 10), the UI overhaul (Section 11), PostgreSQL persistence (Section 12), per-thread document isolation (Section 13), and logging (Section 14).
- [`RERANKING_EXPERIMENT.md`](../RERANKING_EXPERIMENT.md) — real measured data comparing retrieval with vs. without LLM reranking.
- [`ORACLES_TRIAL_READINESS.md`](ORACLES_TRIAL_READINESS.md) — a mission-deliverables/mentor-Q&A readiness check written against an earlier state of the project (predates per-thread isolation and logging); useful context, but re-verify anything upload-related against the current code.
- `git log --oneline` — commit-by-commit build history.
