# SmartDoc — Document Q&A using RAG

## Planning Document (Basic Mandatory Version)

This document explains the plan **before any code is written**. No files other than this doc exist yet. Read this fully, then tell me "next" to start Step 1.

---

## 1. The Problem We Are Solving

Normal LLMs (ChatGPT, Claude, etc.) only know what they were trained on. They:

- Don't know about **your private documents** (company handbook, leave policy, internal PDFs).
- Can **hallucinate** — confidently make up answers that sound correct but aren't.
- Have no way to **cite a source** for what they say.

We want a system where a user can:

1. Upload their own PDF documents.
2. Ask a question in plain English.
3. Get an answer that is **grounded only in those documents**.
4. See exactly **which document and page** the answer came from.
5. Get an honest "I don't know" if the documents don't contain the answer.

This pattern is called **RAG — Retrieval-Augmented Generation**. Instead of asking the LLM to answer from memory, we first **retrieve** the most relevant pieces of text from our documents, and then ask the LLM to **generate** an answer using only that retrieved text.

Think of it like an open-book exam vs a closed-book exam. A plain LLM is closed-book (relies on memory, can guess wrong). RAG makes it open-book (must point to the page it read).

---

## 2. The Basic RAG Flow (Conceptual)

There are two separate "time zones" in a RAG system: things that happen **once when a document is uploaded** (ingestion), and things that happen **every time a question is asked** (querying).

```text
PDF
 ↓
Text Extraction        → pull raw text out of the PDF, page by page
 ↓
Chunking                → break long text into small overlapping pieces
 ↓
Embeddings               → convert each chunk into a vector of numbers (meaning as math)
 ↓
ChromaDB                 → store those vectors + original text + metadata, persisted to disk
 ↓
User Question
 ↓
Question Embedding       → convert the question into a vector the same way
 ↓
Similarity Search        → find chunks whose vectors are closest to the question's vector
 ↓
Relevant Chunks           → top-K matching chunks, each with filename + page
 ↓
LLM                      → answer using ONLY the text in those chunks
 ↓
Answer + Citation         → final answer + "Source: filename.pdf — Page X"
```

### Step-by-step explanation

**PDF (input)**

- What: raw uploaded file.
- Why: it's the source of truth we want the system to answer from.
- Input: a `.pdf` file.
- Output: nothing yet — just stored/received.

**Text Extraction**

- What: pull plain text out of the PDF, one page at a time.
- Why: PDFs are a binary/visual format; we need plain text before we can process language.
- Input: PDF file.
- Output: list of `(page_number, text)` pairs.

**Chunking**

- What: split each page's text into smaller overlapping pieces (e.g. ~700 characters each).
- Why: a whole PDF (or even a whole page) is often too large and too "mixed topic" to embed usefully. Smaller chunks let us retrieve *just* the relevant paragraph, not an entire document. (Explained in depth in Step 5 — Rule 10 requires this.)
- Input: extracted page text.
- Output: list of small text chunks, each tagged with metadata (filename, page, chunk id).

**Embeddings**

- What: convert each text chunk into a vector — a list of numbers that represents its *meaning*. Similar meaning → similar vector (mathematically close).
- Why: computers can't compare "meaning" of two sentences directly, but they can compare vectors using distance/similarity math. This is what lets us search "by meaning" instead of "by exact keyword".
- Input: text chunk.
- Output: a fixed-length numeric vector (e.g. 384 or 1536 numbers, depending on model).

**ChromaDB (Vector Database)**

- What: a database specialized for storing vectors + letting you search "find me the closest vectors to this one".
- Why: a normal SQL database is not built for "similarity search" over thousands of numeric dimensions. ChromaDB stores the vector, the original chunk text, and its metadata together, and persists it to disk so it survives a restart.
- Input: (vector, chunk text, metadata) triples.
- Output: a searchable, persistent store on disk.

**User Question**

- What: the user types a natural language question.
- Input: question string.
- Output: nothing yet — just received.

**Question Embedding**

- What: the same embedding model converts the question into a vector.
- Why: to compare the question against stored chunks, both must live in the same "vector space". Using a *different* embedding model here than during ingestion would make comparisons meaningless.
- Input: question string.
- Output: numeric vector.

**Similarity Search**

- What: ChromaDB compares the question's vector against all stored chunk vectors and returns the closest ones (commonly cosine similarity).
- Why: this is the "Retrieval" in Retrieval-Augmented Generation — we're finding which existing chunks are most likely to contain the answer.
- Input: question vector.
- Output: top-K chunk IDs ranked by similarity.

**Relevant Chunks**

- What: the actual text + metadata (filename, page) for the top-K matches.
- Why: this becomes the "evidence" we hand to the LLM, and later, the basis for citations.
- Input: chunk IDs from similarity search.
- Output: list of `{text, filename, page}`.

**LLM (Generation)**

- What: we build a prompt containing the question + the retrieved chunks, and ask the LLM to answer *using only that context*.
- Why: the LLM is good at reading and summarizing language, but we don't want it inventing facts — so we constrain it to the evidence we retrieved.
- Input: question + retrieved chunk texts.
- Output: a natural language answer.

**Answer + Citation**

- What: we attach the metadata (filename, page) of the chunks that were actually used, next to the answer.
- Why: citations must come from real metadata we tracked — never from the LLM guessing a filename/page, because it can hallucinate that too.
- Input: LLM answer + chunk metadata.
- Output: final response shown to the user, e.g.:
  ```text
  Employees can carry forward up to 10 days of annual leave.

  Sources:
  - employee_handbook.pdf — Page 14
  ```

---

## 3. Ingestion vs Querying — Two Separate Concepts

This is one of the most important mental models in RAG. Keep these two flows completely separate in your head (and later, in code).

### Ingestion (happens once, when a document is uploaded)

```text
PDF
 ↓
extract
 ↓
chunk
 ↓
embed
 ↓
store (ChromaDB)
```

This is a "write" pipeline. It's slower, runs occasionally (on upload), and its job is to prepare data for later searching.

### Querying (happens every time a question is asked)

```text
Question
 ↓
retrieve (search ChromaDB)
 ↓
context (top-K chunks)
 ↓
LLM
 ↓
answer
```

This is a "read" pipeline. It's fast, runs frequently (every question), and never modifies the vector store.

### Why separate them?

- Different triggers: ingestion runs on upload events; querying runs on question events.
- Different performance needs: ingestion can be slow (batch processing); querying must feel responsive.
- Different failure modes: a bad PDF breaks ingestion; a bad prompt breaks querying. Debugging is easier when they're isolated.
- Different code lifecycles: you might swap the LLM without touching ingestion, or change the chunking strategy without touching querying.

They meet at exactly one point: **ChromaDB**, which ingestion writes into and querying reads from.

---

## 4. Minimal Architecture

### Runtime architecture (query time)

```text
                Streamlit (UI)
                    │
                    │ user question
                    ▼
                 FastAPI (backend)
                    │
                    ▼
                RAG Pipeline
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
      ChromaDB              LLM
   (similarity search)   (answer generation)
          │
          ▼
    Relevant chunks
    (feed into LLM prompt)
```

- **Streamlit** is the UI the user interacts with (upload box, question box, answer display).
- **FastAPI** is the backend that exposes HTTP endpoints (`/upload`, `/query`) and coordinates the pipeline. Streamlit talks to FastAPI over HTTP — they are two separate processes.
- **RAG Pipeline** is our own Python logic that ties retrieval and generation together.
- **ChromaDB** stores and searches vectors.
- **LLM** generates the final answer from retrieved context.

### Ingestion architecture (upload time)

```text
PDF
 │
 ▼
PDF Loader          (extract text per page)
 │
 ▼
Text Splitter        (chunking)
 │
 ▼
Embeddings           (text → vectors)
 │
 ▼
ChromaDB             (persisted to data/chroma/)
```

### How the two flows connect

Both flows are triggered through FastAPI, but they use different endpoints and different backend modules:

```text
Streamlit "Upload PDF" button
        │
        ▼
   POST /upload  ──────────────►  ingest.py ──► vectorstore.py ──► ChromaDB (write)

Streamlit "Ask" button
        │
        ▼
   POST /query   ──────────────►  rag.py ──► vectorstore.py ──► ChromaDB (read)
                                      │
                                      ▼
                                     LLM
```

`vectorstore.py` is the shared module both flows depend on — it's the only place that talks directly to ChromaDB. `ingest.py` writes through it; `rag.py` reads through it.

---

## 5. Minimal Repository Structure

```text
smartdoc/
│
├── backend/
│   ├── main.py          # FastAPI app, defines endpoints
│   ├── config.py        # central configuration (paths, chunk size, model names)
│   ├── ingest.py        # PDF loading + chunking (ingestion pipeline)
│   ├── vectorstore.py   # ChromaDB read/write logic (shared by ingest + rag)
│   └── rag.py           # retrieval + LLM answer generation (query pipeline)
│
├── frontend/
│   └── app.py           # Streamlit UI
│
├── data/
│   ├── documents/       # uploaded PDFs saved here
│   └── chroma/          # persistent ChromaDB storage
│
├── .env                 # actual secrets (not committed)
├── .env.example         # template showing which env vars are needed
├── requirements.txt     # Python dependencies
└── README.md            # project overview + how to run it
```

This mirrors the standard structure you proposed — I'm not adding anything beyond it for the basic version.

---

## 6. What Every File Will Do

```text
File:
backend/config.py

Purpose:
Single place to store settings — Chroma path, chunk size, chunk overlap,
embedding model name, LLM model name, API keys (read from .env).

Why separate file:
Every other module needs these same settings. Without this file we'd
hardcode values in multiple places, and changing one setting (e.g. chunk
size) would mean hunting through several files.

Used by:
ingest.py, vectorstore.py, rag.py, main.py
```

```text
File:
backend/ingest.py

Purpose:
Loads a PDF, extracts text page-by-page, and splits it into chunks
with metadata (filename, page, chunk_id).

Why separate file:
Document ingestion (preparing data) is a distinct concern from
answering questions. Keeping it separate means we can test/debug
"did my PDF get chunked correctly?" without touching the LLM at all.

Used by:
main.py (the /upload endpoint calls this)
```

```text
File:
backend/vectorstore.py

Purpose:
The only module that talks directly to ChromaDB. Wraps "add chunks
to the store" and "search for similar chunks" behind simple functions.

Why separate file:
Both ingestion (writing) and querying (reading) need ChromaDB access.
Centralizing it means there's one place that knows how ChromaDB is
configured — if we ever swap vector DBs, only this file changes.

Used by:
ingest.py (writes), rag.py (reads)
```

```text
File:
backend/llm.py

Purpose:
get_llm() — the single place that builds the ChatOpenAI client (model
name, API key, temperature).

Why separate file:
Both rag.py (answer generation) and reranker.py (reranking) need the
exact same LLM client. Putting it in either of those two files would
force the other to import from it, and since both already need to be
importable by each other's caller, that risks a circular import.
This small shared file removes that problem entirely.

Used by:
rag.py, reranker.py
```

```text
File:
backend/reranker.py

Purpose:
rerank_chunks() — takes a wider pool of candidate chunks and asks the
LLM to reorder them by true relevance to the question, returning just
the best few.

Why separate file:
Reranking is its own distinct concern (refining retrieval results),
separate from generating the final answer — keeping it in its own
file makes both easier to read on their own.

Used by:
rag.py (generate_answer() calls this after the initial retrieval)
```

```text
File:
backend/rag.py

Purpose:
The query pipeline: takes a question, retrieves a wide pool of
candidate chunks via vectorstore.py, narrows it down via
reranker.py, builds a strict prompt, calls the LLM, and returns
an answer plus the source metadata used.

Why separate file:
This is the "thinking" part of the system, separate from data
preparation (ingest.py) and from the storage layer (vectorstore.py).

Used by:
main.py (the /query endpoint calls this)
```

```text
File:
backend/main.py

Purpose:
The FastAPI app. Defines HTTP endpoints (health check, upload, query)
and wires them to ingest.py / rag.py.

Why separate file:
This is the "front door" of the backend — the only file that knows
about HTTP concerns (requests, responses, status codes). It stays
thin; real logic lives in the other modules.

Used by:
frontend/app.py (Streamlit calls these endpoints over HTTP)
```

```text
File:
frontend/app.py

Purpose:
Streamlit UI — upload box for PDFs, text box for questions, displays
the answer and its sources.

Why separate file:
UI is a different layer entirely (presentation), and it's a separate
process from the backend. It only talks to FastAPI over HTTP — it
never imports backend modules directly.

Used by:
The end user, directly.
```

```text
File:
data/documents/

Purpose:
Where uploaded PDF files are physically saved.

Why separate folder:
Keeps raw inputs separate from generated data (embeddings), and
gives us a place to re-run ingestion later if needed.
```

```text
File:
data/chroma/

Purpose:
Where ChromaDB persists its vector index to disk.

Why separate folder:
This is the "database files" folder — you'll be able to show your
mentor this folder and say "this is where the embeddings live."
```

```text
File:
.env / .env.example

Purpose:
.env holds real secrets (API keys) and is never committed.
.env.example documents which variables are needed, with placeholder
values, so the project is reproducible.

Used by:
config.py reads .env at startup.
```

```text
File:
requirements.txt

Purpose:
Pinned list of Python packages needed (fastapi, langchain, chromadb,
streamlit, etc.) so the environment is reproducible.
```

```text
File:
README.md

Purpose:
Explains what the project is, how to set it up, and how to run it —
useful for your mentor and for future-you.
```

I'm not adding any files beyond this list for the basic version — no extra abstractions, no service layers, no test framework scaffolding yet. We'll only add things when a real need shows up.

---

## 7. Development Order

We'll follow this order, one step at a time, and I will stop after each one for you to say "next":

```text
STEP 0   Understand project architecture         ← this document
STEP 1   Create repository structure
STEP 2   Environment setup (requirements.txt, .env.example)
STEP 3   config.py
STEP 4   PDF loading
STEP 5   Chunking
STEP 6   Embedding model
STEP 7   Persistent ChromaDB
STEP 8   Document ingestion pipeline
STEP 9   Test ingestion
STEP 10  Retriever
STEP 11  Test retrieval without LLM
STEP 12  RAG prompt + LLM
STEP 13  Citation generation
STEP 14  Out-of-scope behaviour ("I don't know")
STEP 15  FastAPI endpoints
STEP 16  Test backend
STEP 17  Streamlit UI
STEP 18  Upload unseen PDF test
STEP 19  README + architecture explanation
```

Note steps 4–6 (PDF loading, chunking, embeddings) will actually live together inside `ingest.py`, built incrementally function-by-function — I won't create three separate files for them, just three separate small functions, explained one at a time.

---

## 8. What We Build First, and Why

**First real file: `backend/config.py`.**

Why config first, before even the folder skeleton is "done": every other module (ingest, vectorstore, rag, main) will import settings from it. Building it first means we never hardcode a chunk size or a file path somewhere and forget to fix it later. It also forces us to decide, upfront and explicitly:

- where Chroma will persist data (`data/chroma/`)
- what chunk size/overlap we're starting with (~700 / ~100, to be tuned later)
- which embedding model and LLM we're using

Before writing `config.py` itself, we'll first do **Step 1** (create the empty folder skeleton) and **Step 2** (requirements.txt + .env.example), since config.py needs `.env` to exist conceptually before it can read from it.

---

## 9. Future Improvements (Beyond the Basic Version)

The basic mandatory version (Steps 0-19 above) is complete: upload → chunk →
embed → retrieve → generate → cite, with two layers of hallucination
protection. Everything below is a **possible next phase** — none of it is
built yet, this is just a plan of what could be added and why.

### 9.1 Multi-user & Access Control

- **Authentication (login/signup)** — right now anyone who opens the app has
  full access; there's no concept of a "user" at all.
- **RBAC (role-based access control)** — e.g. an *admin* role that can upload/
  delete documents and see everything, vs a *regular user* role that can only
  ask questions. Needed once more than one person uses the same deployment.
- **Per-user document scoping** — each user's uploaded PDFs are only
  searchable by that user (or by whoever they explicitly share with), instead
  of one shared pool everyone's questions search across.
- **Query history** — save each user's past questions + answers + sources to
  a real database (not just Chroma), so they can revisit old answers instead
  of re-asking.
- **Upload audit log** — track who uploaded which file and when.

### 9.2 Retrieval Quality

- **Reranking** — after the initial top-K similarity search, run a dedicated
  reranker model (e.g. a cross-encoder or a hosted rerank API) over those
  candidates to re-score them by actual relevance. Embedding similarity alone
  is a rough signal (we saw this ourselves — score ranges overlapped between
  relevant and irrelevant chunks); a reranker reads the question and chunk
  together and is much better at judging true relevance.
- **Hybrid search (keyword + semantic)** — combine exact keyword search
  (BM25) with vector similarity search. Pure semantic search can miss exact
  matches (names, IDs, specific terms) that keyword search would catch
  instantly.
- **Query rewriting** — before retrieval, expand or rephrase a short/vague
  question into a few fuller variants and search with all of them. Directly
  addresses the "different phrasing gives worse results" issue we hit with
  the resume test.
- **Document-summary chunk** — generate and store one extra "summary" chunk
  per document at ingestion time, specifically so broad questions like "what
  is this document about" have something real to retrieve — this was a
  known gap in the basic version.
- **Document-scoped questions** — let the user pick "search only this PDF"
  instead of always searching across every uploaded document.

### 9.3 Document Parsing

- **Better PDF parsing** — swap/augment `PyPDFLoader` with a layout-aware
  parser (e.g. `unstructured`) that handles tables, multi-column layouts, and
  headers more accurately instead of flattening everything into one text
  stream.
- **OCR for scanned/image-based PDFs** — `PyPDFLoader` only reads embedded
  text; a scanned document (or a resume exported as an image) currently
  produces no usable text at all. An OCR step (e.g. Tesseract, or a
  vision-capable LLM) would fix this.
- **Other file types** — `.docx`, `.txt`, `.pptx`, images — currently PDF-only.
- **Semantic/structure-aware chunking** — split by section/heading instead of
  a fixed character count, so a chunk doesn't cut a table or a paragraph in
  half.

### 9.4 User Experience

- **Conversational memory** — let follow-up questions ("what about his
  education?" after asking about a resume) use the previous question as
  context, instead of every question being treated as brand new.
- **Show retrieved chunks / scores** — a transparency/debug view so the user
  can see exactly which chunks and scores were used for a given answer.
- **Highlight the exact answer snippet**, not just the filename/page.
- **Manage documents from the UI** — list/delete uploaded PDFs without
  touching the server's filesystem directly.
- **Upload progress indicator** for large PDFs.

### 9.5 Engineering & Reliability

- **De-dup on upload** — detect if an identical file was already ingested
  and skip/replace instead of adding duplicate chunks (a real issue we hit
  and worked around manually, not yet fixed at the code level).
- **Background ingestion** — large PDFs currently block the `/upload` request
  until fully processed; a task queue (e.g. Celery, or a simple background
  task) would make uploads feel instant.
- **Automated tests** — unit tests for chunking, retrieval, and citation
  logic, so a future change can't silently break something we already fixed.
- **An evaluation set** — a fixed list of test questions with known-correct
  answers/sources, run automatically to measure retrieval/answer quality
  over time, instead of the ad-hoc manual testing we've been doing so far.
- **Logging & monitoring** — track questions asked, latency, and errors.
- **Containerization (Docker)** — for easier/reproducible deployment.

### 9.6 Security

- **Sanitize uploaded filenames** — `/upload` currently builds the save path
  directly from the client-supplied filename; a crafted filename (e.g. with
  `../` in it) could write outside the intended folder. Worth validating/
  sanitizing before this ever runs somewhere untrusted.
- **File validation** — enforce a max upload size and check it's actually a
  PDF, not just trust the extension/content-type.
- **API rate limiting / cost control** — every question costs a real OpenAI
  API call; without limits, one user (or a bug/loop) could run up unexpected
  cost.

---

**Status:** the basic mandatory version described above (Sections 1-8) is
built and pushed to GitHub. Section 9 is an unbuilt wishlist. Section 10
below is the detailed implementation plan for the item we've decided to
build next: **reranking**.

---

## 10. Reranking — Implementation Plan (Next Version)

### 10.1 The Problem This Solves

Right now, retrieval is **one stage**: embed the question, compare it
against every stored chunk vector, take the top-K closest by distance. This
is fast, but approximate — an embedding compresses meaning into a fixed list
of numbers, so some nuance is lost. We saw this ourselves: the same resume
question, worded two different ways, produced very different scores for the
exact same correct chunk (1.3 vs 1.66).

### 10.2 The Concept: Two-Stage Retrieval

```text
Stage 1 (existing) — RECALL            Stage 2 (new) — PRECISION
"cast a wide net, cheaply"             "carefully judge the shortlist"

Question                                Wider candidate set (e.g. top 15)
   │                                          │
   ▼                                          ▼
Embed question                          For each candidate, judge its
   │                                    actual relevance to the exact
   ▼                                    question (slower, smarter check)
Compare to ALL stored vectors                │
   │                                          ▼
   ▼                                    Re-sort by this new relevance
Top ~15 closest chunks                  score, keep the real top-K
(rough, fast)                           (accurate, small)
```

Analogy: Stage 1 is a librarian quickly grabbing 15 books that look related
by skimming titles. Stage 2 is actually reading the first paragraph of each
of those 15 to pick the true best few. You can't afford to "read the first
paragraph" of every book in the library (too slow) — but you can afford it
for a shortlist of 15.

### 10.3 Tech Choice: LLM-Based Reranking

**Decision: reuse the existing `gpt-4o-mini` model (already wired up via
`ChatOpenAI` in `rag.py`) to do the reranking — no new library, no new
API key, no new account.**

How: after retrieving a wider candidate pool (e.g. top 15 instead of top 6),
send the question + all 15 candidate chunks to the LLM in a single prompt,
and ask it to return the most relevant ones in order. Only those get passed
to the existing answer-generation step — nothing downstream changes.

**Alternatives considered, and why not these (for now):**

- **Local cross-encoder model** (`sentence-transformers`, e.g.
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) — the standard production pattern,
  free per-query, but pulls in a heavy new dependency (PyTorch) and a model
  download. More setup than the basic version needs right now.
- **Cohere Rerank API** — a dedicated hosted reranking service, simple to
  call, but requires signing up for a completely separate account/API key
  just for this one feature.
- Both are reasonable *later* upgrades if the LLM-based approach turns out
  too slow/expensive/inaccurate — noted here so we don't forget the option.

**Tradeoff of the chosen approach:** one extra LLM call per question (for
the reranking step, on top of the existing answer-generation call), so
slightly slower and slightly more expensive per query — but zero new
infrastructure.

### 10.4 Where This Fits in the Existing Pipeline

```text
Before:
  retrieve_with_scores() [top 6]  →  score-threshold filter  →  LLM answer

After:
  retrieve_with_scores() [top 15, WIDER]  →  rerank_chunks() [NEW]  →  keep top 6  →  LLM answer
```

`rerank_chunks()` lives in its own file, `backend/reranker.py`, rather than
inside `rag.py` — this keeps the reranking logic (its own distinct concern)
readable on its own, the same way ingestion and vector-store logic already
each get their own file. This required pulling `get_llm()` out into a new
`backend/llm.py` too: both `rag.py` (answer generation) and
`reranker.py` (reranking) need the same LLM client, and having either file
import the other directly would create a circular import — `backend/llm.py`
is the shared dependency both of them import from instead.

```text
backend/llm.py        - get_llm() — shared by rag.py and reranker.py
backend/reranker.py   - RERANK_PROMPT_TEMPLATE, build_rerank_prompt(), rerank_chunks()
backend/rag.py         - retrieval, answer prompt, citations, generate_answer()
```

### 10.5 Step-by-Step Build Plan

Same small-step approach as the rest of this project — one piece at a time,
testing each layer independently before wiring it into `generate_answer()`.

```text
STEP R1   Add a new config value for the wider candidate pool
          (e.g. RERANK_CANDIDATE_K = 15), separate from RETRIEVAL_TOP_K.

STEP R2   Write the reranking prompt template: given the question and a
          numbered list of candidate chunks, ask the LLM to return the
          IDs of the most relevant ones, in order.

STEP R3   Write rerank_chunks(question, chunks) in rag.py: sends that
          prompt to the LLM, parses which chunk IDs it picked, and
          returns just those chunks in relevance order.

STEP R4   Test rerank_chunks() on its own — print what it picks for a
          few known questions, compare against the plain embedding
          ranking, before touching generate_answer() at all.

STEP R5   Wire it into generate_answer(): widen the initial retrieval to
          RERANK_CANDIDATE_K, pass the results through rerank_chunks(),
          then feed the result into the existing build_prompt() / LLM
          call exactly as before.

STEP R6   Re-run the same test questions we've already used (iso27001,
          FAQ, resume — including the differently-phrased resume
          question that struggled before) and compare answers/sources
          before vs after.
```

Nothing about `build_prompt()`, `get_sources()`, or the FastAPI/Streamlit
layers changes — reranking only affects *which* chunks reach the LLM, not
what happens after that.

### 10.6 Real-World Finding That Confirms Why This Matters

While testing Step R4 (`rerank_chunks()` standalone), we hit a live example
of the exact problem reranking is meant to fix:

- With **only the resume** in the store, "what is the name of the person in
  this resume?" gave a correct answer.
- After also uploading `Calfus_Intern_FAQ.pdf`, the **same kind of question**
  started returning "I don't know based on the uploaded documents." again.

Cause: `generate_answer()` still only pulls the top **6** chunks
(`RETRIEVAL_TOP_K`), and now those 6 slots are shared across every document
in the store. The FAQ has many chunks; when they happen to score closer to
the question than the resume's chunk does, they crowd the resume out of the
top 6 entirely — reranking can't rescue a chunk that was never even
retrieved. This is a general scaling problem: a fixed, small top-K gets
"thinner" per document as more documents are added.

This makes **Step R5 (not yet done)** more important, not optional — widening
the initial pull to `RERANK_CANDIDATE_K` (15) before reranking gives the
resume's chunk a much better chance of surviving into the candidate pool in
the first place.

**Also worth prioritizing sooner rather than later (already listed in
Section 9.2, "Document-scoped questions"):** let the user restrict a
question to one specific uploaded document (e.g. a "search only in:
resume.pdf" selector in the UI), using Chroma's metadata filter (it can
filter by the `source` field we already store). This removes cross-document
competition entirely rather than just widening the net — the more documents
get uploaded over time, the more this becomes the real fix rather than a
nice-to-have.
