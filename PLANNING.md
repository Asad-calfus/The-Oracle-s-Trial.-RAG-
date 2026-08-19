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

- ~~**Sanitize uploaded filenames**~~ — **DONE.** `/upload` used to build the
  save path straight from the client-supplied filename, so a crafted name
  (e.g. with `../` in it) could have written outside the intended folder.
  `safe_filename()` in `main.py` now reduces it to a bare basename and
  rejects empty/`.`/`..` names.
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

---

## 11. UI Overhaul — Implementation Plan

The current `frontend/app.py` is deliberately minimal: an upload box, a text
box, and one answer printed below. It works, but it has real gaps — you
can't see what's already uploaded, every question wipes the previous answer,
and there's no notion of a conversation at all.

### 11.1 The One Concept That Shapes This Whole Plan

**Streamlit re-runs the entire script from top to bottom on every single
interaction** — every button click, every text input, every file upload.
Normal Python variables are wiped and recreated each time.

The only thing that survives a rerun is **`st.session_state`** — a
dictionary Streamlit keeps alive for as long as that browser tab's session
lives. So anything the UI needs to *remember* (chat history, which thread is
open, which documents are selected) has to live in `st.session_state`.

Two consequences worth knowing upfront:

- **Good news for "different users without login":** `st.session_state` is
  already **per browser session**. Two people opening the app get completely
  separate `session_state` dictionaries automatically — no login needed for
  their *chats* to stay separate.
- **Honest limitation:** ChromaDB is **still one shared pool**. Person A's
  uploaded PDF is searchable by Person B's questions, because chunks aren't
  tagged with who uploaded them. Truly isolating documents per user requires
  either real login, or tagging each chunk with a session id and filtering
  on it — a bigger change, noted here rather than assumed away.
- **`session_state` dies on refresh.** Close the tab or hit reload, and the
  chat threads are gone. Making them survive needs real storage on the
  backend (covered in Phase U5 below).

### 11.2 What You Asked For

**A) Multiple chat threads** (like ChatGPT's sidebar — "New chat", switch
between past chats).

- Each thread = a title + a list of messages (question, answer, sources).
- Kept in `st.session_state` as a dict of threads plus "which one is active".
- Auto-name each thread from its first question so the sidebar is readable.

**B) A visible list of uploaded documents**, so the user knows what the
system is actually searching.

- This needs a **new backend endpoint** — the frontend currently has no way
  to ask "what's in the store?". A `GET /documents` endpoint would read the
  distinct `source` values out of Chroma's metadata (the same
  `store.get()` call we already used while debugging) and return each
  filename plus its chunk count.
- Worth pairing with the **duplicate-upload fix** (Section 9.5): the moment
  we render this list, any duplicate ingestion becomes visible to the user,
  so it stops being a hidden annoyance and starts being a visible bug.

### 11.3 Additional Suggestions

Ordered by how much value they add relative to effort:

1. **Chat-style message display** (`st.chat_message` / `st.chat_input`
   instead of a text box + `st.write`). This is the single biggest visual
   upgrade and it's mostly free — Streamlit has these built in. Threads
   (feature A) barely make sense without it.
2. **Document scoping selector** — checkboxes/multiselect over the document
   list: "search only in these PDFs". This is the *root-cause* fix for the
   crowding problem from Section 10.6, not just a UI nicety. Requires
   `/query` to accept an optional list of filenames and pass a Chroma
   metadata filter down into retrieval.
3. **Loading spinners** (`st.spinner`) during upload and query. Right now a
   large PDF upload or a slow query just looks frozen — and reranking made
   every query *slower* (two LLM calls instead of one), so this matters more
   now than it did before.
4. **Expandable "sources" section** showing the actual retrieved chunk text,
   not just filename + page. Turns citations from "trust me" into "here's
   the exact text I read" — and doubles as the debugging view we've been
   building by hand in the terminal all along.
5. **Delete a document from the UI** — needs a `DELETE /documents/{filename}`
   endpoint that removes those chunks from Chroma by metadata filter.
   Currently the only way to clean up is deleting `data/chroma/` and
   re-uploading everything.
6. **Better error surfaces** — the backend being down currently produces a
   raw `requests` exception trace in the UI. A clear "backend not reachable,
   is uvicorn running?" message would save real debugging time.

### 11.4 Phased Build Plan

Each phase is independently useful and testable — same one-step-at-a-time
approach as the rest of this project.

```text
PHASE U1  Chat-style layout (frontend only, no backend changes)
          Replace the text box with st.chat_input, render past
          question/answer pairs as chat bubbles from session_state.

PHASE U2  Multiple threads (frontend only)
          Sidebar: "New chat" button + a list of past threads.
          session_state holds {thread_id: {title, messages}} plus
          the active thread id. Auto-title from the first question.

PHASE U3  Document list (needs backend)
          New GET /documents endpoint reading distinct sources from
          Chroma metadata. Sidebar section listing each uploaded PDF
          and its chunk count. Pair with the dedup fix on /upload.

PHASE U4  Document scoping (needs backend + a small rag.py change)
          Multiselect over that document list; /query accepts an
          optional list of filenames; retrieval passes a Chroma
          metadata filter so only those documents are searched.

PHASE U5  Persistence (bigger — needs real storage)
          Move threads out of session_state into a database behind
          /threads endpoints, so chats survive a page refresh.
          Only worth doing once U1-U4 feel right.
          (Detailed plan in Section 12 — using PostgreSQL.)

PHASE U6  Conversational memory (rag.py change, not a UI change)
          Let follow-up questions ("what about his education?") use
          earlier turns as context — needs the question rewritten
          with history before retrieval, since the raw follow-up on
          its own retrieves nothing useful.
```

### 11.5 What Changes Where

```text
Phase   frontend/app.py   backend/main.py   backend/rag.py   new files
U1      yes               —                 —                —
U2      yes               —                 —                —
U3      yes               yes (/documents)  —                —
U4      yes               yes (/query)      yes (filter)     —
U5      yes               yes (/threads)    —                storage module
U6      —                 —                 yes              —
```

U1 and U2 touch **only** the frontend — the backend contract stays exactly
as it is today. That's deliberate: it means the biggest visible improvement
carries the least risk of breaking the RAG pipeline we've already tested.

---

## 12. Phase U5: Persistence with PostgreSQL — Implementation Plan

### 12.1 The Problem

Chat threads currently live in `st.session_state`, which exists only for as
long as a browser tab's session does. Refresh the page and every thread is
gone. To survive, they have to live **outside the frontend entirely** — on
the backend, in a real database.

### 12.2 Why PostgreSQL (and the honest tradeoff)

SQLite would be simpler here: it's just a file, no server to install or
keep running, and for a single-user local app it would be entirely
sufficient. PostgreSQL is a deliberate choice for the *learning* value —
it's a real client/server database, it's what production deployments
actually use, and it handles multiple concurrent users properly.

**The tradeoff nobody should skip over:** moving threads into a shared
database *removes* the accidental per-user isolation we currently get for
free.

```text
Today (session_state):   separate chats per browser, but lost on refresh
After Postgres:          chats survive forever, but EVERYONE sees the same list
```

`session_state` is per-browser-session, so two people opening the app never
see each other's chats. A shared Postgres database has no such boundary —
without login, there's nothing to tell one person's threads from another's.
Options, from cheapest to most correct:

1. **Accept it** — fine for a single-user project or a mentor demo.
2. **Add an `owner` column** plus a "who are you?" name box — a poor man's
   login. Honest about being unenforced, but keeps lists separate.
3. **Real authentication** (Section 9.1) — the actual fix, bigger scope.

We're taking option 1 for now, with the column design left compatible with
option 2 if we want it later.

### 12.3 Schema Design

Two tables, a classic one-to-many relationship:

```text
threads
  id          SERIAL PRIMARY KEY
  title       TEXT NOT NULL              -- auto-named from the first question
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()

messages
  id          SERIAL PRIMARY KEY
  thread_id   INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE
  role        TEXT NOT NULL              -- 'user' or 'assistant'
  content     TEXT NOT NULL
  sources     JSONB                      -- citations; assistant messages only
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
```

Design decisions worth stating:

- **`sources` as JSONB, not a third table.** Citations are a list of
  `{filename, page}` that only ever exist to be displayed under their own
  message — they're never queried across messages. A separate table would
  add a join for no benefit, and JSONB already matches the exact shape the
  API returns.
- **`ON DELETE CASCADE`** so deleting a thread automatically takes its
  messages with it, rather than leaving orphaned rows behind.
- **`created_at` on both tables** so threads can be listed newest-first and
  messages replayed in the order they were actually sent.
- **`SERIAL` ids instead of the frontend's counter.** The database assigns
  ids now, which means two browsers creating threads at the same time can't
  collide — something the old `next_thread_id` counter could not guarantee.

### 12.4 Step-by-Step Build Plan

```text
STEP P1   Create the database, add DATABASE_URL to .env / .env.example
          and config.py, add the psycopg driver to requirements.txt.
          No application logic yet — just plumbing.

STEP P2   backend/database.py — get_connection() plus init_db(), which
          creates both tables if they don't already exist.

STEP P3   Thread functions — create_thread(title) and list_threads().
          Test them standalone before any endpoint exists.

STEP P4   Message functions — add_message(...) and get_messages(thread_id).

STEP P5   FastAPI endpoints — POST /threads, GET /threads,
          GET /threads/{id}/messages, and /query gains an optional
          thread_id so it persists both messages itself.

STEP P6   Frontend — read and write threads through those endpoints
          instead of session_state.
```

### 12.5 What Does NOT Change

The entire RAG pipeline — `ingest.py`, `vectorstore.py`, `reranker.py`,
`rag.py` — is untouched by this phase. Persistence is purely about
remembering conversations; it has nothing to do with how answers are found
or generated. The one exception is `/query` gaining an optional `thread_id`
parameter, and even that leaves `generate_answer()` itself unchanged.

---

## 13. Per-Thread Document Isolation — Implementation Plan

### 13.1 The Problem

Today, ChromaDB is **one shared pool** — every uploaded PDF is searchable by
every chat thread. "New chat" only resets what's shown on screen; it doesn't
give you a clean slate to upload a fresh, unrelated set of documents into.
The ask: clicking "New chat" should start completely empty, with its own
upload box, and that chat should only ever search documents uploaded into
*it* — not documents uploaded into any other chat.

### 13.2 The Core Idea

Tag every chunk with **which thread it belongs to**, and make that tag a
**mandatory** filter on every search — on top of the existing optional
"search only these specific files" filter from Phase U4.

```text
Today:     search across ALL chunks in the store (optionally narrowed to
           specific files, if the user picked some)

After:     search across ALL chunks belonging to THIS THREAD (optionally
           narrowed further to specific files within that thread)
```

This reuses infrastructure that already exists — Chroma metadata filtering
(built for Phase U4) and Postgres threads (built for Phase P) — rather than
introducing anything new. The two filters combine with `$and` when both are
present.

### 13.3 Three Real Complications

1. **Filename collisions across chats.** Two different chats could each
   upload a file called `resume.pdf`. Today both would be saved to the same
   `data/documents/resume.pdf` and overwrite each other. Fix: save uploads to
   a **per-thread folder**, `data/documents/{thread_id}/{filename}`, so the
   full path — which becomes the chunk's `source` metadata — is naturally
   unique per thread. This also means `delete_document()` (used for
   re-upload dedup) automatically stays thread-safe with no code change:
   the path it matches on already can't collide across threads.
2. **A thread must exist before its first upload, not just its first
   question.** Right now a thread row is only created lazily when the first
   *question* is asked. But the ask here is "New chat → immediately show an
   upload box" — so uploading needs to be able to create the thread too,
   using the same lazy-creation idea, just triggered by whichever comes
   first: a question or an upload.
3. **Old, already-ingested chunks have no `thread_id` at all.** Once the
   filter becomes mandatory, chunks without a `thread_id` tag will never
   match any thread's search — they become orphaned. We're **accepting
   this** rather than writing migration/backfill code: it's a one-time,
   easy-to-fix cost (just re-upload the PDFs you want into whichever new
   chat should have them) that isn't worth the extra complexity of a
   backfill script for a project at this stage.

### 13.4 Step-by-Step Build Plan

```text
STEP T1   ingest_pdf(file_path, thread_id) tags every chunk's metadata
          with thread_id. Test standalone: ingest the same file under two
          different thread_ids, confirm two independently-tagged sets of
          chunks exist side by side.

STEP T2   /upload accepts thread_id as a form field, saves the file to
          data/documents/{thread_id}/{filename} (creating that folder if
          needed), and passes thread_id into ingest_pdf().

STEP T3   Mandatory thread scoping in retrieval: the source-filter builder
          in rag.py takes thread_id and combines it with any optional
          per-file filter via $and. generate_answer() takes thread_id
          through from /query, which already receives it today (currently
          only used for history — now also used to scope the search).

STEP T4   list_documents(thread_id) in vectorstore.py filters by
          thread_id too; /documents takes a thread_id query parameter.

STEP T5   Frontend: uploading when no thread is active creates one first
          (named after the uploaded file, same lazy-creation pattern
          already used for questions); the document list and "search only
          in" selector only render once a thread exists, showing an empty
          "upload a PDF to start this chat" state otherwise.

STEP T6   End-to-end test: two separate chats, each given a different set
          of uploaded PDFs, confirm neither chat's questions can retrieve
          the other's documents.
```

### 13.5 What Does NOT Change

Chunking, embedding, the reranker, the strict answer prompt, and citation
logic are all untouched — this phase only changes *which chunks are
eligible to be searched at all* for a given thread, not how retrieval,
reranking, or answering work once that eligible set is decided.

---

## 14. Logging — Implementation Plan

### 14.1 Why

Right now the only way to see what the pipeline is doing internally
(rewritten questions, retrieval scores, how many candidates the reranker
kept, which endpoint got hit) is to reproduce it by hand with a `python3 -c`
script — exactly what this whole project has been doing for debugging so
far. Logging makes that visible automatically, live in the terminal and
saved to a file for later.

### 14.2 Design

- **Python's built-in `logging` module** — no new dependency.
- **One setup function**, `backend/logging_config.py`, called once from
  `config.py` (which already runs setup code — `load_dotenv()` — at import
  time, so it's the natural place). This avoids a circular import:
  `logging_config.py` takes its settings as plain arguments rather than
  importing `config.py` itself.
- **Two handlers**: console (so the `uvicorn` terminal shows live activity)
  and a rotating log file at `data/logs/app.log` (capped size, a few backups
  kept — so history survives a restart without growing forever).
- **Every module gets its own logger** via `logging.getLogger(__name__)`, so
  each log line shows exactly which file it came from.
- **`LOG_LEVEL`** in `config.py`, read from `.env`, defaulting to `DEBUG` as
  asked. Third-party libraries (`httpx`, `openai`, etc.) are pinned to
  `WARNING` regardless, since they're extremely noisy at `DEBUG` and would
  drown out our own log lines.

### 14.3 What Gets Logged Where

```text
ingest.py       PDF loaded, page count, chunk count, thread_id tag applied
vectorstore.py  chunks added, chunks deleted (re-upload replace), documents listed
rewriter.py     original question, rewritten question (or "unchanged, no history")
rag.py          candidate count, scores, how many survived the threshold,
                how many the reranker kept, final answer type (real vs "I don't know")
reranker.py     the raw LLM reply and how many passage numbers were parsed from it
database.py     thread created, message saved
main.py         every endpoint hit, with its key parameters
```

### 14.4 Step-by-Step Build Plan

```text
STEP L1   backend/logging_config.py — setup_logging(logs_dir, level):
          console handler + rotating file handler, shared formatter,
          noisy third-party loggers turned down.

STEP L2   config.py — add LOG_LEVEL (default "DEBUG") and LOGS_DIR, call
          setup_logging() once at import time. Add data/logs/ to .gitignore.

STEP L3   Instrument ingest.py + vectorstore.py.

STEP L4   Instrument rag.py + reranker.py + rewriter.py.

STEP L5   Instrument main.py + database.py.

STEP L6   Test: run the app, watch the console live, then check
          data/logs/app.log to confirm the same activity was recorded.
```

---

## 15. OCR for Scanned/Image-Based PDFs — Implementation Plan

### 15.1 The Problem

`PyPDFLoader` only reads a PDF's **embedded text layer**. A scanned document
(or a resume exported as a flat image) has no such layer — every page comes
back with empty or near-empty text, so ingestion silently produces useless
chunks and the document is effectively invisible to every question.

### 15.2 The Core Idea (as actually built)

Detection is a deterministic check on the page itself, via `pdfplumber` —
not a guess from extracted text length. Two related but different
questions are asked per page:

```text
PDF page
  → pdfplumber opens the page and checks it directly:
        does extract_text() return real text?
        does the page actually contain embedded images (page.images)?
  → is_image_page: no text AND has images -> fully-scanned page
  → page_has_image: has images, regardless of text -> "mixed" page too

  → page_has_image is False → use PyPDFLoader's extracted text as-is,
    nothing else happens (unchanged, free, today's behavior)

  → page_has_image is True (this is where the opt-in feature runs):
        extract the page's image with pdfplumber
        → send it to a vision-capable LLM, ask for a DETAILED TEXT
          DESCRIPTION (verbatim text where present, plus a plain-language
          description of non-text visual content — charts, diagrams,
          stamps, tables-as-images, etc.)
        → if is_image_page (no original text) → REPLACE page_content
          with the description, since there was nothing else there
        → if it's a "mixed" page (real text + a meaningful image) →
          APPEND the description under the existing text, so a chart's
          content becomes searchable without discarding the real text
          already extracted

  → from here on, nothing changes: the (possibly extended) text is
    chunked with the existing splitter, embedded with the existing
    embedding model, stored in the existing Chroma vector DB, retrieved
    as normal text chunks, and cited by filename + page exactly like any
    other chunk
```

This covers two real cases found during testing, not just "fully scanned
documents": a resume exported as a flat image (fully-scanned case) AND a
normal PDF that has a chart/diagram sitting next to real paragraph text
(mixed case) — both were missing from the original single-case plan.

### 15.3 Tech Choices

**Vision LLM, not Tesseract** — use `gpt-4o-mini`'s vision capability
(already our configured LLM) to read the page image, instead of a local
OCR engine.
- **Reuses existing infrastructure**: the same OpenAI client already
  configured for embeddings/answers/reranking/rewriting now also handles
  this — no new account, no new API key, no new model to manage.
- **Tradeoff accepted**: costs one extra LLM call (with an image) per
  image-containing page. Tesseract would be free and fully local, but
  requires installing a separate system-level binary — extra setup
  friction this project is choosing to avoid, the same reasoning as
  choosing LLM-based reranking over a cross-encoder model (Section 10.3).

**`pdfplumber`, not `pymupdf`** — both were tried and compared side by
side on real project PDFs. Image counts disagreed between the two
libraries in both directions depending on the document (one library
found more on one PDF, the other found more on a different PDF), but the
higher-level "which pages have images at all" signal was close between
them either way. `pdfplumber` was kept as the single implementation to
avoid running two overlapping libraries for the same job — no functional
requirement forced this choice over the other.

**Opt-in checkbox, not automatic** — the vision-LLM call for images only
happens when the user explicitly checks "Describe images..." on upload
(`include_images` — plumbed through the `/upload` form field,
`ingest_pdf()`, and `load_pdf()`, default `False` end to end). A plain
PDF upload with the box unchecked costs and behaves exactly as it did
before this feature existed — no surprise LLM spend on every upload.

### 15.4 Detection Method

Two pdfplumber-backed checks in `backend/ocr.py`, both operating on the
same page object, no text-length guessing involved:

1. `is_image_page(file_path, page_number)` — `extract_text()` is empty
   AND `page.images` is non-empty → the page has nothing but an image.
2. `page_has_image(file_path, page_number)` — `page.images` is non-empty,
   regardless of text → used to also catch "mixed" pages.

A genuinely blank page has no text *and* no images, so both checks
correctly leave it alone.

### 15.5 Step-by-Step Build Plan (as actually built)

```text
STEP O1   backend/ocr.py — is_image_page(file_path, page_number) -> bool,
          using pdfplumber. DONE.

STEP O2   backend/ocr.py — extract_page_image(file_path, page_number) ->
          bytes, rendering the page via pdfplumber's to_image() at 144
          DPI. DONE.

STEP O3   backend/ocr.py — describe_page_image(image_bytes) -> str,
          sending the image to gpt-4o-mini for a detailed transcription +
          non-text-content description. DONE.

STEP O3.5 backend/ocr.py — page_has_image(file_path, page_number) -> bool,
          added after testing surfaced the "mixed page" case that the
          original single is_image_page() check couldn't catch. DONE.

STEP O4   Wired into load_pdf() (backend/ingest.py), gated behind a new
          include_images parameter (default False), threaded through
          ingest_pdf() and the /upload endpoint's include_images form
          field, with a frontend checkbox as the opt-in control. Replaces
          content for fully-scanned pages, appends for mixed pages. Logs
          which pages triggered it and whether it succeeded. DONE.

STEP O5   Real test files used instead of a purpose-built one:
          multimodal_rag_test.pdf (mixed text+image pages) and
          worldhealthstatistics_2022.pdf (image-heavy real-world PDF).
          DONE.

STEP O6   End-to-end UI test: upload with the checkbox checked, ask a
          question whose answer only exists in an image/chart, confirm a
          real answer with a correct filename+page citation. DONE.
```

### 15.6 What Does NOT Change

Chunking, embedding, retrieval, reranking, the answer prompt, and citations
are all untouched — this feature is isolated entirely inside `load_pdf()`.
Nothing downstream needs to know whether a page's text came from direct
extraction, from OCR, or from an appended image description.

### 15.7 Caveats Worth Knowing Going In

- **Per-page cost**: a document with many image-containing pages means one
  vision LLM call per such page when the checkbox is checked — noticeably
  slower/costlier than a plain-text ingest. Rough cost: well under a cent
  per page with `gpt-4o-mini`, but scales with page count.
- **Rendering resolution matters**: if a description comes out garbled for
  small fonts, the fix is usually increasing `extract_page_image()`'s
  `resolution` (currently 144 DPI), not the prompt.
- **Image-count detection is approximate across libraries**: `pdfplumber`
  and `pymupdf` don't always agree on exact image counts on the same PDF —
  harmless here, since the feature only cares about "does this page have
  at least one image," where both libraries agreed closely.
- **The detection heuristic is approximate**, not a guarantee — but a false
  positive (running OCR on a page that didn't need it) is harmless, just a
  wasted LLM call.

---

## 16. Paragraph Citations, Table Extraction, Streaming Answers — Implementation Plan

Three independent features, tackled in increasing order of complexity.
None of them touch each other's code, so they can be built and tested
one at a time without conflicts.

### 16.1 Paragraph-Level Citations

**Problem today**: a citation is only `{filename, page}` — accurate but
coarse. A page can be long; the user still has to scan the whole page to
find the actual sentence that answered their question.

**Two-tier approach**, cheapest first:

- **Tier 1 (recommended starting point) — quote the exact chunk text.**
  Every chunk retrieved for an answer already IS a small, specific span of
  the page (~700 characters, roughly a paragraph, because of
  `CHUNK_SIZE`). The chunk's own text is already sitting in memory at
  answer time — nothing needs to be computed or stored newly. Just include
  a short excerpt (e.g. first ~150 characters) of the chunk alongside
  `{filename, page}` in the citation. This turns "Page 4" into "Page 4 —
  '...average tenure across departments was...'" — the user can
  `Ctrl+F` straight to it. No new dependency, no ingestion change, no
  pipeline risk.

- **Tier 2 (optional stretch) — real paragraph numbers.** Requires
  detecting paragraph boundaries at ingestion time (e.g. via `pdfplumber`
  line-gap analysis: a bigger-than-normal vertical gap between lines marks
  a new paragraph), assigning each detected paragraph an index, and
  carrying that index through to whichever chunk(s) it ends up in as
  `paragraph_number` metadata. More accurate ("Page 4, Paragraph 2") but a
  real ingestion change with its own edge cases (paragraphs split across
  chunks, multi-column layouts confusing line-gap detection).

```text
STEP C1   backend/rag.py — extend get_sources() (or wherever citation
          dicts are built) to include a short excerpt from each cited
          chunk's page_content, alongside filename/page. DONE
          (build_excerpt(), EXCERPT_LENGTH=150, collapses whitespace,
          truncates with "...").

STEP C2   frontend/app.py — render_sources(): show the excerpt under each
          citation, e.g. as a smaller/italic caption line. DONE.

STEP C3   Test: ask a question, confirm the excerpt shown actually
          contains language relevant to the answer (spot-check a few).
          NEXT — do this in the running app.

--- Tier 2, only if C1-C3 feels insufficient ---

STEP C4   backend/ocr.py or a new backend/paragraphs.py — a
          split_into_paragraphs(file_path, page_number) helper using
          pdfplumber's line/word position data.

STEP C5   Wire into load_pdf()/split_documents() so each chunk inherits a
          paragraph_number in its metadata.

STEP C6   Update citations to show "Page X, Paragraph Y" when available.
```

**Recommendation**: build Tier 1 only for now — it solves the actual pain
(finding the answer on the page) with almost no engineering risk. Revisit
Tier 2 only if excerpts alone prove insufficient in practice.

### 16.2 Table Extraction

**Problem today**: `PyPDFLoader`'s flat text extraction reads a table
column-by-column or row-by-row in a jumbled order that loses the
row/column structure — a question like "what was Q3 revenue in the North
region?" becomes hard for the LLM to answer correctly even if the raw
table text is technically present in the chunk.

**Approach** — deliberately mirrors the OCR/image work already built
(Section 15), reusing the same architectural pattern and the same
`pdfplumber` dependency already in the project:

```text
For each page:
  → pdfplumber's page.extract_tables() detects any tables on the page
  → each detected table (a list of rows) is converted into a clean
    Markdown table (e.g. using "| a | b |" syntax) — Markdown tables are
    easy for an LLM to read correctly and cheap to generate, no vision
    LLM call needed (this is a structural extraction, not OCR)
  → the Markdown version is APPENDED to the page's existing text, labeled
    similarly to the image feature (e.g. "[Table content]\n<markdown>"),
    so the clean structured version is available for retrieval alongside
    whatever PyPDFLoader already extracted
```

No vision LLM cost here — `extract_tables()` is a pure parsing operation,
so this feature can safely run automatically on every upload rather than
needing an opt-in checkbox like image description does.

```text
STEP T1   backend/tables.py — extract_tables_as_markdown(file_path,
          page_number) -> list[str]: use pdfplumber's extract_tables(),
          convert each table's rows into a Markdown table string. DONE.

STEP T2   Wire into load_pdf(): for each page, if extract_tables_as_markdown
          returns anything, append it under the page's existing text
          (same append pattern as the image feature). Log how many tables
          were found per page. Runs unconditionally (not opt-in) since
          it's pure parsing, no LLM cost. DONE.

STEP T3   Test standalone against a real PDF containing a table — confirm
          the Markdown table reads correctly and preserves rows/columns.
          DONE (table_rag_test.pdf).

STEP T4   End-to-end test: upload a document with a table, ask a question
          whose answer requires reading a specific row/column, confirm
          the answer is correct (not just "present somewhere"). DONE.
```

**Complexity**: Low — same append pattern as Section 15, no new paid
dependency, no opt-in UI needed since it's free to run.

### 16.3 Streaming Answers

**Problem today**: `/query` waits for the entire answer to be generated
before responding — the user stares at a spinner for the full duration of
retrieval + reranking + the LLM's full response generation.

**Why this one is the most involved**: unlike the other two, this changes
the shape of an existing endpoint, not just what happens during ingestion.
Two things currently depend on having the COMPLETE answer text before
anything else can happen:
- `is_no_answer()` decides whether to show citations, and it needs the
  full answer text to check.
- `add_message()` persists the complete answer to Postgres.

**Approach**: retrieval, reranking, and question-rewriting still happen
up front exactly as today (they're fast and don't need streaming) — only
the FINAL answer-generation LLM call streams token-by-token. The backend
accumulates the full text as it streams past, and only after the stream
ends does it run `is_no_answer()` and persist the message — the same
logic as today, just deferred to the end of the stream instead of before
the response starts.

```text
STEP S1   backend/rag.py — add a streaming variant, e.g.
          generate_answer_stream(), that does the same retrieval/rerank/
          rewrite steps as generate_answer() but calls llm.stream(prompt)
          instead of llm.invoke(prompt), yielding each token as it
          arrives. DONE — yields ("token", text) then one final
          ("done", {answer, sources, rewritten_question}).

STEP S2   backend/main.py — new endpoint, e.g. POST /query/stream, using
          FastAPI's StreamingResponse: yields answer tokens as they're
          generated, then (after the stream ends) figures out
          is_no_answer()/sources and persists both messages via
          add_message() — same as /query does today, just after the loop
          instead of before returning. DONE — newline-delimited JSON
          (application/x-ndjson) response body.

STEP S3   frontend/app.py — replace the blocking requests.post() call for
          questions with a streamed request (requests.post(..., stream=
          True) or Streamlit's st.write_stream), updating the answer
          text incrementally as chunks arrive. Sources are rendered only
          once the stream finishes (mirrors backend timing). DONE.

STEP S4   Test: ask a question, confirm the answer visibly streams in
          instead of appearing all at once; confirm "I don't know"
          questions still correctly show zero citations; confirm the
          full exchange is still saved to Postgres and reappears
          correctly on a thread reload.
```

**Complexity**: Medium-High — the only one of the three that changes an
existing request/response shape rather than purely adding to ingestion.
Recommend building this last, after paragraph citations and table
extraction are done and stable.

### 16.4 Suggested Build Order

1. **Paragraph citations (Tier 1)** — smallest change, immediate value,
   almost no risk.
2. **Table extraction** — reuses the exact append pattern already proven
   in Section 15, free to run, moderate value for table-heavy documents.
3. **Streaming answers** — biggest UX win but the only one requiring a
   real endpoint/response-shape change; do it last once the simpler wins
   are banked.

---

## 17. Knowledge-Graph Retrieval Experiment — `cognee`

**This is explicitly an experiment/comparison, not a decided architecture
change.** The goal is to find out whether graph-based retrieval actually
helps on this project's real documents, before touching the production
pipeline at all.

### 17.1 What `cognee` Is and Why It's Worth Testing

`cognee` (open-source, `pip install cognee`) is a library that builds a
**knowledge graph** from documents: it runs each document through an LLM
to extract entities (people, orgs, dates, concepts) and the relationships
between them, then stores that graph so a question can be answered by
**traversing relationships**, not just by finding the single closest text
chunk.

Where this could help: today's pipeline (Chroma + vector similarity) treats
every chunk independently — it's strong at "find the paragraph that talks
about X" but weaker at multi-hop, relationship-style questions like *"which
policy references the certification reimbursement policy?"* or *"who is
mentioned in both the resume and the referral policy?"* — questions where
the answer isn't sitting in one chunk, it's a connection between two.

Where it likely won't help (worth being honest about going in): this
project's actual documents so far — policies, a resume, an ISO standard,
health statistics — are mostly fact-lookup documents, not
relationship-dense ones. The value of this experiment is finding out
whether that's really true, not assuming the answer either way.

### 17.2 How It Works, Architecturally

```text
cognee.add(document_text)      # ingest — no LLM call yet
cognee.cognify()               # THE EXPENSIVE STEP — LLM calls extract
                                # entities/relationships and build the graph
cognee.search(query, search_type=...)   # query the graph, cheap per call
```

- **Vector store**: LanceDB, embedded locally by default (a second vector
  store alongside the project's existing Chroma — not a replacement, a
  parallel system for this experiment).
- **Graph store**: NetworkX, also embedded locally by default — no Neo4j
  or other separate service required to get started.
- **LLM**: reuses the same `OPENAI_API_KEY` already configured for this
  project (`cognee` reads it from its own `LLM_API_KEY` env var).
- Search types include `GRAPH_COMPLETION` (natural-language answer using
  graph context), `RAG_COMPLETION` (closer to what this project already
  does), and `CHUNKS` (plain semantic chunk search) — worth trying more
  than one during the comparison.

### 17.3 The Real Concern: Citations

This project's single strongest safety property (Rule 16, and the whole
reason `get_sources()` exists) is that **every citation comes from
metadata attached at ingestion, never from the LLM** — so a citation can
never be wrong or invented. `cognee`'s `search()` returns an
LLM-synthesized answer from graph traversal; it does not naturally carry
the same "exact filename + page, guaranteed non-hallucinated" citation
guarantee this project has built everywhere else. **This has to be
checked directly in the experiment** — if `cognee`'s output can't be
traced back to a specific source with the same confidence, that's a real
regression, not just a UX detail, for a project whose stated goal is
grounded, citable answers.

### 17.4 Step-by-Step Experiment Plan

```text
STEP E1   pip install cognee. Set LLM_API_KEY (same value as
          OPENAI_API_KEY) in a scratch script — do NOT touch backend/
          config.py or any existing module yet. Feed the text of ONE
          existing test document through cognee.add() + cognee.cognify()
          in a standalone script. Time it and log how many LLM calls /
          roughly what it costs — cognify() is proportional to
          entity/relationship density, not just page count, so this
          number matters before testing at any scale.

STEP E2   Same standalone script — run cognee.search() with a
          relationship-style question (one that genuinely needs two
          documents/entities connected, not a plain fact lookup) using
          GRAPH_COMPLETION, then again using RAG_COMPLETION. Compare
          both answers to what the existing /query pipeline gives for
          the same question.

STEP E3   Repeat E2 for 4-5 questions total: a mix of plain fact-lookup
          questions (where the current pipeline already does well) and
          relationship-style questions (where it might not). For each,
          record: answer correctness, whether a traceable citation
          exists, and latency. This produces a real side-by-side table,
          not a guess.

STEP E4   Decision point based on E1-E3's actual numbers — not before:
            - If graph retrieval doesn't measurably beat the current
              pipeline on this project's real documents, stop here
              and document the finding (a real "we tried it and it
              didn't help enough to justify the complexity" result is
              still a useful, honest outcome).
            - If it does help on a specific question type, the next
              question is HOW to integrate: a separate opt-in
              "deep search" mode alongside the existing /query, not a
              replacement — the citation-integrity gap in 17.3 would
              need a real solution first, not a shortcut.
```

### 17.5 Caveats Worth Knowing Going In

- **`cognify()` is slow and costs multiple LLM calls per document** —
  budget real time and real API cost for E1 before running it on more
  than one document.
- **Two new embedded databases** (LanceDB + a NetworkX-backed graph
  store) would sit alongside the existing Chroma store if this ever went
  beyond the experiment stage — genuine added operational complexity.
- **Citation integrity (17.3) is the single biggest open question** —
  everything else in this project was built around "never let the LLM
  invent a source"; this experiment is the first time that guarantee is
  even in question, so it needs a direct answer, not an assumption.

### 17.6 Experiment Findings (E1-E3)

- **cognify() is genuinely expensive even on a small document**: a single
  Wikibooks-chapter-sized PDF took **109.9 seconds** and produced 56 raw
  nodes / 115 edges (consolidated to 14 entities / 15 relationships) — a
  double-digit number of LLM calls for a small document. Cost/time scales
  with content, so a large document (e.g. the 494k-character
  `worldhealthstatistics_2022.pdf`) would be substantially more expensive
  — this was deliberately NOT tested at that scale given E1's result.
- **Answer quality was solid** on a relationship-style question — both
  `GRAPH_COMPLETION` and `RAG_COMPLETION` gave accurate, coherent answers.
  No clear graph-specific advantage showed up on this single-document
  test, though the question wasn't genuinely multi-document/multi-hop.
- **Citation traceability — confirmed gap**: neither `GRAPH_COMPLETION`
  nor `RAG_COMPLETION` returns anything beyond a dataset-level identifier
  (no filename, no page). `CHUNKS` does return the actual matched raw
  text plus a `document_name`/`chunk_index`, but `document_name` was a
  generated hash (because the experiment fed extracted text, not the
  original file) and there is no page number in any mode. **Conclusion**:
  `cognee` cannot meet this project's citation-integrity bar (Rule 16) as
  a primary answer source.
- **Decision (per the user)**: do not pursue `cognee` as an answer/
  citation source at all — that requirement doesn't apply here. Instead,
  reuse the one genuinely good part of this experiment — cognee's
  built-in `visualize_graph()` — as a **pure visualization feature**: an
  optional panel showing a document's extracted entities/relationships as
  an interactive graph. See Section 18 for the implementation plan, which
  treats the cost finding above as a hard design constraint.

---

## 18. Knowledge Graph Visualization Panel — Implementation Plan

**Scope, explicitly**: this is a visualization side-feature only. It does
NOT touch retrieval, answers, or citations — `cognee` is disqualified from
that role by Section 17.6's finding. This panel exists purely so a user
can visually explore a document's extracted entities/relationships,
completely separate from asking it questions.

**Every design decision below exists to control cost**, given Section
17.6's measured finding (~110 seconds and dozens of LLM calls for even a
SMALL document):

1. **Strictly opt-in, per document, on demand** — never automatic on
   upload (unlike table extraction) and not even automatic alongside the
   existing `include_images` checkbox. A separate "Generate knowledge
   graph" button per document, clicked only when a user actually wants
   to see one.
2. **Generate once, cache forever** — the generated HTML is saved to disk
   (`data/graphs/{thread_id}/{filename}.html`). If that file already
   exists, clicking the button again just re-opens it — `cognify()`
   never re-runs for a document that already has one, unless the user
   explicitly asks to regenerate.
3. **A hard size cap with a clear message** — documents above a
   configurable character limit (e.g. `GRAPH_MAX_CHARS`) are refused with
   an explanation ("too large for graph visualization — costs scale with
   content"), rather than silently running an expensive job. This
   directly prevents the "494k-character document" scenario from ever
   reaching `cognify()` un-warned.
4. **An explicit cost/time warning in the UI** before the button is even
   clickable — the same pattern as the `include_images` checkbox's help
   text, but more prominent given the larger cost/time (minutes, not
   seconds).
5. **Isolated per document** — each document gets its own `cognee`
   dataset (`dataset_name` scoped to `thread_id` + filename), so
   different documents' graphs never mix, and re-processing one document
   never touches another's already-built graph.

### 18.1 Step-by-Step Build Plan

```text
STEP G1   backend/knowledge_graph.py — dataset_name_for(thread_id,
          filename) and graph_output_path(thread_id, filename) helpers
          (mirrors the per-thread folder pattern already used for
          uploads). Verify, with two small test documents, that scoping
          cognee.add()/cognify() by dataset_name actually keeps their
          graphs separate — this is the one piece of cognee's behavior
          not yet directly confirmed in this project's testing, so
          confirm it before building on top of it.

STEP G2   backend/knowledge_graph.py — build_graph(file_path, thread_id,
          filename) -> str (output path):
            - refuse (raise/return an error) if the document's extracted
              text exceeds GRAPH_MAX_CHARS
            - if graph_output_path() already exists, return it immediately
              — no re-run
            - otherwise: cognee.add(text, dataset_name=...),
              cognee.cognify(datasets=[dataset_name]),
              visualize_graph(output_path), return output_path
          Log start/end and elapsed time (this is exactly the kind of
          activity Section 14's logging work was built to capture).

STEP G3   backend/config.py — add GRAPH_MAX_CHARS (start conservative,
          e.g. 20,000-50,000 — small enough that a full run stays in the
          tens-of-seconds range based on the E1 timing data, not minutes).

STEP G4   backend/main.py — new endpoint, e.g. POST
          /documents/{filename}/graph?thread_id=..., calling build_graph()
          and returning the output path (or the HTML content directly).

STEP G5   frontend/app.py — in the "Documents in use" section, a
          "Generate knowledge graph" button per document, with cost/time
          warning text alongside it (mirroring the include_images
          checkbox's help text). On click, calls the endpoint and embeds
          the returned HTML inline via
          st.components.v1.html(html_content, height=600) — no separate
          file needs to be served or opened manually.

STEP G6   End-to-end test: click the button on a real (small) document,
          confirm the graph renders inline in the app; click it again and
          confirm it does NOT re-run cognify() (near-instant the second
          time); try a document over GRAPH_MAX_CHARS and confirm it's
          refused with a clear message instead of silently running.
```

### 18.2 What Does NOT Change

Nothing about `/query`, `/query/stream`, retrieval, reranking, or
citations changes. This is an entirely additive, opt-in panel — a user
who never clicks "Generate knowledge graph" sees zero difference and
incurs zero extra cost, exactly like `include_images`.

---

## 19. PII Redaction + "Model Thinking" Panel — Implementation Plan

Two independent features, motivated by the same real-world trigger: the
project is now being used with actual company-private documents, not just
test PDFs. Neither touches the other's code.

### 19.1 PII Redaction at Ingestion

**Decision (per the user): redact at ingestion, permanently** — PII is
detected and masked in a page's text BEFORE it's chunked or embedded, so
it is structurally incapable of appearing in a chunk, an embedding, an
answer, or a citation excerpt. This is stronger than masking only at
display time (which would still let PII sit in the vector store and pass
through the LLM's context on every query) — the tradeoff is that it's a
one-way door: once redacted at upload, the original value is gone from
what the system can retrieve or answer with.

**The real tension to be upfront about**: this project's own test
history includes a legitimate use case built around a PERSON'S name and
role — *"What was Ajinkya's role at Airports Authority of India?"*
(`RERANKING_EXPERIMENT.md`). If redaction is too aggressive (e.g., it
masks every proper name), that exact kind of resume/profile lookup stops
working — which is core functionality, not an edge case, for a
document-Q&A tool. So redaction has to target things that are almost
NEVER the actual answer to a legitimate business question, not anything
that looks personally identifiable.

**Two-tier approach**, cheapest and least risky first:

- **Tier 1 — regex-based, structured PII only.** Emails, phone numbers,
  national ID numbers (PAN/Aadhaar-style patterns, given this project's
  documents are India-based — INR salary figures, Indian phone formats),
  and credit-card-like numbers. These are near-universally NOT what a
  legitimate question is asking about, they're a side effect of a
  document mentioning a contact method or an ID number. Pure regex — no
  LLM call, no new dependency, runs automatically on every upload (same
  category as table extraction: free, so no opt-in needed).
- **Tier 2 (optional stretch, NOT built now unless asked) — free-text
  entities (names, addresses).** This needs either an NER
  library/model or an LLM pass, costs more, and is exactly where the
  "Ajinkya" tension above becomes real — would need per-document opt-in,
  clearly separate from Tier 1, and probably a way to say "except this
  document" for legitimate profile/resume lookups. Left as a stretch,
  not part of this build.

```text
STEP PII1   backend/pii.py — PII_PATTERNS: compiled regexes for email,
            phone (Indian + generic international formats), PAN-style
            (AAAAA9999A), Aadhaar-style (12 digits, optionally spaced),
            credit-card-like (13-16 digits). redact_text(text) -> (str,
            int): replace each match with a labeled placeholder (e.g.
            "[REDACTED_EMAIL]"), return the redacted text and a count of
            replacements made.

STEP PII2   Wire into load_pdf() (backend/ingest.py): run redact_text()
            on each page's FINAL text — AFTER the table/image steps have
            already appended their content, so nothing they add escapes
            redaction, and before chunking/before the function returns
            (must run on every path, including when include_images is
            False — an early return before this point would skip it).
            Runs unconditionally, like table extraction. Log how many
            redactions happened per page (fits Section 14's logging
            work) — this is also how a real leak would first get
            noticed, so this log line matters. DONE.

STEP PII3   Test standalone: feed redact_text() a string with a known
            email/phone/PAN-style number, confirm each is replaced and
            the rest of the text is untouched.

STEP PII4   End-to-end test: upload a document containing a fake-but-
            realistic email/phone number, confirm it never appears in an
            answer or a citation excerpt; then upload the existing
            resume test document and confirm a legitimate name/role
            question (the Ajinkya-style question) still works exactly as
            before — this is the regression check that actually matters.
```

**What does NOT change**: chunking, embedding, retrieval, reranking, and
citations are all untouched — redaction happens once, to the page text,
before any of them run. A chunk that was redacted just has different
text; nothing downstream needs to know why.

### 19.2 "Model Thinking" Panel (Pipeline Transparency)

**Decision (per the user): show the pipeline's own real steps, not a
separate LLM-generated explanation.** Every piece of this is already
computed by `generate_answer()`/`generate_answer_stream()` today and
currently thrown away after the request — rewritten question, how many
candidates were retrieved, their similarity scores, how many survived the
score threshold, how many the reranker kept. Surfacing it costs nothing
extra: no new LLM call, no new latency, just returning data that already
exists.

**Not default, collapsible** — a `st.expander("🧠 Show thinking",
expanded=False)` under each assistant message, per the user's own
instruction. A user who never opens it sees no change at all.

```text
STEP MT1   backend/rag.py — extend generate_answer()'s and
            generate_answer_stream()'s return dict with a "thinking" key:
            {"rewritten_question", "retrieved_count", "retrieved_scores"
            (rounded), "passed_threshold_count", "kept_after_rerank_count"}.
            Populate it at each stage of the existing pipeline (no new
            computation — these numbers already exist mid-function today,
            just not returned). DONE — build_thinking() helper, called at
            all three return points in each function.

STEP MT2   backend/main.py — /query and /query/stream both already return
            the full result dict / the "done" event — just make sure
            "thinking" rides along unchanged. DONE — confirmed no code
            change was needed beyond MT1.

STEP MT3   backend/database.py / add_message() — persist "thinking"
            alongside a message (same pattern as `sources`), so it's
            still visible after a thread reload. DONE — new `thinking`
            JSONB column (migration-safe `ALTER TABLE ADD COLUMN IF NOT
            EXISTS`, since `messages` already existed in earlier
            deployments), `add_message()`/`get_messages()` updated, both
            endpoints pass `result.get("thinking")` through.

STEP MT4   frontend/app.py — render_thinking(): an expander showing the
            rewritten question (only if it differs from the original),
            retrieved candidate count + similarity scores, how many
            passed the threshold, how many the reranker kept. DONE —
            called after render_sources() in both the thread-replay loop
            and the live-streaming block.

STEP MT5   End-to-end test: ask a follow-up question (so the rewritten
            question actually differs), confirm the expander shows it
            plus retrieval scores; confirm it's collapsed by default and
            the answer/citations look exactly as before when it's left
            closed.
```

**What does NOT change**: the answer itself, retrieval, reranking, and
citations are all identical to today — this only adds a way to SEE
numbers that were already being computed, nothing about how they're
computed changes.

### 19.3 Suggested Build Order

1. **PII redaction (Tier 1)** — higher stakes (real company documents),
   should land first.
2. **Model thinking panel** — purely additive UI/transparency feature,
   no urgency tied to it, safe to do second.

---

## 20. Multi-Format Document Support (.docx, .txt, .md) — Implementation Plan

### 20.1 Tech Choice: Lightweight Per-Format Loaders, Not Docling

**Decision (per the user, after checking Docling's actual footprint):
lightweight, format-specific loaders — no Docling.** Docling ships with
PyTorch bundled in its own wheel and, for PDFs specifically, downloads its
own layout-detection model weights on first run. That's a large
dependency footprint whose main value (excellent unified PDF/table/layout
parsing) this project doesn't need — the PDF pipeline is already built
and working (`PyPDFLoader` + `pdfplumber` tables + the custom vision-LLM
OCR). Pulling in Docling just to also read `.docx`/`.txt`/`.md` would mean
a multi-hundred-MB install for formats that are structurally simple.

Instead:
- **`.txt` / `.md`** — read as plain text with Python's built-in `open()`.
  No library at all.
- **`.docx`** — `python-docx`, a small, pure-Python library that parses
  the format's XML directly. No ML models, no heavy dependency.
- **`.pdf`** — completely unchanged. The existing `load_pdf()` pipeline
  (PyPDFLoader, table extraction, opt-in image description, PII
  redaction) is untouched.

### 20.2 The "No Pages" Problem

`.txt`/`.md` have no page concept at all, and `.docx` pagination is a
*rendering-time* concern (how Word lays it out on screen/print) — it is
NOT stored data in the file the way a PDF's page boundaries are. So a
citation for these formats can only ever be `{filename}`, never
`{filename}, Page N}`.

`get_sources()` already tolerates this gracefully today: `page_display`
is `None` whenever `metadata["page"]` isn't an int. The one real fix
needed is on the frontend — `render_sources()` currently always prints
`"— Page {page}"`, which would literally show `"— Page None"`. That needs
to become conditional: show the page segment only when a page actually
exists.

### 20.3 What Does NOT Extend to New Formats (This Build)

- **Table extraction** (Section 16.2) and **image description/OCR**
  (Section 15) stay PDF-only for now — both are built around
  `pdfplumber`'s page-level PDF-specific API. `.docx` CAN contain tables/
  images too, but that's a real stretch goal, not part of this build.
- **PII redaction** (Section 19.1), by contrast, SHOULD apply to every
  format equally — it's pure regex over whatever final text a document
  produces, format-agnostic by nature. This build refactors redaction
  into a shared step so `.txt`/`.md`/`.docx` get the same protection as
  `.pdf` already does.

### 20.4 Step-by-Step Build Plan

```text
STEP F1   backend/ingest.py — load_text_file(file_path) -> list[Document]:
          read the whole file as ONE Document (page_content = file text,
          metadata = {"source": file_path, "page": None}). Covers both
          .txt and .md (Markdown syntax is left as-is in the text — it
          still embeds/reads fine, no need to strip it).

STEP F2   backend/ingest.py — load_docx_file(file_path) -> list[Document]:
          use python-docx to join every paragraph's text into ONE
          Document (same shape as F1's output: page_content + {"source",
          "page": None}).

STEP F3   backend/ingest.py — load_document(file_path, include_images) ->
          list[Document]: a dispatcher that picks load_pdf() /
          load_text_file() / load_docx_file() by file extension. The
          existing PII-redaction pass (_redact_pages(), Section 19.1) is
          pulled out so it runs on EVERY format's output here, not just
          inside load_pdf() — a shared, format-agnostic final step.
          ingest_pdf() is renamed to ingest_document() (or kept as an
          alias) and now calls load_document() instead of load_pdf()
          directly.

STEP F4   backend/main.py — extend the /upload endpoint's validation to
          accept .pdf/.docx/.txt/.md and reject anything else with a
          clear 400 error (same "validate before trusting" spirit as
          safe_filename()'s path-traversal check).

STEP F5   frontend/app.py — st.file_uploader(type=["pdf", "docx", "txt",
          "md"]), update the upload section's label/help text away from
          "PDF"-specific wording. Fix render_sources() to omit the
          "— Page N" segment when a source has no page.

STEP F6   requirements.txt — add python-docx.

STEP F7   End-to-end test: upload a .txt, a .md, and a .docx file (one
          each), ask a real question about each, confirm a correct
          answer with a filename-only citation (no page); re-upload an
          existing PDF and confirm its citations still show page numbers
          exactly as before — this is the regression check that matters.
```

### 20.5 What Does NOT Change

Retrieval, reranking, chunking, and the answer prompt are all completely
format-agnostic already (they operate on `Document` objects regardless of
where the text came from) — nothing about them changes. The PDF pipeline
specifically (tables, OCR, its own PII redaction call site) is untouched;
only the dispatch point and the now-shared redaction step are new.

---

## 21. Per-Thread Tunable Settings Panel — Implementation Plan

### 21.1 What This Covers

Five pipeline knobs that are currently fixed constants in `config.py`
become live, per-thread-adjustable settings:

| Setting | Config default | What it controls | Effect of raising it |
|---|---|---|---|
| `similarity_threshold` | 1.8 | "Weak evidence" cutoff (Rule 16) — a chunk only survives if its distance score is at or below this. **Lower score = more similar**, so raising the threshold = LOOSER (more chunks pass); lowering it = STRICTER (more "I don't know"). | Looser retrieval, less likely to falsely decline |
| `retrieval_top_k` | 6 | How many chunks the reranker keeps for the final answer prompt. | More context reaches the LLM, more tokens/cost |
| `rerank_candidate_k` | 15 | How wide the initial candidate pool is BEFORE reranking. | Better recall (less likely to miss a relevant chunk), slightly slower |
| `rewrite_history_messages` | 6 | How many past messages the question-rewriter sees to resolve follow-ups. Setting this to **0 disables conversational memory entirely** (`rewrite_question()` already skips rewriting when there's no history to work with) — a free on/off toggle, no new code path needed. | Better follow-up resolution, one extra LLM call reads more context |
| `llm_temperature` | 0 | How deterministic vs varied the final answer's phrasing is. | More varied phrasing — **caution: 0 is the safest default for a fact-grounded Q&A app; raising this is a real accuracy/consistency tradeoff, not a free customization, and the UI should say so.** |

### 21.2 Decision: Persisted Per-Thread, One JSONB Column

Same per-thread persistence decision as before, but with 5 knobs instead
of 1, a single `settings JSONB` column on `threads` (not 5 separate
columns) keeps this from being 5x the migration/plumbing work:
- `NULL` or a missing key means "use the config default" for that knob.
- Updating one setting only ever sends that one key — Postgres's `||`
  jsonb-concatenation operator merges it into whatever's already stored,
  so the frontend never needs to know the other 4 current values just to
  change one.

### 21.3 Step-by-Step Build Plan

```text
STEP AS1   backend/database.py — ALTER TABLE threads ADD COLUMN IF NOT
           EXISTS settings JSONB. Update list_threads() to include it.
           Add get_thread_settings(thread_id) -> dict (returns {} when
           NULL) and update_thread_settings(thread_id, partial: dict) ->
           dict, using
           `settings = COALESCE(settings, '{}'::jsonb) || %s::jsonb`
           for the merge.

STEP AS2   backend/rag.py (or a new backend/settings.py) —
           DEFAULT_SETTINGS = {the 5 config constants, by the names in
           21.1's table} and resolve_settings(thread_settings: dict) ->
           dict, returning {**DEFAULT_SETTINGS, **thread_settings}. One
           place that knows how every knob falls back, instead of 5
           separate `if x is None: x = CONFIG_CONST` checks scattered
           around.

STEP AS3   Thread each resolved setting through where the matching
           constant is used TODAY:
             - rag.py: similarity_threshold in the score filter,
               rerank_candidate_k as retrieve_with_scores()'s k.
             - reranker.py: rerank_chunks() gets a top_k param (currently
               reads RETRIEVAL_TOP_K itself) for how many it keeps.
             - rewriter.py: rewrite_question() gets a history_limit param
               (currently reads REWRITE_HISTORY_MESSAGES itself).
             - llm.py: get_llm() gets an optional temperature param
               (currently hardcodes LLM_TEMPERATURE), falling back to the
               config constant when not given.
           generate_answer()/generate_answer_stream() take one new
           `settings: dict` parameter (already-resolved, from AS2) and
           pass the right piece to each. build_thinking() already shows
           similarity_threshold — extend it to show all 5.

STEP AS4   backend/main.py — GET /config/defaults ->  DEFAULT_SETTINGS
           (single source of truth for the frontend); PATCH
           /threads/{thread_id}/settings, body = a partial dict of any
           of the 5 keys, calls update_thread_settings(). /query and
           /query/stream resolve the active thread's settings via
           get_thread_settings() + resolve_settings() before calling
           generate_answer()/generate_answer_stream().

STEP AS5   frontend/app.py — a "⚙️ Settings" expander in the sidebar with
           5 sliders, each seeded from the active thread's resolved
           settings (or GET /config/defaults when there's no thread
           yet). Moving any one calls PATCH with just that one key.
           Temperature's slider gets an explicit caution caption per
           21.1. retrieval_top_k's slider max should stay <=
           rerank_candidate_k's current value (a candidate pool smaller
           than what you're asking to keep doesn't make sense) — enforce
           with the slider's own min/max bounds reacting to the other's
           current value, or just a caption noting the relationship.

STEP AS6   End-to-end test, one per knob:
             - similarity_threshold: strict vs loose, as before.
             - retrieval_top_k: lower it to 1-2, confirm shorter/more
               narrowly-sourced answers; confirm citations count matches.
             - rerank_candidate_k: lower it below retrieval_top_k and
               confirm the system still behaves sanely (reranker simply
               can't keep more than it was given).
             - rewrite_history_messages = 0: ask a follow-up question
               ("his role?" after discussing a name) and confirm it NO
               LONGER resolves the pronoun — conversational memory is
               genuinely off.
             - llm_temperature: raise it and ask the same question twice,
               observe whether phrasing varies run to run.
           For all 5: confirm "🧠 Show thinking" reflects the actual
           resolved values, and that settings persist across a reload/
           thread switch, independently per thread.
```

### 21.4 What Does NOT Change

A thread that never has its settings touched behaves identically to
today — `resolve_settings({})` returns exactly today's 5 hardcoded
constants. Retrieval logic, reranking, and citations are unchanged in
HOW they work; only WHERE each number comes from does.
