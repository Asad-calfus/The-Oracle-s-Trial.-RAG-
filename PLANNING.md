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
backend/rag.py

Purpose:
The query pipeline: takes a question, retrieves relevant chunks via
vectorstore.py, builds a strict prompt, calls the LLM, and returns
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

**Stopping here as instructed.** No code or files beyond this planning doc have been created yet. Say **"next"** when you want me to start Step 1 (repository structure).
