# Port·06 — The Oracle's Trial — Readiness Check

A plain-language check of SmartDoc against the mission's deliverables and the 6 mentor questions (M6S1–M6S6). Every "yes" below is backed by something actually checked in the code or the live database — not assumed.

## ⚠️ Fix this before the demo

**`/upload` has a known, unresolved bug** — it started returning a generic `500 Internal Server Error` on every upload partway through the last work session. Calling the ingestion code directly (without going through the API) works fine, which points at the running server process itself (most likely a `--reload` restart that didn't come back up cleanly), not the ingestion logic.

Why this matters: almost every deliverable and mentor question below depends on being able to upload a document live. **Restart `uvicorn` fresh and confirm a real upload works before doing anything else.** If it still fails, the first thing to check is the uvicorn terminal's own traceback — that will say exactly what broke.

## Deliverables checklist

| Deliverable | Status | Evidence |
|---|---|---|
| Parse and chunk 5+ PDFs into clean text | ✅ Done | 5 different real PDFs have actually been uploaded and chunked: a Communication Policy, a Certification Reimbursement Policy, an Employee Referral Policy, an ISO 27001 standard, and a resume. Checked live in the database — 239 chunks total. |
| Generate and store embeddings in a vector DB | ✅ Done | Every chunk is embedded with OpenAI's `text-embedding-3-small` and stored in **ChromaDB**, a real vector database — not a Python list in memory. |
| Build the RAG pipeline (query → retrieve → answer) | ✅ Done | Full pipeline exists: question → retrieve candidates → filter out weak matches → LLM re-ranks the survivors → build a strict prompt → LLM answers. |
| Streamlit UI with source citation on every answer | ✅ Done (with a good nuance) | Every real answer shows filename + page. When the system can't answer, it correctly shows **no citation at all** rather than a fake one — see M6S5 for why that's the right behavior, not a bug. |
| Demo to mentor with 3 test documents | 🟡 Ready, pending the upload fix | You already have 5 real documents on hand, more than enough for the demo — just need `/upload` working again first. |

## Mentor questions — answered

### M6S1 — "What chunk size did you choose, and why?"

**700 characters per chunk, 100 characters of overlap.**

In plain terms: a "chunk" is one bite-sized piece of a document that gets its own entry in the vector database.

- **Too big a chunk** mixes several unrelated topics into one piece of text. That confuses the search (the embedding represents "a bit of everything" instead of one clear idea) and wastes the AI's attention on irrelevant text sitting next to the useful part.
- **Too small a chunk** cuts a sentence or an answer in half, so the important detail might get separated from the context that explains it.
- **700 characters** is roughly one paragraph — big enough to hold one complete idea, small enough that a search for "what's the notice period?" pulls back *just* that answer, not the whole policy document.
- **The 100-character overlap** is a safety margin: if an important sentence happens to sit right at the boundary between two chunks, the overlap means it still shows up whole in at least one of them, instead of being sliced in two and losing its meaning in both.

Be honest if asked further: this size was chosen using standard RAG practice, not by testing multiple sizes against this specific document set and measuring which worked best — the code comment even calls it "an initial guess, not a tuned value." That's a fair thing to say if pushed — it shows you understand the tradeoff even though you haven't exhaustively tuned it.

### M6S2 — "Show me where the embeddings are stored and persisted."

Point to: **`data/chroma/`** on disk. That's a real ChromaDB database folder (a `chroma.sqlite3` file plus a data folder), not something that disappears when the app restarts. The connection is set up in `backend/vectorstore.py`, reading the folder path from `backend/config.py`.

Right now it holds 239 real chunks from 5 real documents — you can prove persistence live by restarting the backend and showing the documents are still listed and still answerable.

### M6S3 — "Retrieval returns relevant chunks for 3 mentor-prepared questions — do citations match?"

This one has to be demonstrated live with the mentor's own questions, so it can't be "pre-answered" here. But you have real precedent it works: earlier testing with questions like *"What are the working hours for interns?"* and *"What was the candidate's role at their previous internship?"* correctly retrieved the right chunk and cited the right file and page every time (see `RERANKING_EXPERIMENT.md` for the full test log).

**Before the demo:** ask 2-3 questions yourself against a document you know well, and manually confirm the cited page actually contains that answer. That's the fastest way to catch a surprise.

### M6S4 — "Source citation shows exact document and section — filename, and ideally paragraph."

**Filename + page number** — both present on every real answer, pulled from the PDF's own metadata (never invented by the AI, so it can't be wrong or made up).

**Gap to know about:** citations currently stop at the page level, not the paragraph/section level. If the mentor specifically asks "which paragraph," the honest answer is: not implemented yet — page number is the current granularity. Worth knowing before it's asked, not worth panicking about; page-level citation is a normal, solid answer for this kind of project.

### M6S5 — "Handles out-of-scope questions without hallucinating — does it say 'I don't know'?"

**Yes, and this is one of the strongest parts of the build.** Two independent safety layers, plus a recent fix:

1. A distance-score filter throws out chunks that aren't even remotely close to the question.
2. The AI is strictly instructed to answer *only* from the given text, and to say exactly *"I don't know based on the uploaded documents"* if the answer isn't there.
3. **Recently tightened:** the re-ranking step now actively *drops* chunks that don't genuinely help answer the question (it used to just reorder them, keeping some irrelevant ones around) — and when the final answer is "I don't know," the system now deliberately shows **zero citations**, instead of citations that would misleadingly suggest something was found.

Tested for real with a deliberately unrelated question ("What is the capital of France?") against uploaded documents that had nothing to do with it — it correctly answered "I don't know" both times, before and after the recent fix.

### M6S6 — "Works on a new document the intern has not tested with before."

**Structurally, yes** — nothing in the pipeline is hardcoded to any specific document; any PDF goes through the exact same load → chunk → embed → store steps. **In practice, this cannot be demonstrated right now** because of the `/upload` bug flagged at the top of this doc. Fix that first, then this becomes a simple "upload something new live" test.

## Summary

Everything the mission asks for has genuinely been built, and most of it is backed by real, checked evidence rather than assumptions — the RAG pipeline, the vector database, the reranking, and especially the hallucination-avoidance (M6S5) are all in solid shape. The two things actually worth doing before the mentor demo:

1. **Fix `/upload`** — nothing else can be demoed live until this works again.
2. **Decide what to say about paragraph-level citation (M6S4)** and the "not empirically tuned" chunk size (M6S1) if asked — both are fine answers, just be ready to give them honestly rather than be caught off guard.
