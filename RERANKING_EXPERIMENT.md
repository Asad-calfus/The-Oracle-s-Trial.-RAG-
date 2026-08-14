# Experiment: Plain Retrieval vs. LLM-Based Reranking

## What was compared

Two pipelines, differing in exactly one step (the reranking step), everything else identical:

**Baseline (no rerank)**
`embed question -> top-6 chunks by embedding distance -> drop chunks above SIMILARITY_SCORE_THRESHOLD -> LLM answer`

**Reranked (current `backend/rag.py`)**
`embed question -> top-15 candidate chunks by embedding distance -> drop chunks above SIMILARITY_SCORE_THRESHOLD -> ask the LLM to rank the 15 by actual relevance (backend/reranker.py) -> keep top-6 -> LLM answer`

Same config for both: `RETRIEVAL_TOP_K=6`, `RERANK_CANDIDATE_K=15`, `SIMILARITY_SCORE_THRESHOLD=1.8`, same LLM (`gpt-4o-mini`), same Chroma store.

**Corpus at test time:** `Calfus_Intern_FAQ-2.pdf` (30 chunks) + `Ajinkya_Mahesh_Pawar_RESUME.pdf` (10 chunks) — 40 chunks total, currently the only documents ingested in `data/chroma`.

**Data quality note (found while running this, not the point of the experiment):** every single result below shows each retrieved chunk appearing twice, back to back. That means both PDFs were likely ingested twice (e.g. `/upload` called twice, or once before and once after a restart). It didn't invalidate this comparison since both pipelines saw the same duplication, but it's worth deduplicating (check for existing `source` in Chroma before re-ingesting) before trusting chunk counts or running bigger experiments.

## Results

| # | Question | Chunk order changed? | Final answer changed? | Baseline latency | Reranked latency | Latency cost |
|---|----------|----------------------|------------------------|-------------------|--------------------|--------------|
| 1 | What are the working hours for interns? | No | No | 2.66s | 3.07s | +0.41s |
| 2 | What is the notice period to leave early? | No | No | 1.92s | 2.75s | +0.83s |
| 3 | What kinds of leave are available (besides holidays)? | No | No | 2.03s | 3.00s | +0.97s |
| 4 | What programming languages does Ajinkya know? | **Yes** — Skills chunk moved from position 3 to position 1 | No (same facts, reworded) | 1.15s | 2.38s | +1.23s |
| 5 | What was Ajinkya's role at Airports Authority of India? | **Yes** — Experience chunk moved from position 5 to position 1 | No (same facts, reworded) | 1.85s | 4.96s | +3.11s |
| 6 | What is the capital of France? *(out-of-scope control)* | No | No — both correctly said "I don't know based on the uploaded documents." | 1.42s | 2.30s | +0.88s |

Average latency cost of reranking: **~+1.24s per query** (one extra LLM round-trip).

## Findings

1. **Reranking never changed the final answer in this test.** In every case, the answer was factually identical between the two pipelines (question 4/5 had slightly reworded phrasing, same facts).
2. **Reranking did visibly fix chunk ordering** in 2 of 6 questions — it correctly promoted the truly relevant chunk (the resume's "Skills" section, the "Airports Authority" experience bullet) from a middling position up to #1, ahead of less relevant chunks like contact info.
3. **Ordering didn't matter here because `top_k=6` already comfortably held all the relevant content.** The LLM reads the *entire* context block at once, not just the first chunk — so as long as the right chunk is *somewhere* in the 6, order alone doesn't change the answer. Reranking's reordering would matter more if `top_k` were smaller (e.g. 2-3), where a demoted chunk would get cut entirely instead of just moved down.
4. **Reranking has a real, consistent latency cost** — every query got slower, from +0.4s up to +3.1s, because it's a second full LLM call.
5. **The out-of-scope control question behaved identically** in both pipelines — reranking doesn't interfere with the "I don't know" path.

## Why keep reranking anyway

This project's own earlier threshold experiment (see the comment in `backend/config.py` around `SIMILARITY_SCORE_THRESHOLD`) already found that embedding-distance scores for relevant and irrelevant chunks *overlap* (relevant: 1.3–1.66, irrelevant: 1.39–1.7 on a real resume). That means distance alone cannot reliably separate "actually relevant" from "not relevant" — there's no clean cutoff. Reranking replaces that unreliable distance signal with the LLM actually reading each candidate passage, which is why it correctly promoted the right chunks in questions 4 and 5 even though embedding distance had ranked them lower.

The benefit didn't show up as *better answers* here only because this corpus is small (40 chunks, 2 documents) — a wrong ordering rarely means the right chunk gets excluded when `top_k=6` already fits almost everything. The real payoff shows up as documents pile up: with more PDFs competing for the same top-6 slots, embedding search is more likely to leave the actually-correct chunk *outside* the top-6 entirely (a recall miss, not just a ranking miss) — that's exactly the case `RERANK_CANDIDATE_K=15` (cast a wider net, then let the LLM pick the true best 6) is designed to catch.

## Recommendation

Keep reranking — it's cheap insurance against the score-overlap problem you already measured, at a real but small latency cost (~1.2s/query with `gpt-4o-mini`). To actually see it change an *answer* (not just chunk order), rerun this same comparison after:
- Ingesting more/larger documents, so more chunks genuinely compete for the top-6 spots.
- Using paraphrased test questions that don't share vocabulary with the source PDF (e.g. asking "How much do I get paid if I get sick?" instead of "sick leave policy") — that's where embedding similarity is weakest and reranking should recover chunks the baseline would have missed entirely.

## Reproducing this experiment

```python
import sys, time, json
sys.path.insert(0, ".")

from backend.config import RETRIEVAL_TOP_K, RERANK_CANDIDATE_K, SIMILARITY_SCORE_THRESHOLD
from backend.rag import retrieve_with_scores, build_prompt, get_sources
from backend.reranker import rerank_chunks
from backend.llm import get_llm

def run_baseline(question):
    t0 = time.time()
    results = retrieve_with_scores(question, k=RETRIEVAL_TOP_K)
    good_chunks = [c for c, score in results if score <= SIMILARITY_SCORE_THRESHOLD]
    if not good_chunks:
        return {"answer": "I don't know based on the uploaded documents.", "elapsed": time.time() - t0}
    prompt = build_prompt(question, good_chunks)
    response = get_llm().invoke(prompt)
    return {"answer": response.content, "chunks": good_chunks, "elapsed": time.time() - t0}

def run_reranked(question):
    t0 = time.time()
    results = retrieve_with_scores(question, k=RERANK_CANDIDATE_K)
    good_chunks = [c for c, score in results if score <= SIMILARITY_SCORE_THRESHOLD]
    if not good_chunks:
        return {"answer": "I don't know based on the uploaded documents.", "elapsed": time.time() - t0}
    top_chunks = rerank_chunks(question, good_chunks)
    prompt = build_prompt(question, top_chunks)
    response = get_llm().invoke(prompt)
    return {"answer": response.content, "chunks": top_chunks, "elapsed": time.time() - t0}

# Run both for each test question and compare chunk order / answer / elapsed time.
```

Run with `python3 compare_rerank.py` from the project root, inside the venv, with `OPENAI_API_KEY` set in `.env`.
