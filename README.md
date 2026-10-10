# Retrieval Evaluation on SQuAD: BM25 vs Dense vs Hybrid

Which retrieval method finds the right paragraph for a question: keyword search (BM25), embeddings (MiniLM), or a weighted hybrid? Evaluated with bootstrap intervals on a fixed split.

## Summary

- **Corpus:** 20,958 unique SQuAD v1.1 paragraphs. **Dev:** 500 validation questions (all tuning). **Test:** 2,000 other validation questions, scored once. Seed 42.
- **Result (test):** hybrid recall@1 0.731, BM25 0.683, MiniLM 0.527.
- **Hybrid minus BM25, recall@1:** +4.7 points (95% paired bootstrap interval +3.3 to +6.1).
- **BM25 minus MiniLM, recall@1:** +15.6 points (interval +13.2 to +18.0). Keyword search clearly beat this dense model.
- **Main limit:** SQuAD questions were written while reading the paragraph, so they reuse its words. This likely favors BM25, but I did not test that.

## Results (test set, 2,000 questions)

| Method | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| BM25 | 0.683 | 0.847 | 0.889 | 0.760 |
| MiniLM (all-MiniLM-L6-v2) | 0.527 | 0.771 | 0.838 | 0.635 |
| Hybrid (w=0.6 on MiniLM) | 0.731 | 0.905 | 0.944 | 0.810 |

95% bootstrap intervals for each cell are printed by `src.final_test`.

## Method

- BM25: rank_bm25 (Okapi), lowercase word tokens.
- Dense: all-MiniLM-L6-v2, cosine similarity on normalized embeddings.
- Hybrid: per-question z-score of each method's scores across the corpus, then (1-w)*BM25 + w*MiniLM. The weight w=0.6 was chosen on dev by MRR (grid 0.0 to 1.0, step 0.1; 0.5 and 0.6 were nearly tied). Dev MRR at w=0.6 was 0.824, test 0.810.
- No model was trained, so there are no seeds.

## Error analysis (test)

- BM25 beat MiniLM on recall@1 in every question type. The hybrid was best or tied in most, not in "when" (tied with BM25) or "why" (below BM25, only 29 questions).
- The hybrid had 112 of 2,000 questions with the right paragraph outside the top 10. Five I read by hand included questions with no content words ("Who discovered this and where did they come from?"), which no retriever could resolve.
- Per-type groups are small (29 to 906 questions), so differences within them are noisy.

## Run it yourself

1. `python -m venv .venv`, activate, `pip install -r requirements.txt`
2. `python -m src.data`
3. `python -m src.bm25_eval dev`, `python -m src.dense_eval dev` (the corpus embedding took about 43 minutes on my CPU), `python -m src.hybrid_dev`
4. `python -m src.bm25_eval test`, `python -m src.dense_eval test`, `python -m src.final_test`, `python -m src.errors`

Tested on Windows with Python 3.12 only.

## What I did not test

- Only one dense model, one tokenizer and one fusion method. Other models (for example larger or retrieval-tuned ones), BM25 settings and rank-based fusion were not tried.
- Why BM25 beats MiniLM here (word overlap with the paragraph) is a hypothesis.
- Paragraph-level retrieval only. No answer generation, no passage chunking.
- Dev and test questions can point to the same paragraph.
- SQuAD is Wikipedia text with written questions. Results may differ on real user queries.
- The weight was tuned on 500 questions and was nearly flat between 0.5 and 0.6.
