# Retrieval Evaluation on SQuAD: BM25 vs Dense vs Hybrid

Which retrieval method finds the right paragraph for a question: keyword search (BM25), embeddings (MiniLM), or a weighted hybrid? Evaluated with bootstrap intervals on a fixed split.

## Summary

- **Corpus:** 20,958 unique SQuAD v1.1 paragraphs. **Dev:** 500 validation questions (all tuning). **Test:** 2,000 other validation questions, scored once. Seed 42.
- **Result (test):** hybrid recall@1 0.731, BM25 0.683, MiniLM 0.527.
- **Hybrid minus BM25, recall@1:** +4.7 points (95% paired bootstrap interval +3.3 to +6.1).
- **BM25 minus MiniLM, recall@1:** +15.6 points (interval +13.2 to +18.0). Keyword search clearly beat this dense model.
- **Main limit:** SQuAD questions were written while reading the paragraph, so they reuse its words. This likely favors BM25, but I did not test that.
- **Second dataset (SciFact, 300 test claims):** hybrid nDCG@10 0.717 vs BM25 0.652 (+6.5 points, 95% paired bootstrap interval +4.1 to +8.9). BM25 and MiniLM are close there (0.652 vs 0.648), unlike on SQuAD.

## Results (test set, 2,000 questions)

| Method | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| BM25 | 0.683 | 0.847 | 0.889 | 0.760 |
| MiniLM (all-MiniLM-L6-v2) | 0.527 | 0.771 | 0.838 | 0.635 |
| Hybrid (w=0.6 on MiniLM) | 0.731 | 0.905 | 0.944 | 0.810 |
| Hybrid, light (ONNX MiniLM + sparse BM25) | 0.732 | 0.904 | 0.942 | 0.811 |

95% bootstrap intervals for each cell are printed by `src.final_test`. The light version's are printed by `python -m src.light_eval test` (recall@1 0.712 to 0.751, MRR 0.796 to 0.826).

## Live demo and light version

- **Live demo:** https://rag-demo-7iru.onrender.com (free Render instance, so the first load can take a minute). Code: https://github.com/dmatharv25-cmd/rag-demo
- **Light version:** MiniLM runs through ONNX Runtime and BM25 uses precomputed sparse matrices, so the demo needs no torch. On the same test split it scores recall@1 0.732 and MRR 0.811, against 0.731 and 0.810 for the full hybrid. The differences are well inside the bootstrap intervals.
- **Memory:** about 355 MB after loading and about 400 MB after a query, measured locally on Windows. Render's free tier has 512 MB. On Render's free tier, the service stayed live after test queries, with no restart or out-of-memory event in Render's Events tab. I did not read the memory peak there.
- **Latency:** on Render's free tier (0.15 CPU, 512 MB), 12 consecutive queries on the live demo, including 4 repeats of the Rome question, all succeeded and returned in 1.6 to 2.0 s each (mean 1.8 s). Timed from a local machine, so this includes the network round trip. Times did not grow across the run.

## Method

- BM25: rank_bm25 (Okapi), lowercase word tokens.
- Dense: all-MiniLM-L6-v2, cosine similarity on normalized embeddings.
- Hybrid: per-question z-score of each method's scores across the corpus, then (1-w)*BM25 + w*MiniLM. The weight w=0.6 was chosen on dev by MRR (grid 0.0 to 1.0, step 0.1; 0.5 and 0.6 were nearly tied). Dev MRR at w=0.6 was 0.824, test 0.810.
- No model was trained, so there are no seeds.

## Second dataset: SciFact

Same pipeline on BEIR SciFact (5,183 abstracts, documents are title plus abstract). The 809 train-split claims were used as dev to choose the fusion weight (w=0.7, dev MRR 0.696; 0.689 at 0.6 and 0.692 at 0.8). The 300 test claims were scored once. Here recall@k means at least one relevant document in the top k.

| Method | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.523 | 0.797 | 0.624 | 0.652 |
| MiniLM (all-MiniLM-L6-v2) | 0.503 | 0.800 | 0.613 | 0.648 |
| Hybrid (w=0.7 on MiniLM) | 0.593 | 0.847 | 0.691 | 0.717 |

- Hybrid minus BM25: nDCG@10 +6.5 points (95% paired bootstrap interval +4.1 to +8.9), recall@1 +7.0 (+3.3 to +11.0). Hybrid minus MiniLM: nDCG@10 +6.9 (+4.2 to +9.7).
- BM25 and  MiniLM are close on SciFact (recall@1 differs by 2.0 points, nDCG@10 by 0.4). On SQuAD BM25 led by 15.6 recall@1 points. This is consistent with the word-overlap explanation for SQuAD, but the domain, task and corpus size also differ, so it does not test that explanation. I did not compute a paired interval for the BM25 vs MiniLM gap on SciFact.
- Intervals are wide (300 test claims, about 5 points either side). The weight was tuned on one dataset at a time, so 0.6 on SQuAD and 0.7 here are separate choices.
- Not checked: whether MiniLM's default sequence length truncates longer abstracts, and how these numbers compare with published BEIR results.
- Run it: `python -m src.scifact_eval` (downloads SciFact from Hugging Face as mteb/scifact; embedding the corpus took about 5 minutes on my CPU).

## Error analysis (test)

- BM25 beat MiniLM on recall@1 in every question type. The hybrid was best or tied in most, not in "when" (tied with BM25) or "why" (below BM25, only 29 questions).
- The hybrid had 112 of 2,000 questions with the right paragraph outside the top 10. Five I read by hand included questions with no content words ("Who discovered this and where did they come from?"), which no retriever could resolve.
- Per-type groups are small (29 to 906 questions), so differences within them are noisy.
- In the live demo, "Who was the first emperor of Rome?" returns a Constantine paragraph at rank 1 and the Augustus paragraph at rank 2. Lexical overlap can outrank the true answer. This is one example, not a measured rate.

## Run it yourself

1. `python -m venv .venv`, activate, `pip install -r requirements.txt`
2. `python -m src.data`
3. `python -m src.bm25_eval dev`, `python -m src.dense_eval dev` (the corpus embedding took about 43 minutes on my CPU), `python -m src.hybrid_dev`
4. `python -m src.bm25_eval test`, `python -m src.dense_eval test`, `python -m src.final_test`, `python -m src.errors`

Tested on Windows with Python 3.12 only.

## Light version: run it yourself

Needs the full-pipeline outputs first: steps 1 and 2 above, plus `python -m src.dense_eval dev` once, which builds `data/minilm_corpus_emb.npy`.

1. `python -m src.build_index` builds the sparse BM25 index (`data/bm25_M.npz`, `bm25_idf.npy`, `bm25_vocab.json`).
2. `python -m src.export_onnx minilm_onnx` exports all-MiniLM-L6-v2 to ONNX (inputs `input_ids` and `attention_mask`) and saves the tokenizer files.
3. `python -m src.onnx_check` compares ONNX and torch embeddings on three queries. I got cosine 1.0 on all three.
4. `python -m src.light_eval test` prints recall@1 0.732, recall@5 0.904, recall@10 0.942 and MRR 0.811 with bootstrap intervals.

Embedding recipe: tokenize with truncation at 256 tokens, run the ONNX graph, mean-pool over the attention mask, then L2-normalize. The graph outputs token states only, so pooling and normalization happen in the code.

I rebuilt the ONNX file with this script and re-ran steps 3 and 4. The metrics matched the file used for the demo to three decimals. The two files differ by about 6 KB in size, so they are not byte-identical. The live demo uses the earlier export.

Light version tested with torch 2.14.1, transformers 5.19.0, onnx 1.23.2 and onnxruntime 1.31.0. The export uses the legacy TorchScript exporter and prints tracer warnings, which did not change the results on this corpus.

## What I did not test

- Only one dense model, one tokenizer and one fusion method. Other models (for example larger or retrieval-tuned ones), BM25 settings and rank-based fusion were not tried.
- Why BM25 beats MiniLM here (word overlap with the paragraph) is a hypothesis.
- Paragraph-level retrieval only. No answer generation, no passage chunking.
- Dev and test questions can point to the same paragraph.
- SQuAD is Wikipedia text with written questions. Results may differ on real user queries.
- The weight was tuned on 500 questions and was nearly flat between 0.5 and 0.6.
- Ties are ranked optimistically (only strictly higher scores count). This affects 13 of 2,000 BM25 test questions and changes no reported metric at the precision shown.
