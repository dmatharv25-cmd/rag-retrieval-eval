"""Encode the corpus and score dev/test with BAAI/bge-small-en-v1.5.
Run: python -m src.bge_eval   (corpus encoding is cached in data/bge_corpus_emb.npy)
"""
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from src.evalutil import load, rank_of_target, metrics, bootstrap_ci

PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    corpus = load("corpus")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    model.max_seq_length = 256

    emb_path = "data/bge_corpus_emb.npy"
    if os.path.exists(emb_path):
        C = np.load(emb_path)
        print("loaded corpus embeddings:", C.shape)
    else:
        start = time.time()
        C = model.encode(corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        np.save(emb_path, C.astype(np.float32))
        print("embedded corpus in", round(time.time() - start), "seconds", C.shape)

    for split in ("dev", "test"):
        qs = load(split)
        Q = model.encode([PREFIX + q["question"] for q in qs], batch_size=64, normalize_embeddings=True)
        S = (Q @ C.T).astype(np.float32)
        ranks = np.array([rank_of_target(S[i], q["context_id"]) for i, q in enumerate(qs)])
        m = metrics(ranks)
        for k, v in m.items():
            vals = (ranks <= int(k.split("@")[1])).astype(float) if k.startswith("recall") else 1.0 / ranks
            lo, hi = bootstrap_ci(vals)
            print(f"BGE {split} {k}: {v:.3f} (95% interval {lo:.3f} to {hi:.3f})")
        np.save(f"data/bge_{split}_scores.npy", S)
        np.save(f"data/bge_{split}_ranks.npy", ranks)


if __name__ == "__main__":
    main()
