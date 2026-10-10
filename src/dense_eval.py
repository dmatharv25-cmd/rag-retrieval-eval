import sys
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from src.evalutil import load, rank_of_target, metrics, bootstrap_ci

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert split in ("dev", "test")

corpus = load("corpus")
qs = load(split)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

import os
emb_path = "data/minilm_corpus_emb.npy"
if os.path.exists(emb_path):
    C = np.load(emb_path)
    print("loaded corpus embeddings:", C.shape)
else:
    start = time.time()
    C = model.encode(corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    np.save(emb_path, C.astype(np.float32))
    print("embedded corpus in", round(time.time() - start), "seconds", C.shape)

start = time.time()
Q = model.encode([q["question"] for q in qs], batch_size=64, normalize_embeddings=True)
S = (Q @ C.T).astype(np.float32)
print("scored", len(qs), "questions in", round(time.time() - start), "seconds")

ranks = np.array([rank_of_target(S[i], q["context_id"]) for i, q in enumerate(qs)])
m = metrics(ranks)
for k, v in m.items():
    vals = (ranks <= int(k.split("@")[1])).astype(float) if k.startswith("recall") else 1.0 / ranks
    lo, hi = bootstrap_ci(vals)
    print(f"MiniLM {split} {k}: {v:.3f} (95% interval {lo:.3f} to {hi:.3f})")

np.save(f"data/minilm_{split}_scores.npy", S)
np.save(f"data/minilm_{split}_ranks.npy", ranks)
