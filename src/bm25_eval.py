import re
import sys
import time
import numpy as np
from rank_bm25 import BM25Okapi
from src.evalutil import load, rank_of_target, metrics, bootstrap_ci

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert split in ("dev", "test")


def tok(s):
    return re.findall(r"\w+", s.lower())


corpus = load("corpus")
qs = load(split)

start = time.time()
bm25 = BM25Okapi([tok(c) for c in corpus])
print("index built in", round(time.time() - start), "seconds")

start = time.time()
ranks = []
score_rows = []
for q in qs:
    s = bm25.get_scores(tok(q["question"]))
    ranks.append(rank_of_target(s, q["context_id"]))
    score_rows.append(s.astype(np.float32))
print("scored", len(qs), "questions in", round(time.time() - start), "seconds")

m = metrics(ranks)
for k, v in m.items():
    lo, hi = bootstrap_ci((np.array(ranks) <= int(k.split("@")[1])).astype(float)) if k.startswith("recall") else bootstrap_ci(1.0 / np.array(ranks))
    print(f"BM25 {split} {k}: {v:.3f} (95% interval {lo:.3f} to {hi:.3f})")

np.save(f"data/bm25_{split}_scores.npy", np.stack(score_rows))
np.save(f"data/bm25_{split}_ranks.npy", np.array(ranks))
