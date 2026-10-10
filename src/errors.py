import numpy as np
from collections import Counter
from src.evalutil import load

qs = load("test")
corpus = load("corpus")
rb = np.load("data/bm25_test_ranks.npy")
rd = np.load("data/minilm_test_ranks.npy")
rh = np.load("data/hybrid_test_ranks.npy")

types = ["what", "who", "when", "where", "which", "how", "why"]


def qtype(q):
    w = q.lower().split()[0]
    return w if w in types else "other"


kinds = [qtype(q["question"]) for q in qs]
print("type | n | bm25 r@1 | minilm r@1 | hybrid r@1")
for t in types + ["other"]:
    idx = [i for i, k in enumerate(kinds) if k == t]
    if idx:
        a = lambda r: np.mean([r[i] == 1 for i in idx])
        print(f"{t:6s} {len(idx):4d}   {a(rb):.3f}   {a(rd):.3f}   {a(rh):.3f}")

miss = np.where(rh > 10)[0]
print("\nhybrid misses outside top 10:", len(miss))
for i in miss[:5]:
    print("-", qs[i]["question"], "| answer:", qs[i]["answer"], "| rank:", int(rh[i]))
