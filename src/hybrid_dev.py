import numpy as np
from src.evalutil import load, rank_of_target, metrics, bootstrap_ci

qs = load("dev")
B = np.load("data/bm25_dev_scores.npy")
D = np.load("data/minilm_dev_scores.npy")


def z(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


Bz, Dz = z(B), z(D)
rb = np.load("data/bm25_dev_ranks.npy")
rd = np.load("data/minilm_dev_ranks.npy")

b_ok, d_ok = rb <= 10, rd <= 10
print("both find it in top 10:", int((b_ok & d_ok).sum()))
print("only BM25 finds it:", int((b_ok & ~d_ok).sum()))
print("only MiniLM finds it:", int((~b_ok & d_ok).sum()))
print("neither finds it:", int((~b_ok & ~d_ok).sum()))

print("weight on MiniLM -> recall@1, recall@10, mrr (dev)")
best = None
for w in [round(x * 0.1, 1) for x in range(11)]:
    S = (1 - w) * Bz + w * Dz
    ranks = np.array([rank_of_target(S[i], q["context_id"]) for i, q in enumerate(qs)])
    m = metrics(ranks)
    print(f"  w={w}: {m['recall@1']:.3f}  {m['recall@10']:.3f}  {m['mrr']:.3f}")
    if best is None or m["mrr"] > best[1]:
        best = (w, m["mrr"])
print("best weight by dev MRR:", best[0], "MRR", round(best[1], 3))
