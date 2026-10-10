import numpy as np
from src.evalutil import load, rank_of_target, metrics, bootstrap_ci

W = 0.6
qs = load("test")
B = np.load("data/bm25_test_scores.npy")
D = np.load("data/minilm_test_scores.npy")


def z(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


H = (1 - W) * z(B) + W * z(D)
ranks = {
    "bm25": np.load("data/bm25_test_ranks.npy"),
    "minilm": np.load("data/minilm_test_ranks.npy"),
    "hybrid": np.array([rank_of_target(H[i], q["context_id"]) for i, q in enumerate(qs)]),
}
np.save("data/hybrid_test_ranks.npy", ranks["hybrid"])

for name, r in ranks.items():
    m = metrics(r)
    parts = []
    for k, v in m.items():
        vals = (r <= int(k.split("@")[1])).astype(float) if k.startswith("recall") else 1.0 / r
        lo, hi = bootstrap_ci(vals)
        parts.append(f"{k} {v:.3f} ({lo:.3f}-{hi:.3f})")
    print(f"{name}: " + " | ".join(parts))

rng = np.random.default_rng(1)
n = len(qs)
for a, b in [("hybrid", "bm25"), ("bm25", "minilm")]:
    ra, rb = ranks[a], ranks[b]
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        diffs.append((ra[i] == 1).mean() - (rb[i] == 1).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"recall@1 {a} minus {b}: {(ra == 1).mean() - (rb == 1).mean():+.3f} (95% interval {lo:+.3f} to {hi:+.3f})")
