"""Paired bootstrap on the test set (same resampled questions for both methods).
Weights are the dev-tuned values printed by src.fusion_bge and src.hybrid_dev.
Run: python -m src.paired_bootstrap
"""
import numpy as np
from src.evalutil import load
from src.fusion_bge import z, ranks_from_scores


def main():
    qs = load("test")
    ids = np.array([q["context_id"] for q in qs])
    Bz = z(np.load("data/bm25_test_scores.npy"))
    Mz = z(np.load("data/minilm_test_scores.npy"))
    Gz = z(np.load("data/bge_test_scores.npy"))

    r = {
        "BM25": ranks_from_scores(Bz, ids),
        "Hyb MiniLM": ranks_from_scores(0.4 * Bz + 0.6 * Mz, ids),
        "Hyb BGE": ranks_from_scores(0.4 * Bz + 0.6 * Gz, ids),
        "Hyb 3-way": ranks_from_scores(0.3 * Bz + 0.3 * Mz + 0.4 * Gz, ids),
    }
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), (10000, len(ids)))

    def diff(a, b, fn):
        va, vb = fn(r[a]), fn(r[b])
        d = va[idx].mean(1) - vb[idx].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {a} minus {b}: {va.mean() - vb.mean():+.4f} (95% interval {lo:+.4f} to {hi:+.4f})")

    for name, fn in (("MRR", lambda x: 1.0 / x), ("recall@1", lambda x: (x <= 1).astype(float))):
        print(name)
        diff("Hyb BGE", "Hyb MiniLM", fn)
        diff("Hyb 3-way", "Hyb BGE", fn)
        diff("Hyb BGE", "BM25", fn)


if __name__ == "__main__":
    main()
