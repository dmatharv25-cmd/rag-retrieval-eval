"""95% bootstrap intervals for every full-pipeline row of the SQuAD results table (test set).
Reads existing score files only, no model is loaded. Weights are the dev-tuned values
from src.fusion_bge and src.hybrid_dev. The light (ONNX) row is covered by src.light_eval.
Run: python -m src.row_intervals
"""
import numpy as np
from src.evalutil import load, metrics, bootstrap_ci
from src.fusion_bge import z, rank_matrix, ranks_from_scores, rrf_ranks


def main():
    qs = load("test")
    ids = np.array([q["context_id"] for q in qs])
    B = np.load("data/bm25_test_scores.npy")
    M = np.load("data/minilm_test_scores.npy")
    G = np.load("data/bge_test_scores.npy")
    Bz, Mz, Gz = z(B), z(M), z(G)
    RB, RM, RG = rank_matrix(B), rank_matrix(M), rank_matrix(G)

    rows = {
        "BM25": ranks_from_scores(B, ids),
        "MiniLM": ranks_from_scores(M, ids),
        "BGE-small": ranks_from_scores(G, ids),
        "Hybrid (w=0.6 on MiniLM)": ranks_from_scores(0.4 * Bz + 0.6 * Mz, ids),
        "RRF (k=0, BM25 + MiniLM)": rrf_ranks([RB, RM], ids, k=0),
        "Hybrid (w=0.6 on BGE-small)": ranks_from_scores(0.4 * Bz + 0.6 * Gz, ids),
        "RRF (k=0, BM25 + BGE-small)": rrf_ranks([RB, RG], ids, k=0),
        "Hybrid 3-way (0.3/0.3/0.4)": ranks_from_scores(0.3 * Bz + 0.3 * Mz + 0.4 * Gz, ids),
    }
    for name, ranks in rows.items():
        m = metrics(ranks)
        parts = []
        for k, v in m.items():
            vals = (ranks <= int(k.split("@")[1])).astype(float) if k.startswith("recall") else 1.0 / ranks
            lo, hi = bootstrap_ci(vals)
            parts.append(f"{k} {v:.3f} ({lo:.3f} to {hi:.3f})")
        print(f"{name}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
