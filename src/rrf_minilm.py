"""BM25 + MiniLM reciprocal rank fusion, plus the BM25 tie check.
k is tuned on dev by MRR, test is scored once.
Run: python -m src.rrf_minilm
RRF ties are broken by BM25 rank (same rule as fusion_bge.py).
"""
import numpy as np
from src.evalutil import load, metrics
from src.fusion_bge import rank_matrix, rrf_ranks

K_GRID = [0, 1, 2, 3, 5, 10, 20, 40, 60, 100]


def load_split(split):
    qs = load(split)
    ids = np.array([q["context_id"] for q in qs])
    B = np.load(f"data/bm25_{split}_scores.npy")
    M = np.load(f"data/minilm_{split}_scores.npy")
    return ids, B, M


def tie_check(B, ids):
    t = B[np.arange(len(ids)), ids][:, None]
    optimistic = (B > t).sum(1) + 1
    pessimistic = (B >= t).sum(1)  # counts the target itself, so this is the worst rank in its tie group
    tied = int((pessimistic != optimistic).sum())
    print(f"BM25 test questions where the target is tied with another paragraph: {tied} of {len(ids)}")
    for name, r in (("optimistic", optimistic), ("pessimistic", pessimistic)):
        m = metrics(r)
        print(f"  {name}: r@1 {m['recall@1']:.3f}  r@5 {m['recall@5']:.3f}  r@10 {m['recall@10']:.3f}  mrr {m['mrr']:.3f}")


def main():
    ids, B, M = load_split("dev")
    RB, RM = rank_matrix(B), rank_matrix(M)
    print("RRF (BM25 + MiniLM): k -> recall@1, recall@10, mrr (dev)")
    best = None
    for k in K_GRID:
        m = metrics(rrf_ranks([RB, RM], ids, k=k))
        print(f"  k={k}: {m['recall@1']:.3f}  {m['recall@10']:.3f}  {m['mrr']:.3f}")
        if best is None or m["mrr"] > best[1]:
            best = (k, m["mrr"])
    k = best[0]
    print("best k by dev MRR:", k)

    ids, B, M = load_split("test")
    RB, RM = rank_matrix(B), rank_matrix(M)
    m = metrics(rrf_ranks([RB, RM], ids, k=k))
    print("TEST (one run, k fixed from dev)")
    print(f"| RRF (k={k}, BM25 + MiniLM, ties by BM25 rank) | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} |")
    tie_check(B, ids)


if __name__ == "__main__":
    main()
