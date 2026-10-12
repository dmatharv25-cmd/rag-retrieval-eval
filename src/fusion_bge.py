"""Fusion of BM25, MiniLM and BGE-small. Parameters are tuned on dev, test is scored once.
Run: python -m src.fusion_bge
RRF ties are broken by BM25 rank; z-score rows count only strictly higher scores.
"""
import numpy as np
from src.evalutil import load, metrics


def z(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def rank_matrix(S):
    R = np.empty(S.shape, dtype=np.int32)
    ar = np.arange(1, S.shape[1] + 1, dtype=np.int32)
    for i in range(S.shape[0]):
        R[i, np.argsort(-S[i], kind="stable")] = ar
    return R


def ranks_from_scores(S, ids, chunk=250):
    out = []
    for a in range(0, len(ids), chunk):
        s = S[a:a + chunk]
        t = s[np.arange(s.shape[0]), ids[a:a + chunk]][:, None]
        out.append((s > t).sum(1) + 1)
    return np.concatenate(out)


def rrf_ranks(Rs, ids, k=0, chunk=250):
    out = []
    for a in range(0, len(ids), chunk):
        sl = slice(a, a + chunk)
        idx = np.arange(Rs[0][sl].shape[0])
        S = sum(1.0 / (k + R[sl]) for R in Rs)
        t = S[idx, ids[sl]][:, None]
        tb = Rs[0][sl][idx, ids[sl]][:, None]
        out.append((S > t).sum(1) + ((S == t) & (Rs[0][sl] < tb)).sum(1) + 1)
    return np.concatenate(out)


def load_split(split):
    qs = load(split)
    ids = np.array([q["context_id"] for q in qs])
    B = np.load(f"data/bm25_{split}_scores.npy")
    M = np.load(f"data/minilm_{split}_scores.npy")
    G = np.load(f"data/bge_{split}_scores.npy")
    return ids, B, M, G


def main():
    ids, B, M, G = load_split("dev")
    Bz, Mz, Gz = z(B), z(M), z(G)

    print("BM25 + BGE: weight on BGE -> recall@1, recall@10, mrr (dev)")
    best2 = None
    for w in [round(x * 0.1, 1) for x in range(11)]:
        m = metrics(ranks_from_scores((1 - w) * Bz + w * Gz, ids))
        print(f"  w={w}: {m['recall@1']:.3f}  {m['recall@10']:.3f}  {m['mrr']:.3f}")
        if best2 is None or m["mrr"] > best2[1]:
            best2 = (w, m["mrr"])
    w2 = best2[0]
    print("best w (BM25+BGE) by dev MRR:", w2)

    best3 = None
    for wm in range(0, 11):
        for wg in range(0, 11 - wm):
            wb = 10 - wm - wg
            m = metrics(ranks_from_scores((wb * Bz + wm * Mz + wg * Gz) / 10, ids))
            if best3 is None or m["mrr"] > best3[3]:
                best3 = (wb / 10, wm / 10, wg / 10, m["mrr"])
    wb3, wm3, wg3 = best3[:3]
    print(f"best 3-way weights by dev MRR: BM25={wb3} MiniLM={wm3} BGE={wg3} (dev MRR {best3[3]:.3f})")

    ids, B, M, G = load_split("test")
    Bz, Mz, Gz = z(B), z(M), z(G)
    RB, RG = rank_matrix(B), rank_matrix(G)

    def row(name, ranks):
        m = metrics(ranks)
        print(f"| {name} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} |")

    print("TEST (one run, parameters fixed from dev)")
    row("BGE (bge-small-en-v1.5)", ranks_from_scores(G, ids))
    row(f"Hybrid (w={w2} on BGE)", ranks_from_scores((1 - w2) * Bz + w2 * Gz, ids))
    row("RRF (k=0, BM25 + BGE, ties by BM25 rank)", rrf_ranks([RB, RG], ids, k=0))
    row(f"Hybrid 3-way (BM25 {wb3}, MiniLM {wm3}, BGE {wg3})", ranks_from_scores(wb3 * Bz + wm3 * Mz + wg3 * Gz, ids))


if __name__ == "__main__":
    main()
