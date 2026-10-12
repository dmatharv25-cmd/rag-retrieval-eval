"""Cross-encoder rerank of the top 20 from the BGE-small hybrid (w=0.6, dev-tuned).
No parameter is tuned here. Run dev first as a sanity check, then test once.
Run: python -m src.rerank dev    then    python -m src.rerank test
Scores are cached in data/rerank_<split>_*.npy.
"""
import os
import sys
import numpy as np
from sentence_transformers import CrossEncoder
from src.evalutil import load, metrics, bootstrap_ci
from src.fusion_bge import z, ranks_from_scores

TOP = 20
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main(split):
    qs = load(split)
    corpus = load("corpus")
    ids = np.array([q["context_id"] for q in qs])
    H = 0.4 * z(np.load(f"data/bm25_{split}_scores.npy")) + 0.6 * z(np.load(f"data/bge_{split}_scores.npy"))
    hyb = ranks_from_scores(H, ids)
    cand_path, ce_path = f"data/rerank_{split}_cands.npy", f"data/rerank_{split}_scores.npy"

    if os.path.exists(ce_path):
        cands, CE = np.load(cand_path), np.load(ce_path)
        print("loaded cached rerank scores:", CE.shape)
    else:
        cands = np.argsort(-H, axis=1, kind="stable")[:, :TOP]
        model = CrossEncoder(MODEL, device="cpu", max_length=256)
        pairs = [(q["question"], corpus[c]) for q, row in zip(qs, cands) for c in row]
        CE = model.predict(pairs, batch_size=32, show_progress_bar=True).reshape(len(qs), TOP)
        np.save(cand_path, cands)
        np.save(ce_path, CE.astype(np.float32))

    rr = hyb.copy()
    for i in range(len(ids)):
        pos = np.where(cands[i] == ids[i])[0]
        if len(pos):
            rr[i] = int((CE[i] > CE[i][pos[0]]).sum()) + 1

    print(f"{split}: target inside hybrid top {TOP}: {(hyb <= TOP).mean():.3f} (upper bound for recall@1)")
    for name, ranks in (("Hybrid (w=0.6 on BGE-small)", hyb), (f"Hybrid + rerank top {TOP}", rr)):
        m = metrics(ranks)
        parts = []
        for k, v in m.items():
            vals = (ranks <= int(k.split("@")[1])).astype(float) if k.startswith("recall") else 1.0 / ranks
            lo, hi = bootstrap_ci(vals)
            parts.append(f"{k} {v:.3f} ({lo:.3f} to {hi:.3f})")
        print(f"{name}: " + " | ".join(parts))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), (10000, len(ids)))
    for name, fn in (("recall@1", lambda x: (x <= 1).astype(float)), ("MRR", lambda x: 1.0 / x)):
        va, vb = fn(rr), fn(hyb)
        d = va[idx].mean(1) - vb[idx].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  rerank minus hybrid, {name}: {va.mean() - vb.mean():+.4f} (95% paired interval {lo:+.4f} to {hi:+.4f})")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("dev", "test"):
        raise SystemExit("usage: python -m src.rerank dev|test")
    main(sys.argv[1])
