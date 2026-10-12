import json, os, re
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

SEED, N_BOOT = 42, 10000
os.makedirs("data", exist_ok=True)
REPO = "mteb/scifact"
corpus_ds = load_dataset(REPO, "corpus")
queries_ds = load_dataset(REPO, "queries")
qrels_ds = load_dataset(REPO)
first = lambda d: d[list(d.keys())[0]]
print("qrels splits:", list(qrels_ds.keys()), "| columns:", qrels_ds[list(qrels_ds.keys())[0]].column_names)

corpus = [{"_id": str(r["_id"]), "title": r.get("title") or "", "text": r["text"]} for r in first(corpus_ds)]
queries = {str(r["_id"]): r["text"] for r in first(queries_ds)}

def qrels(split):
    out = {}
    for r in qrels_ds[split]:
        if r["score"] > 0:
            out.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    return out

doc_ids = [r["_id"] for r in corpus]
docs = [(r.get("title", "") + ". " + r["text"]).strip() for r in corpus]
dev_rel, test_rel = qrels("train"), qrels("test")
dev_q = [q for q in dev_rel if q in queries]
test_q = [q for q in test_rel if q in queries]
print("docs", len(docs), "| dev queries", len(dev_q), "| test queries", len(test_q))

tok = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok(d) for d in docs])
model = SentenceTransformer("all-MiniLM-L6-v2")
print("embedding corpus ...")
E = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

def scores(qids):
    B = np.array([bm25.get_scores(tok(queries[q])) for q in qids])
    Q = model.encode([queries[q] for q in qids], batch_size=64, normalize_embeddings=True)
    return B, Q @ E.T
def zs(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)

idx = {d: i for i, d in enumerate(doc_ids)}
def per_query(S, qids, rel):
    r1, r10, rr, nd = [], [], [], []
    for i, q in enumerate(qids):
        order = np.argsort(-S[i])
        rel_idx = {idx[d] for d in rel[q] if d in idx}
        hits = [k for k, j in enumerate(order[:len(order)]) if j in rel_idx]
        first = hits[0] + 1
        r1.append(first <= 1); r10.append(first <= 10); rr.append(1 / first)
        gains = [1.0 if j in rel_idx else 0.0 for j in order[:10]]
        dcg = sum(g / np.log2(k + 2) for k, g in enumerate(gains))
        ideal = sum(1 / np.log2(k + 2) for k in range(min(len(rel_idx), 10)))
        nd.append(dcg / ideal)
    return {"recall@1": np.array(r1, float), "recall@10": np.array(r10, float),
            "mrr": np.array(rr), "ndcg@10": np.array(nd)}

print("scoring dev ...")
Bd, Dd = scores(dev_q)
best = None
for w in [round(x * 0.1, 1) for x in range(11)]:
    m = per_query((1 - w) * zs(Bd) + w * zs(Dd), dev_q, dev_rel)["mrr"].mean()
    print(f"  dev w={w}: MRR {m:.3f}")
    if best is None or m > best[1]:
        best = (w, m)
w = best[0]
print("chosen w:", w)

print("scoring test (once) ...")
Bt, Dt = scores(test_q)
res = {"BM25": per_query(Bt, test_q, test_rel),
       "MiniLM": per_query(Dt, test_q, test_rel),
       f"Hybrid (w={w})": per_query((1 - w) * zs(Bt) + w * zs(Dt), test_q, test_rel)}
rng = np.random.default_rng(SEED)
boot = rng.integers(0, len(test_q), size=(N_BOOT, len(test_q)))
print("\nmethod | recall@1 | recall@10 | MRR | nDCG@10 (mean, 95% bootstrap CI)")
for name, r in res.items():
    cells = []
    for k in ["recall@1", "recall@10", "mrr", "ndcg@10"]:
        bm = r[k][boot].mean(1)
        cells.append(f"{r[k].mean():.3f} [{np.percentile(bm, 2.5):.3f},{np.percentile(bm, 97.5):.3f}]")
    print(name, "|", " | ".join(cells))
hy = res[f"Hybrid (w={w})"]
for other in ["BM25", "MiniLM"]:
    for k in ["recall@1", "ndcg@10"]:
        d = (hy[k] - res[other][k])[boot].mean(1)
        print(f"hybrid minus {other}, {k}: {(hy[k]-res[other][k]).mean():+.3f} (95% CI {np.percentile(d,2.5):+.3f} to {np.percentile(d,97.5):+.3f})")
bm_m = (res["BM25"]["mrr"] - res["MiniLM"]["mrr"])
print("BM25 minus MiniLM, MRR:", f"{bm_m.mean():+.3f}")
json.dump({"w": w, "n_test": len(test_q), "n_dev": len(dev_q),
           "test": {n: {k: float(v.mean()) for k, v in r.items()} for n, r in res.items()}},
          open("data/scifact_results.json", "w"), indent=1)
