import re
import sys
import numpy as np
import scipy.sparse as sp
import json
import onnxruntime as ort
from tokenizers import Tokenizer
from src.evalutil import load, rank_of_target, metrics, bootstrap_ci

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert split in ("dev", "test")
W = 0.6


def tok(s):
    return re.findall(r"\w+", s.lower())


corpus = load("corpus")
qs = load(split)
C = np.load("data/minilm_corpus_emb.npy")
N = len(corpus)
M = sp.load_npz("data/bm25_M.npz").tocsc()
idf = np.load("data/bm25_idf.npy")
with open("data/bm25_vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

tk = Tokenizer.from_file("minilm_onnx/tokenizer.json")
tk.enable_truncation(max_length=256)
tk.enable_padding()
sess = ort.InferenceSession("minilm_onnx/model.onnx", providers=["CPUExecutionProvider"])


def bm25_scores(q):
    s = np.zeros(N, dtype=np.float32)
    for t in tok(q):
        i = vocab.get(t)
        if i is not None:
            col = M[:, i]
            s[col.indices] += idf[i] * col.data
    return s


def embed_batch(texts):
    encs = tk.encode_batch(texts)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    out = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    v = (out * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


Q = np.vstack([embed_batch([q["question"] for q in qs[i:i + 32]]) for i in range(0, len(qs), 32)])
ranks = []
for i, q in enumerate(qs):
    h = (1 - W) * z(bm25_scores(q["question"])) + W * z(C @ Q[i])
    ranks.append(rank_of_target(h, q["context_id"]))
ranks = np.array(ranks)
m = metrics(ranks)
for k, v in m.items():
    vals = (ranks <= int(k.split("@")[1])).astype(float) if k.startswith("recall") else 1.0 / ranks
    lo, hi = bootstrap_ci(vals)
    print(f"light hybrid {split} {k}: {v:.3f} ({lo:.3f}-{hi:.3f})")
