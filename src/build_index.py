import json
import re
import numpy as np
import scipy.sparse as sp
from collections import Counter

K1, B = 1.5, 0.75


def tok(s):
    return re.findall(r"\w+", s.lower())


with open("data/corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)
N = len(corpus)
docs = [tok(c) for c in corpus]
dl = np.array([len(d) for d in docs], dtype=np.float32)
avg = dl.mean()
df = Counter()
for d in docs:
    df.update(set(d))
vocab = {t: i for i, t in enumerate(df)}
idf = np.zeros(len(vocab), dtype=np.float32)
for t, i in vocab.items():
    idf[i] = np.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
rows, cols, vals = [], [], []
for r, d in enumerate(docs):
    for t, f_ in Counter(d).items():
        rows.append(r)
        cols.append(vocab[t])
        vals.append(f_ * (K1 + 1) / (f_ + K1 * (1 - B + B * dl[r] / avg)))
M = sp.csr_matrix((vals, (rows, cols)), shape=(N, len(vocab)), dtype=np.float32).tocsc()
sp.save_npz("data/bm25_M.npz", M)
np.save("data/bm25_idf.npy", idf)
with open("data/bm25_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f)
print("saved index:", M.shape, "nonzeros:", M.nnz)
