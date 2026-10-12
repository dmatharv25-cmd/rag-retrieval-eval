import json
import numpy as np


def load(name):
    with open(f"data/{name}.json", encoding="utf-8") as f:
        return json.load(f)


def rank_of_target(scores, target):
    """1-based rank of the target paragraph (ties counted pessimistically)."""
    return int((scores > scores[target]).sum()) + 1


def metrics(ranks):
    ranks = np.asarray(ranks)
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in (1, 5, 10)}
    out["mrr"] = float((1.0 / ranks).mean())
    return out


def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = [values[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def zscore(S):
    """Per-row z-score, same formula as final_test.py and hybrid_dev.py."""
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
