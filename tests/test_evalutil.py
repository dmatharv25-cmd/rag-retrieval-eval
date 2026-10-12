import numpy as np
from src.evalutil import rank_of_target, metrics, bootstrap_ci, zscore


def test_rank_of_target_basic():
    s = np.array([0.1, 0.9, 0.5])
    assert rank_of_target(s, 1) == 1
    assert rank_of_target(s, 2) == 2
    assert rank_of_target(s, 0) == 3


def test_rank_of_target_ties_currently_optimistic():
    # Documents current behaviour: only strictly greater scores count,
    # so a tied target gets the best rank of its tie group.
    s = np.array([1.0, 1.0, 0.0])
    assert rank_of_target(s, 0) == 1
    assert rank_of_target(s, 1) == 1


def test_metrics_known_values():
    m = metrics([1, 2, 5, 11])
    assert m["recall@1"] == 0.25
    assert m["recall@5"] == 0.75
    assert m["recall@10"] == 0.75
    assert abs(m["mrr"] - (1 + 0.5 + 0.2 + 1 / 11) / 4) < 1e-12


def test_bootstrap_ci_constant_values():
    lo, hi = bootstrap_ci([0.7] * 50)
    assert abs(lo - 0.7) < 1e-12 and abs(hi - 0.7) < 1e-12


def test_bootstrap_ci_deterministic_and_ordered():
    vals = np.random.default_rng(0).integers(0, 2, 200)
    a = bootstrap_ci(vals, seed=3)
    b = bootstrap_ci(vals, seed=3)
    assert (a == b).all()
    assert a[0] <= vals.mean() <= a[1]


def test_zscore_rows_have_zero_mean_unit_std():
    S = np.random.default_rng(1).normal(5, 3, size=(4, 50))
    Z = zscore(S)
    assert np.allclose(Z.mean(1), 0, atol=1e-9)
    assert np.allclose(Z.std(1), 1, atol=1e-6)


def test_zscore_constant_row_is_zero_not_nan():
    Z = zscore(np.ones((1, 10)))
    assert np.isfinite(Z).all() and np.allclose(Z, 0)


def test_fusion_endpoints_match_single_methods():
    rng = np.random.default_rng(2)
    B, D = rng.normal(size=(3, 30)), rng.normal(size=(3, 30))
    w0 = 1.0 * zscore(B) + 0.0 * zscore(D)
    w1 = 0.0 * zscore(B) + 1.0 * zscore(D)
    assert (np.argsort(-w0, 1) == np.argsort(-B, 1)).all()
    assert (np.argsort(-w1, 1) == np.argsort(-D, 1)).all()
