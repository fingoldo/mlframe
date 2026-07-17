"""PROP1: accuracy_ratio must be row-permutation-invariant on tied scores and
satisfy its documented identity AR == 2*AUC-1 on both tied and distinct data.

Pre-fix the naive cumsum over -score argsort made the CAP area depend on the
arbitrary intra-tie row order, breaking both properties on tie-heavy inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import accuracy_ratio, fast_roc_auc


def _tie_heavy_data(seed: int, n: int = 200):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    # Few distinct score levels -> many ties.
    y_score = rng.integers(0, 4, size=n).astype(np.float64)
    return y_true, y_score


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13, 42])
def test_accuracy_ratio_invariant_under_row_permutation_on_ties(seed):
    y_true, y_score = _tie_heavy_data(seed)
    base = accuracy_ratio(y_true, y_score)
    vals = set()
    rng = np.random.default_rng(1000 + seed)
    for _ in range(20):
        perm = rng.permutation(len(y_true))
        vals.add(round(accuracy_ratio(y_true[perm], y_score[perm]), 12))
    assert len(vals) == 1, f"AR not permutation-invariant on ties: {sorted(vals)}"
    assert abs(base - next(iter(vals))) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13, 42])
def test_accuracy_ratio_equals_2auc_minus_1_on_ties(seed):
    y_true, y_score = _tie_heavy_data(seed)
    ar = accuracy_ratio(y_true, y_score)
    auc = fast_roc_auc(y_true, y_score)
    assert abs(ar - (2.0 * auc - 1.0)) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13, 42])
def test_accuracy_ratio_equals_2auc_minus_1_on_distinct(seed):
    rng = np.random.default_rng(seed)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.standard_normal(n)  # continuous -> no ties
    ar = accuracy_ratio(y_true, y_score)
    auc = fast_roc_auc(y_true, y_score)
    assert abs(ar - (2.0 * auc - 1.0)) < 1e-9


def test_accuracy_ratio_vectorised_tiefold_matches_reference_loop():
    """The tie-fold (spread each equal-score block's TP contribution to its block mean) is vectorised via reduceat over
    sorted-score block starts instead of a per-row Python loop (56x at n=60k). Regression sensor: the vectorised fold
    must be bit-identical to the explicit reference loop across tie-heavy and distinct inputs, and AR == 2*AUC-1 must
    still hold to fp tolerance."""

    def _ref_ar(yt, ys):
        yt = np.asarray(yt).astype(np.int64)
        ys = np.asarray(ys, dtype=np.float64)
        n = yt.shape[0]
        n_pos = int(yt.sum())
        if n_pos == 0 or n_pos == n:
            return float("nan")
        order = np.argsort(-ys, kind="stable")
        yt_s = yt[order].astype(np.float64)
        ys_s = ys[order]
        bs = 0
        for k in range(1, n + 1):
            if k == n or ys_s[k] != ys_s[bs]:
                m = k - bs
                if m > 1:
                    yt_s[bs:k] = yt_s[bs:k].sum() / m
                bs = k
        cum_tp = np.concatenate(([0.0], np.cumsum(yt_s) / n_pos))
        cum_pop = np.concatenate(([0.0], np.arange(1, n + 1, dtype=np.float64) / n))
        area = float(np.sum((cum_pop[1:] - cum_pop[:-1]) * (cum_tp[1:] + cum_tp[:-1]) * 0.5))
        return (area - 0.5) / ((1.0 - n_pos / (2.0 * n)) - 0.5)

    for s in range(60):
        r = np.random.default_rng(s)
        n = int(r.integers(6, 3000))
        ys = r.choice(np.round(r.uniform(0, 1, max(2, n // 8)), 2), n) if s % 2 else r.uniform(0, 1, n)
        yt = r.integers(0, 2, n)
        if yt.sum() in (0, n):
            continue
        got = accuracy_ratio(yt, ys)
        ref = _ref_ar(yt, ys)
        assert abs(got - ref) < 1e-12, (s, got, ref)
        assert abs(got - (2.0 * float(fast_roc_auc(yt, ys.astype(np.float64))) - 1.0)) < 1e-9
