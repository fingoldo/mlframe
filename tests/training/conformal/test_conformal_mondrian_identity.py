"""CPX29 regression: calibrate_conformal_mondrian factorize+argsort block-slicing
must yield the same per-group {label: radius} dict (and None global fallback) as the
pre-optimization ``residuals[g == u]`` masked sweep, across alphas and label dtypes.

Pins the optimization that replaced the O(G*n)-per-alpha boolean-mask loop with a single
factorize + stable argsort grouping (8-37x measured; see
src/mlframe/training/composite/_benchmarks/bench_cpx29_conformal_mondrian.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import conformal as conf


class _Stub:
    """Minimal CompositeTargetEstimator surface the function reads: an
    ``estimator_`` presence flag and a ``predict`` returning fixed y_pred."""

    def __init__(self, y_pred):
        self.estimator_ = object()
        self._y_pred = np.asarray(y_pred, dtype=np.float64)

    def predict(self, X):
        """Predict."""
        return self._y_pred


def _old_mondrian(residuals, g, alphas):
    """The pre-CPX29 reference (git HEAD): per-alpha loop over sorted unique groups,
    each slicing residuals with a fresh ``g == u`` boolean mask."""
    uniq = [u for u in np.unique(g)]
    out = {}
    for a in alphas:
        af = float(a)
        global_r = conf.conformal_quantile(residuals, af)
        per_group = {None: global_r}
        for u in uniq:
            r_g = residuals[g == u]
            rad = conf.conformal_quantile(r_g, af)
            if not np.isfinite(rad) and np.isfinite(global_r):
                rad = global_r
            per_group[u] = float(rad)
        out[round(af, 6)] = per_group
    return out


def _assert_dicts_identical(old, new):
    """Assert dicts identical."""
    assert set(old.keys()) == set(new.keys())
    for k in old:
        o, n = old[k], new[k]
        assert set(o.keys()) == set(n.keys()), (k, set(o) ^ set(n))
        for gk in o:
            ov, nv = o[gk], n[gk]
            if np.isnan(ov) or np.isnan(nv):
                assert np.isnan(ov) and np.isnan(nv), (k, gk, ov, nv)
            else:
                assert ov == nv, (k, gk, ov, nv)  # bit-identical, not approx


@pytest.mark.parametrize(
    "n,n_groups,seed",
    [(20_000, 50, 0), (50_000, 500, 1), (5_000, 7, 2)],
)
def test_mondrian_block_slicing_matches_masked_sweep(n, n_groups, seed):
    """Mondrian block slicing matches masked sweep."""
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=1.0 + rng.random(n), size=n)
    groups = rng.integers(0, n_groups, size=n)
    residuals = y_true - y_pred
    alphas = [0.05, 0.1, 0.2]

    expected = _old_mondrian(residuals, groups, alphas)
    stub = _Stub(y_pred)
    conf.calibrate_conformal_mondrian(stub, None, y_true, groups, alpha=alphas)
    _assert_dicts_identical(expected, stub._mondrian_q_)


def test_mondrian_tiny_group_falls_back_to_global():
    """A singleton group cannot certify alpha=0.1 (rank ceil((1+1)*0.9)=2 > 1 -> inf),
    so its radius must take the finite pooled global radius, not stay +inf."""
    y_pred = np.zeros(11, dtype=np.float64)
    y_true = np.arange(11, dtype=np.float64)  # residuals 0..10, all finite
    groups = np.array([0] * 10 + [1], dtype=np.int64)  # group 1 is a singleton
    stub = _Stub(y_pred)
    conf.calibrate_conformal_mondrian(stub, None, y_true, groups, alpha=0.1)
    table = stub._mondrian_q_[round(0.1, 6)]
    global_r = table[None]
    assert np.isfinite(global_r)
    assert table[1] == global_r  # singleton fell back to the global radius


def test_mondrian_string_group_labels_identity():
    """Object/string group labels must group + key identically to the masked sweep."""
    rng = np.random.default_rng(3)
    n = 8_000
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(size=n)
    labels = np.array(["alpha", "beta", "gamma", "delta"])
    groups = labels[rng.integers(0, labels.size, size=n)]
    residuals = y_true - y_pred
    alphas = [0.1, 0.25]

    expected = _old_mondrian(residuals, groups, alphas)
    stub = _Stub(y_pred)
    conf.calibrate_conformal_mondrian(stub, None, y_true, groups, alpha=alphas)
    new = stub._mondrian_q_
    # The string labels must be present as dict keys (plus the None fallback).
    for k in new:
        assert set(map(str, new[k].keys() - {None})) == set(labels)
    _assert_dicts_identical(expected, new)
