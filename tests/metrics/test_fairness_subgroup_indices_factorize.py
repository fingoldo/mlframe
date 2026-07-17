"""Regression sensors for the factorize-based fast path in ``create_fairness_subgroups_indices``.

The per-bin ``np.where(bins == bin_name)[0]`` loop was O(n*B) and ran a full object-dtype element-wise comparison per categorical bin (~12 s/call at n=1M, B=200).
The fast path partitions all groups with a single ``pd.factorize`` + stable argsort. These tests pin that the fast path is taken AND that it is bit-identical to the
brute-force per-bin reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.metrics import _fairness_metrics as fm
from mlframe.metrics._fairness_metrics import create_fairness_subgroups, create_fairness_subgroups_indices


def _bruteforce_indices(bins_series: pd.Series, arr: np.ndarray) -> dict:
    """Helper: Bruteforce indices."""
    bins = bins_series.loc[arr]
    out = {}
    for bin_name in bins.unique():
        out[bin_name] = np.where((bins == bin_name).to_numpy())[0]
    return out


def test_subgroup_indices_uses_factorize_not_per_bin_compare(monkeypatch):
    """Fast path must call ``pd.factorize`` once per (split, group); the pre-fix per-bin ``np.where`` loop never called factorize, so the spy count is 0 there -> FAIL."""
    n = 5000
    rng = np.random.default_rng(0)
    feat = pd.Series([f"c{v:02d}" for v in rng.integers(0, 30, n)], name="region")
    sg = create_fairness_subgroups(pd.DataFrame({"region": feat}), ["region"], min_pop_cat_thresh=10)
    idx = np.arange(n)

    calls = {"n": 0}
    real_factorize = pd.factorize

    def spy(*a, **k):
        """Spy."""
        calls["n"] += 1
        return real_factorize(*a, **k)

    monkeypatch.setattr(fm.pd, "factorize", spy)
    create_fairness_subgroups_indices(sg, idx, idx[: n // 2], idx[n // 2 :])
    # 1 categorical group x 3 splits.
    assert calls["n"] >= 3, f"expected factorize fast path (>=3 calls), got {calls['n']}"


def test_subgroup_indices_bit_identical_to_per_bin_reference():
    """Subgroup indices bit identical to per bin reference."""
    n = 5000
    rng = np.random.default_rng(1)
    feat = pd.Series([f"c{v:02d}" for v in rng.integers(0, 30, n)], name="region")
    sg = create_fairness_subgroups(pd.DataFrame({"region": feat}), ["region"], min_pop_cat_thresh=10)
    idx = np.arange(n)
    a, b, c = idx, idx[: int(n * 0.6)], idx[int(n * 0.6) :]
    res = create_fairness_subgroups_indices(sg, a, b, c)

    for arr in (a, c, b):
        ref = _bruteforce_indices(sg["region"]["bins"], arr)
        got = res[len(arr)]["region"]["bins"]
        assert list(got.keys()) == list(ref.keys())
        for k in ref:
            assert np.array_equal(got[k], ref[k]), k


def test_subgroup_indices_numeric_qcut_bit_identical():
    """Subgroup indices numeric qcut bit identical."""
    n = 4000
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"age": rng.integers(0, 100, n)})
    sg = create_fairness_subgroups(df, ["age"], cont_nbins=4, min_pop_cat_thresh=10)
    idx = np.arange(n)
    a, b, c = idx, idx[: int(n * 0.6)], idx[int(n * 0.6) :]
    res = create_fairness_subgroups_indices(sg, a, b, c)
    for arr in (a, c, b):
        ref = _bruteforce_indices(sg["age"]["bins"], arr)
        got = res[len(arr)]["age"]["bins"]
        assert list(got.keys()) == list(ref.keys())
        for k in ref:
            assert np.array_equal(got[k], ref[k]), k
