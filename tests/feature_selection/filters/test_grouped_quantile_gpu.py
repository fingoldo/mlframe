"""The batched grouped-quantile statistics (numba form, CUDA form) equal the per-group host loop bit for bit, and the generated features are unchanged by them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _grouped_quantile_fe as GQ
from mlframe.feature_selection.filters import _grouped_quantile_gpu as G


def _block(n=6000, k=6, n_groups=7):
    """Float32 columns with ties, signed zeros and a tiny group, and the group codes."""
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((k, n)).astype(np.float32)
    xs[1] = np.round(xs[1], 1)
    xs[2] = rng.integers(0, 3, n).astype(np.float32)
    xs[3][::5] = -0.0
    xs[3][1::5] = 0.0
    codes = rng.integers(0, n_groups, size=n).astype(np.int64)
    codes[codes == n_groups - 1] = n_groups - 2
    codes[:3] = n_groups - 1
    return xs, codes, n_groups


def test_the_two_forms_agree_exactly():
    """Ranks, sorted segments and both spreads are equal between the numba form and the CUDA form."""
    pytest.importorskip("cupy")
    xs, codes, n_groups = _block()
    host = G._stats_host(xs, codes, n_groups)
    dev = G._stats_device(xs, codes, n_groups)
    if dev is None:
        pytest.skip("no usable device")
    for key in ("pct", "sorted", "iqr", "p90p10"):
        assert np.array_equal(host[key], dev[key]), key


def _frame(n=4000, with_nan=False):
    """Two group columns, five numeric columns (one with ties), optionally one with a missing value."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"g1": rng.integers(0, 6, n), "g2": rng.integers(0, 3, n)})
    for j in range(5):
        X[f"x{j}"] = rng.standard_normal(n).astype(np.float32)
    X["x1"] = X["x1"].round(1)
    if with_nan:
        X.loc[::50, "x2"] = np.nan
    return X


@pytest.mark.parametrize("forced", ["0", "1"])
@pytest.mark.parametrize("with_nan", [False, True])
def test_generated_features_do_not_change(monkeypatch, forced, with_nan):
    """The engineered columns and the recipe payloads (group lookups and sorted segments) equal the per-group host loop, on either backend, also with a non-finite column in the frame."""
    X = _frame(with_nan=with_nan)
    cols = [f"x{j}" for j in range(5)]
    monkeypatch.setattr(GQ, "_device_group_stats", lambda *a, **k: {})
    want_df, want_rec = GQ.generate_grouped_quantile_features(X, ["g1", "g2"], cols)
    monkeypatch.undo()
    monkeypatch.setenv("MLFRAME_GROUPED_QUANTILE_DEVICE", forced)
    got_df, got_rec = GQ.generate_grouped_quantile_features(X, ["g1", "g2"], cols)
    assert list(got_df.columns) == list(want_df.columns)
    for c in want_df.columns:
        assert np.array_equal(got_df[c].to_numpy(), want_df[c].to_numpy(), equal_nan=True), c
    assert set(got_rec) == set(want_rec)
    assert want_rec
    for name, payload in want_rec.items():
        assert payload["iqr_lookup"] == got_rec[name]["iqr_lookup"] and payload["p90p10_lookup"] == got_rec[name]["p90p10_lookup"], name
        if payload["op"] != "pct_rank":
            assert payload["group_sorted"] == {}, name  # the spread recipes carry the lookups only
            continue
        assert payload["group_sorted"], name
        for key, sv in payload["group_sorted"].items():
            assert np.array_equal(np.asarray(sv), np.asarray(got_rec[name]["group_sorted"][key])), (name, key)
