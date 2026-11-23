"""The total-correlation scorer holds its candidate bins as bytes and scores them as before (PLAN-10: int64 codes were the largest holder of a 500000-row fit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_total_correlation_fe as T


def test_bins_are_bytes_and_equal_the_int64_bins():
    """Same codes at one eighth of the bytes."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((4000, 5)), columns=list("abcde"))
    narrow = T._bin_dataframe_batched(df, 10)
    wide = T._quantile_bin_batched(df.to_numpy(), 10)
    assert all(b.dtype == np.uint8 for b in narrow)
    for j, b in enumerate(narrow):
        np.testing.assert_array_equal(b.astype(np.int64), wide[:, j])


def test_the_scores_do_not_change_with_the_code_dtype(monkeypatch):
    """The ranking table of the scorer is the same whether the bins are bytes or int64."""
    rng = np.random.default_rng(1)
    n = 3000
    raw = pd.DataFrame(rng.standard_normal((n, 3)), columns=["x0", "x1", "x2"])
    eng = pd.DataFrame({"x0__He2": raw["x0"] ** 2, "x1__He2": raw["x1"] ** 2, "x2__He3": raw["x2"] ** 3})
    y = (raw["x0"] ** 2 + 0.5 * rng.standard_normal(n) > 1).astype(np.int64)
    got = T.score_features_by_tc_uplift(raw, eng, y, n_bins=10)
    real = T._quantile_bin_batched
    monkeypatch.setattr(T, "_quantile_bin_batched", lambda a, nbins, dtype=np.int64: real(a, nbins, dtype=np.int64))
    ref = T.score_features_by_tc_uplift(raw, eng, y, n_bins=10)
    pd.testing.assert_frame_equal(got, ref)
