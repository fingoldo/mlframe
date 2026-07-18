"""Regression sensor for the per-column NaN-accumulate fast path in
``missingness_count_fit`` / ``apply_missingness_count`` (perf iter121).

The fit/apply count must stay bit-identical to the pandas reference
``X.loc[:, cols].isna().sum(axis=1)`` while avoiding the slow row-wise
``.sum(axis=1)`` reduction. The spy test pins that the helper does NOT
route through ``DataFrame.sum`` (the pre-fix hotspot) -- a future revert to
the row-sum path trips it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _missingness_fe as mf


def _frame():
    """Helper that frame."""
    rng = np.random.default_rng(7)
    n = 4000
    cols = {}
    for j in range(5):
        a = rng.standard_normal(n)
        a[rng.random(n) < 0.15] = np.nan
        cols[f"c{j}"] = a
    o = np.array(["x"] * n, dtype=object)
    o[rng.random(n) < 0.2] = None
    cols["obj"] = o
    return pd.DataFrame(cols), list(cols)


def test_count_helper_bit_identical_to_pandas_row_sum():
    """Count helper bit identical to pandas row sum."""
    X, allcols = _frame()
    ref = X.loc[:, allcols].isna().sum(axis=1).to_numpy().astype(np.int32)
    counts, _recipe = mf.missingness_count_fit(X, allcols)
    assert counts.dtype == np.int32
    assert np.array_equal(counts, ref)
    applied = mf.apply_missingness_count(X, {"cols": tuple(allcols)})
    assert np.array_equal(applied, ref)


def test_count_helper_does_not_use_row_sum(monkeypatch):
    """Count helper does not use row sum."""
    X, allcols = _frame()

    calls = {"n": 0}
    orig_sum = pd.DataFrame.sum

    def _spy(self, *a, **k):
        """Helper that spy."""
        calls["n"] += 1
        return orig_sum(self, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "sum", _spy)
    mf.missingness_count_fit(X, allcols)
    mf.apply_missingness_count(X, {"cols": tuple(allcols)})
    assert calls["n"] == 0, "count path must not route through DataFrame.sum (the pre-fix row-wise hotspot)"


def test_count_helper_subset_and_missing_cols():
    """Count helper subset and missing cols."""
    X, allcols = _frame()
    sub = allcols[:3]
    ref = X.loc[:, sub].isna().sum(axis=1).to_numpy().astype(np.int32)
    assert np.array_equal(mf.apply_missingness_count(X, {"cols": tuple(sub)}), ref)
    # A column absent at test time contributes 0 (graceful schema-drift contract).
    drifted = X.drop(columns=[allcols[0]])
    ref_present = drifted.loc[:, allcols[1:3]].isna().sum(axis=1).to_numpy().astype(np.int32)
    got = mf.apply_missingness_count(drifted, {"cols": tuple(allcols[:3])})
    assert np.array_equal(got, ref_present)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
