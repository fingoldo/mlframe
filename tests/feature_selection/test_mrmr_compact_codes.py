"""Compact codes-matrix storage: the categorized ``factors_data`` holds per-column BIN INDICES, so it is downcast to
int8/int16 (from int32) to cut the big (n, p) matrix 4x/2x. Joint codes stay int32 (deep-joint overflow), and every
consumer casts the storage UP -- so this is SELECTION-EQUIVALENT. These tests pin the equivalence + the narrowing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _fit_selection(X, y, compact: str, monkeypatch, **kw):
    monkeypatch.setenv("MLFRAME_MRMR_COMPACT_CODES", compact)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=1, verbose=0, **kw)
        m.fit(X, y)
    return list(m.get_feature_names_out())


@pytest.mark.parametrize("fe_max_steps", [0, 1])
def test_compact_codes_selection_equivalent(monkeypatch, fe_max_steps):
    """Selection with compact codes ON must equal OFF -- including when FE engineers + appends columns (the append
    path must preserve the narrow dtype without changing which features are picked)."""
    rng = np.random.default_rng(0)
    n = 8000
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(12)})
    y = X["x0"] * 1.5 + X["x3"] * np.sign(X["x5"]) + 0.1 * rng.normal(size=n)
    on = _fit_selection(X, y, "1", monkeypatch, fe_max_steps=fe_max_steps)
    off = _fit_selection(X, y, "0", monkeypatch, fe_max_steps=fe_max_steps)
    assert on == off, f"compact codes changed selection: ON={on} OFF={off}"


def test_compact_codes_downcast_mechanism():
    """The int32 categorized codes matrix must downcast to int8 for nbins<=127 codes (4x) -- the actual memory win, and
    the code VALUES must be identical after the cast (selection-equivalence rests on this)."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(2)
    n = 4000
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(8)})
    data, _cols, _nbins = categorize_dataset(df=X, method="quantile", n_bins=10, dtype=np.int32)
    assert data.dtype == np.int32  # quantization_dtype default -> wide storage before the compact downcast
    # Apply the same range-checked downcast the fit uses.
    dmin, dmax = int(data.min()), int(data.max())
    assert -128 <= dmin and dmax <= 127, "nbins=10 codes must fit int8"
    narrow = data.astype(np.int8)
    assert np.array_equal(narrow.astype(np.int64), data.astype(np.int64)), "downcast must preserve code values"
    assert narrow.nbytes * 4 == data.nbytes, "int8 storage is 4x smaller than int32"
