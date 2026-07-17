"""Regression: nullable-dtype densification is byte-gated and never mutates the caller's frame.

MRMR densifies pandas masked (Int64/Float64/boolean + pd.NA) columns to float64 so the screen/FE numba kernels
see dense NaN, not object pd.NA. A single ``X.assign(**{all nullable})`` materialises every float64 column at
once (peak ~2x the nullable-column bytes); above _NULLABLE_DENSIFY_EAGER_MAX_BYTES the densification switches to
one column per ``assign`` so peak extra RAM stays ~one column. Either path returns a NEW frame, so the caller's
frame must keep its nullable dtypes. Both paths must produce identical selection (the densified values are the
same regardless of single-vs-incremental assign).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR
import mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core as _core


def _nullable_xy(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = pd.Series((a + b > 0).astype(np.int64), name="targ")
    X = pd.DataFrame(
        {
            "a": pd.array(a, dtype="Float64"),
            "b": pd.array(b, dtype="Float64"),
            "c": pd.array(rng.integers(0, 5, n), dtype="Int64"),
            "d": pd.array(rng.integers(0, 2, n).astype(bool), dtype="boolean"),
        }
    )
    # A few pd.NA so the masked path is genuinely exercised.
    X.loc[X.index[:5], "c"] = pd.NA
    return X, y


def _fit_support(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_seed=0).fit(X, y)
    return tuple(np.asarray(m.support_).tolist()), tuple(m.get_feature_names_out())


def test_densify_does_not_mutate_caller_frame():
    X, y = _nullable_xy()
    dtypes_before = X.dtypes.astype(str).to_dict()
    _fit_support(X, y)
    assert X.dtypes.astype(str).to_dict() == dtypes_before, "fit mutated the caller's nullable dtypes"


def test_eager_and_per_column_densify_paths_select_identically(monkeypatch):
    X, y = _nullable_xy()
    # Eager path (default high threshold).
    monkeypatch.setattr(_core, "_NULLABLE_DENSIFY_EAGER_MAX_BYTES", 2 * 1024**3)
    eager = _fit_support(X, y)
    # Per-column path (threshold forced to 0 so the byte check always takes the incremental branch).
    monkeypatch.setattr(_core, "_NULLABLE_DENSIFY_EAGER_MAX_BYTES", 0)
    per_col = _fit_support(X, y)
    assert eager == per_col, f"byte-gated densify path changed selection: eager={eager} per_col={per_col}"


def test_nullable_matches_dense_float64_selection():
    """The whole point of densification: a nullable frame must select the same as its dense-float64 twin."""
    X, y = _nullable_xy()
    X_dense = X.astype("float64")
    assert _fit_support(X, y) == _fit_support(X_dense, y)
