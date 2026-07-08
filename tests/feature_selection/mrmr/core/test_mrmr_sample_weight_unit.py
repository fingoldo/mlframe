"""Regression sentry: MRMR.fit(X, y, sample_weight=None) and MRMR.fit(X, y) must match byte-for-byte.

The sample_weight kwarg is additive; passing None or omitting it must not change support_,
support_indices_, n_features_in_ or any other public fitted attribute. This pins the contract
so the FS cache (keyed on params + content hashes, NOT weights) remains valid across pre-fix
and post-fix snapshots of the codebase.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _toy_dataset(seed=7):
    rng = np.random.default_rng(seed)
    n, p = 400, 6
    X = rng.normal(size=(n, p))
    # y depends primarily on feature 0 with light noise; MRMR should pick x0 reliably.
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    return df, pd.Series(y, name="y")


def test_mrmr_fit_sample_weight_none_matches_omitted():
    """sample_weight=None must produce identical support_ to omitting the kwarg."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _toy_dataset()
    sel_a = MRMR(verbose=0, random_seed=42, max_runtime_mins=0.5).fit(df, y)
    sel_b = MRMR(verbose=0, random_seed=42, max_runtime_mins=0.5).fit(df, y, sample_weight=None)
    assert list(sel_a.support_) == list(sel_b.support_), (
        f"support_ differs: omitted={list(sel_a.support_)} vs None={list(sel_b.support_)}"
    )


def test_mrmr_fit_uniform_sample_weight_matches_unweighted():
    """A constant non-zero weight vector must be treated as uniform -> identical to unweighted."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _toy_dataset()
    n = len(df)
    sw_uniform = np.full(n, 2.5)
    sel_a = MRMR(verbose=0, random_seed=42, max_runtime_mins=0.5).fit(df, y)
    sel_b = MRMR(verbose=0, random_seed=42, max_runtime_mins=0.5).fit(df, y, sample_weight=sw_uniform)
    assert list(sel_a.support_) == list(sel_b.support_), (
        f"uniform-weight support_ {list(sel_b.support_)} != unweighted {list(sel_a.support_)}"
    )


def test_mrmr_fit_sample_weight_validates_shape_and_values():
    """Invalid sample_weight (wrong shape, NaN, negative, zero-sum) must raise ValueError."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _toy_dataset()
    n = len(df)
    sel = MRMR(verbose=0, random_seed=42, max_runtime_mins=0.5)
    with pytest.raises(ValueError, match="sample_weight length"):
        sel.fit(df, y, sample_weight=np.ones(n - 1))
    with pytest.raises(ValueError, match="finite and non-negative"):
        sw = np.ones(n)
        sw[0] = -1.0
        sel.fit(df, y, sample_weight=sw)
    with pytest.raises(ValueError, match="finite and non-negative"):
        sw = np.ones(n)
        sw[0] = np.nan
        sel.fit(df, y, sample_weight=sw)
    with pytest.raises(ValueError, match="sums to zero"):
        sel.fit(df, y, sample_weight=np.zeros(n))


def test_mrmr_fit_nonuniform_sample_weight_runs_without_error():
    """A non-uniform weight vector must trigger the resample branch and still fit cleanly."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _toy_dataset()
    n = len(df)
    rng = np.random.default_rng(0)
    sw = rng.uniform(0.1, 2.0, size=n)
    sel = MRMR(verbose=0, random_seed=42, max_runtime_mins=0.5).fit(df, y, sample_weight=sw)
    # Just assert it produced at least one selected feature and stored the weights internally.
    assert len(sel.support_) >= 1
    assert getattr(sel, "_fit_sample_weight_", None) is not None
    assert sel._fit_sample_weight_.shape == (n,)


def test_mrmr_fit_nonuniform_sample_weight_runs_without_error_on_polars_input():
    """Regression sensor: the resample branch's polars row-select used ``.take()``, which polars removed
    (replaced by ``__getitem__``/``.gather()``); a non-uniform weight on a polars X crashed with
    ``AttributeError: 'DataFrame' object has no attribute 'take'`` before the fix in
    ``_MRMRFitHelpersMixin._maybe_resample_for_sample_weight``."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _toy_dataset()
    n = len(df)
    pl_df = pl.from_pandas(df)
    rng = np.random.default_rng(0)
    sw = rng.uniform(0.1, 2.0, size=n)
    sel = MRMR(verbose=0, random_seed=42, max_runtime_mins=0.5).fit(pl_df, y, sample_weight=sw)
    assert len(sel.support_) >= 1
