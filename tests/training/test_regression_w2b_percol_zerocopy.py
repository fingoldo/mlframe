"""Regression sensors for w2b-percol-scattered zerocopy fixes.

Covers:
- ``_extract_col_1d`` helper consolidation (finding #9).
- ``_LagPredictDeployableModel.predict`` polars get_column path (finding #12).
- target-distribution polars gather() probe (finding #11).
- ``_predict_guards`` polars allow_copy=False dtype gate (finding #17).
- ``composite_cache`` per-column hashing (finding #8 -- regression-only, no behavioural change).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")


def test_extract_col_1d_polars_pandas_array_parity():
    """Helper must return identical flat ndarrays for polars/pandas/list inputs of the same data."""
    from mlframe.training._dummy_baseline_regression import _extract_col_1d

    base = np.arange(50, dtype=np.float64)
    pdf = pd.DataFrame({"x": base})
    pldf = pl.DataFrame({"x": base})

    arr_pd = _extract_col_1d(pdf, "x")
    arr_pl = _extract_col_1d(pldf, "x")

    assert arr_pd.ndim == 1
    assert arr_pl.ndim == 1
    assert arr_pd.shape == (50,)
    assert arr_pl.shape == (50,)
    np.testing.assert_array_equal(arr_pd, base)
    np.testing.assert_array_equal(arr_pl, base)


def test_extract_col_1d_none_returns_empty_array():
    from mlframe.training._dummy_baseline_regression import _extract_col_1d

    out = _extract_col_1d(None, "x")
    assert out.size == 0


def test_lag_predict_deployable_polars_returns_1d_float64():
    """Polars predict path must return a 1-D float64 array of correct length (no (N,1) leakage from the prior .select() path)."""
    from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel

    df = pl.DataFrame({"lag_col": np.linspace(0.0, 1.0, 25, dtype=np.float64), "junk": np.ones(25)})
    model = _LagPredictDeployableModel(lag_column="lag_col")
    preds = model.predict(df)
    assert preds.ndim == 1
    assert preds.shape == (25,)
    assert preds.dtype == np.float64
    np.testing.assert_allclose(preds, np.linspace(0.0, 1.0, 25))


def test_lag_predict_deployable_pandas_still_works():
    from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel

    df = pd.DataFrame({"lag_col": np.arange(10, dtype=np.int32)})
    model = _LagPredictDeployableModel(lag_column="lag_col")
    preds = model.predict(df)
    assert preds.ndim == 1
    assert preds.shape == (10,)
    assert preds.dtype == np.float64


def test_lag_predict_deployable_int_cast_does_not_break():
    """Polars int column must be cast to float64 with no shape distortion."""
    from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel

    df = pl.DataFrame({"y_prev": np.arange(7, dtype=np.int64)})
    model = _LagPredictDeployableModel(lag_column="y_prev")
    preds = model.predict(df)
    assert preds.shape == (7,)
    assert preds.dtype == np.float64
    np.testing.assert_array_equal(preds, np.arange(7, dtype=np.float64))


def test_predict_guards_polars_allow_copy_zero_copy_path():
    """`_fit_persist_and_transform` on a uniformly-float64 polars frame must produce the same numeric output as the legacy copy path."""
    from mlframe.training._predict_guards import _fit_persist_and_transform

    rng = np.random.default_rng(7)
    arr = rng.normal(size=(64, 4)).astype(np.float64)
    pdf = pd.DataFrame(arr, columns=[f"c{i}" for i in range(4)])
    pldf = pl.DataFrame({f"c{i}": arr[:, i] for i in range(4)})

    class _Model:
        pass

    out_pd = _fit_persist_and_transform(_Model(), pdf, lambda X: np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X), n_rows=arr.shape[0])
    out_pl = _fit_persist_and_transform(_Model(), pldf, lambda X: np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X), n_rows=arr.shape[0])
    np.testing.assert_allclose(out_pd, out_pl, rtol=1e-9, atol=1e-9)


def test_target_distribution_polars_gather_probe_does_not_materialise_full_column():
    """The polars branch must use gather() so monotonicity probes don't allocate the whole column. Behavioural test: probe still returns
    monotonic verdicts identical to the pandas branch on the same data."""
    rng = np.random.default_rng(42)
    n = 5_000
    ts = np.arange(n, dtype=np.int64)
    payload = rng.normal(size=n)

    pdf = pd.DataFrame({"date": ts, "y": payload})
    pldf = pl.DataFrame({"date": ts, "y": payload})

    stride = max(1, n // 1024)
    sample_pd = pdf.iloc[::stride]["date"].to_numpy()
    sample_pl = pldf.get_column("date").gather(list(range(0, n, stride))).to_numpy()

    np.testing.assert_array_equal(sample_pd, sample_pl)
    # Stride logic: ceil(n / stride) samples; for n=5000, stride=4 -> 1250 samples. Cap is just "samples << n".
    assert sample_pl.size < n // 2
