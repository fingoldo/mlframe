"""Regression: polars-ds robust/standard/min_max scalers must not emit NaN/inf
on a zero-IQR (constant / near-constant) column, INDEPENDENT of whether
constant-column removal is enabled.

polars-ds ``robust_scale`` divides by ``q_high - q_low`` (``scale`` divides by
std / range). For a constant column that divisor is 0, so the scaled output is
NaN, silently corrupting every downstream row. The fuzz suite previously masked
this by forcing ``remove_constant_columns=True`` whenever degenerate columns
were injected. These tests reproduce the masked scenario directly with
constant-column removal OFF.
"""

import numpy as np
import polars as pl
import pytest

from mlframe.training.pipeline import (
    create_polarsds_pipeline,
    _select_scalable_numeric_columns,
    _apply_safe_scaler,
)
from mlframe.training.configs import PreprocessingBackendConfig

pytest.importorskip("polars_ds")


def _frame(n: int = 200, with_allnull: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "a": rng.normal(size=n),
        "const": np.full(n, 3.5),  # zero-IQR / zero-std / zero-range
        "b": rng.normal(size=n) * 5.0,
    }
    if with_allnull:
        data["allnull"] = np.full(n, np.nan)  # all-null -> quantile None
    return pl.DataFrame(data)


def _numeric_finite(out: pl.DataFrame) -> bool:
    cols = [c for c in out.columns if out[c].dtype.is_numeric()]
    if not cols:
        return True
    arr = out.select(cols).to_numpy()
    return bool(np.isfinite(arr).all())


@pytest.mark.parametrize("scaler_name", ["robust", "standard", "min_max"])
def test_zero_iqr_column_excluded_from_scalable_set(scaler_name):
    method = "robust" if scaler_name == "robust" else scaler_name
    safe = _select_scalable_numeric_columns(_frame(with_allnull=True), method=method)
    assert "const" not in safe
    assert "allnull" not in safe
    assert "a" in safe and "b" in safe


@pytest.mark.parametrize("scaler_name", ["robust", "standard", "min_max"])
def test_pipeline_no_nonfinite_with_constant_column(scaler_name):
    df = _frame()
    cfg = PreprocessingBackendConfig(scaler_name=scaler_name)
    bp = create_polarsds_pipeline(df, cfg, verbose=0)
    assert bp is not None
    out = bp.transform(df)
    assert _numeric_finite(out), f"{scaler_name}: scaled output contains NaN/inf"


def test_apply_safe_scaler_guards_explicit_zero_iqr_column():
    """The guard must hold even when a caller explicitly REQUESTS the zero-IQR
    column -- robustness must not depend on caller-side pre-filtering. (On the
    pre-fix code there was no such helper / the const column would reach
    polars-ds and produce NaN.)"""
    from polars_ds.pipeline import Blueprint as PdsBlueprint

    df = _frame(with_allnull=True)
    bp = PdsBlueprint(df, name="t")
    bp = _apply_safe_scaler(bp, df, scaler_name="robust", requested_cols=["a", "const", "allnull"])
    out = bp.materialize().transform(df)
    # allnull is passed through (NaN by construction); only the scaled/identity
    # columns must stay finite.
    assert _numeric_finite(out.drop("allnull"))
    # const must be passed through unscaled (identity), not NaN.
    assert out["const"].to_list() == df["const"].to_list()


def test_raw_polarsds_robust_scale_on_const_is_nonfinite():
    """Pins the underlying defect: calling polars-ds robust_scale directly on a
    constant column DOES emit NaN. This is what the guard protects against; if a
    future polars-ds release fixes it, this test flips and the guard can relax.
    """
    from polars_ds.pipeline import Blueprint as PdsBlueprint

    df = _frame()
    bp = PdsBlueprint(df, name="t").robust_scale(["const"], q_low=0.25, q_high=0.75)
    out = bp.materialize().transform(df)
    assert not np.isfinite(out["const"].to_numpy()).all()
