"""Row-wise quantile features equal ``np.nanquantile`` exactly, on tied and untied neighbours, through every backend.

The numba kernel and the polars expression interpolated as ``low * (1 - frac) + high * frac``. numpy's ``_lerp`` computes
``low + (high - low) * frac`` below ``frac = 0.5`` and ``high - (high - low) * (1 - frac)`` from there up, so the two disagreed by a ULP on
ordinary rows (-2.8e-17 against 0.0) and on tied ones. The same frame therefore produced different feature values depending on whether numba
was installed and whether it arrived as pandas or polars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.row_wise_summary import row_wise_summary_stats

_QS = ["q10", "q25", "q75", "q90"]


def _rows(with_nan: bool) -> np.ndarray:
    """Rows of decimal values (not exact in binary) with many ties; optionally a few NaNs per row so the NaN-aware kernel runs."""
    rng = np.random.default_rng(11)
    rows = np.round(rng.normal(size=(400, 12)) * 2.0, 1)
    rows[::3, :6] = rows[::3, [0]]  # long runs of one tied value
    rows[1::5] = np.round(rng.choice([-0.9, 0.1, 0.3, 0.7, 1.5704455335485081, 2.2, 5.1], size=(rows[1::5].shape)), 16)
    if with_nan:
        mask = rng.random(rows.shape) < 0.15
        mask[:, 0] = False  # keep at least one value per row
        rows = rows.copy()
        rows[mask] = np.nan
    return rows


def _expected(rows: np.ndarray, stat: str) -> np.ndarray:
    """``np.nanquantile`` per row at the stat's level."""
    return np.nanquantile(rows, int(stat[1:]) / 100.0, axis=1)


def _col(out, stat: str) -> np.ndarray:
    """The output column for ``stat``, whatever the column prefix."""
    cols = [c for c in out.columns if str(c).endswith(stat)]
    assert cols, f"no output column for {stat}: {list(out.columns)}"
    return np.asarray(out[cols[0]], dtype=np.float64)


def _assert_exact(got: np.ndarray, want: np.ndarray, label: str) -> None:
    """Bit-equal, with a count of mismatching rows in the message."""
    bad = np.flatnonzero(~((got == want) | (np.isnan(got) & np.isnan(want))))
    assert bad.size == 0, f"{label}: {bad.size} row(s) differ from np.nanquantile, e.g. row {bad[0]}: {got[bad[0]]!r} vs {want[bad[0]]!r}"


@pytest.mark.parametrize("with_nan", [False, True], ids=["dense", "with_nan"])
def test_pandas_path_quantiles_equal_nanquantile(with_nan):
    """The pandas entry point (vectorised numpy when NaN-free, the numba kernel otherwise) returns np.nanquantile's exact values."""
    rows = _rows(with_nan)
    df = pd.DataFrame(rows, columns=[f"c{i}" for i in range(rows.shape[1])])
    out = row_wise_summary_stats(df, stats=_QS)
    for stat in _QS:
        _assert_exact(_col(out, stat), _expected(rows, stat), f"pandas {stat}")


@pytest.mark.parametrize("with_nan", [False, True], ids=["dense", "with_nan"])
def test_numba_kernel_quantiles_equal_nanquantile(with_nan):
    """The numba kernel itself, called directly, returns np.nanquantile's exact values."""
    import mlframe.feature_engineering.row_wise_summary as rws

    if not getattr(rws, "_HAS_NUMBA", False):
        pytest.skip("numba not installed")
    kernel = next((getattr(rws, n) for n in dir(rws) if n.endswith("_njit") and "quantile" in n), None)
    assert kernel is not None, "no numba row-quantile kernel found"
    rows = _rows(with_nan)
    qs = np.array([int(s[1:]) / 100.0 for s in _QS])
    got = kernel(np.ascontiguousarray(rows), qs)
    assert len(got) == len(_QS), f"kernel returned {len(got)} quantile rows for {len(_QS)} requested"
    for k, stat in enumerate(_QS):
        _assert_exact(np.asarray(got[k]), _expected(rows, stat), f"numba {stat}")


@pytest.mark.parametrize("with_nan", [False, True], ids=["dense", "with_nan"])
def test_polars_path_quantiles_equal_nanquantile(with_nan):
    """The polars entry point returns the identical values."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_engineering.row_wise_summary_polars import row_wise_summary_stats_polars

    rows = _rows(with_nan)
    df = pl.DataFrame({f"c{i}": rows[:, i] for i in range(rows.shape[1])}).fill_nan(None)
    out = row_wise_summary_stats_polars(df, stats=_QS)
    for stat in _QS:
        _assert_exact(_col(out, stat), _expected(rows, stat), f"polars {stat}")
