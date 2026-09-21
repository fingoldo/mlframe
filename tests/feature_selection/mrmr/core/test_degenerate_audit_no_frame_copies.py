"""The degenerate-column audit reports the same reasons without holding full-frame float64 copies.

It kept a float64 copy plus a bool mask of every numeric column for the whole scan, checked its width cap only afterwards, and then built
its correlation matrix with a per-column copy and two more full-size temporaries; the polars branch converted the whole frame at once.
"""

from __future__ import annotations

import logging
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _mrmr_degenerate as dg


def _mixed_frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Every reason at once: all-NaN, constant, duplicate, exact and affine collinear columns, NaN and inf cells, ints, a string column."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = a.copy()
    c[::17] = np.nan
    frame = {
        "a": a,
        "b": b,
        "all_nan": np.full(n, np.nan),
        "const": np.full(n, 3.25),
        "dup_a": a.copy(),
        "affine_b": 2.5 * b - 7.0,
        "a_with_nan": c,
        "ints": rng.integers(0, 50, size=n).astype(np.int64),
        "ints_x3": rng.integers(0, 50, size=n).astype(np.int64) * 3,
        "with_inf": np.where(np.arange(n) % 101 == 0, np.inf, rng.normal(size=n)),
        "label": rng.choice(["u", "v"], size=n),
        "noise": rng.normal(size=n),
    }
    frame["ints_x3"] = frame["ints"] * 3
    return pd.DataFrame(frame)


def test_reasons_match_the_expected_classification():
    """Each constructed column is reported with the reason its construction implies."""
    reasons = dg.audit_degenerate_columns(_mixed_frame())
    assert reasons["all_nan"] == "all_nan"
    assert reasons["const"] == "constant"
    assert reasons["dup_a"] == "duplicate_of:a"
    assert reasons["affine_b"] == "collinear_with:b"
    assert reasons["ints_x3"] == "collinear_with:ints"
    for clean in ("a", "b", "noise", "label", "ints"):
        assert clean not in reasons, f"{clean} flagged: {reasons.get(clean)}"


def test_polars_and_pandas_report_identical_reasons():
    """The per-column polars path gives exactly the pandas result."""
    pl = pytest.importorskip("polars")
    pdf = _mixed_frame(seed=1)
    plf = pl.from_pandas(pdf)
    assert dg.audit_degenerate_columns(plf) == dg.audit_degenerate_columns(pdf)


def test_scan_peak_stays_near_one_matrix():
    """On a wide numeric frame the traced peak stays near the one (K, n) matrix, not several frame-sized copies."""
    rng = np.random.default_rng(2)
    n, p = 50_000, 60
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"c{i}" for i in range(p)])
    frame_bytes = n * p * 8
    dg.audit_degenerate_columns(X.iloc[:500])  # warm
    tracemalloc.start()
    try:
        dg.audit_degenerate_columns(X)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.6 * frame_bytes, f"audit peak {peak / frame_bytes:.2f} frame copies"


def test_byte_budget_skips_the_pass_before_allocating(caplog):
    """A byte budget below the matrix size skips collinearity with a log; the other reasons are still reported."""
    X = _mixed_frame()
    with caplog.at_level(logging.INFO):
        reasons = dg.audit_degenerate_columns(X, max_collinearity_bytes=1024)
    assert not any(r.startswith("collinear_with") for r in reasons.values())
    assert reasons["dup_a"] == "duplicate_of:a" and reasons["const"] == "constant"
    assert any("max_collinearity_bytes" in r.getMessage() for r in caplog.records)
