"""Regression + perf sentinel for the ``_select_scalable_numeric_columns``
finite-count fusion.

The per-column non-null/finite count was computed via a double
``drop_nulls().drop_nans()`` chain (once for the filter predicate, once for the
filtered series) plus a finite-filter + ``len``. It is now a single
``pl.col(c).is_finite().sum()``: bit-identical (is_finite -> null on null,
False on NaN/inf, so the sum counts exactly the finite non-null values) and
~4.8x faster on the count select at 50col x 200k.

These tests pin:
  * the SELECTED scalable-column SET is unchanged on a mixed frame
    (numeric, constant, all-nan, inf-bearing, low-variance) across every
    scaler method -- the bit-identity guarantee;
  * the batched path and the per-col fallback (``stats_row is None``) agree;
  * a wall-time sentinel that the fused count select beats the old chain.
"""

import time

import numpy as np
import polars as pl
import pytest

from mlframe.training.pipeline import _select_scalable_numeric_columns


def _mixed_frame(n: int = 2000) -> pl.DataFrame:
    """Mixed frame."""
    rng = np.random.default_rng(7)
    good = rng.normal(size=n)
    lowvar = rng.normal(size=n) * 1e-9  # tiny but non-zero spread
    inf_col = rng.normal(size=n).copy()
    inf_col[rng.random(n) < 0.02] = np.inf
    inf_col[rng.random(n) < 0.02] = -np.inf
    nan_col = rng.normal(size=n).copy()
    nan_col[rng.random(n) < 0.1] = np.nan
    return pl.DataFrame(
        {
            "good": good,
            "good2": rng.normal(size=n) * 5.0,
            "lowvar": lowvar,
            "const": np.full(n, 3.5),  # zero-spread
            "allnan": np.full(n, np.nan),  # all non-finite
            "infbearing": inf_col,
            "withnan": nan_col,
            "intcol": rng.integers(0, 100, size=n),
            "intconst": np.full(n, 4, dtype=np.int64),
        }
    )


# The expected scalable set is the historical (pre-fusion) selection, hardcoded
# so a future change that perturbs selection trips this regardless of impl.
_EXPECTED = {
    "robust": {"good", "good2", "lowvar", "infbearing", "withnan", "intcol"},
    "standard": {"good", "good2", "lowvar", "infbearing", "withnan", "intcol"},
    "min_max": {"good", "good2", "lowvar", "infbearing", "withnan", "intcol"},
    # abs_max only skips columns whose abs().max() == 0; the constant 3.5 /
    # constant 4 columns have a non-zero abs-max so they survive here.
    "abs_max": {
        "good",
        "good2",
        "lowvar",
        "infbearing",
        "withnan",
        "intcol",
        "const",
        "intconst",
    },
}


@pytest.mark.parametrize("method", ["robust", "standard", "min_max", "abs_max"])
def test_selected_set_unchanged(method):
    """Selected set unchanged."""
    df = _mixed_frame()
    got = set(_select_scalable_numeric_columns(df, method=method))
    assert got == _EXPECTED[method], (method, sorted(got))


@pytest.mark.parametrize("method", ["robust", "standard", "min_max", "abs_max"])
def test_batched_matches_fallback(method, monkeypatch):
    """The per-col fallback path (stats_row is None) must select the same set
    as the batched path -- exercises the fallback finite-count branch too."""
    df = _mixed_frame()
    batched = set(_select_scalable_numeric_columns(df, method=method))

    # Force the batched collect to fail -> drives the per-col fallback.
    # Patch ``LazyFrame.select`` (not ``DataFrame.lazy``): the batched path is
    # ``train_df.lazy().select(...).collect()``, while the per-col fallback's
    # own Series ops (is_finite/quantile/std) route through DataFrame.select_seq
    # and stay intact, so only the batched sweep is broken.
    real_select = pl.LazyFrame.select

    def boom(self, *a, **k):
        """Boom."""
        raise RuntimeError("force fallback")

    monkeypatch.setattr(pl.LazyFrame, "select", boom)
    try:
        fallback = set(_select_scalable_numeric_columns(df, method=method))
    finally:
        monkeypatch.setattr(pl.LazyFrame, "select", real_select)
    assert batched == fallback == _EXPECTED[method], (method, sorted(fallback))


def test_finite_count_expr_bit_identical():
    """Direct expr-level bit-identity: is_finite().sum() == old chain, on a
    column carrying null + NaN + inf + finite values."""
    vals = [1.0, 2.0, np.nan, np.inf, -np.inf, None, 3.0, None, np.nan, 4.0]
    df = pl.DataFrame({"c": pl.Series("c", vals, dtype=pl.Float64)})
    old = pl.col("c").drop_nulls().drop_nans().filter(pl.col("c").drop_nulls().drop_nans().is_finite()).len()
    new = pl.col("c").is_finite().sum()
    r = df.select(old.alias("o"), new.alias("n"))
    assert r["o"][0] == r["n"][0] == 4


def test_perf_sentinel_fused_count_faster():
    """Wall sentinel: the fused single-pass count select is faster than the
    old double-drop chain on a wide frame. Generous margin to avoid noise."""
    rng = np.random.default_rng(1)
    n, ncol = 200_000, 50
    data = {}
    for i in range(ncol):
        a = rng.normal(size=n)
        a[rng.random(n) < 0.05] = np.nan
        a[rng.random(n) < 0.01] = np.inf
        data[f"c{i}"] = a
    df = pl.DataFrame(data)
    cols = df.columns

    def old():
        """Old."""
        e = [pl.col(c).drop_nulls().drop_nans().filter(pl.col(c).drop_nulls().drop_nans().is_finite()).len().alias("n" + c) for c in cols]
        return df.lazy().select(e).collect()

    def new():
        """New."""
        e = [pl.col(c).is_finite().sum().alias("n" + c) for c in cols]
        return df.lazy().select(e).collect()

    old()
    new()  # warm

    def bench(f, k=7):
        """Bench."""
        best = float("inf")
        for _ in range(k):
            t0 = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - t0)
        return best

    t_old = bench(old)
    t_new = bench(new)
    # bit-identity on the perf shape too
    assert (old().to_numpy().ravel() == new().to_numpy().ravel()).all()
    assert t_new < t_old, f"expected fused faster: old={t_old * 1e3:.2f}ms new={t_new * 1e3:.2f}ms"
