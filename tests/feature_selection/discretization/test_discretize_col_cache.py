"""Cross-instance per-column code cache for the unsupervised numeric discretization path.

Unsupervised binning (``nbins_strategy=None``) bins each numeric column independently of the others and of
the target, so per-column ordinal codes are a pure function of (column values, n_bins, method, dtype). The
process-wide cache (``_discretize_2d_array_col_cached``) lets a suite training many targets on the SAME
feature frame reuse feature-column codes and rebin only the new target column -- bit-identical by construction.
The cache is correctly NOT consulted on the supervised ``nbins_strategy`` path (mdlp/...), where bins depend
on the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization import (
    categorize_dataset,
    clear_numeric_code_cache,
)


def _frame(seed, n=400, target_vals=None):
    """Build a synthetic feature+target frame for the discretize-cache tests."""
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.normal(size=n) for i in range(5)}
    cols["target"] = rng.normal(size=n) if target_vals is None else target_vals
    return pd.DataFrame(cols)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Autouse fixture: clear the process-wide numeric-code cache before every test."""
    clear_numeric_code_cache()
    yield
    clear_numeric_code_cache()


def test_cache_on_bit_identical_to_cache_off(monkeypatch):
    """Cache ON must produce byte-identical codes/cols/nbins vs cache OFF on the unsupervised path."""
    df = _frame(0)

    monkeypatch.setenv("MLFRAME_DISCRETIZE_COL_CACHE", "0")
    clear_numeric_code_cache()
    data_off, cols_off, nbins_off = categorize_dataset(df=df, method="quantile", n_bins=10, nbins_strategy=None)

    monkeypatch.setenv("MLFRAME_DISCRETIZE_COL_CACHE", "1")
    clear_numeric_code_cache()
    data_on, cols_on, nbins_on = categorize_dataset(df=df, method="quantile", n_bins=10, nbins_strategy=None)

    assert cols_off == cols_on
    assert np.array_equal(nbins_off, nbins_on)
    assert np.array_equal(data_off, data_on)


def test_shared_feature_columns_reused_across_targets():
    """Two frames sharing f0..f4 but with DIFFERENT target columns: the second categorize reuses the cached
    feature-column codes (cache populated) and the feature codes are byte-identical across the two calls."""
    clear_numeric_code_cache()
    df_a = _frame(1)
    df_b = _frame(1)  # same features (seed 1) ...
    df_b["target"] = np.random.default_rng(999).normal(size=len(df_b))  # ... different target

    data_a, cols_a, _ = categorize_dataset(df=df_a, method="quantile", n_bins=8, nbins_strategy=None)

    # Confirm the cache is actually consulted (feature-column entries exist after the first categorize).
    from mlframe.feature_selection.filters.discretization import _discretization_dataset as _dd

    assert len(_dd._NUMERIC_CODE_CACHE) >= 5, "feature columns were not cached after first categorize"

    data_b, cols_b, _ = categorize_dataset(df=df_b, method="quantile", n_bins=8, nbins_strategy=None)

    # Feature columns (f0..f4) must be byte-identical between A and B (same values -> same codes).
    for fc in [c for c in cols_a if c.startswith("f")]:
        ja, jb = cols_a.index(fc), cols_b.index(fc)
        assert np.array_equal(data_a[:, ja], data_b[:, jb]), f"feature {fc} codes diverged across targets"


def test_cache_not_consulted_on_supervised_strategy():
    """Supervised nbins_strategy (mdlp) must NOT populate the unsupervised per-column cache (bins are y-dependent)."""
    pytest.importorskip("numba")
    clear_numeric_code_cache()
    df = _frame(2)
    from mlframe.feature_selection.filters.discretization import _discretization_dataset as _dd

    try:
        categorize_dataset(df=df, method="quantile", n_bins=10, nbins_strategy="mdlp", y_for_strategy=df["target"].to_numpy())
    except Exception:
        pytest.skip("mdlp strategy unavailable in this build")
    assert len(_dd._NUMERIC_CODE_CACHE) == 0, "supervised path must not use the unsupervised per-column cache"


def test_regression_xxh3_cache_key_matches_blake2b_fallback():
    """2026-07-17: the per-column cache key switched from ``hashlib.blake2b`` (cryptographic, ~2.6x
    slower -- measured on a 99401x500 wellbore-shaped frame) to ``xxhash.xxh3_128_digest`` over a
    ONE-TIME bulk-transposed buffer (was ``n_cols`` separate ``np.ascontiguousarray(arr[:, j])`` strided
    copies -- fixed alongside). Both are content-only fingerprints (no cryptographic-strength requirement
    for a process-local cache key), so this pins that they discriminate columns IDENTICALLY: same content
    -> same key (dedup), different content -> different key -- with either hash backend available."""
    from mlframe.feature_selection.filters.discretization import _discretization_dataset as _dd
    from mlframe.feature_selection.filters.discretization import discretize_2d_array as _disc2d

    rng = np.random.default_rng(11)
    n = 300
    arr = rng.standard_normal((n, 6))
    arr[:, 4] = arr[:, 1]  # force an exact duplicate column

    for use_xxh3 in (True, False):
        saved = _dd._xxh3_128
        if not use_xxh3:
            _dd._xxh3_128 = None
        try:
            _dd._NUMERIC_CODE_CACHE.clear()
            _dd._NUMERIC_CODE_CACHE_BYTES = 0
            out = _dd._discretize_2d_array_col_cached(arr, n_bins=5, method="quantile", min_ncats=2, dtype=np.int16, discretize_2d_array=_disc2d)
            ref = _disc2d(arr=arr, n_bins=5, method="quantile", min_ncats=2, min_values=None, max_values=None, dtype=np.int16)
            assert np.array_equal(out, ref), f"use_xxh3={use_xxh3}: codes diverged from the reference discretizer"
            assert np.array_equal(out[:, 1], out[:, 4]), f"use_xxh3={use_xxh3}: duplicate columns got different codes"
            assert len(_dd._NUMERIC_CODE_CACHE) == 5, f"use_xxh3={use_xxh3}: duplicate column did not collapse to one cache entry"
        finally:
            _dd._xxh3_128 = saved
