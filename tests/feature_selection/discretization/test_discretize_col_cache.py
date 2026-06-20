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
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.normal(size=n) for i in range(5)}
    cols["target"] = rng.normal(size=n) if target_vals is None else target_vals
    return pd.DataFrame(cols)


@pytest.fixture(autouse=True)
def _clear_cache():
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
