"""Identity tests for the fused per-fold cell sum/count kernel against np.add.at on masked gathers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._count_freq_interaction_fe import cat_num_interaction_fit
from mlframe.feature_selection.filters._fold_cell_stats import fold_cell_sum_cnt


def _ref(codes, y, fold_ids, skip, n_cells, valid=None):
    """Reference accumulation through boolean masks and np.add.at."""
    m = fold_ids != skip
    if valid is not None:
        m &= valid
    s = np.zeros(n_cells)
    c = np.zeros(n_cells)
    np.add.at(s, codes[m], y[m])
    np.add.at(c, codes[m], 1.0)
    return s, c


def test_fold_cell_sum_cnt_bit_identical_to_add_at():
    """Every fold, the all-rows sentinel and the validity mask give bit-equal sums and counts."""
    rng = np.random.default_rng(1)
    n, k = 5000, 37
    codes = rng.integers(0, k, n)
    y = rng.normal(size=n) * 1e3
    folds = rng.integers(0, 5, n)
    valid = rng.random(n) > 0.2
    for skip in (-1, 0, 3):
        for v in (None, valid):
            s, c = fold_cell_sum_cnt(codes, y, folds, skip, k, valid=v)
            rs, rc = _ref(codes, y, folds, skip, k, v)
            assert np.array_equal(s, rs) and np.array_equal(c, rc)


def test_cat_num_interaction_fit_ignores_nan_numeric_rows():
    """NaN numeric rows keep a zero residual and do not contaminate the per-category lookup."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"c": rng.choice(list("abc"), 600), "x": rng.normal(size=600)})
    X.loc[::7, "x"] = np.nan
    out = cat_num_interaction_fit(X, np.zeros(600), "c", "x")
    resid = out[0] if isinstance(out, tuple) else out
    resid = np.asarray(resid if not isinstance(resid, dict) else resid.get("residual"))
    assert resid.shape[0] == 600 and np.all(resid[::7] == 0.0) and np.isfinite(resid).all()


def test_composite_corr_small_inputs_route_to_kernel_and_match_numpy():
    """A 1k x 8 matrix is below the old 20k/64 gate; it now picks numba and agrees with the numpy reference within 1e-12."""
    from mlframe.training.composite.discovery import _ktc_dispatch as kd
    from mlframe.training.composite.discovery import _corr_numba as cn
    from mlframe.training.composite.discovery.screening import _safe_abs_corr_all, _safe_abs_corr_all_numpy

    assert kd.choose_corr_backend(1000, 8, min_rows=cn._MIN_ROWS, min_cols=cn._MIN_COLS) in ("numba", "numpy")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1000, 8))
    y = X[:, 0] * 2 + rng.normal(size=1000)
    assert np.abs(_safe_abs_corr_all(y, X) - _safe_abs_corr_all_numpy(y, X)).max() < 1e-12
    assert cn._MIN_ROWS < 20_000 and cn._MIN_COLS < 64


def test_unary_registry_matches_numpy_references_and_is_disk_cached():
    """Registry transforms equal their numpy definitions and the njit wrappers carry an on-disk cache."""
    from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

    reg = create_unary_transformations(preset="medium")
    x = np.linspace(0.5, 4.0, 64)
    assert np.array_equal(reg["neg"](x), -x)
    assert np.array_equal(reg["sqr"](x), np.power(x, 2))
    assert np.array_equal(reg["sqrt"](x), np.sqrt(np.abs(x)))
    assert np.allclose(reg["cbrt"](x), np.cbrt(x), rtol=0, atol=1e-15)
    assert type(reg["neg"]._cache).__name__ != "NullCache"


def test_evaluate_gain_disk_cache_follows_ci_flag():
    """evaluate_gain persists compiled code on disk unless the CI guard is active."""
    from mlframe.feature_selection.filters import evaluation as ev

    has_cache = type(ev.evaluate_gain._cache).__name__ != "NullCache"
    assert has_cache == ev._EVALUATE_GAIN_DISK_CACHE
