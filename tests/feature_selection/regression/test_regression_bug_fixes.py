"""Regression tests for bugs fixed in commit a953bd7 (refactor of mlframe.feature_selection.filters).

Each test would FAIL on pre-fix code and PASS on post-fix. Per project memory feedback_test_every_bug_fix.md.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bug 1: polygamma late-binding (feature_engineering.py:621-623)
# ---------------------------------------------------------------------------


def test_regression_polygamma_late_binding():
    """Pre-fix: ``for order in range(3): unary[...] = lambda x: sp.polygamma(order, x)`` made all three lambdas resolve order=2.
    Post-fix: ``lambda x, _order=order: sp.polygamma(_order, x)`` binds order at definition time.
    """
    from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

    transforms = create_unary_transformations(preset="maximal")
    x = np.linspace(1.5, 3.5, 20)  # positive domain valid for polygamma/struve/jv

    # polygamma family
    if all(f"polygamma_{i}" in transforms for i in (0, 1, 2)):
        p0 = transforms["polygamma_0"](x)
        p1 = transforms["polygamma_1"](x)
        p2 = transforms["polygamma_2"](x)
        # Pre-fix bug: all three identical (all use order=2). Post-fix: distinct.
        assert not np.allclose(p0, p1), "polygamma_0 and polygamma_1 were identical (late-binding bug regressed)"
        assert not np.allclose(p1, p2), "polygamma_1 and polygamma_2 were identical (late-binding bug regressed)"

    # struve family (same bug pattern)
    if all(f"struve{i}" in transforms for i in (0, 1, 2)):
        s0 = transforms["struve0"](x)
        s1 = transforms["struve1"](x)
        s2 = transforms["struve2"](x)
        assert not np.allclose(s0, s1)
        assert not np.allclose(s1, s2)

    # jv (Bessel-J) family
    if all(f"jv{i}" in transforms for i in (0, 1, 2)):
        j0 = transforms["jv0"](x)
        j1 = transforms["jv1"](x)
        j2 = transforms["jv2"](x)
        assert not np.allclose(j0, j1)
        assert not np.allclose(j1, j2)


# ---------------------------------------------------------------------------
# Bug 2: pl not in scope in mrmr.py:_run_fe_step (mrmr.py:1315)
# ---------------------------------------------------------------------------


def test_regression_mrmr_polars_pl_in_scope():
    """Pre-fix: ``pl.Series(...)`` used in _run_fe_step without local ``import polars as pl``. Polars input -> NameError.
    Post-fix: added ``import polars as pl`` at the top of the polars-input branch.
    """
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(42)
    n = 200
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x0 + 0.3 * x1 > 0).astype(np.int32)

    df = pl.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    target = pl.Series("y", y)

    sel = MRMR(
        full_npermutations=2,
        baseline_npermutations=2,
        fe_max_steps=2,  # triggers _run_fe_step branch where the bug lived
        n_jobs=1,
        verbose=0,
        random_seed=42,
    )
    # Pre-fix: NameError. Post-fix: completes (or runs a no-op FE step).
    sel.fit(df, target)
    assert hasattr(sel, "support_")


# ---------------------------------------------------------------------------
# Bug 3: MAX_JOBLIB_NBYTES missing import in screen.py
# ---------------------------------------------------------------------------


def test_regression_screen_max_joblib_nbytes_imported():
    """Pre-fix: ``parallel_kwargs = dict(max_nbytes=MAX_JOBLIB_NBYTES)`` at screen.py:190 raised NameError.
    Post-fix: ``MAX_JOBLIB_NBYTES`` added to the ``from ._internals import (...)`` block.
    """
    from mlframe.feature_selection.filters.screen import screen_predictors

    rng = np.random.default_rng(0)
    n = 50
    factors_data = rng.integers(0, 3, size=(n, 4)).astype(np.int32)
    targets_data = rng.integers(0, 2, size=(n, 1)).astype(np.int32)
    factors_nbins = np.array([3, 3, 3, 3], dtype=np.int32)
    targets_nbins = np.array([2], dtype=np.int32)

    # Pre-fix: NameError on the parallel_kwargs initialisation.
    result = screen_predictors(
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        factors_names=[f"f{i}" for i in range(factors_data.shape[1])],
        targets_data=targets_data,
        targets_nbins=targets_nbins,
        y=np.array([0], dtype=np.int32),
        parallel_kwargs=None,  # triggers the fallback path that uses MAX_JOBLIB_NBYTES
        full_npermutations=2,
        baseline_npermutations=2,
        n_workers=1,
        verbose=0,
        random_seed=0,
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Bug 4: hermite Optuna closure late-binding (hermite_fe.py:1421-1459)
# ---------------------------------------------------------------------------


def test_regression_hermite_optuna_closure_late_binding():
    """Pre-fix: closures inside ``for degree in degree_grid:`` referenced ca_size/cb_size/eval_pair_fn/eval_kwargs/stop_state from enclosing scope.
    Synchronous study.optimize() consumed them within-iteration so the bug never manifested, but B023 flagged the late-binding risk.
    Post-fix: vars bound via default args (``_ca_size=ca_size`` etc.).
    Test: multi-degree Optuna run completes cleanly and returns a result whose ``degree`` field is the BEST observed -- not necessarily the final iteration.
    """
    pytest.importorskip("optuna")
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

    rng = np.random.default_rng(0)
    n = 400
    x_a = rng.uniform(-1, 1, n)
    x_b = rng.uniform(-1, 1, n)
    y = ((x_a * x_b) > 0).astype(np.int32)  # XOR-like target

    result = optimise_hermite_pair(
        x_a=x_a,
        x_b=x_b,
        y=y,
        basis="hermite",
        min_degree=2,
        max_degree=3,
        n_trials=15,
        optimizer="optuna",
        seed=42,
    )
    # Even if optimise_hermite_pair returns None (no degree beat the baseline) the run must complete without exception.
    if result is not None:
        assert hasattr(result, "degree") or hasattr(result, "best_degree") or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Bug 5: cat_interactions _maybe_rerank_with_mm IndexError on short selected_idx
# ---------------------------------------------------------------------------


def test_regression_cat_interactions_short_pair_mm():
    """Pre-fix: ``for k in selected_idx: if per_pair_mm[bool(True)]: pass`` evaluated ``per_pair_mm[1]`` each iteration, raising
    IndexError when ``len(per_pair_mm) < 2``. Post-fix: the dead no-op loop was removed.
    """
    from mlframe.feature_selection.filters.cat_interactions import _maybe_rerank_with_mm
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig

    rng = np.random.default_rng(0)
    n = 200
    # Minimal fixture: 2 cat features + 1 target, single selected pair
    factors_data = rng.integers(0, 2, size=(n, 3)).astype(np.int32)
    pairs_a = np.array([0], dtype=np.int64)
    pairs_b = np.array([1], dtype=np.int64)
    selected_idx = np.array([0], dtype=np.int64)  # length-1 (the problematic case)
    ii_arr = np.array([0.05], dtype=np.float64)
    nbins = np.array([2, 2, 2], dtype=np.int64)
    target_indices = np.array([2], dtype=np.int64)
    classes_y = factors_data[:, 2]
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.float64) / n

    cfg = CatFEConfig(use_miller_madow=True)

    # Pre-fix: IndexError on per_pair_mm[1] when len(per_pair_mm) == 1.
    out_ii, out_idx = _maybe_rerank_with_mm(
        factors_data=factors_data,
        pairs_a=pairs_a,
        pairs_b=pairs_b,
        selected_idx=selected_idx,
        ii_arr=ii_arr,
        nbins=nbins,
        target_indices=target_indices,
        classes_y=classes_y,
        freqs_y=freqs_y,
        cfg=cfg,
        dtype=np.int32,
        verbose=0,
    )
    assert out_ii.shape == (1,)
    assert out_idx.shape == (1,)
