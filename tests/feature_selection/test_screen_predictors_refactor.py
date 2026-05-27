"""Regression tests for the screen_predictors -> _confirm_predictor refactor and the two latent
bugs it surfaced and fixed:

1. ``_pool_warmup_noop`` was referenced in ``_screen_predictors.screen_predictors`` (joblib pool warmup
   on the ``n_workers>1`` path) but only defined in ``screen.py`` -> ``NameError`` whenever n_workers>1.
2. ``evaluation.evaluate_candidate`` assigned the whole ``cached_confident_MIs[X]`` tuple
   ``(gain, confidence)`` to ``direct_gain`` -> ``TypeError`` on the next ``direct_gain > 0`` if that
   branch was ever reached.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _ordinal_inputs(X: np.ndarray, y: np.ndarray, nbins: int = 8):
    from mlframe.feature_selection.filters.discretization import discretize_array

    cols = [discretize_array(arr=X[:, j], n_bins=nbins, method="quantile", dtype=np.int32) for j in range(X.shape[1])]
    data = np.column_stack(cols + [y.astype(np.int32)]).astype(np.int32)
    nb = np.array([nbins] * X.shape[1] + [int(len(np.unique(y)))], dtype=np.int64)
    names = [f"x{j}" for j in range(X.shape[1])] + ["y"]
    return data, nb, names, (X.shape[1],)


# ---------------------------------------------------------------------------
# Bug 1: _pool_warmup_noop NameError on the n_workers>1 path
# ---------------------------------------------------------------------------


def test_pool_warmup_noop_is_module_level():
    """The joblib warmup callable must be importable at module scope (loky pickling + the warmup call)."""
    from mlframe.feature_selection.filters import _screen_predictors as sp

    assert hasattr(sp, "_pool_warmup_noop"), "_pool_warmup_noop missing -> NameError on n_workers>1 path"
    assert sp._pool_warmup_noop(0) is None


def test_screen_predictors_n_workers_2_no_nameerror():
    """screen_predictors with n_workers>1 reaches the pool-warmup line; pre-fix this raised NameError."""
    from mlframe.feature_selection.filters.screen import screen_predictors

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)
    data, nb, names, y_idx = _ordinal_inputs(X, y)
    # Must not raise NameError('_pool_warmup_noop'); threading backend keeps it fast and orphan-free.
    out = screen_predictors(
        factors_data=data, factors_nbins=nb, factors_names=names, y=y_idx,
        n_workers=2, random_seed=42, use_simple_mode=False,
        full_npermutations=10, baseline_npermutations=4, verbose=0,
    )
    assert isinstance(out, tuple) and len(out) == 10


# ---------------------------------------------------------------------------
# Bug 2: evaluate_candidate must unpack the cached_confident_MIs (gain, confidence) tuple
# ---------------------------------------------------------------------------


def test_evaluate_candidate_unpacks_cached_confident_tuple():
    """When X is already in cached_confident_MIs (a (gain, confidence) tuple), evaluate_candidate must use
    the gain, not the whole tuple. Pre-fix this crashed on ``direct_gain > 0``."""
    from mlframe.feature_selection.filters.evaluation import evaluate_candidate

    X = (0,)
    cached_confident_MIs = {X: (0.5, 0.9)}  # (bootstrapped_gain, confidence)
    expected_gains = np.zeros(1, dtype=np.float64)
    # selected_vars empty -> the conditional-MI branch is skipped, so no numba-typed caches are needed.
    gain, _sink = evaluate_candidate(
        cand_idx=0, X=X, y=(1,), nexisting=0, best_gain=-1e9,
        factors_data=np.zeros((10, 2), dtype=np.int32), factors_nbins=np.array([2, 2], dtype=np.int64),
        factors_names=["x0", "y"], expected_gains=expected_gains, partial_gains={}, selected_vars=[],
        baseline_npermutations=2, cached_MIs={}, cached_confident_MIs=cached_confident_MIs,
        cached_cond_MIs={}, entropy_cache={}, use_gpu=False,
    )
    assert gain == pytest.approx(0.5)
    assert expected_gains[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Refactor sanity: shared primitive is wired, greedy still selects the signal.
# ---------------------------------------------------------------------------


def test_confirm_predictor_primitive_wired():
    """screen_predictors imports and delegates to the extracted confirm_one_predictor primitive."""
    from mlframe.feature_selection.filters import _screen_predictors as sp
    from mlframe.feature_selection.filters import _confirm_predictor as cp

    assert sp.confirm_one_predictor is cp.confirm_one_predictor
    assert hasattr(cp, "score_candidates") and hasattr(cp, "confirm_candidate")


def test_greedy_still_selects_signal_after_refactor():
    """End-to-end smoke: MRMR (greedy) recovers the two informative features post-refactor."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    sel = MRMR(verbose=0, random_seed=42, use_simple_mode=False).fit(df, pd.Series(y, name="y"))
    assert {0, 1}.issubset(set(int(i) for i in sel.support_))
