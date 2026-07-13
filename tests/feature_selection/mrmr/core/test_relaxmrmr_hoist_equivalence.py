"""Wave 13 (3): evaluation.py's RelaxMRMR block hoisted y_col/k_y + selected-column materialize_var
results out of the per-candidate ``evaluate_candidate`` call (now computed once per greedy iteration in
``_evaluation_driver._evaluate_candidates_inner`` and threaded through via ``_relax_*`` kwargs).

Uses the real ``MRMR.fit()`` path (rather than hand-built low-level factors_data/factors_nbins, which
is fragile to construct correctly for the njit MI kernels) so the equivalence and call-count checks
exercise the exact production call site.
"""

from __future__ import annotations

import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest


def _build_xor_fixture(n=600, seed=42):
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n)
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    noise = rng.normal(size=(n, 3))
    y = ((x0 ^ x1) ^ x2).astype(np.int64)
    X = np.column_stack([x0, x1, x2, noise])
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    return df, pd.Series(y, name="y")


def test_biz_val_relaxmrmr_fit_still_completes_and_selects():
    """End-to-end: relaxmrmr_alpha>0 fit still completes with the hoisted call site."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, ys = _build_xor_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            verbose=0,
            random_seed=42,
            n_workers=1,
            max_runtime_mins=1,
            quantization_nbins=8,
            relaxmrmr_alpha=1.0,
        )
        sel.fit(df, ys)
    assert len(sel.get_feature_names_out()) >= 1


def test_relaxmrmr_fit_deterministic_same_seed():
    """Same (X, y, seed) with relaxmrmr_alpha>0 must reproduce the identical selection across repeated
    fits -- pins that hoisting y_col/k_y/sel_cols/sel_nbins introduced no nondeterminism/staleness."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, ys = _build_xor_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel1 = MRMR(verbose=0, random_seed=42, n_workers=1, max_runtime_mins=1, quantization_nbins=8, relaxmrmr_alpha=1.0).fit(df, ys)
        sel2 = MRMR(verbose=0, random_seed=42, n_workers=1, max_runtime_mins=1, quantization_nbins=8, relaxmrmr_alpha=1.0).fit(df, ys)
    assert list(sel1.get_feature_names_out()) == list(sel2.get_feature_names_out())


def test_evaluate_candidates_inner_hoists_materialize_var_once_per_round():
    """Isolated unit test of the driver-level hoist (finding 3's actual target): feed
    ``_evaluate_candidates_inner`` a workload of several candidates with a non-empty ``selected_vars``
    and ``relaxmrmr_alpha>0``, stub out ``evaluate_candidate`` to just record the ``_relax_*`` kwargs it
    receives (skipping the real MI kernels entirely -- this test targets the driver's hoist logic, not
    ``evaluate_candidate``'s own fallback, which the end-to-end fit tests above already cover), and
    confirm: (a) every candidate receives the SAME (by identity) precomputed y_col/sel_cols objects, and
    (b) ``_materialize_var`` is called exactly ``1 + len(selected_vars)`` times total -- once per round,
    not once per candidate."""
    from mlframe.feature_selection.filters import _evaluation_driver as drv_mod
    from mlframe.feature_selection.filters.info_theory import set_relaxmrmr_alpha, get_relaxmrmr_alpha

    n = 200
    n_factors = 4
    rng = np.random.default_rng(11)
    factors_data = rng.integers(0, 4, size=(n, n_factors)).astype(np.int32)
    factors_nbins = np.full(n_factors, 4, dtype=np.int32)
    factors_names = [f"f{i}" for i in range(n_factors)]
    y = rng.integers(0, 2, size=n).astype(np.int64)
    selected_vars = [0, 1]
    workload = [(10, (2,), len(selected_vars)), (11, (3,), len(selected_vars))]  # 2 candidates, 1 round

    received: list = []

    def _stub_evaluate_candidate(**kw):
        received.append((kw.get("_relax_y_col"), kw.get("_relax_sel_cols")))
        return 0.0, set()

    calls = {"n": 0}

    _prev = get_relaxmrmr_alpha()
    set_relaxmrmr_alpha(1.0)
    try:
        from mlframe.feature_selection.filters.evaluation import _materialize_var as _real_mv

        def _counting_mv(*a, **kw):
            calls["n"] += 1
            return _real_mv(*a, **kw)

        with (
            mock.patch("mlframe.feature_selection.filters.evaluation._materialize_var", _counting_mv),
            mock.patch("mlframe.feature_selection.filters._evaluation_driver.tqdmu", lambda it, **kw: it),
        ):
            # evaluate_candidate is imported lazily inside _evaluate_candidates_inner via
            # `from .evaluation import evaluate_candidate` -- patch the SOURCE it resolves from.
            with mock.patch("mlframe.feature_selection.filters.evaluation.evaluate_candidate", side_effect=_stub_evaluate_candidate):
                drv_mod._evaluate_candidates_inner(
                    workload=workload,
                    y=y,
                    best_gain=-1e18,
                    factors_data=factors_data,
                    factors_nbins=factors_nbins,
                    factors_names=factors_names,
                    partial_gains={},
                    selected_vars=selected_vars,
                    baseline_npermutations=5,
                    classes_y=np.array([0, 1]),
                    freqs_y=np.array([1, 1]),
                    use_simple_mode=False,
                    verbose=0,
                    entropy_cache={},
                    cached_cond_MIs={},
                    cached_jmim_MIs={},
                )
    finally:
        set_relaxmrmr_alpha(_prev)

    assert len(received) == 2, f"expected 2 stubbed evaluate_candidate calls, got {len(received)}"
    y_col_0, sel_cols_0 = received[0]
    y_col_1, sel_cols_1 = received[1]
    assert y_col_0 is not None and y_col_0 is y_col_1, "y_col must be the SAME precomputed object across candidates in one round"
    assert sel_cols_0 is not None and sel_cols_0 is sel_cols_1, "sel_cols must be the SAME precomputed object across candidates in one round"
    # 1 (y) + len(selected_vars)=2 -> exactly 3 _materialize_var calls for the WHOLE round, not
    # 2 candidates x 3 = 6 (the pre-hoist per-candidate recompute count).
    assert calls["n"] == 1 + len(selected_vars), f"expected {1 + len(selected_vars)} _materialize_var calls, got {calls['n']}"
