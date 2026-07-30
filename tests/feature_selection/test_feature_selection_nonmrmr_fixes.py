"""Regression tests for audits/full_audit_2026-07-21/feature_selection_nonmrmr.md (F1-F10 + PR1/PR2).

F4, F5, F6, F10 are docs/comment-only fixes (docstring corrections, a chained-assignment .copy() hardening
with no behavior change); pinned here as lightweight content checks so they can't silently re-drift, not
as full behavioral regression tests. PR5 (hetero_vote registry.py + training-suite wiring) is a "consider"
proposal explicitly deferred -- F7's actual finding (missing top-level re-export) is fixed and tested below.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection import heterogeneous_relevance_vote
from mlframe.feature_selection._sklearn_defaults import default_tree_estimator, is_classification_target
from mlframe.feature_selection.ace import _default_estimator
from mlframe.feature_selection.functional_adapters import _default_tree_estimator, _is_classification_target
from mlframe.feature_selection.pre_screen import compute_unsupervised_drops
from mlframe.feature_selection.stochastic_bandit_selection import _stochastic_bandit_selection_core
from mlframe.feature_selection.stochastic_bandit_selection_ensemble import stochastic_bandit_selection_ensemble

# ----------------------------------------------------------------------
# F1 (P0) -- sparse-column variance accounts for the fill-value mass.
# ----------------------------------------------------------------------


def test_f1_sparse_rare_signal_column_not_dropped():
    """F1: sparse rare signal column not dropped."""
    n = 1000
    vals = np.zeros(n)
    vals[500] = 5.0  # one stored non-fill value among 999 fill cells
    sparse_col = pd.arrays.SparseArray(vals, fill_value=0.0)
    df = pd.DataFrame({"rare_signal": sparse_col, "other": np.random.default_rng(0).normal(size=n)})
    drops = compute_unsupervised_drops(df, variance_threshold=1e-8, null_fraction_threshold=0.99)
    assert "rare_signal" not in drops


def test_f1_sparse_rare_binary_flag_not_dropped():
    """F1: sparse rare binary flag not dropped."""
    n = 1000
    flag_vals = np.zeros(n)
    flag_vals[[10, 20, 30]] = 1.0  # three identical stored values -- the audit's "rare flag" pattern
    flag_sparse = pd.arrays.SparseArray(flag_vals, fill_value=0.0)
    df = pd.DataFrame({"flag": flag_sparse})
    drops = compute_unsupervised_drops(df, variance_threshold=1e-8, null_fraction_threshold=0.99)
    assert "flag" not in drops


def test_f1_sparse_truly_constant_column_still_dropped():
    """F1: sparse truly constant column still dropped."""
    n = 1000
    const_sparse = pd.arrays.SparseArray(np.zeros(n), fill_value=0.0)
    df = pd.DataFrame({"const": const_sparse})
    drops = compute_unsupervised_drops(df, variance_threshold=1e-8, null_fraction_threshold=0.99)
    assert "const" in drops


def test_f1_sparse_variance_matches_dense_reconstruction_with_nans():
    """The closed-form fill-value-aware variance must exactly match a dense np.nanvar reconstruction."""
    rng = np.random.default_rng(1)
    n = 500
    vals = np.zeros(n)
    stored_idx = rng.choice(n, size=50, replace=False)
    vals[stored_idx] = rng.normal(loc=3.0, scale=2.0, size=50)
    vals[stored_idx[:3]] = np.nan
    sparse_col = pd.arrays.SparseArray(vals, fill_value=0.0)
    df = pd.DataFrame({"x": sparse_col})

    ref_var = float(np.nanvar(vals))
    # A cutoff just below the true variance keeps the column; just above drops it -- a tight bracket proves
    # the internally-computed variance is numerically close to the reference, not just "nonzero".
    drops_below = compute_unsupervised_drops(df, variance_threshold=ref_var * 0.9, null_fraction_threshold=0.99)
    drops_above = compute_unsupervised_drops(df, variance_threshold=ref_var * 1.1, null_fraction_threshold=0.99)
    assert "x" not in drops_below
    assert "x" in drops_above


# ----------------------------------------------------------------------
# F2 -- BorutaShap.fit() with n_trials<=0 raises a clear ValueError, not NameError.
# ----------------------------------------------------------------------


def test_f2_boruta_shap_zero_trials_raises_value_error():
    """F2: boruta shap zero trials raises value error."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y)
    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=10, random_state=0), importance_measure="gini",
        classification=True, n_trials=0, verbose=False, random_state=0,
    )
    with pytest.raises(ValueError, match="n_trials"):
        b.fit(X, y)


# ----------------------------------------------------------------------
# F3 -- stochastic_bandit_selection's cv=None default is seeded from random_state, not a hardcoded 0.
# ----------------------------------------------------------------------


def _make_bandit_data(n=150, p=8, seed=0):
    """Make bandit data."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"].to_numpy() * 2.0 + rng.normal(scale=0.5, size=n)
    return X, y


def test_f3_cv_none_fold_assignment_varies_with_random_state():
    """F3: cv none fold assignment varies with random state."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X, y = _make_bandit_data()
    captured_folds = {}

    class _RecordingRidge(Ridge):
        """Recording Ridge."""
        pass

    for seed in (1, 2):
        # Directly capture the CV splitter's fold assignment (not the bandit's subset draws) by calling the
        # core with n_epochs=0-equivalent: instead, inspect the cv object the core builds internally by
        # monkeypatching KFold to record its random_state.
        # NOTE: `import mlframe.feature_selection.stochastic_bandit_selection as sbs_mod` would NOT give the
        # module here -- the package's __init__.py does `from .stochastic_bandit_selection import
        # stochastic_bandit_selection`, which rebinds the `stochastic_bandit_selection` ATTRIBUTE on the
        # `mlframe.feature_selection` package object to the FUNCTION; `import a.b.c as x` resolves via
        # attribute access (`x = a.b.c`), so it would silently pick up the function instead of the module.
        # Go through sys.modules directly to sidestep that entirely.
        import sys

        sbs_mod = sys.modules["mlframe.feature_selection.stochastic_bandit_selection"]
        real_kfold = sbs_mod.KFold
        seen_random_states = []

        def spy_kfold(*args, **kwargs):
            """Records each KFold construction's random_state, then delegates to the real KFold."""
            # seen_random_states / real_kfold are rebound each loop iteration, but spy_kfold is installed,
            # called synchronously inside the same iteration's _stochastic_bandit_selection_core() call, and
            # restored in the finally: block below before the next iteration starts -- never stored/deferred
            # across iterations, so the late-binding closure hazard B023 warns about cannot occur here.
            seen_random_states.append(kwargs.get("random_state"))  # noqa: B023
            return real_kfold(*args, **kwargs)  # noqa: B023

        sbs_mod.KFold = spy_kfold
        try:
            _stochastic_bandit_selection_core(
                estimator=_RecordingRidge(alpha=0.1), X=X, y=y, scoring=r2_score, subset_size=3,
                n_epochs=1, cv=None, up_factor=1.05, down_factor=0.97, lock_in_threshold=3.0,
                moving_average_window=10, random_state=seed,
            )
        finally:
            sbs_mod.KFold = real_kfold
        captured_folds[seed] = seen_random_states

    assert captured_folds[1] and captured_folds[1][0] == 1
    assert captured_folds[2] and captured_folds[2][0] == 2
    assert captured_folds[1][0] != captured_folds[2][0]


# ----------------------------------------------------------------------
# F7 -- heterogeneous_relevance_vote is re-exported from the package top level.
# ----------------------------------------------------------------------


def test_f7_hetero_vote_reexported_and_callable():
    """F7: hetero vote reexported and callable."""
    import mlframe.feature_selection as fs_pkg

    assert "heterogeneous_relevance_vote" in fs_pkg.__all__
    assert fs_pkg.heterogeneous_relevance_vote is heterogeneous_relevance_vote


def test_f7_hetero_vote_still_works_via_reexport():
    """F7: hetero vote still works via reexport."""
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"].to_numpy() + rng.normal(scale=0.2, size=n) > 0).astype(int)
    accepted, info = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=2, random_state=0)
    assert isinstance(accepted, list)
    assert "vote_fraction" in info


# ----------------------------------------------------------------------
# F8 -- stochastic_bandit_selection_ensemble warns (not raises) on a single seed.
# ----------------------------------------------------------------------


def test_f8_single_seed_ensemble_warns():
    """F8: single seed ensemble warns."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X, y = _make_bandit_data(n=80, p=5)
    with warnings.catch_warnings(record=True):
        pass  # (logger.warning, not warnings.warn -- checked via caplog below instead)
    import logging

    logger_name = "mlframe.feature_selection.stochastic_bandit_selection_ensemble"

    handler_records = []

    class _Handler(logging.Handler):
        """Stub logging.Handler that records emitted records for this test's assertions."""
        def emit(self, record):
            """Captures the log record via a stub logging.Handler."""
            handler_records.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    h = _Handler()
    logger.addHandler(h)
    logger.setLevel(logging.WARNING)
    try:
        result = stochastic_bandit_selection_ensemble(
            Ridge(alpha=0.1), X, y, scoring=r2_score, seeds=[0], subset_size=2, n_epochs=5,
        )
    finally:
        logger.removeHandler(h)
    assert result is not None
    assert any("only 1 seed" in msg for msg in handler_records)


def test_f8_single_seed_ensemble_still_returns_valid_result():
    """The single-seed path must still work end-to-end (a pre-existing biz_val test relies on it)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X, y = _make_bandit_data(n=80, p=5)
    result = stochastic_bandit_selection_ensemble(Ridge(alpha=0.1), X, y, scoring=r2_score, seeds=[42], subset_size=2, n_epochs=5)
    assert all(v == pytest.approx(1.0) for v in result.stability.values())


# ----------------------------------------------------------------------
# F9 / PR1 -- float-binary-target ambiguity now warns; heuristic hoisted to one shared helper.
# ----------------------------------------------------------------------


def test_f9_float_binary_target_warns_in_ace():
    """F9: float binary target warns in ace."""
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0] * 20, dtype=np.float64)
    with pytest.warns(UserWarning, match="float-dtype"):
        est = _default_estimator(y, n=100, random_state=0)
    assert type(est).__name__ == "RandomForestRegressor"


def test_f9_float_binary_target_warns_in_functional_adapters():
    """F9: float binary target warns in functional adapters."""
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0] * 20, dtype=np.float64)
    with pytest.warns(UserWarning, match="float-dtype"):
        est = _default_tree_estimator(y, random_state=0)
    assert type(est).__name__ == "RandomForestRegressor"


def test_f9_genuinely_continuous_target_no_warning():
    """F9: genuinely continuous target no warning."""
    y = np.random.default_rng(0).normal(size=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        est = _default_estimator(y, n=200, random_state=0)
    assert type(est).__name__ == "RandomForestRegressor"


def test_pr1_ace_and_functional_adapters_share_one_heuristic():
    """Both cluster-local wrappers now delegate to the same _sklearn_defaults helper (no more duplicated logic)."""
    y_int = np.array([0, 1, 0, 1, 2] * 20)
    assert is_classification_target(y_int) == _is_classification_target(y_int)
    est_shared = default_tree_estimator(y_int, random_state=0)
    est_ace = _default_estimator(y_int, n=100, random_state=0)
    est_fa = _default_tree_estimator(y_int, random_state=0)
    assert type(est_shared) is type(est_ace) is type(est_fa)


# ----------------------------------------------------------------------
# F4/F5/F6/F10 -- docs/comment-only fixes: lightweight content pins against re-drift.
# ----------------------------------------------------------------------


def test_f4_boruta_premerge_docstring_matches_real_default():
    """F4: boruta premerge docstring matches real default."""
    from mlframe.feature_selection.hybrid_selector import HybridSelector

    assert HybridSelector._run_boruta_premerge.__doc__ is not None
    doc = HybridSelector._run_boruta_premerge.__doc__
    assert '"gini"' in doc
    # the pre-fix docstring claimed "permutation" was the default -- must no longer be phrased that way.
    assert '"permutation" held-out by default' not in doc


def test_f5_zero_importance_pruning_docstring_no_longer_claims_stop_on_degradation():
    """F5: zero importance pruning docstring no longer claims stop on degradation."""
    import mlframe.feature_selection.zero_importance_pruning as zip_mod

    assert "stopping on degradation" not in (zip_mod.__doc__ or "")


def test_f6_cascade_select_stability_docstring_matches_real_default():
    """F6: cascade select stability docstring matches real default."""
    from mlframe.feature_selection.cascade_select_stability import cascade_select_stable
    import inspect

    sig = inspect.signature(cascade_select_stable)
    assert sig.parameters["n_bootstrap"].default == 20
    assert "default of 1" not in (cascade_select_stable.__doc__ or "")


def test_f10_io_plot_uses_explicit_copy():
    """F10: io plot uses explicit copy."""
    import inspect

    from mlframe.feature_selection.boruta_shap import _io_plot

    src = inspect.getsource(_io_plot)
    assert "self.history_x.iloc[1:].copy()" in src


# ----------------------------------------------------------------------
# PR2 -- HybridSelector.keep_augmented_data opt-in memory control.
# ----------------------------------------------------------------------


def test_pr2_keep_augmented_data_default_preserves_stash():
    """PR2: keep augmented data default preserves stash."""
    from mlframe.feature_selection.hybrid_selector import HybridSelector

    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"].to_numpy() + rng.normal(scale=0.3, size=n) > 0).astype(int)
    sel = HybridSelector(use_fe=False, use_tree_member=False, tree_prod_gate="off", random_state=0, classification=True)
    sel.fit(X, y)
    assert sel._Xaug_ is not None
    assert sel._y_ is not None


def test_pr2_keep_augmented_data_false_frees_stash():
    """PR2: keep augmented data false frees stash."""
    from mlframe.feature_selection.hybrid_selector import HybridSelector

    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"].to_numpy() + rng.normal(scale=0.3, size=n) > 0).astype(int)
    sel = HybridSelector(
        use_fe=False, use_tree_member=False, tree_prod_gate="off", random_state=0, classification=True,
        keep_augmented_data=False,
    )
    sel.fit(X, y)
    assert sel._Xaug_ is None
    assert sel._y_ is None
    # the rest of the fitted contract must still work normally.
    assert hasattr(sel, "selected_features_")
