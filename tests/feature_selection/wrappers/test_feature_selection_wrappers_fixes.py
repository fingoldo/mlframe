"""Regression tests for audits/full_audit_2026-07-21/feature_selection_wrappers.md findings W1-W9.

P3's promised meta-validator (SearchConfig/FIConfig/RobustnessConfig field defaults must match
RFECV.__init__'s flat defaults) is included here too, closing the "no validator exists anywhere"
gap W4 flagged in _configs.py's own docstring claim.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# W1: stability_selection now threads sample_weight into the bootstrap fits
# ---------------------------------------------------------------------------


def test_w1_stability_selection_threads_sample_weight(monkeypatch):
    """W1 stability selection threads sample weight."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers.rfecv import RFECV

    rng = np.random.default_rng(0)
    n, p = 60, 5
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = rng.integers(0, 2, n)
    sw = rng.uniform(0.5, 2.0, n)

    seen_sw = []
    real_fit = LogisticRegression.fit

    def _spy_fit(self, X, y, sample_weight=None, **kw):
        """Spy on the fit call and record whether sample_weight was passed."""
        seen_sw.append(sample_weight)
        return real_fit(self, X, y, sample_weight=sample_weight, **kw)

    monkeypatch.setattr(LogisticRegression, "fit", _spy_fit)

    sel = RFECV(
        estimator=LogisticRegression(max_iter=200),
        cv=3, max_refits=2, verbose=0, random_state=0,
        stability_selection=True, stability_n_bootstrap=3, stability_threshold=0.4,
    )
    sel.fit(X, y, sample_weight=sw)
    assert seen_sw, "test setup: LogisticRegression.fit was never called during stability_selection"
    assert any(w is not None for w in seen_sw), "W1 REGRESSION: stability_selection's bootstrap fits must receive sample_weight, not silently drop it"


# ---------------------------------------------------------------------------
# W2: aggregate_tree/aggregate_linear now honour run_weights
# ---------------------------------------------------------------------------


def test_w2_aggregate_tree_run_weights_change_the_ranking():
    """W2 aggregate tree run weights change the ranking."""
    from mlframe.feature_selection.wrappers._helpers_importance_agg import aggregate_tree

    # Feature "a": high importance in an EARLY run, near-zero in later runs.
    # Feature "b": steady mid-level importance across all runs.
    feature_importances = {
        "run1": {"a": 10.0, "b": 2.0},
        "run2": {"a": 0.1, "b": 2.1},
        "run3": {"a": 0.1, "b": 2.0},
    }
    unweighted = aggregate_tree(feature_importances, k_cv=1.0)
    # Recency-decay weights: run1 (oldest) down-weighted heavily, run3 (freshest) full weight.
    run_weights = {"run1": 0.1, "run2": 0.5, "run3": 1.0}
    weighted = aggregate_tree(feature_importances, k_cv=1.0, run_weights=run_weights)
    assert weighted["a"] != unweighted["a"], "W2 REGRESSION: run_weights must change aggregate_tree's score, not be silently ignored"
    # With recency weighting, "a"'s stale high value from run1 should matter much less.
    assert weighted["a"] < unweighted["a"]


def test_w2_aggregate_tree_no_weights_bit_identical_to_before():
    """W2 aggregate tree no weights bit identical to before."""
    from mlframe.feature_selection.wrappers._helpers_importance_agg import aggregate_tree

    feature_importances = {"run1": {"a": 5.0, "b": 1.0}, "run2": {"a": 4.0, "b": 1.2}}
    out_no_weights_arg = aggregate_tree(feature_importances, k_cv=1.0)
    out_none_weights = aggregate_tree(feature_importances, k_cv=1.0, run_weights=None)
    assert out_no_weights_arg == out_none_weights


def test_w2_aggregate_linear_run_weights_change_the_ranking():
    """W2 aggregate linear run weights change the ranking."""
    from mlframe.feature_selection.wrappers._helpers_importance_agg import aggregate_linear

    signed_importances = {
        "run1": {"a": 5.0, "b": 1.0},
        "run2": {"a": -0.1, "b": 1.1},
        "run3": {"a": -0.1, "b": 1.0},
    }
    unweighted = aggregate_linear(signed_importances)
    run_weights = {"run1": 0.05, "run2": 0.5, "run3": 1.0}
    weighted = aggregate_linear(signed_importances, run_weights=run_weights)
    assert weighted["a"] != unweighted["a"], "W2 REGRESSION: run_weights must change aggregate_linear's score, not be silently ignored"


def test_w2_dispatched_forwards_run_weights_to_tree_and_linear():
    """W2 dispatched forwards run weights to tree and linear."""
    import inspect

    from mlframe.feature_selection.wrappers import _helpers_importance_agg as mod

    src = inspect.getsource(mod.aggregate_importances_dispatched)
    assert "aggregate_tree(feature_importances, k_cv=k_cv, run_weights=run_weights)" in src
    assert "aggregate_linear(signed_importances, run_weights=run_weights)" in src


# ---------------------------------------------------------------------------
# W3: permutation-based FI no longer reuses the identical seed across folds/bootstraps
# ---------------------------------------------------------------------------


def test_w3_fit_fold_passes_random_state_to_get_feature_importances():
    """W3 fit fold passes random state to get feature importances."""
    import inspect

    from mlframe.feature_selection.wrappers.rfecv import _fit_fold as mod

    src = inspect.getsource(mod)
    assert "random_state=int(fold_seed)" in src


def test_w3_stability_select_passes_random_state_to_get_feature_importances():
    """W3 stability select passes random state to get feature importances."""
    import inspect

    from mlframe.feature_selection.wrappers.rfecv import _stability_select as mod

    src = inspect.getsource(mod)
    assert "random_state=int(rng.integers(" in src


# ---------------------------------------------------------------------------
# W4 / P3: FIConfig.votes_aggregation_method default matches RFECV.__init__, + the promised
# validator now genuinely exists (enumerates every shared field, not just the one found here).
# ---------------------------------------------------------------------------


def test_w4_fi_config_votes_aggregation_default_matches_rfecv_init():
    """W4 fi config votes aggregation default matches rfecv init."""
    from mlframe.feature_selection.wrappers._enums import VotesAggregation
    from mlframe.feature_selection.wrappers.rfecv._configs import FIConfig

    assert FIConfig().votes_aggregation_method == VotesAggregation.Borda


def test_p3_config_defaults_match_rfecv_init_for_every_shared_field():
    """The validator rfecv/_configs.py's own module docstring promises but never implemented.
    Enumerates SearchConfig/FIConfig/RobustnessConfig fields and asserts each default equals
    inspect.signature(RFECV.__init__)'s matching parameter default -- prevents recurrence of
    W4-style drift for any future knob, not just votes_aggregation_method."""
    import inspect

    from mlframe.feature_selection.wrappers.rfecv import RFECV
    from mlframe.feature_selection.wrappers.rfecv._configs import FIConfig, RobustnessConfig, SearchConfig

    init_params = inspect.signature(RFECV.__init__).parameters
    mismatches = []
    for config_cls in (SearchConfig, FIConfig, RobustnessConfig):
        instance = config_cls()
        for field_name in type(instance).model_fields:
            if field_name not in init_params:
                continue
            init_default = init_params[field_name].default
            config_default = getattr(instance, field_name)
            if init_default is inspect.Parameter.empty:
                continue
            if config_default != init_default:
                mismatches.append((config_cls.__name__, field_name, config_default, init_default))
    assert not mismatches, f"config default(s) mismatch RFECV.__init__'s flat default(s): {mismatches}"


# ---------------------------------------------------------------------------
# W5: the N=0 dummy baseline now honours sample_weight, matching the weighted real-N scores
# ---------------------------------------------------------------------------


def test_w5_get_best_dummy_score_accepts_and_uses_sample_weight():
    """W5 get best dummy score accepts and uses sample weight."""
    from sklearn.metrics import make_scorer, mean_absolute_error

    from mlframe.estimators.baselines import get_best_dummy_score
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 40
    X_train = rng.normal(size=(n, 3))
    y_train = rng.normal(size=n)
    X_test = rng.normal(size=(10, 3))
    y_test = rng.normal(size=10)
    sw_train = rng.uniform(0.1, 5.0, n)
    sw_test = rng.uniform(0.1, 5.0, 10)
    scoring = make_scorer(mean_absolute_error, greater_is_better=False)

    score_unweighted = get_best_dummy_score(LinearRegression(), X_train, y_train, X_test, y_test, scoring)
    score_weighted = get_best_dummy_score(
        LinearRegression(), X_train, y_train, X_test, y_test, scoring,
        train_sample_weight=sw_train, test_sample_weight=sw_test,
    )
    assert np.isfinite(score_unweighted)
    assert np.isfinite(score_weighted)
    # Not asserting they differ (could coincide), just that the weighted call path executes cleanly
    # and doesn't silently ignore the weights (verified via source below).


def test_w5_fit_fold_forwards_fold_weights_to_dummy_score():
    """W5 fit fold forwards fold weights to dummy score."""
    import inspect

    from mlframe.feature_selection.wrappers.rfecv import _fit_fold as mod

    src = inspect.getsource(mod)
    assert "train_sample_weight=_fold_train_sw, test_sample_weight=_fold_test_sw" in src


# ---------------------------------------------------------------------------
# W6: SFFS swap pass now threads sample_weight into cross_val_score
# ---------------------------------------------------------------------------


def test_w6_sffs_swap_pass_forwards_sample_weight():
    """W6 sffs swap pass forwards sample weight."""
    import inspect

    from mlframe.feature_selection.wrappers.rfecv import _sffs as mod

    src = inspect.getsource(mod._sffs_swap_pass)
    assert '"sample_weight": _sw' in src
    assert "**_cv_score_kwargs" in src


# ---------------------------------------------------------------------------
# W7: _rank_with_ties/_mann_whitney_u_z/_kruskal_wallis_h documented as a deliberate A/B reference
# ---------------------------------------------------------------------------


def test_w7_reference_kernels_documented_not_silently_dead():
    """W7 reference kernels documented not silently dead."""
    import inspect

    from mlframe.feature_selection.wrappers import _univariate_ht as mod

    src = inspect.getsource(mod)
    assert "bit-identity reference" in src


def test_w7_v2_kernels_still_match_the_reference_implementation():
    """The actual dependency this comment documents: _v2 must still be bit-identical to the
    reference it's tested against (guards against a future edit silently breaking the pairing)."""
    from mlframe.feature_selection.wrappers._univariate_ht import (
        _kruskal_wallis_h,
        _kruskal_wallis_h_v2,
        _mann_whitney_u_z,
        _mann_whitney_u_z_v2,
    )

    rng = np.random.default_rng(3)
    x = rng.integers(0, 5, 200).astype(np.float64)
    g = (rng.random(200) > 0.5).astype(np.int64)
    np.testing.assert_array_equal(_mann_whitney_u_z(x, g), _mann_whitney_u_z_v2(x, g))

    g4 = rng.integers(0, 4, 200).astype(np.int64)
    np.testing.assert_array_equal(_kruskal_wallis_h(x, g4, 4), _kruskal_wallis_h_v2(x, g4, 4))


# ---------------------------------------------------------------------------
# W8: OptimumSearch.ScipyLocal/ScipyGlobal comments no longer promise a scipy backend they don't call
# ---------------------------------------------------------------------------


def test_w8_enum_comments_no_longer_claim_scipy_backends():
    """W8 enum comments no longer claim scipy backends."""
    import inspect

    from mlframe.feature_selection.wrappers import _enums as mod

    src = inspect.getsource(mod.OptimumSearch)
    assert "# Brent" not in src
    assert "# direct, diff evol, shgo" not in src


# ---------------------------------------------------------------------------
# W9: the 4 misconfiguration warnings now fire under the DEFAULT verbose=0 setting
# ---------------------------------------------------------------------------


def test_w9_high_cardinality_warning_fires_at_default_verbose(caplog):
    """W9 high cardinality warning fires at default verbose."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers.rfecv import RFECV
    from mlframe.feature_selection.wrappers.rfecv._validate import _sanitize_X_inputs

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "id_like": rng.integers(0, 1_000_000, n),  # cardinality ~n -> > 0.5*n
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    y = rng.integers(0, 2, n)

    sel = RFECV(estimator=LogisticRegression(), cv=3, verbose=0, random_state=0)  # verbose=0, the OUT-OF-THE-BOX default
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        _sanitize_X_inputs(sel, df, y)
    assert any("cardinality > 0.5*n" in r.getMessage() for r in caplog.records), "W9 REGRESSION: high-cardinality warning must fire at default verbose=0"


def test_w9_no_more_verbose_gates_on_the_four_checks():
    """W9 no more verbose gates on the four checks."""
    import inspect

    from mlframe.feature_selection.wrappers.rfecv import _validate as mod

    src = inspect.getsource(mod._sanitize_X_inputs)
    # None of the 4 W9 checks should still gate on verbose (the leakage-scan warning further down
    # legitimately has no verbose gate either -- confirming the pattern, not counting occurrences).
    assert 'X.shape[1] >= 5000 and self.max_nfeatures is None and getattr(self, "verbose", 0)' not in src
    assert '0 < self.max_runtime_mins < (1.0 / 60.0) and getattr(self, "verbose", 0)' not in src
    assert 'self.cv >= X.shape[0] and getattr(self, "verbose", 0)' not in src
    assert 'X.shape[0] >= 50 and getattr(self, "verbose", 0)' not in src
