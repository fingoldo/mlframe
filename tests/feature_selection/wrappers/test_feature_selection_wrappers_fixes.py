"""Regression tests for audits/full_audit_2026-07-21/feature_selection_wrappers.md findings W1-W9.

P3's promised meta-validator (SearchConfig/FIConfig/RobustnessConfig field defaults must match
RFECV.__init__'s flat defaults) is included here too, closing the "no validator exists anywhere"
gap W4 flagged in _configs.py's own docstring claim.
"""

from __future__ import annotations

import logging
import pytest

import numpy as np
import pandas as pd

@pytest.fixture(autouse=True, scope="module")
def _suppress_noisy_logging_during_this_module():
    """Suppress WARNING-and-below logging while this module's tests run, then restore.

    A bare module-level ``logging.disable(logging.CRITICAL)`` used to sit here with no matching
    ``logging.disable(logging.NOTSET)`` -- ``logging.disable`` is a process-wide, manager-level
    override (not scoped to one logger), so it fired at IMPORT time and stayed in effect for the
    rest of this pytest-xdist worker's lifetime, silently swallowing every later test's
    ``logger.warning(...)``/``logger.debug(...)`` calls regardless of that test's own
    handler/level setup -- see the identical fix + incident writeup in
    test_x_ml_correctness_meta_fixes.py.
    """
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


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


def test_w2_dispatched_forwards_run_weights_to_tree_and_linear(monkeypatch):
    """W2 dispatched forwards run weights to tree and linear."""
    from mlframe.feature_selection.wrappers import _helpers_importance_agg as mod
    from mlframe.feature_selection.wrappers._enums import VotesAggregation

    seen = {}
    real_tree, real_linear = mod.aggregate_tree, mod.aggregate_linear

    def tree_spy(feature_importances, **kwargs):
        """Record the weights this estimator receives and forward the call."""
        seen["tree"] = kwargs.get("run_weights")
        return real_tree(feature_importances, **kwargs)

    def linear_spy(signed_importances, **kwargs):
        """Record the weights this estimator receives and forward the call."""
        seen["linear"] = kwargs.get("run_weights")
        return real_linear(signed_importances, **kwargs)

    monkeypatch.setattr(mod, "aggregate_tree", tree_spy)
    monkeypatch.setattr(mod, "aggregate_linear", linear_spy)
    run_weights = {"run1": 0.05, "run2": 0.5, "run3": 1.0}
    fi = {"run1": {"a": 5.0, "b": 1.0}, "run2": {"a": 4.0, "b": 1.2}, "run3": {"a": 3.0, "b": 1.1}}
    signed = {"run1": {"a": 5.0, "b": 1.0}, "run2": {"a": -0.1, "b": 1.1}, "run3": {"a": -0.1, "b": 1.0}}

    mod.aggregate_importances_dispatched(fi, "tree", VotesAggregation.Borda, k_cv=1.0, run_weights=run_weights)
    mod.aggregate_importances_dispatched(fi, "linear", VotesAggregation.Borda, signed_importances=signed, run_weights=run_weights)

    assert seen == {"tree": run_weights, "linear": run_weights}


# ---------------------------------------------------------------------------
# W3: permutation-based FI no longer reuses the identical seed across folds/bootstraps
# ---------------------------------------------------------------------------


def _spy_on_get_feature_importances(monkeypatch, module):
    """Wrap ``module.get_feature_importances`` so every call's ``random_state`` is recorded while the real function still runs."""
    from mlframe.feature_selection.wrappers._helpers import get_feature_importances as real_gfi

    seeds = []

    def spy(*args, **kwargs):
        """Record the call and forward it to the real function."""
        seeds.append(kwargs.get("random_state"))
        return real_gfi(*args, **kwargs)

    monkeypatch.setattr(module, "get_feature_importances", spy)
    return seeds


def test_w3_fit_fold_passes_random_state_to_get_feature_importances(monkeypatch):
    """W3 fit fold passes a distinct integer seed per fold to get feature importances, not one shared default."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    from mlframe.feature_selection.wrappers import RFECV
    from mlframe.feature_selection.wrappers.rfecv import _fit_fold as mod

    seeds = _spy_on_get_feature_importances(monkeypatch, mod)
    X, y = make_classification(n_samples=200, n_features=8, n_informative=3, n_redundant=0, shuffle=False, class_sep=2.0, random_state=0)
    RFECV(estimator=LogisticRegression(max_iter=200, random_state=0), cv=3, max_refits=2, verbose=0, random_state=0, leakage_corr_threshold=None).fit(
        pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), y
    )

    assert len(seeds) >= 2, "per-fold permutation importance was never computed; the seed contract is untested"
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == len(seeds), f"folds reused an RNG seed: {seeds}"


def test_w3_stability_select_passes_random_state_to_get_feature_importances(monkeypatch):
    """W3 stability select derives a distinct integer seed per bootstrap, not get_feature_importances' fixed default."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    from mlframe.feature_selection.wrappers import RFECV
    from mlframe.feature_selection.wrappers.rfecv import _stability_select as mod

    seeds = _spy_on_get_feature_importances(monkeypatch, mod)
    X, y = make_classification(n_samples=200, n_features=8, n_informative=3, n_redundant=0, shuffle=False, class_sep=2.0, random_state=0)
    RFECV(
        estimator=LogisticRegression(max_iter=200, random_state=0), stability_selection=True, stability_n_bootstrap=4,
        stability_threshold=0.5, cv=3, max_refits=2, verbose=0, random_state=0, leakage_corr_threshold=None,
    ).fit(pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), y)

    assert len(seeds) >= 2, "stability selection never computed bootstrap importances; the seed contract is untested"
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == len(seeds), f"bootstraps reused an RNG seed: {seeds}"


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
    # Not asserting they differ (could coincide), just that the weighted call path executes cleanly.
    # Weight forwarding into the dummy baseline is pinned by test_w5_fit_fold_forwards_fold_weights_to_dummy_score.


def test_w5_fit_fold_forwards_fold_weights_to_dummy_score(monkeypatch):
    """W5 fit fold forwards the fold's train/test sample weights to the dummy baseline, matching the weighted real-N scores."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers import RFECV
    from mlframe.feature_selection.wrappers.rfecv import _fit_fold as mod

    seen = []
    real_dummy = mod.get_best_dummy_score

    def spy(*args, **kwargs):
        """Record the call and forward it to the real function."""
        seen.append((kwargs.get("train_sample_weight"), kwargs.get("test_sample_weight")))
        return real_dummy(*args, **kwargs)

    monkeypatch.setattr(mod, "get_best_dummy_score", spy)
    X, y = make_classification(n_samples=200, n_features=6, n_informative=3, n_redundant=0, shuffle=False, class_sep=2.0, random_state=0)
    sw = np.random.default_rng(1).uniform(0.5, 2.0, size=200)
    RFECV(estimator=LogisticRegression(max_iter=200, random_state=0), cv=3, max_refits=2, verbose=0, random_state=0, leakage_corr_threshold=None).fit(
        pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), y, sample_weight=sw
    )

    assert seen, "the dummy baseline was never scored; the weight forwarding is untested"
    for train_w, test_w in seen:
        assert train_w is not None and test_w is not None, "dummy baseline dropped the fold sample weights"
        assert len(train_w) < 200 and len(test_w) < 200
        assert np.isin(train_w, sw).all() and np.isin(test_w, sw).all()


# ---------------------------------------------------------------------------
# W6: SFFS swap pass now threads sample_weight into cross_val_score
# ---------------------------------------------------------------------------


def test_w6_sffs_swap_pass_forwards_sample_weight(monkeypatch):
    """W6 sffs swap pass threads the caller's sample_weight into every swap-evaluation cross_val_score call."""
    from types import SimpleNamespace

    import sklearn.model_selection as skms
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers.rfecv import _sffs as mod

    real_cvs = skms.cross_val_score
    seen = []

    def spy(estimator, X, y, **kwargs):
        """Record the call and forward it to the real function."""
        seen.append(kwargs)
        return real_cvs(estimator, X, y, **kwargs)

    monkeypatch.setattr(skms, "cross_val_score", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + 0.1 * rng.normal(size=120) > 0).astype(int)
    sw = rng.uniform(0.5, 2.0, size=120)
    feats = ["f0", "f1", "f2", "f3"]
    fi = {"r0": {"f0": 1.0, "f1": 0.1, "f2": 0.2, "f3": 0.0}}
    self_ = SimpleNamespace(swap_top_k=1, _fit_sample_weight_=sw)

    mod._sffs_swap_pass(
        self_, X, y, LogisticRegression(max_iter=200), 3, "accuracy", 2, 0.5,
        {2: ["f1", "f2"]}, fi, feats, {2: 0.5}, {2: 0.0}, 0, 6,
    )

    assert seen, "the swap pass never evaluated a swap; the weight forwarding is untested"
    for kwargs in seen:
        np.testing.assert_array_equal(kwargs["params"]["sample_weight"], sw)


# ---------------------------------------------------------------------------
# W7: _rank_with_ties/_mann_whitney_u_z/_kruskal_wallis_h documented as a deliberate A/B reference
# ---------------------------------------------------------------------------


def test_w7_reference_kernels_return_textbook_values():
    """W7 reference kernels stay live: on a tie-free input they return the textbook Mann-Whitney U / Kruskal-Wallis H."""
    from mlframe.feature_selection.wrappers._univariate_ht import _kruskal_wallis_h, _mann_whitney_u_z

    x = np.array([1.0, 2.0, 3.0, 4.0])
    # Group 1 holds ranks 3 and 4: U1 = 7 - 3 = 4, mu = 2, var = (2*2/12) * 5 = 5/3.
    u_stat, mu_u, sigma_u = _mann_whitney_u_z(x, np.array([0, 0, 1, 1], dtype=np.int64))
    assert u_stat == pytest.approx(4.0)
    assert mu_u == pytest.approx(2.0)
    assert sigma_u == pytest.approx(np.sqrt(5.0 / 3.0))

    # Rank sums 3 and 7 over groups of 2: H = 12/(4*5) * (3^2/2 + 7^2/2) - 3*5 = 2.4, df = 1.
    h_stat, df = _kruskal_wallis_h(x, np.array([0, 0, 1, 1], dtype=np.int64), 2)
    assert h_stat == pytest.approx(2.4)
    assert df == 1


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


def test_w8_scipy_named_strategies_delegate_to_dichotomic_search():
    """W8 the ScipyLocal/ScipyGlobal strategies run dichotomic search (no scipy optimizer), so they must pick the same N as it."""
    from mlframe.feature_selection.wrappers._enums import OptimumSearch
    from mlframe.feature_selection.wrappers._helpers import _suggest_dichotomic, _suggest_scipy_global, _suggest_scipy_local

    remaining = list(range(1, 21))
    evaluated = {5: 0.5, 10: 0.7, 15: 0.6}
    expected = _suggest_dichotomic(remaining, evaluated, 20, epsilon=0.0, rng=None, step="midpoint")
    assert expected is not None
    assert _suggest_scipy_local(remaining, evaluated, 20, epsilon=0.0, rng=None, step="midpoint") == expected
    assert _suggest_scipy_global(remaining, evaluated, 20, epsilon=0.0, rng=None, step="midpoint") == expected
    assert OptimumSearch.ScipyLocal.value == "ScipyLocal"
    assert OptimumSearch.ScipyGlobal.value == "ScipyGlobal"


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


@pytest.mark.parametrize(
    ("case", "expected_fragment"),
    [
        ("wide", "p=5000 features with no max_nfeatures cap"),
        ("runtime", "max_runtime_mins=0.0010 is < 1 second"),
        ("cv_ge_n", "cv=60 >= n_samples=50"),
    ],
)
def test_w9_misconfiguration_warnings_fire_at_default_verbose(caplog, case, expected_fragment):
    """W9 the wide-data, sub-second-runtime and cv>=n misconfiguration warnings fire under the default verbose=0."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers.rfecv import RFECV
    from mlframe.feature_selection.wrappers.rfecv._validate import _sanitize_X_inputs

    rng = np.random.default_rng(0)
    if case == "wide":
        X = pd.DataFrame(rng.normal(size=(20, 5000)), columns=[f"f{i}" for i in range(5000)])
        y = rng.integers(0, 2, 20)
        sel = RFECV(estimator=LogisticRegression(), cv=3, max_nfeatures=None, verbose=0, random_state=0)
    elif case == "runtime":
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
        y = rng.integers(0, 2, 60)
        sel = RFECV(estimator=LogisticRegression(), cv=3, max_runtime_mins=0.001, verbose=0, random_state=0)
    else:
        X = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
        y = rng.integers(0, 2, 50)
        sel = RFECV(estimator=LogisticRegression(), cv=60, verbose=0, random_state=0)

    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        _sanitize_X_inputs(sel, X, y)
    assert any(expected_fragment in r.getMessage() for r in caplog.records), f"W9 REGRESSION: {case} warning must fire at verbose=0"
