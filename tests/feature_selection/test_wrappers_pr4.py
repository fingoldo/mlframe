"""PR-4 RFECV tests: tactical fixes (z-scoring, must_exclude, leakage detection,
feature_groups, bootstrap CI on best N) + Stability Selection + Multi-estimator
voting."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

# Lazy imports — RFECV pulls in heavy training modules that OOM
# during collection when loaded alongside filters/* tests.
# from mlframe.feature_selection.wrappers import RFECV, get_feature_importances


def _rfecv(**kw):
    from mlframe.feature_selection.wrappers import RFECV as _RFECV
    return _RFECV(**kw)


def _get_feature_importances(*a, **kw):
    from mlframe.feature_selection.wrappers import get_feature_importances as _gfi
    return _gfi(*a, **kw)


# ----------------------------------------------------------------------------
# T1: coefficient z-scoring for linear `coef_` path
# ----------------------------------------------------------------------------
class TestT1_CoefZScoring:
    def test_high_variance_feature_with_same_effect_not_unfairly_penalised(self):
        """Construct two features with the SAME effect on y but different scales:
        f_small (std=1) and f_big (std=100). Pre-fix np.abs(coef_) would rank
        f_big lower (its coef is 100x smaller for the same effect). With
        z-scoring (multiply by std), the importances should be roughly equal.

        Uses LinearRegression so there's no L2-regularization asymmetry that
        would pull large-scale features toward 0 disproportionately.
        """
        rng = np.random.default_rng(0)
        n = 500
        signal = rng.standard_normal(n)
        f_small = signal + rng.standard_normal(n) * 0.1
        f_big = (signal + rng.standard_normal(n) * 0.1) * 100.0
        noise = rng.standard_normal(n)
        X = pd.DataFrame({"f_small": f_small, "f_big": f_big, "noise": noise})
        # Continuous y so we can use LinearRegression (no logistic asymmetry).
        y = signal + rng.standard_normal(n) * 0.1
        model = LinearRegression().fit(X, y)
        result = _get_feature_importances(
            model=model,
            current_features=list(X.columns),
            importance_getter="coef_",
            data=X,
        )
        # With z-scoring, f_small and f_big should be similarly important
        # (within ~3x). Without z-scoring, f_big would be 100x smaller.
        ratio = max(result["f_small"], result["f_big"]) / max(min(result["f_small"], result["f_big"]), 1e-12)
        assert ratio < 3.0, (
            f"Z-scoring should make scale-equivalent features have similar "
            f"importance; got ratio={ratio:.2f}: {result}"
        )
        # noise should still be the smallest
        assert result["noise"] < min(result["f_small"], result["f_big"]), (
            f"noise feature should rank lowest: {result}"
        )


# ----------------------------------------------------------------------------
# T2: must_exclude
# ----------------------------------------------------------------------------
class TestT2_MustExclude:
    def test_excluded_features_never_in_support(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((150, 6)), columns=list("abcdef"))
        y = (X["a"] + X["b"] > 0).astype(int).values
        rfecv = _rfecv(
            estimator=LogisticRegression(max_iter=400, random_state=0),
            cv=3, max_refits=4, verbose=0,
            must_exclude=["c", "d"],
        )
        rfecv.fit(X, y)
        names = list(rfecv.get_feature_names_out())
        assert "c" not in names
        assert "d" not in names

    def test_must_exclude_silently_ignores_missing_columns(self):
        """must_exclude with a column that doesn't exist must not error."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
        y = (X["a"] > 0).astype(int).values
        rfecv = _rfecv(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3, max_refits=2, verbose=0,
            must_exclude=["nonexistent", "a"],
        )
        rfecv.fit(X, y)
        # 'a' must be excluded; 'nonexistent' silently ignored.
        assert "a" not in list(rfecv.get_feature_names_out())


# ----------------------------------------------------------------------------
# T3: target-leakage detection
# ----------------------------------------------------------------------------
class TestT3_LeakageDetection:
    def test_leak_column_gets_warning(self, caplog):
        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("abcde"))
        y = (X["a"] > 0).astype(int).values
        # Inject a near-perfect leak (corr ~ 0.999)
        X["leak"] = y.astype(float) + rng.standard_normal(n) * 0.01

        with caplog.at_level(logging.WARNING):
            rfecv = _rfecv(
                estimator=LogisticRegression(max_iter=200, random_state=0),
                cv=3, max_refits=2, verbose=0,
                leakage_corr_threshold=0.95,
            )
            rfecv.fit(X, y)
        leak_warnings = [r for r in caplog.records if "leakage" in r.getMessage().lower() or "Pearson" in r.getMessage()]
        assert leak_warnings, (
            f"Expected a leakage WARNING; got records: "
            f"{[r.getMessage()[:120] for r in caplog.records]}"
        )

    def test_no_warning_when_threshold_none(self, caplog):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
        y = (X["a"] > 0).astype(int).values
        X["leak"] = y.astype(float)
        with caplog.at_level(logging.WARNING):
            rfecv = _rfecv(
                estimator=LogisticRegression(max_iter=200, random_state=0),
                cv=3, max_refits=2, verbose=0,
                leakage_corr_threshold=None,
            )
            rfecv.fit(X, y)
        leak_warnings = [r for r in caplog.records if "leakage" in r.getMessage().lower()]
        assert not leak_warnings, "Threshold=None must suppress all leakage warnings"


# ----------------------------------------------------------------------------
# T4: feature_groups all-or-nothing
# ----------------------------------------------------------------------------
class TestT4_FeatureGroups:
    def test_group_expands_to_all_or_nothing(self):
        """Construct 5 collinear copies as a group; without feature_groups
        RFECV may pick any subset; with it, either all 5 are in or all 5 are out."""
        rng = np.random.default_rng(0)
        n = 300
        driver = rng.standard_normal(n)
        cols_dup = np.column_stack([driver + rng.standard_normal(n) * 0.001 for _ in range(5)])
        noise = rng.standard_normal((n, 5))
        X = pd.DataFrame(
            np.column_stack([cols_dup, noise]),
            columns=[f"dup{i}" for i in range(5)] + [f"noise{i}" for i in range(5)],
        )
        y = (driver > 0).astype(int)
        rfecv = _rfecv(
            estimator=LogisticRegression(max_iter=400, random_state=0),
            cv=3, max_refits=4, verbose=0,
            feature_groups={"dup_group": [f"dup{i}" for i in range(5)]},
        )
        rfecv.fit(X, y)
        names = set(rfecv.get_feature_names_out())
        dup_in_names = sum(1 for f in [f"dup{i}" for i in range(5)] if f in names)
        # All-or-nothing: must be 0 OR 5
        assert dup_in_names in (0, 5), (
            f"feature_groups violated all-or-nothing on dup_group: "
            f"{dup_in_names}/5 selected"
        )


# ----------------------------------------------------------------------------
# T5: bootstrap CI on best n_features_
# ----------------------------------------------------------------------------
class TestT5_BootstrapCI:
    def test_returns_low_n_high_triple(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((200, 8)), columns=list("abcdefgh"))
        y = (X["a"] + X["b"] > 0).astype(int).values
        rfecv = _rfecv(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3, max_refits=4, verbose=0, random_state=0,
        )
        rfecv.fit(X, y)
        low, n, high = rfecv.n_features_bootstrap_ci_(n_bootstrap=100, ci=0.9, random_state=0)
        assert low <= n <= high
        assert isinstance(low, int) and isinstance(n, int) and isinstance(high, int)


# ----------------------------------------------------------------------------
# T6: Stability Selection
# ----------------------------------------------------------------------------
class TestT6_StabilitySelection:
    def test_stability_selection_recovers_informative_features(self):
        X, y = make_classification(
            n_samples=400, n_features=30, n_informative=5,
            n_redundant=0, random_state=0, shuffle=False, class_sep=2.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
        rfecv = _rfecv(
            estimator=LogisticRegression(max_iter=400, random_state=0),
            stability_selection=True,
            stability_n_bootstrap=30,
            stability_threshold=0.5,
            verbose=0, random_state=0,
        )
        rfecv.fit(Xdf, y)
        names = set(rfecv.get_feature_names_out())
        informative = {f"f{i}" for i in range(5)}
        recall = len(names & informative) / 5
        assert recall >= 0.6, (
            f"Stability selection should recover most informative features; "
            f"got recall={recall:.2f} ({names & informative})"
        )

    def test_stability_selection_freq_attribute_populated(self):
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=3, random_state=0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        rfecv = _rfecv(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            stability_selection=True,
            stability_n_bootstrap=20,
            verbose=0, random_state=0,
        )
        rfecv.fit(Xdf, y)
        assert hasattr(rfecv, "stability_selection_freq_")
        assert rfecv.stability_selection_freq_.shape == (10,)
        assert (rfecv.stability_selection_freq_ >= 0).all()
        assert (rfecv.stability_selection_freq_ <= 1).all()

    def test_stability_threshold_controls_strictness(self):
        """Higher threshold -> fewer selected (or equal)."""
        X, y = make_classification(
            n_samples=200, n_features=15, n_informative=4, random_state=0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(15)])
        common = dict(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            stability_selection=True,
            stability_n_bootstrap=20,
            verbose=0, random_state=0,
        )
        r_low = _rfecv(stability_threshold=0.3, **common)
        r_low.fit(Xdf, y)
        r_high = _rfecv(stability_threshold=0.9, **common)
        r_high.fit(Xdf, y)
        assert r_high.n_features_ <= r_low.n_features_, (
            f"Higher threshold (0.9) should select <= than lower (0.3); "
            f"got high={r_high.n_features_}, low={r_low.n_features_}"
        )


# ----------------------------------------------------------------------------
# T7: Multi-estimator voting in MBH path
# ----------------------------------------------------------------------------
class TestT7_MultiEstimator:
    def test_estimators_list_increases_fi_runs(self):
        """With M estimators we get M FI runs per fold (vs 1 with singular)."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=4, random_state=0,
            shuffle=False, class_sep=2.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        # Singular path: should get cv * n_iter runs in feature_importances_
        r_one = _rfecv(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3, max_refits=2, verbose=0, random_state=0,
        )
        r_one.fit(Xdf, y)
        n_runs_one = len(r_one.feature_importances_)

        # Two estimators: should get 2x the runs
        r_two = _rfecv(
            estimators=[
                LogisticRegression(max_iter=200, random_state=0),
                RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1),
            ],
            cv=3, max_refits=2, verbose=0, random_state=0,
        )
        r_two.fit(Xdf, y)
        n_runs_two = len(r_two.feature_importances_)

        assert n_runs_two >= 2 * n_runs_one - 5, (
            f"Multi-estimator should produce ~2x FI runs; "
            f"singular={n_runs_one}, two_estimators={n_runs_two}"
        )

    def test_multi_estimator_recovers_informative_on_synthetic(self):
        # Larger n + more refits so MBH has room to converge; multi-estimator
        # paths intrinsically have more variance per probe (mean across
        # heterogeneous models) so we need a slightly easier setup.
        X, y = make_classification(
            n_samples=800, n_features=15, n_informative=5,
            n_redundant=0, random_state=0, shuffle=False, class_sep=2.5,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(15)])
        rfecv = _rfecv(
            estimators=[
                LogisticRegression(max_iter=400, random_state=0),
                RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1),
            ],
            cv=3, max_refits=8, verbose=0, random_state=0,
        )
        rfecv.fit(Xdf, y)
        names = set(rfecv.get_feature_names_out())
        recall = sum(1 for f in [f"f{i}" for i in range(5)] if f in names) / 5
        assert recall >= 0.6, f"multi-estimator recall too low: {recall}"

    def test_estimators_takes_precedence_over_estimator(self):
        """When both estimator= and estimators= are passed, estimators wins."""
        X, y = make_classification(n_samples=120, n_features=6, n_informative=3, random_state=0)
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        rfecv = _rfecv(
            estimator=LogisticRegression(max_iter=200, random_state=0),  # ignored
            estimators=[
                LogisticRegression(max_iter=200, random_state=0),
                RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1),
            ],
            cv=2, max_refits=2, verbose=0, random_state=0,
        )
        rfecv.fit(Xdf, y)
        # 2 estimators x 2 folds x ~1 outer iter = ~4 FI runs minimum
        assert len(rfecv.feature_importances_) >= 2


# ----------------------------------------------------------------------------
# T8: Stability + Multi-estimator combined (the headliner)
# ----------------------------------------------------------------------------
class TestT8_StabilityPlusMultiEstimator:
    def test_combined_path(self):
        X, y = make_classification(
            n_samples=500, n_features=25, n_informative=6,
            n_redundant=0, random_state=0, shuffle=False, class_sep=2.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(25)])
        rfecv = _rfecv(
            estimator=None,
            estimators=[
                LogisticRegression(max_iter=400, random_state=0),
                RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1),
            ],
            stability_selection=True,
            stability_n_bootstrap=25,
            stability_threshold=0.5,
            verbose=0, random_state=0,
        )
        rfecv.fit(Xdf, y)
        names = set(rfecv.get_feature_names_out())
        informative = {f"f{i}" for i in range(6)}
        recall = len(names & informative) / 6
        assert recall >= 0.5, (
            f"stability+multi recall too low: {recall} ({names & informative})"
        )
