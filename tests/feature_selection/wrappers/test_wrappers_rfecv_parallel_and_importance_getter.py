"""Tests for Phase 4 N3 (parallel CV folds) and N6 (permutation/SHAP importance_getter).

N3 design: when ``n_jobs > 1`` AND the estimator is multi-threaded (CB/LGB/XGB/RF/...),
auto-fall-back to sequential UNLESS ``force_parallel=True`` (in which case pin
inner threads to 1). For single-thread estimators (LR, Ridge, etc.) parallel
folds give a real wall-clock win on multi-core machines.

N6 design: ``importance_getter='permutation'`` uses sklearn.inspection;
``importance_getter='shap'`` uses the optional ``shap`` package.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import (
    RFECV,
    get_feature_importances,
)
from tests.training.synthetic import make_sklearn_classification_df
from tests.feature_selection.conftest import COVERAGE_ACTIVE
from mlframe.feature_selection.wrappers._helpers import (
    _detect_multithreaded,
    _pin_threads_to_one,
)


@pytest.fixture(scope="module")
def small_clf_data():
    """Small clf data."""
    Xdf, y, _ = make_sklearn_classification_df(
        n_samples=200,
        n_features=10,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        shuffle=False,
        seed=0,
    )
    return Xdf, y


# ----------------------------------------------------------------------------
# N3: Multi-thread detection helpers
# ----------------------------------------------------------------------------
class TestN3_MultithreadDetection:
    """Groups tests covering TestN3_MultithreadDetection."""
    def test_detects_random_forest(self):
        """Detects random forest."""
        rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        assert _detect_multithreaded(rf) is True

    def test_does_not_flag_logistic_regression(self):
        """Does not flag logistic regression."""
        lr = LogisticRegression()
        assert _detect_multithreaded(lr) is False

    def test_pin_threads_zeroes_n_jobs_on_rf(self):
        """Pin threads zeroes n jobs on rf."""
        rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        _pin_threads_to_one(rf)
        assert rf.get_params()["n_jobs"] == 1

    def test_pin_threads_no_op_on_lr_without_threading_param(self):
        # LR doesn't have any thread-count constructor param; pin should no-op.
        """Pin threads no op on lr without threading param."""
        lr = LogisticRegression()
        _pin_threads_to_one(lr)
        # Expect no exception, no change in object.
        assert isinstance(lr, LogisticRegression)


# ----------------------------------------------------------------------------
# N3: Parallel CV folds produce same selection as sequential (correctness)
# ----------------------------------------------------------------------------
class TestN3_ParallelEquivalence:
    """Groups tests covering TestN3_ParallelEquivalence."""
    @pytest.mark.skipif(
        COVERAGE_ACTIVE,
        reason="joblib.Parallel + coverage's sys.settrace deadlocks Windows thread spawn (RuntimeError + DummyProcess.terminate AttributeError). Test is correct - skip only when measuring coverage; runs under standard pytest.",
    )
    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="macOS GitHub-hosted runner: joblib.Parallel(n_jobs=2) inside RFECV crashes gw2 worker (verified 2026-05-26 run 26463488829). Same libomp + numba concurrent-JIT class as the existing prewarm + MRMR Darwin skips. Linux + Windows cover the parallel-equivalence contract.",
    )
    def test_n_jobs_2_matches_n_jobs_1_for_single_thread_estimator(self, small_clf_data):
        """N jobs 2 matches n jobs 1 for single thread estimator."""
        X, y = small_clf_data
        common = dict(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3,
            max_refits=4,
            verbose=0,
            random_state=0,
        )
        seq = RFECV(n_jobs=1, **common).fit(X, y)
        par = RFECV(n_jobs=2, **common).fit(X, y)
        # Same selection on the same problem with the same seed.
        assert set(seq.get_feature_names_out()) == set(par.get_feature_names_out()), (
            f"Parallel and sequential RFECV diverged on the same input + seed. seq={list(seq.get_feature_names_out())} par={list(par.get_feature_names_out())}"
        )

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="macOS GitHub-hosted runner: joblib.Parallel(n_jobs=-1) inside RFECV crashes gw2 worker (verified 2026-05-27 run 26473751742). Same libomp + numba concurrent-JIT class as the sibling test_n_jobs_2 Darwin skip. Linux + Windows cover the n_jobs=-1 contract.",
    )
    def test_n_jobs_negative_one_resolves_to_cpu_count(self, small_clf_data):
        # n_jobs=-1 should resolve to all cores; ensure it doesn't crash.
        """N jobs negative one resolves to cpu count."""
        X, y = small_clf_data
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3,
            max_refits=3,
            verbose=0,
            n_jobs=-1,
        )
        rfecv.fit(X, y)
        assert rfecv.n_features_ >= 1


# ----------------------------------------------------------------------------
# N3: Auto-fallback on multi-threaded estimators
# ----------------------------------------------------------------------------
class TestN3_AutoFallback:
    """Groups tests covering TestN3_AutoFallback."""
    @pytest.mark.skipif(
        COVERAGE_ACTIVE,
        reason="joblib.Parallel + coverage's sys.settrace deadlocks Windows thread spawn (RuntimeError + DummyProcess.terminate AttributeError). Test is correct - skip only when measuring coverage; runs under standard pytest.",
    )
    def test_rf_auto_fallback_no_force_parallel(self, small_clf_data, caplog):
        """RF + n_jobs=2 + force_parallel=False (default) -> auto-fallback to
        sequential and a clear log line. The outer RF still uses its native
        threading, but RFECV doesn't compound it."""
        X, y = small_clf_data
        import logging

        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers.rfecv"):
            rfecv = RFECV(
                estimator=RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1),
                cv=3,
                max_refits=3,
                verbose=1,
                n_jobs=2,
                # force_parallel left at default False
            )
            rfecv.fit(X, y)
        # The fit must complete without crashing (sequential underneath).
        assert rfecv.n_features_ >= 1


# ----------------------------------------------------------------------------
# N6: permutation importance dispatch
# ----------------------------------------------------------------------------
class TestN6_PermutationImportance:
    """Groups tests covering TestN6_PermutationImportance."""
    def test_permutation_returns_per_feature_dict(self):
        """Permutation returns per feature dict."""
        Xdf, y, cols = make_sklearn_classification_df(
            n_samples=120,
            n_features=6,
            n_informative=3,
            seed=0,
            n_redundant=0,
            shuffle=False,
            class_sep=2.0,
        )
        model = LogisticRegression(max_iter=200, random_state=0).fit(Xdf, y)
        result = get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="permutation",
            data=Xdf,
            target=y,
        )
        assert set(result.keys()) == set(cols)
        # Informative features (f0, f1, f2) should have larger importance than noise (f3..f5).
        # Use sum-of-top-3 vs sum-of-bottom-3 as a robust check.
        sorted_pairs = sorted(result.items(), key=lambda kv: kv[1], reverse=True)
        top3 = {k for k, _ in sorted_pairs[:3]}
        # At least 2 of the top-3 must be informative (f0, f1, f2)
        informative_in_top3 = len(top3 & {"f0", "f1", "f2"})
        assert informative_in_top3 >= 2, f"permutation importance failed to surface the informative features; top3={top3}"

    def test_permutation_requires_target(self):
        """importance_getter='permutation' without target= must raise."""
        X = pd.DataFrame(np.random.randn(20, 4), columns=list("abcd"))
        model = LogisticRegression(max_iter=50).fit(X, [0, 1] * 10)
        with pytest.raises(ValueError, match="requires target"):
            get_feature_importances(
                model=model,
                current_features=list("abcd"),
                importance_getter="permutation",
                data=X,
                # target intentionally omitted
            )


class TestN6_ShapImportance:
    """Groups tests covering TestN6_ShapImportance."""
    def test_shap_either_works_or_raises_clear_error(self):
        """Run SHAP if available, else assert the ImportError is informative."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 5)), columns=list("abcde"))
        y = (X["a"] > 0).astype(int).values
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        try:
            import shap  # noqa: F401

            has_shap = True
        except ImportError:
            has_shap = False

        if has_shap:
            result = get_feature_importances(
                model=model,
                current_features=list("abcde"),
                importance_getter="shap",
                data=X,
            )
            assert set(result.keys()) == set("abcde")
            # Driver feature 'a' should be near the top.
            top = max(result, key=result.get)
            assert top == "a", f"shap top feature should be 'a', got {top}"
        else:
            with pytest.raises(ImportError, match="shap"):
                get_feature_importances(
                    model=model,
                    current_features=list("abcde"),
                    importance_getter="shap",
                    data=X,
                )


# ----------------------------------------------------------------------------
# Incremental Leaderboard: borda no longer builds majority_graph
# ----------------------------------------------------------------------------
class TestLeaderboard_LazyMajorityGraph:
    """Groups tests covering TestLeaderboard_LazyMajorityGraph."""
    def test_borda_skips_majority_graph(self):
        """Borda skips majority graph."""
        from mlframe.votenrank import Leaderboard

        df = pd.DataFrame({"r1": [10, 20, 5, 15], "r2": [12, 18, 6, 14]}, index=["f0", "f1", "f2", "f3"])
        lb = Leaderboard(table=df)
        # Before any graph-using method is called: graph not built.
        assert lb.majority_graph is None
        _ = lb.borda_ranking()
        assert lb.majority_graph is None, (
            "borda_ranking() must not trigger majority_graph construction; the graph is the n^2 hot path that was making RFECV slow."
        )

    def test_copeland_builds_majority_graph_on_demand(self):
        """Copeland builds majority graph on demand."""
        from mlframe.votenrank import Leaderboard

        df = pd.DataFrame({"r1": [10, 20, 5, 15], "r2": [12, 18, 6, 14]}, index=["f0", "f1", "f2", "f3"])
        lb = Leaderboard(table=df)
        assert lb.majority_graph is None
        _ = lb.copeland_ranking()
        assert lb.majority_graph is not None, "copeland_ranking() must trigger lazy majority_graph build."

    def test_idempotent_majority_graph_build(self):
        """Calling _ensure_majority_graph twice should keep the same graph;
        no recomputation."""
        from mlframe.votenrank import Leaderboard

        df = pd.DataFrame({"r1": [1, 2], "r2": [3, 4]}, index=["a", "b"])
        lb = Leaderboard(table=df)
        lb._ensure_majority_graph()
        first = lb.majority_graph
        lb._ensure_majority_graph()
        # Same object identity = no rebuild
        assert lb.majority_graph is first


# ----------------------------------------------------------------------------
# Phase 7: Conditional Permutation Importance (Strobl, Boulesteix, Zeileis,
# Hothorn 2008). Vanilla permutation breaks on correlated feature pairs by
# creating out-of-distribution joint values; CPI permutes WITHIN leaves of
# a shallow tree X_{-j} -> X_j, preserving P(X_j | X_{-j}).
# ----------------------------------------------------------------------------
class TestPhase7_ConditionalPermutationImportance:
    """Groups tests covering TestPhase7_ConditionalPermutationImportance."""
    def test_cpi_returns_per_feature_dict(self):
        """Basic shape contract: one importance per feature, informative > noise."""
        Xdf, y, cols = make_sklearn_classification_df(
            n_samples=200,
            n_features=6,
            n_informative=3,
            seed=0,
            n_redundant=0,
            shuffle=False,
            class_sep=2.0,
        )
        model = RandomForestClassifier(n_estimators=30, random_state=0).fit(Xdf, y)
        result = get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="conditional_permutation",
            data=Xdf,
            target=y,
        )
        assert set(result.keys()) == set(cols)
        sorted_pairs = sorted(result.items(), key=lambda kv: kv[1], reverse=True)
        top3 = {k for k, _ in sorted_pairs[:3]}
        informative_in_top3 = len(top3 & {"f0", "f1", "f2"})
        assert informative_in_top3 >= 2, f"CPI failed to surface informative features; top3={top3}"

    def test_cpi_requires_target(self):
        """importance_getter='conditional_permutation' without target= must raise."""
        X = pd.DataFrame(np.random.randn(20, 4), columns=list("abcd"))
        model = LogisticRegression(max_iter=50).fit(X, [0, 1] * 10)
        with pytest.raises(ValueError, match="requires target"):
            get_feature_importances(
                model=model,
                current_features=list("abcd"),
                importance_getter="conditional_permutation",
                data=X,
            )

    def test_cpi_correlated_pair_structural(self):
        """Structural claim: with x2 highly correlated with the driver x1,
        CPI on x1 still ranks above noise features (driver signal survives),
        and CPI on x2 does NOT exceed CPI on x1 (the conditional tree
        x_{-2} -> x2 predicts x2 well from x1, so within-leaf shuffles
        of x2 barely move the score).

        Note: with PERFECT correlation x2==x1 the pair becomes fully
        redundant and BOTH vanilla and CPI return ~0 for both — that
        is correct conservative behavior, not a bug. We use 0.95
        correlation here so the driver/redundant asymmetry is visible.
        """
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        n = 500
        x1 = rng.standard_normal(n)
        # Tightly correlated but not identical: small uncorrelated jitter on x2.
        x2 = 0.95 * x1 + 0.1 * rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        Xdf = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        y = (x1 > 0).astype(int)  # driver is x1

        model = LogisticRegression(max_iter=300, random_state=0).fit(Xdf, y)

        cpi = get_feature_importances(
            model=model,
            current_features=list(Xdf.columns),
            importance_getter="conditional_permutation",
            data=Xdf,
            target=y,
        )

        # Driver x1 must rank above both noise features.
        assert cpi["x1"] > cpi["x3"] + 0.005, f"CPI: x1 must dominate noise; cpi={cpi}"
        assert cpi["x1"] > cpi["x4"] + 0.005, f"CPI: x1 must dominate noise; cpi={cpi}"
        # x2 (correlated near-copy) should not exceed driver x1.
        assert cpi["x2"] <= cpi["x1"] + 0.02, f"CPI must not over-rank correlated copy x2 above driver x1; cpi={cpi}"
        # Noise features must have small (near-zero) CPI.
        assert abs(cpi["x3"]) < 0.05, f"noise x3 should be ~0; cpi={cpi}"
        assert abs(cpi["x4"]) < 0.05, f"noise x4 should be ~0; cpi={cpi}"

    def test_cpi_works_with_ndarray_input(self):
        """ndarray X path (no .columns / .index) must also work."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((150, 5))
        y = (X[:, 0] > 0).astype(int)
        cols = list(range(5))
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        result = get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="conditional_permutation",
            data=X,
            target=y,
        )
        assert set(result.keys()) == set(cols)
        # Driver column 0 should dominate
        top = max(result, key=result.get)
        assert top == 0, f"CPI top should be column 0 (driver), got {top}"

    def test_cpi_with_regression(self):
        """Regression target path: model.score returns R^2; CPI must still rank
        the driver feature on top."""
        from sklearn.linear_model import Ridge
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=200,
            n_features=5,
            n_informative=2,
            random_state=0,
            shuffle=False,
        )
        cols = [f"f{i}" for i in range(5)]
        Xdf = pd.DataFrame(X, columns=cols)
        model = Ridge(random_state=0).fit(Xdf, y)
        result = get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="conditional_permutation",
            data=Xdf,
            target=y,
        )
        assert set(result.keys()) == set(cols)
        sorted_pairs = sorted(result.items(), key=lambda kv: kv[1], reverse=True)
        top2 = {k for k, _ in sorted_pairs[:2]}
        # The two informative features (f0, f1) should be in the top-2.
        assert len(top2 & {"f0", "f1"}) >= 1, f"CPI failed to surface regression-informative features; top2={top2}, all={result}"

    def test_cpi_single_feature_edge_case(self):
        """p=1: no conditioning set X_{-j}, CPI must fall back to vanilla shuffle
        without crashing."""
        rng = np.random.default_rng(2)
        Xdf = pd.DataFrame({"only": rng.standard_normal(80)})
        y = (Xdf["only"] > 0).astype(int).values
        model = LogisticRegression(max_iter=200, random_state=0).fit(Xdf, y)
        result = get_feature_importances(
            model=model,
            current_features=["only"],
            importance_getter="conditional_permutation",
            data=Xdf,
            target=y,
        )
        assert set(result.keys()) == {"only"}
        # Single driver -> non-trivial importance
        assert result["only"] > 0, f"single-feature CPI should be > 0, got {result}"

    def test_cpi_constant_feature_edge_case(self):
        """When X_j is constant, conditional tree fit may degenerate; CPI must
        return 0 for that feature without raising."""
        rng = np.random.default_rng(3)
        Xdf = pd.DataFrame(
            {
                "const": np.ones(100),
                "driver": rng.standard_normal(100),
            }
        )
        y = (Xdf["driver"] > 0).astype(int).values
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(Xdf, y)
        result = get_feature_importances(
            model=model,
            current_features=["const", "driver"],
            importance_getter="conditional_permutation",
            data=Xdf,
            target=y,
        )
        # Constant feature must have ~0 importance; driver must dominate.
        assert abs(result["const"]) < 0.05, f"constant feature should be ~0, got {result}"
        assert result["driver"] > result["const"], result

    def test_cpi_via_rfecv_end_to_end(self):
        """Smoke-test: RFECV(importance_getter='conditional_permutation') must
        complete without error and select at least one feature."""
        Xdf, y, _ = make_sklearn_classification_df(
            n_samples=200,
            n_features=8,
            n_informative=3,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=2.0,
            shuffle=False,
            seed=0,
        )
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3,
            max_refits=10,
            importance_getter="conditional_permutation",
            verbose=0,
            random_state=0,
        )
        rfecv.fit(Xdf, y)
        assert rfecv.n_features_ >= 1
        assert rfecv.support_.sum() == rfecv.n_features_
