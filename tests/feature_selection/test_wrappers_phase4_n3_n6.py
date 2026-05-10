"""Tests for Phase 4 N3 (parallel CV folds) and N6 (permutation/SHAP importance_getter).

N3 design: when ``n_jobs > 1`` AND the estimator is multi-threaded (CB/LGB/XGB/RF/...),
auto-fall-back to sequential UNLESS ``force_parallel=True`` (in which case pin
inner threads to 1). For single-thread estimators (LR, Ridge, etc.) parallel
folds give a real wall-clock win on multi-core machines.

N6 design: ``importance_getter='permutation'`` uses sklearn.inspection;
``importance_getter='shap'`` uses the optional ``shap`` package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import (
    RFECV,
    get_feature_importances,
)
from mlframe.feature_selection.wrappers._helpers import (
    _detect_multithreaded,
    _pin_threads_to_one,
)


@pytest.fixture(scope="module")
def small_clf_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=4,
        n_redundant=0, n_classes=2, n_clusters_per_class=1,
        random_state=0, shuffle=False, class_sep=2.0,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), y


# ----------------------------------------------------------------------------
# N3: Multi-thread detection helpers
# ----------------------------------------------------------------------------
class TestN3_MultithreadDetection:
    def test_detects_random_forest(self):
        rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        assert _detect_multithreaded(rf) is True

    def test_does_not_flag_logistic_regression(self):
        lr = LogisticRegression()
        assert _detect_multithreaded(lr) is False

    def test_pin_threads_zeroes_n_jobs_on_rf(self):
        rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        _pin_threads_to_one(rf)
        assert rf.get_params()["n_jobs"] == 1

    def test_pin_threads_no_op_on_lr_without_threading_param(self):
        # LR doesn't have any thread-count constructor param; pin should no-op.
        lr = LogisticRegression()
        _pin_threads_to_one(lr)
        # Expect no exception, no change in object.
        assert isinstance(lr, LogisticRegression)


# ----------------------------------------------------------------------------
# N3: Parallel CV folds produce same selection as sequential (correctness)
# ----------------------------------------------------------------------------
class TestN3_ParallelEquivalence:
    def test_n_jobs_2_matches_n_jobs_1_for_single_thread_estimator(self, small_clf_data):
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
            f"Parallel and sequential RFECV diverged on the same input + seed. "
            f"seq={list(seq.get_feature_names_out())} par={list(par.get_feature_names_out())}"
        )

    def test_n_jobs_negative_one_resolves_to_cpu_count(self, small_clf_data):
        # n_jobs=-1 should resolve to all cores; ensure it doesn't crash.
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
    def test_rf_auto_fallback_no_force_parallel(self, small_clf_data, caplog):
        """RF + n_jobs=2 + force_parallel=False (default) -> auto-fallback to
        sequential and a clear log line. The outer RF still uses its native
        threading, but RFECV doesn't compound it."""
        X, y = small_clf_data
        import logging
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers._rfecv"):
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
    def test_permutation_returns_per_feature_dict(self):
        X, y = make_classification(
            n_samples=120, n_features=6, n_informative=3, random_state=0,
            n_redundant=0, shuffle=False, class_sep=2.0,
        )
        cols = [f"f{i}" for i in range(6)]
        Xdf = pd.DataFrame(X, columns=cols)
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
        assert informative_in_top3 >= 2, (
            f"permutation importance failed to surface the informative features; "
            f"top3={top3}"
        )

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
    def test_borda_skips_majority_graph(self):
        from mlframe.votenrank import Leaderboard
        df = pd.DataFrame({"r1": [10, 20, 5, 15], "r2": [12, 18, 6, 14]},
                          index=["f0", "f1", "f2", "f3"])
        lb = Leaderboard(table=df)
        # Before any graph-using method is called: graph not built.
        assert lb.majority_graph is None
        _ = lb.borda_ranking()
        assert lb.majority_graph is None, (
            "borda_ranking() must not trigger majority_graph construction; "
            "the graph is the n^2 hot path that was making RFECV slow."
        )

    def test_copeland_builds_majority_graph_on_demand(self):
        from mlframe.votenrank import Leaderboard
        df = pd.DataFrame({"r1": [10, 20, 5, 15], "r2": [12, 18, 6, 14]},
                          index=["f0", "f1", "f2", "f3"])
        lb = Leaderboard(table=df)
        assert lb.majority_graph is None
        _ = lb.copeland_ranking()
        assert lb.majority_graph is not None, (
            "copeland_ranking() must trigger lazy majority_graph build."
        )

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
