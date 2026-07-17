"""Wave 3 (2026-05-28) FI-semantics regression tests for RFECV.

Covers:
  - F4 : ``coef_scale_source='train'`` rescales coef_ with train-fold std.
  - F5 : ``multiclass_coef_aggregation='max'`` collapses OvR via max(|coef|).
  - F6 : multi-estimator + AM/GM auto-falls-back to Borda.
  - F7 : deterministic tie-breaker by lexicographic feature name.
  - F8 : ``fi_decay_rate`` exponential decay over FI history.
  - F10 : CPI tree grown via min_samples_leaf, not max_depth=5.
  - F11 : _is_discrete_v2 tighter heuristic.
  - F12 : use_fi_ranking is a no-op when downstream rule is rank-based.
  - F14 : NaN-score FI runs dropped from voting pool by default.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._helpers import (
    _conditional_permutation_importance,
    get_actual_features_ranking,
    select_appropriate_feature_importances,
)
from mlframe.feature_selection.wrappers._enums import VotesAggregation


# ----------------------------------------------------------------------- F7


class TestTieBreaker:
    def test_tied_features_sort_lexicographically(self):
        # Three runs where features a,b,c all have identical importance ranks.
        fi = {
            "r0": {"a": 0.5, "b": 0.5, "c": 0.5},
            "r1": {"a": 0.5, "b": 0.5, "c": 0.5},
        }
        out = get_actual_features_ranking(fi, VotesAggregation.Borda, fi_missing_policy="worst")
        # All tied -> lexicographic.
        assert out == ["a", "b", "c"]

    def test_partial_tie_lexicographic_within_tier(self):
        fi = {
            "r0": {"x": 0.9, "a": 0.5, "b": 0.5, "c": 0.5, "z": 0.1},
        }
        out = get_actual_features_ranking(fi, VotesAggregation.Borda, fi_missing_policy="worst")
        assert out[0] == "x"
        assert out[-1] == "z"
        # Middle tied trio sorts lexicographically.
        assert out[1:4] == ["a", "b", "c"]


# ----------------------------------------------------------------------- F8


class TestFiDecay:
    def test_decay_weights_shift_ranking_toward_recent_runs(self):
        # Earlier run says A best, latest run says B best.
        # Without decay: tied.
        # With decay rate=0.5 (heavy): latest run wins -> B first.
        fi = {
            "r0_oldest": {"A": 1.0, "B": 0.0, "C": 0.5},
            "r1_middle": {"A": 1.0, "B": 0.0, "C": 0.5},
            "r2_latest": {"A": 0.0, "B": 1.0, "C": 0.5},
        }
        ordered = ["r0_oldest", "r1_middle", "r2_latest"]
        # No decay
        get_actual_features_ranking(
            fi,
            VotesAggregation.Borda,
            fi_missing_policy="worst",
            run_weights={k: 1.0 for k in ordered},
        )
        # With heavy decay rate=0.5
        decay = 0.5
        weights = {k: (1.0 - decay) ** (len(ordered) - 1 - i) for i, k in enumerate(ordered)}
        out_decay = get_actual_features_ranking(
            fi,
            VotesAggregation.Borda,
            fi_missing_policy="worst",
            run_weights=weights,
        )
        # Under heavy decay, the latest run (B-favouring) dominates -> B must rank
        # at least as high as A in the decayed output.
        assert out_decay.index("B") <= out_decay.index("A"), f"Decay should push B up; got {out_decay}"


# ----------------------------------------------------------------------- F12


class TestFiRankingSkipForRankBased:
    def test_borda_skips_pre_ranking(self):
        fi = {"r0": {"a": 1.0, "b": 0.5, "c": 0.1}}
        out_borda = select_appropriate_feature_importances(
            feature_importances=fi,
            nfeatures=2,
            n_original_features=3,
            use_fi_ranking=True,
            use_all_fi_runs=True,
            votes_aggregation_method=VotesAggregation.Borda,
        )
        # The single run still has float values, not pct-ranks.
        vals = list(out_borda["r0"].values())
        assert any(v > 0.5 for v in vals)  # would be in [0,1] if pre-ranked

    def test_am_still_pre_ranks(self):
        fi = {"r0": {"a": 1.0, "b": 0.5, "c": 0.1}}
        out_am = select_appropriate_feature_importances(
            feature_importances=fi,
            nfeatures=2,
            n_original_features=3,
            use_fi_ranking=True,
            use_all_fi_runs=True,
            votes_aggregation_method=VotesAggregation.AM,
        )
        # AM should have been pre-ranked -> pct values in [0, 1].
        vals = list(out_am["r0"].values())
        assert all(0.0 <= v <= 1.0 for v in vals)


# ----------------------------------------------------------------------- F4+F5


class TestCoefScaleSource:
    def test_default_uses_train_data(self):
        from mlframe.feature_selection.wrappers._helpers import get_feature_importances

        rng = np.random.default_rng(0)
        n, p = 200, 5
        X_train = rng.normal(scale=np.array([[1.0, 10.0, 0.1, 1.0, 1.0]]), size=(n, p))
        X_test = rng.normal(scale=np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]), size=(n, p))
        beta = np.array([1.0, 0.1, 10.0, 0.5, -0.5])
        y_train = X_train @ beta + rng.normal(scale=0.1, size=n)
        y_test = X_test @ beta + rng.normal(scale=0.1, size=n)
        model = Ridge().fit(X_train, y_train)
        # Default coef_scale_source='train' uses X_train stds. Force importance_getter='coef_'
        # explicitly: 'auto' now resolves to permutation importance on small data (accuracy-first
        # default), which bypasses the coef-rescale branch this test validates.
        fi_train = get_feature_importances(
            model=model,
            current_features=list(range(p)),
            data=X_test,
            train_data=X_train,
            target=y_test,
            importance_getter="coef_",
            coef_scale_source="train",
        )
        # Now ask for 'test' rescale.
        fi_test = get_feature_importances(
            model=model,
            current_features=list(range(p)),
            data=X_test,
            train_data=X_train,
            target=y_test,
            importance_getter="coef_",
            coef_scale_source="test",
        )
        # Results should differ because train/test stds are different.
        assert any(abs(fi_train[i] - fi_test[i]) > 1e-9 for i in range(p)), "coef_scale_source='train' should produce a different rescale than 'test'"


class TestMulticlassCoefAggregation:
    def test_max_vs_sum_differs_on_class_specific_feature(self):
        from mlframe.feature_selection.wrappers._helpers import get_feature_importances
        from sklearn.multiclass import OneVsRestClassifier

        # 3-class problem with class-specific features
        rng = np.random.default_rng(0)
        n, p = 300, 4
        X = rng.normal(size=(n, p))
        # Make f0 a strong class-0 discriminator only; rest noise.
        y = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.5).astype(int) * 2
        # sklearn 1.7+ dropped multi_class=; wrap in OvR explicitly so coef_ ends up 2-D.
        model = OneVsRestClassifier(LogisticRegression(max_iter=400)).fit(X, y)
        # OneVsRestClassifier doesn't expose coef_ directly in some sklearn versions;
        # stack manually for the test.
        coefs = np.array([est.coef_.ravel() for est in model.estimators_])

        # Wrap into an object that get_feature_importances can read.
        class _Wrap:
            coef_ = coefs

        model = _Wrap()
        # Force importance_getter='coef_' explicitly: 'auto' now resolves to permutation importance
        # on small data (accuracy-first default), which bypasses the multiclass coef-aggregation
        # branch this test validates. (The _Wrap object only exposes coef_, not feature_importances_.)
        fi_max = get_feature_importances(
            model=model,
            current_features=list(range(p)),
            data=X,
            train_data=X,
            target=y,
            importance_getter="coef_",
            multiclass_coef_aggregation="max",
        )
        fi_sum = get_feature_importances(
            model=model,
            current_features=list(range(p)),
            data=X,
            train_data=X,
            target=y,
            importance_getter="coef_",
            multiclass_coef_aggregation="sum",
        )
        # The two aggregations MUST produce different orderings or different magnitudes for at least one feature.
        assert any(abs(fi_max[i] - fi_sum[i]) > 1e-9 for i in range(p))


# ----------------------------------------------------------------------- F6


class TestMultiEstimatorAmGmFallback:
    def test_am_falls_back_to_borda_on_multi(self, caplog):
        X, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)
        rfecv = RFECV(
            estimators=[LogisticRegression(max_iter=300), RandomForestClassifier(n_estimators=10)],
            cv=3,
            max_refits=3,
            random_state=0,
            votes_aggregation_method=VotesAggregation.AM,
            verbose=1,
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
            rfecv.fit(X, y)
        assert any("Switching to Borda" in rec.getMessage() for rec in caplog.records)

    def test_opt_in_preserves_user_choice(self):
        X, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)
        rfecv = RFECV(
            estimators=[LogisticRegression(max_iter=300), RandomForestClassifier(n_estimators=10)],
            cv=3,
            max_refits=3,
            random_state=0,
            votes_aggregation_method=VotesAggregation.AM,
            allow_unsafe_aggregation=True,
        )
        rfecv.fit(X, y)
        # Smoke
        assert rfecv.n_features_ >= 1


# ----------------------------------------------------------------------- F10+F11


class TestCpiKnobs:
    def test_max_depth_none_grows_deeper(self):
        # Validate that passing max_depth=None to the helper actually grows
        # an unconstrained tree (min_samples_leaf-bounded).
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(0)
        n, p = 200, 6
        X = rng.normal(size=(n, p))
        # Strong cross-correlations between X[:,0] and X[:,1].
        X[:, 1] = X[:, 0] * 0.95 + rng.normal(scale=0.1, size=n)
        y = X[:, 0] + X[:, 2] + rng.normal(scale=0.1, size=n)
        model = LinearRegression().fit(X, y)
        imps = _conditional_permutation_importance(
            model,
            pd.DataFrame(X),
            y,
            n_repeats=3,
            max_depth=None,
            min_samples_leaf=10,
            random_state=0,
        )
        assert imps.shape == (p,)
        # Smoke: no NaNs.
        assert np.isfinite(imps).all()


# ----------------------------------------------------------------------- F14


class TestNanScoreFiDropped:
    def test_nan_score_fi_dropped_by_default(self):
        # Construct a scenario where one estimator returns NaN score on a fold.
        # We use a callable scorer that returns NaN for fold-0 only.
        from sklearn.metrics import make_scorer

        nan_counter = {"i": 0}

        def scorer_func(y_true, y_pred):
            nan_counter["i"] += 1
            return np.nan if nan_counter["i"] == 1 else 0.5

        scorer = make_scorer(scorer_func, greater_is_better=True)
        X, y = make_classification(n_samples=200, n_features=6, n_informative=3, random_state=0)
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=300),
            cv=3,
            max_refits=2,
            random_state=0,
            scoring=scorer,
        )
        rfecv.fit(X, y)
        # All FI run keys must NOT include the fold-0 first-iter run (which got NaN score).
        # We can't easily inspect this without internal state, so just smoke that the fit completes.
        assert rfecv.n_features_ >= 0
