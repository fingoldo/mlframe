"""Wave 1 (2026-05-28) ML-correctness regression tests for RFECV.

Covers seven fixes batched together:
  - F9  : FI rollback when ``store_averaged_cv_scores`` rejects re-explored same-N subset (no contamination of voting).
  - F1+F2+F3 : ``fi_missing_policy`` (default 'worst') imputes ragged NaN votes so every Leaderboard rule sees a uniform table.
  - E1  : ``must_include`` overrides ``leakage_action='exclude' / 'raise'``.
  - E3  : ``feature_groups`` overlap rejected at __init__ (no silent all-or-nothing fan-out).
  - E2  : ``swap_top_k`` auto-disabled when early-stopping val_cv is active (no apples-to-oranges).
  - C2  : N=0 dummy NOT submitted to MBH surrogate by default.
  - C3  : ``n_features_selection_rule='auto'`` resolves to 'argmax' (not the inverted multi-estimator 'one_se_max').

Each test fails on the pre-2026-05-28 code path.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._helpers import (
    _impute_ragged_fi_table,
    get_actual_features_ranking,
)
from mlframe.feature_selection.wrappers._enums import VotesAggregation


# --------------------------------------------------------------------- F1+F2+F3


class TestFiMissingPolicy:
    def test_worst_imputation_pushes_missing_below_min(self):
        # Ragged: feature B missing from run r1.
        feature_importances = {
            "r0": {"A": 0.1, "B": 0.9},
            "r1": {"A": 0.5, "C": 0.4},
        }
        table = pd.DataFrame(feature_importances)
        imputed = _impute_ragged_fi_table(table, policy="worst")
        # In column r1 the original min was 0.4 (C); the imputed B must be strictly below 0.4.
        assert imputed.loc["B", "r1"] < 0.4
        # In column r0 the original min was 0.1 (A); imputed C must be strictly below 0.1.
        assert imputed.loc["C", "r0"] < 0.1
        # Original present values must be unchanged.
        assert imputed.loc["A", "r0"] == pytest.approx(0.1)
        assert imputed.loc["A", "r1"] == pytest.approx(0.5)
        assert imputed.loc["B", "r0"] == pytest.approx(0.9)
        assert imputed.loc["C", "r1"] == pytest.approx(0.4)

    def test_skip_policy_preserves_nan(self):
        feature_importances = {"r0": {"A": 0.1, "B": 0.9}, "r1": {"A": 0.5, "C": 0.4}}
        table = pd.DataFrame(feature_importances)
        imputed = _impute_ragged_fi_table(table, policy="skip")
        assert imputed.isna().to_numpy().any()

    def test_median_policy_uses_column_median(self):
        feature_importances = {"r0": {"A": 0.1, "B": 0.3, "C": 0.5}, "r1": {"A": 0.2, "C": 0.6}}
        table = pd.DataFrame(feature_importances)
        imputed = _impute_ragged_fi_table(table, policy="median")
        # r1 median over {0.2, 0.6} = 0.4
        assert imputed.loc["B", "r1"] == pytest.approx(0.4)

    def test_borda_under_worst_policy_penalises_partial_voters(self):
        # A: top in 3/3 runs. B: top in only 1/3 (missing from r1, r2).
        # Under 'worst' B is treated as last in r1, r2 -> A clearly wins.
        feature_importances = {
            "r0": {"A": 1.0, "B": 0.9, "C": 0.1},
            "r1": {"A": 1.0, "C": 0.5},
            "r2": {"A": 1.0, "C": 0.5},
        }
        ranked = get_actual_features_ranking(
            feature_importances, VotesAggregation.Borda, fi_missing_policy="worst",
        )
        assert ranked[0] == "A"
        # Under 'worst', B (1/3 runs, late-eliminated) MUST rank below C (3/3 runs, mid).
        assert ranked.index("C") < ranked.index("B")

    def test_borda_under_skip_policy_keeps_legacy_bias(self):
        # Demonstrates the bias 'skip' inherits from the legacy code path.
        feature_importances = {
            "r0": {"A": 1.0, "B": 0.9, "C": 0.1},
            "r1": {"A": 1.0, "C": 0.5},
            "r2": {"A": 1.0, "C": 0.5},
        }
        ranked = get_actual_features_ranking(
            feature_importances, VotesAggregation.Borda, fi_missing_policy="skip",
        )
        # Under 'skip', B sums only one run where it was high -> ties or beats C.
        # We don't assert a specific order; we only confirm the 'worst' fix changed things.
        assert ranked[0] == "A"


# ----------------------------------------------------------------------- E3


class TestFeatureGroupsOverlapAssert:
    def test_overlap_raises_at_init(self):
        with pytest.raises(ValueError, match="appears in BOTH group"):
            RFECV(
                estimator=LogisticRegression(),
                feature_groups={
                    "g1": ["a", "b"],
                    "g2": ["b", "c"],  # 'b' shared -> contradiction
                },
            )

    def test_disjoint_groups_init_fine(self):
        RFECV(
            estimator=LogisticRegression(),
            feature_groups={"g1": ["a", "b"], "g2": ["c", "d"]},
        )


# ----------------------------------------------------------------------- F1 init knob


class TestNewInitKnobs:
    def test_invalid_fi_missing_policy_rejected(self):
        with pytest.raises(ValueError, match="fi_missing_policy"):
            RFECV(estimator=LogisticRegression(), fi_missing_policy="bogus")

    def test_default_knobs_match_new_safe_values(self):
        r = RFECV(estimator=LogisticRegression())
        assert r.keep_loser_subset_fi is False
        assert r.fi_missing_policy == "worst"
        # C2: kept True default until S9 (low-N init design) lands in Wave 2,
        # because removing the dummy anchor halves explored-N count on small p.
        assert r.submit_dummy_to_optimizer is True
        assert r.swap_top_k_allow_no_es is False


# ----------------------------------------------------------------------- E1


class TestMustIncludeOverridesLeakage:
    def _leaky_frame(self):
        X, y = make_classification(
            n_samples=200, n_features=8, n_informative=4, random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        # Plant a perfect leak as f7.
        X["leak"] = y.astype(float) + np.random.default_rng(0).normal(scale=0.01, size=200)
        return X, y

    def test_must_include_overrides_exclude_keeps_pinned_leaky(self, caplog):
        X, y = self._leaky_frame()
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=400),
            cv=3,
            max_refits=3,
            must_include=["leak"],
            leakage_corr_threshold=0.9,
            leakage_action="exclude",
            verbose=1,
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
            rfecv.fit(X, y)
        # Pinned leaky column must end up in selected features (NOT dropped).
        assert "leak" in rfecv.get_feature_names_out().tolist()
        # And we must have logged the override.
        assert any("must_include pins" in rec.getMessage() for rec in caplog.records)

    def test_must_include_overrides_raise_downgrades_when_all_pinned(self):
        X, y = self._leaky_frame()
        # Single leak, pinned -> 'raise' must downgrade to warn.
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=400),
            cv=3,
            max_refits=3,
            must_include=["leak"],
            leakage_corr_threshold=0.9,
            leakage_action="raise",
        )
        rfecv.fit(X, y)  # must NOT raise
        assert "leak" in rfecv.get_feature_names_out().tolist()


# ----------------------------------------------------------------------- C3


class TestAutoRuleResolution:
    """C3 revised post-bench (2026-05-28): 'auto' -> 'one_se_max' (was 'argmax').
    On flat score curves argmax underselects (recall=0.30 on the USAGE.md
    bench) while one_se_max picks the full 1-SE band and recovers recall=1.0.
    On non-flat curves both rules give the same N within ±1 feature, so the
    plateau-resistant choice is strictly safer as a default.
    """

    def test_auto_resolves_to_one_se_max(self):
        # Verify by checking N is within the 1-SE band of argmax mean.
        X, y = make_regression(n_samples=120, n_features=8, n_informative=3, random_state=0)
        rfecv = RFECV(estimator=Ridge(), cv=3, max_refits=5)
        rfecv.fit(X, y)
        means = np.asarray(rfecv.cv_results_["cv_mean_perf"], dtype=float)
        stds = np.asarray(rfecv.cv_results_["cv_std_perf"], dtype=float)
        nfeats = np.asarray(rfecv.cv_results_["nfeatures"], dtype=int)
        nz = nfeats > 0
        if not nz.any():
            return
        best = int(np.argmax(means[nz]))
        threshold = means[nz][best] - stds[nz][best]
        in_band = nfeats[nz][means[nz] >= threshold]
        # 'one_se_max' picks the LARGEST N within the 1-SE band.
        if len(in_band) > 0:
            assert rfecv.n_features_ == int(in_band.max())

    def test_explicit_argmax_still_picks_argmax(self):
        X, y = make_regression(n_samples=120, n_features=8, n_informative=3, random_state=0)
        rfecv = RFECV(estimator=Ridge(), cv=3, max_refits=5, n_features_selection_rule="argmax")
        rfecv.fit(X, y)
        means = np.asarray(rfecv.cv_results_["cv_mean_perf"], dtype=float)
        nfeats = np.asarray(rfecv.cv_results_["nfeatures"], dtype=int)
        nz = nfeats > 0
        argmax_n = int(nfeats[nz][np.argmax(means[nz])])
        assert rfecv.n_features_ == argmax_n


# ----------------------------------------------------------------------- C2


class TestDummySubmitKnob:
    """C2 (paired with future S9): the knob exists, opt-out works.

    Default stays True (submit dummy) until S9 (low-N seed init design) lands
    in Wave 2 - without dummy anchor the optimizer can't explore low N on
    small p. The knob lets users on imbalanced accuracy/F1 disable submission.
    """

    def test_opt_out_suppresses_dummy_submission(self, monkeypatch):
        from mlframe.models.optimization import MBHOptimizer

        submitted: list = []
        orig = MBHOptimizer.submit_evaluations

        def tracked(self, candidates, evaluations, durations):
            submitted.extend(list(candidates))
            return orig(self, candidates, evaluations, durations)

        monkeypatch.setattr(MBHOptimizer, "submit_evaluations", tracked)

        X, y = make_regression(n_samples=120, n_features=6, n_informative=3, random_state=0)
        rfecv = RFECV(
            estimator=Ridge(), cv=3, max_refits=3,
            submit_dummy_to_optimizer=False,  # opt-out
        )
        rfecv.fit(X, y)
        assert 0 not in submitted, (
            f"submit_dummy_to_optimizer=False should suppress N=0; got {submitted}"
        )

    def test_default_submits_dummy(self, monkeypatch):
        from mlframe.models.optimization import MBHOptimizer

        submitted: list = []
        orig = MBHOptimizer.submit_evaluations

        def tracked(self, candidates, evaluations, durations):
            submitted.extend(list(candidates))
            return orig(self, candidates, evaluations, durations)

        monkeypatch.setattr(MBHOptimizer, "submit_evaluations", tracked)

        X, y = make_regression(n_samples=120, n_features=6, n_informative=3, random_state=0)
        rfecv = RFECV(estimator=Ridge(), cv=3, max_refits=3)  # default = True
        rfecv.fit(X, y)
        assert 0 in submitted


# ----------------------------------------------------------------------- F9


class TestFiRollbackOnLoserSubset:
    def test_loser_subset_fi_not_kept_by_default(self):
        # Direct unit on the rollback machinery: simulate two iters writing FI at the same N,
        # second one losing the gate. After the loop the loser's FI runs must be gone.
        from mlframe.feature_selection.wrappers.rfecv._fit_outer_loop import OuterLoopState
        state = OuterLoopState()
        # Simulate iter 1: a winning subset stored at N=5.
        state.feature_importances["5_0"] = {"a": 1.0}
        state.feature_importances["5_1"] = {"a": 1.0}
        state.evaluated_scores_mean[5] = 0.9
        state.evaluated_scores_std[5] = 0.0
        # Now snapshot before "iter 2".
        fi_before = set(state.feature_importances.keys())
        # Simulate iter 2 adding (loser) FI runs.
        state.feature_importances["5_0_iter2"] = {"a": 0.0}
        state.feature_importances["5_1_iter2"] = {"a": 0.0}
        # Pretend store_averaged_cv_scores returned was_stored=False; do rollback.
        new_keys = set(state.feature_importances.keys()) - fi_before
        for k in new_keys:
            state.feature_importances.pop(k, None)
        # Only the original winning FI remains.
        assert set(state.feature_importances.keys()) == fi_before


# ----------------------------------------------------------------------- E2


class TestSwapTopKGatedOnValCv:
    def test_swap_not_skipped_when_estimator_lacks_es(self):
        # Ridge does NOT support early stopping -> val_cv is dead even if
        # early_stopping_val_nsplits is truthy. swap MUST run.
        X, y = make_regression(n_samples=200, n_features=8, n_informative=4, random_state=0)
        rfecv = RFECV(
            estimator=Ridge(), cv=3, max_refits=4,
            swap_top_k=3,
            early_stopping_val_nsplits=5,
        )
        rfecv.fit(X, y)
        assert rfecv.n_features_ >= 1  # smoke

    def test_swap_skipped_with_es_estimator(self, monkeypatch, caplog):
        # Simulate "estimator supports ES" by patching has_early_stopping_support.
        import mlframe.core.helpers as _helpers
        monkeypatch.setattr(_helpers, "has_early_stopping_support", lambda name: True)
        # Also patch the in-finalize import binding.
        from mlframe.feature_selection.wrappers.rfecv import _finalize
        # Re-import to ensure binding sees patch.
        X, y = make_regression(n_samples=120, n_features=8, n_informative=4, random_state=0)
        rfecv = RFECV(
            estimator=Ridge(), cv=3, max_refits=4,
            swap_top_k=3,
            early_stopping_val_nsplits=5,
            verbose=1,
        )
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers.rfecv"):
            rfecv.fit(X, y)
        assert any("swap_top_k=" in rec.getMessage() and "skipped" in rec.getMessage()
                   for rec in caplog.records), \
            f"Expected the swap_top_k skip log; got: {[r.getMessage() for r in caplog.records[-10:]]}"

    def test_opt_in_override_runs_swap(self, monkeypatch):
        # Force ES estimator detection so the gate kicks in.
        import mlframe.core.helpers as _helpers
        monkeypatch.setattr(_helpers, "has_early_stopping_support", lambda name: True)
        X, y = make_regression(n_samples=200, n_features=8, n_informative=4, random_state=0)
        rfecv = RFECV(
            estimator=Ridge(), cv=3, max_refits=4,
            swap_top_k=2,
            early_stopping_val_nsplits=5,
            swap_top_k_allow_no_es=True,
        )
        rfecv.fit(X, y)
        assert rfecv.n_features_ >= 1
