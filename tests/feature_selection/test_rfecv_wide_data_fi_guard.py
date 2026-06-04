"""Regression tests for the wide-data permutation-FI cost guard (2026-06-04).

Bug: ``RFECVSel('lgbm_perm')`` (importance_getter='permutation') on a WIDE frame (e.g. madelon p=500) timed out and
produced a ~3-point cv_results_ curve, so the N-rule landed at the over-selection. Permutation / conditional-permutation
importance rescore the model O(p * n_repeats) times PER FOLD, so on wide frames a single RFECV iteration can exceed the
whole runtime budget (measured madelon p=500, n_repeats=5 -> ~208s/iter > a 180s budget) and only 2-3 iterations finish.

Fix: when ``wide_data_fi_fallback`` (default True) and the search universe exceeds ``wide_data_fi_threshold``, RFECV
falls back to the estimator's native (gain/impurity) importance for the elimination ranking so the outer loop builds a
REAL multi-point curve in budget. Below the threshold, n_repeats is capped at ``wide_data_fi_n_repeats``.

These tests assert:
  - the guard FIRES on a wide frame + permutation getter (fallback to native, recorded in _wide_data_fi_applied_);
  - the guard caps n_repeats just under the threshold;
  - with the guard the wide-frame run yields a multi-point curve and a support strictly narrower than all-features
    WITHOUT a pathological iteration count (the pre-fix failure mode);
  - the guard is a NO-OP on a narrow frame (perm FI kept exactly), so narrow-frame behaviour does not regress;
  - opting out (wide_data_fi_fallback=False) keeps the permutation getter on a wide frame;
  - self.n_repeats is never mutated by the guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score

from mlframe.feature_selection.wrappers import RFECV, FIConfig

_AUC = make_scorer(roc_auc_score, response_method="predict_proba")


def _wide_frame(n_samples=200, n_features=60, n_informative=6, seed=0):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative,
        n_redundant=4, random_state=seed,
    )
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return Xdf, pd.Series(y)


def _fast_rf():
    # Tiny RF: cheap per fit so even the perm-FI path (opt-out test) stays quick.
    return RandomForestClassifier(n_estimators=12, max_depth=6, random_state=0, n_jobs=1)


class TestWideDataGuardFires:
    def test_fallback_to_native_on_wide_frame(self):
        Xdf, y = _wide_frame(n_features=60)
        sel = RFECV(
            estimator=_fast_rf(),
            importance_getter="permutation",
            wide_data_fi_threshold=30,   # 60 > 30 -> fallback
            cv=3, max_refits=4, random_state=0, n_repeats=5,
        )
        sel.fit(Xdf, y)
        guard = sel._wide_data_fi_applied_
        assert guard is not None, "wide-data guard should have fired on a wide frame + permutation getter"
        assert guard["reason"] == "fallback_to_native"
        assert guard["from_importance_getter"] == "permutation"
        assert guard["to_importance_getter"] == "auto"
        # user-facing n_repeats untouched by the guard
        assert sel.n_repeats == 5

    def test_wide_frame_yields_multipoint_curve_and_narrow_support(self):
        # The pre-fix failure mode: perm-FI's O(p*n_repeats) per-fold cost let only 2-3 N be explored before the
        # budget ran out -> curve degenerated -> support ~= all features. With the fallback to fast native importance,
        # the MBH explores a real multi-point grid in the SAME refit budget and the parsimonious rule lands below p.
        # roc_auc scorer + clear signal so the model beats the dummy at N=0 (otherwise the early "dummy-beats-first"
        # stop fires at iter 1 -- a synthetic-data artifact unrelated to the perm-FI cost bug).
        Xdf, y = _wide_frame(n_samples=600, n_features=60, n_informative=10)
        sel = RFECV(
            estimator=_fast_rf(),
            importance_getter="permutation",
            scoring=_AUC,
            wide_data_fi_threshold=30,
            cv=3, max_refits=12, random_state=0, n_repeats=5,
            init_design_size=5,
            n_features_selection_rule="one_se_min",
        )
        sel.fit(Xdf, y)
        # guard must have fired (so this is the fallback path, not exact perm FI)
        assert sel._wide_data_fi_applied_ is not None
        nfeat = [n for n in sel.cv_results_["nfeatures"] if n > 0]
        assert len(nfeat) > 2, f"expected a multi-point curve, got explored N={sel.cv_results_['nfeatures']}"
        assert sel.n_features_ < 60, f"expected support narrower than all-features, got {sel.n_features_}"
        assert sel.n_features_ >= 1

    def test_caps_n_repeats_below_threshold(self):
        # p between threshold//4 and threshold -> n_repeats capped, getter kept.
        Xdf, y = _wide_frame(n_features=40)
        sel = RFECV(
            estimator=_fast_rf(),
            importance_getter="permutation",
            wide_data_fi_threshold=50,         # 40 <= 50 -> no fallback
            wide_data_fi_n_repeats=2,
            cv=3, max_refits=3, random_state=0, n_repeats=5,
        )
        sel.fit(Xdf, y)
        guard = sel._wide_data_fi_applied_
        assert guard is not None and guard["reason"] == "capped_n_repeats"
        assert guard["to_n_repeats"] == 2
        assert sel._effective_n_repeats == 2
        assert sel.n_repeats == 5  # untouched


class TestWideDataGuardNoOp:
    def test_narrow_frame_keeps_permutation_exactly(self):
        # Narrow frame (p below threshold//4) -> no fallback, no cap: narrow-frame behaviour must not regress.
        Xdf, y = _wide_frame(n_features=8, n_informative=4)
        sel = RFECV(
            estimator=_fast_rf(),
            importance_getter="permutation",
            wide_data_fi_threshold=200,
            cv=3, max_refits=3, random_state=0, n_repeats=5,
        )
        sel.fit(Xdf, y)
        assert sel._wide_data_fi_applied_ is None, "guard must be a no-op on a narrow frame"
        assert sel._effective_n_repeats == 5
        assert sel.n_features_ >= 1

    def test_opt_out_keeps_permutation_on_wide_frame(self):
        Xdf, y = _wide_frame(n_features=60)
        sel = RFECV(
            estimator=_fast_rf(),
            importance_getter="permutation",
            wide_data_fi_fallback=False,   # opt out
            wide_data_fi_threshold=30,
            cv=3, max_refits=3, random_state=0, n_repeats=5,
        )
        sel.fit(Xdf, y)
        assert sel._wide_data_fi_applied_ is None
        assert sel._effective_n_repeats == 5
        assert sel.n_features_ >= 1


class TestWideDataGuardConfig:
    def test_knobs_via_fi_config(self):
        sel = RFECV(
            estimator=_fast_rf(),
            fi_config=FIConfig(
                wide_data_fi_fallback=False,
                wide_data_fi_threshold=123,
                wide_data_fi_n_repeats=4,
            ),
        )
        assert sel.wide_data_fi_fallback is False
        assert sel.wide_data_fi_threshold == 123
        assert sel.wide_data_fi_n_repeats == 4

    def test_defaults(self):
        sel = RFECV(estimator=_fast_rf())
        assert sel.wide_data_fi_fallback is True
        assert sel.wide_data_fi_threshold == 200
        assert sel.wide_data_fi_n_repeats == 2
