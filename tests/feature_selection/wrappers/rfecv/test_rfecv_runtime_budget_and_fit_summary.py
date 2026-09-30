"""RFECV max_runtime_mins enforcement, end-of-fit best tracking, and the end-of-fit selection summary log.

The budget tests run on a fake clock that advances only when the wrapped estimator fits, so iteration counts are deterministic regardless of machine load.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers.rfecv import _fit as fit_mod
from mlframe.feature_selection.wrappers.rfecv import _fit_outer_loop as loop_mod
from mlframe.feature_selection.wrappers.rfecv._fit_summary import build_rfecv_fit_summary

RFECV_LOGGER = "mlframe.feature_selection.wrappers.rfecv"
SECONDS_PER_FIT = 60.0


class _Clock:
    """Manually advanced clock callable."""
    now = 0.0

    def __call__(self) -> float:
        return self.now


_CLOCK = _Clock()


class _ClockedLR(LogisticRegression):
    """LogisticRegression whose every fit costs SECONDS_PER_FIT on the fake clock."""

    def fit(self, X, y, sample_weight=None):
        """Advance the fake clock by SECONDS_PER_FIT, then fit."""
        _CLOCK.now += SECONDS_PER_FIT
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace the fit and loop timers with the manual clock, reset to zero."""
    _CLOCK.now = 0.0
    monkeypatch.setattr(fit_mod, "timer", _CLOCK)
    monkeypatch.setattr(loop_mod, "timer", _CLOCK)
    return _CLOCK


def _data(n=400, p=12, seed=0):
    """Build a frame with two informative columns and a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = (X["x0"] + 0.5 * X["x1"] + rng.normal(scale=0.5, size=n) > 0).astype(int).to_numpy()
    return X, y


def _rfecv(**kw):
    """Build a fast RFECV around the clocked estimator, with kwargs overrides."""
    params = dict(
        estimator=_ClockedLR(max_iter=200), cv=2, random_state=0, verbose=0, leave_progressbars=False, optimizer_plotting="No",
        max_noimproving_iters=1000, early_stopping_val_nsplits=None, skip_retraining_on_same_shape=False, leakage_corr_threshold=None,
    )
    params.update(kw)
    return RFECV(**params)


def test_max_runtime_does_not_start_iteration_predicted_to_overrun(fake_clock, caplog):
    """Max runtime does not start iteration predicted to overrun."""
    # cv=2 -> one iteration = 2 estimator fits = 2 fake minutes. Budget 5 min: after 2 iterations (4 min) a 3rd would end at 6 min.
    X, y = _data()
    sel = _rfecv(max_runtime_mins=5)
    with caplog.at_level(logging.INFO, logger=RFECV_LOGGER):
        sel.fit(X, y)
    n_iters = len([n for n in sel.cv_results_["nfeatures"] if n > 0])
    assert n_iters == 2, f"expected the 3rd iteration to be skipped as predicted-over-budget, got {n_iters} iterations"
    assert fake_clock.now <= 5 * 60
    assert "would end past the budget" in caplog.text


def test_max_runtime_clock_starts_at_fit_entry(fake_clock, monkeypatch):
    """Max runtime clock starts at fit entry."""
    # Pre-loop setup (hashing / leakage / cardinality checks) costs real wall time on big frames and must count against the budget.
    real_init = fit_mod._init_fit_state

    def _slow_init(*args, **kwargs):
        """Advance the fake clock by four minutes before running the real init."""
        _CLOCK.now += 4 * 60
        return real_init(*args, **kwargs)

    monkeypatch.setattr(fit_mod, "_init_fit_state", _slow_init)
    X, y = _data()
    sel = _rfecv(max_runtime_mins=5)
    sel.fit(X, y)
    n_iters = len([n for n in sel.cv_results_["nfeatures"] if n > 0])
    assert n_iters == 1, f"4 min of setup + 2 min/iteration leaves room for exactly one iteration in 5 min, got {n_iters}"


def test_stop_on_best_iteration_still_updates_best_nfeatures(monkeypatch):
    """Stop on best iteration still updates best nfeatures."""
    # A stop condition used to return before the best-score update, so a fit ending on max_refits/max_runtime never counted its last
    # iteration: with max_refits=1 best_nfeatures stayed 0 and the swap_top_k pass (gated on best_nfeatures > 0) was silently skipped.
    captured = {}
    real_finalize = fit_mod._finalize_fit_results

    def _spy(self, **kwargs):
        """Capture best_nfeatures and best_score passed to finalization, then delegate."""
        captured.update(best_nfeatures=kwargs["best_nfeatures"], best_score=kwargs["best_score"])
        return real_finalize(self, **kwargs)

    monkeypatch.setattr(fit_mod, "_finalize_fit_results", _spy)
    X, y = _data()
    sel = _rfecv(estimator=LogisticRegression(max_iter=200), max_refits=1)
    sel.fit(X, y)
    assert captured["best_nfeatures"] == X.shape[1]
    assert np.isfinite(captured["best_score"])


def test_fit_logs_selected_count_names_and_stop_reason(caplog):
    """Fit logs selected count names and stop reason."""
    X, y = _data()
    sel = _rfecv(estimator=LogisticRegression(max_iter=200), max_refits=4)
    with caplog.at_level(logging.INFO, logger=RFECV_LOGGER):
        sel.fit(X, y)
    kept = [c for c, m in zip(X.columns, sel.support_) if m]
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("RFECV: selected")]
    assert len(summary) == 1
    assert f"RFECV: selected {len(kept)} of {X.shape[1]} features" in summary[0]
    assert "max_refits=4 reached" in summary[0]
    assert f"Selected: [{', '.join(kept)}]" in summary[0]


def test_summary_explains_one_se_rule_keeping_more_than_best_scoring_subset():
    """Summary explains one se rule keeping more than best scoring subset."""
    # The reported case: the progress bar said "best was 56F" but the default one_se_max rule kept all 88 evaluated medoids.
    names = [f"f{i}" for i in range(88)]
    fitted = SimpleNamespace(
        _selected_cols_cache=names, n_features_in_=88, n_features_=88, resolved_n_features_rule_="one_se_max",
        mean_perf_weight=1.0, std_perf_weight=0.1, feature_cost=0.0,
        cv_results_={"nfeatures": [0, 7, 56, 71, 88], "cv_mean_perf": [-0.2, -0.06, -0.0385, -0.0425, -0.0410], "cv_std_perf": [0.0, 0.01, 0.006, 0.006, 0.005]},
    )
    text = build_rfecv_fit_summary(fitted, stop_reason="max_runtime_mins=180.0 reached (186.0 min elapsed)", n_iters=17, elapsed_s=186 * 60)
    assert "RFECV: selected 88 of 88 features after 17 iteration(s)" in text
    assert "stopped: max_runtime_mins=180.0 reached" in text
    assert "Best-scoring subset: 56 features" in text
    assert "Rule one_se_max keeps the largest evaluated size" in text
    assert "n_features_selection_rule='argmax'" in text
    assert "... (+58 more)" in text


def _one_se_fitted(rule, k):
    """Fake fitted RFECV with 88 features whose CV curve plateaus, for one-SE rule replays."""
    names = [f"f{i}" for i in range(88)]
    return SimpleNamespace(
        _selected_cols_cache=names, n_features_in_=88, n_features_=88, resolved_n_features_rule_=rule,
        mean_perf_weight=1.0, std_perf_weight=0.0, feature_cost=0.0, _per_fold_scores={n: [0.0] * k for n in (7, 56, 71, 88)},
        cv_results_={"nfeatures": [0, 7, 56, 71, 88], "cv_mean_perf": [-0.2, -0.06, -0.04, -0.0425, -0.041], "cv_std_perf": [0.0, 0.01, 0.006, 0.006, 0.005]},
    )


def test_summary_states_standard_error_band_arithmetic():
    """Summary states standard error band arithmetic."""
    # floor = -0.0400 - 0.0060/sqrt(9) = -0.0420
    text = build_rfecv_fit_summary(_one_se_fitted("one_se_max", 9), stop_reason=None, n_iters=3, elapsed_s=1.0)
    assert "one standard error (fold std / sqrt(k))" in text
    assert "(-0.0400 - 0.0060/sqrt(9) = -0.0420)" in text


def test_summary_states_legacy_fold_std_band_for_foldstd_rule():
    """Summary states legacy fold std band for foldstd rule."""
    # floor = -0.0400 - 0.0060 = -0.0460
    text = build_rfecv_fit_summary(_one_se_fitted("one_se_max_foldstd", 9), stop_reason=None, n_iters=3, elapsed_s=1.0)
    assert "Rule one_se_max_foldstd keeps the largest evaluated size" in text
    assert "one across-fold std" in text and "(-0.0400 - 0.0060 = -0.0460)" in text
