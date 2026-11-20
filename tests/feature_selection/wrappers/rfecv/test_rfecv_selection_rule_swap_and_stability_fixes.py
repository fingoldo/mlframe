"""NaN candidates in the N-selection rules, the SFFS swap objective/groups, and the stability-selection group/seed/pin handling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge

from mlframe.feature_selection.wrappers.rfecv import RFECV, _sffs, _stability_select


def _pick(rule, mean):
    """Run the N-selection rule over N = 1..len(mean) with a unit std and return the chosen N."""
    mean = np.asarray(mean, dtype=float)
    n = np.arange(1, len(mean) + 1)
    me = SimpleNamespace()
    idx = _stability_select._pick_best_idx_by_rule(me, rule, "se", mean, n > 0, n, n, mean, np.full(len(mean), 0.01), 0.0)
    return int(n[idx])


def test_argmax_rule_never_picks_a_nan_candidate():
    """argsort puts NaN last, so reversing it made a NaN-scored N the 'best'; the rule must skip it."""
    assert _pick("argmax", [0.1, 0.9, float("nan"), 0.5]) == 2


def _swap_self(**kw):
    """A minimal selector stand-in for the SFFS swap pass."""
    base = dict(swap_top_k=1, mean_perf_weight=1.0, std_perf_weight=1.0, _fit_sample_weight_=None, _fit_groups_=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _run_swap(monkeypatch, me, trial_scores, best_score_ref):
    """Run one swap against patched cross_val_score; return (mean dict, captured kwargs)."""
    import sklearn.model_selection as ms

    captured = {}

    def fake_cvs(est, X, y, **kw):
        """Record the call and return the canned fold scores."""
        captured.update(kw)
        return np.asarray(trial_scores)

    monkeypatch.setattr(ms, "cross_val_score", fake_cvs)
    X = pd.DataFrame(np.zeros((6, 2)), columns=["a", "b"])
    means = {1: 0.8}
    _sffs._sffs_swap_pass(
        me, X, np.zeros(6), Ridge(), 2, None, 1, best_score_ref, {1: ["a"]}, {"1_0": {"a": 0.1, "b": 0.9}}, ["a", "b"], means, {1: 0.0}, 0, 4,
    )
    return means, captured


def test_swap_pass_compares_the_penalised_objective(monkeypatch):
    """A swap whose raw mean beats the baseline objective only because the std penalty was left off the trial must be rejected."""
    # trial: mean 0.75, std 0.25 -> objective 0.50; baseline objective 0.69 (mean 0.8, std 0.11).
    means, _ = _run_swap(monkeypatch, _swap_self(), [0.5, 1.0], 0.69)
    assert means[1] == 0.8


def test_swap_pass_accepts_a_genuinely_better_objective(monkeypatch):
    """The corrected comparison still accepts a swap that wins on the objective."""
    means, _ = _run_swap(monkeypatch, _swap_self(), [0.95, 0.95], 0.69)
    assert means[1] == pytest.approx(0.95)


def test_swap_pass_forwards_groups_to_cross_val_score(monkeypatch):
    """Group-aware CV needs the fit's groups; without them every swap raised and was skipped."""
    groups = np.array([0, 0, 1, 1, 2, 2])
    _, captured = _run_swap(monkeypatch, _swap_self(_fit_groups_=groups), [0.9, 0.9], 0.1)
    assert captured["groups"] is groups


class _Recorder(BaseEstimator, ClassifierMixin):
    """Classifier that records the row ids it was fit on and importance-ranks feature 0."""

    fit_ids: list = []
    eval_ids: list = []

    def fit(self, X, y, sample_weight=None):
        """Remember the fitted row ids."""
        type(self).fit_ids.append(set(int(i) for i in X.index))
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.array([1.0] + [0.0] * (X.shape[1] - 1))
        return self

    def predict(self, X):
        """Constant prediction."""
        return np.zeros(len(X), dtype=int)


def _stab_frame(n=60, p=4, seed=0):
    """Frame whose RangeIndex carries the row id (so fitted/held-out rows can be traced)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=["f_a"] + [f"f{i}" for i in range(p - 1)])
    y = (X["f0"].to_numpy() > 0).astype(int)
    return X, y


def test_stability_bootstraps_keep_groups_apart(monkeypatch):
    """With groups given, a bootstrap's fitted rows and its held-out rows must not share a group."""
    X, y = _stab_frame()
    groups = np.repeat(np.arange(12), 5)
    held_out = []

    def fake_fi(**kw):
        """Capture the held-out ids the importances are measured on."""
        held_out.append(set(int(i) for i in kw["data"].index))
        return {c: (1.0 if c == "f0" else 0.0) for c in kw["current_features"]}

    monkeypatch.setattr(_stability_select, "get_feature_importances", fake_fi)
    _Recorder.fit_ids = []
    sel = RFECV(estimator=_Recorder(), stability_selection=True, stability_n_bootstrap=6, random_state=0, verbose=0)
    sel.fit(X, y, groups=groups)
    assert len(_Recorder.fit_ids) == 6 == len(held_out)
    for fit_ids, ho in zip(_Recorder.fit_ids, held_out):
        assert {groups[i] for i in fit_ids}.isdisjoint({groups[i] for i in ho})


def test_stability_without_random_state_is_repeatable():
    """random_state=None must derive its seed from the data (as the main path does), not from system entropy."""
    rng = np.random.default_rng(0)
    n, p = 120, 12
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"].to_numpy() + 3 * rng.normal(size=n)
    freqs = []
    for _ in range(2):
        sel = RFECV(estimator=Ridge(), stability_selection=True, stability_n_bootstrap=10, stability_top_k=3, importance_getter="auto", random_state=None, verbose=0)
        sel.fit(X, y)
        freqs.append(np.array(sel.stability_selection_freq_))
    assert np.array_equal(freqs[0], freqs[1])


def test_stability_pins_must_include_by_index_on_ndarray():
    """must_include given as integer column indices on an ndarray X must be pinned in support_ on the stability path too."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = X[:, 0] + 0.5 * rng.normal(size=80)
    sel = RFECV(estimator=Ridge(), stability_selection=True, stability_n_bootstrap=4, stability_top_k=1, stability_threshold=0.99, must_include=[3], random_state=0, verbose=0)
    sel.fit(X, y)
    assert bool(np.asarray(sel.support_)[3])
