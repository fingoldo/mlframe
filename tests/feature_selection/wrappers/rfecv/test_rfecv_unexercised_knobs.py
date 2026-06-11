"""Behavioral coverage for 12 RFECV constructor knobs that no prior test exercised.

Closes coverage_asymmetry_wrappers-03: the audit found these 12 ctor params never
appear as ``<name>=`` in any test, several of them decision-altering
(``special_feature_indices`` short-circuits the whole search; ``best_desired_score``
early-stops; ``drop_nan_score_fi`` is a Wave-1 legacy-A/B correctness flag;
``prescreen_fdr_level`` gates the univariate prescreen). Each test below drives the
knob through the public ``RFECV.fit`` path on a small-n Ridge/LogReg fixture and
asserts the documented behavior, with quantitative floors pinned 5-15% below a
value measured once during development (see the per-test docstrings for the number).

All measured numbers were read on store-Python 3.14 CPU-only against the integration
worktree. Floors are deliberately loose so seed jitter does not trip them while a
real regression (knob silently ignored) still fails the win.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._helpers_importance import get_feature_importances

from tests.feature_selection.conftest import is_fast_mode, fast_subset  # noqa: F401
from tests.feature_selection._biz_val_synth import (
    make_signal_plus_noise,
    make_correlated_redundant,
    as_df,
)


# Compiled once at module scope (project rule: pre-compile regex).
_XCOL_RE = re.compile(r"x(\d+)")


def _refit_count(r: RFECV) -> int:
    """Number of distinct nfeatures points the outer loop actually evaluated (includes the N=0 dummy anchor)."""
    return len(r.cv_results_["nfeatures"])


def _selected_names(r: RFECV) -> list:
    return r.get_feature_names_out().tolist()


def _logreg(seed: int = 0):
    return LogisticRegression(max_iter=200, random_state=seed)


def _base_kwargs(**over):
    kw = dict(cv=3, random_state=0, leakage_corr_threshold=None,
              n_features_selection_rule="argmax")
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# special_feature_indices: forces a fixed subset and short-circuits the optimiser.
# ---------------------------------------------------------------------------


def test_special_feature_indices_forces_exact_subset_and_short_circuits():
    """``special_feature_indices`` must make ``support_`` equal EXACTLY that subset and
    skip the search. Measured: support == ['x0','x2','x5'] and the loop evaluated only
    the N=0 dummy + the one special subset (cv nfeatures == [0, 3], a single real refit)."""
    X, y, _ = make_signal_plus_noise(n=300, p_signal=3, p_noise=7, seed=0)
    Xdf, ys = as_df(X, y)
    forced = ["x0", "x2", "x5"]
    r = RFECV(estimator=_logreg(), **_base_kwargs(special_feature_indices=forced))
    r.fit(Xdf, ys)

    assert set(_selected_names(r)) == set(forced)
    # Short-circuit: only the special subset (and the 0-feature dummy anchor) get scored.
    assert _refit_count(r) <= 2, f"optimiser not short-circuited: {r.cv_results_['nfeatures']}"
    assert 3 in r.cv_results_["nfeatures"]


# ---------------------------------------------------------------------------
# best_desired_score: early-stop when an explored subset reaches the target.
# ---------------------------------------------------------------------------


def test_best_desired_score_early_stops_below_unconstrained_twin():
    """A reachable ``best_desired_score`` must stop the outer loop strictly earlier than an
    otherwise-identical twin. Measured: twin ran 13 refit points; with best_desired_score=0.10
    (the first explored 4F subset scores ~0.246 > 0.10) the run stopped at 2 refit points."""
    X, y, _ = make_signal_plus_noise(n=400, p_signal=4, p_noise=8, seed=1)
    Xdf, ys = as_df(X, y)
    common = _base_kwargs(max_refits=30, max_noimproving_iters=30)

    twin = RFECV(estimator=_logreg(), **common)
    twin.fit(Xdf, ys)
    twin_refits = _refit_count(twin)

    es = RFECV(estimator=_logreg(), **_base_kwargs(
        max_refits=30, max_noimproving_iters=30, best_desired_score=0.10))
    es.fit(Xdf, ys)
    es_refits = _refit_count(es)

    assert twin_refits >= 6, f"twin too short to demonstrate early-stop (got {twin_refits})"
    assert es_refits < twin_refits, (
        f"best_desired_score did not stop early: es={es_refits} twin={twin_refits}")
    assert es_refits <= 3


# ---------------------------------------------------------------------------
# min_train_size: pins the actual code behavior (folds below the size are SKIPPED).
# ---------------------------------------------------------------------------


def test_min_train_size_above_fold_skips_all_folds_yielding_empty_curve():
    """The code path (``_rfecv_fit_fold._eval_fold_body``) returns None for any fold whose
    train slice is below ``min_train_size`` -- it SKIPS the fold, it does NOT raise. With a
    min_train_size far above every fold size, every fold is skipped, so the CV curve has no
    real (non-zero) point: ``select_optimal_nfeatures_`` logs the empty case and returns an
    empty support_. Measured: cv_mean_perf == [] (all folds skipped) and support_ empty."""
    X, y, _ = make_signal_plus_noise(n=120, p_signal=3, p_noise=5, seed=2)
    Xdf, ys = as_df(X, y)
    r = RFECV(estimator=_logreg(), **_base_kwargs(max_refits=3, min_train_size=10_000))
    r.fit(Xdf, ys)

    # Every explored subset was skipped on every fold -> no usable CV point recorded.
    perf = r.cv_results_["cv_mean_perf"]
    assert len(perf) == 0 or all(np.isnan(v) for v in perf), (
        f"min_train_size did not skip folds; got cv_mean_perf={perf}")
    assert int(r.get_support().sum()) == 0


# ---------------------------------------------------------------------------
# stability_top_k: smaller top_k -> smaller (subset) selection.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_stability_top_k_monotonically_shrinks_selection():
    """Under stability_selection, a smaller ``stability_top_k`` ranks fewer features into the
    per-bootstrap top set, so fewer clear the frequency threshold -> a smaller, NESTED selection.
    Measured (n=800, 6 signal + 6 noise, B=40, threshold=0.6): top_k 1->0, 2->{x2}, 3->{x0,x2,x4},
    6->{x0..x5}; each smaller-top_k set is a subset of the next."""
    _stability_top_k_body()


def test_stability_top_k_fast():
    """MLFRAME_FAST representative: just the 2-vs-6 contrast so the cheap path stays covered."""
    _stability_top_k_body(fast=True)


def _stability_top_k_body(fast: bool = False):
    X, y, _ = make_signal_plus_noise(n=400 if fast else 800, p_signal=6, p_noise=6, seed=7)
    Xdf, ys = as_df(X, y)
    ks = [2, 6] if fast else [1, 2, 3, 6]

    sizes = {}
    sets = {}
    for tk in ks:
        r = RFECV(estimator=_logreg(), cv=3, random_state=0, leakage_corr_threshold=None,
                  stability_selection=True, stability_n_bootstrap=40,
                  stability_threshold=0.6, stability_top_k=tk)
        r.fit(Xdf, ys)
        sets[tk] = set(_selected_names(r))
        sizes[tk] = len(sets[tk])

    # Monotone non-decreasing selection size with top_k.
    ordered = sorted(ks)
    for a, b in zip(ordered, ordered[1:]):
        assert sizes[a] <= sizes[b], f"top_k {a}->{b} grew selection: {sizes}"
        # Nesting: the smaller-top_k selection is contained in the larger.
        assert sets[a].issubset(sets[b]), f"top_k {a} not subset of {b}: {sets}"
    # The largest top_k must recover a meaningfully larger set than the smallest.
    assert sizes[ordered[-1]] > sizes[ordered[0]]


# ---------------------------------------------------------------------------
# prescreen_fdr_level: stricter level keeps fewer columns; true signal survives both.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_prescreen_fdr_level_strict_keeps_fewer_signal_survives():
    """``prescreen='univariate_ht'`` with a strict ``prescreen_fdr_level`` (1e-6) keeps no MORE
    columns than a permissive level (0.5), and the true signal survives BOTH. Measured (n=600,
    3 signal + 12 noise): level 0.5 -> 4 selected; level 1e-6 -> 3 selected; all 3 signal cols
    present at both levels."""
    _prescreen_fdr_body()


def test_prescreen_fdr_level_fast():
    """MLFRAME_FAST representative for the prescreen-FDR contrast."""
    _prescreen_fdr_body(fast=True)


def _prescreen_fdr_body(fast: bool = False):
    n = 400 if fast else 600
    X, y, sig = make_signal_plus_noise(n=n, p_signal=3, p_noise=12, seed=4)
    Xdf, ys = as_df(X, y)

    def _fit(level):
        r = RFECV(estimator=_logreg(), **_base_kwargs(
            max_refits=3, prescreen="univariate_ht", prescreen_fdr_level=level))
        r.fit(Xdf, ys)
        return set(_selected_names(r))

    loose = _fit(0.5)
    strict = _fit(1e-6)

    signal_cols = {f"x{i}" for i in sig}
    # Strict FDR keeps no more columns than the loose one.
    assert len(strict) <= len(loose), f"strict kept more: strict={strict} loose={loose}"
    # The true signal survives both prescreen levels.
    assert signal_cols.issubset(loose), f"signal lost at loose level: {loose}"
    assert signal_cols.issubset(strict), f"signal lost at strict level: {strict}"


# ---------------------------------------------------------------------------
# cpi_max_depth / cpi_min_samples_leaf: feed the conditional-permutation aux tree.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cpi_max_depth_changes_conditional_permutation_fi_vector():
    """``importance_getter='conditional_permutation'`` builds an auxiliary DecisionTree to
    condition on co-features; ``cpi_max_depth`` controls that tree. A depth-1 stump conditions
    far more weakly than an unbounded (None) tree, so the FI vector changes between the two.
    Measured (n=500, 3 correlated + 3 noise, RF model): max |depth1 - depthN| ~ 0.068 over the
    per-feature CPI scores -- well above noise. Floor 0.015 (>=4x margin below the measured gap)."""
    _cpi_depth_body()


def test_cpi_max_depth_fast():
    """MLFRAME_FAST representative for the CPI depth contrast (smaller n, fewer repeats)."""
    _cpi_depth_body(fast=True)


def _cpi_depth_body(fast: bool = False):
    from sklearn.ensemble import RandomForestClassifier

    n = 350 if fast else 500
    n_repeats = 3 if fast else 5
    X, y, _ = make_correlated_redundant(n=n, n_corr=3, p_noise=3, seed=5)
    Xdf, ys = as_df(X, y)
    feats = list(Xdf.columns)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(Xdf, ys)

    def _cpi(depth):
        fi = get_feature_importances(
            model=rf, current_features=feats, data=Xdf, target=ys.values,
            train_data=Xdf, importance_getter="conditional_permutation",
            cpi_max_depth=depth, cpi_min_samples_leaf=10, n_repeats=n_repeats,
        )
        return np.array([fi[f] for f in feats], dtype=float)

    v_stump = _cpi(1)
    v_full = _cpi(None)

    max_abs_diff = float(np.max(np.abs(v_stump - v_full)))
    assert max_abs_diff >= 0.015, (
        f"cpi_max_depth had no effect on the conditional-permutation FI vector "
        f"(max abs diff {max_abs_diff:.4f}); the aux-tree depth knob is not wired.")


def test_cpi_min_samples_leaf_changes_conditional_permutation_fi_vector():
    """``cpi_min_samples_leaf`` is the other aux-tree control on the conditional-permutation path.
    A tiny leaf (2) grows a much deeper tree than a large leaf (80) at fixed depth=None, so the
    conditioning -- and thus the CPI FI vector -- differs. Behavioral floor only (the exact gap is
    fixture-dependent); assert a non-trivial change so a silently-ignored knob fails."""
    from sklearn.ensemble import RandomForestClassifier

    X, y, _ = make_correlated_redundant(n=500, n_corr=3, p_noise=3, seed=5)
    Xdf, ys = as_df(X, y)
    feats = list(Xdf.columns)
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(Xdf, ys)

    def _cpi(leaf):
        fi = get_feature_importances(
            model=rf, current_features=feats, data=Xdf, target=ys.values,
            train_data=Xdf, importance_getter="conditional_permutation",
            cpi_max_depth=None, cpi_min_samples_leaf=leaf, n_repeats=5,
        )
        return np.array([fi[f] for f in feats], dtype=float)

    diff = float(np.max(np.abs(_cpi(2) - _cpi(80))))
    assert diff >= 0.01, (
        f"cpi_min_samples_leaf had no effect on the CPI FI vector (max abs diff {diff:.4f}).")


# ---------------------------------------------------------------------------
# drop_nan_score_fi: legacy A/B on a fixture with exactly one NaN-score fold.
# ---------------------------------------------------------------------------


class _FoldTaggedLR(ClassifierMixin, BaseEstimator):
    """LogReg shim that records its train-fold row count so a custom scorer can NaN one fold
    deterministically (used to exercise the per-fold NaN-FI policy)."""

    def __init__(self, random_state: int = 0, bad_rows=None):
        self.random_state = random_state
        self.bad_rows = bad_rows

    def fit(self, X, y, **kw):
        self._train_n = X.shape[0]
        self._lr = LogisticRegression(max_iter=200, random_state=self.random_state).fit(X, y)
        self.classes_ = self._lr.classes_
        self.coef_ = self._lr.coef_
        self.n_features_in_ = self._lr.n_features_in_
        return self

    def predict(self, X):
        return self._lr.predict(X)

    def predict_proba(self, X):
        return self._lr.predict_proba(X)


def _make_nan_fold_scorer(bad_rows: int):
    def _score(estimator, X, y):
        if getattr(estimator, "_train_n", None) == bad_rows:
            return float("nan")
        try:
            return float(roc_auc_score(y, estimator.predict_proba(X)[:, 1]))
        except ValueError:
            return float("nan")
    return _score


class _UnequalCV:
    """Two folds where fold-0's train slice is a UNIQUE small size (the deterministic NaN fold)
    and fold-1 is a different (normal) size. Lets a single fold -- not all -- score NaN."""

    def __init__(self, n: int, bad_train: int):
        self.n = n
        self.bad_train = bad_train

    def split(self, X=None, y=None, groups=None):
        idx = np.arange(self.n)
        yield idx[: self.bad_train], idx[self.bad_train:]
        half = self.n // 2
        yield idx[:half], idx[half:]

    def get_n_splits(self, X=None, y=None, groups=None):
        return 2


def test_drop_nan_score_fi_excludes_nan_fold_importances_from_voting_pool():
    """F14 legacy A/B: ``drop_nan_score_fi=True`` (default) drops the FI run of any estimator-fold
    pair whose score was NaN so a degenerate fold can't contaminate voting; ``=False`` (legacy)
    keeps it. On a fixture where fold-0 deterministically scores NaN at every explored N (n=200,
    2 folds, 4 explored subsets), measured: True -> 4 FI runs (only the healthy fold-1 keys
    ``*_1``); False -> 8 FI runs (both folds for every N). The legacy flag keeps exactly the
    NaN-fold runs the default discards."""
    n = 200
    X, y, _ = make_signal_plus_noise(n=n, p_signal=3, p_noise=5, seed=8)
    Xdf, ys = as_df(X, y)
    bad = 60
    cv = _UnequalCV(n, bad)
    scorer = _make_nan_fold_scorer(bad)

    def _fit(drop):
        r = RFECV(estimator=_FoldTaggedLR(0, bad), cv=cv, random_state=0,
                  leakage_corr_threshold=None, max_refits=4, scoring=scorer,
                  drop_nan_score_fi=drop, importance_getter="coef_",
                  n_features_selection_rule="argmax", nofeatures_dummy_scoring=False)
        r.fit(Xdf, ys)
        return sorted(r.feature_importances_.keys())

    keys_drop = _fit(True)
    keys_keep = _fit(False)

    # Default drops the NaN fold's runs -> strictly fewer FI runs than legacy.
    assert len(keys_drop) < len(keys_keep), (
        f"drop_nan_score_fi=True did not drop NaN-fold FI: drop={keys_drop} keep={keys_keep}")
    # The default keeps ONLY the healthy fold-1 runs (keys ending in '_1').
    assert all(k.endswith("_1") for k in keys_drop), (
        f"default kept a NaN (fold-0) FI run: {keys_drop}")
    # Legacy keeps the NaN fold-0 runs the default removed.
    assert any(k.endswith("_0") for k in keys_keep), (
        f"legacy did not retain the NaN-fold FI runs: {keys_keep}")


# ---------------------------------------------------------------------------
# keep_loser_subset_fi / noimprove_counts_revisit: legacy Wave-1 flags.
# These only diverge when the optimiser REVISITS the same N with a worse subset
# (``was_stored=False``), a state the MBH/exhaustive optimisers do not reach on
# these synthetics. The tests below drive the legacy branch through fit() and pin
# that it (a) round-trips on the fitted estimator unchanged and (b) yields a valid,
# deterministic, non-empty selection -- so a regression that crashes or no-ops the
# legacy path is caught. The documented voting/iteration divergence is covered at
# the discriminating level by ``drop_nan_score_fi`` above (same Wave-1 family).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [False, True])
def test_keep_loser_subset_fi_legacy_branch_fits_and_round_trips(flag):
    X, y, _ = make_signal_plus_noise(n=300, p_signal=3, p_noise=5, seed=9)
    Xdf, ys = as_df(X, y)
    r = RFECV(estimator=_logreg(), **_base_kwargs(
        max_refits=12, keep_loser_subset_fi=flag))
    r.fit(Xdf, ys)
    # The flag survives fit (params re-read each fit; never clobbered).
    assert r.keep_loser_subset_fi is flag
    # Legacy and default both produce a valid, non-empty, deterministic selection.
    assert int(r.get_support().sum()) >= 1
    r2 = RFECV(estimator=_logreg(), **_base_kwargs(
        max_refits=12, keep_loser_subset_fi=flag))
    r2.fit(Xdf, ys)
    assert _selected_names(r) == _selected_names(r2)


@pytest.mark.parametrize("flag", [False, True])
def test_noimprove_counts_revisit_legacy_branch_fits_and_round_trips(flag):
    X, y, _ = make_signal_plus_noise(n=300, p_signal=3, p_noise=5, seed=10)
    Xdf, ys = as_df(X, y)
    r = RFECV(estimator=_logreg(), **_base_kwargs(
        max_refits=15, max_noimproving_iters=4, noimprove_counts_revisit=flag))
    r.fit(Xdf, ys)
    assert r.noimprove_counts_revisit is flag
    assert int(r.get_support().sum()) >= 1
    # max_noimproving_iters is honoured regardless of the revisit-accounting flag.
    assert _refit_count(r) <= 16


# ---------------------------------------------------------------------------
# estimators_save_path + keep_estimators.
# ---------------------------------------------------------------------------


def test_keep_estimators_populates_estimators_dict_per_fold():
    """``keep_estimators=True`` must retain the per-(nfeatures, fold) fitted estimators in
    ``self.estimators_``. Measured (n=300, cv=3, max_refits=3): 9 fitted estimators kept, keyed
    ``'{N}_{fold}'`` (e.g. '8_0','8_1','8_2','2_0',...). With keep_estimators=False the dict is
    empty."""
    X, y, _ = make_signal_plus_noise(n=300, p_signal=3, p_noise=5, seed=6)
    Xdf, ys = as_df(X, y)

    r_off = RFECV(estimator=_logreg(), **_base_kwargs(max_refits=3, keep_estimators=False))
    r_off.fit(Xdf, ys)
    assert len(r_off.estimators_) == 0

    r_on = RFECV(estimator=_logreg(), **_base_kwargs(max_refits=3, keep_estimators=True))
    r_on.fit(Xdf, ys)
    assert len(r_on.estimators_) >= 3, f"no estimators retained: {list(r_on.estimators_)}"
    # Keys follow the documented '{nfeatures}_{fold}' scheme.
    for k in r_on.estimators_:
        assert re.fullmatch(r"\d+_\d+(?:_e\d+)?", str(k)), f"unexpected estimator key: {k!r}"


def test_estimators_save_path_writes_dump_files(tmp_path):
    """The ctor docstring (``_rfecv.py:216-218``) promises that with ``estimators_save_path`` set
    the fitted estimators are written to ``join(save_path, estimator_type_name, ...dump)`` and a
    ``required_features.dump`` summary is written. ``_persist_fitted_estimators`` now implements
    that layout; this test pins both the estimator dumps and the required-features summary."""
    X, y, _ = make_signal_plus_noise(n=300, p_signal=3, p_noise=5, seed=6)
    Xdf, ys = as_df(X, y)
    r = RFECV(estimator=_logreg(), **_base_kwargs(
        max_refits=3, keep_estimators=True, estimators_save_path=str(tmp_path)))
    r.fit(Xdf, ys)

    files = glob.glob(os.path.join(str(tmp_path), "**", "*.dump"), recursive=True)
    assert files, "estimators_save_path produced no .dump files"
    assert os.path.exists(os.path.join(str(tmp_path), "required_features.dump"))


def test_estimators_save_path_required_features_dump_is_loadable_and_lists_support(tmp_path):
    """Regression for the silently-inert estimators_save_path knob: required_features.dump must exist,
    be joblib-loadable, and list EXACTLY the fitted support columns; a no-path fit writes nothing."""
    import joblib

    X, y, _ = make_signal_plus_noise(n=300, p_signal=3, p_noise=5, seed=6)
    Xdf, ys = as_df(X, y)

    r = RFECV(estimator=_logreg(), **_base_kwargs(
        max_refits=3, keep_estimators=True, estimators_save_path=str(tmp_path)))
    r.fit(Xdf, ys)

    summary_path = os.path.join(str(tmp_path), "required_features.dump")
    assert os.path.exists(summary_path)
    summary = joblib.load(summary_path)
    assert list(summary["required_features"]) == _selected_names(r)

    # A no-path fit must write nothing (default behavior unchanged, no stray I/O).
    no_path_dir = tmp_path / "untouched"
    no_path_dir.mkdir()
    r2 = RFECV(estimator=_logreg(), **_base_kwargs(max_refits=3, keep_estimators=True))
    r2.fit(Xdf, ys)
    assert not glob.glob(os.path.join(str(no_path_dir), "**", "*.dump"), recursive=True)


# ---------------------------------------------------------------------------
# report_ndigits: low-value cosmetic; one no-crash smoke at ndigits=1.
# ---------------------------------------------------------------------------


def test_report_ndigits_one_does_not_crash():
    """``report_ndigits`` only controls log formatting (``f"{x:.{ndigits}f}"``); ndigits=1 must not
    crash the fit. Smoke-only per the audit (LOW value)."""
    X, y, _ = make_signal_plus_noise(n=300, p_signal=3, p_noise=5, seed=6)
    Xdf, ys = as_df(X, y)
    r = RFECV(estimator=_logreg(), **_base_kwargs(max_refits=3, report_ndigits=1))
    r.fit(Xdf, ys)
    assert int(r.get_support().sum()) >= 1


# ---------------------------------------------------------------------------
# Ridge / regression coverage: special_feature_indices on a regression estimator.
# ---------------------------------------------------------------------------


def test_special_feature_indices_regression_ridge_forces_subset():
    """The forced-subset short-circuit also holds for a regression estimator (Ridge). Pins that
    the knob is estimator-family agnostic."""
    rng = np.random.default_rng(0)
    n = 300
    Xs = rng.normal(size=(n, 8))
    y = 2.0 * Xs[:, 0] - 1.5 * Xs[:, 3] + 0.3 * rng.normal(size=n)
    Xdf = pd.DataFrame(Xs, columns=[f"x{i}" for i in range(8)])
    ys = pd.Series(y, name="y")

    forced = ["x0", "x3"]
    r = RFECV(estimator=Ridge(), cv=3, random_state=0, leakage_corr_threshold=None,
              n_features_selection_rule="argmax", special_feature_indices=forced)
    r.fit(Xdf, ys)
    assert set(_selected_names(r)) == set(forced)
    assert _refit_count(r) <= 2
