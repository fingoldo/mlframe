"""Stale per-fit state, nullable-dtype blind spots and silent fallbacks in the RFECV fit path."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from mlframe.feature_selection.wrappers._enums import OptimumSearch
from mlframe.feature_selection.wrappers.rfecv import RFECV, _fit, _fit_fold, _prescreen
from mlframe.utils.log_throttle import reset_throttle_counts

_LOGGER = "mlframe.feature_selection.wrappers.rfecv"


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Start every test with empty log-throttle counters so a WARNING is never swallowed by an earlier test."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _reg_frame(n=120, p=4, seed=0):
    """A small regression problem."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"].to_numpy() * 2 + rng.normal(scale=0.1, size=n)
    return X, y


def test_consecutive_nan_counter_resets_between_fits():
    """A NaN streak left on the instance by an earlier fit must not count toward the 5-NaN abort of the next fit."""

    def scorer(est, X, y):
        """NaN on the full 4-feature subset only."""
        return float("nan") if X.shape[1] == 4 else -float(np.mean((est.predict(X) - y) ** 2))

    X, y = _reg_frame()
    sel = RFECV(estimator=Ridge(), cv=2, scoring=scorer, top_predictors_search_method=OptimumSearch.ExhaustiveRandom, random_state=0, verbose=0)
    sel._consecutive_nan_iters = 4
    sel.fit(X, y)
    assert sel._consecutive_nan_iters == 1


def test_refit_after_minority_reduction_restores_requested_cv():
    """cv reduced for a rare-positive target must go back to the requested value when the next fit has enough positives."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y_rare = np.zeros(n, dtype=int)
    y_rare[:2] = 1
    sel = RFECV(estimator=RandomForestClassifier(n_estimators=5, random_state=0), cv=5, random_state=0, verbose=0)
    sel.fit(X, y_rare)
    assert sel.cv == 2
    sel.fit(X, rng.integers(0, 2, n))
    assert sel.cv == 5


def test_scorer_rejecting_sample_weight_is_logged(caplog):
    """A scorer that cannot take sample_weight scores the fold unweighted, and that is announced at WARNING."""

    def scorer(fitted, X, y):
        """No sample_weight keyword."""
        return 1.0

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _fit_fold._score_with_optional_weight(scorer, None, None, None, np.ones(3))
    assert out == 1.0
    assert any("UNWEIGHTED" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_estimator_without_sample_weight_is_logged(caplog):
    """An estimator whose fit() lacks sample_weight is fit unweighted while the scorer is weighted: that must be logged."""

    class NoWeightRidge(Ridge):
        """Ridge whose fit() does not take sample_weight."""

        def fit(self, X, y):
            """Fit without weights."""
            return super().fit(X, y)

    X, y = _reg_frame()
    sel = RFECV(estimator=NoWeightRidge(), cv=2, top_predictors_search_method=OptimumSearch.ExhaustiveRandom, max_refits=2, random_state=0, verbose=0)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        sel.fit(X, y, sample_weight=np.ones(len(y)))
    assert any("does not accept sample_weight" in r.getMessage() for r in caplog.records)


def test_empty_fold_prescreen_mask_is_logged(caplog):
    """When the train-only prescreen leaves none of the searched features, the fold falls back to the unmasked subset and says so."""
    train_index = np.arange(10)
    key = _prescreen._fold_universe_key(train_index)
    me = SimpleNamespace(_prescreen_fold_universes={key: frozenset({"zzz"})})
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _fit_fold._apply_fold_prescreen(me, train_index, ["a", "b"])
    assert out == ["a", "b"]
    assert any("train-only prescreen" in r.getMessage() for r in caplog.records)


def test_fit_keyword_arguments_that_are_dropped_are_logged(caplog):
    """fit(**kw) reads only 'timestamps'; any other keyword is ignored and the caller is told."""
    me = SimpleNamespace(fit_params={"keep": 1})
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _fit._resolve_fit_params(me, {"eval_set": 1, "keep": 2, "timestamps": [1]})
    assert out == {"keep": 1, "timestamps": [1]}
    assert any("eval_set" in r.getMessage() for r in caplog.records)


class _FailingMRMR:
    """MRMR stand-in whose fit raises."""

    def __init__(self, **kw):
        """Accept and ignore the constructor kwargs."""

    def fit(self, X, y):
        """Always fail."""
        raise RuntimeError("boom")


def _raising_callable(X, y):
    """Prescreen callable that fails."""
    raise RuntimeError("boom")


@pytest.mark.parametrize("case", ["callable", "mrmr", "univariate_ndarray", "univariate_error", "unknown"])
def test_prescreen_fallbacks_are_logged_at_warning_without_verbose(case, caplog, monkeypatch):
    """Every prescreen fallback (the user asked for a filter and got none) logs a WARNING even at verbose=0."""
    X, y = _reg_frame()
    feats = list(X.columns)
    prescreen = {"callable": _raising_callable, "mrmr": "mrmr", "univariate_ndarray": "univariate_ht", "univariate_error": "univariate_ht", "unknown": "nonsense"}[case]
    me = SimpleNamespace(prescreen=prescreen, prescreen_top_k=None, random_state=0)
    if case == "mrmr":
        monkeypatch.setattr("mlframe.feature_selection.filters.mrmr.MRMR", _FailingMRMR)
    if case == "univariate_error":
        monkeypatch.setattr("mlframe.feature_selection.wrappers._univariate_ht.calculate_relevance_table", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    data = X.to_numpy() if case == "univariate_ndarray" else X
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _prescreen._apply_prescreen(me, X=data, y=y, candidate_features=feats)
    assert out == feats
    assert any(r.levelno == logging.WARNING for r in caplog.records), caplog.records


def test_nested_prescreen_precompute_failure_is_logged_at_warning(caplog):
    """A failing per-fold prescreen precompute falls back to the global estimate and logs a WARNING at verbose=0."""

    class BadCV:
        """Splitter whose split raises."""

        def split(self, **kw):
            """Always fail."""
            raise RuntimeError("boom")

    X, y = _reg_frame()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _prescreen._precompute_prescreen_fold_universes(SimpleNamespace(), X=X, y=y, groups=None, cv=BadCV(), full_features=list(X.columns), verbose=0)
    assert out is None
    assert any(r.levelno == logging.WARNING and "Nested prescreen" in r.getMessage() for r in caplog.records)


def test_must_include_duplicates_are_collapsed():
    """A feature listed twice in must_include is pinned once, not fed to the estimator as two identical columns."""
    from mlframe.feature_selection.wrappers.rfecv._must_include import _resolve_must_include

    X, _ = _reg_frame()
    me = SimpleNamespace(must_include=["f0", "f0", "f1"])
    rest, pinned = _resolve_must_include(me, X=X, original_features=list(X.columns), verbose=0)
    assert pinned == ["f0", "f1"]
    assert rest == ["f2", "f3"]


def test_mbh_search_space_has_no_duplicate_when_cap_reaches_p(monkeypatch):
    """max_nfeatures >= p must not list the full-set size twice in the optimizer's search space."""
    from mlframe.feature_selection.wrappers.rfecv import _mbh_optimizer

    seen = {}
    monkeypatch.setattr(_mbh_optimizer, "MBHOptimizer", lambda **kw: seen.update(kw))
    me = SimpleNamespace(
        optimizer_plotting=None, max_nfeatures=10, optimizer_config=None, _rng=np.random.default_rng(0), mbh_adaptive_threshold=30, init_design_size="auto",
    )
    _mbh_optimizer._build_mbh_optimizer(
        me, original_features=list("abcde"), max_refits=None, top_predictors_search_method=OptimumSearch.ModelBasedHeuristic,
    )
    space = list(seen["search_space"])
    assert len(space) == len(set(space))


def test_multioutput_over_polars_frame_keeps_feature_names():
    """A polars X must map the sub-fits' selected names onto its real columns, not onto '0','1',..."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    n = 200
    X = pl.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
    Y = np.column_stack([X["f0"].to_numpy() * 2, X["f1"].to_numpy() * 2])
    sel = RFECV(estimator=Ridge(), cv=2, multioutput_strategy="union", random_state=0, verbose=0)
    sel.fit(X, Y)
    assert sel.feature_names_in_ == ["f0", "f1", "f2", "f3"]
    assert sel.support_.any()
