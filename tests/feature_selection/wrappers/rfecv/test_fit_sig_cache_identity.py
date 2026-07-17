"""Regression: the cached _fit_accepts_sample_weight must return the same verdict as the
pre-cache per-call inspect.signature(est.fit) introspection, for sw-accepting, sw-rejecting,
and **kwargs estimators."""

import inspect

import pytest

from mlframe.feature_selection.wrappers.rfecv._fit_fold import _fit_accepts_sample_weight


def _raw(est):
    """Pre-fix logic: introspect the BOUND fit method directly."""
    try:
        sig = inspect.signature(est.fit)
    except (TypeError, ValueError):
        return False
    p = sig.parameters
    return "sample_weight" in p or any(x.kind == inspect.Parameter.VAR_KEYWORD for x in p.values())


class _AcceptsSW:
    def fit(self, X, y, sample_weight=None):
        return self


class _RejectsSW:
    def fit(self, X, y):
        return self


class _Kwargs:
    def fit(self, X, y, **kwargs):
        return self


def _cached(est):
    key = getattr(est.fit, "__func__", est.fit)
    return _fit_accepts_sample_weight(key)


@pytest.mark.parametrize(
    "cls, expected",
    [(_AcceptsSW, True), (_RejectsSW, False), (_Kwargs, True)],
)
def test_cached_matches_raw_and_expected(cls, expected):
    est = cls()
    assert _cached(est) == _raw(est) == expected


def test_sklearn_estimators_match_raw():
    rf = pytest.importorskip("sklearn.ensemble").RandomForestClassifier()
    assert _cached(rf) == _raw(rf)


def test_cache_hits_across_instances():
    a, b = _AcceptsSW(), _AcceptsSW()
    _fit_accepts_sample_weight.cache_clear()
    _cached(a)
    info_before = _fit_accepts_sample_weight.cache_info()
    _cached(b)  # same class -> same __func__ key -> must be a cache hit
    info_after = _fit_accepts_sample_weight.cache_info()
    assert info_after.hits == info_before.hits + 1
