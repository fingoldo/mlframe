"""Regression test for audit2 leaks-F2: the BorutaShap binom-test memo was ``lru_cache(maxsize=None)``,
which grows for the whole process lifetime across many differently-sized fits. It is now bounded, so a
long-lived worker cannot pin it unbounded. Assert the cache is bounded AND still serves repeat calls.
"""

from mlframe.feature_selection.boruta_shap import _binom_test_cached


def test_binom_cache_is_bounded_not_unbounded():
    info = _binom_test_cached.cache_info()
    assert info.maxsize is not None, "binom-test cache must be bounded (lru_cache maxsize=None leaks)"
    assert info.maxsize >= 10_000, "bound must be generous enough not to evict within a single fit"


def test_binom_cache_still_memoizes():
    _binom_test_cached.cache_clear()
    a = _binom_test_cached(3, 10, 0.5, "two-sided")
    hits_before = _binom_test_cached.cache_info().hits
    b = _binom_test_cached(3, 10, 0.5, "two-sided")
    assert a == b
    assert _binom_test_cached.cache_info().hits == hits_before + 1, "second identical call must be a cache hit"
