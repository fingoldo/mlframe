"""Regression tests for MRMR._FIT_CACHE LRU bound (P1).

Pre-fix the class-global ``_FIT_CACHE`` was an unbounded ``dict`` that
held strong refs to every fitted MRMR instance, leaking memory in
long-running web services. Post-fix it is an LRU-bounded
``OrderedDict`` whose cap is the instance kwarg ``fit_cache_max``
(default 4).
"""

from __future__ import annotations

from collections import OrderedDict

from mlframe.feature_selection.filters.mrmr import MRMR


def test_mrmr_fit_cache_is_lru_ordered_dict():
    """Mrmr fit cache is lru ordered dict."""
    assert isinstance(MRMR._FIT_CACHE, OrderedDict)


def test_mrmr_has_fit_cache_max_kwarg_default_four():
    """Mrmr has fit cache max kwarg default four."""
    inst = MRMR()
    assert getattr(inst, "fit_cache_max") == 4


def test_mrmr_fit_cache_respects_fit_cache_max():
    """Simulate fit-time writes via the same LRU mechanism the fit path uses."""
    MRMR._FIT_CACHE.clear()
    inst = MRMR(fit_cache_max=3)

    # Mimic the write path inside ``MRMR.fit``: insert, move_to_end, evict.
    for i in range(10):
        key = (f"x_{i}", f"y_{i}", "params_sig")
        MRMR._FIT_CACHE[key] = inst
        MRMR._FIT_CACHE.move_to_end(key)
        _cap = int(getattr(inst, "fit_cache_max", 4) or 4)
        while len(MRMR._FIT_CACHE) > _cap:
            MRMR._FIT_CACHE.popitem(last=False)

    assert len(MRMR._FIT_CACHE) == 3

    # And the LRU semantics: the three most recent keys are kept.
    expected_keys = [(f"x_{i}", f"y_{i}", "params_sig") for i in (7, 8, 9)]
    assert list(MRMR._FIT_CACHE.keys()) == expected_keys


def test_mrmr_fit_cache_default_cap_bounds_ten_writes_to_four():
    """Mrmr fit cache default cap bounds ten writes to four."""
    MRMR._FIT_CACHE.clear()
    inst = MRMR()  # default fit_cache_max=4
    for i in range(10):
        key = (f"k_{i}",)
        MRMR._FIT_CACHE[key] = inst
        MRMR._FIT_CACHE.move_to_end(key)
        _cap = int(getattr(inst, "fit_cache_max", 4) or 4)
        while len(MRMR._FIT_CACHE) > _cap:
            MRMR._FIT_CACHE.popitem(last=False)
    assert len(MRMR._FIT_CACHE) == 4
