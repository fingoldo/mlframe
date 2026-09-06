"""The same trend line was fitted once per backend from the same frozen spec.

``plot_outputs`` renders BOTH backends from one ``FigureSpec``, and ``robust_fit_endpoints`` is a pure
function of ``(x, y, method)`` called from inside each renderer -- so every trend panel paid for two
identical Theil-Sen fits. The cache is keyed on array IDENTITY (hashing two million-row arrays to save a
0.25s fit would cost more than the fit), which is only sound because each entry holds weak references and
a hit is confirmed against them: a freed array's id is reused, and an unconfirmed hit would draw one
panel's trend line across another panel's data.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import mlframe.reporting.renderers._trend as trend_mod
from mlframe.reporting.renderers._trend import robust_fit_endpoints


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts from an empty cache; a leaked entry would make the next test pass for the wrong reason."""
    trend_mod._FIT_CACHE.clear()
    yield
    trend_mod._FIT_CACHE.clear()


def _cloud(n: int = 4_000, seed: int = 0):
    """A noisy linear cloud with a slope worth fitting."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, 2.0 * x + rng.normal(size=n)


def test_the_second_call_does_not_refit():
    """One pair of arrays, one fit -- counted, not inferred from timing."""
    x, y = _cloud()
    calls = {"n": 0}
    real = trend_mod._fit_endpoints_uncached

    def counted(*a, **kw):
        """Count the real fits so the cache hit is observable rather than inferred from timing."""
        calls["n"] += 1
        return real(*a, **kw)

    trend_mod._fit_endpoints_uncached = counted
    try:
        first = robust_fit_endpoints(x, y, "theil-sen")
        second = robust_fit_endpoints(x, y, "theil-sen")
    finally:
        trend_mod._fit_endpoints_uncached = real
    assert calls["n"] == 1, f"the fit ran {calls['n']} times for one pair of arrays"
    assert first == second


def test_different_data_is_not_served_from_the_cache():
    """Guard: memoising must not collapse two different clouds onto one line."""
    x, y = _cloud()
    other_x, other_y = _cloud(seed=7)
    a = robust_fit_endpoints(x, y, "theil-sen")
    b = robust_fit_endpoints(other_x, other_y, "theil-sen")
    assert a != b, "two different clouds got the same fit"


def test_a_stale_entry_under_a_live_id_is_not_served():
    """The whole reason the entries hold weak references: CPython reuses the id of a freed array.

    Waiting for a real id collision makes a test that usually skips, so the collision is constructed: an
    entry is planted under THIS array's key carrying a fit of different data and references that no longer
    resolve to it. A cache that trusted the id alone would draw that other panel's trend line here.
    """
    x, y = _cloud()
    other = robust_fit_endpoints(*_cloud(seed=7), "theil-sen")
    honest = robust_fit_endpoints(x, y, "theil-sen")
    assert other != honest, "the fixture cannot detect a mis-serve: both clouds fit the same line"

    trend_mod._FIT_CACHE.clear()
    dead_x, dead_y = _cloud(seed=99)
    dead_refs = (weakref.ref(dead_x), weakref.ref(dead_y))
    del dead_x, dead_y
    gc.collect()
    trend_mod._FIT_CACHE[(id(x), id(y), "theil-sen")] = (dead_refs[0], dead_refs[1], other)

    assert robust_fit_endpoints(x, y, "theil-sen") == honest, "a stale entry under a recycled id was served as this array's fit"


def test_an_undefined_fit_is_remembered_too():
    """``None`` is an answer; recomputing it on every render is the same waste as recomputing a real fit."""
    x = np.zeros(50)
    y = np.arange(50.0)
    calls = {"n": 0}
    real = trend_mod._fit_endpoints_uncached

    def counted(*a, **kw):
        """Count the real fits, so a cached ``None`` is distinguishable from a miss that returns ``None``."""
        calls["n"] += 1
        return real(*a, **kw)

    trend_mod._fit_endpoints_uncached = counted
    try:
        assert robust_fit_endpoints(x, y, "theil-sen") is None
        assert robust_fit_endpoints(x, y, "theil-sen") is None
    finally:
        trend_mod._fit_endpoints_uncached = real
    assert calls["n"] == 1, "an undefined fit was recomputed"


def test_the_cache_is_bounded():
    """It holds weak references, but the dict itself must not grow without limit across a long report."""
    for seed in range(trend_mod._FIT_CACHE_MAX * 3):
        x, y = _cloud(n=200, seed=seed)
        robust_fit_endpoints(x, y, "theil-sen")
    assert len(trend_mod._FIT_CACHE) <= trend_mod._FIT_CACHE_MAX


def test_both_backends_fit_a_shared_spec_once():
    """The defect as the report meets it: one spec, two renders, two identical fits."""
    import matplotlib.pyplot as plt

    from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec

    x, y = _cloud()
    spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=y, title="pred vs actual", trend_line="theil-sen"),),), figsize=(6.0, 4.0))
    calls = {"n": 0}
    real = trend_mod._fit_endpoints_uncached

    def counted(*a, **kw):
        """Count fits across both renderers."""
        calls["n"] += 1
        return real(*a, **kw)

    trend_mod._fit_endpoints_uncached = counted
    try:
        fig = MatplotlibRenderer().render(spec)
        plt.close(fig)
        PlotlyRenderer().render(spec)
    finally:
        trend_mod._fit_endpoints_uncached = real
    assert calls["n"] == 1, f"the two backends fitted the same panel {calls['n']} times"
