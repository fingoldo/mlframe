"""``_joblib_safe.fit_constant_memmap``: dump a fit-constant array to a read-only memmap ONCE per
process per content, so joblib ships it to loky workers by filename instead of re-dumping the whole
buffer on every ``Parallel(...)`` invocation (wellbore-100k profile: 45 dumps / ~315s of
_pickle.dumps for the same ~hundreds-of-MB ``data`` matrix)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._joblib_safe import _FIT_MEMMAP_CACHE, fit_constant_memmap


def test_same_content_returns_same_memmap_instance():
    """Same content returns same memmap instance."""
    a = np.random.default_rng(0).standard_normal((500, 20))
    before = len(_FIT_MEMMAP_CACHE)
    m1 = fit_constant_memmap(a)
    m2 = fit_constant_memmap(a.copy())  # same CONTENT, different object -> same cache entry
    assert isinstance(m1, np.memmap)
    assert m1 is m2
    assert len(_FIT_MEMMAP_CACHE) == before + 1
    np.testing.assert_array_equal(np.asarray(m1), a)


def test_different_content_gets_own_entry_and_readonly_view():
    """Different content gets own entry and readonly view."""
    a = np.random.default_rng(1).standard_normal((300, 7))
    b = a + 1.0
    ma, mb = fit_constant_memmap(a), fit_constant_memmap(b)
    assert ma is not mb
    np.testing.assert_array_equal(np.asarray(mb), b)
    # Read-only: a worker bug must never be able to corrupt the shared file.
    try:
        ma[0, 0] = 123.0
        raised = False
    except (ValueError, OSError):
        raised = True
    assert raised, "memmap view must be read-only"


def test_existing_memmap_passes_through():
    """Existing memmap passes through."""
    a = np.random.default_rng(2).standard_normal((100, 3))
    m = fit_constant_memmap(a)
    assert fit_constant_memmap(m) is m
