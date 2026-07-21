"""``_joblib_safe.fit_constant_memmap``: dump a fit-constant array to a read-only memmap ONCE per
process per content, so joblib ships it to loky workers by filename instead of re-dumping the whole
buffer on every ``Parallel(...)`` invocation (wellbore-100k profile: 45 dumps / ~315s of
_pickle.dumps for the same ~hundreds-of-MB ``data`` matrix)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._joblib_safe import _FIT_MEMMAP_CACHE, _fit_constant_key, fit_constant_memmap


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


def _pre_fix_sampled_key(arr):
    """The pre-B-21 sampled key (first/last 64KB + coarse stride) -- reimplemented here ONLY as a
    reference to demonstrate the exact collision the fix closes; not exercised by production code."""
    a = np.ascontiguousarray(arr)
    raw = a.view(np.uint8).ravel()
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(raw[:65536].tobytes())
    if raw.size > 65536:
        h.update(raw[-65536:].tobytes())
        h.update(raw[:: max(1, raw.size // 65536)].tobytes())
    return (a.shape, str(a.dtype), h.hexdigest())


def test_key_differs_when_only_untested_middle_byte_differs():
    """mrmr_audit_2026-07-20 B-21: two same-shape/dtype arrays that agree at the FIRST 64KB, the LAST
    64KB, and every coarse-stride sample point the pre-fix key inspected, but differ at ONE byte
    squarely in the untested middle, must get DIFFERENT cache keys under the fixed (full-buffer) key.
    Constructed to PROVE the pre-fix sampled key genuinely collides on this exact byte layout (not a
    hypothetical) -- ``fit_constant_memmap`` would have silently returned the WRONG cached array."""
    n_rows = 20_000
    n_cols = 20
    rng = np.random.default_rng(3)
    a = rng.standard_normal((n_rows, n_cols))
    b = a.copy()

    a_c = np.ascontiguousarray(a)
    raw_size = a_c.nbytes
    stride = max(1, raw_size // 65536)
    # A byte offset outside the first/last 64KB windows and NOT a multiple of the old stride --
    # verified below, not assumed.
    mid_offset = raw_size // 2
    assert mid_offset > 65536 and mid_offset < raw_size - 65536
    assert mid_offset % stride != 0, "fixture must land off the old sampled stride"
    b.view(np.uint8).ravel()[mid_offset] ^= 0xFF

    # Prove the OLD algorithm genuinely collides on this construction (not just "differs somewhere").
    assert _pre_fix_sampled_key(a) == _pre_fix_sampled_key(b), (
        "fixture sanity check failed: the pre-fix sampled key must collide on this construction " "for the regression below to demonstrate anything"
    )

    key_a = _fit_constant_key(a)
    key_b = _fit_constant_key(b)
    assert key_a != key_b, "B-21 regression: differing-content arrays produced the SAME cache key -- collision risk"

    ma = fit_constant_memmap(a)
    mb = fit_constant_memmap(b)
    assert ma is not mb, "B-21 regression: fit_constant_memmap returned the SAME cached memmap for different content"
    np.testing.assert_array_equal(np.asarray(ma), a)
    np.testing.assert_array_equal(np.asarray(mb), b)
