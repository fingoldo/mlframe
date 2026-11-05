"""The MRMR fit cache has one lock, and every holder gets that same object.

Two module-level locks guarded the same ``MRMR._FIT_CACHE``, so a holder of one did not exclude a holder of the other.
"""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fit_impl import _fit_impl_core, _fit_impl_core_parts


def test_fit_cache_lock_is_a_single_object_across_modules():
    """The core module and the parts module refer to one RLock."""
    assert _fit_impl_core._MRMR_FIT_CACHE_LOCK is _fit_impl_core_parts._MRMR_FIT_CACHE_LOCK
