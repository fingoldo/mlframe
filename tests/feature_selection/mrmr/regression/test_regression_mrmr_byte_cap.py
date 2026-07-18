"""Regression sensor for A5 P1 #6.

Pre-fix MRMR._FIT_CACHE had only an ENTRY-count cap (fit_cache_max=4). On a 1k-feature suite each cached instance can pin 100+ MB of _selectors_ / _engineered_features_ state, so 4 entries x 100 MB = 400 MB pinned for an inactive R&D suite. The fix adds an aggregate byte cap (fit_cache_max_mb / env MLFRAME_MRMR_FIT_CACHE_MAX_MB) that LRU-evicts when crossed.

We exercise the helpers directly rather than driving a 100+ MB MRMR fit to keep the sensor fast (<1s) -- the count/byte eviction policy is purely a wrapping concern around the helpers.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl import (
    _mrmr_instance_state_size_bytes,
    _mrmr_cache_bytes_total,
)


class _ToyMRMRInstance:
    """Pseudo-MRMR instance carrying the state attributes the byte estimator inspects."""

    def __init__(self, n_features: int = 2000):
        self.mi_scores_ = np.zeros(n_features, dtype=np.float64)
        self._selectors_ = {f"col_{i}": np.zeros(64, dtype=np.float64) for i in range(50)}
        self._engineered_features_ = np.zeros((n_features, 8), dtype=np.float64)
        self.ranking_ = np.arange(n_features, dtype=np.int64)


def test_byte_estimator_counts_dominant_arrays():
    """Byte estimator counts dominant arrays."""
    inst = _ToyMRMRInstance(n_features=2000)
    n = _mrmr_instance_state_size_bytes(inst)
    # mi_scores_ (16k) + _engineered_features_ (128k) + ranking_ (16k) + 50 selector arrays x 512B
    # gives a lower bound around 175 KB. Use a conservative floor.
    assert n >= 16_000 + 128_000 + 16_000, f"byte estimator must walk the major state attributes; got {n}"


def test_byte_estimator_handles_attributes_missing():
    """Byte estimator handles attributes missing."""
    class _Empty:
        """Groups tests covering Empty."""
        pass

    assert _mrmr_instance_state_size_bytes(_Empty()) == 0


def test_cache_bytes_total_aggregates_across_entries():
    """Cache bytes total aggregates across entries."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    _saved = dict(MRMR._FIT_CACHE)
    try:
        MRMR._FIT_CACHE.clear()
        MRMR._FIT_CACHE["k1"] = _ToyMRMRInstance(n_features=500)
        MRMR._FIT_CACHE["k2"] = _ToyMRMRInstance(n_features=500)
        total = _mrmr_cache_bytes_total()
        single = _mrmr_instance_state_size_bytes(MRMR._FIT_CACHE["k1"])
        assert total >= 2 * single * 0.9, f"aggregate must approximate sum across entries; got {total} vs 2*{single}"
    finally:
        MRMR._FIT_CACHE.clear()
        MRMR._FIT_CACHE.update(_saved)
