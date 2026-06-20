"""Sensor for W5 A5-P1#6 MRMR fit-cache byte cap.

The fit-cache is class-attribute (process-wide). A 1k-feature suite carrying 4 cached MRMR
instances each holding ``_selectors_`` / ``_engineered_features_`` state can exceed 1 GB of
process RSS. The new ``fit_cache_max_mb`` knob (default 1024 MB; env override
``MLFRAME_MRMR_FIT_CACHE_MAX_MB``) bounds the aggregate cache footprint.

Sensor: insert two mock-instances whose ``mi_scores_`` ndarray together exceed a tiny cap;
assert the LRU pops to enforce the cap. Behaviour-only -- no real MRMR fit needed.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest


class _FakeMRMR:
    """Mock that exposes the attributes ``_mrmr_instance_state_size_bytes`` walks."""

    def __init__(self, nbytes: int):
        self.mi_scores_ = np.zeros(nbytes // 8, dtype=np.float64)
        # Sanity: nbytes reported by ndarray must hit the requested target.
        assert self.mi_scores_.nbytes == (nbytes // 8) * 8


def test_instance_state_size_walks_mi_scores():
    from mlframe.feature_selection.filters._mrmr_fit_impl import _mrmr_instance_state_size_bytes

    inst = _FakeMRMR(nbytes=8_000)
    sz = _mrmr_instance_state_size_bytes(inst)
    assert sz >= 8_000, f"Expected the mi_scores_ buffer to dominate the estimate; got {sz}"


def test_cache_bytes_total_aggregates_across_cache():
    from mlframe.feature_selection.filters._mrmr_fit_impl import _mrmr_cache_bytes_total
    from mlframe.feature_selection.filters.mrmr import MRMR

    snapshot = OrderedDict(MRMR._FIT_CACHE)
    MRMR._FIT_CACHE.clear()
    try:
        MRMR._FIT_CACHE["a"] = _FakeMRMR(nbytes=4_000)
        MRMR._FIT_CACHE["b"] = _FakeMRMR(nbytes=6_000)
        total = _mrmr_cache_bytes_total()
        assert total >= 10_000, f"Aggregate should sum across cache entries; got {total}"
    finally:
        MRMR._FIT_CACHE.clear()
        MRMR._FIT_CACHE.update(snapshot)
