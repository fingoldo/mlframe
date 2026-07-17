"""Regression sensor for A5 Low #13.

DiscoveryCache(max_entries=None, max_size_mb=None) previously only warned via warnings.warn at construction. CI runs commonly suppress warnings, so the cache filled the disk silently. The fix promotes the unbounded-cap construction to a hard ValueError so the operator must opt in explicitly (pass float('inf') or 10**9 if unbounded growth is genuinely desired).
"""

from __future__ import annotations

import pytest

from mlframe.training.composite.cache import DiscoveryCache


def test_both_caps_none_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="grow without bound"):
        DiscoveryCache(str(tmp_path), max_entries=None, max_size_mb=None)


def test_one_explicit_cap_constructs_cleanly(tmp_path):
    c = DiscoveryCache(str(tmp_path), max_entries=128, max_size_mb=None)
    assert c.max_entries == 128


def test_explicit_unbounded_construction_allowed(tmp_path):
    c = DiscoveryCache(str(tmp_path), max_entries=10**9, max_size_mb=None)
    assert c.max_entries == 10**9
