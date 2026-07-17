"""DiscoveryCache must have non-None default caps so R&D runs don't grow unboundedly.

Pre-fix the defaults were (None, None) which silently let the cache balloon to many
GB on long sessions, surfacing only as a one-time WARN at construction. New defaults
1000 entries / 2000 MB protect operators by default; passing ``None`` is still
honored for explicit opt-out.
"""

from __future__ import annotations


import pytest


@pytest.mark.fast
def test_default_caps_are_non_none(tmp_path):
    from mlframe.training.composite.cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path))
    assert cache.max_entries is not None and cache.max_entries > 0
    assert cache.max_size_mb is not None and cache.max_size_mb > 0


@pytest.mark.fast
def test_explicit_none_pair_raises(tmp_path):
    """Both caps None is an unauditable footgun. Per audit A5 Low #13 the constructor now refuses
    so the operator must explicitly pass a large finite cap (or float('inf') / 10**9) to opt into unbounded growth."""
    from mlframe.training.composite.cache import DiscoveryCache

    with pytest.raises(ValueError, match="grow without bound"):
        DiscoveryCache(str(tmp_path), max_entries=None, max_size_mb=None)


@pytest.mark.fast
def test_explicit_one_none_other_set_works(tmp_path):
    """Disabling ONE cap is fine -- the other cap still gates eviction."""
    from mlframe.training.composite.cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path), max_entries=None, max_size_mb=2000.0)
    assert cache.max_entries is None
    assert cache.max_size_mb == 2000.0


@pytest.mark.fast
def test_partial_override(tmp_path):
    from mlframe.training.composite.cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path), max_entries=50)
    assert cache.max_entries == 50
    assert cache.max_size_mb is not None  # default still applied


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "--no-cov", "-x", "-s", "--tb=short"]))
