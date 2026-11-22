"""Cluster 39 regression: persistence on with no DiskKey is reported, not silent (audit 2026-11-20, finding 39-F20)."""

import logging

from mlframe.training.feature_handling.cache import FeatureCache
from mlframe.training.feature_handling.config import CacheConfig


def test_disk_tier_without_key_logs_warning(tmp_path, caplog):
    """persistence='auto' with no disk_key must warn that the disk tier is unused (39-F20)."""
    cfg = CacheConfig(persistence="auto", dir=str(tmp_path))
    cache = FeatureCache(cache_cfg=cfg)
    in_mem_key = ("cluster39", "key-without-disk")
    with caplog.at_level(logging.WARNING):
        value = cache.get_or_compute(in_mem_key, lambda: 42)
    assert value == 42
    assert any("disk tier is unused" in r.getMessage() for r in caplog.records), "no warning when the disk tier was skipped"
