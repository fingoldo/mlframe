"""Cluster 32 regression: the threshold benchmark seeds from a process-stable hash (audit 2026-11-20, finding 32-F17)."""

from pathlib import Path

import mlframe.training._benchmarks.bench_threshold_objective as bench


def test_seed_uses_crc32_not_salted_hash():
    """The per-scenario seed reads zlib.crc32; the salted builtin hash() is not used (32-F17)."""
    src = Path(bench.__file__).read_text(encoding="utf-8")
    assert "zlib.crc32(name.encode())" in src
    assert "hash(name)" not in src
