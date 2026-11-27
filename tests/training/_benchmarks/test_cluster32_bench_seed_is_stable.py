"""Cluster 32 regression: the threshold benchmark seeds each cell from a process-stable hash of its scenario name (audit 2026-11-20, finding 32-F17).

``hash(name)`` of a string is salted per process, so two runs drew different validation and test samples. The seed now uses ``zlib.crc32``.
"""

import builtins
import zlib

import numpy as np
import pytest

from mlframe.training._benchmarks import bench_threshold_objective as bench


_REAL_DEFAULT_RNG = np.random.default_rng


class _Stop(Exception):
    """Stops the benchmark once the seeds of its first scenario are recorded."""


def _seeds_of_the_first_scenario(monkeypatch, salted_hash):
    """The integer seeds ``run`` hands to ``default_rng`` for its first scenario, under a given built-in ``hash``."""
    seeds = []

    def recorder(seed=None):
        """Record the seed, and stop after the oracle draw plus the first cell."""
        seeds.append(seed)
        if len(seeds) > 2:
            raise _Stop
        return _REAL_DEFAULT_RNG(seed)

    monkeypatch.setattr(builtins, "hash", salted_hash)
    monkeypatch.setattr(np.random, "default_rng", recorder)
    with pytest.raises(_Stop):
        bench.run()
    return seeds


def test_cell_seeds_do_not_depend_on_the_process_hash_salt(monkeypatch):
    """Two different string-hash salts give the same cell seed, and it equals the crc32 form (32-F17)."""
    first = _seeds_of_the_first_scenario(monkeypatch, lambda obj: 12345)
    second = _seeds_of_the_first_scenario(monkeypatch, lambda obj: 987654321)
    assert first == second
    name = bench.SCENARIOS[0][0]
    assert first[1] == 1000 * bench.SEEDS[0] + zlib.crc32(name.encode()) % 997
