"""Cluster 32 regression: benchmark model names are unique across scenario, mode, val size and seed (audit 2026-11-20, finding 32-F25)."""

import itertools

from mlframe.training._benchmarks import bench_val_size_earlystop as bench


def test_model_names_are_unique_across_all_cells():
    """Every (scenario, regression, val_size, seed) cell gets its own model name in the shared workdir (32-F25)."""
    cells = itertools.product(["s1", "s2"], [False, True], [0.1, 0.2], [0, 1])
    names = [bench._model_name(sc, reg, vs, seed) for sc, reg, vs, seed in cells]
    assert len(set(names)) == len(names)
