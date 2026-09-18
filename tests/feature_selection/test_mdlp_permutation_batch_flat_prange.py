"""The batched MDLP permutation kernel must parallelise a BFS level that holds a single node.

``_mdlp_permutation_batch_njit`` used to ``prange`` over NODES only, so the root level (one node, and the
largest one) ran all its permutation scans on one thread; the flat prange over (node, permutation) pairs
spreads them. Accept flags must stay bit-identical to the per-node layout (same per-draw seeds).
"""

from __future__ import annotations

import numba
import numpy as np
import pytest

from mlframe.feature_selection.filters._benchmarks.bench_mdlp_permutation_batch_flat_prange import _batch_prange_over_nodes, _level
from mlframe.feature_selection.filters._mdlp_validated_split import _mdlp_permutation_batch_njit
from tests._perf_paired import assert_paired_speedup


@pytest.mark.parametrize("n_nodes", [1, 2, 7, 32])
def test_flat_prange_accepts_match_per_node_layout(n_nodes):
    x, y, sizes, gains, ncls, seeds = _level(n_nodes, 4000, 6, np.random.default_rng(n_nodes))
    # Mix of gains around the null so both verdicts occur.
    gains = np.linspace(0.0, 0.02, n_nodes)
    call = (x, y, sizes, gains, ncls, 5, 30, seeds, 0.05)
    np.testing.assert_array_equal(_mdlp_permutation_batch_njit(*call), _batch_prange_over_nodes(*call))


@pytest.mark.skipif(numba.get_num_threads() < 2, reason="needs >= 2 numba threads to parallelise anything")
def test_single_node_level_uses_more_than_one_thread():
    x, y, sizes, gains, ncls, seeds = _level(1, 20_000, 10, np.random.default_rng(0))
    call = (x, y, sizes, gains, ncls, 5, 30, seeds, 0.05)
    assert_paired_speedup(
        lambda: _batch_prange_over_nodes(*call),
        lambda: _mdlp_permutation_batch_njit(*call),
        base_ratio=1.3,
        n_trials=5,
        what="the flat (node, permutation) prange on a one-node BFS level",
    )
