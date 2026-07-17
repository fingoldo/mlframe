"""Unit gate for the parallel-njit maxT shuffle generator (large-n lever, 2026-06-24).

``_permutation_null._gen_target_shuffles_par_njit`` thread-parallelises the maxT floor's per-shuffle
Fisher-Yates across permutations -- the dominant LARGE-N cost of the order-1 floor (~88% of its wall at
n>=600k; floor-direct A/B 3.5s -> 1.3s, 2.7x). Each row is seeded ONLY by ``base_seed + k`` (not thread
id), so the matrix is reproducible, thread-count-independent, and every row is a true uniform permutation.

The dispatcher (``_generate_target_shuffles``) routes between the legacy sequential numpy stream and the
parallel njit by a per-host KTC crossover; ``MLFRAME_FDR_SHUFFLEGEN`` forces a backend (A/B + escape hatch).
The numba stream is a DIFFERENT-but-valid uniform draw sequence, so the floor it yields is statistically
equivalent (validated selection-identical on the canonical biz_value suite), and remains DETERMINISTIC for a
fixed seed (the contract ``test_pooled_gain_floor_perms_prange_identity.test_public_floor_*`` already pins).
"""

from __future__ import annotations

import os

import numpy as np

from mlframe.feature_selection.filters._permutation_null import (
    _gen_target_shuffles_par_njit,
    pooled_permutation_null_gain_floor,
)


def test_parallel_gen_rows_are_uniform_permutations():
    """Parallel gen rows are uniform permutations."""
    y = np.random.default_rng(0).integers(0, 10, size=5000).astype(np.int32)
    out = _gen_target_shuffles_par_njit(y, 32, np.int64(123))
    assert out.shape == (32, 5000)
    sorted_y = np.sort(y)
    for k in range(out.shape[0]):
        assert np.array_equal(np.sort(out[k]), sorted_y), f"row {k} is not a permutation of y"


def test_parallel_gen_reproducible_and_thread_independent():
    """Parallel gen reproducible and thread independent."""
    import numba

    y = np.random.default_rng(1).integers(0, 8, size=4000).astype(np.int32)
    a = _gen_target_shuffles_par_njit(y, 40, np.int64(777))
    b = _gen_target_shuffles_par_njit(y, 40, np.int64(777))
    assert np.array_equal(a, b), "same seed must give an identical shuffle matrix"
    prev = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        c = _gen_target_shuffles_par_njit(y, 40, np.int64(777))
    finally:
        numba.set_num_threads(prev)
    assert np.array_equal(a, c), "result must be independent of the numba thread count"


def test_dispatcher_floor_deterministic_under_both_backends():
    """Dispatcher floor deterministic under both backends."""
    rng = np.random.default_rng(7)
    n, p, nbins = 3000, 24, 10
    cols = [rng.integers(0, nbins, size=n).astype(np.int32) for _ in range(p)]
    y = rng.integers(0, nbins, size=n).astype(np.int32)
    data = np.column_stack([*cols, y]).astype(np.int32)
    nb = np.array([nbins] * (p + 1), dtype=np.int64)
    cand = np.arange(p, dtype=np.int64)
    saved = os.environ.get("MLFRAME_FDR_SHUFFLEGEN")
    try:
        for backend in ("numpy", "numba"):
            os.environ["MLFRAME_FDR_SHUFFLEGEN"] = backend
            f1 = pooled_permutation_null_gain_floor(data, nb, cand, p, n_permutations=64, random_seed=42)
            f2 = pooled_permutation_null_gain_floor(data, nb, cand, p, n_permutations=64, random_seed=42)
            assert f1 == f2, f"{backend}: same seed must yield an identical floor"
            assert f1 >= 0.0
    finally:
        if saved is None:
            os.environ.pop("MLFRAME_FDR_SHUFFLEGEN", None)
        else:
            os.environ["MLFRAME_FDR_SHUFFLEGEN"] = saved
