"""Pairs and triples whose joint histogram is too large for the parallel region run one at a time after it, with unchanged values."""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.info_theory._batch_kernels import (
    PARALLEL_JOINT_CELLS,
    batch_pair_mi_perm_batched,
    batch_pair_mi_prange,
    batch_triple_mi_perm_batched,
    batch_triple_mi_prange,
)


def _plugin_mi(codes: np.ndarray, y: np.ndarray) -> float:
    """Reference plug-in MI (nats) of an integer code vector against ``y`` from a dense crosstab over the occupied codes."""
    _, xi = np.unique(codes, return_inverse=True)
    n = len(y)
    joint = np.zeros((xi.max() + 1, int(y.max()) + 1))
    np.add.at(joint, (xi, y), 1.0)
    pxy = joint / n
    px, py = pxy.sum(axis=1, keepdims=True), pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())


def _fixture(big: int, n: int = 150, seed: int = 0):
    """Two low-cardinality columns, three columns of cardinality ``big``, a binary target and its frequencies."""
    rng = np.random.default_rng(seed)
    nbins = np.array([3, 3, big, big, big], dtype=np.int64)
    data = np.column_stack([rng.integers(0, int(b), n) for b in nbins]).astype(np.int32)
    y = ((data[:, 0] + rng.integers(0, 2, n)) % 2).astype(np.int32)
    freqs_y = np.bincount(y, minlength=2).astype(np.float64) / n
    return data, nbins, y, freqs_y


def test_pairs_above_and_below_the_parallel_limit_match_the_reference():
    """A pair over the cell limit (deferred, serial) and pairs under it (parallel) both equal the dense reference MI."""
    big = 1_500
    assert big * big * 2 > PARALLEL_JOINT_CELLS
    data, nbins, y, freqs_y = _fixture(big)
    pa = np.array([0, 2, 0, 3], dtype=np.int64)
    pb = np.array([1, 3, 2, 4], dtype=np.int64)
    got = batch_pair_mi_prange(data, pa, pb, nbins, y, freqs_y)
    for k in range(len(pa)):
        merged = data[:, pa[k]].astype(np.int64) * nbins[pb[k]] + data[:, pb[k]]
        assert abs(got[k] - _plugin_mi(merged, y)) < 1e-12
    alone = batch_pair_mi_prange(data, pa[1:2], pb[1:2], nbins, y, freqs_y)
    assert got[1] == alone[0]


def test_pair_perm_batched_defers_large_pairs_without_changing_values():
    """The permutation-batched pair kernel equals the per-shuffle kernel for every pair, including the deferred one."""
    big = 1_500
    data, nbins, y, freqs_y = _fixture(big)
    rng = np.random.default_rng(3)
    perms = np.stack([rng.permutation(y) for _ in range(3)]).astype(np.int32)
    pa = np.array([0, 2, 0], dtype=np.int64)
    pb = np.array([1, 3, 2], dtype=np.int64)
    got = batch_pair_mi_perm_batched(data, pa, pb, nbins, perms, freqs_y)
    for k in range(3):
        np.testing.assert_array_equal(got[k], batch_pair_mi_prange(data, pa, pb, nbins, perms[k], freqs_y))


def test_triples_above_the_parallel_limit_match_the_reference():
    """A triple whose raw cardinality exceeds the limit (deferred) equals the dense reference joint MI, as does a small triple beside it."""
    big = 170
    assert big**3 > PARALLEL_JOINT_CELLS
    data, nbins, y, freqs_y = _fixture(big)
    ta = np.array([0, 2], dtype=np.int64)
    tb = np.array([1, 3], dtype=np.int64)
    tc = np.array([0, 4], dtype=np.int64)
    got = batch_triple_mi_prange(data, ta, tb, tc, nbins, y, freqs_y)
    for k in range(2):
        a, b, c = ta[k], tb[k], tc[k]
        merged = (data[:, a].astype(np.int64) * nbins[b] + data[:, b]) * nbins[c] + data[:, c]
        assert abs(got[k] - _plugin_mi(merged, y)) < 1e-12


def test_triple_perm_batched_defers_large_triples_without_changing_values():
    """The permutation-batched triple kernel equals the per-shuffle kernel, including for the deferred triple."""
    big = 170
    data, nbins, y, freqs_y = _fixture(big)
    rng = np.random.default_rng(4)
    perms = np.stack([rng.permutation(y) for _ in range(2)]).astype(np.int32)
    ta, tb, tc = np.array([0, 2], dtype=np.int64), np.array([1, 3], dtype=np.int64), np.array([0, 4], dtype=np.int64)
    got = batch_triple_mi_perm_batched(data, ta, tb, tc, nbins, perms, freqs_y)
    for k in range(2):
        np.testing.assert_array_equal(got[k], batch_triple_mi_prange(data, ta, tb, tc, nbins, perms[k], freqs_y))
