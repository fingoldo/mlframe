"""Byte-identity regression tests for ``joint_freqs_2var`` -- the 2-variable joint-frequency
fast path that replaces ``merge_vars`` in the DCD pairwise-SU hot loop.

``joint_freqs_2var(fd, ia, ib, nb_a, nb_b)`` MUST return the EXACT same normalized nonzero
frequencies as ``merge_vars(fd, [ia, ib], None, factors_nbins, dtype)[1]`` (and hence drive
a bit-identical ``entropy``), while skipping the per-sample ``final_classes`` build + remap
that ``merge_vars`` does and the SU path discards. These tests pin the bit-identity contract
so a future edit to either kernel cannot silently drift the DCD redundancy score.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    merge_vars,
    joint_freqs_2var,
    entropy,
)


def _ref_merge_freqs(fd, a, b, fn_arr, dtype=np.int32):
    """Reference joint frequencies for columns (a, b) computed via the legacy merge_vars path."""
    pair_buf = np.array([a, b], dtype=np.int64)
    _, freqs_ab, _ = merge_vars(fd, pair_buf, None, fn_arr, dtype=dtype)
    return freqs_ab


def _make(kind, n, k, nbins, seed=0):
    """Build an (n, k) integer factor matrix in one of four occupancy regimes (uniform/sparse/skew/const) for joint-frequency fuzzing."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.integers(0, nbins, size=(n, k)).astype(np.int32)
    if kind == "sparse":  # only a few bins occupied
        return rng.integers(0, 3, size=(n, k)).astype(np.int32)
    if kind == "skew":  # heavy mass on bin 0
        d = rng.integers(0, nbins, size=(n, k))
        d[d > 2] = 0
        return d.astype(np.int32)
    if kind == "const":  # some constant columns (single occupied bin)
        d = rng.integers(0, nbins, size=(n, k)).astype(np.int32)
        d[:, ::3] = 5
        return d
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["uniform", "sparse", "skew", "const"])
@pytest.mark.parametrize("n", [37, 600, 2407])
def test_joint_freqs_2var_byte_identical_to_merge_vars(kind, n):
    """Normalized nonzero joint freqs are EXACTLY (max-abs-diff 0.0) ``merge_vars``'s 2-var
    output across data shapes/distributions -- so the downstream ``entropy`` is bit-identical."""
    nbins = 10
    k = 40
    fd = _make(kind, n, k, nbins, seed=hash((kind, n)) % 2**31)
    fn_arr = np.full(k, nbins, dtype=np.int64)
    # warm
    joint_freqs_2var(fd, 0, 1, nbins, nbins)
    max_freq_diff = 0.0
    max_entropy_diff = 0.0
    for a in range(0, 20):
        for b in range(a + 1, 25):
            ref = _ref_merge_freqs(fd, a, b, fn_arr)
            got = joint_freqs_2var(fd, a, b, int(fn_arr[a]), int(fn_arr[b]))
            # SAME length, SAME order, SAME values -> identical pruned histogram
            assert ref.shape == got.shape, f"shape mismatch a={a} b={b}: {ref.shape} vs {got.shape}"
            max_freq_diff = max(max_freq_diff, float(np.max(np.abs(ref - got))))
            max_entropy_diff = max(max_entropy_diff, abs(float(entropy(ref)) - float(entropy(got))))
    assert max_freq_diff == 0.0, f"freqs not byte-identical: max diff {max_freq_diff}"
    assert max_entropy_diff == 0.0, f"entropy not byte-identical: max diff {max_entropy_diff}"


def test_joint_freqs_2var_unequal_nbins():
    """Columns with DIFFERENT bin counts: class id ``ca + cb*nb_a`` must match ``merge_vars``."""
    rng = np.random.default_rng(7)
    n = 500
    # col 0: 4 bins, col 1: 7 bins
    fd = np.empty((n, 2), dtype=np.int32)
    fd[:, 0] = rng.integers(0, 4, size=n)
    fd[:, 1] = rng.integers(0, 7, size=n)
    fn_arr = np.array([4, 7], dtype=np.int64)
    ref = _ref_merge_freqs(fd, 0, 1, fn_arr)
    got = joint_freqs_2var(fd, 0, 1, 4, 7)
    assert ref.shape == got.shape
    assert np.max(np.abs(ref - got)) == 0.0
    assert abs(float(entropy(ref)) - float(entropy(got))) == 0.0


def test_joint_freqs_2var_constant_pair():
    """Both columns constant -> single nonzero bin, freq 1.0, entropy 0.0."""
    n = 300
    fd = np.zeros((n, 2), dtype=np.int32)
    fd[:, 0] = 2
    fd[:, 1] = 5
    got = joint_freqs_2var(fd, 0, 1, 10, 10)
    assert got.shape == (1,)
    assert got[0] == 1.0
    assert float(entropy(got)) == 0.0
