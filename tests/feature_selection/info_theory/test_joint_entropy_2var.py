"""Byte-identity regression tests for ``joint_entropy_2var`` -- the FUSED 2-variable joint
ENTROPY fast path that replaces ``entropy(joint_freqs_2var(...))`` in the DCD pairwise-SU hot
loop (``_dcd_metrics.pair_su`` su branch).

``joint_entropy_2var(fd, ia, ib, nb_a, nb_b)`` MUST return the EXACT same float64 as
``entropy(joint_freqs_2var(fd, ia, ib, nb_a, nb_b))`` (and hence, transitively, as
``entropy(merge_vars(fd, [ia, ib], ...)[1])``), while skipping the intermediate normalized-freqs
array, the ``freqs[freqs > 0]`` mask, and entropy's ``log(freqs) * freqs`` temporary that the
two-call form allocates-then-discards. These tests pin the bit-identity contract so a future edit
to either kernel cannot silently drift the DCD redundancy score.
"""
import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    merge_vars,
    joint_freqs_2var,
    joint_entropy_2var,
    entropy,
)


def _ref_merge_entropy(fd, a, b, fn_arr, dtype=np.int32):
    pair_buf = np.array([a, b], dtype=np.int64)
    _, freqs_ab, _ = merge_vars(fd, pair_buf, None, fn_arr, dtype=dtype)
    return float(entropy(freqs_ab))


def _make(kind, n, k, nbins, seed=0):
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
def test_joint_entropy_2var_byte_identical(kind, n):
    """Fused joint entropy is EXACTLY (abs-diff 0.0) ``entropy(joint_freqs_2var(...))`` AND
    ``entropy(merge_vars(...)[1])`` across data shapes/distributions -- so the DCD SU score is
    bit-identical to the two-call form it replaces."""
    nbins = 10
    k = 40
    fd = _make(kind, n, k, nbins, seed=hash((kind, n)) % 2**31)
    fn_arr = np.full(k, nbins, dtype=np.int64)
    # warm
    joint_entropy_2var(fd, 0, 1, nbins, nbins)
    joint_freqs_2var(fd, 0, 1, nbins, nbins)
    max_vs_freqs = 0.0
    max_vs_merge = 0.0
    for a in range(0, 20):
        for b in range(a + 1, 25):
            two_call = float(entropy(joint_freqs_2var(fd, a, b, int(fn_arr[a]), int(fn_arr[b]))))
            fused = float(joint_entropy_2var(fd, a, b, int(fn_arr[a]), int(fn_arr[b])))
            merge_ref = _ref_merge_entropy(fd, a, b, fn_arr)
            max_vs_freqs = max(max_vs_freqs, abs(two_call - fused))
            max_vs_merge = max(max_vs_merge, abs(merge_ref - fused))
    assert max_vs_freqs == 0.0, f"fused != entropy(joint_freqs_2var): max diff {max_vs_freqs}"
    assert max_vs_merge == 0.0, f"fused != entropy(merge_vars): max diff {max_vs_merge}"


def test_joint_entropy_2var_unequal_nbins():
    """Columns with DIFFERENT bin counts: class id ``ca + cb*nb_a`` -> same entropy as the
    two-call form (which inherits merge_vars' encoding)."""
    rng = np.random.default_rng(7)
    n = 500
    fd = np.empty((n, 2), dtype=np.int32)
    fd[:, 0] = rng.integers(0, 4, size=n)
    fd[:, 1] = rng.integers(0, 7, size=n)
    fn_arr = np.array([4, 7], dtype=np.int64)
    two_call = float(entropy(joint_freqs_2var(fd, 0, 1, 4, 7)))
    fused = float(joint_entropy_2var(fd, 0, 1, 4, 7))
    merge_ref = _ref_merge_entropy(fd, 0, 1, fn_arr)
    assert abs(two_call - fused) == 0.0
    assert abs(merge_ref - fused) == 0.0


def test_joint_entropy_2var_constant_pair():
    """Both columns constant -> single nonzero bin -> entropy exactly 0.0."""
    n = 300
    fd = np.zeros((n, 2), dtype=np.int32)
    fd[:, 0] = 2
    fd[:, 1] = 5
    assert float(joint_entropy_2var(fd, 0, 1, 10, 10)) == 0.0
