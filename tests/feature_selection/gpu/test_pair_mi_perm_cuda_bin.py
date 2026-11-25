"""The outcome-code kernel of the pair maxT floor (binary target) gives the MI of the general kernel to rounding, groups the pairs by their permutations per pass and leaves other targets to the general kernel."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _pair_mi_perm_cuda as P
from mlframe.feature_selection.filters import _pair_mi_perm_cuda_bin as B

pytestmark = pytest.mark.gpu


def _case(n: int, bins: "list[int]", n_pairs: int, k: int, ky: int = 2, seed: int = 0):
    """Byte codes of columns with the given bin counts, random pairs, ``k`` permutations of a ``ky``-class target and its frequencies."""
    rng = np.random.default_rng(seed)
    nb = np.asarray(bins, dtype=np.int64)
    x = np.stack([rng.integers(0, b, n) for b in nb], axis=1).astype(np.uint8)
    a = rng.integers(0, len(bins), n_pairs).astype(np.int64)
    b = rng.integers(0, len(bins), n_pairs).astype(np.int64)
    y = rng.integers(0, ky, n).astype(np.uint8)
    perms = np.stack([rng.permutation(y) for _ in range(k)])
    return x, a, b, nb, perms, np.bincount(y, minlength=ky).astype(np.float64) / n


def _both(monkeypatch, args, cap=10_000):
    """``(general kernel, outcome-code kernel)`` matrices of one case."""
    monkeypatch.setenv("MLFRAME_PAIR_PERM_BIN", "0")
    general = P.pair_mi_perm_matrix_cuda(*args, cap)
    monkeypatch.setenv("MLFRAME_PAIR_PERM_BIN", "1")
    return general, P.pair_mi_perm_matrix_cuda(*args, cap)


@pytest.mark.parametrize("n", [1, 15, 16, 17, 4001, 70_003])
@pytest.mark.parametrize("k", [1, 2, 8, 9, 25])
def test_the_matrix_equals_the_general_kernel_for_every_row_count_and_permutation_count(monkeypatch, n, k):
    """Row counts around the 16-row load and the word, permutation counts around the 8 of a pass: equal to rounding."""
    args = _case(n, [2, 5, 7, 12], 60, k)
    general, binary = _both(monkeypatch, args)
    assert general is not None and binary is not None and binary.shape == general.shape == (k, 60)
    assert np.allclose(binary, general, rtol=0, atol=1e-14)


def test_pairs_of_very_different_joint_cardinality_in_one_call_are_grouped_and_all_equal(monkeypatch):
    """Joint tables from 4 to 900 cells need 8 down to 3 permutations per pass: each group is launched with its own, and every pair matches the general kernel."""
    args = _case(5000, [2, 5, 12, 30, 30, 9], 400, 25, seed=3)
    _x, a, b, nb, _perms, _freqs = args
    joint = nb[a] * nb[b]
    assert len(set(B.perms_per_pass(joint, 25).tolist())) >= 3
    general, binary = _both(monkeypatch, args)
    assert np.allclose(binary, general, rtol=0, atol=1e-14)


def test_permutations_per_pass_follow_the_shared_budget():
    """8 for a small joint table, fewer as it grows, never more than the permutations asked for, 0 when not even one fits."""
    assert B.perms_per_pass(25, 25) == 8 and B.perms_per_pass(44, 25) == 8 and B.perms_per_pass(100, 25) == 6
    assert B.perms_per_pass(225, 25) == 5 and B.perms_per_pass(900, 25) == 3 and B.perms_per_pass(5632, 25) == 1
    assert B.perms_per_pass(25, 3) == 3 and B.perms_per_pass(5633, 25) == 0
    assert B.perms_per_pass(np.array([25, 900]), 25).tolist() == [8, 3]


def test_the_planes_hold_the_class_of_each_permutation_in_its_own_bit():
    """Bit q of plane p, row i is the class of row i under permutation p * kb + q; the padding rows are zero."""
    y = np.array([[0, 1, 1, 0, 1], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=np.uint8)
    planes = B.pack_planes(y, 2, 8)
    assert planes.shape == (2, 8)
    assert planes[0, :5].tolist() == [2, 3, 1, 0, 1] and planes[1, :5].tolist() == [0, 0, 0, 1, 1]
    assert not planes[:, 5:].any()


def test_a_pair_over_the_cardinality_cap_gets_zeros_as_in_the_general_kernel(monkeypatch):
    """The cap makes the pair of two 30-bin columns invalid (900 cells against a cap of 400): zeros in both kernels, the other pairs unchanged."""
    args = _case(3000, [30, 30, 5, 5], 80, 9, seed=5)
    general, binary = _both(monkeypatch, args, cap=400)
    assert np.allclose(binary, general, rtol=0, atol=1e-14)
    _x, a, b, nb, _perms, _freqs = args
    assert (binary[:, nb[a] * nb[b] == 900] == 0).all() and (binary[:, nb[a] * nb[b] == 25] != 0).any()


def test_a_three_class_target_stays_on_the_general_kernel(monkeypatch):
    """The outcome-code kernel counts one bit per permutation: a target of three classes never reaches it."""
    monkeypatch.setattr(B, "launch_pairs_binary", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the binary kernel was used")))
    args = _case(2000, [4, 6, 5], 30, 5, ky=3)
    assert P.pair_mi_perm_matrix_cuda(*args, 10_000) is not None
