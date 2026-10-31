"""The blocked bitset null kernel (four permutations per pass over the column bitsets) returns exactly the values of the one-permutation kernel."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_null_v2_njit as V


@pytest.mark.parametrize("n_y", [2, 3])
@pytest.mark.parametrize("b_count", [1, 3, 4, 7, 10])
@pytest.mark.parametrize("n", [100, 1000, 4099])
def test_blocked_kernel_equals_the_single_permutation_kernel(n_y, b_count, n):
    """Bit-equal maxima for permutation counts that do and do not fill the last block, label counts and row counts that do and do not fill the last word."""
    rng = np.random.default_rng(n + b_count)
    k = 7
    cols = rng.integers(0, 6, (k, n)).astype(np.int64)
    n_x = np.full(k, 6, dtype=np.int64)
    y = rng.integers(0, n_y, n).astype(np.int64)
    y_perms = np.stack([y[rng.permutation(n)] for _ in range(b_count)])
    table, c_a, px, groups = V.bitset_groups(cols, n_x)
    bits = groups[0][2]
    want = V.family_null_max_bitsets(bits, n_x, y_perms, n_y, table, c_a, px)
    got = V.family_null_max_bitsets_blocked(bits, n_x, y_perms, n_y, table, c_a, px)
    assert np.array_equal(want, got)
    assert np.array_equal(V.null_max_bitsets((table, c_a, px, groups), n_x, y_perms, n_y), want)


def test_family_null_max_from_class_bitsets_equals_the_label_matrix_route():
    """The family gate packs the class bit sets from the permutation indices; its maxima equal those of the kernel run on the permuted label matrix."""
    from mlframe.feature_selection.filters import _family_maxt_gate as gate

    rng = np.random.default_rng(11)
    n, k = 900, 9
    cols = np.ascontiguousarray(rng.integers(0, 5, size=(k, n)).astype(np.int64))
    n_x = (cols.max(axis=1) + 1).astype(np.int64)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    got = gate.family_null_max(cols, n_x, y, 3, 300, np.random.default_rng(2))
    idx = gate._permutation_block(np.random.default_rng(2), y, 256)
    idx2 = gate._permutation_block(np.random.default_rng(2), y, 256)
    assert np.array_equal(idx, idx2)
    table, c_a, px, groups = V.bitset_groups(cols, n_x)
    want = V.null_max_bitsets((table, c_a, px, groups), n_x, y[idx], 3)
    assert np.array_equal(got[:256], want)


def test_column_bit_sets_are_built_once_for_all_blocks_of_one_exact_test(monkeypatch):
    """The blocks of one family test share the column bit sets (one build), and the pass flags equal those with a build per block."""
    from mlframe.feature_selection.filters import _family_maxt_gate as gate

    rng = np.random.default_rng(13)
    n, k = 2000, 8
    cols = np.ascontiguousarray(rng.integers(0, 5, size=(k, n)).astype(np.int64))
    n_x = (cols.max(axis=1) + 1).astype(np.int64)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    observed_max = np.array([0.0005, 0.02])
    built = []
    real = gate.bitset_groups
    monkeypatch.setattr(gate, "bitset_groups", lambda *a, **kw: (built.append(1), real(*a, **kw))[1])
    shared = gate._permutation_pass_flags(cols, n_x, y, 2, observed_max, 0.05, 700, 0)
    n_shared = len(built)
    built.clear()
    per_block = gate._permutation_pass_flags_with(lambda m, r: gate.family_null_max(cols, n_x, y, 2, m, r), observed_max, 0.05, 700, 0)
    assert n_shared == 1 and len(built) > 1
    assert np.array_equal(shared, per_block)


def test_a_strong_family_is_settled_early_and_flags_equal_the_full_count(monkeypatch):
    """With a planted signal the batch passes after a few blocks (far fewer null draws than the full count), and with the early stop off the same flags come out after every draw."""
    from mlframe.feature_selection.filters import _family_maxt_gate as gate

    rng = np.random.default_rng(21)
    n, k = 3000, 6
    cols = np.ascontiguousarray(rng.integers(0, 4, size=(k, n)).astype(np.int64))
    n_x = (cols.max(axis=1) + 1).astype(np.int64)
    y = np.where(rng.random(n) < 0.85, cols[2] % 2, rng.integers(0, 2, n)).astype(np.int64)
    observed = np.array([0.35, 0.0002])  # one batch far above the null, one at noise level
    drawn = []
    real = gate.family_null_max
    monkeypatch.setattr(gate, "family_null_max", lambda c, nx, yy, ny, m, r, **kw: (drawn.append(m), real(c, nx, yy, ny, m, r, **kw))[1])
    early = gate._permutation_pass_flags(cols, n_x, y, 2, observed[:1], 0.01, 9999, 0)
    n_early = sum(drawn)
    drawn.clear()
    monkeypatch.setattr(gate, "_CERTAINTY_DELTA", 0.0)
    full = gate._permutation_pass_flags(cols, n_x, y, 2, observed[:1], 0.01, 9999, 0)
    assert np.array_equal(early, full) and bool(early[0])
    assert n_early < 0.5 * sum(drawn) and sum(drawn) == 9999
    assert not gate._passes_for_certain(np.array([60]), 1500, 9999, 0.01)  # a count near the limit is never settled early
