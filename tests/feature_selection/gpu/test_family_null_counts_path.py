"""The fused-table family null equals the thread-per-pair kernel and the CPU kernel bit for bit, and ``family_null_max_device`` runs it first."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _family_maxt_counts_device as counts_path
from mlframe.feature_selection.filters import _family_maxt_cuda as cuda
from mlframe.feature_selection.filters._family_maxt_gate import family_null_max_njit


def _family(seed: int, k: int, n: int, n_y: int, b: int, max_bins: int = 20):
    """Columns with different bin counts, a signal column, labels and ``b`` permutations of them."""
    rng = np.random.default_rng(seed)
    n_x = rng.integers(2, max_bins + 1, k).astype(np.int64)
    cols = np.stack([rng.integers(0, int(v), n) for v in n_x]).astype(np.int16)
    y = rng.integers(0, n_y, n)
    cols[0] = np.where(rng.random(n) < 0.5, y % int(n_x[0]), cols[0])
    perms = np.stack([rng.permutation(y) for _ in range(b)]).astype(np.int64)
    return cols, n_x, y, perms


@pytest.mark.parametrize("n_y", [2, 7, 20])
def test_counts_path_equals_the_thread_per_pair_kernel_bit_for_bit_and_the_cpu_kernel_to_an_ulp(n_y, monkeypatch):
    """Bit-identical to the numba CUDA kernel it replaces (same arithmetic and summation order); the njit CPU kernel differs from both in the last bit at most."""
    cols, n_x, _y, perms = _family(n_y, 40, 3001, n_y, 9)
    got = counts_path.family_null_max_counts(cols, n_x, perms, n_y)
    assert got is not None
    monkeypatch.setattr(counts_path, "family_null_max_counts", lambda *a, **k: None)  # the entry then runs the thread-per-pair kernel
    thread_per_pair = cuda.family_null_max_device(cols, n_x, perms, n_y)
    np.testing.assert_array_equal(got, thread_per_pair)
    np.testing.assert_allclose(got, family_null_max_njit(cols, n_x, perms, n_y), rtol=1e-14, atol=0)


def test_tiles_over_columns_and_permutations_give_the_same_maxima(monkeypatch):
    """A tiny table budget and a two-permutation block force many (column block x permutation block) tiles; the maxima do not move."""
    cols, n_x, _y, perms = _family(5, 70, 2000, 20, 11)
    whole = counts_path.family_null_max_counts(cols, n_x, perms, 20)
    monkeypatch.setattr(counts_path, "_TABLE_BUDGET_BYTES", 3 * 2 * 20 * 20 * 4)
    monkeypatch.setattr(counts_path, "_PERM_BLOCK", 2)
    tiled = counts_path.family_null_max_counts(cols, n_x, perms, 20)
    np.testing.assert_array_equal(tiled, whole)


def test_shapes_the_kernel_does_not_hold_are_declined():
    """More classes than the register array, or a table over 48 KB, returns None (the caller keeps the thread-per-pair kernel)."""
    cols, n_x, _y, perms = _family(1, 4, 500, 70, 3, max_bins=4)
    assert counts_path.family_null_max_counts(cols, n_x, perms, 70) is None
    cols2, n_x2, _y2, perms2 = _family(2, 3, 500, 20, 3, max_bins=20)
    wide = np.full_like(n_x2, 700)
    assert counts_path.family_null_max_counts(cols2, wide, perms2, 20) is None


def test_family_null_max_device_runs_the_counts_path_first_and_falls_back_on_error(monkeypatch):
    """The device entry asks the fused path first; when it raises a non-OOM error the thread-per-pair kernel answers with the same maxima."""
    cols, n_x, _y, perms = _family(4, 30, 1500, 20, 6)
    calls = []
    real = counts_path.family_null_max_counts
    monkeypatch.setattr(counts_path, "family_null_max_counts", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = cuda.family_null_max_device(cols, n_x, perms, 20)
    assert calls, "the fused path was not asked"

    def broken(*a, **k):
        """A fused path that fails."""
        raise RuntimeError("boom")

    monkeypatch.setattr(counts_path, "family_null_max_counts", broken)
    second = cuda.family_null_max_device(cols, n_x, perms, 20)
    np.testing.assert_array_equal(first, second)
