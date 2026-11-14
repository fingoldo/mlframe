"""The radix select builds its prefix windows with a warp scan (ascending ranks) and keeps the serial dedup for anything else; both give the order statistics a column sort gives."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_select as S
from mlframe.feature_selection.filters import _gpu_resident_select_f32_compact as C


def _launch(x, n, K, ranks_h, threads):
    """Run the f32 v5 select on the (K, n) column-major matrix ``x`` for the given rank vector; returns the (R, K) values."""
    kern = C._get_radix_select_f32_v5_kernel()
    R = len(ranks_h)
    out = cp.empty((R, K), dtype=cp.float32)
    cap = max(1024, n // 4)
    scratch = cp.empty((K, cap), dtype=cp.uint32)
    kern((K,), (threads,), (x, np.int64(n), np.int32(K), cp.asarray(ranks_h), np.int32(R), out, scratch, np.int64(cap)), shared_mem=min(256 * 4 * R, 48 * 1024))
    return out


def _reference(x, ranks_h):
    """What sorting every column and reading the ranks gives."""
    return cp.sort(x, axis=1)[:, cp.asarray(ranks_h)].T


def _ascending_ranks(n, r):
    """``r`` distinct ascending ranks spread over ``n`` rows."""
    return np.unique(np.linspace(0, n - 1, r).astype(np.int64))


@pytest.mark.parametrize("r", [1, 2, 5, 18, 32, 33, 38])
@pytest.mark.parametrize("threads", [64, 768, 1024])
@pytest.mark.parametrize("kind", ["normal", "ties", "lognormal"])
def test_ascending_ranks_equal_the_column_sort(r, threads, kind):
    """Rank counts either side of one warp (32, 33) and up to 38 (the most the 48 KB of shared windows hold on this card, the callers refuse more), three block widths, and data with heavy ties, a skewed tail and a plain normal."""
    rng = np.random.default_rng(r * 7 + threads)
    n, K = 4001, 9
    h = {"normal": rng.standard_normal((K, n)), "ties": np.round(rng.standard_normal((K, n)) * 3), "lognormal": rng.lognormal(size=(K, n))}[kind]
    x = cp.asarray(np.ascontiguousarray(h.astype(np.float32)))
    ranks_h = _ascending_ranks(n, r)
    got = _launch(x, n, K, ranks_h, threads)
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(_reference(x, ranks_h)))


@pytest.mark.parametrize("r", [5, 38])
def test_non_ascending_ranks_take_the_serial_dedup_and_still_equal_the_sort(r):
    """A shuffled rank vector (never produced by the callers) makes the prefixes decrease: the serial path answers with the same values."""
    rng = np.random.default_rng(r)
    n, K = 3000, 6
    x = cp.asarray(np.ascontiguousarray(rng.standard_normal((K, n)).astype(np.float32)))
    ranks_h = rng.permutation(_ascending_ranks(n, r))
    got = _launch(x, n, K, ranks_h, 1024)
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(_reference(x, ranks_h)))


def test_the_public_entry_accepts_unordered_and_repeated_ranks_and_equals_the_sort():
    """The warp scan relies on ascending distinct ranks; the public entry builds them itself (np.unique), so shuffled and repeated requests still return the sort's values in the requested order."""
    rng = np.random.default_rng(11)
    n, K = 3000, 5
    x = cp.asarray(rng.standard_normal((n, K)).astype(np.float32))
    ranks = np.array([2500, 3, 3, 1200, 0, 2999, 1200, 77], dtype=np.int64)
    got = S.radix_order_statistics(x, ranks)
    assert got is not None
    want = cp.sort(x, axis=0)[cp.asarray(ranks)]
    assert np.array_equal(cp.asnumpy(got), cp.asnumpy(want))
