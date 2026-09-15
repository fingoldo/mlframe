"""The order-2 maxT floor builds its pair operand arrays without the C(k, 2) triu index arrays, and fills its bias dict in chunks.

``compute_pair_maxt_floor`` called ``np.triu_indices`` (two C(k, 2) int64 arrays) and then gathered both operand arrays from them, so four
C(k, 2) arrays were alive at once (~400 MB at k=5000); the MM-bias dict was then filled from two full-length ``.tolist()`` copies.
"""

from __future__ import annotations

import tracemalloc
from itertools import combinations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _pair_operand_arrays as poa


@pytest.mark.parametrize("k", [0, 1, 2, 3, 7, 64, 301])
def test_operand_arrays_equal_triu_gather_in_combinations_order(k):
    """Both halves equal the triu gather, and the pair order is itertools.combinations' order."""
    rng = np.random.default_rng(k)
    k_vars = rng.permutation(np.arange(1000, 1000 + k)).astype(np.int64)
    pa, pb = poa.pair_operand_arrays(k_vars)
    ia, ib = np.triu_indices(k, k=1)
    np.testing.assert_array_equal(pa, k_vars[ia])
    np.testing.assert_array_equal(pb, k_vars[ib])
    assert list(zip(pa.tolist(), pb.tolist())) == list(combinations(k_vars.tolist(), 2))
    assert pa.dtype == np.int64 and pb.dtype == np.int64


def test_operand_arrays_follow_set_iteration_order():
    """The caller passes a set through np.fromiter; the arrays follow that iteration order exactly."""
    vars_set = {17, 3, 99, 42, 5, 250}
    k_vars = np.fromiter(vars_set, dtype=np.int64, count=len(vars_set))
    pa, pb = poa.pair_operand_arrays(k_vars)
    assert list(zip(pa.tolist(), pb.tolist())) == list(combinations(list(vars_set), 2))


def _peak_bytes(fn) -> int:
    """Peak traced allocation, in bytes, while ``fn`` runs."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def _triu_gather(k_vars: np.ndarray):
    """The replaced construction: two C(k, 2) index arrays, then a gather for each operand half."""
    ia, ib = np.triu_indices(k_vars.shape[0], k=1)
    return k_vars[ia], k_vars[ib]


def test_no_triu_index_arrays_allocated():
    """The helper's peak stays near its two output arrays and well below the triu-gather construction it replaces."""
    k = 3000
    m = k * (k - 1) // 2
    k_vars = np.arange(k, dtype=np.int64)
    helper_peak = _peak_bytes(lambda: poa.pair_operand_arrays(k_vars))
    triu_peak = _peak_bytes(lambda: _triu_gather(k_vars))
    two_arrays = 2 * 8 * m
    assert helper_peak < 1.25 * two_arrays, f"helper peak {helper_peak / 1e6:.1f} MB vs two operand arrays {two_arrays / 1e6:.1f} MB"
    assert helper_peak < 0.7 * triu_peak, f"helper peak {helper_peak / 1e6:.1f} MB is not clearly below the triu gather's {triu_peak / 1e6:.1f} MB"


def test_chunked_bias_fill_equals_full_comprehension(monkeypatch):
    """Filling across many chunks gives exactly the dict the one-shot comprehension built, with canonical (min, max) keys."""
    rng = np.random.default_rng(0)
    k_vars = rng.permutation(np.arange(60)).astype(np.int64)
    pa, pb = poa.pair_operand_arrays(k_vars)
    bias = rng.random(pa.shape[0])
    expected = {}
    for pi, (a, b) in enumerate(zip(pa.tolist(), pb.tolist())):
        expected[(a, b) if a <= b else (b, a)] = float(bias[pi])
    monkeypatch.setattr(poa, "_DICT_FILL_CHUNK", 97)
    got: dict = {}
    poa.fill_pair_bias(got, pa, pb, bias)
    assert got == expected
    assert all(a <= b for a, b in got)
