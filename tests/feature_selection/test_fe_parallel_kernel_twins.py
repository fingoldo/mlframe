"""OPT-A (2026-06-07): byte-identity of the ``parallel=True`` column-prange twins of the
FE materialise + searchsorted kernels vs their serial ``nogil`` originals.

On the SERIAL-MAIN-THREAD FE path (``len(X) < 50000`` -> ``check_prospective_fe_pairs`` runs
with NO joblib threading nest) the FE pipeline may dispatch to the parallel twins so the
embarrassingly-parallel per-column work spreads across cores. The twins MUST produce the
EXACT SAME bytes as the serial kernels (each column is reduced independently, so the result
is thread-count-invariant) -- any drift would flip the float ``>=`` noise-gate and change
feature selection. These tests assert that bit-identity directly on the kernels, plus that
the joblib (>=50000-row) path keeps the serial variant (parallel dispatch is gated OFF when
``serial_main_thread`` is False).
"""
from __future__ import annotations

import numpy as np
import pytest


# All 9 njit-coded binary op codes (mul/add/sub/div/max/min/abs_diff/signed/ratio_abs).
_ALL_OPS = list(range(9))


def _rand_inputs(n_rows, K, n_operands, seed, with_nan=False, with_inf=False):
    rng = np.random.default_rng(seed)
    tv = rng.standard_normal((n_rows, n_operands)).astype(np.float32)
    if with_nan:
        # sprinkle NaN to exercise the nan-propagating max/min/signed branches + nan_to_num
        nan_idx = rng.integers(0, n_rows, size=max(1, n_rows // 20))
        tv[nan_idx, rng.integers(0, n_operands, size=nan_idx.shape[0])] = np.nan
    if with_inf:
        inf_idx = rng.integers(0, n_rows, size=max(1, n_rows // 30))
        tv[inf_idx, rng.integers(0, n_operands, size=inf_idx.shape[0])] = np.inf
    a_cols = rng.integers(0, n_operands, size=K).astype(np.int64)
    b_cols = rng.integers(0, n_operands, size=K).astype(np.int64)
    # cycle through every op so all branches are covered
    ops = np.asarray([_ALL_OPS[i % len(_ALL_OPS)] for i in range(K)], dtype=np.int8)
    return tv, a_cols, b_cols, ops


@pytest.mark.parametrize("with_nan,with_inf", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("n_rows,K,n_operands", [(7, 5, 3), (256, 64, 12), (2407, 300, 20)])
def test_materialise_parallel_eq_serial(n_rows, K, n_operands, with_nan, with_inf):
    from mlframe.feature_selection.filters._feature_engineering_pairs import (
        _materialise_chunk_njit,
        _materialise_chunk_njit_parallel,
    )
    tv, a_cols, b_cols, ops = _rand_inputs(n_rows, K, n_operands, seed=11 + n_rows + K,
                                           with_nan=with_nan, with_inf=with_inf)
    out_serial = np.empty((n_rows, K), dtype=np.float32)
    out_parallel = np.empty((n_rows, K), dtype=np.float32)
    _materialise_chunk_njit(tv, a_cols, b_cols, ops, out_serial)
    _materialise_chunk_njit_parallel(tv, a_cols, b_cols, ops, out_parallel)
    # BYTE-identical: same bits (view as uint32 so NaN/-0.0/+0.0 distinctions are exact).
    assert np.array_equal(out_serial.view(np.uint32), out_parallel.view(np.uint32)), (
        "parallel materialise twin diverged from serial -- would flip the noise-gate"
    )


@pytest.mark.parametrize("dtype_in", [np.float32, np.float64])
@pytest.mark.parametrize("n_rows,K", [(7, 4), (256, 64), (2407, 300)])
def test_searchsorted_parallel_eq_serial(n_rows, K, dtype_in):
    from mlframe.feature_selection.filters.discretization import (
        _searchsorted_2d_right_njit,
        _searchsorted_2d_right_njit_parallel,
    )
    rng = np.random.default_rng(7 + n_rows + K)
    arr2d = np.ascontiguousarray(rng.standard_normal((n_rows, K)).astype(dtype_in))
    # n_bins-1 interior edges per column, ascending (mirrors the percentile edge slice).
    n_edges = 9
    edges_inner = np.ascontiguousarray(
        np.sort(rng.standard_normal((n_edges, K)), axis=0).astype(np.float64)
    )
    out_serial = np.empty((n_rows, K), dtype=np.int8)
    out_parallel = np.empty((n_rows, K), dtype=np.int8)
    _searchsorted_2d_right_njit(edges_inner, arr2d, out_serial)
    _searchsorted_2d_right_njit_parallel(edges_inner, arr2d, out_parallel)
    assert np.array_equal(out_serial, out_parallel), (
        "parallel searchsorted twin diverged from serial"
    )


def test_searchsorted_parallel_eq_serial_with_nan():
    """NaN routes to the rightmost bin (n_edges) identically in both kernels."""
    from mlframe.feature_selection.filters.discretization import (
        _searchsorted_2d_right_njit,
        _searchsorted_2d_right_njit_parallel,
    )
    rng = np.random.default_rng(99)
    n_rows, K, n_edges = 500, 40, 9
    arr2d = rng.standard_normal((n_rows, K)).astype(np.float32)
    arr2d[rng.integers(0, n_rows, 25), rng.integers(0, K, 25)] = np.nan
    arr2d = np.ascontiguousarray(arr2d)
    edges_inner = np.ascontiguousarray(np.sort(rng.standard_normal((n_edges, K)), axis=0).astype(np.float64))
    out_s = np.empty((n_rows, K), dtype=np.int8)
    out_p = np.empty((n_rows, K), dtype=np.int8)
    _searchsorted_2d_right_njit(edges_inner, arr2d, out_s)
    _searchsorted_2d_right_njit_parallel(edges_inner, arr2d, out_p)
    assert np.array_equal(out_s, out_p)


def test_discretize_2d_quantile_batch_parallel_flag_byte_identical():
    """The public ``discretize_2d_quantile_batch(parallel=True)`` produces byte-identical
    codes to ``parallel=False`` (the dispatch must not perturb output)."""
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch
    rng = np.random.default_rng(3)
    arr2d = rng.standard_normal((2407, 400)).astype(np.float32)
    a = discretize_2d_quantile_batch(arr2d, n_bins=10, dtype=np.int32, parallel=False)
    b = discretize_2d_quantile_batch(arr2d, n_bins=10, dtype=np.int32, parallel=True)
    assert np.array_equal(a, b)


def test_dispatch_predicate_gates_on_serial_main_thread():
    """The OPT-A dispatch predicate MUST return False on the joblib path
    (``serial_main_thread`` False) regardless of column count -- nesting a numba prange
    inside the threading layer deadlocks."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _fe_use_parallel_kernels
    # joblib path: never parallel, even for a huge chunk.
    assert _fe_use_parallel_kernels(100_000, serial_main_thread=False) is False
    # main-thread path, large chunk: parallel (fallback heuristic >= 256 cols).
    assert _fe_use_parallel_kernels(4096, serial_main_thread=True) is True
    # main-thread path, tiny chunk: serial (prange overhead not worth it).
    assert _fe_use_parallel_kernels(8, serial_main_thread=True) is False
