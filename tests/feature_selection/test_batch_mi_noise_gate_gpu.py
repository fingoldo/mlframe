"""Bit-identity tests for the GPU backend of ``batch_mi_with_noise_gate``.

The cupy + numba.cuda twins (``batch_mi_noise_gate_gpu``) must reproduce the
EXACT ``fe_mi`` the CPU njit ``batch_mi_with_noise_gate`` produces -- bit-identity
is non-negotiable because the noise-gate rejection is a float comparison
(``mi_perm >= original_mi``); a single-ULP drift would flip borderline rejections
and change which engineered features MRMR keeps.

Covered: varied n / K / nbins, npermutations in {0, 3, 10}, min_nonzero_confidence
in {0.99, 0.0}, plus tie-heavy + pure-noise (rejection-triggering) columns and
SU-normalization mode. The dispatcher (``dispatch_batch_mi_with_noise_gate_gpu``)
is exercised via ``force_backend`` overrides.

Auto-skips when CUDA / CuPy is unavailable (CI without GPU).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    batch_mi_with_noise_gate,
    merge_vars,
)
from mlframe.feature_selection.filters.batch_mi_noise_gate_gpu import (
    _CUDA_AVAIL,
    _CUPY_AVAIL,
    batch_mi_with_noise_gate_cupy,
    batch_mi_with_noise_gate_cuda,
    dispatch_batch_mi_with_noise_gate_gpu,
)


def _make_frame(n, K, nbins, seed):
    """(n, K) int frame mixing informative / pure-noise / tie-heavy / strongly-
    informative columns -- identical construction to the CPU bit-identity test."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n).astype(np.int32)
    cols = np.empty((n, K), dtype=np.int32)
    for k in range(K):
        kind = k % 4
        if kind == 0:
            c = (y + rng.integers(0, 2, size=n)) % nbins
        elif kind == 1:
            c = rng.integers(0, nbins, size=n)
        elif kind == 2:
            c = np.zeros(n, dtype=np.int64)
            idx = rng.choice(n, size=max(1, n // 20), replace=False)
            c[idx] = rng.integers(1, nbins, size=idx.size)
        else:
            c = y.copy().astype(np.int64)
            flip = rng.choice(n, size=max(1, n // 10), replace=False)
            c[flip] = rng.integers(0, nbins, size=flip.size)
        cols[:, k] = (c % nbins).astype(np.int32)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1),
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64),
        dtype=np.int32,
    )
    return cols, classes_y, freqs_y


def _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, npermutations, mnc, use_su):
    return batch_mi_with_noise_gate(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y.copy(),
        freqs_y=freqs_y,
        npermutations=npermutations,
        base_seed=np.uint64(0),
        min_nonzero_confidence=float(mnc),
        use_su=use_su,
        dtype=np.int32,
    )


_BACKENDS = []
if _CUPY_AVAIL:
    _BACKENDS.append(("cupy", batch_mi_with_noise_gate_cupy))
if _CUDA_AVAIL:
    _BACKENDS.append(("cuda", batch_mi_with_noise_gate_cuda))


@pytest.mark.skipif(not (_CUDA_AVAIL or _CUPY_AVAIL), reason="no GPU backend available")
@pytest.mark.parametrize("backend_name,backend_fn", _BACKENDS or [("none", None)])
@pytest.mark.parametrize("n,K,nbins", [(200, 8, 4), (500, 13, 6), (1000, 20, 5), (2407, 64, 10)])
@pytest.mark.parametrize("npermutations", [0, 3, 10])
@pytest.mark.parametrize("min_nonzero_confidence", [0.99, 0.0])
def test_gpu_bit_identical_to_cpu(backend_name, backend_fn, n, K, nbins, npermutations, min_nonzero_confidence):
    if backend_fn is None:
        pytest.skip("no GPU backend available")
    disc_2d, classes_y, freqs_y = _make_frame(n, K, nbins, seed=1234 + n + K + nbins)
    factors_nbins = np.full(K, nbins, dtype=np.int64)

    ref = _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, npermutations, min_nonzero_confidence, False)
    got = backend_fn(
        disc_2d, factors_nbins, classes_y, classes_y.copy(), freqs_y,
        npermutations, np.uint64(0), float(min_nonzero_confidence), False, np.int32,
    )
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"{backend_name} mismatch n={n} K={K} nbins={nbins} nperm={npermutations} "
        f"mnc={min_nonzero_confidence}\n ref={ref}\n got={got}\n diff={got - ref}"
    )


@pytest.mark.skipif(not (_CUDA_AVAIL or _CUPY_AVAIL), reason="no GPU backend available")
@pytest.mark.parametrize("backend_name,backend_fn", _BACKENDS or [("none", None)])
def test_gpu_su_mode_bit_identical(backend_name, backend_fn, monkeypatch):
    if backend_fn is None:
        pytest.skip("no GPU backend available")
    import mlframe.feature_selection.filters.info_theory as it
    monkeypatch.setattr(it, "use_su_normalization", lambda: True)
    disc_2d, classes_y, freqs_y = _make_frame(400, 10, 5, seed=7)
    factors_nbins = np.full(10, 5, dtype=np.int64)
    ref = _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, 10, 0.99, True)
    got = backend_fn(
        disc_2d, factors_nbins, classes_y, classes_y.copy(), freqs_y,
        10, np.uint64(0), 0.99, True, np.int32,
    )
    assert np.array_equal(got, ref)


@pytest.mark.skipif(not (_CUDA_AVAIL or _CUPY_AVAIL), reason="no GPU backend available")
def test_gpu_pure_noise_zeroed_identically():
    n = 800
    rng = np.random.default_rng(99)
    y = rng.integers(0, 4, size=n).astype(np.int32)
    noise = rng.integers(0, 6, size=n).astype(np.int32)
    disc_2d = noise.reshape(-1, 1)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1),
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64),
        dtype=np.int32,
    )
    factors_nbins = np.array([6], dtype=np.int64)
    ref = _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, 10, 0.99, False)
    for _name, fn in _BACKENDS:
        got = fn(disc_2d, factors_nbins, classes_y, classes_y.copy(), freqs_y,
                 10, np.uint64(0), 0.99, False, np.int32)
        assert np.array_equal(got, ref), _name


@pytest.mark.skipif(not (_CUDA_AVAIL or _CUPY_AVAIL), reason="no GPU backend available")
@pytest.mark.parametrize("force", [b for b, _ in _BACKENDS])
def test_dispatch_force_backend_bit_identical(force):
    disc_2d, classes_y, freqs_y = _make_frame(700, 32, 8, seed=321)
    factors_nbins = np.full(32, 8, dtype=np.int64)
    ref = _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, 3, 0.99, False)
    out = dispatch_batch_mi_with_noise_gate_gpu(
        disc_2d, factors_nbins, classes_y, classes_y.copy(), freqs_y,
        3, np.uint64(0), 0.99, False, np.int32, force_backend=force,
    )
    assert out is not None
    fe_mi, backend_name = out
    assert backend_name == force
    assert np.array_equal(fe_mi, ref)


@pytest.mark.skipif(not (_CUDA_AVAIL or _CUPY_AVAIL), reason="no GPU backend available")
def test_empty_and_single_edge_cases():
    """K==0 and n==0 return empty/zero arrays without touching the device."""
    for _name, fn in _BACKENDS:
        # K == 0
        out = fn(np.empty((5, 0), dtype=np.int32), np.empty(0, dtype=np.int64),
                 np.zeros(5, dtype=np.int32), np.zeros(5, dtype=np.int32),
                 np.array([1.0]), 3, np.uint64(0), 0.99, False, np.int32)
        assert out.shape == (0,)
        # n == 0
        out = fn(np.empty((0, 3), dtype=np.int32), np.full(3, 4, dtype=np.int64),
                 np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                 np.array([0.25, 0.25, 0.25, 0.25]), 3, np.uint64(0), 0.99, False, np.int32)
        assert out.shape == (3,)
        assert np.array_equal(out, np.zeros(3))


@pytest.mark.skipif(not _CUPY_AVAIL, reason="cupy unavailable")
@pytest.mark.parametrize("size,flatlen", [(64, 5000), (1000, 120000), (24000, 800000), (50, 200)])
def test_cupy_bincount_known_size_byte_identical(size, flatlen):
    """OPT-D: the known-size bincount helper (skips cupy.bincount's (x<0).any() +
    cupy.max(x) host-sync validations) must be BYTE-IDENTICAL to cupy.bincount, for
    every (size, flatlen) the FE-MI gate exercises. Covers a flat index whose max
    is < size-1 (so the size-difference path is hit) and a fully-saturated one."""
    import cupy as cp
    from mlframe.feature_selection.filters.batch_mi_noise_gate_gpu import _cupy_bincount_known_size
    rng = np.random.default_rng(size * 7 + flatlen)
    # indices strictly < size (non-negative by construction, as in the gate)
    idx = rng.integers(0, size, size=flatlen).astype(np.int64)
    d_idx = cp.asarray(idx)
    ref = cp.asnumpy(cp.bincount(d_idx, minlength=size)[:size])
    got = cp.asnumpy(_cupy_bincount_known_size(d_idx, size))
    assert got.shape == (size,)
    assert got.dtype == ref.dtype
    assert np.array_equal(got, ref)
    # an index set whose observed max is well below size-1: cupy.bincount would size
    # via minlength here; the helper must still produce all `size` slots (trailing zeros).
    idx2 = rng.integers(0, max(1, size // 2), size=flatlen).astype(np.int64)
    d_idx2 = cp.asarray(idx2)
    ref2 = cp.asnumpy(cp.bincount(d_idx2, minlength=size)[:size])
    got2 = cp.asnumpy(_cupy_bincount_known_size(d_idx2, size))
    assert np.array_equal(got2, ref2)
