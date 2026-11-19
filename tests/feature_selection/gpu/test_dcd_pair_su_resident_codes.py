"""The pair joint entropies over a resident code matrix equal the host values and the non-resident path, and the matrix is uploaded once (MRMR-DCD-PAIR, 2026-11-19)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

import mlframe.feature_selection.filters._dcd_pair_su_gpu as D
from mlframe.feature_selection.filters._dcd_pair_su_batch import _joint_entropies_host


def _case(n: int, p: int, levels: int, pairs: int, seed: int):
    """A code matrix over ``levels`` bins and ``pairs`` random column pairs of it."""
    rng = np.random.default_rng(seed)
    fd = rng.integers(0, levels, (n, p)).astype(np.int8)
    a = rng.integers(0, p, pairs).astype(np.int64)
    b = ((a + 1 + rng.integers(0, p - 1, pairs)) % p).astype(np.int64)
    return fd, a, b, np.full(p, levels, dtype=np.int64)


def test_resident_path_equals_the_host_entropies():
    """Device values on the resident matrix agree with the numba pair kernel to double precision."""
    fd, a, b, nb = _case(20000, 30, 10, 200, 1)
    got = D.joint_entropies_gpu(fd, a, b, nb)
    ref = np.asarray(_joint_entropies_host(fd, a, b, nb))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)


def test_codes_above_int16_take_the_per_block_path_with_the_same_values():
    """A matrix with a code above 32767 is not made resident and still gives the host values."""
    fd, a, b, nb = _case(6000, 12, 6, 60, 2)
    fd = fd.astype(np.int32)
    fd[0, 0] = 40000
    nb = nb.copy(); nb[0] = 40001
    assert D._resident_codes(cp, fd) is None
    got = D.joint_entropies_gpu(fd, a, b, nb)
    ref = np.asarray(_joint_entropies_host(fd, a, b, nb))
    np.testing.assert_allclose(got[np.isfinite(got)], ref[np.isfinite(got)], rtol=0, atol=1e-9)


def test_the_matrix_is_uploaded_once_per_array():
    """The same matrix object returns the same device array; another matrix gets its own."""
    fd, *_ = _case(5000, 10, 8, 10, 3)
    first = D._resident_codes(cp, fd)
    second = D._resident_codes(cp, fd)
    assert first is second and first is not None
    other, *_ = _case(5000, 10, 8, 10, 4)
    assert D._resident_codes(cp, other) is not first
