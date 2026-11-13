"""The fused joint-entropy kernels read a byte code matrix as bytes and give the 64-bit matrix's entropies and occupied-cell counts."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_batched_mi_cmi as C

pytestmark = pytest.mark.gpu


def _codes(n=4000, k=13, kx=10, kb=40, seed=0):
    """A random code matrix as int64 and the same codes as bytes, with a conditioning code vector."""
    rng = np.random.default_rng(seed)
    wide = cp.asarray(rng.integers(0, kx, size=(n, k), dtype=np.int64))
    return wide, wide.astype(cp.uint8), cp.asarray(rng.integers(0, kb, size=n, dtype=np.int64)), kx, kb


@pytest.mark.parametrize("route", ["tiled", "untiled"])
def test_global_scratch_kernels_equal_for_bytes_and_int64(route, monkeypatch):
    """Both global-scratch routes (column groups and one column per block) agree exactly between the byte and the 64-bit matrix."""
    monkeypatch.setattr(C, "_JOINT_ENTROPY_GLOBAL_TILE_MAX_CELLS", 6144 if route == "tiled" else 0)
    wide, byte, b, kx, kb = _codes()
    n = wide.shape[0]
    h64, k64 = C.batched_joint_entropy_global(wide, b, kx, kb, 1.0 / n, 3)
    h8, k8 = C.batched_joint_entropy_global(byte, b, kx, kb, 1.0 / n, 3)
    assert bool((h64 == h8).all()) and bool((k64 == k8).all())


def test_shared_memory_kernel_equal_for_bytes_and_int64(monkeypatch):
    """The shared-histogram kernel (forced on) agrees between the matrices."""
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY", "1")
    wide, byte, b, kx, kb = _codes(k=21, kb=12)
    n = wide.shape[0]
    r64 = C._batched_joint_entropy_and_k2(wide, b, kx, kb, 1.0 / n)
    r8 = C._batched_joint_entropy_and_k2(byte, b, kx, kb, 1.0 / n)
    assert r64 is not None and r8 is not None
    assert bool((r64[0] == r8[0]).all()) and bool((r64[1] == r8[1]).all())


def test_code_type_follows_the_dtype():
    """Bytes are read as unsigned char, any other integer matrix as long long."""
    assert C._code_type(cp.zeros((2, 2), dtype=cp.uint8)) == "unsigned char"
    assert C._code_type(cp.zeros((2, 2), dtype=cp.int64)) == "long long"
    assert C._code_type(cp.zeros((2, 2), dtype=cp.int32)) == "long long"
