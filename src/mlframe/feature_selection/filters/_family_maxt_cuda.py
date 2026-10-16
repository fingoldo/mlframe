"""CUDA twin of :func:`_family_maxt_gate.family_null_max_njit` (maximum plug-in MI over a candidate family for each label permutation).

One CUDA thread per (permutation, candidate column) pair. The thread builds the joint histogram of its column with the permutation in
local memory and evaluates the same plug-in MI formula, in the same summation order, as the CPU kernel. The per-pair values come back as a
(B, K) matrix and the per-permutation maximum is taken on the device, so only B values cross the bus.

Scope: histograms up to ``_MAX_CELLS`` joint cells (local-memory budget). Larger tables fall back to the CPU kernel.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda

_MAX_CELLS = 1024
_THREADS = 128


def _make_family_mi_kernel(max_cells: int):
    """Build the family-MI kernel with a local joint table of ``max_cells`` cells (a compile-time size, so the table lives in registers/L1
    for small tables rather than the full worst-case budget)."""

    @cuda.jit
    def _family_mi_kernel(cols_t, n_x, perms, n_y, out):
        """out[b, k] = plug-in MI of column k with permutation b; one thread per (b, k)."""
        tid = cuda.grid(1)
        K = cols_t.shape[1]
        B = perms.shape[0]
        if tid >= B * K:
            return
        # Consecutive threads take consecutive columns of one permutation: the permutation row is a broadcast read and the transposed
        # column matrix ``cols_t[i, k]`` is read contiguously across the warp.
        b = tid // K
        k = tid - b * K
        n = perms.shape[1]
        nx = n_x[k]
        cells = nx * n_y
        joint = cuda.local.array(max_cells, dtype=np.int64)
        for c in range(cells):
            joint[c] = 0
        for i in range(n):
            joint[cols_t[i, k] * n_y + perms[b, i]] += 1
        mi = 0.0
        for a in range(nx):
            pxa = 0
            for bb in range(n_y):
                pxa += joint[a * n_y + bb]
            if pxa == 0:
                continue
            for bb in range(n_y):
                nab = joint[a * n_y + bb]
                if nab == 0:
                    continue
                pyb = 0
                for aa in range(nx):
                    pyb += joint[aa * n_y + bb]
                mi += (nab / n) * math.log((nab * n) / (pxa * pyb))
        out[b, k] = mi

    return _family_mi_kernel


_KERNELS = {64: _make_family_mi_kernel(64), _MAX_CELLS: _make_family_mi_kernel(_MAX_CELLS)}


def device_supported(n_x: np.ndarray, n_y: int) -> bool:
    """True when every column's joint table fits the kernel's local-memory budget."""
    return int(n_y) * (int(np.max(n_x)) if np.size(n_x) else 0) <= _MAX_CELLS


def family_null_max_device(cols: np.ndarray, n_x: np.ndarray, perms: np.ndarray, n_y: int) -> np.ndarray:
    """Per-permutation maximum MI over the columns, computed on the GPU (see module docstring)."""
    import cupy as cp

    from mlframe.utils.cupy_pinned import install_pinned_policy

    install_pinned_policy()
    B = int(perms.shape[0])
    K = int(cols.shape[0])
    # Codes are uploaded in their own (int16) width: a (K, n) int64 copy of a wide family was 4x the bytes for the same values.
    cols_d = cp.asarray(np.ascontiguousarray(np.asarray(cols).T))
    nx_d = cp.asarray(np.ascontiguousarray(n_x, dtype=np.int64))
    perms_d = cp.asarray(np.ascontiguousarray(perms, dtype=np.int64))
    out_d = cp.zeros((B, K), dtype=cp.float64)
    total = B * K
    blocks = (total + _THREADS - 1) // _THREADS
    cap = 64 if int(n_y) * (int(np.max(n_x)) if np.size(n_x) else 0) <= 64 else _MAX_CELLS
    _KERNELS[cap][blocks, _THREADS](cols_d, nx_d, perms_d, np.int64(n_y), out_d)
    best = cp.maximum(out_d.max(axis=1), 0.0)
    host: np.ndarray = cp.asnumpy(best)
    return host


__all__ = ["device_supported", "family_null_max_device"]
