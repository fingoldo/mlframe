"""Fused (n, n_tau) mask/select build kernels for the conditional-gate scan.

Carved out of ``_conditional_gate_fe`` purely for file size: that module crossed the 1000-LOC budget, and these
two kernels are the largest block in it with no dependency on anything the parent defines -- they take arrays
and return an array. Re-exported from the parent, which is the only caller, so import sites are unchanged.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, parallel=True, fastmath=False)
def _gate_mask_grid_njit(cv, av, taus):
    """Fused (n, n_tau) mask block: ``feats[i, j] = av[i] * (cv[i] > taus[j])``. The off-branch value ``a * 0.0``
    preserves the NaN semantics of the numpy ``av * (cv > tau)`` form (0*NaN=NaN), so the kernel is bit-identical incl NaN."""
    n = cv.shape[0]; k = taus.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for i in numba.prange(n):
        c = cv[i]; a = av[i]; off = a * 0.0
        for j in range(k):
            out[i, j] = a if c > taus[j] else off
    return out


@numba.njit(cache=True, parallel=True, fastmath=False)
def _gate_select_grid_njit(cv, av, bv, taus):
    """Fused (n, n_tau) select block: ``feats[i, j] = av[i] if cv[i] > taus[j] else bv[i]``. Pure gather (no arithmetic),
    so bit-identical to the numpy ``np.where(cv > tau, av, bv)`` form incl NaN operands."""
    n = cv.shape[0]; k = taus.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for i in numba.prange(n):
        c = cv[i]; a = av[i]; b = bv[i]
        for j in range(k):
            out[i, j] = a if c > taus[j] else b
    return out

# The per-candidate (n, 17-tau) mask/select build is fused into one njit(parallel) kernel ABOVE ``_GATE_BUILD_NJIT_MIN_N``
# (default 20000), numpy per-tau loop below. History: the fusion was first e2e-rejected at SMALL n (2026-06-13, iter53):
# isolated njit(parallel) won 1.26-2.46x over the numpy loop at n=533..12000 but LOST end-to-end (whole-scan njit 0.89-0.90x)
# because at small n the build is a tiny fraction of the scan and the kernel's prange contends with the MI prange
# (``_gate_grid_mi``). At LARGE n that flips: a full-suite profile put ``_build_feats`` at 22s tottime / 2058 calls (n=40k)
# and a paired end-to-end A/B of ``cheap_conditional_gate_scan`` @n=40k measured numpy 22.3s vs njit 20.8s = 1.07x, 3/3 wins,
# scan output BIT-IDENTICAL (705 hits both). So the fusion is gated ON only where the build is a real fraction of the scan.
# The mask off-value is ``a*0.0`` (not 0.0) to preserve the numpy off-region NaN semantics. A numpy broadcast build
# (``cv[:,None]>taus[None,:]``) was rejected outright (0.65x@n12000, the (n,17) bool temp blows cache).
# bench: _benchmarks/bench_gate_grid_njit.py.
