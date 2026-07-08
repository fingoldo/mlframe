"""Orthogonal-polynomial basis evaluation kernels for the hermite_fe pair-FE search.

Carved from the hermite_fe package facade: the per-degree Horner njit evaluators (single + prange),
the GEMV-style basis-matrix builders, and the ``build_basis_matrix`` dispatcher. Depends only on
numpy + numba; the facade re-exports every public name so importers stay unchanged.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

    def prange(n):
        return range(n)


# njit polynomial evaluators. numpy's polyval-family is C-optimized but carries Python dispatch overhead
# (~30-40us); for n~2000 with degree<=4 dispatch dominates. Empirical: njit hermeval ~12us vs numpy 46us (3.7x);
# njit legval ~10us vs numpy 64us (6.3x). Gap shrinks at n>=20k where numpy's vectorization wins.
#
# Recurrences (probabilist's variants where applicable):
# * Hermite_e (He_n): He_0=1, He_1=x, He_n = x*He_{n-1} - (n-1)*He_{n-2}
# * Legendre  (P_n) : P_0=1,  P_1=x,  P_n = ((2n-1)*x*P_{n-1} - (n-1)*P_{n-2}) / n
# * Chebyshev (T_n) : T_0=1,  T_1=x,  T_n = 2*x*T_{n-1} - T_{n-2}
# * Laguerre  (L_n) : L_0=1,  L_1=1-x, L_n = ((2n-1-x)*L_{n-1} - (n-1)*L_{n-2}) / n
#
# Single-thread perf note (2026-06-24, bench_polyeval_single_thread_register.py):
# These keep the per-degree array form (each fixed-coefficient inner i-loop SIMD-vectorizes), with
# the prologue FUSED: no x.copy() (P_1 == x is read-only here) and the out[i]=c[0] + out[i]+=c[1]*x[i]
# passes collapsed into one. Bit-identical, 1.13-1.39x over the prior form across n=500..50k.
# bench-attempt-rejected: the register single-pass form used by the *_parallel variants (scalar
# p_prev/p_curr, serial k-recurrence per element) is SLOWER single-thread (0.52-0.65x) -- it blocks
# vectorization -- and not bit-identical (fma reassociation). It only wins once prange spreads it
# across cores (the existing _parallel path). Do not re-try the register rewrite for n<50k.


@njit(cache=True, fastmath=True)
def _hermeval_njit(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    c0 = c[0]
    if nc == 1:
        for i in range(n):
            out[i] = c0
        return out
    c1 = c[1]
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = x
    for i in range(n):
        out[i] = c0 + c1 * x[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        km1 = k - 1
        for i in range(n):
            p_next[i] = x[i] * p_curr[i] - km1 * p_prev[i]
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


@njit(cache=True, fastmath=True)
def _legval_njit(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    c0 = c[0]
    if nc == 1:
        for i in range(n):
            out[i] = c0
        return out
    c1 = c[1]
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = x
    for i in range(n):
        out[i] = c0 + c1 * x[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        inv_k = 1.0 / k
        two_km1 = 2 * k - 1
        km1 = k - 1
        for i in range(n):
            p_next[i] = (two_km1 * x[i] * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


@njit(cache=True, fastmath=True)
def _chebval_njit(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    c0 = c[0]
    if nc == 1:
        for i in range(n):
            out[i] = c0
        return out
    c1 = c[1]
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = x
    for i in range(n):
        out[i] = c0 + c1 * x[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        for i in range(n):
            p_next[i] = 2.0 * x[i] * p_curr[i] - p_prev[i]
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


@njit(cache=True, fastmath=True)
def _lagval_njit(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    c0 = c[0]
    if nc == 1:
        for i in range(n):
            out[i] = c0
        return out
    c1 = c[1]
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = np.empty(n, dtype=np.float64)
    for i in range(n):
        pc = 1.0 - x[i]
        p_curr[i] = pc
        out[i] = c0 + c1 * pc
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        inv_k = 1.0 / k
        two_km1 = 2 * k - 1
        km1 = k - 1
        for i in range(n):
            p_next[i] = ((two_km1 - x[i]) * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


# Parallel-prange variants of the polynomial evaluators. Per-element Horner recurrence runs in registers
# (no intermediate p_prev / p_curr arrays), so prange over array elements scales linearly with cores at the
# cost of recomputing the recurrence per element. Wins for n >= 50k where memory bandwidth + thread-spawn
# overhead is amortised.


@njit(cache=True, fastmath=True, parallel=True)
def _hermeval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            p_next = xi * p_curr - (k - 1) * p_prev
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _legval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = (two_km1 * xi * p_curr - km1 * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _chebval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            p_next = 2.0 * xi * p_curr - p_prev
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _lagval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = 1.0 - xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = ((two_km1 - xi) * p_curr - km1 * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


# 2026-05-18 Basis-matrix builders for GEMV-style polynomial evaluation.
#
# Per-iteration Horner (above) recomputes the polynomial basis ``T_k(x[i])``
# from scratch for every (coef_a, coef_b) trial inside CMA-ES. That's
# wasteful when the SAME ``z_a`` / ``z_b`` arrays are evaluated against
# many different coef vectors: the basis values ``T_k(z[i])`` are coef-
# independent.
#
# These builders compute the full basis matrix ``B[i, k] = T_k(z[i])`` of
# shape ``(n, max_degree + 1)`` ONCE per (pair, basis). Then per-trial
# evaluation reduces to a BLAS GEMV ``h = B[:, :len(c)] @ c`` - typically
# 2-5x faster than custom njit Horner because BLAS is decades-tuned
# (SIMD, cache blocking).
#
# Memory cost: at n=200_000 and max_degree=8, B is 200k x 9 = 14 MB
# float64 - negligible. At the multi-fidelity 1500-sample inner-search
# path it's ~110 KB.


@njit(cache=True, fastmath=True, parallel=True)
def _build_basis_hermite(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Build ``B[i, k] = He_k(x[i])`` for k=0..max_degree using the He recurrence."""
    n = x.shape[0]
    nc = max_degree + 1
    B = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        B[i, 0] = 1.0
        if nc > 1:
            B[i, 1] = xi
            for k in range(2, nc):
                B[i, k] = xi * B[i, k - 1] - (k - 1) * B[i, k - 2]
    return B


@njit(cache=True, fastmath=True, parallel=True)
def _build_basis_legendre(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Build ``B[i, k] = P_k(x[i])`` via Bonnet's recurrence."""
    n = x.shape[0]
    nc = max_degree + 1
    B = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        B[i, 0] = 1.0
        if nc > 1:
            B[i, 1] = xi
            for k in range(2, nc):
                inv_k = 1.0 / k
                two_km1 = 2 * k - 1
                km1 = k - 1
                B[i, k] = (two_km1 * xi * B[i, k - 1] - km1 * B[i, k - 2]) * inv_k
    return B


@njit(cache=True, fastmath=True, parallel=True)
def _build_basis_chebyshev(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Build ``B[i, k] = T_k(x[i])`` via T_{k+1} = 2x*T_k - T_{k-1}."""
    n = x.shape[0]
    nc = max_degree + 1
    B = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        two_xi = 2.0 * xi
        B[i, 0] = 1.0
        if nc > 1:
            B[i, 1] = xi
            for k in range(2, nc):
                B[i, k] = two_xi * B[i, k - 1] - B[i, k - 2]
    return B


@njit(cache=True, fastmath=True, parallel=True)
def _build_basis_laguerre(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Build ``B[i, k] = L_k(x[i])`` via the Laguerre recurrence."""
    n = x.shape[0]
    nc = max_degree + 1
    B = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        B[i, 0] = 1.0
        if nc > 1:
            B[i, 1] = 1.0 - xi
            for k in range(2, nc):
                inv_k = 1.0 / k
                two_km1 = 2 * k - 1
                km1 = k - 1
                B[i, k] = ((two_km1 - xi) * B[i, k - 1] - km1 * B[i, k - 2]) * inv_k
    return B


_BASIS_BUILDERS = {
    "hermite": _build_basis_hermite,
    "legendre": _build_basis_legendre,
    "chebyshev": _build_basis_chebyshev,
    "laguerre": _build_basis_laguerre,
}


def build_basis_matrix(basis: str, z: np.ndarray, max_degree: int) -> np.ndarray:
    """Public dispatcher: returns ``B[i, k] = T_k(z[i])`` for the named basis.

    Raises ``KeyError`` if the basis is not one of the orthogonal-polynomial
    family supported here (hermite / legendre / chebyshev / laguerre).
    Factory-based bases (RBF / Sigmoid / Fourier / Pade) need per-feature
    eval closures and are NOT compatible with basis-matrix caching - the
    caller checks via ``basis_info.get('eval_njit_factory') is None``
    before precomputing.
    """
    builder = _BASIS_BUILDERS.get(basis)
    if builder is None:
        raise KeyError(
            f"build_basis_matrix: basis {basis!r} not in " f"{sorted(_BASIS_BUILDERS)}; factory-based bases must use " f"the per-call eval_func path."
        )
    z_c = np.ascontiguousarray(z, dtype=np.float64)
    return np.asarray(builder(z_c, int(max_degree)))
