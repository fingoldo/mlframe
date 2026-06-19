"""Kernel variants + dispatcher for the discrete (one-hot) interaction-propensity score.

The discrete branch of ``second_moment_propensity`` computes, summed over the K one-hot class
indicators ``1[y==c]``:

    score(x_j) = sum_c |corr(V2[:,j], 1[y==c])| + |corr(V[:,j], 1[y==c])|

where V = column matrix (n, p) and V2 = V*V. The original implementation looped over classes and
recomputed the per-column means / L2 norms of V and V2 ONCE PER CLASS (K times). Two facts make this
wasteful:

  * the per-column standardization of V and V2 is class-independent -- hoist it OUT of the class loop
    so it is done exactly once;
  * after standardizing the indicator columns too, ``|corr|`` is a single centered dot product, so the
    whole K-class sweep is ONE GEMM: Zc (p, n) @ Yc (n, K) -> (p, K), then sum |.| over K. Same for V2.

All three variants below return the IDENTICAL score vector (the numpy variant is the bit-reference;
the numba and cupy variants are validated to agree to a tight tolerance and produce the same RANKING,
which is all top-k selection consumes).

Dispatch: ``compute_discrete_score`` picks the fastest variant for the (n, p, K) work via the
kernel_tuning registry (mirrors random_features.rff_matmul); a source-default heuristic is the
fallback when no per-host sweep entry exists.

Bench (this dev box, GTX 1050 Ti; min-of-runs wall time, discrete binary path K=2; 2026-06-19,
machine under concurrent load -- absolute numbers noisy but the ratio is stable):
  p=10000, n=8000 : per-class loop 2.33s -> numpy-gemm 0.96s (2.4x); numba 3.08s.
  p=100000,n=1000 : numpy-gemm 1.16s.
The win is two parts: (1) reformulating the 2*K |corr| calls as a single (p,n)@(n,K) GEMM per matrix;
(2) standardizing V/V2 ONCE with an in-place column-scale (einsum norm, no np.where (n,p) temporaries),
which removed the dominant allocation cost cProfile flagged in _abs_col_corr. numba lost here (BLAS GEMM
beats a hand loop for the standardized matmul); it is kept as a HW-dependent option the sweep may elect.
cupy GEMM wins at large p once GPU memory is free (H2D of the (n,p) standardized matrix amortises);
the dispatcher routes to it above _DEFAULT_CUPY_WORK_THRESHOLD when a GPU is present.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Shared standardization: hoisted ONCE (was recomputed per class K times).
# ---------------------------------------------------------------------------
def _standardize_cols(M: np.ndarray) -> np.ndarray:
    """Return column-standardized copy of ``M`` (n, p): centered, unit-L2 per column.

    A constant (zero-variance) column becomes all-zeros, so its dot product with any
    indicator is 0 -> |corr| = 0, matching the reference's zero-for-constant contract.

    In-place after the initial center copy: np.where(norm>0, Mc/norm, 0) allocated three
    (n,p) temporaries per call (the dominant cost at p>=10k -- cProfile 2026-06-19); here we
    reciprocate the per-column norm (length p) once, zero the dead columns' scale, and scale
    the centered matrix in place, so only the one centered copy is held."""
    Mc = M - M.mean(axis=0)                      # one (n,p) copy (centered)
    norm = np.sqrt(np.einsum("ij,ij->j", Mc, Mc))  # (p,) column L2; einsum avoids a (n,p) square temp
    inv = np.zeros_like(norm)
    nz = norm > 0.0
    inv[nz] = 1.0 / norm[nz]                      # constant columns -> 0 scale -> zero column
    Mc *= inv                                     # broadcast (p,) over rows, in place
    return np.ascontiguousarray(Mc)


def _standardize_indicators(yf: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Build the (n, K) standardized one-hot indicator matrix for ``classes``.

    Each column is the centered, unit-L2 indicator 1[y==c]. A class present in every row
    (impossible here -- classes are distinct values) or absent would give a zero column."""
    n = yf.shape[0]
    Y = np.zeros((n, classes.size), dtype=np.float64)
    for k, c in enumerate(classes):
        Y[:, k] = (yf == c)
    Yc = Y - Y.mean(axis=0)
    norm = np.sqrt((Yc * Yc).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        Yc = np.where(norm > 0.0, Yc / norm, 0.0)
    return np.ascontiguousarray(Yc)


# ---------------------------------------------------------------------------
# Variant: numpy GEMM (default CPU; the bit-reference for parity checks).
# ---------------------------------------------------------------------------
def discrete_score_numpy(ZV: np.ndarray, ZV2: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    """Reference: two GEMMs (p,n)@(n,K) then sum |.| over the K classes.

    ZV, ZV2 are the standardized V and V2 (n, p); Yc the standardized indicators (n, K).
    corr(col, indicator) = standardized_col . standardized_indicator, so |corr| is |ZV.T @ Yc|."""
    c1 = ZV.T @ Yc          # (p, K) = sum_c corr(V[:,j], 1[y=c])
    c2 = ZV2.T @ Yc         # (p, K) = sum_c corr(V2[:,j], 1[y=c])
    return np.abs(c1).sum(axis=1) + np.abs(c2).sum(axis=1)


# ---------------------------------------------------------------------------
# Variant: numba parallel (CPU, no BLAS dependency / better at some sizes).
# ---------------------------------------------------------------------------
_NUMBA_FN = None


def _get_numba_fn():
    global _NUMBA_FN
    if _NUMBA_FN is not None:
        return _NUMBA_FN
    import numba

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _kernel(ZV, ZV2, Yc):  # ZV,ZV2 (n,p) ; Yc (n,K)
        n, p = ZV.shape
        K = Yc.shape[1]
        out = np.zeros(p, dtype=np.float64)
        for j in numba.prange(p):
            acc = 0.0
            for k in range(K):
                s1 = 0.0
                s2 = 0.0
                for i in range(n):
                    yik = Yc[i, k]
                    s1 += ZV[i, j] * yik
                    s2 += ZV2[i, j] * yik
                acc += abs(s1) + abs(s2)
            out[j] = acc
        return out

    _NUMBA_FN = _kernel
    return _kernel


def discrete_score_numba(ZV: np.ndarray, ZV2: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    fn = _get_numba_fn()
    return fn(np.ascontiguousarray(ZV), np.ascontiguousarray(ZV2), np.ascontiguousarray(Yc))


# ---------------------------------------------------------------------------
# Variant: cupy GEMM (GPU; wins at p>=100k once H2D/D2H amortised).
# ---------------------------------------------------------------------------
def discrete_score_cupy(ZV: np.ndarray, ZV2: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    import cupy as cp

    ZVd = cp.asarray(ZV)
    ZV2d = cp.asarray(ZV2)
    Ycd = cp.asarray(Yc)
    c1 = cp.abs(ZVd.T @ Ycd).sum(axis=1)
    c2 = cp.abs(ZV2d.T @ Ycd).sum(axis=1)
    return cp.asnumpy(c1 + c2)


# ---------------------------------------------------------------------------
# Dispatcher (kernel_tuning registry; mirrors random_features.rff_matmul).
# ---------------------------------------------------------------------------
_VARIANTS = {
    "numpy": discrete_score_numpy,
    "numba": discrete_score_numba,
    "cupy": discrete_score_cupy,
}

# work = n * p * K (size of the GEMM). Source-default crossover bands; replaced per-host
# after a kernel_tuning sweep. cupy wins large; numpy GEMM is fine small (numba rarely best
# given BLAS, but kept as a HW-dependent option the sweep can elect).
_DEFAULT_CUPY_WORK_THRESHOLD = 8000 * 50000 * 2  # ~ p>=50k at n=8000,K=2

_SWEEP_WORK = [8000 * 1000 * 2, 8000 * 10000 * 2, 8000 * 50000 * 2,
               8000 * 100000 * 2, 20000 * 100000 * 2]


def _gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _fallback_choice(work: int = 0, **_dims) -> str:
    if _gpu_available() and work >= _DEFAULT_CUPY_WORK_THRESHOLD:
        return "cupy"
    return "numpy"


def _make_inputs(work: int):
    """Representative standardized operands whose n*p*K ~= work (n=8000, K=2 fixed -> derive p)."""
    n, K = 8000, 2
    p = max(2, int(work / (n * K)))
    rng = np.random.default_rng(0)
    ZV = _standardize_cols(rng.standard_normal((n, p)))
    ZV2 = _standardize_cols(ZV * ZV)
    yf = (rng.random(n) < 0.5).astype(np.float64)
    Yc = _standardize_indicators(yf, np.array([0.0, 1.0]))
    return (ZV, ZV2, Yc)


def _run_sweep() -> list:
    from pyutilz.dev.benchmarking import sweep_backend_crossover

    # numba is benched-slower than the BLAS GEMM for the standardized matmul on this class of inputs
    # (it is retained as a callable variant for re-bench on BLAS-poor hardware, but excluded from the
    # default sweep so its multi-second JIT does not dominate tuning). numpy GEMM vs cupy GEMM only.
    variants = {"numpy": discrete_score_numpy}
    if _gpu_available():
        variants["cupy"] = discrete_score_cupy
    return sweep_backend_crossover(
        variants, _SWEEP_WORK, _make_inputs, "work",
        reference="numpy", repeats=3, equiv_rtol=1e-6, equiv_atol=1e-8,
    )


_SPEC = None


def _get_spec():
    global _SPEC
    if _SPEC is not None:
        return _SPEC
    try:
        from pyutilz.performance.kernel_tuning.registry import kernel_tuner

        _SPEC = kernel_tuner(
            kernel_name="fe_interaction_prerank_discrete",
            variant_fns=(discrete_score_numpy, discrete_score_numba, discrete_score_cupy),
            tuner=_run_sweep,
            axes={"work": list(_SWEEP_WORK)},
            fallback=_fallback_choice,
            gpu_capable=True,
            salt=1,
            cli_label="fe_interaction_prerank_discrete",
        )
    except Exception:
        _SPEC = False
    return _SPEC


def compute_discrete_score(V: np.ndarray, V2: np.ndarray, yf: np.ndarray,
                           classes: np.ndarray, backend: str | None = None) -> np.ndarray:
    """Standardize once + dispatch the K-class GEMM to the fastest backend for the work size.

    ``backend`` forces a specific variant ("numpy"/"numba"/"cupy") -- used by parity tests; when
    None the kernel_tuning spec (or the source-default heuristic) chooses by work = n*p*K."""
    ZV = _standardize_cols(V)
    ZV2 = _standardize_cols(V2)
    Yc = _standardize_indicators(yf, classes)
    work = int(V.shape[0]) * int(V.shape[1]) * int(classes.size)
    if backend is None:
        spec = _get_spec()
        if spec:
            try:
                backend = spec.choose(work=work)
            except Exception:
                backend = _fallback_choice(work=work)
        else:
            backend = _fallback_choice(work=work)
    fn = _VARIANTS.get(backend, discrete_score_numpy)
    try:
        return fn(ZV, ZV2, Yc)
    except Exception:
        return discrete_score_numpy(ZV, ZV2, Yc)


__all__ = [
    "compute_discrete_score",
    "discrete_score_numpy",
    "discrete_score_numba",
    "discrete_score_cupy",
]
