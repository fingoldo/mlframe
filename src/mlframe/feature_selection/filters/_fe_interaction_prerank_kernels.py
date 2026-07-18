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

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


# ---------------------------------------------------------------------------
# MEASURED-AND-CACHED LightGBM-fit cost predictor for the "auto" pre-rank gate.
# ---------------------------------------------------------------------------
# The "auto" criterion in top_k_by_interaction_propensity uses the high-recall ``fused``
# criterion (which pays one full LightGBM fit over all p candidate columns) ONLY when that
# fit is affordable for the (n_rows, n_candidates) at hand, else it falls back to the cheap
# ``second_moment`` score. Affordability is a PREDICTED fit wall-time vs a budget; the
# throughput model below mirrors _fe_synergy_exhaustive (measured_pairs_per_second /
# warm_*_cache / predict_*_seconds): the gbm cols/sec is MEASURED on a small synthetic cell
# and cached per host via pyutilz kernel_tuning_cache, NEVER hardcoded into the decision.
#
# Bench (this dev box, GTX 1050 Ti host, LightGBM CPU fit, num_boost_round=100, binary;
# 2026-06-19, machine under concurrent load): ~16.8 s @ p=2000 / n=8000, ~87.7 s @ p=10000
# / n=8000 -> roughly O(p) at fixed n (the per-column split search dominates). From the two
# points: cols/sec ~ (10000-2000)/(87.7-16.8) ~= 113 cols/s with a small fixed offset. We
# model fit_seconds ~ p / cols_per_second(n) and store cols_per_second per (n) region. The
# ~113 cols/s figure is ONLY the cold-cache analytic FALLBACK (per feedback_use_kernel_
# tuning_cache_for_gpu: the live path measures-and-caches per (n) below).
_GBM_FALLBACK_COLS_PER_SEC = 113.0  # cold-cache analytic fallback (bench 2026-06-19, n=8000)
_GBM_FALLBACK_FIXED_SEC = 2.0  # constant fit overhead (dataset build + boosting setup)
_GBM_COST_SWEEP_N_SAMPLES = [2000, 8000, 32000]
_GBM_COST_SALT = 1
_GBM_BENCH_P = 256  # small synthetic cell width (kept tiny so the warm-up bench is ~sub-second)
_GBM_BENCH_ROUNDS = 100


def _measure_gbm_cols_per_second(n_samples: int) -> float:
    """Time a real LightGBM fit on a small synthetic (n_samples, _GBM_BENCH_P) binary cell and
    return achieved cols/second (p / fit_seconds, fixed-offset corrected). Used as the
    kernel_tuning tuner body; raises on any failure so the orchestrator records the fallback."""
    import lightgbm as lgb

    n = int(n_samples)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, _GBM_BENCH_P))
    y = (rng.random(n) < 0.5).astype(np.float64)
    params = dict(objective="binary", num_leaves=31, learning_rate=0.1, verbose=-1, min_child_samples=20, feature_fraction=1.0)
    ds = lgb.Dataset(X, label=y)
    t0 = time.perf_counter()
    lgb.train(params, ds, num_boost_round=_GBM_BENCH_ROUNDS)
    dt = time.perf_counter() - t0
    eff = dt - _GBM_FALLBACK_FIXED_SEC
    if eff <= 0.0:
        return float(_GBM_FALLBACK_COLS_PER_SEC)
    cps = float(_GBM_BENCH_P) / eff
    if cps <= 0.0:
        return float(_GBM_FALLBACK_COLS_PER_SEC)
    return cps


def measured_gbm_cols_per_second(n_samples: int) -> tuple[float, str]:
    """Per-host measured LightGBM-fit throughput (cols/s) for ``n_samples`` rows, looked up
    from the kernel_tuning_cache (measured on first miss via ``warm_gbm_cost_cache``). Returns
    ``(cols_per_second, source)`` where source is "cache" | "fallback".

    NEVER hardcodes the throughput in the gate decision -- the ~113 cols/s figure is only the
    cold-cache analytic fallback (mirrors _fe_synergy_exhaustive.measured_pairs_per_second)."""
    try:
        from ._kernel_tuning import get_kernel_tuning_cache

        cache = get_kernel_tuning_cache()
        if cache is not None:
            region = cache.lookup("fe_interaction_prerank_gbm_cost", n_samples=int(n_samples))
            if region is not None:
                val = region.get("value", region.get("choice"))
                if val is not None:
                    try:
                        cps = float(val)
                        if cps > 0:
                            return cps, "cache"
                    except (TypeError, ValueError):
                        pass
    except Exception as exc:  # cache miss / pyutilz unavailable -> fallback
        logger.debug("gbm-cost cache lookup failed (%s: %s); using fallback", type(exc).__name__, exc)
    return float(_GBM_FALLBACK_COLS_PER_SEC), "fallback"


def warm_gbm_cost_cache() -> None:
    """Populate the per-host gbm-fit cost cache via ``get_or_tune`` (one-time, async-safe).
    Best effort: callers may skip this and rely on ``measured_gbm_cols_per_second``'s fallback.
    No-op if LightGBM or pyutilz is unavailable."""
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        logger.debug("lightgbm unavailable, skipping gbm-cost cache warm (%s: %s)", type(exc).__name__, exc)
        return
    try:
        from ._kernel_tuning import get_kernel_tuning_cache

        cache = get_kernel_tuning_cache()
        if cache is None:
            return

        def _tuner() -> list:
            """kernel_tuning tuner body: measure gbm cols/sec at each sweep n_samples, falling back per-point on failure."""
            regions = []
            for n in _GBM_COST_SWEEP_N_SAMPLES:
                try:
                    cps = _measure_gbm_cols_per_second(n)
                except Exception as exc:
                    logger.debug("gbm-cost measurement failed at n_samples=%d, using fallback (%s: %s)", n, type(exc).__name__, exc)
                    cps = float(_GBM_FALLBACK_COLS_PER_SEC)
                regions.append({"n_samples": n, "value": cps})
            return regions

        cache.get_or_tune(
            "fe_interaction_prerank_gbm_cost",
            dims={"n_samples": _GBM_COST_SWEEP_N_SAMPLES[0]},
            tuner=_tuner,
            axes=["n_samples"],
            fallback=lambda n_samples: _GBM_FALLBACK_COLS_PER_SEC,
            salt=_GBM_COST_SALT,
            once_per_process=True,
        )
    except Exception as exc:
        logger.debug("gbm-cost cache warm failed (%s: %s)", type(exc).__name__, exc)


def predict_gbm_fit_seconds(n_samples: int, n_candidates: int) -> tuple[float, float, str]:
    """Predicted LightGBM-fit wall-time (seconds) for one ``fused`` booster fit over
    ``n_candidates`` columns at ``n_samples`` rows. Returns
    ``(predicted_seconds, cols_per_second, throughput_source)``.

    Model: fit_seconds ~ fixed_overhead + n_candidates / cols_per_second(n_samples), with
    cols_per_second measured-and-cached per host (analytic fallback on a cold cache)."""
    cps, source = measured_gbm_cols_per_second(n_samples)
    if cps <= 0:
        cps = float(_GBM_FALLBACK_COLS_PER_SEC)
    predicted = float(_GBM_FALLBACK_FIXED_SEC) + float(n_candidates) / cps
    return predicted, cps, source


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
    Mc = M - M.mean(axis=0)  # one (n,p) copy (centered)
    norm = np.sqrt(np.einsum("ij,ij->j", Mc, Mc))  # (p,) column L2; einsum avoids a (n,p) square temp
    inv = np.zeros_like(norm)
    nz = norm > 0.0
    inv[nz] = 1.0 / norm[nz]  # constant columns -> 0 scale -> zero column
    Mc *= inv  # broadcast (p,) over rows, in place
    return np.ascontiguousarray(Mc)


def _standardize_indicators(yf: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Build the (n, K) standardized one-hot indicator matrix for ``classes``.

    Each column is the centered, unit-L2 indicator 1[y==c]. A class present in every row
    (impossible here -- classes are distinct values) or absent would give a zero column."""
    n = yf.shape[0]
    Y = np.zeros((n, classes.size), dtype=np.float64)
    for k, c in enumerate(classes):
        Y[:, k] = yf == c
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
    c1 = ZV.T @ Yc  # (p, K) = sum_c corr(V[:,j], 1[y=c])
    c2 = ZV2.T @ Yc  # (p, K) = sum_c corr(V2[:,j], 1[y=c])
    return np.asarray(np.abs(c1).sum(axis=1) + np.abs(c2).sum(axis=1))


# ---------------------------------------------------------------------------
# Variant: numba parallel (CPU, no BLAS dependency / better at some sizes).
# ---------------------------------------------------------------------------
_NUMBA_FN = None


def _get_numba_fn():
    """Lazily compile and cache the parallel numba kernel (module-level singleton; import + JIT paid once)."""
    global _NUMBA_FN
    if _NUMBA_FN is not None:
        return _NUMBA_FN
    import numba

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _kernel(ZV, ZV2, Yc):  # ZV,ZV2 (n,p) ; Yc (n,K)
        """Per-column-parallel accumulation of sum_c(|corr(V,1[y=c])| + |corr(V2,1[y=c])|) via explicit dot-product loops (no BLAS)."""
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
    """numba parallel-prange variant of ``discrete_score_numpy``; no BLAS dependency, retained as a HW-dependent sweep option."""
    fn = _get_numba_fn()
    return np.asarray(fn(np.ascontiguousarray(ZV), np.ascontiguousarray(ZV2), np.ascontiguousarray(Yc)))


# ---------------------------------------------------------------------------
# Variant: cupy GEMM (GPU; wins at p>=100k once H2D/D2H amortised).
# ---------------------------------------------------------------------------
def discrete_score_cupy(ZV: np.ndarray, ZV2: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    """cupy GEMM variant of ``discrete_score_numpy``; H2D-uploads the standardized operands then mirrors the numpy math on GPU."""
    import cupy as cp

    ZVd = cp.asarray(ZV)
    ZV2d = cp.asarray(ZV2)
    Ycd = cp.asarray(Yc)
    c1 = cp.abs(ZVd.T @ Ycd).sum(axis=1)
    c2 = cp.abs(ZV2d.T @ Ycd).sum(axis=1)
    return np.asarray(cp.asnumpy(c1 + c2))


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

_SWEEP_WORK = [8000 * 1000 * 2, 8000 * 10000 * 2, 8000 * 50000 * 2, 8000 * 100000 * 2, 20000 * 100000 * 2]


def _gpu_available() -> bool:
    """True iff cupy is importable and at least one CUDA device is visible."""
    try:
        import cupy as cp

        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception as exc:
        logger.debug("cupy/CUDA probe failed, treating as no GPU available (%s: %s)", type(exc).__name__, exc)
        return False


def _fallback_choice(work: int = 0, **_dims) -> str:
    """Source-default backend heuristic (cache miss / no pyutilz): cupy above the work threshold on a GPU host, else numpy."""
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
    """kernel_tuning tuner body: bench numpy-vs-cupy across ``_SWEEP_WORK`` sizes and return the crossover regions."""
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


_SPEC: "TunerSpec | bool | None" = None


def _get_spec():
    """Lazily build and cache the kernel_tuning ``KernelSpec`` (module-level singleton); ``False`` if pyutilz is unavailable, signalling callers to use ``_fallback_choice``."""
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
    except Exception as exc:
        logger.debug("kernel_tuning registry unavailable (%s: %s); using source-default backend heuristic", type(exc).__name__, exc)
        _SPEC = False
    return _SPEC


def compute_discrete_score(V: np.ndarray, V2: np.ndarray, yf: np.ndarray, classes: np.ndarray, backend: str | None = None) -> np.ndarray:
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
            except Exception as exc:
                # Hot per-call dispatch path: debug-only, no per-call warning spam.
                logger.debug("kernel_tuning backend choice failed, using heuristic fallback (%s: %s)", type(exc).__name__, exc)
                backend = _fallback_choice(work=work)
        else:
            backend = _fallback_choice(work=work)
    fn = _VARIANTS.get(backend, discrete_score_numpy)
    try:
        return fn(ZV, ZV2, Yc)
    except Exception as exc:
        # Hot per-call dispatch path: debug-only, no per-call warning spam.
        logger.debug("backend %r kernel failed, falling back to numpy variant (%s: %s)", backend, type(exc).__name__, exc)
        return discrete_score_numpy(ZV, ZV2, Yc)


__all__ = [
    "compute_discrete_score",
    "discrete_score_numpy",
    "discrete_score_numba",
    "discrete_score_cupy",
    "measured_gbm_cols_per_second",
    "warm_gbm_cost_cache",
    "predict_gbm_fit_seconds",
]
