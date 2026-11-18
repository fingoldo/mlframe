"""Source-side collinear-column dedup for the orthogonal-univariate FE stage.

Carved out of ``_orthogonal_univariate_fe/__init__.py`` (2026-06-22, monolith-split: the package facade
re-exports ``_dedup_collinear_source_cols`` from its bottom). Self-contained - depends only on numpy/pandas.
"""
from __future__ import annotations

from mlframe.feature_selection.filters._fe_frame_ops import fe_column_is_numeric, fe_column_values
import contextlib
import logging
import threading
from typing import Any, Iterator, Optional, Sequence

import numpy as np
from .._fe_gpu_batch._devices import crit_float_dtype
from .._fe_usability_signal import _crit_np_dtype

from mlframe.utils.env_flags import env_int
from mlframe.utils.log_throttle import log_throttle
import pandas as pd
from numba import njit, prange

logger = logging.getLogger(__name__)

#: below this many rows per tile, `_pc_corr_cupy`'s room-bound tiling pays more in per-tile transfer/sync
#: overhead than the matmul work is worth; env-overridable for the kernel-tuning-cache style per-host tuning other
#: dedup backends already use.
_MIN_CUPY_DEDUP_TILE = env_int("MLFRAME_FE_DEDUP_CUPY_MIN_TILE", 32)

# The dedup is a DECISION (drop near-duplicate source columns), not a final statistic, so the correlations do not need
# every row: a Pearson |r| estimate from a large row sample has standard error ~ (1-r^2)/sqrt(m), far under the slack of
# a 0.999 threshold at m=100k. Capping the rows keeps the pass O(P^2 * cap) instead of O(P^2 * N) and - critical on
# mlframe's 100+ GB frames - bounds the transient (P, cap) work matrices regardless of N. Only engages when N > cap, so
# every small-n path (all tests) is byte-for-byte unchanged. Override via MLFRAME_FE_DEDUP_MAX_CORR_ROWS.
import os as _os

_MAX_CORR_ROWS = 100000  # default; MLFRAME_FE_DEDUP_MAX_CORR_ROWS is read on every call


def _dedup_sample_idx(n_rows: int) -> Optional[np.ndarray]:
    """Deterministic row-sample index for the correlation pass (see ``_MAX_CORR_ROWS``), factored out so the
    fit-scoped memo's cache-key hash (:func:`_dedup_cache_key`) samples the EXACT same rows the correlation
    pass itself would - same seed, same ``n_rows`` -> same draw both times."""
    cap = env_int("MLFRAME_FE_DEDUP_MAX_CORR_ROWS", _MAX_CORR_ROWS, minimum=2)
    if n_rows > cap:
        return np.sort(np.random.default_rng(0).choice(n_rows, size=cap, replace=False))
    return None


# Fit-scoped memo for _dedup_collinear_source_cols: mirrors heavy_tail_memo_scope() in
# hermite_fe._hermite_robust. Five independent opt-in FE-family builders (orth / extra-basis / gpu-resident /
# wavelet / hinge) each call this dedup on their own candidate columns; a "kitchen sink" config with several
# families on would otherwise repeat the identical O(P^2) corrcoef pass 2-5x within ONE MRMR.fit() call. The
# memo is OFF by default (cache is None) so every call outside an explicit dedup_collinear_memo_scope() is
# byte-for-byte unchanged. threading.local -> no cross-worker contamination.
_DEDUP_MEMO_LOCK = threading.Lock()
_DEDUP_MEMO: dict = {"cache": None, "depth": 0}


@contextlib.contextmanager
def dedup_collinear_memo_scope() -> Iterator[None]:
    """Enable the fit-scoped ``_dedup_collinear_source_cols`` memo for one MRMR.fit() call, then clear it.

    The memo is shared by every thread of the process while a scope is open (the FE stage cascades run on worker threads, and a thread-local memo was invisible to them: each of about seven
    families then repeated the classification and the correlation pass, 1.9 s of a 100000 x 100 fit). It is keyed by content, so threads of different fits can share it safely; nested and
    concurrent scopes are counted and the last one to close clears it."""
    with _DEDUP_MEMO_LOCK:
        if _DEDUP_MEMO["depth"] == 0:
            _DEDUP_MEMO["cache"] = {}
        _DEDUP_MEMO["depth"] += 1
    try:
        yield
    finally:
        with _DEDUP_MEMO_LOCK:
            _DEDUP_MEMO["depth"] -= 1
            if _DEDUP_MEMO["depth"] == 0:
                _DEDUP_MEMO["cache"] = None


def _dedup_cache_key(X: pd.DataFrame, cols: Sequence[str], corr_threshold: float) -> tuple:
    """Content-hash key for the fit-scoped dedup memo: per-column ``(name, shape, content hash)`` over the
    SAME row-capped sample the correlation pass itself uses (:func:`_dedup_sample_idx`), so two calls that
    would run the identical corrcoef pass collapse to one cache entry regardless of which DataFrame OBJECT
    each FE-family caller built its view from. Non-numeric / missing columns always pass through
    unconditionally (Pass 1 below never reads their values), so only their identity needs to be in the key."""
    from .._fe_resident_operands import _content_hash

    n_rows = len(X)
    sample_idx = _dedup_sample_idx(n_rows)
    parts: list[tuple] = []
    for c in cols:
        if not fe_column_is_numeric(X, c):
            parts.append((c, "non_numeric"))
            continue
        arr = np.asarray(X[c].to_numpy(), dtype=_crit_np_dtype())
        if sample_idx is not None:
            arr = arr[sample_idx]
        arr = np.ascontiguousarray(arr)
        parts.append((c, arr.shape, _content_hash(arr)))
    return (tuple(parts), float(corr_threshold))


def _pc_corr_numpy(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """numpy(BLAS) backend: 6 masked matmuls. Multithreaded via the platform BLAS; the CPU default (see the dispatcher)."""
    Qm = np.isfinite(Q)
    Rm = np.isfinite(R)
    Q0 = np.where(Qm, Q, 0.0)
    R0 = np.where(Rm, R, 0.0)
    Qmf = Qm.astype(_crit_np_dtype())
    Rmf = Rm.astype(_crit_np_dtype())
    n = Qmf @ Rmf.T
    Sx = Q0 @ Rmf.T
    Sy = Qmf @ R0.T
    Sxx = (Q0 * Q0) @ Rmf.T
    Syy = Qmf @ (R0 * R0).T
    Sxy = Q0 @ R0.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = Sxy - Sx * Sy / n
        varx = Sxx - Sx * Sx / n
        vary = Syy - Sy * Sy / n
        corr = np.abs(cov / np.sqrt(varx * vary))
    # nan -> 'not a duplicate' for the caller. Mirror the legacy per-pair skips: <8 common rows, or either side
    # near-constant on the common rows (std <= 1e-12 => var <= 1e-24) - otherwise a 0/~0 ratio could read as inf >= thr.
    corr[(n < 8) | (varx <= 1e-24) | (vary <= 1e-24)] = np.nan
    return np.asarray(corr)


@njit(parallel=True, cache=True, nogil=True)  # NO fastmath: the kernel relies on isfinite() for the NaN mask; nnan/ninf would elide it.
def _pc_corr_njit(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """njit(parallel) backend: one fused pass per pair, no temporaries. MEASURED SLOWER than numpy (0.23-0.80x) - a
    naive triple-loop cannot beat BLAS - so it is never auto-selected; kept as a force-selectable option per
    REJECTED != DELETED. See ``_benchmarks/bench_pairwise_complete_corr.py``."""
    q = Q.shape[0]
    r = R.shape[0]
    n = Q.shape[1]
    # Intentional float64: accumulator kept f64 (statistic / count sums need full precision)
    out = np.empty((q, r), dtype=np.float64)
    for i in prange(q):
        for j in range(r):
            cnt = 0
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            for k in range(n):
                a = Q[i, k]
                b = R[j, k]
                if np.isfinite(a) and np.isfinite(b):
                    cnt += 1
                    sx += a
                    sy += b
                    sxx += a * a
                    syy += b * b
                    sxy += a * b
            if cnt < 8:
                out[i, j] = np.nan
            else:
                inv = 1.0 / cnt
                cov = sxy - sx * sy * inv
                vx = sxx - sx * sx * inv  # moment-ok: Q was row-centred by _pairwise_complete_abs_corr before dispatch
                vy = syy - sy * sy * inv  # moment-ok: R was row-centred by _pairwise_complete_abs_corr before dispatch
                out[i, j] = np.nan if (vx <= 1e-24 or vy <= 1e-24) else abs(cov / np.sqrt(vx * vy))
    return out


def _pc_corr_cupy(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """cupy(CUDA) backend: the same 6 masked matmuls on the GPU. Solo-benched 2.0-2.7x over numpy in a moderate band but
    LOSES at large P*n (GPU memory pressure) - and, like the plug-in MI dispatcher, a solo win is eroded under joblib-
    worker GPU contention, so it is not the auto-default. OOM / no-cupy -> the dispatcher falls back to numpy."""
    import cupy as cp

    # Q/R (the partial-NaN candidate block and its dense/partial comparison block) can recur with IDENTICAL
    # content across the <=6 _dedup_collinear_source_cols calls/fit (independent FE families - orth/
    # extra-basis/gpu-resident/wavelet/hinge - each deduping their own candidate columns against the SAME
    # underlying source frame): resident-cache the upload so a repeat-content call hits instead of re-
    # uploading. When Q/R genuinely differ per call (the common case, since each family's candidate set
    # differs), this is a plain cache miss -> a fresh upload, identical cost to the old raw cp.asarray - so
    # the fix is correctness-neutral either way, only a possible win when content does recur. No ``dtype=``
    # override: Q/R keep their CALLER dtype (f32 under MLFRAME_CRIT_DTYPE_RELAXED), matching the original
    # ``cp.asarray(Q)``'s native-dtype passthrough exactly (unlike the njit backend, which explicitly
    # upcasts to f64) - forcing double precision here would change the matmul precision, not just cache it.
    from .._fe_resident_operands import resident_operand
    from .._device_sort_safe import device_room

    q, n_obs = Q.shape
    r = R.shape[0]
    item = int(Q.dtype.itemsize)
    # the 6 masked matmuls hold about 5 (rows, n) operand arrays per side and ~9 (q, r) result arrays; whole when that fits a quarter of the room, else in (row block x row block) tiles
    need = 5 * (q + r) * n_obs * item + 9 * q * r * item
    room = device_room(cp)
    if need <= room // 4:
        return _pc_corr_cupy_arrays(cp, resident_operand(Q, "orth_dedup_Q"), resident_operand(R, "orth_dedup_R"))
    tile = int(max(1, min(max(q, r), (room // 4 - 9 * 1024 * 1024 * item) // max(1, 10 * n_obs * item))))
    # a tight pool (little free VRAM, or several FE families contending for the same card at a large column
    # count) can shrink the tile below a floor where the loop below would launch thousands of tiny cupy calls instead
    # of a handful - each paying its own host<->device transfer and sync, far slower than one numpy(BLAS) call on the
    # same data. Bail to the dispatcher's existing cupy-failure fallback (numpy) rather than thrash through that many
    # tiles; _MIN_TILE is the tile width below which the per-tile overhead dominates the actual matmul work.
    if tile < _MIN_CUPY_DEDUP_TILE:
        raise RuntimeError(f"pairwise-corr tile shrank to {tile} (room-bound); too fragmented to pay off on the device")
    out = np.empty((q, r), dtype=np.float64)  # Intentional float64: the result contract of the numpy backend
    # R does not change across the outer (i) loop, so its tiles upload ONCE here rather than once per i - the inner
    # loop below used to re-upload the same R tile q // tile times.
    r_tiles = [cp.asarray(np.ascontiguousarray(R[j : j + tile])) for j in range(0, r, tile)]
    for i in range(0, q, tile):
        Qd = cp.asarray(np.ascontiguousarray(Q[i : i + tile]))
        for Rd, j in zip(r_tiles, range(0, r, tile)):
            out[i : i + tile, j : j + tile] = _pc_corr_cupy_arrays(cp, Qd, Rd)
    return out


_FINAL_KERNELS: dict = {}
_FINAL_KERNELS_LOCK = threading.Lock()


def _corr_final_kernel(cp: Any) -> Any:
    """One fused elementwise pass for the last stage of the masked-matmul correlation: covariance and both variances from the six sums, ``|r|``, and ``nan`` where fewer than 8 common rows or either
    side is near-constant (the 12 or so cupy elementwise launches of the unfused form, one launch here)."""
    with _FINAL_KERNELS_LOCK:
        ker = _FINAL_KERNELS.get("final")
        if ker is None:
            ker = _FINAL_KERNELS["final"] = _build_corr_final_kernel(cp)
    return ker


def _build_corr_final_kernel(cp: Any) -> Any:
    """The elementwise kernel of :func:`_corr_final_kernel`."""
    return cp.ElementwiseKernel(
        "S n, T Sx, T Sy, T Sxx, T Syy, T Sxy",
        "T out",
        """
        const T cov = Sxy - Sx * Sy / n;
        const T vx = Sxx - Sx * Sx / n;
        const T vy = Syy - Sy * Sy / n;
        out = (n < 8 || vx <= 1e-24 || vy <= 1e-24) ? (T)nan("") : (T)fabs(cov / sqrt(vx * vy));
        """,
        "pc_corr_final",
    )


def _pc_corr_cupy_arrays(cp: Any, Qd: Any, Rd: Any) -> np.ndarray:
    """The masked-matmul correlation of two resident blocks ``(q, n)`` and ``(r, n)``: six explicit cuBLAS GEMMs and one fused elementwise pass for the rest."""
    Qm = cp.isfinite(Qd)
    Rm = cp.isfinite(Rd)
    Q0 = cp.where(Qm, Qd, 0.0)
    R0 = cp.where(Rm, Rd, 0.0)
    mask_dtype = Qd.dtype if (Qd.dtype == Rd.dtype and Qd.dtype in (cp.float32, cp.float64)) else crit_float_dtype()  # the masks in the data's own dtype: a float32 mask against float64 data was copied to float64 inside each of three GEMMs  # Intentional float64: dtype check: float64 is a supported column type here, nothing is computed in it
    Qmf = Qm.astype(mask_dtype)
    Rmf = Rm.astype(mask_dtype)
    n = cp.matmul(Qmf, Rmf.T)
    Sx = cp.matmul(Q0, Rmf.T)
    Sy = cp.matmul(Qmf, R0.T)
    Sxx = cp.matmul(Q0 * Q0, Rmf.T)
    Syy = cp.matmul(Qmf, (R0 * R0).T)
    Sxy = cp.matmul(Q0, R0.T)
    corr = _corr_final_kernel(cp)(n, Sx.astype(Sxy.dtype, copy=False), Sy.astype(Sxy.dtype, copy=False), Sxx.astype(Sxy.dtype, copy=False), Syy.astype(Sxy.dtype, copy=False), Sxy)
    return np.asarray(cp.asnumpy(corr))


def _row_center_finite(A: np.ndarray) -> np.ndarray:
    """Subtract each row's own finite mean, so the three backends' ``Sxx - Sx*Sx/n`` variance stays conditioned.

    Pearson r is translation-invariant, so shifting a row by ANY constant leaves the true correlation exactly
    unchanged -- but the raw-power-sum form all three backends use (``var = E[x**2] - E[x]**2`` at k=2) loses
    catastrophic precision when a column's offset dwarfs its spread. A price-like column centred at 8.5e3 with
    spread 0.05 drives ``varx`` to numerical garbage, it trips this file's own ``var <= 1e-24`` guard, the entry
    becomes ``nan``, and ``nan`` means "not a duplicate" here -- so genuinely collinear engineered columns
    survive dedup. Centring first is the cheap exact fix (one O(q*n) pass against six matmuls).

    The mean is taken in double precision for accuracy, then cast back so the cupy backend's documented caller-dtype
    passthrough is unaffected. All-NaN rows yield a NaN mean and stay all-NaN; they are excluded downstream by
    the ``n < 8`` common-row guard either way.
    """
    A = np.ascontiguousarray(A)
    # Intentional float64: dtype identity check on the input, not a compute dtype
    if A.ndim == 2 and A.dtype in (np.float32, np.float64):
        out = np.empty_like(A)
        _row_center_kernel(A, out)
        return out
    finite = np.isfinite(A)
    counts = finite.sum(axis=1, keepdims=True)
    # Intentional float64: mean taken in f64 for accuracy, then cast back (documented)
    sums = np.where(finite, np.asarray(A, dtype=np.float64), 0.0).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    # Intentional float64: mean taken in f64 for accuracy, then cast back (documented)
    return np.asarray(np.asarray(A, dtype=np.float64) - means, dtype=A.dtype)


@njit(parallel=True, cache=True, nogil=True)
def _row_center_kernel(A: np.ndarray, out: np.ndarray) -> None:
    """One pass per row: the mean of the finite entries in double precision, then ``out = A - mean`` cast to the dtype of ``A`` (all-NaN rows stay as they are; no temporaries).

    Replaces the isfinite / where / sum / subtract chain of numpy (five passes over the matrix and two double-precision copies of it)."""
    q, n = A.shape
    for i in prange(q):
        s = 0.0
        c = 0
        for j in range(n):
            v = A[i, j]
            if np.isfinite(v):
                s += v
                c += 1
        m = s / c if c > 0 else 0.0
        for j in range(n):
            out[i, j] = A[i, j] - m


def _pairwise_complete_abs_corr(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``|Pearson r|`` between every row of ``Q`` and every row of ``R`` over PAIRWISE-COMPLETE (both-finite) rows,
    vectorized instead of a Python per-pair ``np.corrcoef`` loop. ``Q``/``R``: ``(q, n)`` / ``(r, n)`` double precision with
    ``NaN`` for missing. Returns ``(q, r)``; entries with <8 common obs or zero variance -> ``nan`` ('not a duplicate').

    Backend dispatch (all three bit-identical to ~1e-16): numpy(BLAS) is the CPU default; ``cupy`` and ``njit`` are
    force-selectable via ``MLFRAME_FE_DEDUP_CORR_BACKEND=cupy|njit|numpy`` and consulted per-host via the kernel tuning
    cache. numpy is the default because (a) njit measured slower everywhere and (b) cupy's solo win is eroded under the
    FE pipeline's joblib-worker GPU contention (same lesson as ``lookup_mi_classif_backend``). cupy OOM -> numpy."""
    # Centre ONCE here rather than in each backend: all three share the raw-power-sum variance form and would
    # otherwise each need the same fix (and could drift apart). See ``_row_center_finite``.
    Q = _row_center_finite(Q)
    R = _row_center_finite(R)
    backend = _resolve_pc_backend(Q.shape[0], R.shape[0], Q.shape[1])
    if backend == "njit":
        # Intentional float64: njit kernel operand contract (f64 signature)
        return np.asarray(_pc_corr_njit(np.ascontiguousarray(Q, dtype=np.float64), np.ascontiguousarray(R, dtype=np.float64)))
    if backend == "cupy":
        try:
            return _pc_corr_cupy(Q, R)
        except Exception as exc:  # cupy missing / OOM / driver error -> safe CPU fallback
            logger.warning("[FE dedup] cupy pairwise-corr backend failed (%s); falling back to numpy.", type(exc).__name__)
    return _pc_corr_numpy(Q, R)


_SWEEP_P = [16, 64, 256]
_SWEEP_N = [5_000, 20_000, 100_000]
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(Q, R)``: ``p`` x ``n`` row-centred blocks in the criterion dtype (what the dedup hands its backends) with about 3 % missing values; ``R`` repeats a tenth of ``Q``'s rows with small noise, so the
    block has true near-duplicates (|r| near 1) for the decision the dedup makes."""
    rng = np.random.default_rng(0)
    p, n = int(dims["p"]), int(dims["n"])
    base = rng.standard_normal((p, n))
    other = rng.standard_normal((p, n))
    k = max(1, p // 10)
    other[:k] = base[:k] + 0.02 * rng.standard_normal((k, n))
    for a in (base, other):
        a[rng.random((p, n)) < 0.03] = np.nan
    dt = _crit_np_dtype()
    return (_row_center_finite(base.astype(dt)), _row_center_finite(other.astype(dt)))


def _nan_to_flag(a: np.ndarray) -> np.ndarray:
    """The correlation block with ``nan`` ('not a duplicate') as -1, so the sweep's equivalence check compares like with like."""
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=-1.0)  # Intentional float64: comparison array


def _sweep_variants() -> dict:
    """The backends the sweep times (cupy only where a device is usable)."""
    variants = {
        "numpy": lambda q, r: _nan_to_flag(_pc_corr_numpy(q, r)),
        "njit": lambda q, r: _nan_to_flag(_pc_corr_njit(np.ascontiguousarray(q, dtype=np.float64), np.ascontiguousarray(r, dtype=np.float64))),  # Intentional float64: njit signature
    }
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        variants["cupy"] = lambda q, r: _nan_to_flag(_pc_corr_cupy(q, r))
    except Exception as exc:  # no usable device: the sweep decides between the host backends
        logger.debug("pairwise-corr sweep without the cupy backend (%s: %s)", type(exc).__name__, exc)
    return variants


#: Equivalence of the backends in the sweep, absolute on |r|. The block is in the criterion dtype (float32 by default), so a backend that accumulates in float64 (njit) differs from the float32 GEMMs by the
#: rounding of an n-term float32 sum. Measured on the sweep grid (p up to 256, n up to 100000): njit 6e-7, cupy 9e-7 against the numpy reference; 1e-5 is ten times that, and the dedup only compares |r| with a 0.999 threshold.
_SWEEP_TOL = 1e-5


def _run_pc_sweep() -> list:
    """Grid sweep over (columns, rows): the three backends timed per cell, the fastest kept; every backend must equal the numpy reference to ``_SWEEP_TOL``."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return list(sweep_backend_grid(_sweep_variants(), {"p": _SWEEP_P, "n": _SWEEP_N}, _sweep_inputs_adapter, reference="numpy", equiv_rtol=_SWEEP_TOL, equiv_atol=_SWEEP_TOL, repeats=3))


def _sweep_inputs_adapter(dims: dict) -> tuple:
    """Sweep input factory (named so the grid sweep can call it with the cell dims)."""
    return _sweep_inputs(dims)


def _pc_fallback_choice(p: int, n: int) -> str:
    """Pre-sweep choice: numpy (the BLAS form), the measured default before any host has been swept."""
    return "numpy"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fe_pairwise_complete_corr",
        variant_fns=(_pc_corr_numpy, _pc_corr_njit, _pc_corr_cupy_arrays),  # the three bodies: an edit of any invalidates the entry
        tuner=_run_pc_sweep,
        axes={"p": list(_SWEEP_P), "n": list(_SWEEP_N)},
        fallback=_pc_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_pairwise_complete_corr",
    )
except Exception as e:  # best-effort: without a tuner entry the numpy default decides
    logger.warning("kernel_tuner spec build for fe_pairwise_complete_corr failed: %s", e)
    _SPEC = None


def _resolve_pc_backend(q: int, r: int, n: int) -> str:
    """Pick 'numpy' | 'njit' | 'cupy'. Env force wins; else the per-host kernel-tuning cache (fallback 'numpy')."""
    forced = _os.environ.get("MLFRAME_FE_DEDUP_CORR_BACKEND", "").strip().lower()
    if forced in ("numpy", "njit"):
        return forced
    if forced in ("cupy", "cuda"):
        return "cupy"
    # STRICT GPU mode: prefer the cupy backend (solo-benched 2.0-2.7x over numpy in the moderate band;
    # 12 numpy calls cost 335s on the wellbore-100k GPU-strict cProfile). The numpy default exists because
    # cupy's solo win erodes under joblib-worker GPU contention - but STRICT mode's explicit contract is
    # "carry the FE compute on the device", and the per-call work floor inside fe_gpu_strict_enabled already
    # keeps trivially small dedups (p < 64 or n*p < 1M) on the CPU. Non-strict hosts keep the measured
    # default via the KTC lookup below, unchanged.
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled

        if fe_gpu_strict_enabled(n=n, p=max(q, r)):
            return "cupy"
    except Exception as e:  # nosec B110 - strict-gate probe failure must never break the dedup itself
        log_throttle(logger, "_orth_dedup._resolve_pc_backend.strict_probe", logging.WARNING, "fe_gpu_strict_enabled() probe failed, dedup keeps its default backend: %s", e)
    if _SPEC is None:
        return "numpy"
    try:
        choice = str(_SPEC.choose(p=max(q, r), n=n))
        return choice if choice in ("numpy", "njit", "cupy") else "numpy"
    except Exception as e:
        logger.warning("_resolve_pc_backend: kernel-tuning choice failed, defaulting the pairwise-corr backend to numpy: %s", e)
        return "numpy"


def _dedup_walk(
    cols: Sequence[str], classes: list, dense_idx: list, abs_corr: Optional[np.ndarray], pc_pd: Optional[np.ndarray], pc_pp: Optional[np.ndarray],
    partial_pos: dict, corr_threshold: float,
) -> list[str]:
    """Pass 3: walk the columns in order and keep those not duplicated (|corr| >= threshold) by an already-kept column."""
    def _hits(vec: np.ndarray, idxs: list[int]) -> bool:
        """True iff any kept reference in ``idxs`` has finite |corr| >= threshold (nan == not-a-duplicate)."""
        if not idxs:
            return False
        sub = vec[idxs]
        return bool(np.any(np.isfinite(sub) & (sub >= corr_threshold)))

    # ---- Pass 3: walk in order, apply dedup verdict --------------------------
    kept: list[str] = []
    kept_dense_rows: list[int] = []  # dense-matrix row indices already kept
    kept_partial_pos: list[int] = []  # partial-matrix row indices already kept
    dense_pos = 0  # which dense candidate slot we're currently at
    for i, c in enumerate(cols):
        cls = classes[i]
        if cls == "pass_through":
            kept.append(c)
            # Pass-through columns are never used as a corr reference for downstream candidates (legacy stored arr but
            # the comparison always short-circuited via `.sum() < 8` or `.std() <= 1e-12`).
            continue
        if cls == "dense":
            row_idx = dense_idx[dense_pos]
            dense_pos += 1
            assert abs_corr is not None  # this "dense" cls only occurs when dense_rows was non-empty, which set abs_corr
            is_dup = _hits(abs_corr[row_idx], kept_dense_rows)
            # Also compare against any kept partial-NaN columns (rare): pc_pd[p, row_idx] is partial-p vs this dense col.
            if not is_dup and kept_partial_pos:
                assert pc_pd is not None  # kept_partial_pos non-empty implies partial_order was non-empty, which set pc_pd
                col = pc_pd[:, row_idx]
                is_dup = bool(np.any(np.isfinite(col[kept_partial_pos]) & (col[kept_partial_pos] >= corr_threshold)))
            if not is_dup:
                kept.append(c)
                kept_dense_rows.append(row_idx)
            continue
        # cls == "partial_nan": vectorized lookups against kept dense + kept partial.
        p = partial_pos[i]
        assert pc_pd is not None and pc_pp is not None  # this "partial_nan" cls only occurs when partial_order was non-empty, which set both
        is_dup = True if _hits(pc_pd[p], kept_dense_rows) else _hits(pc_pp[p], kept_partial_pos)
        if not is_dup:
            kept.append(c)
            kept_partial_pos.append(p)
    return kept


def _dense_abs_corr(dense_rows: list) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Pass 2: ``(dense_matrix, |corr| matrix)`` of the fully-finite columns (both None when there are none)."""
    # ---- Pass 2: bulk corrcoef on the dense block ---------------------------
    # One C call replaces p_dense * (p_dense - 1) / 2 per-pair Python+numpy
    # roundtrips. Numerically equivalent to per-pair np.corrcoef because
    # numpy.corrcoef(M)[i, j] uses the same _cov / _std reduction order as
    # numpy.corrcoef(M[i], M[j])[0, 1] (verified bit-identical at p=200).
    if dense_rows:
        dense_matrix = np.vstack(dense_rows)
        # This block used to hardcode np.corrcoef regardless of backend, unlike the partial-NaN block below
        # which already dispatches through _resolve_pc_backend/_pairwise_complete_abs_corr (GPU-strict aware).
        # On the wellbore-100k GPU-strict cProfile this hardcoded CPU call cost 116.9s cumtime over just 9
        # calls (p_dense in the thousands after FE families stack their candidate columns) while the STRICT
        # contract is "carry FE compute on the device". A fully-finite matrix is the degenerate case of
        # pairwise-complete-obs correlation (every row kept), so routing it through the same bit-verified
        # backend as the partial-NaN path is correctness-neutral and picks up the cupy/njit dispatch for free.
        _backend = _resolve_pc_backend(len(dense_rows), len(dense_rows), dense_matrix.shape[1])
        if _backend in ("cupy", "njit"):
            abs_corr = _pairwise_complete_abs_corr(dense_matrix, dense_matrix)
        else:
            # Empty (0, n) matrix corrcoef raises; only call when we have rows.
            corr_matrix = np.corrcoef(dense_matrix)
            # Single-row corrcoef returns a scalar 1.0 instead of (1, 1); normalize.
            if corr_matrix.ndim == 0:
                # Intentional float64: accumulator kept f64 (statistic / count sums need full precision)
                corr_matrix = np.array([[1.0]], dtype=np.float64)
            # Absolute corrs only; NaN -> not duplicate (matches legacy's
            # `if not np.isfinite(corr): continue` skip).
            abs_corr = np.abs(corr_matrix)
    else:
        abs_corr = None
    return (dense_matrix if dense_rows else None), abs_corr


def _classify_dedup_columns(X: pd.DataFrame, cols: Sequence[str], _sample_idx: Optional[np.ndarray]) -> tuple[list, list, list, dict]:
    """Pass 1: classify each column as pass_through / dense / partial_nan; returns ``(classes, dense_idx, dense_rows, partial_arrays)``."""
    classes: list[str] = []  # one of: "pass_through", "dense", "partial_nan"
    dense_idx: list[int] = []  # candidate index in cols -> dense-matrix row
    dense_rows: list[np.ndarray] = []  # the dense arrays themselves
    partial_arrays: dict[int, np.ndarray] = {}  # candidate index -> arr (with NaN)

    from .._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); coarse corr-threshold dedup is scale-robust. Hoisted out of the loop (was re-invoked once per column).
    for i, c in enumerate(cols):
        if not fe_column_is_numeric(X, c):
            classes.append("pass_through")
            continue
        arr = np.asarray(fe_column_values(X, c), dtype=_dt)
        if _sample_idx is not None:
            arr = arr[_sample_idx]
        finite = np.isfinite(arr)
        all_finite = bool(finite.all())
        if not all_finite and not finite.any():
            # All-NaN: pass-through, no kept_array stored. (Matches legacy:
            # legacy stored kept_arrays[c] = arr but immediately continued
            # on the next iteration's `mask.sum() < 8` check; the only
            # observable effect is that downstream partial-NaN candidates
            # don't compute corr against an all-NaN kept column anyway.)
            classes.append("pass_through")
            continue
        # std on the finite subset, matching legacy's constant-detection.
        # (an all-finite column is its own finite subset: no boolean-index copy of the whole column)
        if (arr.std() if all_finite else arr[finite].std()) <= 1e-12:
            classes.append("pass_through")
            continue
        if all_finite:
            classes.append("dense")
            dense_idx.append(len(dense_rows))
            dense_rows.append(arr)
        else:
            classes.append("partial_nan")
            partial_arrays[i] = arr
    return classes, dense_idx, dense_rows, partial_arrays


def _dedup_collinear_source_cols(
    X: pd.DataFrame, cols: Sequence[str], *, corr_threshold: float = 0.999,
) -> list[str]:
    """Drop near-duplicate source columns BEFORE basis enumeration.

    Layer 27 incident: on 10 collinear sources (x2..x10 = x1 +
    1% jitter), the constructor emitted 10 He_2 columns and every one
    survived MRMR's redundancy gate because their CMI-residuals under
    quantile binning differed by tiny amounts above the relevance floor.
    Hybrid stage exploded the candidate set 10x and MRMR couldn't
    distinguish the duplicates.

    Fix: a cheap source-side dedup pass. Walks cols in order, computes the
    abs Pearson correlation against every column already kept; drops the
    candidate if it correlates above ``corr_threshold`` with anything in
    the kept set. ``0.999`` matches the 1% jitter test fixture while
    leaving real-world near-duplicates (corr in [0.95, 0.99]) untouched.

    Non-numeric / constant / all-NaN columns are passed through (not
    deduped, not dropped) so downstream basis evaluation handles them as
    before.

    Layer 30 perf: the original implementation called
    ``np.corrcoef`` per (candidate, kept) pair which is O(p^2) numpy calls
    plus Python overhead. At p=200 cProfile attributed 5.0s out of 4.8s
    wall (cumulative) to this dedup pass — the dominant hotspot. The new
    implementation:

    1. Pre-classifies columns into pass-through (non-numeric / all-NaN /
       constant), partial-NaN (rare path), and fully-finite-and-varying.
    2. Stacks all fully-finite columns into one (p_dense, n) matrix and
       calls ``np.corrcoef`` once on the bulk matrix — one C call instead
       of O(p^2). Numerically bit-identical to the per-pair recipe (same
       reduction order in numpy's cov / std).
    3. For each candidate, looks up its row of the precomputed matrix
       against indices of kept dense columns: O(K) per candidate, O(p*K)
       total, no Python-side reductions.
    4. Partial-NaN columns fall back to the original masked-corr path
       (still per-pair, but the count of these is typically 0 — production
       hybrid path uses np.nanmean fill before reaching this dedup).

    Bench at p=200 n=2000 all-finite synthetic frame: 5.0s -> ~0.05s
    (100x).

    2026-07-12 FIT-SCOPED MEMO: wrap the call site in ``dedup_collinear_memo_scope()`` to memoize repeated
    calls with the SAME (cols content, corr_threshold) within one MRMR.fit() call - five independent opt-in
    FE-family builders (orth / extra-basis / gpu-resident / wavelet / hinge) each call this dedup on their own
    candidate columns, and a "kitchen sink" config with several families on would otherwise repeat the
    identical O(P^2) corrcoef pass 2-5x. The memo is OFF by default (cache is None) so every call outside an
    explicit scope is byte-for-byte unchanged.
    """
    if not cols:
        return list(cols)
    _memo = _DEDUP_MEMO["cache"]
    _memo_key = None
    if _memo is not None:
        _memo_key = _dedup_cache_key(X, cols, corr_threshold)
        _hit = _memo.get(_memo_key)
        if _hit is not None:
            return list(_hit)
    # ---- Pass 1: classify columns -------------------------------------------
    # Pre-classify each column so the bulk corrcoef in pass 2 only sees fully-
    # finite varying columns. Order-preservation matters because the kept
    # list mirrors the input order; we record per-col disposition first then
    # walk in order again in pass 3.
    n_rows = len(X)
    # Row-cap the correlation estimate (see _MAX_CORR_ROWS note). Deterministic seed so a fit is reproducible; sorted so
    # the sampled rows keep their original order (irrelevant to Pearson, but avoids surprising downstream if reused).
    _sample_idx = _dedup_sample_idx(n_rows)
    classes, dense_idx, dense_rows, partial_arrays = _classify_dedup_columns(X, cols, _sample_idx)

    dense_matrix, abs_corr = _dense_abs_corr(dense_rows)

    # Precompute the partial-NaN comparison blocks ONCE via the vectorized pairwise-complete kernel, replacing the
    # legacy O(P^2) per-pair ``np.corrcoef`` loop (the hot path on frames with many NaN-bearing columns, e.g. well logs
    # with missing measurements - the case the legacy "typically 0 partial columns" assumption failed on). ``partial_mat``
    # rows are indexed by ``partial_pos`` in candidate order; ``pc_pd`` = |corr| vs each dense row, ``pc_pp`` = vs each
    # other partial row. Empty when there are no partial columns (the common all-dense case is untouched -> bit-identical).
    partial_order = [i for i in range(len(cols)) if classes[i] == "partial_nan"]
    partial_pos = {i: p for p, i in enumerate(partial_order)}
    pc_pd: np.ndarray | None
    pc_pp: np.ndarray | None
    if partial_order:
        partial_mat = np.vstack([partial_arrays[i] for i in partial_order])
        dense_mat_for_pc = dense_matrix if dense_matrix is not None else np.empty((0, partial_mat.shape[1]), dtype=_crit_np_dtype())
        pc_pd = _pairwise_complete_abs_corr(partial_mat, dense_mat_for_pc)  # (n_partial, n_dense)
        pc_pp = _pairwise_complete_abs_corr(partial_mat, partial_mat)  # (n_partial, n_partial)
    else:
        pc_pd = pc_pp = None

    kept = _dedup_walk(cols, classes, dense_idx, abs_corr, pc_pd, pc_pp, partial_pos, corr_threshold)
    if _memo is not None:
        _memo[_memo_key] = kept
    return kept
