"""Shared GPU-resident feature-engineering state and helpers, kept in a leaf module.

Leaf module: the resident discretiser, pair, select, k-chunk and materialise siblings import these, and this module imports nothing from the resident package.
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from collections import OrderedDict
import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters._gpu_resident_fe")


_MINIMAL_UNARY = ("identity", "neg", "abs", "sqr", "reciproc", "sqrt", "log", "sin")


_MINIMAL_BINARY = ("mul", "add", "sub", "div", "max", "min")


def _cuda_present() -> bool:
    """Best-effort CUDA-availability probe for the resident-codes default. Mirrors
    ``batch_mi_noise_gate_gpu._CUDA_AVAIL`` (pyutilz ``is_cuda_available`` -> numba.cuda fallback) without
    importing the GPU twin at module-import time. Any failure -> False (CPU path, never a regression).

    The global opt-out is checked first because neither backing probe covers it: numba honours
    ``CUDA_VISIBLE_DEVICES=""`` but nothing below knows about ``MLFRAME_DISABLE_GPU=1``, so a run that
    declared no GPU would still engage the whole resident-operand family through this one predicate."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return bool(is_cuda_available())
    except Exception as e:
        logger.debug("_cuda_present: pyutilz.is_cuda_available() failed, trying numba.cuda.is_available(): %s", e)
        try:
            from numba import cuda as _c

            return bool(getattr(_c, "is_available", lambda: False)())
        except Exception as e2:
            logger.debug("_cuda_present: numba.cuda.is_available() also failed, defaulting to no CUDA: %s", e2)
            return False


def _gpu_min_free_vram_mb() -> int:
    """Minimum FREE device VRAM (MB) for the FE GPU paths to default ON. Below this the (n, K) FE matrices cannot
    fit and the GPU work stalls/OOM-falls-back while the (parallelized) CPU cores idle - so default to CPU instead.
    Env-tunable ``MLFRAME_FE_GPU_MIN_FREE_VRAM_MB`` (default 1024). 0 disables the check (legacy: present-only gate)."""
    try:
        return int(os.environ.get("MLFRAME_FE_GPU_MIN_FREE_VRAM_MB", "1024"))
    except (ValueError, TypeError) as _exc:
        logger.warning("_gpu_min_free_vram_mb fell back to its default after %s", type(_exc).__name__)
        return 1024


def _gpu_has_free_vram() -> bool:
    """True iff the current CUDA device has at least ``_gpu_min_free_vram_mb()`` MB FREE. On a small/shared card whose
    VRAM is mostly consumed (e.g. a 4GB desktop GPU eaten by the compositor/browsers) this is False, so the FE GPU
    default-on gates route to CPU rather than starving on ~hundreds of MB. A query failure -> True (defer to the
    existing per-path OOM fallbacks). Skipped when the threshold is 0."""
    _min_mb = _gpu_min_free_vram_mb()
    if _min_mb <= 0:
        return True
    try:
        import cupy as cp

        free_b, _total = cp.cuda.runtime.memGetInfo()
        # bytes cupy's pool holds but has not handed out are reusable by this process: counting them as used flips the gate off for the rest of a large fit
        free_b += int(cp.get_default_memory_pool().free_bytes())
        return bool(free_b >= _min_mb * 1024 * 1024)
    except Exception as e:
        logger.debug("_gpu_has_free_vram: VRAM query failed, deferring to the existing per-path OOM fallbacks: %s", e)
        return True


def _env_gpu_default_on(name: str) -> bool:
    """Shared GPU opt-out gate: explicit env ``<name>`` = 0/1 (off/on) wins; otherwise default ON when a CUDA device
    is usable AND has enough FREE VRAM - False if CUDA_VISIBLE_DEVICES='' or MLFRAME_DISABLE_GPU=1, no CUDA present,
    or the device is nearly full (a tiny / desktop-shared card). Single source for the fe_gpu_*_enabled default-on
    gates. Explicit ``=1`` still forces GPU past the free-VRAM check (user override / big-GPU host)."""
    _v = os.environ.get(name, "").strip().lower()
    if _v in ("0", "false", "off", "no"):
        return False
    if _v in ("1", "true", "on", "yes"):
        return True
    # Delegate the CUDA_VISIBLE_DEVICES=""/MLFRAME_DISABLE_GPU=1
    # opt-out check to the shared _gpu_policy module instead of reimplementing it inline, so a future change
    # to the shared policy's semantics reaches this cluster's default-on gates too.
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    return _cuda_present() and _gpu_has_free_vram()


def fe_gpu_resident_codes_enabled() -> bool:
    """Whether the GPU-RESIDENT-CODES handoff is active. DEFAULT ON when CUDA is present (env opt-out
    ``MLFRAME_FE_GPU_RESIDENT_CODES=0``); explicit truthy forces it on even without a detected device.

    When ON, ``gpu_materialise_discretize_codes_host`` keeps the on-device int codes RESIDENT (a single
    (n, K) cupy array in the narrow code dtype) and stashes them in a module-level handoff (keyed on the
    identity of the host codes array it returns). The noise-gate dispatch then feeds those device codes
    STRAIGHT into ``batch_mi_with_noise_gate_cuda_resident`` - skipping the H2D re-upload of the codes the
    resident gate would otherwise pay (the codes were just produced on the GPU, D2H'd, and re-H2D'd: a
    pointless GPU->host->GPU round-trip of the (n, K) int codes). The host codes are STILL produced, so the
    CPU dispatch / analytic gate / GPU-opt-out / SU / any-failure branches are byte-for-byte unchanged and
    the round-trip is skipped ONLY when the resident CUDA gate is the chosen consumer (so this is
    selection-equivalent - the device codes are the EXACT bytes the host ``out`` carries). Gate:
    ``MLFRAME_FE_GPU_RESIDENT_CODES`` (0/false/off -> off; 1/true/on/yes -> on; UNSET -> on iff CUDA
    present). Honours the CUDA opt-out conventions (``CUDA_VISIBLE_DEVICES=""`` / ``MLFRAME_DISABLE_GPU=1``)
    by short-circuiting to OFF - the resident gate is skipped on those runs anyway, so never stash."""
    return _env_gpu_default_on("MLFRAME_FE_GPU_RESIDENT_CODES")


def fe_gpu_defer_host_codes_enabled() -> bool:
    """Whether the DEFERRED host-codes D2H is active. DEFAULT ON whenever the resident-codes handoff is on
    (the device codes already exist resident, so skipping the eager host D2H of those SAME codes is free
    upside - the canonical fit's single largest D2H, 1691 MB / 160 transfers @100k, paid only on the host
    paths that actually read it, which is NONE when the resident gate consumes the device copy). Opt-out
    ``MLFRAME_FE_GPU_DEFER_HOST_CODES=0`` forces the eager fill back (host ``out`` filled per block).

    SELECTION-EQUIVALENT: when a host consumer (analytic gate / CPU njit / non-resident GPU) reads the
    codes, the dispatch first calls ``ensure_host_codes_filled`` which D2Hs the resident device codes into
    ``out`` - the EXACT bytes the eager fill produced -> identical codes -> identical MI/selection. The
    only behavioural change is WHEN the D2H happens (lazily, on demand) vs eagerly (always)."""
    _v = os.environ.get("MLFRAME_FE_GPU_DEFER_HOST_CODES", "").strip().lower()
    if _v in ("0", "false", "off", "no"):
        return False
    if _v in ("1", "true", "on", "yes"):
        return True
    return fe_gpu_resident_codes_enabled()  # default ON iff the resident-codes handoff is on


_RESIDENT_CODES_HANDOFF: "OrderedDict[int, tuple]" = OrderedDict()  # id(host) -> (device_codes, shape, dtype, weakref(host))


_RESIDENT_CODES_HANDOFF_MAX = 8


_DEFERRED_HOST_FILL: "OrderedDict[int, tuple]" = OrderedDict()  # id(host) -> (weakref(host_codes), device_codes, shape, dtype, filled[list[bool]])


_DEFERRED_HOST_FILL_MAX = 8


_DEFERRED_HOST_FILL_LOCK = threading.Lock()


def _fill_deferred(entry) -> None:
    """D2H one registry entry's device codes into its host buffer, once. An entry whose host buffer has been garbage-collected has no reader left
    and is skipped: filling it would copy the whole (n, K) codes matrix to a buffer nobody can read."""
    host_ref, dev, _shape, _dtype, filled = entry
    host = host_ref()
    if host is not None and not filled[0]:
        dev.get(out=host)
        filled[0] = True


def _stash_resident_codes(host_codes, device_codes) -> None:
    """Record the resident device codes for the host array ``host_codes`` (keyed on its id)."""
    with _DEFERRED_HOST_FILL_LOCK:
        c = _RESIDENT_CODES_HANDOFF
        c[id(host_codes)] = (device_codes, tuple(host_codes.shape), np.dtype(host_codes.dtype), weakref.ref(host_codes))
        c.move_to_end(id(host_codes))
        while len(c) > _RESIDENT_CODES_HANDOFF_MAX:
            # evict-ok: a miss routes to the host codes, which ensure_host_codes_filled always delivers
            c.popitem(last=False)


def _stash_deferred_host_fill(host_codes, device_codes) -> None:
    """Register ``host_codes`` (an UNFILLED host buffer) for a lazy D2H fill from ``device_codes`` - used
    when the producer skips the eager codes D2H because the resident gate will likely consume the device
    copy. ``ensure_host_codes_filled`` performs the fill on demand; ``clear_resident_codes_handoff`` drops
    the record so device memory is not pinned past the dispatch."""
    with _DEFERRED_HOST_FILL_LOCK:
        c = _DEFERRED_HOST_FILL
        # Records whose host buffer is gone (the dispatch that owned it ended without a host read) would otherwise pin their device codes until
        # evicted, and eviction used to copy them back to a buffer nobody can read.
        for _dead in [k for k, e in c.items() if e[0]() is None]:
            del c[_dead]
        c[id(host_codes)] = (weakref.ref(host_codes), device_codes, tuple(host_codes.shape), np.dtype(host_codes.dtype), [False])
        c.move_to_end(id(host_codes))
        while len(c) > _DEFERRED_HOST_FILL_MAX:
            # A buffer whose owner is alive is never dropped unfilled: the owner may still be waiting to read it.
            _fill_deferred(c.popitem(last=False)[1])


def _xp_special(xp):
    """The scipy.special-compatible module for array module ``xp``: ``scipy.special`` for numpy,
    ``cupyx.scipy.special`` for cupy. Used for the GPU full-catalog special functions (erf / gammaln /
    beta / binom are present in cupyx; dawsn / agm are NOT - see _GPU_UNAVAILABLE_* below)."""
    if xp is np:
        import scipy.special as _s

        return _s
    import cupyx.scipy.special as _s

    return _s


def _unary_apply(xp, name, x):
    """Apply unary ``name`` to ``x`` using array module ``xp`` (numpy or cupy). Semantics mirror
    ``feature_engineering.create_unary_transformations`` EXACTLY (full maximal catalog), incl.
    smart_log's full-column nanmin shift. Raises ValueError for an op with no ``xp`` equivalent (the
    caller excludes it from the GPU grid - see _GPU_UNAVAILABLE_UNARY)."""
    # --- minimal ---
    if name == "identity":
        return x
    if name == "neg":
        return -x
    if name == "abs":
        return xp.abs(x)
    if name == "sqr":
        return xp.power(x, 2)
    if name == "reciproc":
        return xp.power(x, -1.0)
    if name == "sqrt":
        return xp.sqrt(xp.abs(x))
    if name == "log":
        x_min = xp.nanmin(x)
        # smart_log: shift only when the column reaches <=0 (anchor frozen over the FULL column).
        return xp.log(x) if float(x_min) > 0 else xp.log(x + (1e-5 - x_min))
    if name == "sin":
        return xp.sin(x)
    # --- medium ---
    if name == "sign":
        return xp.sign(x)
    if name == "rint":
        return xp.rint(x)
    if name == "qubed":
        return xp.power(x, 3)
    if name == "invsquared":
        return xp.power(x, -2)
    if name == "invqubed":
        return xp.power(x, -3)
    if name == "cbrt":
        return xp.cbrt(x)
    if name == "invcbrt":
        return xp.power(x, -1.0 / 3.0)
    if name == "invsqrt":
        return xp.power(x, -1.0 / 2.0)
    if name == "exp":
        return xp.exp(x)
    # --- maximal: trig / inverse-trig / hyperbolic / inverse-hyperbolic (all native to numpy AND cupy) ---
    if name in ("sinc", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh"):
        return getattr(xp, name)(x)
    # --- maximal: special functions (scipy.special / cupyx.scipy.special) ---
    if name == "erf":
        return _xp_special(xp).erf(x)
    if name == "gammaln":
        return _xp_special(xp).gammaln(x)
    # GPU-DISABLED(restore): uncomment these + the matching _FULL_UNARY entries + the
    # feature_engineering.py catalog lines to re-enable. grad1/grad2 need a FULL-column gradient (never a
    # row-block); dawsn needs a cupyx/custom kernel (absent on this stack).
    # if name == "grad1":
    #     return xp.gradient(x)
    # if name == "grad2":
    #     return xp.gradient(x, edge_order=2)
    # if name == "dawsn":
    #     return _xp_special(xp).dawsn(x)
    raise ValueError(f"unknown unary {name!r}")


_COMBOS = [(ua, ub, bop) for ua in _MINIMAL_UNARY for ub in _MINIMAL_UNARY for bop in _MINIMAL_BINARY]


_UNARY_IDX = {u: i for i, u in enumerate(_MINIMAL_UNARY)}


_BINOP_CODE = {"mul": 0, "add": 1, "sub": 2, "div": 3, "max": 4, "min": 5}


_COMBO_IDX_CACHE: dict = {}  # block-tuple -> (ua_idx, ub_idx, bop) device int32; module-level -> pickle-safe


_QLEVELS_CACHE: dict = {}  # (nbins, work-dtype) -> cp.linspace(0,100,nbins+1) device vector; read-only, shared


_GPU_RESIDENT_FE_CACHE_LOCK = threading.Lock()


def _quantile_levels_dev(cp, nbins: int, work):
    """Cached cp.linspace(0,100,nbins+1, dtype=work) percentile-level vector (depends only on (nbins, work),
    rebuilt every chunk otherwise). Read-only - cp.percentile never mutates it - so sharing is safe and
    the returned array is byte-identical to the inline linspace (deterministic in its args)."""
    key = (int(nbins), work)
    with _GPU_RESIDENT_FE_CACHE_LOCK:
        hit = _QLEVELS_CACHE.get(key)
        if hit is None:
            hit = cp.linspace(0.0, 100.0, int(nbins) + 1, dtype=work)
            _QLEVELS_CACHE[key] = hit
    return hit


_GPU_MI_WORKING_MULTIPLE = 5


_GPU_K_CHUNK_VRAM_FRACTION_DEFAULT = 0.25


_GPU_K_CHUNK_VRAM_FRACTIONS = (0.25, 0.40, 0.55, 0.70)  # discrete grid the per-host sweep ranks


_GPU_RESIDENT_MIN_N = 50_000


def _gpu_k_chunk_vram_fraction(n: int) -> float:
    """Per-host VRAM fraction for :func:`_gpu_k_chunk` (lazy KTC resolver in sibling); safe default on miss."""
    try:
        from ._gpu_resident_k_chunk_ktc import gpu_k_chunk_vram_fraction as _resolve

        return _resolve(n)
    except Exception as e:
        logger.debug("_gpu_k_chunk_vram_fraction: KTC lookup failed, using the safe default fraction %.2f: %s", _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT, e)
        return _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT


def _gpu_k_chunk(
    n: int,
    *,
    free_bytes: int | None = None,
    bytes_per_elem: int = 8,
    working_multiple: int | None = None,
    max_cols: int | None = None,
    vram_fraction: float | None = None,
) -> int:
    """Max candidate columns to score in ONE on-device batch so the working set stays within a FRACTION of
    free VRAM - bounds peak GPU memory like the CPU RAM governor, removing the large-n cliff.

    ``bytes_per_elem`` / ``working_multiple`` describe the caller's per-column resident footprint; defaults
    (8 x ``_GPU_MI_WORKING_MULTIPLE``) match the f64 MI prototype, the CODES path passes 4 x ~4 for a ~3x
    wider sub-chunk = ~3x fewer launches. Per-column-INDEPENDENT, so codes/MI are bit-identical regardless of
    chunk boundaries. ``max_cols`` caps to the caller's column count (defaults len(_COMBOS)). G3: the VRAM
    fraction is per-host KTC-tuned; an explicit ``vram_fraction`` (sweep probe) overrides; clamp to (0, 0.9]
    so a stale cache entry never zeros the budget nor exceeds 90% of free VRAM."""
    if free_bytes is None:
        # DEVICE-free ALONE is wrong when cupy's mempool holds the card (gap-analysis root cause):
        # cudaMemGetInfo reports ~0 device-free once the resident FE caches + operand tables have grown the
        # pool to fill VRAM, yet the pool holds hundreds of MB of REUSABLE free blocks that this batch's
        # (n, k_chunk) working set can allocate WITHOUT an OS request. Sizing on device-free alone collapsed
        # k_chunk to 1 -> the materialise+bin loop ran ONE candidate per launch (measured 11,664 per-candidate
        # launches + their host syncs = the dominant GPU-idle source). Add the pool's free capacity
        # (total - used) so the budget reflects what can actually be allocated -> the loop batches again.
        # cupy is only imported here, on the path that actually needs live device queries -- callers that
        # pass free_bytes explicitly (CPU-fallback sizing math, unit tests, no-GPU hosts) never pay the import.
        import cupy as cp

        _dev_free, _dev_total = cp.cuda.runtime.memGetInfo()
        try:
            _mp = cp.get_default_memory_pool()
            _pool_free = int(_mp.total_bytes()) - int(_mp.used_bytes())
        except Exception as e:
            logger.warning("querying the cupy memory pool free bytes failed: %s", e)
            _pool_free = 0
        # Cap the pool-free contribution at a fraction of TOTAL VRAM so one batch's working set stays bounded
        # (the full pool_free would size a multi-GB (n, k_chunk) allocation that churns the pool on this 4 GB
        # card without shortening the host-bound wall). Enough to collapse the per-candidate loop to a handful
        # of block launches, not enough to thrash. Adaptive (no hardcoded byte threshold).
        free_bytes = int(_dev_free) + min(max(0, _pool_free), int(_dev_total * 0.25))
    frac = _gpu_k_chunk_vram_fraction(n) if vram_fraction is None else float(vram_fraction)
    frac = min(0.9, max(1e-3, frac))
    budget = max(1, int(free_bytes * frac))
    wm = _GPU_MI_WORKING_MULTIPLE if working_multiple is None else int(working_multiple)
    per_col = max(1, int(n) * int(bytes_per_elem) * wm)
    cap = len(_COMBOS) if max_cols is None else int(max_cols)
    return int(min(cap, max(1, budget // per_col)))


def _binary_apply(xp, name, x, y):
    """Apply binary ``name`` to ``(x, y)`` mirroring ``feature_engineering.create_binary_transformations``
    EXACTLY (full catalog minus the temporarily-disabled agm/logn/binom - see _FULL_BINARY), incl. safe
    div's y==0 branch. Raises ValueError for an unknown/disabled op."""
    # --- minimal ---
    if name == "mul":
        return x * y
    if name == "add":
        return x + y
    if name == "sub":
        return x - y
    if name == "div":
        safe_y = xp.where(y == 0.0, 1e-9, y)
        return x / safe_y
    if name == "max":
        return xp.maximum(x, y)
    if name == "min":
        return xp.minimum(x, y)
    # --- medium ---
    if name == "abs_diff":
        return xp.abs(x - y)
    if name == "hypot":
        return xp.hypot(x, y)
    if name == "signed":  # sign(x)*|y| - non-symmetrical
        return xp.sign(x) * xp.abs(y)
    if name == "ratio_abs":  # x/(|y|+1) - non-symmetrical
        return x / (xp.abs(y) + 1.0)
    # --- maximal ---
    if name == "logaddexp":
        return xp.logaddexp(x, y)
    if name == "pow":  # non-symmetrical; negative^frac -> nan, scrubbed downstream (matches np.power)
        return xp.power(x, y)
    if name == "heaviside":
        return xp.heaviside(x, y)
    if name == "greater":
        return (x > y).astype(int)
    if name == "less":
        return (x < y).astype(int)
    if name == "equal":
        return (x == y).astype(int)
    if name == "beta":
        return _xp_special(xp).beta(x, y)
    # GPU-DISABLED(restore): uncomment these + the matching _FULL_BINARY entries + the
    # feature_engineering.py catalog lines to re-enable. agm needs a cupyx/custom kernel (absent here);
    # logn is np.emath.logn (complex base) - a real GPU form is log(y-ymin+0.1)/log(x-xmin+0.1); binom
    # is deferred with the batch.
    # if name == "agm":
    #     return _xp_special(xp).agm(x, y)
    # if name == "logn":
    #     xs = x - xp.min(x) + 0.1
    #     ys = y - xp.min(y) + 0.1
    #     return xp.log(ys) / xp.log(xs)
    # if name == "binom":
    #     return _xp_special(xp).binom(x, y)
    raise ValueError(f"unknown binary {name!r}")


def _candidate_names(a_label: str = "a", b_label: str = "b") -> list[str]:
    """Builds the human-readable ``bop(ua(a),ub(b))`` candidate name strings in the same nested-loop order as ``_COMBOS``/the candidate matrix columns."""
    return [f"{bop}({ua}({a_label}),{ub}({b_label}))" for ua in _MINIMAL_UNARY for ub in _MINIMAL_UNARY for bop in _MINIMAL_BINARY]


_FUSED_GEN_SRC = r"""
extern "C" __global__
void fused_gen(const double* __restrict__ ua, const double* __restrict__ ub,
               const int* __restrict__ ua_idx, const int* __restrict__ ub_idx,
               const int* __restrict__ bop, const long long n, const int K,
               double* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    int c = (int)(tid % (long long)K);
    long long i = tid / (long long)K;
    double x = ua[(long long)ua_idx[c] * n + i];
    double y = ub[(long long)ub_idx[c] * n + i];
    double v;
    switch (bop[c]) {
        case 0: v = x * y; break;
        case 1: v = x + y; break;
        case 2: v = x - y; break;
        case 3: v = x / ((y == 0.0) ? 1e-9 : y); break;
        case 4: v = (x > y) ? x : y; break;
        case 5: v = (x < y) ? x : y; break;
        default: v = 0.0;
    }
    if (isnan(v) || isinf(v)) v = 0.0;
    out[i * (long long)K + c] = v;
}
"""


_FUSED_GEN_KERNEL = None  # module-level singleton (lazy-compiled; never on an instance -> pickle-safe)


_FUSED_GEN_CM_SRC = r"""
extern "C" __global__
void fused_gen_cm(const double* __restrict__ ua, const double* __restrict__ ub,
                  const int* __restrict__ ua_idx, const int* __restrict__ ub_idx,
                  const int* __restrict__ bop, const long long n, const int K,
                  double* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    long long i = tid % n;
    int c = (int)(tid / n);
    double x = ua[(long long)ua_idx[c] * n + i];
    double y = ub[(long long)ub_idx[c] * n + i];
    double v;
    switch (bop[c]) {
        case 0: v = x * y; break;
        case 1: v = x + y; break;
        case 2: v = x - y; break;
        case 3: v = x / ((y == 0.0) ? 1e-9 : y); break;
        case 4: v = (x > y) ? x : y; break;
        case 5: v = (x < y) ? x : y; break;
        default: v = 0.0;
    }
    if (isnan(v) || isinf(v)) v = 0.0;
    out[(long long)c * n + i] = v;   // COLUMN-MAJOR (K, n): out_cm[c, i] == out_rm[i, c], c*n+i < K*n < 2^63
}
"""


_FUSED_GEN_CM_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe twin of _FUSED_GEN_KERNEL)


_FUSED_GEN_CM_F32_SRC = (
    _FUSED_GEN_CM_SRC.replace("fused_gen_cm", "fused_gen_cm_f32")
    .replace("const double*", "const float*")
    .replace("double* __restrict__ out", "float* __restrict__ out")
    .replace("double x", "float x")
    .replace("double y", "float y")
    .replace("double v", "float v")
    .replace("(y == 0.0) ? 1e-9 : y", "(y == 0.0f) ? 1e-9f : y")
    .replace("v = 0.0;", "v = 0.0f;")
)


_FUSED_GEN_CM_F32_KERNEL = None


def _get_fused_gen_kernel():
    """Lazily compiles and caches the row-major fused-generation ``RawKernel`` (one launch fills the whole (n,K) candidate matrix)."""
    global _FUSED_GEN_KERNEL
    if _FUSED_GEN_KERNEL is None:
        import cupy as cp

        _FUSED_GEN_KERNEL = cp.RawKernel(_FUSED_GEN_SRC, "fused_gen")
    return _FUSED_GEN_KERNEL


def _get_fused_gen_cm_kernel(float32: bool = False):
    """Lazily compiles and caches the column-major twin of ``_get_fused_gen_kernel`` (writes a (K,n) buffer for coalesced downstream reads); ``float32`` selects the float32 twin."""
    global _FUSED_GEN_CM_KERNEL, _FUSED_GEN_CM_F32_KERNEL
    import cupy as cp

    if float32:
        if _FUSED_GEN_CM_F32_KERNEL is None:
            _FUSED_GEN_CM_F32_KERNEL = cp.RawKernel(_FUSED_GEN_CM_F32_SRC, "fused_gen_cm_f32")
        return _FUSED_GEN_CM_F32_KERNEL
    if _FUSED_GEN_CM_KERNEL is None:
        _FUSED_GEN_CM_KERNEL = cp.RawKernel(_FUSED_GEN_CM_SRC, "fused_gen_cm")
    return _FUSED_GEN_CM_KERNEL


_FUSED_GEN_BIN_HIST_SRC = r"""
extern "C" __global__
void fused_gen_bin_hist(const double* __restrict__ ua, const double* __restrict__ ub,
                        const int* __restrict__ ua_idx, const int* __restrict__ ub_idx,
                        const int* __restrict__ bop,
                        const double* __restrict__ edges,   // (K, nbins-1) interior edges
                        const long long* __restrict__ col_off,  // (K,) histogram offset of column c
                        const int* __restrict__ y_all,      // (P1, n) int32 y-vectors
                        const long long n, const int K, const int nbins, const int K_y,
                        const int P1, const long long total_size,
                        long long* __restrict__ counts) {   // (P1, total_size) int64, host-zeroed
    // ONE BLOCK PER CANDIDATE COLUMN (roadmap #3 fused SHARED-MEM atomic histogram). The whole column's
    // (P1, nbins, K_y) joint histogram lives in shared memory as int32 (counts <= n < 2^31). Each thread
    // strides over rows: regenerates the candidate value ONCE, bins it ONCE, then does P1 SHARED-MEM
    // atomics (orders of magnitude cheaper than the global-mem atomics the naive version paid) - the
    // recompute-instead-of-store of Option F1 fused with the shared-atomic histogram of #3. The shared
    // tile is flushed to global counts once at the end. ``hist_size = P1 * nbins * K_y`` int32 elements
    // must fit shared memory; the host launches this path only when it does (else the global-atomic
    // fallback kernel). col_off[c] = c * nbins * K_y, so the column's global slot is contiguous.
    extern __shared__ int sh[];   // (P1, nbins, K_y) int32, dynamic shared
    int c = blockIdx.x;
    if (c >= K) return;
    int hist_size = P1 * nbins * K_y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    for (int s = tid; s < hist_size; s += nthreads) sh[s] = 0;
    __syncthreads();
    int ne = nbins - 1;
    const double* ec = edges + (long long)c * ne;
    int ua_c = ua_idx[c], ub_c = ub_idx[c], bop_c = bop[c];
    int nbky = nbins * K_y;
    for (long long i = tid; i < n; i += nthreads) {
        double x = ua[(long long)ua_c * n + i];
        double y = ub[(long long)ub_c * n + i];
        double v;
        switch (bop_c) {
            case 0: v = x * y; break;
            case 1: v = x + y; break;
            case 2: v = x - y; break;
            case 3: v = x / ((y == 0.0) ? 1e-9 : y); break;
            case 4: v = (x > y) ? x : y; break;
            case 5: v = (x < y) ? x : y; break;
            default: v = 0.0;
        }
        if (isnan(v) || isinf(v)) v = 0.0;
        // bin: searchsorted(edges[c], v, side='right')
        int lo = 0, hi = ne;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (ec[mid] > v) hi = mid; else lo = mid + 1; }
        int slot = lo * K_y;  // width-ok: bounded by hist_size, which must fit in dynamic shared memory
        for (int p = 0; p < P1; ++p) {
            int yp = y_all[(long long)p * n + i];
            atomicAdd(&sh[p * nbky + slot + yp], 1);
        }
    }
    __syncthreads();
    // flush shared int32 histogram to the global int64 counts (one column block per y-vector).
    for (int s = tid; s < hist_size; s += nthreads) {
        int p = s / nbky;
        int rem = s - p * nbky;
        counts[(long long)p * total_size + col_off[c] + rem] = (long long)sh[s];
    }
}
"""


_FUSED_GEN_BIN_HIST_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe)


def _get_fused_gen_bin_hist_kernel():
    """Lazily compiles and caches the grand-fusion RawKernel that generates, bins, and joint-histograms every candidate against all y-vectors in one launch, without ever materialising the (n,K) candidate matrix."""
    global _FUSED_GEN_BIN_HIST_KERNEL
    if _FUSED_GEN_BIN_HIST_KERNEL is None:
        import cupy as cp

        _FUSED_GEN_BIN_HIST_KERNEL = cp.RawKernel(_FUSED_GEN_BIN_HIST_SRC, "fused_gen_bin_hist")
    return _FUSED_GEN_BIN_HIST_KERNEL


def fe_gpu_grand_fusion_enabled() -> bool:
    """Whether the grand-fusion histogram path (never materialise (n,K)) is active. ON unless
    ``MLFRAME_FE_GPU_GRAND_FUSION`` is explicitly falsy (it is the selection-safe data-movement win;
    the non-fused ``grand_fused_pair_mi`` path stays one env flip away as the exact fallback)."""
    return os.environ.get("MLFRAME_FE_GPU_GRAND_FUSION", "1").strip().lower() in ("1", "true", "on", "yes")


def _fused_generate_block(ua_cm, ub_cm, combos_block, *, column_major: bool = False):
    """Generate the (n, len(combos_block)) candidate matrix for ``combos_block`` in ONE kernel launch.

    ``ua_cm`` / ``ub_cm`` are the (U, n) C-CONTIGUOUS post-unary caches for operands a / b, where
    U = len(_MINIMAL_UNARY) (row u = _UNARY_IDX[name]). This layout lets the kernel address column u
    via ``ua[u*n + i]``; the caller builds them ONCE and reuses across chunks. Returns a row-major
    (n, K) cupy float64 matrix, bit-equal to the cupy elementwise path (same ops, same safe-div, same
    nan_to_num - validated maxdiff 0).

    ``column_major=True`` runs the COLUMN-MAJOR twin kernel instead: same per-cell value, written into a
    (K, n) C-order buffer (``out_cm[c, i] == out_rm[i, c]``). Callers that only need a column-major view of
    the candidate values (e.g. the exact per-candidate percentile edges) take this to SKIP a downstream
    transpose / strided-axis sort - BIT-IDENTICAL values, only the layout differs. Consumers that read the
    matrix as (n, K) (the resident MI / discretize kernels) MUST keep the default row-major output."""
    import cupy as cp

    # Pin the operand-plane row count to the unary set: the kernel does NO bounds check on ua_idx, so a
    # silent row/index drift would be an out-of-bounds device read. Assert it can't.
    assert ua_cm.shape[0] == len(_MINIMAL_UNARY) == ub_cm.shape[0], (
        ua_cm.shape,
        ub_cm.shape,
    )  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters, not reachable with untrusted input
    n = int(ua_cm.shape[1])
    K = len(combos_block)
    # combos_block is always a slice of the module constant _COMBOS, so the int32 index trio depends only on
    # the block contents (identical across every pair at a fixed k_chunk). Cache the device vectors keyed on
    # the block tuple to drop the per-chunk-per-pair list-comp + tiny-H2D allocs (cupy._core.core.array).
    _ck = tuple(combos_block)
    with _GPU_RESIDENT_FE_CACHE_LOCK:
        _cc = _COMBO_IDX_CACHE.get(_ck)
        if _cc is None:
            ua_idx = cp.asarray(np.asarray([_UNARY_IDX[ua] for ua, _, _ in combos_block], dtype=np.int32))
            ub_idx = cp.asarray(np.asarray([_UNARY_IDX[ub] for _, ub, _ in combos_block], dtype=np.int32))
            bop = cp.asarray(np.asarray([_BINOP_CODE[bp] for _, _, bp in combos_block], dtype=np.int32))
            _COMBO_IDX_CACHE[_ck] = (ua_idx, ub_idx, bop)
        else:
            ua_idx, ub_idx, bop = _cc
    total = n * K
    threads = 256
    blocks = (total + threads - 1) // threads
    if column_major:
        out = cp.empty((K, n), dtype=cp.float64)  # (K, n) C-order == column-major over the (n, K) matrix
        _get_fused_gen_cm_kernel()((blocks,), (threads,), (ua_cm, ub_cm, ua_idx, ub_idx, bop, np.int64(n), np.int32(K), out))
        return out
    out = cp.empty((n, K), dtype=cp.float64)
    _get_fused_gen_kernel()((blocks,), (threads,), (ua_cm, ub_cm, ua_idx, ub_idx, bop, np.int64(n), np.int32(K), out))
    return out


def _unary_stack_cm(xp, x):
    """(U, n) C-contiguous stack of the minimal unary transforms of ``x`` (U=len(_MINIMAL_UNARY), row u = _UNARY_IDX[name]).

    Writes each unary DIRECTLY into a preallocated (U, n) buffer instead of ``xp.stack([...])`` (nsys-driven,
    2026-07-03): stack materialises ALL U unary temporaries in the list SIMULTANEOUSLY, then copies each into
    its output -> peak = U operand-plane temporaries + the output. The per-row assignment holds only ONE unary
    temporary at a time (peak = 1 + the output), dropping the stack copy AND the U-fold transient VRAM. The
    plane is C-contiguous by construction (``xp.empty`` default order), so the prior ``ascontiguousarray`` is a
    no-op removed. Row values are byte-identical to the stack path (same ``_unary_apply`` output, copied into
    row u). ``x`` is a float64 device operand here (both callers pass ``cp.asarray(.., float64)``); every
    minimal unary preserves that dtype, so ``plane`` dtype = ``x.dtype`` matches what stack would infer."""
    plane = xp.empty((len(_MINIMAL_UNARY), x.shape[0]), dtype=x.dtype)
    for u in _MINIMAL_UNARY:
        plane[_UNARY_IDX[u]] = _unary_apply(xp, u, x)  # copies the fresh unary result into its row slice
    return plane


def _build_candidate_matrix(xp, a, b):
    """Generate the full minimal unary x unary x binary candidate grid for operands ``a``, ``b`` as one
    contiguous ``(n, K)`` matrix in array module ``xp``. Non-finite cells -> 0 (the FE scrub). With ``xp``
    = cupy and ``a``/``b`` already device-resident, the WHOLE grid is built on the GPU with no transfer."""
    ua_cache = {u: _unary_apply(xp, u, a) for u in _MINIMAL_UNARY}
    ub_cache = {u: _unary_apply(xp, u, b) for u in _MINIMAL_UNARY}
    n = a.shape[0]
    K = len(_MINIMAL_UNARY) * len(_MINIMAL_UNARY) * len(_MINIMAL_BINARY)
    out = xp.empty((n, K), dtype=xp.float64)
    j = 0
    for ua in _MINIMAL_UNARY:
        for ub in _MINIMAL_UNARY:
            for bop in _MINIMAL_BINARY:
                col = _binary_apply(xp, bop, ua_cache[ua], ub_cache[ub])
                out[:, j] = xp.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
                j += 1
    return out


def cpu_pair_candidate_mi(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20) -> tuple[list[str], np.ndarray]:
    """Reference CPU path: build the grid in numpy + score with the production njit batch MI. Returns
    ``(names, mi)`` - the baseline the GPU-resident path must match (ranking + values to fp round-off)."""
    from mlframe.feature_selection.filters.hermite_fe.shared import plugin_mi_classif_batch_njit as _plugin_mi_classif_batch_njit

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b))
    mi = _plugin_mi_classif_batch_njit(cand, np.ascontiguousarray(y_codes, dtype=np.int64), nbins)
    return _candidate_names(), np.asarray(mi, dtype=np.float64)
