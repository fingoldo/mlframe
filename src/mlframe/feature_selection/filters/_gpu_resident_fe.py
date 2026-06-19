"""GPU-resident FE candidate generation + MI (prototype, gated, un-wired).

The terminal phase of the matrix-native FE replatform, and the only part with genuine NEW value: the
reason the GPU LOST the MI dispatch (see _hermite_fe_mi / the 2026-06-19 perf series) was the per-call
H2D upload + many tiny kernels -- ~700ms/call of pure overhead the on-device compute (~10-36ms) was
dwarfed by. The fix is to keep the data RESIDENT: upload the raw operands ONCE, generate the whole
unary x binary candidate grid ON the GPU (cupy elementwise), and score the entire grid in ONE big-k
batch-MI call. No per-candidate transfer, one large kernel -- exactly the regime the contention-aware
sweep showed the GPU winning (n=100k k>=100: cuda < njit).

GATED behind ``MLFRAME_FE_GPU_RESIDENT`` and imported by nothing in the production FE path: this is a
validated prototype proving the approach (correct MI vs the CPU path + faster at large n), not yet the
production recipe-integrated generator. It mirrors the MINIMAL unary/binary preset (enough to express
a**2/b and log(c)*sin(d)); the full catalog + recipe replay is the follow-up once the win is locked.

Non-pure op handling: ``smart_log`` shifts by the FULL-column nanmin (computed once on-device here, the
same anchor the CPU recipe freezes), ``div`` reproduces the exact ``y==0 -> eps`` branch -- so the
on-device candidate equals the CPU one to fp round-off.

BENCH (GTX 1050 Ti, K=384 minimal-preset candidates per pair, median of 3, warm; vs numpy-gen + njit
batch MI). Keeping data resident flips the GPU from the old 3x LOSER (per-call H2D path) to a WINNER
that SCALES -- and the VRAM-bounded K-chunk (``_gpu_k_chunk``, mirroring the CPU RAM governor on-device)
removes the large-n cliff entirely, with the on-device MI bit-matching the CPU path (argmax + values):
  * n=20k   : CPU 287ms   / GPU 379ms  -> 0.76x  (small n: GPU launch dominates -> dispatcher routes CPU)
  * n=100k  : CPU 1771ms  / GPU 854ms  -> 2.07x  (k_chunk=141)
  * n=300k  : CPU 7013ms  / GPU 2046ms -> 3.43x  (k_chunk=47 -- was a 0.12x cliff before chunking)
  * n=1M    : CPU 29731ms / GPU 6424ms -> 4.63x  (k_chunk=14; the win GROWS with n)
``pair_candidate_mi_dispatch`` routes >= _GPU_RESIDENT_MIN_N (50k) to the chunked GPU path, CPU below.

KERNEL-TUNING INVESTIGATION (2026-06-19, GTX 1050 Ti, the on-device MI = cupy, not numba.cuda). The
on-device MI breaks down (n=1M, k=14) as: ``cp.argsort`` quantile-binning = 161ms (69%), histogram+MI
math = 73ms (31%). So the FULL O(n log n) sort cupy uses for equi-frequency binning is the dominant
cost and the obvious tuning target. Two sort-free replacements were prototyped + measured:
  * equi-WIDTH sub-histogram -> CDF -> quantile edges (no sort): 4.36x faster binning (43 vs 189ms) BUT
    BREAKS on heavy-tailed FE candidates (a**2/b's outliers stretch the range so ~all mass collapses to
    bin 0): bin-code agreement 98.6%, but candidate-MI Spearman only 0.88, MI maxdiff ~1.9, argmax flips
    -> REJECTED (would change selection).
  * monotone tail-compressed (``sign(x)*log1p|x|``, rank-preserving so quantiles are invariant) THEN
    equi-width sub-hist: Spearman 0.9993, MI maxdiff 0.029 -- statistically excellent and still 4.36x
    faster binning. But the TOP candidate is a near-tie among EQUIVALENT a**2/b spellings, and a 0.03 MI
    perturbation reorders that tie -> argmax flips -> NOT bit-exact -> unsafe as a drop-in for the
    exact-result contract. Good as an opt-in APPROXIMATE fast mode only.
The exact+fast path (NEXT): prescreen all candidates with the sort-free MI, then re-score only the
top-K (margin-guarded) with the exact ``cp.argsort`` MI -- exact winner at a fraction of the sort cost.
Numbers recorded so this is not re-derived blind; the exact ``cp.argsort`` path stays the default.
"""
from __future__ import annotations

import os

import numpy as np

# Minimal-preset op NAMES (kept in sync with feature_engineering.create_*_transformations "minimal").
_MINIMAL_UNARY = ("identity", "neg", "abs", "sqr", "reciproc", "sqrt", "log", "sin")
_MINIMAL_BINARY = ("mul", "add", "sub", "div", "max", "min")


def fe_gpu_resident_enabled() -> bool:
    """Whether the GPU-resident FE prototype is active. OFF unless ``MLFRAME_FE_GPU_RESIDENT`` truthy."""
    return os.environ.get("MLFRAME_FE_GPU_RESIDENT", "").strip().lower() in ("1", "true", "on", "yes")


def _unary_apply(xp, name, x):
    """Apply unary ``name`` to ``x`` using array module ``xp`` (numpy or cupy). Semantics mirror
    feature_engineering's minimal preset exactly (incl. smart_log's full-column nanmin shift)."""
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
    raise ValueError(f"unknown unary {name!r}")


def _binary_apply(xp, name, x, y):
    """Apply binary ``name`` to ``(x, y)`` mirroring the minimal preset (incl. safe div's y==0 branch)."""
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
    raise ValueError(f"unknown binary {name!r}")


def _candidate_names(a_label: str = "a", b_label: str = "b") -> list[str]:
    return [
        f"{bop}({ua}({a_label}),{ub}({b_label}))"
        for ua in _MINIMAL_UNARY for ub in _MINIMAL_UNARY for bop in _MINIMAL_BINARY
    ]


# (ua, ub, bop) combo order, matching _candidate_names / _build_candidate_matrix column order.
_COMBOS = [(ua, ub, bop) for ua in _MINIMAL_UNARY for ub in _MINIMAL_UNARY for bop in _MINIMAL_BINARY]

_UNARY_IDX = {u: i for i, u in enumerate(_MINIMAL_UNARY)}
_BINOP_CODE = {"mul": 0, "add": 1, "sub": 2, "div": 3, "max": 4, "min": 5}

# FUSED-GENERATION CUDA RawKernel: one launch computes the WHOLE (n, K) candidate block from the cached
# post-unary columns, replacing the Python loop of ~K separate cupy binary ops + nan_to_num + temporaries.
# Each thread owns one (row, candidate) cell: gather its two operand columns by op-code index, apply the
# binary op (safe-div mirrors the CPU y==0 -> eps branch), scrub non-finite to 0, write row-major (n,K).
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


def _get_fused_gen_kernel():
    global _FUSED_GEN_KERNEL
    if _FUSED_GEN_KERNEL is None:
        import cupy as cp
        _FUSED_GEN_KERNEL = cp.RawKernel(_FUSED_GEN_SRC, "fused_gen")
    return _FUSED_GEN_KERNEL


def _fused_generate_block(ua_cm, ub_cm, combos_block):
    """Generate the (n, len(combos_block)) candidate matrix for ``combos_block`` in ONE kernel launch.

    ``ua_cm`` / ``ub_cm`` are the (16, n) C-CONTIGUOUS post-unary caches for operands a / b (row
    u = _UNARY_IDX[name]) -- this layout lets the kernel read ``ua[u*n + i]`` coalesced; the caller
    builds them ONCE and reuses across chunks. Returns a row-major (n, K) cupy float64 matrix, bit-equal
    to the cupy elementwise path (same ops, same safe-div, same nan_to_num -- validated maxdiff 0)."""
    import cupy as cp

    n = int(ua_cm.shape[1])
    K = len(combos_block)
    ua_idx = cp.asarray([_UNARY_IDX[ua] for ua, _, _ in combos_block], dtype=cp.int32)
    ub_idx = cp.asarray([_UNARY_IDX[ub] for _, ub, _ in combos_block], dtype=cp.int32)
    bop = cp.asarray([_BINOP_CODE[bp] for _, _, bp in combos_block], dtype=cp.int32)
    out = cp.empty((n, K), dtype=cp.float64)
    total = n * K
    threads = 256
    blocks = (total + threads - 1) // threads
    _get_fused_gen_kernel()((blocks,), (threads,), (ua_cm, ub_cm, ua_idx, ub_idx, bop, np.int64(n), np.int32(K), out))
    return out


def _unary_stack_cm(xp, x):
    """(16, n) C-contiguous stack of the minimal unary transforms of ``x`` (row u = _UNARY_IDX[name])."""
    return xp.ascontiguousarray(xp.stack([_unary_apply(xp, u, x) for u in _MINIMAL_UNARY], axis=0))

# Per-element GPU working-set multiple for the cupy plug-in MI: the (n, k) cand f64 + argsort int64 +
# X_binned int64 + flat int64 coexist, so budget ~5x the bare cand bytes. Conservative -> avoids the
# n=300k VRAM cliff (measured: unchunked (300k,384) thrashed the 4GB card to 60s).
_GPU_MI_WORKING_MULTIPLE = 5
# Below this n the GPU launch/transfer dominates and the CPU njit grid wins (bench: 20k -> 0.76x,
# 100k -> 1.79x); the dispatcher routes < this to CPU. Provisional crossover; a later sweep can tune it.
_GPU_RESIDENT_MIN_N = 50_000


def _gpu_k_chunk(n: int, *, free_bytes: int | None = None) -> int:
    """Max candidate columns to materialise+score in ONE on-device batch so the cupy MI working set
    stays within a fraction of free VRAM -- bounds peak GPU memory the same way the CPU RAM governor
    bounds host memory, removing the large-n cliff."""
    import cupy as cp

    if free_bytes is None:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
    budget = max(1, int(free_bytes * 0.25))
    per_col = max(1, int(n) * 8 * _GPU_MI_WORKING_MULTIPLE)
    return int(min(len(_COMBOS), max(1, budget // per_col)))


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


def cpu_pair_candidate_mi(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20):
    """Reference CPU path: build the grid in numpy + score with the production njit batch MI. Returns
    ``(names, mi)`` -- the baseline the GPU-resident path must match (ranking + values to fp round-off)."""
    from .hermite_fe import _plugin_mi_classif_batch_njit

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b))
    mi = _plugin_mi_classif_batch_njit(cand, np.ascontiguousarray(y_codes, dtype=np.int64), nbins)
    return _candidate_names(), np.asarray(mi, dtype=np.float64)


def gpu_resident_pair_candidate_mi(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20):
    """GPU-RESIDENT path: upload ``a``, ``b``, ``y`` ONCE, build the whole candidate grid on the device,
    and score it in ONE big-k batch-MI call -- the array never round-trips per candidate. Returns
    ``(names, mi)`` with ``mi`` brought back to host (the only D2H, a (K,) vector). Raises if cupy is
    unavailable (callers gate on :func:`fe_gpu_resident_enabled` + availability)."""
    import cupy as cp

    from . import hermite_fe as _hf  # noqa: F401 -- full-init the parent first so the direct
    # ``_hermite_fe_mi`` import below can't trip the _ensure_cuda_kernels back-import cycle.
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda

    a_gpu = cp.asarray(a, dtype=cp.float64)   # the ONE H2D of the raw operands
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    # (16, n) unary caches stay resident + reused across every chunk; the FUSED kernel then generates
    # each candidate CHUNK in ONE launch (vs a Python loop of ~K cupy binary ops + nan_to_num + temps --
    # bit-equal, ~15x faster generation). Only the chunk matrix is bounded, so peak VRAM is governed.
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)
    k_chunk = _gpu_k_chunk(n)
    mi_parts: list[np.ndarray] = []
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start:start + k_chunk]
        cand = _fused_generate_block(ua_cm, ub_cm, block)   # one-launch fused generation
        # _plugin_mi_classif_batch_cuda's cp.asarray is a no-op for an already-device array -> no extra
        # transfer; one big-k kernel scores the resident chunk.
        mi_parts.append(np.asarray(_plugin_mi_classif_batch_cuda(cand, y_i64, nbins), dtype=np.float64))
        del cand
    return _candidate_names(), np.concatenate(mi_parts) if mi_parts else np.empty(0)


def _sortfree_mi_gpu(cand_gpu, y_i64, nbins, *, sub: int = 4096):
    """On-device APPROXIMATE plug-in MI for an (n, k) cupy candidate block, with NO sort: bin via a
    monotone tail-compressed (``sign*log1p``, rank-preserving so equi-frequency quantiles are invariant)
    equi-width sub-histogram -> CDF -> quantile edges, then the same joint-histogram MI. Spearman ~0.999
    vs the exact argsort MI but ~4.4x faster on the binning step -- used only to PRESCREEN candidates."""
    import math

    import cupy as cp

    n, k = cand_gpu.shape
    Xt = cp.sign(cand_gpu) * cp.log1p(cp.abs(cand_gpu))   # monotone -> preserves ranks/quantiles
    mn = Xt.min(axis=0); mx = Xt.max(axis=0)
    rng = cp.where(mx > mn, mx - mn, 1.0)
    sb = cp.minimum(((Xt - mn) / rng * sub).astype(cp.int32), sub - 1)
    hist = cp.bincount((cp.arange(k)[None, :] * sub + sb).ravel(), minlength=k * sub).reshape(k, sub).astype(cp.float64)
    cdf = cp.cumsum(hist, axis=1)
    targets = cp.arange(1, nbins) / nbins * n
    yg = cp.asarray(y_i64); ymin = int(yg.min()); yg = yg - ymin; nc = int(yg.max()) + 1
    Xb = cp.empty((n, k), dtype=cp.int64)
    for j in range(k):
        e = cp.searchsorted(cdf[j], targets, side="left")
        Xb[:, j] = cp.searchsorted(e.astype(cp.int32), sb[:, j], side="right")
    flat = ((cp.arange(k)[None, :] * nbins + Xb) * nc + yg[:, None]).ravel()
    h = cp.bincount(flat, minlength=k * nbins * nc).reshape(k, nbins, nc).astype(cp.float64)
    hx = h.sum(2); hy = h.sum(1); ln = math.log(n); m = h > 0
    term = (h / n) * (cp.log(cp.where(m, h, 1.0)) + ln
                      - cp.log(cp.where(hx > 0, hx, 1.0))[:, :, None]
                      - cp.log(cp.where(hy > 0, hy, 1.0))[:, None, :])
    return cp.maximum(cp.where(m, term, 0.0).sum(axis=(1, 2)), 0.0)


def gpu_resident_pair_candidate_mi_fast(a, b, y_codes, *, nbins: int = 20, refine_k: int = 48):
    """EXACT-winner, faster GPU-resident pair MI: prescreen ALL candidates with the sort-free MI (no
    O(n log n) sort), then re-score only the top ``refine_k`` with the EXACT argsort MI. The exact
    winner is preserved (it sits in the high-approx-MI tie cluster of equivalent a**2/b spellings, well
    within top-K), at a fraction of the full-grid sort cost. Returns ``(names, mi)`` where the top-K
    entries carry EXACT MI and the tail carries the (excellent, Spearman ~0.999) approx MI -- so argmax
    and the high-MI head are exact while the low-MI tail stays cheap."""
    import cupy as cp

    from . import hermite_fe as _hf  # noqa: F401 -- full-init parent before the _hermite_fe_mi import
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda

    a_gpu = cp.asarray(a, dtype=cp.float64)
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    ua_cache = {u: _unary_apply(cp, u, a_gpu) for u in _MINIMAL_UNARY}
    ub_cache = {u: _unary_apply(cp, u, b_gpu) for u in _MINIMAL_UNARY}

    def _col(idx):
        ua, ub, bop = _COMBOS[idx]
        return cp.nan_to_num(_binary_apply(cp, bop, ua_cache[ua], ub_cache[ub]), nan=0.0, posinf=0.0, neginf=0.0)

    # PRESCREEN: sort-free approx MI over all candidates, VRAM-chunked.
    k_chunk = _gpu_k_chunk(n)
    approx = np.empty(len(_COMBOS), dtype=np.float64)
    for start in range(0, len(_COMBOS), k_chunk):
        idxs = range(start, min(start + k_chunk, len(_COMBOS)))
        block = cp.empty((n, len(idxs)), dtype=cp.float64)
        for jj, idx in enumerate(idxs):
            block[:, jj] = _col(idx)
        approx[start:start + len(idxs)] = cp.asnumpy(_sortfree_mi_gpu(block, y_i64, nbins))
        del block
    # REFINE: exact argsort MI on the top-refine_k by approx MI.
    k = min(int(refine_k), len(_COMBOS))
    top = np.argsort(approx)[-k:]
    refine_mat = cp.empty((n, k), dtype=cp.float64)
    for jj, idx in enumerate(top):
        refine_mat[:, jj] = _col(int(idx))
    exact_top = np.asarray(_plugin_mi_classif_batch_cuda(refine_mat, y_i64, nbins), dtype=np.float64)
    mi = approx.copy()
    mi[top] = exact_top   # exact MI for the head, approx for the cheap tail
    return _candidate_names(), mi
    # MEASURED (GTX 1050 Ti, K=384, refine_k=48): exact winner preserved 6/6 seeds @ n=100k, but the
    # end-to-end speedup over the pure-exact GPU path is only ~1.16x @100k / ~1.06x @1M -- NOT the ~2x the
    # MI-only argsort=69% microbench implied. Once candidate GENERATION + the histogram MATH (paid over
    # all 384 in the prescreen) are counted, trimming only the argsort to the top-48 saves less than
    # argsort's in-kernel share. Real + exact, but modest; the bigger lever is cutting generation/math
    # (or a fused sort-free EXACT kernel), not just the sort. Kept as a validated option, not the default.


def pair_candidate_mi_dispatch(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20):
    """Route a pair's candidate-MI to the measured-fastest backend: GPU-resident in the sweet spot
    (cupy present, n >= the crossover, VRAM-chunked so it can't thrash), CPU njit otherwise. Returns
    ``(names, mi)`` identical in shape/order to both paths. The default FE pipeline does NOT call this
    yet (gated prototype); it is the dispatcher the production wiring will use."""
    n = int(np.asarray(a).shape[0])
    if n >= _GPU_RESIDENT_MIN_N:
        try:
            import cupy  # noqa: F401

            return gpu_resident_pair_candidate_mi(a, b, y_codes, nbins=nbins)
        except Exception:
            pass  # no cupy / GPU error -> CPU fallback
    return cpu_pair_candidate_mi(a, b, y_codes, nbins=nbins)
