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
    # Unary results (16 columns) stay resident + reused across every binary combo; only the candidate
    # CHUNK matrix is bounded, so peak VRAM is governed (no large-n cliff).
    ua_cache = {u: _unary_apply(cp, u, a_gpu) for u in _MINIMAL_UNARY}
    ub_cache = {u: _unary_apply(cp, u, b_gpu) for u in _MINIMAL_UNARY}
    k_chunk = _gpu_k_chunk(n)
    mi_parts: list[np.ndarray] = []
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start:start + k_chunk]
        cand = cp.empty((n, len(block)), dtype=cp.float64)
        for j, (ua, ub, bop) in enumerate(block):
            col = _binary_apply(cp, bop, ua_cache[ua], ub_cache[ub])
            cand[:, j] = cp.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
        # _plugin_mi_classif_batch_cuda's cp.asarray is a no-op for an already-device array -> no extra
        # transfer; one big-k kernel scores the resident chunk.
        mi_parts.append(np.asarray(_plugin_mi_classif_batch_cuda(cand, y_i64, nbins), dtype=np.float64))
        del cand
    return _candidate_names(), np.concatenate(mi_parts) if mi_parts else np.empty(0)


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
