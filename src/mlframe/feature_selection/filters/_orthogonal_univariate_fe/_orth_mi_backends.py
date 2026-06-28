"""Batch MI(X_j; y) backends for the orthogonal-basis FE selectors.

Two interchangeable implementations -- the sklearn ``mutual_info_score``
reference loop and the numba/cupy batch dispatcher routed through
``hermite_fe.plugin_mi_classif_batch_dispatch`` -- plus the import-time
backend chooser (`_select_mi_backend` / `_MI_BACKEND`) and the public
``_mi_classif_batch`` entry point the orth-FE family and many sibling
modules import.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _mi_classif_batch_sklearn(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Per-column quantile-bin + sklearn ``mutual_info_score`` reference path.

    Kept as the fallback when numba is unavailable AND when the caller
    explicitly opts out via ``MLFRAME_NUMBA_MI=0``. Returns MI in nats.
    """
    from sklearn.metrics import mutual_info_score
    n, p = X.shape
    mis = np.zeros(p, dtype=np.float64)
    for j in range(p):
        col = X[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            mis[j] = 0.0
            continue
        col_f = col[finite]
        try:
            edges = np.quantile(col_f, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
            edges = np.unique(edges)
            if edges.size == 0:
                mis[j] = 0.0
                continue
            binned = np.searchsorted(edges, col_f)
            mis[j] = float(mutual_info_score(binned, y[finite]))
        except Exception:
            mis[j] = 0.0
    return mis


def _orth_mi_gpu_enabled() -> bool:
    """STRICT-GPU full-coverage mode: route the orth-FE batch MI through the resident-GPU plugin-MI so the
    whole FE path runs on the device (MLFRAME_FE_GPU_STRICT / MLFRAME_CMI_GPU). Default OFF -> njit
    dispatcher. STRICT is a diagnostic FULL-GPU mode (every gateable kernel on the device for nsys GPU-load
    profiling), not a wall-optimised default -- the per-call H2D of host-built candidates is paid here; the
    wall win needs born-on-device candidates."""
    import os as _os
    if _os.environ.get("MLFRAME_CMI_GPU", "") == "1":
        return True
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled
        return bool(fe_gpu_strict_enabled())
    except Exception:
        return False


def _fe_edge_binning_enabled() -> bool:
    """Whether the CPU orth-MI uses PERCENTILE-EDGE binning (``MLFRAME_FE_EDGE_BINNING``, default OFF -> the
    legacy RANK binning). Edge matches the GPU twin on tied columns too (full CPU==GPU), but on tie-bearing
    fixtures it perturbs a razor-edge redundancy decision (the over-drop pin), so it is gated until the
    redundancy/admission gate is hardened to absorb a sub-noise MI perturbation. See gate-hardening."""
    import os as _os
    return _os.environ.get("MLFRAME_FE_EDGE_BINNING", "").strip().lower() in ("1", "true", "on", "yes")


def _mi_classif_batch_numba(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Numba prange batch MI(X_j; y) for classification.

    Defers to ``plugin_mi_classif_batch_dispatch`` from ``hermite_fe``, which routes (n, k) to the njit
    prange kernel (CPU, argsort equi-frequency RANK binning) or cupy batch kernel (GPU) via the kernel
    tuning cache. Bench at p=200 n=2000: ~6ms vs ~317ms for the per-column sklearn loop (~53x speedup).

    BINNING CHOICE (2026-06-26): the CPU default stays RANK binning. The GPU STRICT-residency twin
    ``_plugin_mi_classif_batch_cuda_resident`` uses PERCENTILE-EDGE binning, and its bit-faithful CPU twin
    ``_fe_edge_mi.plugin_mi_classif_batch_edge_njit`` is proven equal to it to ~1e-9 on continuous AND tied
    columns (``test_fe_edge_mi_parity``). Edge and rank agree bit-for-bit ONLY on tie-free data; on tied
    columns they diverge (rank splits equal values across a bin boundary, edge keeps them together). It is
    TEMPTING to switch this default to edge so the CPU and GPU FE paths bin identically -- but doing so
    regresses the canonical OVER-DROP pin (``test_private_raw_a_kept_alongside_engineered_multi_seed``,
    seeds 1-4): on that fixture the ~1e-12 rank-vs-edge MI perturbation, even on continuous data, is enough
    to push a SPURIOUS cross-signal form ``mul(qubed(a),cbrt(c))`` over the redundancy admission gate, which
    then drops the genuine private-signal raw ``a`` (verified by causality A/B). The GPU-edge path has the
    same latent fragility. True CPU==GPU identity therefore needs the redundancy/admission gate HARDENED
    with a tolerance band (so a sub-noise MI perturbation cannot flip the decision), tracked separately;
    until then the CPU default keeps rank to avoid the quality regression, and the edge twin is reserved for
    the FE batcher path where both backends bin edge-identically. See ``_fe_mi_contract``.

    Handles partial-NaN columns by masking to the finite subset per column,
    matching ``_mi_classif_batch_sklearn`` semantics. An all-NaN column or a
    column where every value collapses to a single bin returns 0.0.
    """
    from ..hermite_fe import plugin_mi_classif_batch_dispatch

    n, p = X.shape
    y_i64 = np.ascontiguousarray(y, dtype=np.int64)
    mis = np.zeros(p, dtype=np.float64)
    # Partition columns into "all-finite" (bulk path) and "partial-NaN"
    # (per-column fallback). In the hybrid_orth_mi_fe production path the
    # source frames are nan-filled upstream so partial_idx is empty and
    # everything goes through the single batch dispatch call.
    finite_per_col = np.isfinite(X).all(axis=0)
    dense_cols = np.where(finite_per_col)[0]
    partial_cols = np.where(~finite_per_col)[0]

    if dense_cols.size:
        # When EVERY column is finite (the production nan-filled path), the
        # ``X[:, dense_cols]`` fancy-index is a full (n, p) gather COPY that
        # reproduces X verbatim -- skip it and hand the (already-contiguous)
        # frame straight to the batch kernel. On a 40k x 200 all-finite frame
        # this setup dropped 3109ms -> 212ms (~14.6x) across 23 calls; the
        # gather copy was the entire self-time. Partial-NaN columns still take
        # the real gather below.
        if dense_cols.size == p:
            X_dense = np.ascontiguousarray(X)
        else:
            X_dense = np.ascontiguousarray(X[:, dense_cols])
        try:
            if _orth_mi_gpu_enabled():
                # Full-residency GPU path -> the FE batcher: VRAM-budget column-chunked, CP-SAT-packed across
                # heterogeneous GPUs (multi_gpu_fe_batch_mi collapses to one device on a 1-GPU host). Same
                # resident edge-binned plug-in MI as the prior direct _plugin_mi_classif_batch_cuda_resident
                # call, so selection is unchanged vs the pre-batcher STRICT path; it only adds VRAM-chunking
                # + multi-GPU. The CPU default branch below stays RANK binning (see the docstring).
                from .._fe_gpu_batch import multi_gpu_fe_batch_mi
                from .._fe_gpu_batch._devices import fe_gpu_f32_enabled
                # X_dense is the FINITE-filtered dense subset (finite_per_col) -> scrub=False skips cupy's
                # full-array nan scan (~12% of the GPU MI wall) that would otherwise be a pure no-op cost here.
                # f32 opt-in (MLFRAME_FE_VRAM_F32): ~2.2x faster, selection-equivalent (Spearman rank 1.0).
                _dt = np.float32 if fe_gpu_f32_enabled() else np.float64
                mis[dense_cols] = multi_gpu_fe_batch_mi(X_dense, y_i64, nbins, scrub=False, dtype=_dt)
            elif _fe_edge_binning_enabled():
                from .._fe_edge_mi import plugin_mi_classif_batch_edge_njit
                mis[dense_cols] = plugin_mi_classif_batch_edge_njit(X_dense, y_i64, nbins)
            else:
                mis_dense = plugin_mi_classif_batch_dispatch(X_dense, y_i64, nbins)
                mis[dense_cols] = mis_dense
        except Exception:
            # If the batch path fails for any reason (cupy import error,
            # kernel tuning miss, etc.), fall back to sklearn for the
            # affected slice rather than poisoning the whole call.
            mis[dense_cols] = _mi_classif_batch_sklearn(
                X_dense, y_i64, nbins=nbins,
            )

    if partial_cols.size:
        # Partial-NaN columns get the per-column path (mask + sklearn). The
        # production hybrid path nan-fills before calling MI so this branch
        # is essentially dead code in practice; keep it for API parity.
        for j in partial_cols:
            col = X[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                mis[j] = 0.0
                continue
            col_f = np.ascontiguousarray(col[finite].reshape(-1, 1))
            y_f = np.ascontiguousarray(y_i64[finite])
            try:
                mis[j] = float(
                    plugin_mi_classif_batch_dispatch(col_f, y_f, nbins)[0],
                )
            except Exception:
                mis[j] = 0.0
    return mis


# Module-import-time decision: which backend does ``_mi_classif_batch`` use?
# - ``MLFRAME_NUMBA_MI=0``  -> force sklearn loop reference
# - ``MLFRAME_NUMBA_MI=1``  -> force numba batch (raises at first call if numba missing)
# - unset / any other value -> auto: numba batch when ``hermite_fe.plugin_mi_classif_batch_dispatch``
#   imports cleanly (the standard case in this repo), sklearn otherwise.
# Cached because hybrid_orth_mi_fe calls _mi_classif_batch twice per fit and
# the dispatcher decision is constant per process.
def _select_mi_backend() -> str:
    import os as _os
    flag = _os.environ.get("MLFRAME_NUMBA_MI", "").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return "sklearn"
    if flag in ("1", "true", "on", "yes"):
        return "numba"
    # auto: try-import the numba dispatcher; on failure (e.g. numba absent
    # in a stripped-down install) fall back to sklearn rather than crashing
    # at first call.
    try:
        from ..hermite_fe import plugin_mi_classif_batch_dispatch  # noqa: F401
        return "numba"
    except Exception:
        return "sklearn"


_MI_BACKEND = _select_mi_backend()


def _mi_classif_batch(X: np.ndarray, y: np.ndarray, *, nbins: int = 10, rank_binning: bool = False) -> np.ndarray:
    """Batch MI(X_j; y) for classification target.

    ``rank_binning`` (GATE MI ONLY, default False): the STRICT-residency MI core bins by percentile EDGES,
    which does NOT byte-match the CPU njit RANK binning the conditional-gate scoring uses on heavily-tied
    columns (the gate_mask output ``1[c>0]*a`` is ~50% exact zeros). When ``rank_binning=True`` AND the
    resident opt-in ``MLFRAME_FE_GPU_STRICT_RESIDENT`` is on, the resident MI is computed over RANK codes
    (argsort equi-frequency, ``_gpu_resident_rank_bin``) so the gate STRICT MI matches the CPU rank MI. The
    FE-candidate path leaves this False -> the edge path is untouched. Falls back to the CPU njit rank path on
    any GPU failure; default flag-off is byte-for-byte unchanged regardless of this flag.

    Layer 31 (2026-05-31): routes to the numba prange batch dispatcher
    (``_mi_classif_batch_numba``) when available — ~53x speedup at
    p=200 n=2000 over the per-column sklearn loop, bit-equivalent to within
    machine epsilon (< 2e-15 across 40 seeds). Set ``MLFRAME_NUMBA_MI=0``
    to force the sklearn reference if a downstream regression demands it.

    Idea #18 (2026-06-10) -- bench-rejected, default OFF: an inverse-prior
    class-balanced MI was added to test whether plain plug-in MI under-RANKS
    rare-class-discriminative features under imbalance. It does NOT: balancing is
    a near-uniform multiplicative rescale (Kendall tau 0.989 vs plain), so it
    almost never changes the rank-based selection, and where it does (13/120
    imbalanced frames) the downstream rare-class AP is a net-negative coin-flip
    (mean dAP -0.0037). Kept opt-in via ``MLFRAME_FE_IMBALANCE_MI=on`` (default
    ``off`` => this branch is skipped and the path is byte-for-byte the plain MI
    below). Full numbers in ``_imbalance_mi`` module docstring + the regression
    test ``tests/feature_selection/test_imbalance_mi.py``.
    """
    # Fast OFF short-circuit (the default): a single env read, no import / no
    # bincount, so the common path is byte-for-byte and ~free vs plain numba.
    import os as _os
    if _os.environ.get("MLFRAME_FE_IMBALANCE_MI", "").strip().lower() in ("on", "1", "true", "yes", "auto"):
        class_w = _maybe_class_weights(y)  # opt-in only
        if class_w is not None:
            cb = _mi_classif_batch_balanced(X, y, class_w, nbins=nbins)
            if cb is not None:
                return cb
    # bench-attempt-rejected (2026-06-23, GTX 1050 Ti, F2 100k MRMR wall /loop "drive CPU share -> 0"):
    # tried routing this CPU njit batch-MI (the #1 mlframe CPU-compute kernel: cProfile tottime 3.05s of a
    # 34.8s warm F2 100k fit, 157 calls via plugin_mi_classif_batch_dispatch, callers = _conditional_gate_fe
    # best_existing_op_mi / _gate_grid_mi ~2.1s + _pairwise_modular_fe._mi 0.35s + _unified_fe_gate, each
    # building a FRESH host candidate matrix one-shot) to its GPU twin _plugin_mi_classif_batch_cuda.
    # ISOLATED A/B (fresh-host-array, WITH H2D) at n=100k showed GPU ~2-3x FASTER (k=14 45.8->20.1ms,
    # k=5 28.5->8.1ms, k=30 106->39ms) -- but END-TO-END is a LOSS: MLFRAME_MI_BACKEND=cuda (forces every
    # batch dispatch to GPU) measured 34.8s -> 36.0s wall. The per-call H2D of each freshly-built host
    # matrix + GPU contention with the already-resident pair kernels eats the isolated speedup (the classic
    # "isolated-kernel microbench lies" trap; see plugin_mi_classif_batch_dispatch ground-truth note).
    # Selection WAS byte-identical under the force (F2 100k recipe hash 962a4c7b / produced e92339f7 both
    # match njit -- the percentile-edge GPU binning is selection-equivalent here despite ~1e-4 ULP MI drift),
    # so the ONLY blocker is the wall regression. A real win needs the candidate matrices GPU-RESIDENT
    # (eliminating the per-call H2D the microbench omits) -- the matrix-native FE replatform, tracked
    # separately; a dispatch flag alone cannot win.
    # residency-blocker quantified (2026-06-23, MRMR FE wall /loop iter14 -- "complete end-to-end GPU
    # residency for the dominant CPU MI"): the FE-PAIR path is ALREADY fully resident (verified: 47/47
    # _dispatch_batch_mi_with_noise_gate calls hit take_resident_codes -> batch_mi_with_noise_gate_cuda_-
    # resident with device codes in place, 0 H2D), so the pair-MI is NOT the remaining CPU cost. The #1 CPU
    # kernel _plugin_mi_classif_batch_njit (2.94s tottime / 34.7s wall, 157 calls) is UNROUTABLE to a resident
    # GPU MI for TWO independent structural reasons, measured this iter:
    #   (1) NO DEVICE CODES TO HAND OFF. Every one of the 157 candidate matrices is built on the HOST with
    #       numpy (best_existing_op_mi: u*v / u-v / u/(|v|+eps) / u+v / stack.max|min|sum via np.column_stack;
    #       _gate_grid_mi / cheap_conditional_gate_scan likewise). The candidates were NEVER on the GPU, so
    #       there is no resident handoff to extend -- "residency" presupposes device data this path doesn't
    #       have. Making it resident = porting the host numpy candidate GENERATION to cupy, a far larger change
    #       than the pair path's (which already binned on-device).
    #   (2) CALL SHAPE IS BELOW THE GPU CROSSOVER even at ZERO transfer cost. Measured k-distribution of the
    #       157 calls (warm F2 100k seed-7): k=1 x96 (0.33s, the partial-NaN per-column fallback), k<=18 x53
    #       (1.15s, 21.7ms/call), k<=40 x4 (0.29s), k>40 x4 (1.17s: k=306 x1, k=527 x2, k=80 x1). The resident
    #       GPU MI crossover on this HW is k>=100 @ n=100k (see _gpu_resident_fe BENCH). So 153/157 calls
    #       (1.77s of the 2.94s) are sub-crossover -- a per-call GPU launch+sync would LOSE on them regardless
    #       of H2D. Only the 4 large-k calls (1.17s, 40%) are GPU-favorable, and those are exactly the
    #       host-built orth-univariate candidates of (1) AND the shapes the flag-route A/B above already
    #       benched to a 34.8->36.0s end-to-end LOSS (per-call H2D + same-GPU contention with the resident pair
    #       kernels). NET: there is no resident-MI win available here; the only lever would be a from-scratch
    #       cupy candidate-generation replatform of the orth-univariate FE families (the matrix-native FE work,
    #       tracked separately), and even then only the ~4 large-k calls could win. The 2.94s is irreducible
    #       CPU on this path absent that replatform. cProfile remainder (34.7s wall): _plugin_mi 2.94s,
    #       gpu_materialise_discretize_codes_host 1.12s (GPU launch/sync, already resident), _pair_combo_mi
    #       1.03s, permutation-null/CMI njit ~1.9s (orchestration), the rest one-time JIT + cuda driver.
    # FULL GPU-RESIDENCY (STRICT, 2026-06-26). Priority #1 is that the FE path runs entirely on the GPU with
    # no CPU MI compute. Under MLFRAME_FE_GPU_STRICT the batch MI routes to the resident GPU plug-in kernel
    # (host candidate matrix -> device once -> resident percentile-edge binning + plug-in MI), so EVERY
    # _mi_classif_batch caller (gate-grid, pairwise-modular, perm-null, unified-gate, chunked) becomes GPU.
    # Selection-equivalent to the CPU njit (percentile-edge vs rank binning; recipe-hash byte-identical in the
    # bench-note force-test). Falls back to the CPU njit on any GPU failure / when STRICT is off (the default
    # path stays byte-for-byte the numba batch). Launches rise (priority #2) but residency is the goal.
    # Narrow device-fault set (FIX1, 2026-06-28): the prior broad ``except Exception: pass`` here
    # SWALLOWED the ValueError that ``_assert_codes_in_range`` raises on a -1 / out-of-range code,
    # silently degrading a genuine OOB (illegal-address) bug to the CPU njit and DEFEATING the guard
    # added in 6c127567. Catch only genuine cupy/device faults below so a true device error still
    # falls back to CPU, while a ValueError/IndexError (real OOB / logic bug) propagates to surface.
    _dev_errs = [np.linalg.LinAlgError]
    try:
        import cupy as _cp_e  # type: ignore

        _dev_errs.append(_cp_e.cuda.runtime.CUDARuntimeError)
        _dev_errs.append(_cp_e.cuda.memory.OutOfMemoryError)
        from cupy_backends.cuda.libs import cusolver as _cusolver_e  # type: ignore
        _dev_errs.append(getattr(_cusolver_e, "CUSOLVERError", None))
        from cupy_backends.cuda.libs import cublas as _cublas_e  # type: ignore
        _dev_errs.append(getattr(_cublas_e, "CUBLASError", None))
    except Exception:
        pass
    _DEV_ERRS = tuple(e for e in _dev_errs if isinstance(e, type) and issubclass(e, BaseException))
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled

        if fe_gpu_strict_enabled():
            import cupy as cp

            from ..hermite_fe import _plugin_mi_classif_batch_cuda_resident

            Xd = cp.asarray(np.ascontiguousarray(np.asarray(X, dtype=np.float64)))
            if Xd.ndim == 1:
                Xd = Xd[:, None]
            _yi = np.ascontiguousarray(np.asarray(y)).astype(np.int64).ravel()
            yd = cp.asarray(_yi)
            # y is a fit-constant: derive y_min / n_classes on the HOST (cheap O(n) pass) and pass them down so
            # the resident plug-in skips the per-call GPU cp.min + cp.max + stack reduction (nsys's #1
            # cuLaunchKernel source on this STRICT MI path, hit by every gate-grid / pairwise / perm-null /
            # chunked caller). Same data -> identical min/max -> bit-identical bincount layout and MI.
            _ymin = int(_yi.min()) if _yi.size else 0
            _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1
            # GATE MI rank route: when the caller is the conditional gate (rank_binning=True) AND the resident
            # opt-in is on, bin by argsort equi-frequency RANK so the STRICT gate MI byte-matches the CPU njit
            # rank MI on the gate's heavily-tied columns (edge would lump tied zeros into one bin -> lower MI).
            # Returns None on any GPU failure -> the exact CPU njit rank path below. The FE-candidate path
            # leaves rank_binning=False so the radix-edge resident MI is untouched.
            if rank_binning:
                try:
                    from .._gpu_strict_fe import fe_gpu_strict_resident_enabled
                    _resident_on = fe_gpu_strict_resident_enabled()
                except Exception:
                    _resident_on = False
                if _resident_on:
                    from .._gpu_resident_rank_bin import plugin_mi_classif_batch_rank_cuda_resident
                    _rank_mi = plugin_mi_classif_batch_rank_cuda_resident(Xd, yd, int(nbins), y_min=_ymin,
                                                                          n_classes=_ncls)
                    if _rank_mi is not None:
                        return np.asarray(_rank_mi, dtype=np.float64)
            return np.asarray(_plugin_mi_classif_batch_cuda_resident(Xd, yd, int(nbins), y_min=_ymin,
                                                                     n_classes=_ncls), dtype=np.float64)
    except (ImportError, *_DEV_ERRS):
        pass   # cupy/strict-module absent OR a genuine device fault -> exact CPU njit below.
        # NOTE (FIX1): ValueError / IndexError are intentionally NOT caught here -- a -1 / out-of-range
        # code raised by _assert_codes_in_range (illegal-address guard) must surface, not degrade to CPU.
    if _MI_BACKEND == "numba":
        return _mi_classif_batch_numba(X, y, nbins=nbins)
    return _mi_classif_batch_sklearn(X, y, nbins=nbins)


def _mi_chunk_cols_for(n_rows: int) -> int:
    """RAM-aware column-block width: bound the per-block float64 materialization (n_rows * cols * 8 B, plus a
    ~3x factor for V, V^2 and the binning transient) to ~10% of free RAM, capped at 1024 cols, floored at 1.
    A fixed COLUMN count alone is unsafe at large n (1024 cols x n=1M float64 = 8 GiB still OOMs); bounding the
    block BYTES makes the chunked scorer safe in n as well as p. Conservative 2 GiB fallback if psutil missing."""
    try:
        import psutil
        free = int(psutil.virtual_memory().available)
    except Exception:
        free = 2 * 1024 ** 3
    budget = max(1, int(free * 0.10))
    by_ram = budget // (max(1, int(n_rows)) * 8 * 3)
    return int(min(1024, max(1, by_ram)))


def mi_classif_batch_chunked(X, y, *, nbins: int = 10, chunk_cols: int = None) -> np.ndarray:
    """Column-CHUNKED ``_mi_classif_batch`` for WIDE engineered matrices (2026-06-19).

    The FE MI-uplift scorers (univariate / pair-cross / triplet / quadruplet / adaptive-arity / mi-greedy)
    each materialised the FULL engineered matrix as one float64 array to batch-score MI -- O(n * n_engineered)
    peak RAM that OOMs at scale (measured (16000, 20000) float64 = 2.38 GiB), worst for the combinatorial
    triplet/quadruplet cross-basis families. MI is PER-COLUMN, so scoring in column blocks is BIT-IDENTICAL to
    the all-at-once call FOR ANY chunk size while bounding peak extra RAM. ``chunk_cols`` defaults to a RAM-aware
    width (see ``_mi_chunk_cols_for`` -- bounds the BLOCK BYTES, so it is safe at large n too, not just wide p);
    pass an explicit value to override. Accepts a pandas DataFrame (sliced via ``iloc``, so only the block is
    materialised) or a 2-D ndarray. Returns the (p,) per-column MI array."""
    # The block loop below stays SEQUENTIAL by design: ``_mi_classif_batch`` already parallelises across the
    # block's columns via the numba prange batch kernel, so each per-block call saturates all cores. Threading
    # the block loop on top would OVERSUBSCRIBE the prange pool (no speedup, likely slower) -- bench-rejected
    # (2026-06-19); the chunking is a MEMORY bound, not a missing parallelism lever.
    is_df = hasattr(X, "iloc")
    n = int(X.shape[0])
    p = int(X.shape[1])
    if chunk_cols is None:
        chunk_cols = _mi_chunk_cols_for(n)
    if p <= chunk_cols:
        arr = X.to_numpy(dtype=np.float64) if is_df else np.asarray(X, dtype=np.float64)
        return _mi_classif_batch(arr, y, nbins=nbins)
    parts = []
    for j0 in range(0, p, chunk_cols):
        if is_df:
            blk = X.iloc[:, j0:j0 + chunk_cols].to_numpy(dtype=np.float64)
        else:
            blk = np.asarray(X[:, j0:j0 + chunk_cols], dtype=np.float64)
        parts.append(_mi_classif_batch(blk, y, nbins=nbins))
        del blk
    return np.concatenate(parts)


def _maybe_class_weights(y: np.ndarray):
    """Auto-detect imbalance and return inverse-prior class weights, or ``None``.

    ``None`` => the caller falls through to the plain-MI path unchanged
    (balanced data, below the n_rare gate, non-discrete y, or override ``off``).
    Cheap: a single ``bincount`` + two comparisons; adds ~0 to the balanced path.
    """
    try:
        from ._imbalance_mi import compute_class_weights
        return compute_class_weights(y)
    except Exception:
        return None


def _mi_classif_batch_balanced(X: np.ndarray, y: np.ndarray, class_w, *, nbins: int = 10):
    """Class-balanced batch MI; mirrors ``_mi_classif_batch_numba``'s NaN handling.

    Returns ``None`` on any failure so the caller falls back to plain MI rather
    than poisoning the whole call.
    """
    try:
        from ._imbalance_mi import _class_balanced_mi_batch_njit
    except Exception:
        return None
    n, p = X.shape
    y_i64 = np.ascontiguousarray(y, dtype=np.int64)
    class_w = np.ascontiguousarray(class_w, dtype=np.float64)
    mis = np.zeros(p, dtype=np.float64)
    finite_per_col = np.isfinite(X).all(axis=0)
    dense_cols = np.where(finite_per_col)[0]
    partial_cols = np.where(~finite_per_col)[0]
    try:
        if dense_cols.size:
            if dense_cols.size == p:
                X_dense = np.ascontiguousarray(X)
            else:
                X_dense = np.ascontiguousarray(X[:, dense_cols])
            mis[dense_cols] = _class_balanced_mi_batch_njit(X_dense, y_i64, class_w, nbins)
        for j in partial_cols:
            col = X[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                mis[j] = 0.0
                continue
            col_f = np.ascontiguousarray(col[finite].reshape(-1, 1))
            y_f = np.ascontiguousarray(y_i64[finite])
            mis[j] = float(_class_balanced_mi_batch_njit(col_f, y_f, class_w, nbins)[0])
    except Exception:
        return None
    return mis
