"""Batched CUDA conditional-mutual-information kernel for the MRMR/Fleuret greedy redundancy loop.

The dominant large-``n`` cost of MRMR is the greedy Fleuret redundancy step: for every
candidate column ``X_j`` it evaluates ``I(X_j; Y | Z)`` against the just-selected feature ``Z``,
each greedy round. The CPU path (``info_theory._entropy_kernels.conditional_mi``) is a serial
njit that rebuilds four joint histograms per call via the per-row ``merge_vars`` melt -- O(p * n)
per round with no parallelism.

This module adds a BATCHED GPU kernel that computes ``I(X_j; Y | Z)`` for ALL ``p`` candidates in a
single launch, returning a ``(p,)`` float64 array. ``Y`` and ``Z`` are fixed across candidates, so
``H(Y, Z)`` and ``H(Z)`` are computed ONCE per round; only the per-candidate 3-variable joint
``H(X_j, Y, Z)`` (and the 2-variable ``H(X_j, Z)``) are the per-candidate piece. One block per
candidate, shared-memory histogram (mirrors ``filters/gpu.py``'s ``compute_joint_hist_*`` design).

    I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

returned RAW (clamped at 0), bit-comparable (within ~1e-9) to the CPU ``conditional_mi`` on the
SAME discretized inputs -- see ``test_cmi_cuda_kernel.py``.

PARITY: the CPU ``entropy`` reduces ``-(log(p) * p)`` over the NONZERO joint bins in ascending
merged-class-id order, with ``p = count / n`` in float64. The kernel walks the dense per-candidate
histogram in the same ascending merged-class order, skips zero bins, and accumulates in float64 --
reproducing the CPU reduction value (sequential add order matches numpy ``.sum`` for the small
nonzero-bin counts here). The merged class id ``cz + cx * nbz + cy * (nbx * nbz)`` keeps ``Z`` the
fastest-varying axis so the (X, Z) marginal and (Y, Z) marginal collapse in the same monotone order
the CPU ``merge_vars`` pruning produces.

PICKLE: NO live CUDA context or RawKernel object is ever stored on a class instance -- the kernel is
a module-level singleton built lazily in ``_get_kernel()`` and attached to THIS module's namespace.
Callers hold only plain numpy arrays. Safe to pickle any consumer.

DISPATCH: ``conditional_mi_batched_dispatch`` is the public entry. It routes to CUDA at large
``(n, p)`` where the launch wins, and to the CPU per-candidate loop otherwise / when no GPU. The
size/HW gate integrates with ``pyutilz`` ``kernel_tuning_cache`` (via the FS ``_kernel_tuning``
singleton) -- NO hardcoded thresholds baked into the hot path; a hand fallback only applies on
cache miss / pyutilz-unavailable.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Optional

import numpy as np
from numba import njit, prange

from ._entropy_kernels import conditional_mi, entropy, _entropy_xz_fused, _entropy_x_onto_classes
from ._class_encoding import merge_vars

logger = logging.getLogger(__name__)

# Below this many candidates the njit-parallel CMI loop's thread-spawn overhead is not worth it -> exact serial path.
# Lowered 32 -> 8 (2026-07): at the default 30k SCREEN-SUBSAMPLE size (the typical per-candidate CMI shape once the
# Q1 screen subsample engages) prange beats the serial loop 2.4-7.8x for EVERY p>=4, and p>=8 wins universally
# (1.7-8.6x) across n=1k..30k -- each candidate's exact conditional_mi on ~30k rows dwarfs the ~50us thread spawn.
# The old 32 floor kept small candidate pools (the wellbore's serial single-core / ~44%-CPU window) on ONE core.
# 8 captures nearly all the win while keeping the tiniest pools (p<8, sub-ms, where spawn variance matters at very
# small n) serial. Both branches are exact CMI (selection-equivalent); the threshold only trades cores for spawn.
# Bench: info_theory/_benchmarks/bench_cmi_parallel_threshold.py. Override via MLFRAME_CMI_PARALLEL_MIN_CANDS.
import os as _os_thr
try:
    _CMI_PARALLEL_MIN_CANDS = int(_os_thr.environ.get("MLFRAME_CMI_PARALLEL_MIN_CANDS", "8"))
except (TypeError, ValueError):
    _CMI_PARALLEL_MIN_CANDS = 8

# F-ORDER (column-contiguous) VIEW CACHE for the CPU CMI melt.
# ``factors_data`` is (n, nfeat) C-contiguous, so reading candidate column ``xi`` in the per-candidate
# melt strides ``nfeat*4`` bytes -- every O(n) pass thrashes cache lines. The batched-GPU path already
# side-steps this by transposing to (candidates, n) contiguous (line ~684); the CPU melt did not. A
# column-contiguous (F-order) copy of the SAME matrix makes every candidate column a contiguous run:
# the identical loop runs 2-6x faster, BIT-IDENTICAL (asfortranarray only changes physical order, so
# ``data[r, col]`` yields the same value -> same histograms -> same CMI). See bench_cmi_layout_probe.py.
# ``factors_data`` is a fit-constant across the greedy loop (the dispatch already caches y/z uploads on
# ``id(factors_data)``), so the O(n*nfeat) transpose is paid ONCE per fit and reused across every round
# and every Z. Cached by ``id`` with a WEAKREF identity guard: a recycled id (allocator reuse after the
# original is GC'd) fails the ``ref() is factors_data`` check and rebuilds -> never returns a stale copy.
# Gated by a byte cap (backstop against a surprise large-n alloc) + an A/B toggle (REJECTED!=DELETED).
_FORDER_CACHE: dict = {}
_FORDER_LOCK = threading.Lock()
try:
    _CMI_FORDER_MAX_MB = float(_os_thr.environ.get("MLFRAME_CMI_FORDER_MAX_MB", "4096"))
except (TypeError, ValueError):
    _CMI_FORDER_MAX_MB = 4096.0


def _cmi_forder_enabled() -> bool:
    """True unless ``MLFRAME_CMI_FORDER`` explicitly disables the F-order caching path (default on)."""
    return _os_thr.environ.get("MLFRAME_CMI_FORDER", "1").strip().lower() not in ("0", "false", "off", "no")


def _cmi_forder_view(factors_data: np.ndarray) -> np.ndarray:
    """Return a column-contiguous (F-order) copy of ``factors_data``, cached per fit.

    Bit-identical to the input (same values, same shape, same column indexing) but with contiguous
    column reads for the melt. Returns the input unchanged when already F-contiguous, when disabled,
    when not a 2-D array, or when the copy would exceed ``MLFRAME_CMI_FORDER_MAX_MB``.
    """
    import weakref
    if not _cmi_forder_enabled():
        return factors_data
    if getattr(factors_data, "ndim", 0) != 2 or factors_data.flags.f_contiguous:
        return factors_data
    if factors_data.nbytes > _CMI_FORDER_MAX_MB * (1 << 20):
        return factors_data
    key = id(factors_data)
    with _FORDER_LOCK:
        ent = _FORDER_CACHE.get(key)
        if ent is not None and ent[0]() is factors_data:
            return np.asarray(ent[1])
        # Purge entries whose original array has been GC'd (frees their cached F-copy; also clears any
        # recycled-id collision) before inserting the fresh one.
        dead = [k for k, v in _FORDER_CACHE.items() if v[0]() is None]
        for k in dead:
            del _FORDER_CACHE[k]
        farr = np.asfortranarray(factors_data)
        _FORDER_CACHE[key] = (weakref.ref(factors_data), farr)
        return farr


def reset_cmi_forder_cache() -> None:
    """Drop all cached F-order copies (tests / free memory between fits)."""
    with _FORDER_LOCK:
        _FORDER_CACHE.clear()

# Module-level kernel singleton (pickle-safe: never stored on instances).
_cmi_joint_hist_cuda = None
_KERNEL_LOCK = threading.Lock()
_CUPY_OK: Optional[bool] = None

# Tuning-region name for the kernel_tuning_cache dispatch (mirrors the
# joint-hist dispatch keys in filters/gpu.py).
_TUNING_REGION = "cmi_batched"


def cupy_available() -> bool:
    """True if cupy imports AND a CUDA device is visible. Cached after first probe."""
    global _CUPY_OK
    if _CUPY_OK is not None:
        return _CUPY_OK
    try:
        import cupy as cp

        _CUPY_OK = cp.cuda.runtime.getDeviceCount() > 0
    except Exception as _exc:
        logger.debug("cmi_cuda: cupy unavailable (%s); CPU fallback", _exc)
        _CUPY_OK = False
    return _CUPY_OK


def _get_kernel():
    """Build (idempotently) and return the batched 3-var joint-hist RawKernel.

    One block per candidate; shared-memory histogram of size ``nbx * nby * nbz`` ints. Each block
    strides over the n samples of its candidate column, atomically bumping the shared histogram,
    then reduces shared -> the candidate's global slot in one pass.
    """
    global _cmi_joint_hist_cuda
    if _cmi_joint_hist_cuda is not None:
        return _cmi_joint_hist_cuda
    import cupy as cp

    with _KERNEL_LOCK:
        if _cmi_joint_hist_cuda is not None:
            return _cmi_joint_hist_cuda
        module = sys.modules[__name__]
        module._cmi_joint_hist_cuda = cp.RawKernel(  # type: ignore[attr-defined]
            r"""
        extern "C" __global__
        void cmi_joint_hist_cuda(
            const int *Xc,            // (p, n) row-major: candidate codes
            const int *y,             // (n,)  target codes
            const int *z,             // (n,)  conditioning codes
            int *joint_counts,        // (p, nbx*nby*nbz) row-major output
            int n,
            int p,
            int nbx,
            int nby,
            int nbz
        ) {
            extern __shared__ int sm[];   // nbx*nby*nbz ints
            int cand = blockIdx.x;
            if (cand >= p) return;
            int joint_size = nbx * nby * nbz;
            int tid = threadIdx.x;
            int nthreads = blockDim.x;

            for (int i = tid; i < joint_size; i += nthreads) sm[i] = 0;
            __syncthreads();

            const int *xrow = Xc + (long)cand * n;
            // merged id = cz + cx*nbz + cy*(nbx*nbz)  -> Z fastest, then X, then Y.
            for (int r = tid; r < n; r += nthreads) {
                int cx = xrow[r];
                int cy = y[r];
                int cz = z[r];
                int cell = cz + cx * nbz + cy * (nbx * nbz);
                atomicAdd(&sm[cell], 1);
            }
            __syncthreads();

            int *out = joint_counts + (long)cand * joint_size;
            for (int i = tid; i < joint_size; i += nthreads) {
                if (sm[i] != 0) out[i] = sm[i];
            }
        }
        """,
            "cmi_joint_hist_cuda",
        )
        return module._cmi_joint_hist_cuda


# FUSED CMI-from-joint (launch-reduction, 2026-06-25). conditional_mi_batched_cuda called
# _entropy_from_counts_axis FOUR times (H(Z), H(X,Z), H(Y,Z), H(X,Y,Z)), each marg.sum(axis) + xlogx EK +
# .sum(axis=1) = ~3 cuLaunchKernel -> ~12 per call (the measured #1 hidden launch source, 228+95). One block
# per candidate now reads the global joint once, builds the (X,Z)/(Y,Z)/(Z) marginals in shared memory via
# atomicAdd, reduces all four plug-in entropies (block tree reduction, 256 threads), and writes the clamped
# CMI -- ONE launch. Same float64 p*log p over nonzero cells -> within the documented ~1e-9 parity gate.
_cmi_from_joint_cuda = None


def _get_cmi_from_joint_kernel():
    """Build (idempotently) and return the fused CMI-from-joint-counts RawKernel.

    One block per candidate reads the global joint histogram once, accumulates the (X,Z)/(Y,Z)/(Z) marginals in
    shared memory via ``atomicAdd``, and reduces all four plug-in entropies in a single launch -- replacing the
    four separate ``_entropy_from_counts_axis`` launches (H(Z), H(X,Z), H(Y,Z), H(X,Y,Z)) that used to dominate
    per-call CUDA launch count. Result is clamped to ``>= 0`` (plug-in CMI can go slightly negative from bias).
    """
    global _cmi_from_joint_cuda
    if _cmi_from_joint_cuda is not None:
        return _cmi_from_joint_cuda
    import cupy as cp

    with _KERNEL_LOCK:
        if _cmi_from_joint_cuda is not None:
            return _cmi_from_joint_cuda
        module = sys.modules[__name__]
        module._cmi_from_joint_cuda = cp.RawKernel(  # type: ignore[attr-defined]
            r"""
        extern "C" __global__
        void cmi_from_joint(const int* __restrict__ joint, const double inv_n, const int p,
                            const int nbx, const int nby, const int nbz, double* __restrict__ cmi_out) {
            int cand = blockIdx.x;
            if (cand >= p) return;
            int tid = threadIdx.x, nt = blockDim.x;
            int nxz = nbx * nbz, nyz = nby * nbz, joint_size = nbx * nby * nbz;
            extern __shared__ char shmem[];
            double* red = (double*)shmem;            // nt doubles (8-byte aligned base)
            int* m_xz = (int*)(red + nt);            // nxz ints
            int* m_yz = m_xz + nxz;                  // nyz ints
            int* m_z  = m_yz + nyz;                  // nbz ints
            for (int i = tid; i < nxz; i += nt) m_xz[i] = 0;
            for (int i = tid; i < nyz; i += nt) m_yz[i] = 0;
            for (int i = tid; i < nbz; i += nt) m_z[i] = 0;
            __syncthreads();
            const int* J = joint + (long)cand * joint_size;
            double hxyz_loc = 0.0;
            for (int i = tid; i < joint_size; i += nt) {
                int c = J[i];
                if (c > 0) {
                    double pp = (double)c * inv_n; hxyz_loc += pp * log(pp);
                    int cz = i % nbz, rem = i / nbz, cx = rem % nbx, cy = rem / nbx;
                    atomicAdd(&m_xz[cx * nbz + cz], c);
                    atomicAdd(&m_yz[cy * nbz + cz], c);
                    atomicAdd(&m_z[cz], c);
                }
            }
            __syncthreads();
            red[tid] = hxyz_loc; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
            double h_xyz = -red[0]; __syncthreads();
            double loc = 0.0;
            for (int i = tid; i < nxz; i += nt) { int c = m_xz[i]; if (c > 0) { double pp = (double)c * inv_n; loc += pp * log(pp); } }
            red[tid] = loc; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
            double h_xz = -red[0]; __syncthreads();
            loc = 0.0;
            for (int i = tid; i < nyz; i += nt) { int c = m_yz[i]; if (c > 0) { double pp = (double)c * inv_n; loc += pp * log(pp); } }
            red[tid] = loc; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
            double h_yz = -red[0]; __syncthreads();
            loc = 0.0;
            for (int i = tid; i < nbz; i += nt) { int c = m_z[i]; if (c > 0) { double pp = (double)c * inv_n; loc += pp * log(pp); } }
            red[tid] = loc; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
            double h_z = -red[0];
            if (tid == 0) { double cmi = h_xz + h_yz - h_z - h_xyz; cmi_out[cand] = cmi > 0.0 ? cmi : 0.0; }
        }
        """,
            "cmi_from_joint",
        )
        return module._cmi_from_joint_cuda


def _entropy_from_counts_axis(counts_3d, axes, n):
    """Shannon entropy (nats) of a marginal of the 3-D joint count tensor.

    ``counts_3d`` is (p, nby, nbx, nbz) int (Y, X, Z layout matching the merged id Z-fastest /
    X-next / Y-slowest). ``axes`` are the axes to SUM OUT before the entropy reduction (so the
    remaining axes form the marginal). Returns a (p,) float64 entropy vector.

    Reduction matches the CPU ``entropy``: ``p_bin = count / n`` in float64, ``-sum p log p`` over
    nonzero bins. CuPy reduces a contiguous flattened marginal; the small nonzero-bin counts here
    make the float64 accumulation track numpy's within ~1e-12 (well inside the 1e-9 gate).
    """
    import cupy as cp

    marg = counts_3d.sum(axis=axes)  # (p, ...) remaining axes
    marg = marg.reshape(marg.shape[0], -1)
    # Fused x*log(x) (launch-reduction): one ElementwiseKernel folds the /n + zero-mask + log + multiply
    # (cp.divide + cp.where + cp.log + multiply, ~4 cuLaunchKernel) into a single launch -- ``c>0 ?
    # (c*invn)*log(c*invn) : 0`` -- then one sum(axis=1). Same float64 plug-in entropy -> within the 1e-9
    # selection gate (cmi_cuda parity tests stay < 1e-9). ElementwiseKernel launches via the same
    # cuLaunchKernel driver API -> genuine count reduction, not a counter shift.
    global _XLOGX_EK
    if _XLOGX_EK is None:
        _XLOGX_EK = cp.ElementwiseKernel("T c, float64 invn", "float64 o", "o = c > 0 ? (c * invn) * log(c * invn) : 0.0", "mrmr_xlogx_ek")
    contrib = _XLOGX_EK(marg, 1.0 / float(n))
    return -contrib.sum(axis=1)


_XLOGX_EK = None

# FIX3 (2026-06-28): y and z are fit-CONSTANTS across the greedy CMI loop (the same target / just-selected
# columns for every candidate batch in a round, and z changes only ONCE per greedy round). The prior code
# re-uploaded BOTH via cp.asarray on EVERY call -> per-candidate-batch H2D churn (the measured safe_cuda_api
# 0.63s + .get() 0.45s overhead on the CPU-strict path, where the CMI GPU launch is neutral-to-hurting vs the
# njit pool MI it replaces). Cache the uploaded device copy keyed on array IDENTITY (id + shape + dtype) so
# repeated calls in the same greedy loop reuse it. Selection-equivalent (same values, just not re-uploaded).
# Pickle-safe: module-level (never on an instance), cleared at FE-step teardown via clear_cmi_resident_cache().
# WALL A/B (2026-06-28, CPU-strict 90k uniform, quiet box, MLFRAME_CMI_RESIDENT_CACHE 0 vs 1): warm median
# 8.82s -> 8.96s, min 8.79s -> 8.84s == FLAT (within run-to-run noise). At this shape (n_feat=1) the greedy
# CMI loop runs too few rounds/candidate-batches for the per-call y/z H2D to be on the critical path, so the
# cache does not move the wall here. Mechanism verified (5 same-y/z dispatch calls -> exactly 2 cache entries
# = y,z uploaded ONCE not per call). KEPT: it is selection-equivalent and can only help (never hurt) at the
# many-round shapes where the H2D churn IS the safe_cuda_api/.get() overhead; the value is shape-dependent.
_CMI_RESIDENT_CACHE: dict = {}


def _resident_upload(arr, key):
    """Return a device int32 copy of host ``arr``, reusing a cached upload keyed on the caller's ``key``.

    The dispatch keys on ``(id(factors_data), column_index, nbins)`` -- a STABLE identity across the greedy
    loop (the freshly-sliced host column is re-allocated every call, so keying on the slice's own id would
    never hit; the parent ``factors_data`` + column index is the fit-constant). ``id()`` can be recycled
    after factors_data is GC'd, so shape/dtype are folded into the cached entry and a recycled id with a
    different shape simply misses and re-uploads (never a stale wrong-shape buffer). Bounded; cleared at
    FE-step teardown.
    """
    import cupy as cp

    # Diagnostic A/B switch only (default ON): MLFRAME_CMI_RESIDENT_CACHE=0 forces a fresh upload every call
    # to reproduce the pre-FIX3 H2D churn for the wall A/B. NOT a perf gate -- the cache is always on in prod.
    import os as _os
    if _os.environ.get("MLFRAME_CMI_RESIDENT_CACHE", "1").strip().lower() in ("0", "false", "off", "no"):
        return cp.asarray(arr)
    # CONTENT FINGERPRINT (recycled-id guard): id(factors_data) is reused by the allocator after the parent
    # is GC'd, so a recycled id with the SAME column index + shape + dtype but DIFFERENT values would return a
    # STALE device copy (a silent correctness bug -- caught by the cmi parity tests). Fold a cheap O(n) host
    # content hash of the (small, 1-D, int) array into the key so different contents never alias. The hash is
    # pure CPU and far cheaper than the H2D it guards; on a real greedy loop the same y/z hash identically ->
    # the cache still hits every candidate batch.
    sig = (arr.shape, arr.dtype.str, hash(arr.tobytes()))
    cached = _CMI_RESIDENT_CACHE.get(key)
    if cached is not None:
        g, csig = cached
        if csig == sig:
            return g
    if len(_CMI_RESIDENT_CACHE) > 16:
        _CMI_RESIDENT_CACHE.clear()
    g = cp.asarray(arr)
    _CMI_RESIDENT_CACHE[key] = (g, sig)
    return g


def clear_cmi_resident_cache() -> None:
    """Drop the resident y/z device cache (call at FE-step teardown; mirrors the mempool teardown)."""
    _CMI_RESIDENT_CACHE.clear()


# --------------------------------------------------------------------------------------------------------------------
# XC RESIDENT MATRIX (2026-07-12): ``factors_data`` (ALL candidate columns) is a fit-CONSTANT across the whole
# greedy CMI loop -- same object for every round -- yet the dispatch below re-sliced the round's candidate subset
# via ``factors_data[:, cand_indices].T`` on the HOST every round and handed the fresh host array to
# ``conditional_mi_batched_cuda``, which then re-uploaded it via ``cp.asarray`` -- a host slice-copy PLUS an H2D
# transfer, repeated every round, of the single LARGEST operand any GPU kernel in this package touches (bigger
# than y/z by a factor of p, the audit's biggest-magnitude finding: tens of GB of redundant PCIe traffic over a
# 100k x 500 fit). y/z already got the resident treatment above (FIX3) for exactly this reason; Xc did not,
# because it is genuinely PER-CANDIDATE -- but every round's candidates are always a SUBSET of the SAME
# fit-constant ``factors_data``, so the fix is to upload ``factors_data`` itself ONCE (keyed on its identity,
# mirroring ``_cmi_forder_view``'s weakref-identity guard a few dozen lines above) and GATHER the round's
# candidate columns ON-DEVICE (cupy fancy-indexing + transpose) -- the host never sees the big matrix again
# after the first round. Selection-equivalent by construction: the gathered device values are byte-identical to
# the old host-sliced-then-uploaded ones (same source array, same column selection, same dtype cast).
# OrderedDict (not plain dict) for O(1) LRU: bounded at a handful of entries (each a FULL factors_data
# copy, potentially ~100s of MB of VRAM) so a long-lived process running many fits/CV-folds never grows
# this cache unboundedly -- move-to-end on hit, evict only the single coldest entry on overflow (mirrors
# ``_fe_resident_operands.py``'s ``_FE_RESIDENT_OPERANDS`` LRU discipline for the exact same VRAM-safety
# reason). 4 is enough to keep 2-3 concurrently-alive folds/fits resident without ever approaching a 4 GB
# card's budget (at ~200 MB/entry that is <=800 MB worst case, comfortable alongside the other resident
# caches + working buffers documented in ``_fe_resident_operands.py``).
from collections import OrderedDict as _OrderedDict

_FACTORS_DEVICE_CACHE: "_OrderedDict" = _OrderedDict()
_FACTORS_DEVICE_MAX_ENTRIES = 4
_FACTORS_DEVICE_LOCK = threading.Lock()


def _xc_resident_enabled() -> bool:
    """Diagnostic A/B switch only (default ON): MLFRAME_CMI_XC_RESIDENT=0 forces a fresh host slice + upload of
    the candidate matrix every call, reproducing the pre-fix H2D/host-slice churn for the wall A/B."""
    return _os_thr.environ.get("MLFRAME_CMI_XC_RESIDENT", "1").strip().lower() not in ("0", "false", "off", "no")


def _assert_codes_in_range_2d_per_column(host2d: np.ndarray, factors_nbins: np.ndarray) -> None:
    """One-time, WHOLE-matrix OOB guard: every column's codes must lie in ``[0, factors_nbins[col])``.

    Strictly a SUPERSET of the per-round candidate-subset check the old code ran every round (that check
    only validated the round's own ``cand_indices`` against the round's ``nbins_x = max(factors_nbins[cand])``
    upper bound); checking every column against its OWN nbins entry is at least as strict, so it can only catch
    an invalid code EARLIER, never later. Run ONCE per ``factors_data`` identity (see
    ``_resident_factors_device``) instead of every greedy round -- cheaper in total than the old per-round
    re-scan (which repeats across dozens of rounds over overlapping candidate sets) while never weakening the
    safety net the OOB screen exists for (a code indexing outside the device histogram is a hard
    ``cudaErrorIllegalAddress`` crash, not a catchable Python error).
    """
    if host2d.size == 0:
        return
    bins = np.asarray(factors_nbins)
    col_min = host2d.min(axis=0)
    col_max = host2d.max(axis=0)
    if bool((col_min < 0).any()):
        j = int(np.argmax(col_min < 0))
        raise ValueError(f"factors_data column {j} contains a negative integer code (min={int(col_min[j])}); codes must be 0-based.")
    bad = col_max >= bins
    if bool(bad.any()):
        j = int(np.argmax(bad))
        raise ValueError(f"factors_data column {j} code out of range (max={int(col_max[j])} >= nbins={int(bins[j])}).")


def _resident_factors_device(factors_data: np.ndarray, factors_nbins: np.ndarray) -> Any:
    """Return the WHOLE ``factors_data`` matrix resident on-device as int32, uploaded ONCE per fit.

    Keyed on ``id(factors_data)`` with a weakref identity guard -- mirrors ``_cmi_forder_view`` a few dozen
    lines above (the SAME object, already trusted with an id-only guard in this exact file): a recycled id
    (allocator reuse after the original ``factors_data`` is GC'd) fails the ``ref() is factors_data`` check and
    re-uploads, never returning a stale buffer. The one-time OOB guard (see
    ``_assert_codes_in_range_2d_per_column``) runs on every fresh upload (cache miss OR the disabled A/B path),
    so every device array this function returns has already been fully validated -- callers may pass
    ``codes_trusted=True`` downstream.
    """
    import weakref

    import cupy as cp

    if not _xc_resident_enabled():
        host = np.ascontiguousarray(factors_data, dtype=np.int32)
        _assert_codes_in_range_2d_per_column(host, factors_nbins)
        return cp.asarray(host)

    # Cache-hit fast path never touches the host array at all (no ascontiguousarray/dtype check on the
    # hot loop) -- the identity key alone decides the hit, matching the "gather on-device, never re-touch
    # the host" goal this cache exists for.
    key = id(factors_data)
    with _FACTORS_DEVICE_LOCK:
        ent = _FACTORS_DEVICE_CACHE.get(key)
        if ent is not None and ent[0]() is factors_data:
            _FACTORS_DEVICE_CACHE.move_to_end(key)  # LRU: this factors_data is hot
            return ent[1]
        dead = [k for k, v in _FACTORS_DEVICE_CACHE.items() if v[0]() is None]
        for k in dead:
            del _FACTORS_DEVICE_CACHE[k]
        host = np.ascontiguousarray(factors_data, dtype=np.int32)
        _assert_codes_in_range_2d_per_column(host, factors_nbins)
        dev = cp.asarray(host)
        _FACTORS_DEVICE_CACHE[key] = (weakref.ref(factors_data), dev)
        if len(_FACTORS_DEVICE_CACHE) > _FACTORS_DEVICE_MAX_ENTRIES:
            _FACTORS_DEVICE_CACHE.popitem(last=False)  # evict ONLY the coldest entry, never the whole table
        return dev


def clear_cmi_xc_resident_cache() -> None:
    """Drop the resident whole-factors_data device cache (call at FE-step teardown alongside the y/z cache)."""
    _FACTORS_DEVICE_CACHE.clear()


def conditional_mi_batched_cuda(
    Xc: Optional[np.ndarray],
    y: np.ndarray,
    z: np.ndarray,
    nbins_x: int,
    nbins_y: int,
    nbins_z: int,
    block_size: int = 256,
    y_g: Any = None,
    z_g: Any = None,
    Xc_g: Any = None,
    codes_trusted: bool = False,
) -> np.ndarray:
    """Compute ``I(X_j; Y | Z)`` for all ``p`` candidate columns in one launch -> (p,) float64.

    Parameters
    ----------
    Xc : (p, n) int32 -- candidate codes, one row per candidate (0..nbins_x-1). May be ``None`` when
        ``Xc_g`` (an already-uploaded, already-validated device array) is supplied instead -- the
        resident-gather dispatch path below does exactly this so the host never rebuilds the big matrix.
    y, z : (n,) int32 -- target / conditioning codes.
    nbins_x, nbins_y, nbins_z : per-variable cardinalities (max code + 1 is fine; empty bins cost
        only shared-mem, not correctness).
    block_size : threads per block (one block per candidate).
    Xc_g : pre-uploaded, pre-gathered (p, n) int32 device array (skips the host ``Xc`` -> ``cp.asarray``
        round-trip entirely when supplied).
    codes_trusted : when True, skips the OOB guard on ``Xc``/``Xc_g`` (the caller has already validated the
        WHOLE parent matrix once via ``_resident_factors_device`` -- see that function's docstring).

    Returns RAW CMI clamped at 0, mirroring the CPU ``conditional_mi``.
    """
    import cupy as cp

    y = np.ascontiguousarray(y, dtype=np.int32)
    z = np.ascontiguousarray(z, dtype=np.int32)
    if Xc_g is None:
        Xc = np.ascontiguousarray(Xc, dtype=np.int32)
        p, n = Xc.shape
    else:
        p, n = int(Xc_g.shape[0]), int(Xc_g.shape[1])
    joint_size = int(nbins_x) * int(nbins_y) * int(nbins_z)

    # OOB SCREEN (FIX2, 2026-06-28): the joint-hist kernel uses raw codes DIRECTLY as a flat shared-mem
    # offset (``cell = cz + cx*nbz + cy*(nbx*nbz)``) sized from nbins_* (= factors_nbins.max()). A -1
    # sentinel or a code >= its nbins indexes OUTSIDE the histogram -> cudaErrorIllegalAddress (a hard,
    # unrecoverable GPU crash, NOT a Python error). Mirror the host min/max screen batch_pair_mi_cuda
    # already does so an upstream OOB surfaces as a clear ValueError instead. Cheap: one min+max per
    # array on the host inputs (already materialized as numpy above). Skipped when ``codes_trusted`` (the
    # resident dispatch path already ran the equivalent, strictly-superset check once for the whole matrix).
    from .._fe_batched_mi import _assert_codes_in_range
    if not codes_trusted:
        _assert_codes_in_range(Xc if Xc_g is None else Xc_g, int(nbins_x), "conditional_mi_batched_cuda X codes")
    _assert_codes_in_range(y, int(nbins_y), "conditional_mi_batched_cuda y codes")
    _assert_codes_in_range(z, int(nbins_z), "conditional_mi_batched_cuda z codes")

    kern = _get_kernel()

    # FIX3/XC-RESIDENT: y, z, and now Xc are fit-constants (Xc a SUBSET-gather of the fit-constant
    # factors_data) -> the dispatch passes their cached device uploads so none are re-uploaded per candidate
    # batch (kills the per-call H2D churn). When called directly (tests/benches) they arrive None -> upload
    # once here (the old behaviour, preserved for direct callers).
    if Xc_g is None:
        Xc_g = cp.asarray(Xc)
    if y_g is None:
        y_g = cp.asarray(y)
    if z_g is None:
        z_g = cp.asarray(z)
    joint = cp.zeros((p, joint_size), dtype=cp.int32)

    kern(
        (p,),
        (block_size,),
        (Xc_g, y_g, z_g, joint, np.int32(n), np.int32(p), np.int32(nbins_x), np.int32(nbins_y), np.int32(nbins_z)),
        shared_mem=joint_size * 4,
    )

    # FUSED (launch-reduction): one block-per-candidate kernel reads the global joint once, builds the
    # (X,Z)/(Y,Z)/(Z) marginals in shared memory, reduces all four plug-in entropies, and writes the clamped
    # CMI in ONE launch -- replacing the four _entropy_from_counts_axis calls (~12 cuLaunchKernel) + the
    # cmi assembly + maximum. Same float64 p*log p over nonzero cells -> within the ~1e-9 parity gate.
    # Falls back to the four-call cupy path on any kernel error.
    try:
        cmi_g = cp.empty(p, dtype=cp.float64)
        red_threads = 256
        shmem_bytes = red_threads * 8 + (nbins_x * nbins_z + nbins_y * nbins_z + nbins_z) * 4
        _get_cmi_from_joint_kernel()(
            (p,), (red_threads,),
            (joint, 1.0 / float(n), np.int32(p), np.int32(nbins_x), np.int32(nbins_y), np.int32(nbins_z), cmi_g),
            shared_mem=shmem_bytes,
        )
        return np.asarray(cp.asnumpy(cmi_g))
    except Exception as _exc:
        logger.debug("cmi_from_joint kernel failed (%s); four-call entropy fallback", _exc)
        counts = joint.reshape(p, nbins_y, nbins_x, nbins_z)
        h_z = _entropy_from_counts_axis(counts, (1, 2), n)
        h_xz = _entropy_from_counts_axis(counts, (1,), n)
        h_yz = _entropy_from_counts_axis(counts, (2,), n)
        h_xyz = _entropy_from_counts_axis(counts, (), n)
        return np.asarray(cp.asnumpy(cp.maximum(h_xz + h_yz - h_z - h_xyz, 0.0)))


@njit(parallel=True, cache=True)
def _cpu_cmi_loop_parallel(factors_data, cand_indices, y, z, factors_nbins, var_is_nominal) -> np.ndarray:
    """``prange`` CMI over candidates -- each ``conditional_mi(X_j; Y | Z)`` is independent and stateless
    (``entropy_cache=None``), so this fans the exact per-candidate kernel across ALL cores. Bit-identical to the serial
    loop (verified maxdiff 0.0; ~9x on 16 cores at p=400). ``var_is_nominal`` is passed through but unused inside
    ``conditional_mi`` (it hardcodes ``None`` to ``merge_vars``)."""
    p = len(cand_indices)
    out = np.empty(p, dtype=np.float64)
    for i in prange(p):
        xi = np.array([cand_indices[i]], dtype=np.int64)
        out[i] = conditional_mi(factors_data, xi, y, z, var_is_nominal, factors_nbins)
    return out


# --------------------------------------------------------------------------------------------------------------------
# Y,Z-ENTROPY HOIST (2026-07): in an MRMR greedy round Y (target) and Z (just-selected feature) are FIXED across the
# WHOLE candidate pool, yet ``conditional_mi`` rebuilds H(Z), the (Y,Z) joint melt (``classes_yz``) and H(Y,Z) on EVERY
# candidate -- its existing ``entropy_z``/``entropy_yz`` params cannot remove that work because the H(X,Y,Z) term is
# built by melting X on top of ``classes_yz`` (needed per candidate). This pair of kernels computes the fixed (Y,Z)/Z
# terms + ``classes_yz`` ONCE per call and reuses a per-candidate COPY of ``classes_yz`` for the H(X,Y,Z) melt, so each
# candidate does only the two X-dependent melts (X,Z and X-on-YZ) instead of all four. BIT-IDENTICAL by construction:
# same merge order (unpack_and_sort), same ``entropy`` reduction -- verified maxabsdiff EXACTLY 0.0 in
# ``_benchmarks/bench_cmi_yz_hoist.py`` and ``tests/.../test_cpu_cmi_loop_yz_hoist_identity.py``. Measured 1.60-1.74x
# (serial AND prange, n=1e6, nb=10/16, p=10..300) -- this path is the dominant wellbore main-process hotspot (the GPU
# CMI cascade routes all redundancy through it). Mirrors the already-shipped ``_mi_greedy_cmi_fe.cmi_from_binned_fixed_yz``
# hoist for the binned-values FE path; this applies the SAME win to the merge_vars ``conditional_mi`` path.


@njit(cache=True)
def _cmi_yz_fixed_terms(factors_data, y, z, factors_nbins, dtype):
    """H(Z), the (Y,Z) merged dense classes, H(Y,Z) and nclasses_yz -- the per-round CONSTANTS. Melt order matches
    ``conditional_mi``: Z alone for H(Z); ``unpack_and_sort(y, z)`` for the (Y,Z) joint."""
    _, freqs_z, _ = merge_vars(factors_data, z, None, factors_nbins, dtype=dtype)
    ent_z = entropy(freqs_z)
    yi = y[0]; zi = z[0]
    yz = np.empty(2, dtype=np.int64)
    if zi <= yi:
        yz[0] = zi; yz[1] = yi
    else:
        yz[0] = yi; yz[1] = zi
    classes_yz, freqs_yz, nclasses_yz = merge_vars(factors_data, yz, None, factors_nbins, dtype=dtype)
    return ent_z, classes_yz, entropy(freqs_yz), nclasses_yz


@njit(cache=True)
def _cmi_one_fixed_yz(factors_data, xi, zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype):
    """I(X_i; Y | Z) reusing the fixed (Y,Z)/Z terms: only the X-dependent (X,Z) and (X-on-YZ) melts run.

    Both melts use the freqs-ONLY pruned kernels ``conditional_mi`` itself already uses -- ``_entropy_xz_fused``
    (single-pass ``joint_freqs_2var`` for the 2-var X u Z union) and ``_entropy_x_onto_classes`` (histograms X onto
    the precomputed (Y,Z) labels ``classes_yz`` READ-ONLY, no length-n scratch copy, no discarded relabel/remap).
    Wasted-work audit (2026-07): the prior body called full ``merge_vars`` twice, each building + remapping a
    length-n ``final_classes`` array (and a ``classes_yz.copy()``) that this path immediately discards -- only the
    joint freqs feed ``entropy``. Bit-identical BY CONSTRUCTION (same kernels as ``conditional_mi``, maxabsdiff
    EXACTLY 0.0), measured 1.33-1.65x at the wellbore redundancy shape (n=30k, p=100..2000, nbins 10-16, |Z|=1).
    Bench: ``_benchmarks/bench_cmi_pruned_melts.py``. ``classes_yz`` is never mutated -> prange-safe with no copy."""
    xz = np.empty(2, dtype=np.int64)
    if xi <= zi:
        xz[0] = xi; xz[1] = zi
    else:
        xz[0] = zi; xz[1] = xi
    ent_xz = _entropy_xz_fused(factors_data, xz, factors_nbins, dtype)
    ent_xyz = _entropy_x_onto_classes(factors_data, xi, classes_yz, nclasses_yz, factors_nbins[xi])
    r = ent_xz + ent_yz - ent_z - ent_xyz
    return r if r > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _cpu_cmi_loop_hoisted_parallel(factors_data, cand_indices, y, z, factors_nbins, dtype=np.int32) -> np.ndarray:
    """prange CMI with the Y,Z terms hoisted out of the per-candidate loop. Bit-identical to
    ``_cpu_cmi_loop_parallel`` (maxdiff 0.0), 1.6-1.7x faster (two melts per candidate instead of four)."""
    p = len(cand_indices)
    out = np.empty(p, dtype=np.float64)
    ent_z, classes_yz, ent_yz, nclasses_yz = _cmi_yz_fixed_terms(factors_data, y, z, factors_nbins, dtype)
    zi = z[0]
    for i in prange(p):
        out[i] = _cmi_one_fixed_yz(factors_data, cand_indices[i], zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype)
    return out


@njit(cache=True)
def _cpu_cmi_loop_hoisted_serial(factors_data, cand_indices, y, z, factors_nbins, dtype=np.int32) -> np.ndarray:
    """Serial (tiny-pool) CMI with the Y,Z terms hoisted. Bit-identical to the serial ``conditional_mi`` loop."""
    p = len(cand_indices)
    out = np.empty(p, dtype=np.float64)
    ent_z, classes_yz, ent_yz, nclasses_yz = _cmi_yz_fixed_terms(factors_data, y, z, factors_nbins, dtype)
    zi = z[0]
    for i in range(p):
        out[i] = _cmi_one_fixed_yz(factors_data, cand_indices[i], zi, classes_yz, nclasses_yz, ent_yz, ent_z, factors_nbins, dtype)
    return out


def _cpu_cmi_loop(factors_data, cand_indices, y, z, factors_nbins, dtype=np.int32) -> np.ndarray:
    """Exact CPU CMI over each candidate column index -> (p,) float64. Parallel (prange) across cores once the
    candidate pool clears ``_CMI_PARALLEL_MIN_CANDS``; exact serial for a tiny pool (thread-spawn not worth it).

    ``cand_indices`` is a (p,) array of column indices into ``factors_data``; ``y`` / ``z`` are 1-element column-index
    arrays (target / selected-feature). Was a SERIAL Python loop -- the 1-core CMI-screen bottleneck on the greedy path.

    Routes through the Y,Z-entropy hoist by default (1.6-1.7x, bit-identical); MLFRAME_CMI_YZ_HOIST=0 restores the
    un-hoisted ``conditional_mi`` recompute path for the A/B (bench_cmi_yz_hoist.py). Both branches are exact.
    """
    cand_indices = np.asarray(cand_indices, dtype=np.int64)
    p = len(cand_indices)
    if p == 0:
        return np.empty(0, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    z = np.asarray(z, dtype=np.int64)
    factors_nbins = np.asarray(factors_nbins)
    _vin = np.empty(0, dtype=np.int64)  # var_is_nominal placeholder (unused inside conditional_mi)

    # Column-contiguous view of the fit-constant matrix -> contiguous candidate-column reads in the
    # melt (2-6x, bit-identical; cached once per fit). Falls through to the C-order input when disabled
    # / oversized / already F-contiguous.
    fd = _cmi_forder_view(factors_data)

    import os as _os
    if _os.environ.get("MLFRAME_CMI_YZ_HOIST", "1").strip().lower() not in ("0", "false", "off", "no"):
        if p >= _CMI_PARALLEL_MIN_CANDS:
            return np.asarray(_cpu_cmi_loop_hoisted_parallel(fd, cand_indices, y, z, factors_nbins, dtype))
        return np.asarray(_cpu_cmi_loop_hoisted_serial(fd, cand_indices, y, z, factors_nbins, dtype))

    # bench-attempt-baseline (MLFRAME_CMI_YZ_HOIST=0): un-hoisted recompute path kept for the A/B (REJECTED!=DELETED).
    if p >= _CMI_PARALLEL_MIN_CANDS:
        return np.asarray(_cpu_cmi_loop_parallel(fd, cand_indices, y, z, factors_nbins, _vin))
    out = np.empty(p, dtype=np.float64)
    for i in range(p):
        out[i] = conditional_mi(fd, np.array([cand_indices[i]], dtype=np.int64), y, z, _vin, factors_nbins)
    return out


# CIRCUIT BREAKER (2026-07): a cudaErrorLaunchFailure / illegal-address is an EXECUTION fault that POISONS the CUDA
# context -- every SUBSEQUENT launch on that context then fails identically. In the wellbore run the first fault
# cascaded into 1515 futile GPU retries (each logged + each paying the host-side setup before failing) before falling
# to CPU. Once ANY GPU CMI launch faults we trip this flag and route ALL further CMI to the (now-hoisted) CPU path for
# the rest of the process -- no re-attempt on a dead context. Cheap, correct, and kills the retry spam. Reset only via
# reset_cmi_gpu_circuit_breaker() (tests). NOT a substitute for preventing the fault; a resilience backstop.
_CMI_GPU_FAILED = False


def reset_cmi_gpu_circuit_breaker() -> None:
    """Re-arm the GPU CMI path (tests / after a fresh CUDA context)."""
    global _CMI_GPU_FAILED
    _CMI_GPU_FAILED = False


def _cmi_cuda_shmem_fits(joint_size: int, nbins_x: int = 0, nbins_y: int = 0, nbins_z: int = 0, cc_smem_bytes: int = 48 * 1024) -> bool:
    """Do BOTH batched-CMI kernels' shared-memory requests fit the per-block limit (cc 6.x = 48 KB)?

    Kernel 1 (joint histogram) needs ``joint_size * 4`` bytes. Kernel 2 (``cmi_from_joint``) needs
    ``256*8 + (nbx*nbz + nby*nbz + nbz)*4`` -- which can EXCEED kernel 1 when an axis is degenerate (e.g. ``nby=1``
    makes ``nbx*nbz == joint_size``, so kernel 2 = ``joint_size*4 + 2048 + 8*nbz``). Guarding only kernel 1 let that
    over-request launch-fail. When the individual nbins are unknown (default 0) only kernel 1 is checked (legacy)."""
    if joint_size * 4 > cc_smem_bytes:
        return False
    if nbins_x and nbins_y and nbins_z:
        k2 = 256 * 8 + (nbins_x * nbins_z + nbins_y * nbins_z + nbins_z) * 4
        if k2 > cc_smem_bytes:
            return False
    return True


def _should_use_cuda(n: int, p: int, joint_size: int, nbins_x: int = 0, nbins_y: int = 0, nbins_z: int = 0) -> bool:
    """Size/HW gate. Routes to CUDA at large (n, p) where the batched launch wins; CPU otherwise.

    Integrates with the FS kernel_tuning_cache singleton (pyutilz). On cache hit the decision is the
    measured one; on cache miss / pyutilz-unavailable a hand fallback applies (NO hardcoded value on
    the hot path -- the fallback is the documented bootstrap heuristic only). VRAM guard: the joint
    buffer is ``p * joint_size * 4`` bytes; reject if it would exceed a conservative slice.
    """
    if _CMI_GPU_FAILED:  # context poisoned by a prior launch fault -> never re-attempt the GPU this process.
        return False
    if not cupy_available():
        return False
    # VRAM guard (GTX 1050 Ti = 4 GB): the dominant device buffer is the candidate matrix Xc
    # (p * n * 4 bytes); the joint output is p * joint_size * 4. Reject if the combined working
    # set would exceed a conservative slice of free VRAM (leave headroom for the model / other
    # worker). Best-effort free-VRAM probe; falls back to a fixed cap.
    bytes_needed = p * n * 4 + p * joint_size * 4
    cap = 1536 * 1024 * 1024  # 1.5 GB conservative cap (shared 4 GB card)
    free_b: Optional[int] = None
    total_b: Optional[int] = None
    try:
        import cupy as cp

        free_b, total_b = cp.cuda.runtime.memGetInfo()
        cap = min(cap, int(free_b * 0.5))
    except Exception as e:
        logger.debug("swallowed exception in _cmi_cuda.py: %s", e)
        pass
    if bytes_needed > cap:
        return False
    # ABSOLUTE cushion guard (2026-07-05): the relative cap above is computed only AFTER the cupy pool may
    # have already eaten the card; on a near-full / SHARED 4 GB card that lets the next launch fault. Require
    # an ABSOLUTE free-VRAM floor (default >=1 GB) BEFORE touching the GPU. Pure ADD -- tightens, never loosens.
    # Reuses the ``memGetInfo`` probe already taken above for the relative cap (``free_b``/``total_b``) instead
    # of letting ``fe_gpu_has_vram_cushion`` re-query the device a second time per dispatch (2026-07-13 fix) --
    # falls back to that function's own probe when the probe above failed (``free_b``/``total_b`` still None).
    try:
        from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion
        if free_b is not None and total_b is not None:
            cushion_ok = fe_gpu_has_vram_cushion(bytes_needed, free_b=free_b, total_b=total_b)
        else:
            cushion_ok = fe_gpu_has_vram_cushion(bytes_needed)
        if not cushion_ok:
            return False
    except Exception as e:
        logger.debug("swallowed exception in _cmi_cuda.py: %s", e)
        pass
    # Shared-mem guard: cc 6.x has 48 KB/block and BOTH kernels must fit (see _cmi_cuda_shmem_fits).
    if not _cmi_cuda_shmem_fits(joint_size, nbins_x, nbins_y, nbins_z):
        return False

    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, diagnostic, default OFF): force CUDA past the KTC crossover /
    # size threshold once the VRAM + shared-mem guards above have passed (forcing past those would OOM, not
    # diagnose). The CPU/CUDA conditional-MI backends are numerically equivalent (~1e-9) -> selection-equivalent.
    try:
        from mlframe.feature_selection.filters._fe_gpu_strict import fe_gpu_strict_enabled
        # Pass THIS call's own (n, p) shape (2026-07-09 fix), not just the fit-level row count: STRICT used to
        # force every dispatch to CUDA for the rest of a >=100k-row fit even when a LATE-round call has shrunk
        # to a handful of candidates (conditional_mi_batched_dispatch's p = len(cand_indices), which shrinks
        # every greedy round) -- forcing GPU on a trivially small call pays full host<->device round-trip +
        # kernel-launch overhead for negligible compute, which is a plausible driver of low CPU+GPU utilization
        # simultaneously. See fe_gpu_strict_enabled's docstring / _STRICT_MIN_CALL_WORK.
        if fe_gpu_strict_enabled(n=n, p=p):
            return True
    except Exception as e:
        logger.debug("swallowed exception in _cmi_cuda.py: %s", e)
        pass

    try:
        from mlframe.feature_selection.filters._kernel_tuning import get_kernel_tuning_cache

        cache = get_kernel_tuning_cache()
        if cache is not None:
            key = {"region": _TUNING_REGION, "n": int(n), "p": int(p), "joint_size": int(joint_size)}
            try:
                decision = cache.lookup(_TUNING_REGION, key)
                if decision is not None and "use_cuda" in decision:
                    return bool(decision["use_cuda"])
            except Exception as _exc:
                logger.debug("cmi_cuda: kernel_tuning_cache lookup failed (%s); hand fallback", _exc)
    except Exception as _exc:
        logger.debug("cmi_cuda: kernel_tuning_cache unavailable (%s); hand fallback", _exc)

    # MANDATE-1 (2026-06-23): per-host KTC-derived crossover (sibling _cmi_cuda_ktc). The legacy "FS
    # kernel_tuning_cache singleton" lookup above is a manual cache; this adds the SWEPT crossover the repo
    # rule mandates -- a kernel_tuner that benches CPU vs CUDA on the real (n, p) conditional-MI shapes and
    # records the faster backend per region. On a measured hit it IS the gate; the hardcoded heuristic below
    # survives ONLY as the un-tuned (pre-sweep / no-cupy / lookup-failure) bootstrap default, per the rule.
    try:
        from ._cmi_cuda_ktc import cmi_use_cuda as _ktc_cmi_use_cuda

        _decision = _ktc_cmi_use_cuda(n, p)
        if _decision is not None:
            return bool(_decision)
    except Exception as _exc:
        logger.debug("cmi_cuda: KTC crossover unavailable (%s); hand fallback", _exc)

    # Hand bootstrap heuristic (un-tuned default ONLY): the batched launch amortizes its host<->device
    # transfer + launch overhead once p and n are both sizable. Measured crossover (GTX 1050 Ti,
    # bench_cmi_cuda): CUDA wins from roughly n*p >= ~1e6 with p >= ~100. KEPT as the safe default for
    # un-tuned hosts; superseded per-host by the swept crossover above once the sweep has run.
    return (n * p) >= 1_000_000 and p >= 64


def conditional_mi_batched_dispatch(
    factors_data: np.ndarray,
    cand_indices: np.ndarray,
    y_index,
    z_index,
    factors_nbins: np.ndarray,
    dtype=np.int32,
    force: Optional[str] = None,
) -> np.ndarray:
    """Public entry: ``I(X_j; Y | Z)`` for every candidate column in ``cand_indices`` -> (p,) float64.

    Routes to the batched CUDA kernel at large ``(n, p)`` (where it wins), else to the exact CPU
    ``conditional_mi`` loop. ``y_index`` / ``z_index`` are the single column indices of the target
    and the just-selected feature; ``cand_indices`` are the candidate column indices.

    ``force`` -- ``"cuda"`` or ``"cpu"`` overrides the size/HW gate (tests / benchmarks). Default
    ``None`` uses the dispatcher.
    """
    cand_indices = np.asarray(cand_indices)
    y_index = int(np.asarray(y_index).ravel()[0])
    z_index = int(np.asarray(z_index).ravel()[0])
    factors_nbins = np.asarray(factors_nbins)
    n = factors_data.shape[0]
    p = len(cand_indices)

    nbins_x = int(factors_nbins[cand_indices].max()) if p else 0
    nbins_y = int(factors_nbins[y_index])
    nbins_z = int(factors_nbins[z_index])
    joint_size = nbins_x * nbins_y * nbins_z

    use_cuda = force == "cuda" or (force is None and _should_use_cuda(n, p, joint_size, nbins_x, nbins_y, nbins_z))
    if force == "cpu":
        use_cuda = False

    if use_cuda:
        y = np.ascontiguousarray(factors_data[:, y_index], dtype=np.int32)
        z = np.ascontiguousarray(factors_data[:, z_index], dtype=np.int32)
        # OOB SCREEN (FIX2, 2026-06-28): screen HERE, BEFORE the GPU try/except below, so a -1 /
        # out-of-range code (illegal-address) surfaces as a ValueError instead of being swallowed by
        # the device-fault fallback (which would silently mask a real bug -> CPU). Cheap host min/max.
        # Xc's OOB check is now the ONE-TIME whole-matrix guard inside ``_resident_factors_device``
        # (see XC RESIDENT MATRIX above) -- it is strictly a superset of the old per-round Xc-subset
        # check, so it is not repeated here.
        from .._fe_batched_mi import _assert_codes_in_range
        _assert_codes_in_range(y, int(nbins_y), "conditional_mi_batched_dispatch y codes")
        _assert_codes_in_range(z, int(nbins_z), "conditional_mi_batched_dispatch z codes")
        try:
            import cupy as cp

            # FIX3: y/z are fit-constants across the greedy loop -> upload ONCE per (factors_data, column)
            # and reuse the device copy on every candidate batch, eliminating the per-call H2D churn (the
            # safe_cuda_api / .get() overhead measured on the CPU-strict path). Keyed on the stable parent
            # factors_data identity + column index (the freshly-sliced y/z host arrays change id per call).
            _fid = id(factors_data)
            y_g = _resident_upload(y, (_fid, "y", int(y_index), int(nbins_y)))
            z_g = _resident_upload(z, (_fid, "z", int(z_index), int(nbins_z)))
            # XC RESIDENT: factors_data itself is uploaded ONCE per fit (id-keyed, weakref-guarded) and the
            # round's candidate columns are GATHERED on-device (fancy-index + transpose) -- no host re-slice,
            # no re-upload, of the single largest operand any GPU kernel here touches. codes_trusted=True
            # because _resident_factors_device already ran the (strictly stronger) whole-matrix OOB guard.
            factors_dev = _resident_factors_device(factors_data, factors_nbins)
            Xc_g = cp.ascontiguousarray(factors_dev[:, cand_indices].T)
            return conditional_mi_batched_cuda(
                None, y, z, nbins_x, nbins_y, nbins_z, y_g=y_g, z_g=z_g, Xc_g=Xc_g, codes_trusted=True,
            )
        except Exception as _exc:
            # Trip the process-level circuit breaker: a launch fault poisons the CUDA context, so every subsequent
            # GPU CMI would fault identically (the wellbore 1515-retry cascade). Route all further CMI to CPU.
            global _CMI_GPU_FAILED
            _CMI_GPU_FAILED = True
            logger.warning("cmi_cuda: GPU path failed (%s); circuit breaker tripped -> CPU for the rest of the process", _exc)

    return _cpu_cmi_loop(factors_data, cand_indices, np.asarray([y_index]), np.asarray([z_index]), factors_nbins, dtype=dtype)


__all__ = [
    "conditional_mi_batched_cuda",
    "conditional_mi_batched_dispatch",
    "cupy_available",
    "clear_cmi_resident_cache",
    "clear_cmi_xc_resident_cache",
]
