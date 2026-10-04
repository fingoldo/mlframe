"""CMI-greedy feature constructor (Layer 60, 2026-05-31).

Sibling to :mod:`_mi_greedy_fe` (Layer 26). Where Layer 26 ranks the same
candidate transform pool by MARGINAL ``MI(candidate; y)`` and de-duplicates
selected winners post-hoc via Spearman, THIS module ranks by CONDITIONAL
``MI(candidate; y | current_support)`` - i.e. each step directly measures
the NEW information the candidate adds on top of the already-selected
columns.

Why CMI ranking matters
-----------------------

Marginal MI ranks ``log_abs(x)``, ``square(x)``, ``abs(x)`` all near the top
when ``y = sign(x^2 - 1)`` because each is monotone in ``|x|`` and so
captures the same signal. The marginal-MI greedy path then picks all three
and the downstream Spearman dedup drops two of them post-hoc - waste.
CMI ranking sees that once ``square(x)`` is in the support, ``CMI(abs(x); y |
square(x))`` is near zero, so ``abs(x)`` is never picked.

Algorithm
---------

1. Materialise the candidate library via :func:`_mi_greedy_fe.iter_candidates`
   over the top-N seed columns (same enumeration as Layer 26).
2. Quantile-bin every candidate column to ``nbins`` integer bins once.
3. Quantile-bin the seed columns identically.
4. Seed the support with the top-``seed_cols_count`` raw columns by marginal
   MI(x; y).
5. Greedy loop: at each step compute
   ``CMI(candidate; y | joint_support)`` for every remaining candidate,
   pick the one with the highest CMI provided it clears ``min_cmi_gain``.
   Stop when no candidate clears the gate or ``top_k`` winners are seated.
6. Emit recipes of kind ``"mi_greedy_transform"`` (same as Layer 26) so
   transform-time replay is shared infrastructure.

The conditional joint Z is the per-row class id of the cross-product of
the currently-selected binned columns - collapsed via the densely-renumbered
contingency table so the joint stays computable even at d=8+ support cols
(memory dominated by ``n``, not by the cartesian bin space).
"""
from __future__ import annotations

import functools
import logging
import threading
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
    from numba.core import types as _nb_types
    from numba.typed import Dict as _NbDict
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a hard dep in practice
    _NUMBA_AVAILABLE = False
    _nb_types = None
    _NbDict = None

    def prange(*a):  # no-op fallback (serial range) when numba is absent
        """Serial ``range`` stand-in used when numba is unavailable so parallel loops still run correctly, just single-threaded."""
        return range(*a)

    from mlframe._numba_fallback import njit

from mlframe.utils.log_throttle import log_throttle
from types import SimpleNamespace as _SimpleNamespace

# GPU quantile-bin crossover (2026-06-28, synchronized micro-bench of _quantile_bin_gpu incl. code D2H, GTX
# 1050 Ti, nbins=10): a single host column round-tripped to the device for equi-frequency binning (H2D +
# cp.percentile sort + cp.searchsorted + code D2H) only beats the host introselect-partition np.quantile path
# well above the launch/transfer floor - n=20k CPU 0.92ms vs GPU 1.67ms; n=35k near-tie 1.53 vs 1.71ms; n=100k
# GPU 2.10 vs CPU 4.12ms = 2x; n=300k GPU 3.06 vs CPU 14.1ms = 4.6x. The gate is set at 50k (clear of the 35k
# near-tie) so every routed call is a decisive win; the small (3k/20k) gate-redundancy columns stay on the
# host, where the fixed ~1.7ms device round-trip overhead loses to numpy.
_GPU_QBIN_MIN_ROWS = 50_000


__all__ = [
    "score_candidates_by_cmi",
    "greedy_cmi_fe_construct",
    "greedy_cmi_fe_construct_with_recipes",
]


# ---------------------------------------------------------------------------
# Binning + entropy helpers (self-contained — mirrors ``_mi_classif_batch``'s
# equi-frequency binning so CMI numbers are directly comparable to the
# marginal-MI numbers Layer 26 reports).
# ---------------------------------------------------------------------------


# Direct-array factorize is used while the joint's max id keeps the ``seen``
# lookup buffer under this many int64 entries (~128 MB at the cap). Above it
# (cartesian blow-up: a high-cardinality support col times a large running
# class count) we fall back to the hash path so memory stays bounded.
# Parallel-memset crossover for the dense ``seen`` buffer in ``_combine_factorize_njit``. The first-seen
# factorize WALK is irreducibly sequential (each id depends on the running ``seen``/``nc`` state), but its
# ``np.full(kmax+1, -1)`` initialisation is an independent fill that prange-splits across threads. The fill
# is only worth the thread spin-up once it is large: synchronized micro-bench (2026-06-29, 4 threads, GTX
# box, n=1M) put the crossover near kmax~500k - below it parallel loses (kmax~40k 0.99x, ~200k 0.92x),
# above it wins and scales (500k 1.08x, 1M 1.06x, 2M 1.09x, 4M 1.12x, 16M 1.24x). At the cap (16M int64 =
# 128 MB) the fill alone is ~28ms of the ~55ms call, so parallelising it is the single largest safe win on
# this sequential kernel. Below the gate the fill stays serial -> bit-identical, zero regression.
from ._mi_greedy_cmi_fe_steps import (  # noqa: F401  -- carved helpers
    logger,
    _qbin_float_dtype,
    _sync_free_qbin_codes,
    _quantile_bin_gpu,
    _quantile_bin_gpu_resident,
    _quantile_bin,
    _FAC_ARRAY_CAP,
    _factorize_dense_njit,
    _FAC_PAR_MEMSET_MIN,
    _combine_factorize_njit,
    _renumber_two_dense_njit,
    _renumber_joint,
    _dense_renumber_device,
    _renumber_joint_gpu,
    _entropy_from_classes_njit,
    _joint_entropy_two_dense_njit,
    _joint_entropy_two,
    _entropy_from_classes,
    precompute_marginal_y_terms,
    marginal_mi_binned_fixed_y,
    precompute_cmi_yz_terms,
    cmi_from_binned_fixed_yz,
    _cmi_gpu_enabled,
    _CARD_MAX_CACHE,
    _CARD_MAX_CACHE_LOCK,
    _cached_card,
    _cmi_from_binned_fixed_yz_cupy,
    _greedy_cmi_fe_constr_step1_content_hash_already,
    _greedy_cmi_fe_constr_step2_rng_raw_small,
    _greedy_cmi_fe_constr_step3_while_st_remaining,
)


@njit(cache=True)
def _combine_factorize_serial_njit(joint: np.ndarray, c: np.ndarray, mult: int) -> tuple:
    """Fully-serial reference form of :func:`_combine_factorize_njit` (the bit-identical baseline + the
    numba-absent fallback). Kept per the repo "keep all kernel versions" rule and used by the parity test."""
    n = joint.size
    if n == 0:
        return joint, 0
    kmax = 0
    for i in range(n):
        v = joint[i] + c[i] * mult
        if v > kmax:
            kmax = v
    inv = np.empty(n, dtype=np.int64)
    nc = 0
    if 0 <= kmax < _FAC_ARRAY_CAP:
        seen = np.full(kmax + 1, -1, dtype=np.int64)
        for i in range(n):
            v = joint[i] + c[i] * mult
            s = seen[v]
            if s >= 0:
                inv[i] = s
            else:
                seen[v] = nc
                inv[i] = nc
                nc += 1
    else:
        d = _NbDict.empty(key_type=_nb_types.int64, value_type=_nb_types.int64)
        for i in range(n):
            v = joint[i] + c[i] * mult
            s = d.get(v, -1)
            if s >= 0:
                inv[i] = s
            else:
                d[v] = nc
                inv[i] = nc
                nc += 1
    return inv, nc


def _cmi_from_binned(
    x: np.ndarray, y: np.ndarray, z_joint: Optional[np.ndarray], return_cards: bool = False,
    kx: int = 0, kz: int = 0,
):
    """``CMI(X; Y | Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)`` from binned
    integer arrays. Miller-Madow bias correction applied: each plug-in
    entropy is reduced by ``(K-1)/(2n)`` where K is the number of
    occupied cells. The four entropy bias terms combine in CMI to
    ``-(K_xz + K_yz - K_z - K_xyz) / (2n)`` (subtracted from the plug-in
    CMI). On noise data this drives the CMI estimate toward zero where
    the unbiased MLE would inflate to e.g. 0.005 - 0.02 nats and admit
    false positives. On signal data the bias term is dwarfed by the
    true CMI so the correction is benign.

    When ``z_joint is None`` (empty support), reduces to marginal
    ``MI(X; Y) = H(X) + H(Y) - H(X, Y)`` (also Miller-Madow corrected).

    ``return_cards`` (conditional path only): also return the OCCUPIED-cell cards ``(k_z, k_xz, k_yz, k_xyz)``
    already computed here, so a same-(x,y,z) analytic CMI null can reuse them (``precomp_cards``) instead of
    recomputing the four joints. Returns ``(cmi, cards)``; ``cards`` is ``None`` on the marginal path.
    """
    # GPU route: same partition-entropy-via-cp.unique offload as cmi_from_binned_fixed_yz,
    # covering the general CMI callers (conditional-perm null, candidate scoring). value-order densify ->
    # same partition -> same CMI; selection-identical. Gated (STRICT / MLFRAME_CMI_GPU), default CPU.
    #
    # ``x`` may ALREADY be device-resident (e.g. _step_score.py's DEVICE-BORN marginal-MI path binning a
    # candidate via _quantile_bin_gpu_resident before ever calling here) - the shape-based gate below only
    # decides whether it's worth PROACTIVELY uploading, it has no way to know the array is already on-device.
    # An already-device x MUST take the cupy path regardless of that heuristic (np.ascontiguousarray on a
    # cupy array raises TypeError - cupy forbids the implicit host conversion __array__ would trigger), and
    # a cupy failure there must explicitly pull x/y/z_joint to host before falling back (the swallow-and-
    # retry-on-CPU pattern below assumes host arrays).
    _x_device = hasattr(x, "__cuda_array_interface__")
    if _x_device or _cmi_gpu_enabled(n=int(x.size), p=1):
        try:
            return _cmi_from_binned_cupy(x, y, z_joint, return_cards=return_cards, kx=kx, kz=kz)
        except Exception as e:
            # A cupy import error, a kernel-shape miss, transient GPU contention and a genuine numeric
            # regression in the cupy path all land here, so the message has to name which kernel and at what
            # shape. The CPU recomputation below keeps the answer correct, which is exactly why a real kernel
            # regression would otherwise never be noticed.
            log_throttle(
                logger,
                "cmi_gpu_kernel_fallback",
                logging.WARNING,
                "_cmi_from_binned_cupy failed (%s: %s) at n=%d; recomputing this CMI on the CPU path. Correctness is preserved, the GPU cost is not.",
                type(e).__name__,
                e,
                int(getattr(x, "size", 0)),
            )
            if _x_device:
                import cupy as cp
                x = cp.asnumpy(x)
                if hasattr(y, "__cuda_array_interface__"):
                    y = cp.asnumpy(y)
                if z_joint is not None and hasattr(z_joint, "__cuda_array_interface__"):
                    z_joint = cp.asnumpy(z_joint)
    x_i = np.ascontiguousarray(x, dtype=np.int64)
    y_i = np.ascontiguousarray(y, dtype=np.int64)
    n = float(max(1, x_i.size))
    if z_joint is None or z_joint.size == 0:
        h_x, k_x = _entropy_from_classes(x_i)
        h_y, k_y = _entropy_from_classes(y_i)
        h_xy, k_xy = _joint_entropy_two(x_i, y_i)  # fused densify+entropy; xy labels discarded
        mi_plugin = h_x + h_y - h_xy
        # Plug-in MLE entropy underestimates the true entropy by
        # ``(K-1)/(2n)`` (Miller 1955). MI = H(X) + H(Y) - H(XY)
        # therefore OVERESTIMATES the true MI by
        # ``((K_x-1) + (K_y-1) - (K_xy-1))/(2n)``
        # = ``(K_x + K_y - K_xy - 1)/(2n)``. Subtract this bias from
        # the plug-in to denoise.
        mi_bias = (k_x + k_y - k_xy - 1) / (2.0 * n)
        mi = max(0.0, mi_plugin - mi_bias)
        return (mi, None) if return_cards else mi
    z_i = np.ascontiguousarray(z_joint, dtype=np.int64)
    # Fused densify+entropy for the joints (labels feed only the entropy). yz is materialised DENSE once
    # and reused for H(X,Y,Z): partition(x, part(y,z)) == partition(x,y,z) -> identical counts -> the
    # 3-column renumber(x,y,z) collapses to a 2-array densify against yz_dense.
    yz_dense, _ = _renumber_joint(y_i, z_i)
    h_z, k_z = _entropy_from_classes(z_i)
    h_xz, k_xz = _joint_entropy_two(x_i, z_i)
    h_yz, k_yz = _entropy_from_classes(yz_dense)
    h_xyz, k_xyz = _joint_entropy_two(x_i, yz_dense)
    cmi_plugin = h_xz + h_yz - h_z - h_xyz
    # Plug-in CMI = H(XZ) + H(YZ) - H(Z) - H(XYZ). Each plug-in entropy
    # is biased low by (K-1)/(2n). The CMI bias from combining them
    # (with signs +H_xz +H_yz -H_z -H_xyz, where each contributes
    # -(K-1)/(2n) to the plug-in vs true entropy) is:
    #   E[CMI_plugin] - CMI_true
    #   = -((k_xz-1) + (k_yz-1) - (k_z-1) - (k_xyz-1))/(2n)
    #   = (k_xyz + k_z - k_xz - k_yz)/(2n).
    # On noise frames k_xyz - k_xz dominates (XYZ has many empty cells
    # filled by noise) so plug-in CMI is biased UP - subtract the
    # bias to denoise.
    cmi_bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
    cmi = max(0.0, cmi_plugin - cmi_bias)
    return (cmi, (int(k_z), int(k_xz), int(k_yz), int(k_xyz))) if return_cards else cmi


# Fused entropy + occupied-cell reduction over a bincount histogram (launch-reduction, 2026-06-25). The
# entropy tail ``c[c>0]; p=c*inv_n; -(p*log p).sum()`` plus the ``c.shape[0]`` count expanded to a
# boolean-mask getitem + astype + multiply + log + sum (~5 cuLaunchKernel). Two cupy ReductionKernels
# (each ONE launch, same cuLaunchKernel driver API -> genuine count reduction) fold it: ENT_RK maps each
# count to ``(c*inv_n)*log(c*inv_n)`` (0 when c==0) and reduces, NNZ_RK counts occupied cells. The c>0
# guard inside the map removes the separate boolean filter. Same plug-in entropy math -> selection-equiv.
_ENT_RK = None
_NNZ_RK = None

# FUSED entropy + occupied-cell in ONE RawKernel (launch-reduction, 2026-06-25). _ent_from_counts ran TWO
# cupy ReductionKernels (ENT_RK for sum xlog x, NNZ_RK for occupied cells) - two cuLaunchKernel per call,
# and after the per-candidate CMI scoring it was the measured #1 launch source (477). One grid-stride kernel
# now reduces both: each thread accumulates (sum xlogx, nnz) over its strided cells, a block reduction in
# shared memory folds the block, and one atomicAdd per block adds into a 2-slot output (out[0]=sum xlogx,
# out[1]=nnz). out is cp.zeros(2) - a cudaMemsetAsync, NOT a cuLaunchKernel - so the call is ONE launch.
# Same float64 plug-in entropy / same occupied-cell definition -> selection-equivalent (bit-identical sum
# up to float reduction order, which the plug-in MI is already order-tolerant to).
_ENT_NNZ_SRC = r"""
extern "C" __global__
void ent_nnz(const long long* __restrict__ c, const double inv_n, const long long M,
             double* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    double hloc = 0.0, kloc = 0.0;
    for (long long i = t; i < M; i += stride) {
        long long ci = c[i];
        if (ci > 0) { double p = (double)ci * inv_n; hloc += p * log(p); kloc += 1.0; }
    }
    __shared__ double sh_h[256];
    __shared__ double sh_k[256];
    int tid = threadIdx.x;
    sh_h[tid] = hloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(&out[0], sh_h[0]); atomicAdd(&out[1], sh_k[0]); }
}
"""
_ENT_NNZ_KERNEL = None


def _get_ent_nnz_kernel(cp):
    """Lazily compile and cache the fused entropy+nnz RawKernel (compiled once per process via NVRTC, ~30ms) so repeated calls pay only the ~30us launch cost."""
    global _ENT_NNZ_KERNEL
    if _ENT_NNZ_KERNEL is None:
        _ENT_NNZ_KERNEL = cp.RawKernel(_ENT_NNZ_SRC, "ent_nnz")
    return _ENT_NNZ_KERNEL


def _ent_from_counts(c, inv_n: float):
    """Plug-in entropy (h) + occupied-cell count (k) of an int64 count vector in ONE fused RawKernel launch.
    Falls back to the two cupy ReductionKernels on any kernel error (bit-equivalent)."""
    import cupy as cp

    global _ENT_RK, _NNZ_RK
    try:
        M = int(c.size)
        out = cp.zeros(2, dtype=cp.float64)  # cudaMemsetAsync, not a cuLaunchKernel
        threads = 256
        blocks = min(1024, max(1, (M + threads - 1) // threads))
        _get_ent_nnz_kernel(cp)((blocks,), (threads,), (c, float(inv_n), np.int64(M), out))
        h_k = cp.asnumpy(out)
        return float(-h_k[0]), round(h_k[1])
    except Exception as e:
        logger.debug("fused entropy-nnz kernel failed, falling back to the reduction kernel path: %s", e)
        if _ENT_RK is None:
            _ENT_RK = cp.ReductionKernel("int64 c, float64 inv_n", "float64 h",
                                         "c > 0 ? (c * inv_n) * log(c * inv_n) : 0.0", "a + b", "h = -a", "0.0", "mrmr_ent_rk")
            _NNZ_RK = cp.ReductionKernel("int64 c", "int64 k", "c > 0 ? 1 : 0", "a + b", "k = a", "0", "mrmr_nnz_rk")
        return float(_ENT_RK(c, float(inv_n))), int(_NNZ_RK(c))


def _nnz_from_counts(c) -> int:
    """Occupied-cell count of a bincount histogram in one ReductionKernel launch (vs >0 elementwise + sum)."""
    import cupy as cp
    global _ENT_RK, _NNZ_RK
    if _NNZ_RK is None:
        _ENT_RK = cp.ReductionKernel("int64 c, float64 inv_n", "float64 h",
                                     "c > 0 ? (c * inv_n) * log(c * inv_n) : 0.0", "a + b", "h = -a", "0.0", "mrmr_ent_rk")
        _NNZ_RK = cp.ReductionKernel("int64 c", "int64 k", "c > 0 ? 1 : 0", "a + b", "k = a", "0", "mrmr_nnz_rk")
    return int(_NNZ_RK(c))


# content-fingerprint -> max(dev_codes)+1 cardinality. y is a fit-constant and z_support a round-constant
# across the per-candidate CMI calls, so their dev .max() (a reduction + scalar D2H) recurs identically;
# memoize it. x varies -> never cached. Selection-exact (same integer cardinality). Module-level -> never on a
# pickled instance.
#
# CORRECTNESS (cudaErrorIllegalAddress fix): the cached cardinality is consumed as the per-axis HISTOGRAM WIDTH
# of the device joint-histogram kernels (``cmi_joint_entropies`` / ``joint_entropy`` size the shared tile from
# ``Kx*Ky*Kz`` and index it ``(x*Ky+y)*Kz+z`` directly). A card SMALLER than the operand's true max+1 makes a
# code index past the shared tile -> an out-of-bounds ``__shared__`` atomic (a stray write that lands on the
# neighbouring shared/adjacent allocation, surfacing later as a misattributed illegal-address at the first sync
# boundary, masked/unmasked by mempool arena rounding). The previous key ``(id(host), size, first, last)`` is
# NOT a safe content fingerprint: after the host array is GC'd a DIFFERENT operand can reuse the same id AND
# match size+endpoints yet have a LARGER max, so the cache returned a STALE-too-small card -> the OOB above.
# Key on a full O(n) content hash (the recycled-id-no-alias guard ``_fe_resident_operands.resident_operand``
# uses); the hash is pure CPU and far cheaper than the H2D .max() it guards, and CANNOT collide two operands of
# different content onto the same (too-small) card. Selection-IDENTICAL on the happy path (same card value).
# The greedy candidate search this cache serves can run under joblib backend="threading"; the lock covers
# the whole get-or-compute-or-evict sequence so two threads racing the same content-hash key don't both
# pay the device .max() and interleave a clear() with a concurrent insert.
# bench-attempt-rejected (2026-06-26): caching the y/z device codes (a _cached_dev resident-operand cache) to
# skip re-uploading the fit-constant y / round-constant z H2D per candidate saved only ~61 MB / ~10 ms on the
# F2 300k STRICT wall (below the 0.5% ship bar) - nsys shows the redundancy gate is overhead/orchestration-
# bound (GPU idle ~90%), NOT operand-H2D-bound (only ~118 operand uploads; the 1790 cudaMemcpyAsync are
# per-kernel scalar D2H, not operand re-uploads). Not worth a DATA cache's stale id()-reuse collision risk for
# ~10 ms. The real redundancy is the DOUBLE card computation, eliminated by the precomp_cards reuse below.


def _cmi_from_binned_cupy(x, y, z_joint, return_cards: bool = False, kx: int = 0, kz: int = 0):
    """Device twin of :func:`_cmi_from_binned` (marginal + conditional) via cp.unique partition counts.
    Value-order densify -> same partition -> same MI/CMI (selection-identical, fp-order ~1e-15).

    ``return_cards`` (conditional path only): also return the OCCUPIED-cell cards ``(k_z, k_xz, k_yz, k_xyz)``
    this call already computes as the byproduct of the fused entropy+nnz kernel, so the caller can feed them to
    the analytic CMI null's ``precomp_cards`` instead of recomputing the identical four histograms in
    ``joint_cardinalities_cupy``. Returns ``(cmi, cards)``; ``cards`` is ``None`` on the marginal path."""
    # bench-attempt-rejected (2026-06-26): fusing dx/dy/dz.max() into one multi_max RawKernel REGRESSED F2
    # STRICT (2021 -> 2066, +45). The y/z cardinalities are already cache-hits via _cached_card on the
    # stable-id greedy/perm paths (free), so the kernel only ADDED a launch where the cache had removed one.
    # Keep dx.max() + _cached_card(y)/_cached_card(z_support).
    import cupy as cp

    from ._fe_resident_operands import resident_operand
    # The candidate binned column is RE-SCORED across greedy steps with IDENTICAL content (H2D audit of a 1M
    # strict-resident fit: 62 calls / 25 distinct -> 296 MB of re-uploads), so route it through the content-keyed
    # resident cache too: a repeat score of the same codes reuses the resident copy (no re-upload), a genuinely
    # distinct candidate still uploads once. Selection-identical (same int64 codes). y is a FIT-CONSTANT and
    # z_joint (below) round-constant -> cached likewise (uploaded once per fit).
    # RESIDENT-INPUT fast path (device-born-generation foundation): a caller may hand ALREADY-RESIDENT int64
    # candidate codes (binned on device from the resident raw operand, e.g. the raw-redundancy CMI) -> use them
    # as-is so they never re-cross H2D (and np.asarray on a cupy array would raise). Host inputs take the
    # content-keyed cache path above.
    from ._fe_resident_operands import resident_code_operand
    if isinstance(x, cp.ndarray):
        dx = x.astype(cp.int64, copy=False).ravel()
    else:
        dx = resident_code_operand(np.asarray(x).ravel(), "cmi_cand_x")
    dy = resident_code_operand(y, "cmi_y")
    n = float(max(1, int(dx.size)))
    inv_n = 1.0 / n

    from ._fe_batched_mi import joint_entropy_gpu

    _entc = functools.partial(joint_entropy_gpu, inv_n=inv_n)

    # Content-cache the candidate cardinality on the host-input path: the same candidate is re-scored across
    # greedy steps (identical content), so its max-code fingerprint hits and skips the int(dx.max()) D2H sync.
    # A resident cp.ndarray input has no cheap host fingerprint (that would itself need a D2H), so keep the
    # direct max there.
    if kx and kx > 0:
        Kx = int(kx)  # caller-known upper bound (e.g. nbins) -> skip the int(dx.max()) sync
    elif isinstance(x, cp.ndarray):
        Kx = (int(dx.max()) + 1) if dx.size else 1
    else:
        Kx = _cached_card(x, dx)
    ky = _cached_card(y, dy)  # y is a fit-constant -> its cardinality is cached
    if z_joint is None or (hasattr(z_joint, "size") and z_joint.size == 0):
        # H(x), H(y), H(x,y) in ONE launch when the (x,y) joint fits shared (always tiny). All three from this
        # kernel -> self-consistent (the safe fusion pattern); bit-identical, falls back to the per-joint path.
        from ._fe_batched_mi import marginal_mi_entropies_gpu
        _three = marginal_mi_entropies_gpu(dx, dy, Kx, ky, inv_n)
        if _three is not None:
            (h_x, k_x), (h_y, k_y), (h_xy, k_xy) = _three
        else:
            h_x, k_x = _entc([dx], [Kx])
            h_y, k_y = _entc([dy], [ky])
            h_xy, k_xy = _entc([dx, dy], [Kx, ky])
        mi = max(0.0, (h_x + h_y - h_xy) - (k_x + k_y - k_xy - 1) / (2.0 * n))
        return (mi, None) if return_cards else mi
    # RESIDENT-INPUT: the CMI-redundancy gate hands a DEVICE-BORN z_support (``_renumber_joint_gpu`` of the
    # resident candidate codes) -> use it as-is so the round conditioning support never crosses H2D (the
    # ``cmi_z`` upload); ``resident_operand``/``np.asarray`` on a cupy array would raise. Host z rides the
    # content-keyed cache (uploaded once per round). Selection-identical: same partition -> same CMI.
    if kz and kz > 0:
        dz = z_joint.astype(cp.int64, copy=False).ravel() if isinstance(z_joint, cp.ndarray) else resident_operand(z_joint, "cmi_z", dtype=np.int64)
        _kz_local = int(kz)  # caller-known z cardinality (the mult from _combine_codes_gpu) -> no sync
    elif isinstance(z_joint, cp.ndarray):
        dz = z_joint.astype(cp.int64, copy=False).ravel()
        _kz_local = (int(dz.max()) + 1) if dz.size else 1  # device z: cardinality on device (host-byte cache N/A)
    else:
        dz = resident_operand(z_joint, "cmi_z", dtype=np.int64)  # z_support round-constant -> cached
        _kz_local = _cached_card(z_joint, dz)  # z_support is round-constant -> cardinality cached
    kz = _kz_local
    # FOUR joint entropies (z, xz, yz, xyz) in ONE launch when the (x,y,z) joint fits shared - the #1
    # cuLaunchKernel source on the STRICT redundancy gate. Bit-identical; falls back to the per-joint path.
    from ._fe_batched_mi import cmi_joint_entropies_gpu
    _four = cmi_joint_entropies_gpu(dx, dy, dz, Kx, ky, kz, inv_n)
    if _four is not None:
        (h_z, k_z), (h_xz, k_xz), (h_yz, k_yz), (h_xyz, k_xyz) = _four
    else:
        h_z, k_z = _entc([dz], [kz])
        h_xz, k_xz = _entc([dx, dz], [Kx, kz])
        h_yz, k_yz = _entc([dy, dz], [ky, kz])
        h_xyz, k_xyz = _entc([dx, dy, dz], [Kx, ky, kz])
    cmi_plugin = h_xz + h_yz - h_z - h_xyz
    cmi_bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
    cmi = max(0.0, cmi_plugin - cmi_bias)
    # the analytic-null df needs EXACTLY these occupied-cell cards (same joints, same value-order densify) ->
    # hand them back so the perm-null reuses them instead of recomputing the four histograms (precomp_cards).
    return (cmi, (int(k_z), int(k_xz), int(k_yz), int(k_xyz))) if return_cards else cmi


_YZ_CARD_CACHE: dict = {}  # (y,z)-identity -> (k_z, k_yz, ky, kz); invariant across a round's candidates
# Same joblib-threading race class as _CARD_MAX_CACHE_LOCK: covers the whole get-or-compute-or-evict sequence.
_YZ_CARD_CACHE_LOCK = threading.Lock()


def joint_cardinalities_cupy(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[int, int, int, int]:
    """Occupied-cell counts (k_z, k_xz, k_yz, k_xyz) for the analytic CMI-null df, via device cp.unique.
    Only the cardinalities (number of distinct joint codes) are needed -> cp.unique(...).size on the
    device replaces the host renumber+entropy. Value-order densify -> SAME occupied-cell count (the df is
    label-invariant). Raises on any cupy error so the caller falls back to the host path."""
    import cupy as cp

    # RESIDENT-INPUT fast path (device-born candidate-code foundation): a caller may hand ALREADY-RESIDENT int64
    # candidate codes (device-binned once, e.g. the CMI-redundancy gate) -> use them as-is so they never re-cross
    # H2D at the ``card_cand_x`` site (and ``np.asarray`` on a cupy array would raise). A host candidate is routed
    # through the content-keyed resident cache: it typically HITS the entry the per-candidate CMI already uploaded
    # (content key is role-agnostic) -> no extra H2D, and a re-evaluated candidate never re-uploads. y / z are
    # fit/round-constants. Cardinalities are label-invariant -> selection-identical.
    from ._fe_resident_operands import resident_operand, resident_code_operand
    if isinstance(x, cp.ndarray):
        dx = x.astype(cp.int64, copy=False).ravel()
    else:
        dx = resident_code_operand(np.asarray(x).ravel(), "card_cand_x")
    dy = resident_code_operand(y, "card_y")
    dz = resident_operand(z, "card_z", dtype=np.int64)
    Kx = (int(dx.max()) + 1) if dx.size else 1

    from ._fe_batched_mi import joint_nnz_gpu

    def _nc(codes, cards):
        """Occupied-cell count of the joint, fused into the histogram pass (atomicAdd 0->1 trick) in ONE launch, vs the two-launch ``joint_counts_gpu`` + ``_nnz_from_counts`` path; same integer cardinality -> identical analytic-null df."""
        return joint_nnz_gpu(codes, cards)

    # CROSS-CALL CACHE of the (y,z)-only cardinalities (launch-reduction). In the analytic-null df the gate
    # scores MANY candidates against the SAME (y target, z support) within a greedy round, so k_z / k_yz
    # (and ky / kz) are INVARIANT across those per-candidate calls - only k_xz / k_xyz depend on the
    # candidate x. Recomputing k_z + k_yz per candidate was 2 of the 4 occupied-cell histograms each call.
    # Memoize them keyed by a CONTENT FINGERPRINT of (y, z) (stable within a round, distinct across rounds).
    # CORRECTNESS: ky / kz returned here are consumed as device joint-histogram WIDTHS (``_nc([dx, dz], [Kx,
    # kz])`` etc. index ``counts[(x*Kb+...)*Kc + z]`` by card stride); a stale-too-small card from an id-reuse
    # fingerprint collision makes a code index past the histogram -> an out-of-bounds atomic (the misattributed
    # illegal-address class fixed in _cached_card). The previous ``(id(y), id(z), size, endpoints)`` key was NOT
    # collision-safe (a GC'd y/z whose id is reused with matching size+endpoints but a LARGER max returns the
    # stale card); key on full O(n) content hashes instead (cheap CPU vs the H2D .max()+nnz it guards). Same
    # cardinalities on the happy path -> df / selection unchanged.
    ya = np.ascontiguousarray(np.asarray(y).ravel())
    za = np.ascontiguousarray(np.asarray(z).ravel())
    yzkey = (int(ya.size), ya.dtype.str, hash(ya.tobytes()), int(za.size), za.dtype.str, hash(za.tobytes()))
    with _YZ_CARD_CACHE_LOCK:
        _cached = _YZ_CARD_CACHE.get(yzkey)
        if _cached is not None:
            k_z, k_yz, ky, kz = _cached
        else:
            ky = (int(dy.max()) + 1) if dy.size else 1
            kz = (int(dz.max()) + 1) if dz.size else 1
            k_z = _nc([dz], [kz])
            k_yz = _nc([dy, dz], [ky, kz])
            if len(_YZ_CARD_CACHE) > 128:
                _YZ_CARD_CACHE.clear()
            _YZ_CARD_CACHE[yzkey] = (k_z, k_yz, ky, kz)
    k_xz = _nc([dx, dz], [Kx, kz])
    k_xyz = _nc([dx, dy, dz], [Kx, ky, kz])
    return k_z, k_xz, k_yz, k_xyz


# ---------------------------------------------------------------------------
# Public CMI scorer
# ---------------------------------------------------------------------------


def score_candidates_by_cmi(
    X_cand: pd.DataFrame,
    y: np.ndarray,
    X_support: Optional[pd.DataFrame] = None,
    *,
    nbins: int = 10,
) -> pd.Series:
    """Score every candidate column by ``CMI(candidate; y | support_joint)``.

    Parameters
    ----------
    X_cand : DataFrame
        Remaining candidate columns.
    y : ndarray
        Target; promoted to int64 if not already integer-typed.
    X_support : DataFrame or None
        Currently-selected support columns. ``None`` (or empty) -> CMI
        reduces to marginal ``MI(candidate; y)`` and the function behaves
        as a batch-MI scorer (useful for the seed step).
    nbins : int
        Bins per column for equi-frequency quantile binning.

    Returns
    -------
    pd.Series indexed by ``X_cand.columns`` holding the CMI value for each.
    """
    if X_cand.empty:
        return pd.Series(dtype=np.float64)
    # This used to `.astype(np.int64)` (TRUNCATE) any
    # non-integer y BEFORE the np.unique densify below - for a continuous y confined to one integer
    # bucket (e.g. a [0,1) probability), truncation collapses every distinct value to the SAME integer
    # first, so the subsequent np.unique can no longer recover the distinctness (the exact B-18 bug class
    # already fixed in 7 sibling orth-scoring files via _coerce_y_int64, but never applied here). Densify
    # directly via np.unique on the RAW y instead - safe for an already-integer y too (same result).
    y_arr = np.asarray(y)
    # Bin y by unique-value remap (renumbers to dense 0..K-1; the caller's y may already be class-typed,
    # in which case this is a no-op renumber, or a raw continuous target, in which case this is what
    # actually turns it into usable class codes without truncation).
    _, y_bin = np.unique(y_arr, return_inverse=True)
    y_bin = y_bin.astype(np.int64)

    if X_support is None or X_support.shape[1] == 0:
        z_joint: Optional[np.ndarray] = None
    else:
        sup_bins = [_quantile_bin(X_support[c].to_numpy(), nbins=nbins) for c in X_support.columns]
        z_joint, _ = _renumber_joint(*sup_bins)

    cand_cols = list(X_cand.columns)
    # BATCHED born-on-device path under STRICT (default OFF -> per-candidate CPU loop, byte-identical):
    # bin all candidates into one (n, K) code matrix and score CMI for EVERY candidate in ONE device
    # workload (batched_cmi_gpu), instead of a per-candidate cp.unique CMI. Parity-pinned selection-equiv.
    if _cmi_gpu_enabled(n=int(y_bin.shape[0]), p=len(cand_cols)) and len(cand_cols) > 1:
        try:
            from ._fe_batched_mi import batched_cmi_gpu, batched_quantile_bin_gpu
            X_float = np.empty((y_bin.shape[0], len(cand_cols)), dtype=np.float64)
            for j, c in enumerate(cand_cols):
                X_float[:, j] = X_cand[c].to_numpy()
            if np.isfinite(X_float).all():
                # Born-on-device: bin the whole candidate matrix on the GPU (one batched cp.percentile
                # sort) and keep the codes RESIDENT, scoring CMI on them without a code H2D round-trip.
                import cupy as cp
                X_codes_dev = batched_quantile_bin_gpu(cp.asarray(X_float), nbins)
                # codes_trusted: binner-produced (batched_quantile_bin_gpu) + dense renumbered y/z are 0-based ->
                # the range guard cannot fire; skip its 2 blocking min/max syncs on the resident hot path (FIX1).
                cmis = batched_cmi_gpu(X_codes_dev, y_bin, z_joint, codes_trusted=True)
            else:
                # Non-finite columns -> host equi-freq binning (handles nan/inf), then device CMI.
                X_codes = np.empty((y_bin.shape[0], len(cand_cols)), dtype=np.int64)
                for j in range(len(cand_cols)):
                    X_codes[:, j] = _quantile_bin(X_float[:, j], nbins=nbins)
                cmis = batched_cmi_gpu(X_codes, y_bin, z_joint, codes_trusted=True)  # host equi-freq binner -> 0-based
            return pd.Series({c: float(cmis[j]) for j, c in enumerate(cand_cols)}, dtype=np.float64)
        except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
            logger.debug("GPU-resident batched CMI path failed (%s: %s) -- falling back to exact CPU loop", type(e).__name__, e)
    out = {}
    for c in cand_cols:
        x_bin = _quantile_bin(X_cand[c].to_numpy(), nbins=nbins)
        out[c] = _cmi_from_binned(x_bin, y_bin, z_joint)
    return pd.Series(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# End-to-end greedy CMI constructor
# ---------------------------------------------------------------------------


def greedy_cmi_fe_construct(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    seed_cols_count: int = 4,
    top_k: int = 5,
    include_unary: bool = True,
    include_binary: bool = True,
    include_trig_on_bounded: bool = True,
    min_cmi_gain: float = 0.005,
    nbins: int = 10,
    seed: int = 0xC011,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end CMI-greedy feature constructor.

    ``seed``: seeds the noise-floor permutation RNG
    (previously hardcoded to ``0xC011`` with no way to vary it). Defaults to the historical constant so
    existing pinned tests/behaviour stay byte-identical when the caller does not override it; pass
    ``self.random_seed`` (or any other seed) to decorrelate the FE admission gate across nominally-
    independent bootstrap/multi-seed replicates (stability selection, seed ensembles).

    Pipeline:

    1. Enumerate UNARY candidates over the FULL numeric column pool (NOT a
       top-N seed pool). The whole point of CMI ranking is that columns
       with near-zero marginal ``MI(x; y)`` can still emit transforms that
       carry the signal - on ``y = sign(x^2 - 1)``, ``MI(x; y) ~= 0``
       because ``x`` is symmetric, yet ``square(x)`` is perfectly
       informative. Restricting unary enumeration to the top-N raw-MI
       cols would discard exactly the signal Layer 60 is designed to
       recover. BINARY candidates are enumerated only over the
       top-``seed_cols_count`` raw cols by marginal ``MI(x; y)`` because
       the pair explosion is O(N^2 * |BINARY_TRANSFORMS|) and quickly
       exceeds the gain.
    2. Materialise every candidate; drop near-constants.
    3. Start the conditioning support Z EMPTY (Z grows step-by-step from
       the greedy loop). This avoids the fragmentation trap of dumping
       several raw cols into Z up front: when several raw cols enter Z,
       the joint Z cardinality climbs into the hundreds and the CMI of
       any candidate collapses toward noise (cells average < 5 samples).
       The greedy loop itself caps Z growth with the contingency budget
       below.
    4. Greedy loop: compute ``CMI(cand; y | support)`` for every remaining
       candidate, pick the highest, add it to support if it clears
       ``min_cmi_gain``; otherwise stop. Z is grown ONLY when the resulting
       joint cardinality stays under ``n / 5`` (chi-squared rule of
       thumb: cells must average >= 5 samples for CMI to be stable).
       Past that cap, the winner is still appended but Z is frozen, so
       subsequent CMI gains stay measurable.
    5. Append winners to X; return (X_augmented, scores) where ``scores``
       is a DataFrame with one row per appended column ordered by
       selection sequence.
    """
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    from ._mi_greedy_fe import (
        generate_mi_greedy_features,
        iter_candidates,
    )
    from ._orthogonal_univariate_fe import _mi_classif_batch

    st.empty_scores = pd.DataFrame(columns=[
        "engineered_col", "transform", "source_cols",
        "cmi_at_selection", "step",
    ])

    st._cols_source = cols if cols is not None else X.columns
    st.candidates_pool = [c for c in st._cols_source if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if not st.candidates_pool:
        return X, st.empty_scores

    # This used to `.astype(np.int64)` (TRUNCATE) any
    # non-integer y here, BEFORE both the raw_mi scoring below AND the np.unique densify further down -
    # for a continuous y confined to one integer bucket, truncation collapses every distinct value to the
    # SAME integer first, so densification downstream can no longer recover the distinctness (the B-18 bug
    # class). Densify via np.unique ONCE, up front, and reuse the dense y_bin everywhere below - safe for
    # an already-integer/already-dense y too (pure renumber, no-op if already 0..K-1).
    st.y_arr = np.asarray(y)
    _, y_bin = np.unique(st.y_arr, return_inverse=True)
    y_bin = y_bin.astype(np.int64)

    # 1. Pick the top-N raw cols by marginal MI as the BINARY-pair source
    #    pool (controls the O(N^2 * |BINARY_TRANSFORMS|) explosion).
    #    Unary candidates still enumerate over the full pool below.
    st.raw_arr = X[st.candidates_pool].to_numpy(dtype=np.float64)
    st.raw_mi = _mi_classif_batch(st.raw_arr, y_bin, nbins=nbins)
    st.order = np.argsort(-st.raw_mi, kind="stable")  # plug-in MI is quantised, so ties are real: break them by position, reproducibly
    st.binary_seed_cols = [st.candidates_pool[i] for i in st.order[: int(seed_cols_count)]] if int(seed_cols_count) > 0 else list(st.candidates_pool)

    # 2. Enumerate candidates. UNARY over the full pool (so transforms on
    #    symmetric / interaction-only cols are never silently dropped),
    #    BINARY only over the seeded subset (pair explosion control).
    st.cands = []
    if include_unary:
        st.cands.extend(iter_candidates(
            X, cols=st.candidates_pool,
            include_unary=True, include_binary=False,
            include_trig_on_bounded=include_trig_on_bounded,
        ))
    if include_binary:
        st.cands.extend(iter_candidates(
            X, cols=st.binary_seed_cols,
            include_unary=False, include_binary=True,
            include_trig_on_bounded=False,
        ))
    engineered, st.parsed = generate_mi_greedy_features(X, st.cands)
    if engineered.empty:
        return X, st.empty_scores

    # 3. y_bin was already densified up front (reuses it here, no re-densify
    #    needed). Start Z EMPTY. Z grows step-by-step from greedy picks (under the fragmentation cap
    #    below). Starting with several raw cols in Z up front pushes joint Z cardinality past the
    #    chi-squared contingency budget and collapses every candidate's CMI toward noise - defeats the
    #    purpose of CMI ranking.
    st.n_samples = int(y_bin.size)
    st.frag_cap = max(2, st.n_samples // 5)
    # RESIDENT fit-constant y for the GPU-strict greedy hot path: upload y_bin to the device ONCE here and hand
    # the SAME cupy array to every per-round ``batched_cmi_gpu`` call (it uses an already-resident y as-is, no
    # re-upload). Routed through ``resident_operand`` only to upload (and to be cleared at FE teardown); kept on a
    # DISTINCT role from the per-permutation shuffled y so the transient null-draw y can never evict it. None when
    # GPU is off (host path uses y_bin directly, byte-identical).
    st.y_bin_dev = None
    if _cmi_gpu_enabled(n=st.n_samples):
        try:
            from ._fe_resident_operands import resident_code_operand as _resident_code_operand
            st.y_bin_dev = _resident_code_operand(y_bin, "cmi_greedy_y_fixed")
        except Exception as e:
            logger.debug("resident_code_operand for y_bin failed, falling back to the per-call upload path: %s", e)
            st.y_bin_dev = None
    z_joint: Optional[np.ndarray] = None
    z_joint_dev = None  # resident twin of z_joint, maintained alongside it once cand_bins_dev is available
    st.z_card = 1

    # 4. Bin every engineered candidate up front. Compute a sortable
    #    bin fingerprint (tuple of the sorted unique bin counts) used
    #    below for monotone-equivalence dedup against already-picked
    #    winners - when Z hits the fragmentation cap (frozen Z),
    #    monotone-equivalent candidates would otherwise tie at the same
    #    plug-in CMI and all get picked.
    cand_names = list(engineered.columns)
    st.name_to_parsed = dict(zip(cand_names, st.parsed))

    def _bin_fingerprint(b: np.ndarray) -> bytes:
        """Hashable fingerprint of a binned candidate array; monotone-equivalent candidates (identical bin assignment under equi-frequency quantization) collapse to the same fingerprint for dedup against already-picked winners."""
        return b.tobytes()

    # RESIDENT candidate codes: when GPU-strict-resident is on, bin every candidate ONCE on-device
    # (``cand_bins_dev``) and fingerprint via a device-side reduction instead of the host ``_quantile_bin`` +
    # ``.tobytes()`` pair below. This closes the confirmed residency gap (``_fe_gpu_strict.py`` docstring, "KNOWN
    # NON-PRODUCTION RESIDENCY GAP"): the host path forces one bulk (n,) D2H PER CANDIDATE just to hash its
    # bytes. ``cand_bins`` (host) stays LAZY here - populated on demand only for the small subset of names the
    # Z-fold step actually touches (at most ``top_k`` winners, via ``_host_bins`` below), not eagerly for every
    # candidate. Falls back to the fully-eager host path (unchanged, byte-identical) on any cupy failure or when
    # GPU-strict-resident is off - selection-identical either way (fingerprint is a dedup key, not a scored
    # quantity; a 64-bit reduction hash is the same collision-domain acceptance the resident cache
    # (``_content_hash``) already uses elsewhere in this codebase).
    cand_bins_dev: dict = {}
    st.cand_fp = {}
    st._resident_fp_ok = False
    _host_bins, cand_bins_dev, rng_floor = _greedy_cmi_fe_constr_step1_content_hash_already(cand_names, engineered, nbins, st, _bin_fingerprint, seed, cand_bins_dev)

    _noise_floor_for_current_z = _greedy_cmi_fe_constr_step2_rng_raw_small(cand_names, rng_floor, y_bin, z_joint_dev, z_joint, cand_bins_dev, _host_bins, st)
    _greedy_cmi_fe_constr_step3_while_st_remaining(st, top_k, _noise_floor_for_current_z, min_cmi_gain, z_joint_dev, z_joint, y_bin, cand_bins_dev, _host_bins, nbins)

    st.scores = pd.DataFrame(st.rows, columns=[
        "engineered_col", "transform", "source_cols",
        "cmi_at_selection", "step",
    ])
    if st.winners:
        st.X_aug = pd.concat([X, engineered[st.winners]], axis=1)
    else:
        st.X_aug = X.copy(deep=False)
    return st.X_aug, st.scores


def greedy_cmi_fe_construct_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    seed_cols_count: int = 4,
    top_k: int = 5,
    include_unary: bool = True,
    include_binary: bool = True,
    include_trig_on_bounded: bool = True,
    min_cmi_gain: float = 0.005,
    nbins: int = 10,
    seed: int = 0xC011,
):
    """Same as :func:`greedy_cmi_fe_construct` but additionally returns a list
    of ``EngineeredRecipe`` objects (one per appended column) so MRMR.transform
    can replay each column on test data without re-running CMI scoring AND
    without referencing y.

    Recipes reuse kind ``"mi_greedy_transform"`` (same as Layer 26) so the
    replay code path is shared infrastructure.

    ``seed``: see :func:`greedy_cmi_fe_construct`'s matching parameter.
    """
    from ._mi_greedy_fe import _parse_binary_name, _parse_unary_name
    from .engineered_recipes import build_mi_greedy_transform_recipe

    X_aug, scores = greedy_cmi_fe_construct(
        X, y,
        cols=cols, seed_cols_count=seed_cols_count, top_k=top_k,
        include_unary=include_unary, include_binary=include_binary,
        include_trig_on_bounded=include_trig_on_bounded,
        min_cmi_gain=min_cmi_gain, nbins=nbins, seed=seed,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    for name in appended:
        parsed_bin = _parse_binary_name(name)
        parsed_un = _parse_unary_name(name)
        if parsed_bin is not None:
            tname, col_i, col_j = parsed_bin
            recipes.append(build_mi_greedy_transform_recipe(
                name=name, transform=tname, src_names=(col_i, col_j),
            ))
        elif parsed_un is not None:
            tname, col = parsed_un
            recipes.append(build_mi_greedy_transform_recipe(
                name=name, transform=tname, src_names=(col,),
            ))
        else:
            log_throttle(
                logger,
                "mi_greedy_cmi_fe_cannot_parse_column",
                logging.WARNING,
                "greedy_cmi_fe_construct_with_recipes: cannot parse " "engineered column %r back to (transform, source); skipping " "recipe.",
                name,
            )
    return X_aug, scores, recipes
