"""Split out of `src/mlframe/feature_selection/filters/_mi_greedy_cmi_fe.py` to keep both modules under the size limit.

Moved verbatim: every body was compared byte-for-byte against the pre-split file, and the moved range
was checked to read no module-level name left behind - see
`pyutilz.dev.freevar_analysis.split_out_module`.

See `src/mlframe/feature_selection/filters/_mi_greedy_cmi_fe.py` for the surrounding context these definitions were written in.
"""

from __future__ import annotations

import functools
import threading
import numpy as np
from typing import Any
from ._fe_resident_operands import _content_hash_memoized
from ._kernel_types import render
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
    _cmi_candidate_batch_on_device,
    _cmi_gpu_enabled,
    _CARD_MAX_CACHE,
    _CARD_MAX_CACHE_LOCK,
    _cached_card,
    _cmi_from_binned_fixed_yz_cupy,
    _greedy_cmi_fe_constr_step1_content_hash_already,
    _greedy_cmi_fe_constr_step2_rng_raw_small,
    _greedy_cmi_fe_constr_step3_while_st_remaining,
)

# Fused entropy + occupied-cell reduction over a bincount histogram (launch-reduction). The
# entropy tail ``c[c>0]; p=c*inv_n; -(p*log p).sum()`` plus the ``c.shape[0]`` count expanded to a
# boolean-mask getitem + astype + multiply + log + sum (~5 cuLaunchKernel). Two cupy ReductionKernels
# (each ONE launch, same cuLaunchKernel driver API -> genuine count reduction) fold it: ENT_RK maps each
# count to ``(c*inv_n)*log(c*inv_n)`` (0 when c==0) and reduces, NNZ_RK counts occupied cells. The c>0
# guard inside the map removes the separate boolean filter. Same plug-in entropy math -> selection-equiv.
_ENT_RK = None
_NNZ_RK = None

# FUSED entropy + occupied-cell in ONE RawKernel (launch-reduction). _ent_from_counts ran TWO
# cupy ReductionKernels (ENT_RK for sum xlog x, NNZ_RK for occupied cells) - two cuLaunchKernel per call,
# and after the per-candidate CMI scoring it was the measured #1 launch source (477). One grid-stride kernel
# now reduces both: each thread accumulates (sum xlogx, nnz) over its strided cells, a block reduction in
# shared memory folds the block, and one atomicAdd per block adds into a 2-slot output (out[0]=sum xlogx,
# out[1]=nnz). out is cp.zeros(2) - a cudaMemsetAsync, NOT a cuLaunchKernel - so the call is ONE launch.
# Same double precision plug-in entropy / same occupied-cell definition -> selection-equivalent (bit-identical sum
# up to float reduction order, which the plug-in MI is already order-tolerant to).
_ENT_NNZ_SRC = render(r"""
extern "C" __global__
void ent_nnz(const long long* __restrict__ c, const {ACC} inv_n, const long long M,
             {ACC}* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    {ACC} hloc = 0.0, kloc = 0.0;
    for (long long i = t; i < M; i += stride) {
        long long ci = c[i];
        if (ci > 0) { {ACC} p = ({ACC})ci * inv_n; hloc += p * log(p); kloc += 1.0; }
    }
    __shared__ {ACC} sh_h[256];
    __shared__ {ACC} sh_k[256];
    int tid = threadIdx.x;
    sh_h[tid] = hloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(&out[0], sh_h[0]); atomicAdd(&out[1], sh_k[0]); }
}
""")
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
        # Intentional float64: entropy accumulator kept f64 (cudaMemset dtype)
        out = cp.zeros(2, dtype=cp.float64)  # cudaMemsetAsync, not a cuLaunchKernel
        threads = 256
        blocks = min(1024, max(1, (M + threads - 1) // threads))
        _get_ent_nnz_kernel(cp)((blocks,), (threads,), (c, float(inv_n), np.int64(M), out))
        h_k = cp.asnumpy(out)
        return float(-h_k[0]), round(h_k[1])
    except Exception as e:
        logger.warning("fused entropy-nnz kernel failed, falling back to the reduction kernel path: %s", e)
        if _ENT_RK is None:
            # Intentional float64: ReductionKernel signature contract (float64)
            _ENT_RK = cp.ReductionKernel("int64 c, float64 inv_n", "float64 h",
                                         "c > 0 ? (c * inv_n) * log(c * inv_n) : 0.0", "a + b", "h = -a", "0.0", "mrmr_ent_rk")
            _NNZ_RK = cp.ReductionKernel("int64 c", "int64 k", "c > 0 ? 1 : 0", "a + b", "k = a", "0", "mrmr_nnz_rk")
        return float(_ENT_RK(c, float(inv_n))), int(_NNZ_RK(c))

def _nnz_from_counts(c) -> int:
    """Occupied-cell count of a bincount histogram in one ReductionKernel launch (vs >0 elementwise + sum)."""
    import cupy as cp
    global _ENT_RK, _NNZ_RK
    if _NNZ_RK is None:
        # Intentional float64: ReductionKernel signature contract (float64)
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
        mi = max(0.0, (h_x + h_y - h_xy) + (k_x + k_y - k_xy - 1) / (2.0 * n))
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

def _host_key_array(cp: Any, v: Any) -> "tuple[np.ndarray, bool]":
    """``(host array, stable)`` of ``v`` for the content key: ``stable`` is True when ``v`` is a host array that is already 1-D and C-contiguous (returned as the SAME object, so an identity-keyed hash memo can hit)."""
    if isinstance(v, cp.ndarray):
        return np.ascontiguousarray(cp.asnumpy(v).ravel()), False
    a = np.asarray(v)
    if a.ndim == 1 and a.flags.c_contiguous:
        return a, True
    return np.ascontiguousarray(a.ravel()), False


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
    dy = y.astype(cp.int64, copy=False).ravel() if isinstance(y, cp.ndarray) else resident_code_operand(y, "card_y")
    dz = z.astype(cp.int64, copy=False).ravel() if isinstance(z, cp.ndarray) else resident_operand(z, "card_z", dtype=np.int64)
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
    # y / z may be device arrays (the resident support): the content key needs host bytes, and ``np.asarray`` on a cupy array raises
    ya, y_stable = _host_key_array(cp, y)
    za, z_stable = _host_key_array(cp, z)
    # A host array that is already 1-D and contiguous is the same object across the calls of a round, so its content hash is memoised on its identity (0.004 ms) instead of hashing a fresh ``tobytes()`` copy of both arrays per call (0.45 ms,
    # most of the cost of a per-pair null at 100000 rows); an array taken from the device is new on every call and keeps the byte hash.
    yzkey = (
        int(ya.size), ya.dtype.str, _content_hash_memoized(ya) if y_stable else hash(ya.tobytes()),
        int(za.size), za.dtype.str, _content_hash_memoized(za) if z_stable else hash(za.tobytes()),
    )
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
