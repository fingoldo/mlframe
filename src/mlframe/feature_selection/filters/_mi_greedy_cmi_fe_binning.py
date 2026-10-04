"""Helpers carved out of ``_mi_greedy_cmi_fe_steps`` to keep that module under its size budget."""
from __future__ import annotations

import logging

import numpy as np

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


logger = logging.getLogger(__name__)

# GPU quantile-bin crossover (2026-06-28, synchronized micro-bench of _quantile_bin_gpu incl. code D2H, GTX
# 1050 Ti, nbins=10): a single host column round-tripped to the device for equi-frequency binning (H2D +
# cp.percentile sort + cp.searchsorted + code D2H) only beats the host introselect-partition np.quantile path
# well above the launch/transfer floor - n=20k CPU 0.92ms vs GPU 1.67ms; n=35k near-tie 1.53 vs 1.71ms; n=100k
# GPU 2.10 vs CPU 4.12ms = 2x; n=300k GPU 3.06 vs CPU 14.1ms = 4.6x. The gate is set at 50k (clear of the 35k
# near-tie) so every routed call is a decisive win; the small (3k/20k) gate-redundancy columns stay on the
# host, where the fixed ~1.7ms device round-trip overhead loses to numpy.
_GPU_QBIN_MIN_ROWS = 50_000


logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def _qbin_float_dtype():
    """Float dtype for the ``qbin_x`` candidate-column upload the device quantile binners do. Under
    ``MLFRAME_FE_VRAM_F32`` (the FE-generation dtype discipline) bin in FLOAT32 - the candidate float upload is
    HALF the bytes and the equi-frequency partition is selection-equivalent to float64 (f32 percentile edges
    agree with f64 at all but ~1e-5 of near-edge rows, below the bin resolution the redundancy gate keys on;
    this is the SAME f32/f64 discipline the FE materialise + the GPU discretiser already use). Falls back to
    float64 when the flag is off or the probe fails, keeping BOTH binners on ONE dtype so identical candidate
    content still content-dedups in the resident-operand cache."""
    import cupy as cp
    try:
        from ._fe_gpu_batch._devices import fe_gpu_f32_enabled
        return cp.float32 if fe_gpu_f32_enabled() else cp.float64
    except Exception as e:
        logger.debug("_qbin_float_dtype: fe_gpu_f32_enabled() probe failed, falling back to float64: %s", e)
        return cp.float64


def _sync_free_qbin_codes(cp, xd, nbins: int):
    """Equi-frequency bin codes for a resident 1-D column WITHOUT any device->host sync.

    Replaces the ``cp.unique(cp.percentile(...))`` edge-dedup (which syncs to size its output) + the
    ``int(edges.size) <= 2`` degenerate check (a second sync) with a branchless device-only construction that is
    partition- AND cardinality-equivalent to the host ``np.unique(np.quantile(a,qs))[1:-1]`` binning for EVERY
    column shape (verified across continuous / low-card / mass-point / binary / two-value / constant columns).

    numpy takes ``unique(edges)[1:-1]`` - i.e. it drops duplicate edges, the global-min edge, and the
    global-max edge, EXCEPT the degenerate 2-distinct-value case keeps one interior split (2 bins). The
    device twin marks for exclusion (-> +inf, pushed past the data by the sort so ``searchsorted`` ignores it):
    every entry equal to the min value, every adjacent duplicate, and every entry equal to the max value ONLY
    when an interior distinct value exists (``has_interior``) - so a 2-distinct column keeps its single split
    while a 3+-distinct column merges its top two bins exactly as ``unique[1:-1]`` does. All ops are elementwise
    / reductions returning device scalars, so nothing crosses the bus; the caller returns the codes resident or
    does the one bulk (n,) result copy.
    """
    # Percentile edges via an explicit sort + linear interpolation (numpy's default 'linear' method) rather
    # than cp.percentile, which does an internal host read (int index) that syncs. Sort is O(n log n) - the
    # same order cp.percentile pays internally - but stays fully device-side. n is a host shape read (no sync).
    #
    # bench-attempt-rejected (2026-07-03): replacing this full cp.sort with cp.partition(xd, kth) at only the
    # nbins+1 quantile ranks - kth = floor/ceil of qs*(n-1), buildable ENTIRELY host-side from n + the fixed qs
    # linspace, so partition needs NO device read and STAYS sync-free (unlike a prior attempt that read
    # cp.asnumpy(ranks)). Rejected because it is not a meaningful win: cupy's partition is a bitonic partial sort
    # whose cost is governed by the LARGEST kth, and the top interior quantile edge sits at rank ~0.9*(n-1), so
    # partition does near-full-sort work anyway (a prior micro-bench measured cp.partition only MARGINALLY faster
    # than cp.sort). The content-keyed code cache below (resident_qbin_codes) removes the sort ENTIRELY on the
    # ~half of calls that re-bin an already-seen column - a strictly larger, sync-free, bit-identical win than
    # shaving a partial-sort constant off every call. Keep the simple full sort here.
    n = int(xd.size)
    qs = cp.linspace(0.0, 100.0, int(nbins) + 1)
    xs = cp.sort(xd.ravel())
    pos = qs / 100.0 * (n - 1)
    lo = cp.floor(pos).astype(cp.int64)
    hi = cp.minimum(lo + 1, n - 1)
    frac = pos - lo
    # numpy's interpolation form: exact when xs[lo] == xs[hi]. a*(1-f) + b*f rounds a tied value one ULP off (-0.9 -> -0.9000000000000001),
    # and searchsorted(side="right") then moves that value's whole mass into the neighbouring bin.
    e = xs[lo] + (xs[hi] - xs[lo]) * frac  # sorted ascending, nbins+1 edges
    dup = cp.empty(e.shape, dtype=bool)
    dup[0] = False
    dup[1:] = e[1:] == e[:-1]
    emin = e[0]
    emax = e[-1]
    has_interior = cp.any((e > emin) & (e < emax))  # 0-dim device bool - no sync
    excl = (e == emin) | dup | (has_interior & (e == emax))
    e2 = cp.sort(cp.where(excl, cp.inf, e))  # excluded edges -> +inf, sorted to the tail
    return cp.searchsorted(e2, xd, side="right").astype(cp.int64)


def _quantile_bin_gpu(a: np.ndarray, nbins: int):
    """Device equi-frequency bin of an all-finite 1-D float column -> host int64 codes, selection-equivalent
    to the numpy ``_quantile_bin`` fast path.
    Returns ``None`` on any failure so the caller transparently keeps the numpy path. NEVER frees the cupy
    memory pool. Codes can 1-off the numpy codes at <~1e-5 of rows where cp.percentile and np.quantile round a
    boundary differently - below the bin resolution, MI/cardinality selection-equivalent (the acceptance bar).

    Device 1-D twin of the numpy fast path: ``cp.percentile`` (on the RAVELLED array - cp.percentile(X,axis=0)
    returns WRONG edges for an (n,1) column, the known cupy single-column bug guarded in
    ``_gpu_resident_discretize_codes``) + ``cp.unique`` edge-dedup + ``cp.searchsorted`` on the deduped interior
    edges. The unique-dedup is load-bearing on low-cardinality / mass-point columns: skipping it (or hitting the
    (n,1) percentile bug) splits a tied bin and breaks the occupied-bin partition the redundancy gate keys on.
    This mirrors the numpy ``np.unique(np.quantile(a,qs))`` + ``searchsorted(edges[1:-1], a, 'right')`` exactly
    (cp.percentile uses [0,100] vs np.quantile's [0,1] - same linear-interpolation edges)."""
    try:
        import cupy as cp

        from ._fe_resident_operands import resident_qbin_codes
        # The same candidate column is re-binned across greedy steps with identical content (H2D audit of a 1M
        # strict-resident fit: 42 calls / 20 distinct). resident_qbin_codes content-caches BOTH the float upload
        # (via resident_operand) AND the codes, so a repeat bin of identical content skips the re-upload AND the
        # O(n log n) sort (the cub DeviceMergeSort hotspot); bit-identical (same values -> same edges -> same
        # codes). Distinct columns compute once (genuine data).
        # sync-free device dedup (was cp.unique + size check, two D2H syncs); only the bulk (n,) result copies.
        return cp.asnumpy(resident_qbin_codes(a, nbins, _qbin_float_dtype(), _sync_free_qbin_codes))
    except Exception:
        logger.debug("GPU _quantile_bin failed; numpy fallback", exc_info=True)
        return None


def _quantile_bin_gpu_resident(a: np.ndarray, nbins: int):
    """DEVICE-RESIDENT equi-frequency bin of an all-finite 1-D float column -> RESIDENT cupy int64 codes.

    Identical partition to :func:`_quantile_bin_gpu` (same content-keyed ``qbin_x`` float upload, same ravelled
    ``cp.percentile`` + ``cp.unique`` edge-dedup + ``cp.searchsorted``), but returns the codes RESIDENT on the
    device instead of ``cp.asnumpy``-ing them. This is the device-born candidate-code foundation for the
    CMI-redundancy gate: a candidate is binned ONCE here and its resident int64 codes are handed straight to the
    resident-input branches of ``_cmi_from_binned_cupy`` / ``batched_cmi_gpu`` / ``conditional_perm_null_gpu``,
    so the derived candidate codes never re-cross H2D (the ``cmi_cand_x`` / ``card_cand_x`` / ``permnull_cand_x``
    re-uploads the host-code path incurred). Returns ``None`` on any cupy failure so the caller keeps the
    host ``_quantile_bin`` path (which yields a host int64 array the existing content-keyed cache then uploads).
    NEVER frees the cupy memory pool. Selection-equivalent to the host binning (documented in ``_quantile_bin``)."""
    try:

        from ._fe_resident_operands import resident_qbin_codes
        # Fully sync-free: codes stay RESIDENT, no cp.unique / size D2H (the whole point of the resident path).
        # resident_qbin_codes content-caches the codes so a repeat bin skips the sort entirely (bit-identical).
        return resident_qbin_codes(a, nbins, _qbin_float_dtype(), _sync_free_qbin_codes)
    except Exception:
        logger.debug("GPU-resident _quantile_bin failed; host fallback", exc_info=True)
        return None


def _quantile_bin(col: np.ndarray, nbins: int) -> np.ndarray:
    """Equi-frequency bin a 1-D float column into ``nbins`` integer classes.

    Constant or near-constant columns degenerate to a single class (0). NaN
    / Inf are mapped to bin 0 (caller is expected to scrub upstream; we keep
    the fallback for safety).

    By design, a low-cardinality column can collapse to a single (or two) bin even when it is informative: ``np.unique(np.quantile(...))`` dedupes the
    equi-frequency edges, so a column with few distinct values yields ``edges.size <= 2`` and reads MI ~= 0 here. This is the price of monotone-invariance
    (the binning depends only on rank order, not raw spacing) and is intentional, NOT a bug - the marginal-MI path (Layer 26) sees such columns through its
    own binning, and the CMI-greedy step is meant to score CONDITIONAL gain on top of that. Do not "fix" this by switching to value-width bins; that would
    break the rank-invariance the CMI numbers rely on (see the bench-attempt-rejected note below for why rank-based rebinning was rejected too).
    """
    # bench-attempt-rejected (2026-06-01): replacing np.quantile value-edge
    # binning with a numba argsort rank-based equi-frequency binner was BOTH
    # slower (0.60x: 1222ms vs 730ms / 411 calls - numpy np.quantile uses
    # introselect partition, not a full sort, and beats numba argsort) AND
    # NOT MI-equivalent here: on tied/discrete columns rank-binning splits ties
    # across bins, shifting MI(X;y) ~2x (disc: 5.8e-5 -> 1.1e-4) and thus the
    # CMI-greedy selection. The "binning-tie-invariance" note at
    # _orthogonal_univariate_fe.py:451 applies only to that hermite MI kernel,
    # NOT to this CMI-greedy path. Keep the value-edge np.quantile binning.
    a = np.asarray(col, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, nbins + 1)
    # Fast path: an all-finite column (the production nan-filled case) skips the
    # boolean-mask materialisation + the ``a[finite_mask]`` gather copy and bins
    # ``a`` in place. Bit-identical (when every value is finite, finite == a and
    # finite_mask selects every row). ~1.3x at the CMI-greedy call volume.
    if np.isfinite(a).all():
        # GPU path (STRICT-resident): the n-sized equi-frequency binning of the gate-redundancy /
        # subsumption / additive-fusion continuous columns is 6x on-device at n=300k (synchronized bench).
        # NO size gate under STRICT (2026-07-02, user contract): strict mode is 100% GPU residency of data
        # and kernels - a size crossover is exactly the KTC-style host dispatch strict forbids, so EVERY
        # finite column bins on the device (the small-column device round-trip overhead is the accepted
        # residency price; the _GPU_QBIN_MIN_ROWS crossover note below documents its wall cost). The no-CUDA
        # / non-strict default keeps the byte-identical numpy path.
        try:
            from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
            _gpu_on = fe_gpu_strict_resident_enabled()
        except Exception as e:
            logger.debug("fe_gpu_strict_resident_enabled() check failed, defaulting to non-resident: %s", e)
            _gpu_on = False
        if _gpu_on:
            _g = _quantile_bin_gpu(a, nbins)
            if _g is not None:
                return np.asarray(_g)
        edges = np.unique(np.quantile(a, qs))
        out = np.zeros(a.size, dtype=np.int64)
        if edges.size <= 2:
            if edges.size == 2:
                out[:] = (a >= edges[1]).astype(np.int64)
            return out
        return np.searchsorted(edges[1:-1], a, side="right").astype(np.int64)

    finite_mask = np.isfinite(a)
    out = np.zeros(a.size, dtype=np.int64)
    if not finite_mask.any():
        return out
    finite = a[finite_mask]
    # Quantile edges; drop dupes so constant-tail columns don't crash.
    edges = np.unique(np.quantile(finite, qs))
    if edges.size <= 2:
        # All finite values identical (or just two unique values) -> nothing
        # to bin against; return a 2-bin indicator if there are two values,
        # else all-zero.
        if edges.size == 2:
            out[finite_mask] = (a[finite_mask] >= edges[1]).astype(np.int64)
        return out
    # ``np.searchsorted(edges[1:-1], a)`` gives bin indices in [0, nbins-1]
    # robust to the equi-frequency-edges path (rightmost edge dropped).
    inner = edges[1:-1]
    bins_finite = np.searchsorted(inner, finite, side="right")
    out[finite_mask] = bins_finite.astype(np.int64)
    return out


_FAC_ARRAY_CAP = 16_000_000


@njit(cache=True)
def _factorize_dense_njit(joint: np.ndarray) -> tuple:
    """Factorize an int64 array to dense first-seen ids in one O(n) pass.

    Replaces ``np.unique(joint, return_inverse=True)``'s O(n log n) sort. The
    per-fold joint is bounded (``old_dense(0..mult-1) + c*mult``), so when the
    max id keeps the ``seen`` buffer small we use a direct-array counting pass
    (array indexing, no hashing - ~10x over the typed.Dict form, ~17x over
    np.unique on the common low-cardinality group/cat joints). A typed.Dict
    fallback guards the rare cartesian-blow-up (high-card col x large running
    class count) so the lookup buffer never explodes.

    Ids are assigned FIRST-SEEN, not sorted - semantically equivalent for every
    consumer: the joint feeds only plug-in entropy (count-based, label-
    permutation-invariant) and further renumbering, and the next
    ``joint + c*mult`` step is a bijection regardless of the 0..k-1 permutation.
    nclasses + the induced partition are identical to the numpy form (verified).
    """
    n = joint.size
    if n == 0:
        return joint, 0
    jmax = 0
    for i in range(n):
        v = joint[i]
        if v > jmax:
            jmax = v
    inv = np.empty(n, dtype=np.int64)
    nc = 0
    if 0 <= jmax < _FAC_ARRAY_CAP:
        # Direct-array counting path (fast common case).
        seen = np.full(jmax + 1, -1, dtype=np.int64)
        for i in range(n):
            v = joint[i]
            s = seen[v]
            if s >= 0:
                inv[i] = s
            else:
                seen[v] = nc
                inv[i] = nc
                nc += 1
    else:
        # Hash fallback for cartesian blow-up (or pathological negative ids).
        d = _NbDict.empty(key_type=_nb_types.int64, value_type=_nb_types.int64)
        for i in range(n):
            v = joint[i]
            s = d.get(v, -1)
            if s >= 0:
                inv[i] = s
            else:
                d[v] = nc
                inv[i] = nc
                nc += 1
    return inv, nc
