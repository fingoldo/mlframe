"""Helpers carved out of ``_mi_greedy_cmi_fe`` to keep that module under its size budget."""
from __future__ import annotations

import functools
import logging
import math
import threading
from typing import Optional

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

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# GPU quantile-bin crossover (2026-06-28, synchronized micro-bench of _quantile_bin_gpu incl. code D2H, GTX
# 1050 Ti, nbins=10): a single host column round-tripped to the device for equi-frequency binning (H2D +
# cp.percentile sort + cp.searchsorted + code D2H) only beats the host introselect-partition np.quantile path
# well above the launch/transfer floor - n=20k CPU 0.92ms vs GPU 1.67ms; n=35k near-tie 1.53 vs 1.71ms; n=100k
# GPU 2.10 vs CPU 4.12ms = 2x; n=300k GPU 3.06 vs CPU 14.1ms = 4.6x. The gate is set at 50k (clear of the 35k
# near-tie) so every routed call is a decisive win; the small (3k/20k) gate-redundancy columns stay on the
# host, where the fixed ~1.7ms device round-trip overhead loses to numpy.
_GPU_QBIN_MIN_ROWS = 50_000


_FAC_PAR_MEMSET_MIN = 500_000


from ._mi_greedy_cmi_fe_binning import (  # noqa: F401  -- carved helpers
    _qbin_float_dtype,
    _sync_free_qbin_codes,
    _quantile_bin_gpu,
    _quantile_bin_gpu_resident,
    _quantile_bin,
    _FAC_ARRAY_CAP,
    _factorize_dense_njit,
)


@njit(cache=True, parallel=True)
def _combine_factorize_njit(joint: np.ndarray, c: np.ndarray, mult: int) -> tuple:
    """Fused ``factorize(joint + c*mult)`` in ONE pass, no temporaries.

    Equivalent to ``_factorize_dense_njit(joint + c*mult)`` but folds the
    multiply-add into the factorize walk - avoids the two numpy temp arrays
    (``c*mult`` and the sum) the `_renumber_joint` per-column step allocated, and
    walks the data once instead of three times. First-seen dense ids, so the
    induced partition + nclasses match the numpy form exactly (bit-identical).

    The first-seen WALK stays serial (data-dependent on the running ``seen``/``nc`` state - see the
    iter16 GPU bench-note below). The only parallel part is the large dense ``seen`` initialisation, prange-
    filled when ``kmax+1 >= _FAC_PAR_MEMSET_MIN`` (gate keeps small buffers serial -> bit-identical, no spin-
    up tax). Result is bit-identical to :func:`_combine_factorize_serial_njit` for every input (parity-tested).

    bench-note (iter16, 2026-06-23, resident-GPU /loop): NOT routed to GPU. The dense renumber is a
    first-seen sequential scan - each output id depends on the running ``seen`` table + ``nc`` counter, a
    data-dependent sequential dependency with no parallel form that preserves the FIRST-SEEN id assignment
    ORDER. A GPU sort+unique+searchsorted twin would assign ids in VALUE order, not first-seen order, changing
    the dense codes (the partition is equivalent but the integer labels differ) -> downstream joint-MI bin
    indices shift, breaking bit-identity. cProfile ~0.87s is single-pass njit already; the resident win this
    iter went to the maxT permutation-null floor instead (see _permutation_null_resident.py).

    bench-attempt-rejected (2026-06-29): full ``parallel=True`` over the factorize walk is impossible (first-
    seen race); parallel max-scan reduction via ``prange`` + ``if v>kmax`` hung/mis-compiled (numba does not
    recognise conditional-max as a parallel reduction) so the cheap ~1ms max scan stays serial."""
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
        span = kmax + 1
        seen = np.empty(span, dtype=np.int64)
        if span >= _FAC_PAR_MEMSET_MIN:
            for i in prange(span):  # parallel fill of the large dense lookup buffer
                seen[i] = -1
        else:
            for i in range(span):
                seen[i] = -1
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


@njit(cache=True)
def _renumber_two_dense_njit(a: np.ndarray, b: np.ndarray) -> tuple:
    """Densify the joint of TWO non-negative int class arrays in ONE pass, skipping the
    separate ``factorize(a)`` pass the generic per-column path runs first.

    When ``(max_a+1)*(max_b+1)`` keeps a flat ``seen`` buffer under the array cap, index it by
    ``a[i]*(max_b+1)+b[i]`` - the same fast array-counting trick as ``_factorize_dense_njit``,
    applied directly to the pair so the pair is densified in a single data walk (~1.7-2.5x over
    factorize-then-combine at the FE call volume). First-seen dense ids: the induced partition +
    nclasses are identical to the two-step path (verified), so every consumer (plug-in entropy,
    further renumbering) is bit-identical. Returns ``(inv, nc)``; ``nc == -1`` signals the caller
    to fall back to the generic path (negative ids or a cartesian span over the cap)."""
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    amax = a[0]; amin = a[0]; bmax = b[0]; bmin = b[0]
    for i in range(1, n):
        av = a[i]; bv = b[i]
        if av > amax: amax = av
        if av < amin: amin = av
        if bv > bmax: bmax = bv
        if bv < bmin: bmin = bv
    if amin < 0 or bmin < 0:
        return np.empty(0, dtype=np.int64), -1
    stride = bmax + 1
    span = (amax + 1) * stride
    if span < 0 or span >= _FAC_ARRAY_CAP:
        return np.empty(0, dtype=np.int64), -1
    seen = np.full(span, -1, dtype=np.int64)
    inv = np.empty(n, dtype=np.int64)
    nc = 0
    for i in range(n):
        k = a[i] * stride + b[i]
        s = seen[k]
        if s >= 0:
            inv[i] = s
        else:
            seen[k] = nc
            inv[i] = nc
            nc += 1
    return inv, nc


def _renumber_joint(*cols: np.ndarray) -> tuple[np.ndarray, int]:
    """Collapse multiple integer class arrays into a single dense class id.

    Returns ``(joint_classes, nclasses)``. Empty bins are pruned so the
    resulting ids are densely numbered 0..nclasses-1 - this is what makes
    multivariate Z trackable: even with d=8 support cols * 10 bins each
    (10**8 cartesian space) the actual occupied bins are <= n_samples, so
    we never allocate the cartesian space.

    Per-fold renumbering uses the njit hash-factorize (first-seen dense ids)
    instead of ``np.unique`` - see :func:`_factorize_dense_njit`.

    # bench-attempt-rejected (2026-07-18): per-call GPU dispatch of THIS host entry point (upload cols,
    # call :func:`_renumber_joint_gpu`, D2H the scalar cardinality) was A/B'd against the njit path at the
    # actual call shapes (n=100k, k=2..8 conditioning columns - see the wellbore-100k cProfile that flagged
    # 51.9s/6766 calls here). ``bench_renumber_joint_gpu_dispatch.py``: host 0.28-1.42ms (k=2..4) vs GPU
    # 3.4-5.8ms even with columns ALREADY resident (no H2D) - GPU is 4-16x SLOWER at the k<=4 shapes that
    # dominate the call sites (pairwise xy/xz/yz/xyz joins, one-sibling-at-a-time budget loops with an
    # early-exit that makes cross-candidate batching change the amount of work done, not just its shape).
    # GPU only wins (1.4x) at k=8+ columns, wider than any call site actually builds. The already-shipped
    # ``_renumber_joint_gpu`` remains correctly used wherever callers already hold device-resident operands
    # (``_fe_raw_redundancy_helpers.py``, ``_fe_cmi_redundancy_gate.py``); this host path is the genuine
    # residual for fresh-host-array / sequential-early-exit call sites, not a missed dispatch.
    """
    if not cols:
        # No conditioning -> caller handles the marginal-MI case explicitly.
        return np.zeros(0, dtype=np.int64), 1
    n = cols[0].size
    # Two-column joints (the marginal xy + the conditional xz / yz) dominate the call volume; densify
    # the pair in ONE pass via the array-counting fast path, skipping the separate factorize(col0) pass.
    # ``nc < 0`` signals an unsupported case (negative ids / cartesian span over the cap) -> generic path.
    if len(cols) == 2 and n:
        a = np.ascontiguousarray(cols[0], dtype=np.int64).ravel()
        b = np.ascontiguousarray(cols[1], dtype=np.int64).ravel()
        inv, nc = _renumber_two_dense_njit(a, b)
        if nc >= 0:
            return inv, int(nc)
    # First column: with ``joint`` all-zeros and ``mult`` == 1 the original
    # ``joint + c64 * mult`` reduced to ``c64``, so seed directly from col 0 and
    # skip both the ``np.zeros(n)`` allocation and the redundant add (2.9x on the
    # common single-col conditioning case; bit-identical).
    # Conditioning cols are 1-D class arrays; a stray singleton 2nd dim ((n, 1) from an upstream reshape) would make
    # the njit factorize see a 2-D array -> numba "Cannot unify Literal[int](0) and array(int64)" at compile. ravel()
    # normalises (no-op for 1-D, squeezes (n, 1)); a genuine (n, k>1) col surfaces downstream as a shape error.
    joint = np.ascontiguousarray(cols[0], dtype=np.int64).ravel()
    if n:
        joint, mult = _factorize_dense_njit(joint)
    else:
        mult = 1
    for c in cols[1:]:
        c64 = np.ascontiguousarray(c, dtype=np.int64).ravel()
        # Fused multiply-add + refactorize: one njit walk, no ``c64*mult`` /
        # sum temp arrays. Renumber after every fold so ``mult`` stays bounded by
        # the actual occupied joint cardinality (~ <= n) instead of the cartesian
        # product (which would blow up at d=4+ support cols * 10 bins).
        if n:
            joint, mult = _combine_factorize_njit(joint, c64, mult)
        else:
            joint = joint + c64 * mult
    return joint, int(mult)


def _dense_renumber_device(cp, keys):
    """Value-order dense renumber of a resident int64 key array WITHOUT a device->host sync.

    Bit-identical to ``cp.unique(keys, return_inverse=True)`` (the dense ids follow sorted-value order), but
    cp.unique syncs to size its unique-value output, whereas this keeps the cardinality as a DEVICE 0-dim scalar:
    argsort the keys, mark run boundaries (a value is new iff it differs from its sorted predecessor), cumsum the
    boundary flags to dense ids in sorted order, then scatter those back to the original positions. Returns
    ``(inv, k)`` - ``inv`` the resident densified codes, ``k`` the occupied cardinality as a device 0-dim int
    (so the caller can broadcast it into ``joint + c*k`` with no host read; read it once at the very end)."""
    if keys.size == 0:
        return keys.astype(cp.int64, copy=False), cp.asarray(1, dtype=cp.int64)
    order = cp.argsort(keys, kind="stable")
    ks = keys[order]
    is_new = cp.empty(ks.shape, dtype=cp.bool_)
    is_new[0] = True
    is_new[1:] = ks[1:] != ks[:-1]
    dense_sorted = cp.cumsum(is_new.astype(cp.int64)) - 1  # dense ids in sorted-value order
    inv = cp.empty(keys.shape, dtype=cp.int64)
    inv[order] = dense_sorted
    return inv, dense_sorted[-1] + 1  # k stays a device 0-dim scalar


def _renumber_joint_gpu(*cols):
    """Device twin of :func:`_renumber_joint`: collapse ALREADY-RESIDENT integer class arrays into a single
    dense class id ON the device, returning ``(joint_dev, nclasses)`` - ``joint_dev`` a resident int64 cupy
    array densely numbered 0..nclasses-1.

    Used by the CMI-redundancy gate to build the round conditioning support Z as a device-born join of the
    resident candidate codes, so Z never crosses H2D (the ``cmi_z`` upload). The dense-id ASSIGNMENT differs
    from the host njit first-seen numbering (``cupy.unique`` numbers in sorted-value order), but the PARTITION
    - which rows share a class - is IDENTICAL to :func:`_renumber_joint`, and every downstream consumer (the
    entropy / joint-count histograms in ``batched_cmi_gpu`` / ``_cmi_from_binned_cupy`` / the perm-null) depends
    ONLY on the partition, so the CMI - and the gate's admit/reject decision - is selection-identical. Each
    fold refactorises so ``mult`` stays bounded by the occupied cardinality (<= n), so the combine key
    ``joint + c*mult`` (c <= nbins-1) cannot overflow int64. Any cupy fault raises to the caller (host fallback)."""
    import cupy as cp

    if not cols:
        return cp.zeros(0, dtype=cp.int64), 1
    joint = cp.ascontiguousarray(cols[0].astype(cp.int64, copy=False).ravel())
    # Device-driven dense renumber: replaces cp.unique(return_inverse), which SYNCS to size its
    # output, with a manual argsort -> boundary-mark -> cumsum renumber that is bit-identical (value-order dense
    # ids == cp.unique inverse) but keeps the fold multiplier ``mult`` as a DEVICE 0-dim scalar. ``joint + c*mult``
    # broadcasts the device scalar with no host read, so the whole multi-column densify runs sync-free; only the
    # final cardinality crosses the bus once (int(mult) in the return).
    joint, mult = _dense_renumber_device(cp, joint)
    for c in cols[1:]:
        c64 = c.astype(cp.int64, copy=False).ravel()
        # joint in [0, mult) -> ``joint + c*mult`` is a unique key per (joint, c) pair; refactorise to dense so
        # mult tracks the occupied joint cardinality (not the cartesian product) fold-to-fold.
        joint, mult = _dense_renumber_device(cp, joint + c64 * mult)
    return joint, int(mult)


@njit(cache=True)
def _entropy_from_classes_njit(classes: np.ndarray) -> tuple:
    """Single-pass plug-in entropy + occupied-cell count for a dense integer
    class array. Fuses the numpy ``bincount -> mask-copy -> p-array ->
    log-array -> sum`` chain into one allocation-light C loop (2.54x over the
    numpy form at the CMI-greedy call volume; bit-identical to 1e-9). ``classes``
    MUST be non-negative dense ids (``_renumber_joint`` guarantees 0..k-1)."""
    n = classes.size
    if n == 0:
        return 0.0, 0
    cmax = 0
    for i in range(n):
        v = classes[i]
        if v > cmax:
            cmax = v
    counts = np.zeros(cmax + 1, dtype=np.int64)
    for i in range(n):
        counts[classes[i]] += 1
    H = 0.0
    k = 0
    inv_n = 1.0 / n
    for c in counts:
        if c > 0:
            p = c * inv_n
            H -= p * math.log(p)
            k += 1
    return H, k


@njit(cache=True)
def _joint_entropy_two_dense_njit(a: np.ndarray, b: np.ndarray) -> tuple:
    """Fused densify+entropy of the JOINT of two non-negative int class arrays in ONE walk.

    The CMI callers renumber a joint (``xz`` / ``xyz`` / ``xy``) ONLY to hand the dense-id array to
    :func:`_entropy_from_classes`, then discard the labels - the classic "caller uses only part of the
    kernel output" waste (see CLAUDE.md). This fuses the two: index a flat ``seen`` buffer by
    ``a[i]*stride+b[i]`` (the ``_renumber_two_dense_njit`` array-counting trick) but, instead of writing a
    length-n relabel array + a second ``bincount`` pass, it accumulates the per-class COUNT inline and
    reduces the plug-in entropy over the occupied cells at the end - O(n + k), no length-n ``inv`` array,
    no second data pass, no separate entropy call.

    Returns ``(H, k)`` - plug-in entropy (natural log) + occupied-cell count, bit-identical to
    ``_entropy_from_classes(_renumber_joint(a, b)[0])`` (both are functions of the joint COUNT multiset,
    which is label-permutation-invariant; only the fp summation ORDER can differ ~1e-15). ``k == -1``
    signals an unsupported case (negative ids or a cartesian span over the cap) -> caller falls back to the
    generic renumber+entropy path."""
    n = a.shape[0]
    if n == 0:
        return 0.0, 0
    amax = a[0]; amin = a[0]; bmax = b[0]; bmin = b[0]
    for i in range(1, n):
        av = a[i]; bv = b[i]
        if av > amax: amax = av
        if av < amin: amin = av
        if bv > bmax: bmax = bv
        if bv < bmin: bmin = bv
    if amin < 0 or bmin < 0:
        return 0.0, -1
    stride = bmax + 1
    span = (amax + 1) * stride
    if span < 0 or span >= _FAC_ARRAY_CAP:
        return 0.0, -1
    seen = np.full(span, -1, dtype=np.int64)
    counts = np.empty(n, dtype=np.int64)  # count per dense id (<= n distinct); no length-n label array
    nc = 0
    for i in range(n):
        k = a[i] * stride + b[i]
        s = seen[k]
        if s >= 0:
            counts[s] += 1
        else:
            seen[k] = nc
            counts[nc] = 1
            nc += 1
    H = 0.0
    inv_n = 1.0 / n
    for j in range(nc):
        c = counts[j]
        p = c * inv_n
        H -= p * math.log(p)
    return H, nc


def _joint_entropy_two(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Plug-in entropy ``(H, k)`` of the joint of two int class arrays, fusing the densify+entropy so the
    dense-id labels are never materialised (see :func:`_joint_entropy_two_dense_njit`). Bit-identical to
    ``_entropy_from_classes(_renumber_joint(a, b)[0])`` (to ~1e-9 fp reduction order). Falls back to the
    generic renumber+entropy on the unsupported case (negative ids / cartesian span over the cap)."""
    a_i = np.ascontiguousarray(a, dtype=np.int64).ravel()
    b_i = np.ascontiguousarray(b, dtype=np.int64).ravel()
    if a_i.size == 0:
        return 0.0, 0
    H, k = _joint_entropy_two_dense_njit(a_i, b_i)
    if k >= 0:
        return float(H), int(k)
    joint, _ = _renumber_joint(a_i, b_i)
    return _entropy_from_classes(joint)


def _entropy_from_classes(classes: np.ndarray) -> tuple[float, int]:
    """``H = -sum p_i log p_i`` from an integer class array (natural log).

    Returns ``(H_plugin, n_nonempty_cells)``. The cell count is used by
    Miller-Madow bias correction in :func:`_cmi_from_binned` - plug-in
    MLE entropy has positive bias O((K-1)/(2n)); subtracting the same
    quantity from CMI cancels at first order.

    Delegates to the njit kernel after ensuring a contiguous int64 array (the
    kernel indexes a ``counts`` buffer by class id, so non-negative dense ids
    are required - guaranteed by the binned / ``_renumber_joint`` callers).
    """
    if classes.size == 0:
        return 0.0, 0
    classes = np.ascontiguousarray(classes, dtype=np.int64)
    H, k = _entropy_from_classes_njit(classes)
    return float(H), int(k)


def precompute_marginal_y_terms(y_codes: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Hoist the y-only terms of the marginal ``_cmi_from_binned(x, y, None)`` out of a
    loop that scores many candidate ``x`` against ONE fixed ``y``.

    The marginal-MI path computes ``MI(X;Y) = H(X) + H(Y) - H(X,Y)``; ``H(Y)`` and its
    occupied-cell count ``k_y`` are invariant across candidates, yet the plain helper
    re-binned/re-entropied ``y`` on every call (the usability candidate-pool enumeration
    evaluates ``|unary|^2 * |binary|`` forms per pair against the same ``y_codes``).

    Returns ``(y_i, h_y, k_y)`` where ``y_i`` is the contiguous int64 view reused by
    every :func:`marginal_mi_binned_fixed_y` call. Bit-identical to the inline path.
    """
    y_i = np.ascontiguousarray(y_codes, dtype=np.int64)
    h_y, k_y = _entropy_from_classes(y_i)
    return y_i, h_y, k_y


def marginal_mi_binned_fixed_y(
    x_binned: np.ndarray, y_i: np.ndarray, h_y: float, k_y: int,
) -> float:
    """Marginal binned MI ``MI(X;Y)`` reusing precomputed y terms from
    :func:`precompute_marginal_y_terms`. Bit-identical to ``_cmi_from_binned(x_binned,
    y_i, None)`` (same plug-in entropies, same Miller-Madow bias), minus the per-call
    ``H(Y)`` recompute + the ``y`` int64 cast."""
    x_i = np.ascontiguousarray(x_binned, dtype=np.int64)
    n = float(max(1, x_i.size))
    h_x, k_x = _entropy_from_classes(x_i)
    h_xy, k_xy = _joint_entropy_two(x_i, y_i)  # fused densify+entropy; xy labels discarded
    mi_plugin = h_x + h_y - h_xy
    mi_bias = (k_x + k_y - k_xy - 1) / (2.0 * n)
    return max(0.0, mi_plugin - mi_bias)


def precompute_cmi_yz_terms(
    y: np.ndarray, z_joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int, float]:
    """Hoist the y/z-only terms of the conditional ``_cmi_from_binned`` out of a
    permutation loop that only resamples ``x``.

    Within a conditional-permutation null only the candidate ``x`` is reshuffled
    (within support strata); ``y`` and ``z`` are fixed across all permutations,
    so ``H(Y,Z)``, ``H(Z)`` and their occupied-cell counts ``k_yz`` / ``k_z`` are
    invariant. Recomputing them per permutation (the plain ``_cmi_from_binned``
    path) re-renumbers ``yz`` and re-bins ``z`` every iteration and discards the
    result - pure wasted work. This returns the invariant block once; pair with
    :func:`cmi_from_binned_fixed_yz` for the per-permutation evaluation.

    Returns ``(y_i, z_i, h_yz, h_z, k_yz, k_z, n)`` where ``y_i`` / ``z_i`` are
    contiguous int64 views reused by every permutation.
    """
    y_i = np.ascontiguousarray(y, dtype=np.int64).ravel()
    z_i = np.ascontiguousarray(z_joint, dtype=np.int64).ravel()
    n = float(max(1, y_i.size))
    # H(Y,Z): the (y,z) dense labels are consumed ONLY by this entropy here, so fuse the densify+entropy
    # (``_joint_entropy_two``) instead of materialising the length-n relabel via ``_renumber_joint`` and
    # discarding it. Bit-identical (both are functions of the (y,z) count multiset; ~1e-15 fp order only).
    h_z, k_z = _entropy_from_classes(z_i)
    h_yz, k_yz = _joint_entropy_two(y_i, z_i)
    return y_i, z_i, h_yz, h_z, k_yz, k_z, n


def cmi_from_binned_fixed_yz(
    x: np.ndarray,
    y_i: np.ndarray,
    z_i: np.ndarray,
    h_yz: float,
    h_z: float,
    k_yz: int,
    k_z: int,
    n: float,
    yz_i: Optional[np.ndarray] = None,
) -> float:
    """``CMI(X; Y | Z)`` for a fresh ``x`` reusing the y/z-invariant terms from
    :func:`precompute_cmi_yz_terms`. Computes only the x-dependent ``xz`` / ``xyz``
    renumberings + their entropies; bit-identical to :func:`_cmi_from_binned` on
    the same inputs (it is the same arithmetic with the y/z block factored out).

    ``yz_i`` (optional): the round-fixed DENSE ``(y,z)`` joint codes from
    :func:`precompute_cmi_yz_terms` (its ``yz_dense`` return). When supplied, the
    x-dependent ``H(X,Y,Z)`` is taken as the joint entropy of ``(x, yz_i)`` - a
    TWO-array densify (partition of ``x`` with the fixed ``(y,z)`` partition ==
    partition of ``(x,y,z)``, identical counts -> identical entropy) instead of the
    3-column ``renumber(x,y,z)`` factorize (1 factorize + 2 combine walks -> 1 fused
    walk). ``None`` (external callers) keeps the exact 3-column path."""
    # GPU route: the xz / xyz joint ENTROPIES are partition statistics - cp.unique(flat_key,
    # return_counts) densifies on the device (sort+unique, value-order labels) and the counts give the SAME
    # partition -> the SAME entropy -> the SAME CMI (only fp reduction order differs ~1e-15; selection
    # identical). Routes the dominant mi_greedy CMI compute (combine_factorize + entropy) onto the GPU under
    # MLFRAME_FE_GPU_STRICT / the KTC gate, instead of the host njit renumber+entropy. Falls back to CPU on
    # any cupy error.
    #
    # ``x`` may already be device-resident (see _cmi_from_binned's identical note) - the shape gate cannot
    # see that, so an already-device x forces the cupy path regardless, and a cupy failure there pulls x/y_i/
    # z_i to host explicitly before falling through (np.ascontiguousarray cannot accept a cupy array).
    _x_device = hasattr(x, "__cuda_array_interface__")
    if _x_device or _cmi_gpu_enabled(n=int(np.asarray(x).size), p=1):
        try:
            return _cmi_from_binned_fixed_yz_cupy(x, y_i, z_i, h_yz, h_z, k_yz, k_z, n)
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
                if hasattr(y_i, "__cuda_array_interface__"):
                    y_i = cp.asnumpy(y_i)
                if hasattr(z_i, "__cuda_array_interface__"):
                    z_i = cp.asnumpy(z_i)
    x_i = np.ascontiguousarray(x, dtype=np.int64).ravel()
    # Fused densify+entropy: xz / xyz labels are consumed ONLY by the entropy, so build the joint histogram
    # inline and skip the length-n relabel array + the second bincount pass (see _joint_entropy_two).
    h_xz, k_xz = _joint_entropy_two(x_i, z_i)
    if yz_i is not None:
        # 2-array densify against the round-fixed (y,z) partition == partition(x,y,z) -> identical counts.
        h_xyz, k_xyz = _joint_entropy_two(x_i, yz_i)
    else:
        xyz, _ = _renumber_joint(x_i, y_i, z_i)
        h_xyz, k_xyz = _entropy_from_classes(xyz)
    cmi_plugin = h_xz + h_yz - h_z - h_xyz
    cmi_bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
    return max(0.0, cmi_plugin - cmi_bias)


def _cmi_gpu_enabled(*, n: Optional[int] = None, p: Optional[int] = None, min_p: Optional[int] = None) -> bool:
    """Route the mi_greedy CMI entropies to the GPU when STRICT-GPU is on (or a future KTC gate). Default
    OFF -> host path, byte-identical. STRICT_GPU=1 forces it (the user's "make GPU actually carry the FE
    compute" knob: most FE families were CPU-only, this puts the dominant CMI on the device).

    ``n``/``p`` (optional): the calling dispatch's own shape, forwarded to ``fe_gpu_strict_enabled`` so the
    STRICT/AUTO decision is size-aware for THIS call instead of shape-blind. Omit when no natural shape is
    available at the call site (preserves prior behavior)."""
    import os as _os
    if _os.environ.get("MLFRAME_CMI_GPU", "") == "1":
        return True
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        return bool(fe_gpu_strict_enabled(n=n, p=p, min_p=min_p))
    except Exception as e:
        logger.debug("fe_gpu_strict_enabled probe failed (%s: %s) -- defaulting to CPU host path", type(e).__name__, e)
        return False


_CARD_MAX_CACHE: dict = {}


_CARD_MAX_CACHE_LOCK = threading.Lock()


def _cached_card(host_arr, dev_codes) -> int:
    """Return ``max(dev_codes)+1``, memoized by a content-hash of ``host_arr`` (never the operand's ``id()`` - an id can be reused after GC and silently return a too-small cardinality, corrupting the device joint-histogram kernels' shared-tile sizing)."""
    if dev_codes.size == 0:
        return 1
    ha = np.ascontiguousarray(np.asarray(host_arr).ravel())
    key = (int(ha.size), ha.dtype.str, hash(ha.tobytes()))  # content fingerprint (no id-reuse collision)
    with _CARD_MAX_CACHE_LOCK:
        v = _CARD_MAX_CACHE.get(key)
        if v is None:
            v = int(dev_codes.max()) + 1
            if len(_CARD_MAX_CACHE) > 64:
                _CARD_MAX_CACHE.clear()
            _CARD_MAX_CACHE[key] = v
    return v


def _cmi_from_binned_fixed_yz_cupy(x, y_i, z_i, h_yz, h_z, k_yz, k_z, n) -> float:
    """Device twin of :func:`cmi_from_binned_fixed_yz`: the xz / xyz joint entropies via cp.unique counts.
    Value-order densification -> same partition -> same CMI (selection-identical, fp-order ~1e-15)."""
    import cupy as cp

    from ._fe_batched_mi import joint_entropy_gpu

    # x is the per-permutation candidate (transient) -> NOT cached. y_i / z_i are fit/round-constants reused
    # across every permutation in this CMI-null loop (H2D instrumentation: 100x / 800 MB each at 1M) ->
    # resident operand cache (uploaded once per fit; CMI is value-order invariant).
    from ._fe_resident_operands import resident_operand, resident_code_operand
    dx = cp.asarray(np.ascontiguousarray(x, dtype=np.int64).ravel())
    dy = resident_code_operand(y_i, "fixedyz_y")
    dz = resident_operand(z_i, "fixedyz_z", dtype=np.int64)
    inv_n = 1.0 / float(n)

    _entc = functools.partial(joint_entropy_gpu, inv_n=inv_n)

    Kx = (int(dx.max()) + 1) if dx.size else 1
    ky = _cached_card(y_i, dy)  # y is a fit-constant -> cardinality cached
    kz = int(k_z) if int(k_z) > 0 else (int(dz.max()) + 1 if dz.size else 1)
    h_xz, k_xz = _entc([dx, dz], [Kx, kz])
    h_xyz, k_xyz = _entc([dx, dy, dz], [Kx, ky, kz])
    cmi_plugin = h_xz + float(h_yz) - float(h_z) - h_xyz
    cmi_bias = (k_xyz + int(k_z) - k_xz - int(k_yz)) / (2.0 * float(n))
    return float(max(0.0, cmi_plugin - cmi_bias))


def _greedy_cmi_fe_constr_step1_content_hash_already(cand_names, engineered, nbins, st, _bin_fingerprint, seed, cand_bins_dev):
    """Step 1 of greedy_cmi_fe_construct: lines starting at ``try:``."""
    try:
        from mlframe.feature_selection.filters._gpu_strict_fe import fe_gpu_strict_resident_enabled
        if fe_gpu_strict_resident_enabled():
            import cupy as _cp

            for _name in cand_names:
                _codes_dev = _quantile_bin_gpu_resident(engineered[_name].to_numpy(dtype=np.float64), nbins)
                if _codes_dev is None:
                    cand_bins_dev = {}
                    break
                cand_bins_dev[_name] = _codes_dev
            if cand_bins_dev:
                # cupy (this version) has no ``bitwise_xor.reduce`` - a weighted-sum reduction is an
                # equally-cheap, equally-accepted (see module docstring) content hash: same n for every
                # candidate this fit, so the odd-prime weight vector is built once and reused.
                _fp_n = next(iter(cand_bins_dev.values())).size
                _weights = _cp.arange(1, _fp_n + 1, dtype=_cp.int64) * _cp.int64(2654435761)
                for _name in cand_names:
                    _c = cand_bins_dev[_name].astype(_cp.int64, copy=False)
                    st.cand_fp[_name] = int(_cp.sum((_c + 1) * _weights, dtype=_cp.int64).item())
                st._resident_fp_ok = True
    except Exception:
        logger.debug("resident candidate binning failed; host fingerprint fallback", exc_info=True)
        cand_bins_dev = {}
        st.cand_fp = {}
        st._resident_fp_ok = False
    if not st._resident_fp_ok:
        cand_bins = {name: _quantile_bin(engineered[name].to_numpy(), nbins=nbins) for name in cand_names}
        st.cand_fp = {name: _bin_fingerprint(cand_bins[name]) for name in cand_names}

    def _host_bins(name: str) -> np.ndarray:
        """Host projection of a candidate's binned codes, materialized LAZILY (only when the host-only Z-fold
        step below needs it) instead of eagerly for the whole candidate pool - the one remaining, deliberate
        per-winner D2H (bounded by ``top_k``, not by candidate count)."""
        b = cand_bins.get(name)
        if b is None:
            if cand_bins_dev:
                import cupy as _cp

                b = _cp.asnumpy(cand_bins_dev[name])
            else:
                b = _quantile_bin(engineered[name].to_numpy(), nbins=nbins)
            cand_bins[name] = b
        return b

    # Permutation-based noise-floor for the current Z: shuffle y once,
    # rebin, sample 24 candidates' CMI; take the 95th percentile as
    # the floor. Combined with the user's ``min_cmi_gain`` via max().
    # Avoids the "noise CMI ~ 0.01 with k=4 Z and small n still admits
    # spurious transforms" failure mode that bias correction alone
    # can't fully suppress at finite n. Recomputed when Z grows so the
    # floor scales with the conditioning's fragmentation.
    # Salt the caller's seed against the historical constant via SeedSequence rather than
    # handing small user-chosen integers (e.g. random_state=0) straight to the permutation
    # RNG: raw small seeds can land on unlucky permutation draws.
    rng_floor = np.random.default_rng(np.random.SeedSequence([0xC011, seed & 0xFFFFFFFF]))
    return _host_bins, cand_bins_dev, rng_floor


def _greedy_cmi_fe_constr_step2_rng_raw_small(cand_names, rng_floor, y_bin, z_joint_dev, z_joint, cand_bins_dev, _host_bins, st):
    """Step 2 of greedy_cmi_fe_construct: lines starting at ``def _noise_floor_for_current_z() -> float:``."""
    def _noise_floor_for_current_z() -> float:
        """Permutation-based CMI noise floor for the CURRENT conditioning Z: shuffles y once, samples up to 24 candidates' CMI against the shuffled target, and returns the 95th percentile as the floor a real candidate must clear (combined with the user's ``min_cmi_gain`` via ``max()``). Recomputed as Z grows so the floor tracks the conditioning's fragmentation."""
        if not cand_names:
            return 0.0
        idx = rng_floor.permutation(y_bin.size)
        y_shuf = y_bin[idx]
        sample_size = min(24, len(cand_names))
        sample_names = rng_floor.choice(
            np.array(cand_names, dtype=object), size=sample_size, replace=False,
        )
        # BATCHED (launch-reduction): y_shuf / z_joint are fixed across the sampled candidates -> score their
        # CMI in ONE batched_cmi_gpu workload instead of a per-candidate loop. The floor is the 0.95 quantile
        # (order-independent) -> selection-equivalent. Per-candidate loop fallback on any error / GPU-off.
        try:
            if _cmi_gpu_enabled(n=int(y_shuf.size), p=len(sample_names)) and len(sample_names) > 1:
                from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu

                _zc = z_joint_dev if z_joint_dev is not None else (z_joint if (z_joint is not None and z_joint.size > 0) else None)
                if cand_bins_dev:
                    # RESIDENT: assemble the sampled columns from the already-device-resident candidate codes
                    # (no per-round H2D of X) and keep the (K,) CMI vector resident too - only the FINAL scalar
                    # 0.95-quantile crosses back, not the bulk (K,) vector the non-resident branch below D2Hs.
                    import cupy as _cp

                    _Xs_dev = _cp.stack([cand_bins_dev[_nm] for _nm in sample_names], axis=1)
                    _mi_dev = batched_cmi_gpu(_Xs_dev, y_shuf, _zc, return_device=True)
                    return float(_cp.percentile(_mi_dev, 95.0).item())
                # bench-attempt-rejected (2026-07): np.column_stack([cand_bins[nm] ...]).astype(int64) vs this per-column loop was noise
                # across n{50k,200k,1M} x k{24,100} x dtype{i32,i64}: -9.8%..+8.8%, no consistent >=5% win (memory-bandwidth-bound copy either way).
                _Xs = np.empty((int(y_shuf.shape[0]), len(sample_names)), dtype=np.int64)
                for _j, _nm in enumerate(sample_names):
                    _Xs[:, _j] = _host_bins(_nm)
                _cmis = np.asarray(batched_cmi_gpu(_Xs, y_shuf, _zc), dtype=np.float64)
                return float(np.quantile(_cmis, 0.95))
        except Exception as e:
            # Same reasoning as the two kernel handlers above: the CPU path below keeps the answer correct, so
            # a real regression in the batched GPU permutation-null scan would otherwise never be noticed.
            log_throttle(
                logger,
                "cmi_gpu_perm_null_fallback",
                logging.WARNING,
                "batched GPU permutation-null scan failed (%s: %s); recomputing the null floor on the CPU path. Correctness is preserved, the GPU cost is not.",
                type(e).__name__,
                e,
            )
        # Hoist the y/z-invariant CMI block out of the sampled-candidate scan: y_shuf and z_joint are FIXED
        # across the 24 samples, so the plain per-sample ``_cmi_from_binned`` recomputed-and-discarded
        # ``renumber(y_shuf, z)`` + ``H(Z)`` + ``H(Y,Z)`` (or ``H(Y)`` on the marginal path) 24 times per step
        # - pure wasted _renumber_joint / _entropy_from_classes work. Precompute the invariant block once and
        # score each sample via the x-only fixed helper. Bit-identical to the plain path (same MM plug-in CMI;
        # only fp reduction order can differ ~1e-15), and the floor is the order-independent 0.95 quantile.
        if z_joint is not None and z_joint.size > 0:
            _zc = z_joint
        elif z_joint_dev is not None:
            import cupy as _cp

            _zc = _cp.asnumpy(z_joint_dev)
        else:
            _zc = None
        cmis_shuf: list[float]
        if _zc is None:
            _yt = precompute_marginal_y_terms(y_shuf)
            cmis_shuf = [marginal_mi_binned_fixed_y(_host_bins(nm), *_yt) for nm in sample_names]
        else:
            _yi, _zi, _hyz, _hz, _kyz, _kz, _nf = precompute_cmi_yz_terms(y_shuf, _zc)
            _yzd, _ = _renumber_joint(_yi, _zi)  # round-fixed (y,z) dense codes reused per sampled candidate
            cmis_shuf = [cmi_from_binned_fixed_yz(_host_bins(nm), _yi, _zi, _hyz, _hz, _kyz, _kz, _nf, yz_i=_yzd) for nm in sample_names]
        if not cmis_shuf:
            return 0.0
        return float(np.quantile(np.asarray(cmis_shuf), 0.95))

    # 5. Greedy CMI loop.
    st.winners = []
    st.winner_fps = set()
    st.rows = []
    st.remaining = set(cand_names)
    st.step = 0
    st.z_card_at_floor = -1
    st.cur_floor = 0.0
    return _noise_floor_for_current_z


def _greedy_cmi_fe_constr_step3_while_st_remaining(st, top_k, _noise_floor_for_current_z, min_cmi_gain, z_joint_dev, z_joint, y_bin, cand_bins_dev, _host_bins, nbins):
    """Step 3 of greedy_cmi_fe_construct: lines starting at ``while st.remaining and len(st.winners) < int(top_k):``."""
    while st.remaining and len(st.winners) < int(top_k):
        # Recompute the noise floor when Z cardinality changes (i.e. Z
        # was grown since last iter). Step 0 always recomputes.
        if st.z_card != st.z_card_at_floor:
            st.cur_floor = _noise_floor_for_current_z()
            st.z_card_at_floor = st.z_card
        effective_floor = max(float(min_cmi_gain), st.cur_floor)
        best_name = None
        best_cmi = -1.0
        _have_z_dev = z_joint_dev is not None
        _have_z_host = z_joint is not None and z_joint.size > 0
        _have_z = False
        if _have_z_dev:
            _have_z = True
        elif _have_z_host:
            _have_z = True
        # The fingerprint-skip set is fixed for this step -> the candidates to score are
        # ``_scan`` (a list over ``remaining`` preserving its iteration order). y_bin / z_joint are fixed
        # across them, so score ALL their CMI in ONE batched_cmi_gpu workload and take the first-max argmax
        # (== the sequential ``cmi > best_cmi`` tie-break). Same MM plug-in CMI -> selection-equivalent;
        # per-candidate loop fallback on any error / GPU-off.
        _scan = [name for name in st.remaining if st.cand_fp[name] not in st.winner_fps]
        _batched_done = False
        try:
            if _cmi_gpu_enabled(n=int(y_bin.shape[0]), p=len(_scan)) and len(_scan) > 1:
                from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu, cmi_device_argmax

                if cand_bins_dev:
                    # RESIDENT: the candidate codes are already device-resident (binned once up front via
                    # ``_quantile_bin_gpu_resident``) - stack them directly instead of rebuilding + re-uploading
                    # a host (n, K) matrix every round.
                    import cupy as _cp

                    _Xc = _cp.stack([cand_bins_dev[_nm] for _nm in _scan], axis=1)
                else:
                    # bench-attempt-rejected (2026-07): np.column_stack(...).astype(int64) vs this per-column loop was noise (-9.8%..+8.8%, no consistent >=5%).
                    _Xc = np.empty((int(y_bin.shape[0]), len(_scan)), dtype=np.int64)
                    for _j, _nm in enumerate(_scan):
                        _Xc[:, _j] = _host_bins(_nm)
                # RESIDENT Z: prefer the resident conditioning support - folded fully on-device by
                # ``_renumber_joint_gpu`` at the winner-fold step below - over the host mirror. batched_cmi_gpu
                # accepts a resident z as-is (no H2D), closing the round-constant ``cmi_z`` re-upload this loop
                # otherwise pays every round Z changes.
                _zc = z_joint_dev if z_joint_dev is not None else (z_joint if _have_z else None)
                # RESIDENCY: the greedy loop only needs the argmax of the (K,) CMI vector (rest is discarded),
                # and y_bin is fit-constant / z_joint round-constant. Pass the RESIDENT y_bin_dev (uploaded once)
                # so y never re-crosses H2D; return_device keeps the (K,) vector on the device (no per-round bulk
                # D2H); cmi_device_argmax pulls only the winning (idx, val) scalars. first-max == the sequential
                # ``cmi > best_cmi`` tie-break. Falls back to host y_bin if the one-time device upload failed.
                _yarg = st.y_bin_dev if st.y_bin_dev is not None else y_bin
                # kx=nbins: candidate codes are equi-frequency bins in [0, nbins-1] -> width is known, skip int(max) sync.
                _mi_d = batched_cmi_gpu(_Xc, _yarg, _zc, codes_trusted=True, return_device=True, kx=int(nbins))
                _bi, _bv = cmi_device_argmax(_mi_d)
                best_cmi = float(_bv); best_name = _scan[_bi]
                _batched_done = True
        except Exception as e:
            logger.debug("batched CMI argmax computation failed, falling back to the per-candidate path: %s", e)
            _batched_done = False
        if _batched_done:
            pass
        else:
            # FALLBACK ONLY (GPU-off / batched call faulted): the y/z-invariant CMI block
            # (H(Y,Z)/H(Z) + their renumberings) is only needed by this per-candidate loop, not the batched
            # path above - computed here, lazily, instead of eagerly every round (2026-07-17: it used to run
            # unconditionally even when the batched path succeeded and its output went unused, and it forced
            # ``z_joint`` to be a HOST array every round). ``_host_z()`` materializes the host mirror of
            # ``z_joint_dev`` on first use in this fallback only.
            _yz_dense = None
            _y_marg = None
            if not _have_z:
                _y_marg = precompute_marginal_y_terms(y_bin)
            else:
                if z_joint is not None:
                    _z_host = z_joint
                else:
                    import cupy as _cp

                    _z_host = _cp.asnumpy(z_joint_dev)
                _y_i, _z_i, _h_yz, _h_z, _k_yz, _k_z, _n = precompute_cmi_yz_terms(y_bin, _z_host)
                _yz_dense, _ = _renumber_joint(_y_i, _z_i)
            for name in _scan:
                if _have_z:
                    cmi = cmi_from_binned_fixed_yz(_host_bins(name), _y_i, _z_i, _h_yz, _h_z, _k_yz, _k_z, _n, yz_i=_yz_dense)
                else:
                    assert _y_marg is not None  # not _have_z guarantees this was precomputed above
                    cmi = marginal_mi_binned_fixed_y(_host_bins(name), *_y_marg)
                if cmi > best_cmi:
                    best_cmi = cmi
                    best_name = name
        if best_name is None:
            break
        if best_cmi < effective_floor:
            # No remaining candidate adds enough new info; stop.
            break
        st.winners.append(best_name)
        st.winner_fps.add(st.cand_fp[best_name])
        src_cols, tname = st.name_to_parsed[best_name]
        st.rows.append({
            "engineered_col": best_name,
            "transform": tname,
            "source_cols": tuple(src_cols),
            "cmi_at_selection": float(best_cmi),
            "step": st.step,
        })
        # Fold the winner into the conditioning support so the next CMI
        # measures the gain ON TOP OF this column. Same fragmentation
        # cap as the seed-support build: if folding would push joint Z
        # past ``frag_cap`` cells, freeze Z (the winner still counts as
        # selected, but later CMI continues against the previous Z so
        # downstream candidates are still measurable).
        if cand_bins_dev:
            # RESIDENT fold: the winner's codes are already device-resident
            # (``cand_bins_dev[best_name]``) - fold them into Z entirely on-device via
            # ``_renumber_joint_gpu`` instead of materializing the winner's (n,) codes host-side just to
            # call the host ``_renumber_joint``. ``z_joint`` (host) is left ``None`` and only backfilled
            # lazily (``_host_z`` in the fallback branch above) if a later round's batched CMI call faults
            # and needs a host mirror - the common, all-batched-rounds-succeed path never materializes it.
            import cupy as _cp

            new_support_dev = cand_bins_dev[best_name]
            if z_joint_dev is None:
                z_joint_dev = new_support_dev.copy()
                st.z_card = int(_cp.unique(z_joint_dev).size)
            else:
                candidate_joint_dev, _ = _renumber_joint_gpu(z_joint_dev, new_support_dev)
                cand_card = int(_cp.unique(candidate_joint_dev).size)
                if cand_card <= st.frag_cap:
                    z_joint_dev = candidate_joint_dev
                    st.z_card = cand_card
        else:
            new_support_bin = _host_bins(best_name)
            if z_joint is None or z_joint.size == 0:
                z_joint = new_support_bin.copy()
                st.z_card = int(np.unique(z_joint).size)
            else:
                candidate_joint, _ = _renumber_joint(z_joint, new_support_bin)
                cand_card = int(np.unique(candidate_joint).size)
                if cand_card <= st.frag_cap:
                    z_joint = candidate_joint
                    st.z_card = cand_card
            # else: leave z_joint unchanged; subsequent CMI uses prev Z.
        st.remaining.discard(best_name)
        st.step += 1
