"""CMI-greedy feature constructor (Layer 60, 2026-05-31).

Sibling to :mod:`_mi_greedy_fe` (Layer 26). Where Layer 26 ranks the same
candidate transform pool by MARGINAL ``MI(candidate; y)`` and de-duplicates
selected winners post-hoc via Spearman, THIS module ranks by CONDITIONAL
``MI(candidate; y | current_support)`` -- i.e. each step directly measures
the NEW information the candidate adds on top of the already-selected
columns.

Why CMI ranking matters
-----------------------

Marginal MI ranks ``log_abs(x)``, ``square(x)``, ``abs(x)`` all near the top
when ``y = sign(x^2 - 1)`` because each is monotone in ``|x|`` and so
captures the same signal. The marginal-MI greedy path then picks all three
and the downstream Spearman dedup drops two of them post-hoc -- waste.
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
the currently-selected binned columns -- collapsed via the densely-renumbered
contingency table so the joint stays computable even at d=8+ support cols
(memory dominated by ``n``, not by the cartesian bin space).
"""
from __future__ import annotations

import logging
import math
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
        return range(*a)

    def njit(*args, **kwargs):  # no-op fallback so the module imports
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

logger = logging.getLogger(__name__)

# GPU quantile-bin crossover (2026-06-28, synchronized micro-bench of _quantile_bin_gpu incl. code D2H, GTX
# 1050 Ti, nbins=10): a single host column round-tripped to the device for equi-frequency binning (H2D +
# cp.percentile sort + cp.searchsorted + code D2H) only beats the host introselect-partition np.quantile path
# well above the launch/transfer floor -- n=20k CPU 0.92ms vs GPU 1.67ms; n=35k near-tie 1.53 vs 1.71ms; n=100k
# GPU 2.10 vs CPU 4.12ms = 2x; n=300k GPU 3.06 vs CPU 14.1ms = 4.6x. The gate is set at 50k (clear of the 35k
# near-tie) so every routed call is a decisive win; the small (3k/20k) gate-redundancy columns stay on the
# host, where the fixed ~1.7ms device round-trip overhead loses to numpy.
_GPU_QBIN_MIN_ROWS = 50_000


def _quantile_bin_gpu(a: np.ndarray, nbins: int):
    """Device equi-frequency bin of an all-finite 1-D float column -> host int64 codes, selection-equivalent
    to the numpy ``_quantile_bin`` fast path.
    Returns ``None`` on any failure so the caller transparently keeps the numpy path. NEVER frees the cupy
    memory pool. Codes can 1-off the numpy codes at <~1e-5 of rows where cp.percentile and np.quantile round a
    boundary differently -- below the bin resolution, MI/cardinality selection-equivalent (the acceptance bar).

    Device 1-D twin of the numpy fast path: ``cp.percentile`` (on the RAVELLED array -- cp.percentile(X,axis=0)
    returns WRONG edges for an (n,1) column, the known cupy single-column bug guarded in
    ``_gpu_resident_discretize_codes``) + ``cp.unique`` edge-dedup + ``cp.searchsorted`` on the deduped interior
    edges. The unique-dedup is load-bearing on low-cardinality / mass-point columns: skipping it (or hitting the
    (n,1) percentile bug) splits a tied bin and breaks the occupied-bin partition the redundancy gate keys on.
    This mirrors the numpy ``np.unique(np.quantile(a,qs))`` + ``searchsorted(edges[1:-1], a, 'right')`` exactly
    (cp.percentile uses [0,100] vs np.quantile's [0,1] -- same linear-interpolation edges)."""
    try:
        import cupy as cp

        xd = cp.asarray(np.ascontiguousarray(a, dtype=np.float64))
        qs = cp.linspace(0.0, 100.0, int(nbins) + 1)
        edges = cp.unique(cp.percentile(xd, qs))   # raveled 1-D -> cp.percentile is correct (no axis=0 bug)
        if int(edges.size) <= 2:
            out = cp.zeros(xd.size, dtype=cp.int64)
            if int(edges.size) == 2:
                out[:] = (xd >= edges[1]).astype(cp.int64)
            return cp.asnumpy(out)
        codes = cp.searchsorted(edges[1:-1], xd, side="right").astype(cp.int64)
        return cp.asnumpy(codes)
    except Exception:
        logger.debug("GPU _quantile_bin failed; numpy fallback", exc_info=True)
        return None


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


def _quantile_bin(col: np.ndarray, nbins: int) -> np.ndarray:
    """Equi-frequency bin a 1-D float column into ``nbins`` integer classes.

    Constant or near-constant columns degenerate to a single class (0). NaN
    / Inf are mapped to bin 0 (caller is expected to scrub upstream; we keep
    the fallback for safety).

    By design, a low-cardinality column can collapse to a single (or two) bin even when it is informative: ``np.unique(np.quantile(...))`` dedupes the
    equi-frequency edges, so a column with few distinct values yields ``edges.size <= 2`` and reads MI ~= 0 here. This is the price of monotone-invariance
    (the binning depends only on rank order, not raw spacing) and is intentional, NOT a bug -- the marginal-MI path (Layer 26) sees such columns through its
    own binning, and the CMI-greedy step is meant to score CONDITIONAL gain on top of that. Do not "fix" this by switching to value-width bins; that would
    break the rank-invariance the CMI numbers rely on (see the bench-attempt-rejected note below for why rank-based rebinning was rejected too).
    """
    # bench-attempt-rejected (2026-06-01): replacing np.quantile value-edge
    # binning with a numba argsort rank-based equi-frequency binner was BOTH
    # slower (0.60x: 1222ms vs 730ms / 411 calls -- numpy np.quantile uses
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
        # GPU fast path (STRICT-resident, large-operand only): the n-sized equi-frequency binning of the
        # gate-redundancy / subsumption / additive-fusion continuous columns is 6x on-device at n=300k
        # (synchronized bench). Size-gated (_GPU_QBIN_MIN_ROWS) so only above-crossover columns route to the
        # GPU; the small columns and the no-CUDA / non-strict default keep the byte-identical numpy path.
        if a.size >= _GPU_QBIN_MIN_ROWS:
            try:
                from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
                _gpu_on = fe_gpu_strict_resident_enabled()
            except Exception:
                _gpu_on = False
            if _gpu_on:
                _g = _quantile_bin_gpu(a, nbins)
                if _g is not None:
                    return _g
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


# Direct-array factorize is used while the joint's max id keeps the ``seen``
# lookup buffer under this many int64 entries (~128 MB at the cap). Above it
# (cartesian blow-up: a high-cardinality support col times a large running
# class count) we fall back to the hash path so memory stays bounded.
_FAC_ARRAY_CAP = 16_000_000


@njit(cache=True)
def _factorize_dense_njit(joint: np.ndarray) -> tuple:
    """Factorize an int64 array to dense first-seen ids in one O(n) pass.

    Replaces ``np.unique(joint, return_inverse=True)``'s O(n log n) sort. The
    per-fold joint is bounded (``old_dense(0..mult-1) + c*mult``), so when the
    max id keeps the ``seen`` buffer small we use a direct-array counting pass
    (array indexing, no hashing -- ~10x over the typed.Dict form, ~17x over
    np.unique on the common low-cardinality group/cat joints). A typed.Dict
    fallback guards the rare cartesian-blow-up (high-card col x large running
    class count) so the lookup buffer never explodes.

    Ids are assigned FIRST-SEEN, not sorted -- semantically equivalent for every
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


# Parallel-memset crossover for the dense ``seen`` buffer in ``_combine_factorize_njit``. The first-seen
# factorize WALK is irreducibly sequential (each id depends on the running ``seen``/``nc`` state), but its
# ``np.full(kmax+1, -1)`` initialisation is an independent fill that prange-splits across threads. The fill
# is only worth the thread spin-up once it is large: synchronized micro-bench (2026-06-29, 4 threads, GTX
# box, n=1M) put the crossover near kmax~500k -- below it parallel loses (kmax~40k 0.99x, ~200k 0.92x),
# above it wins and scales (500k 1.08x, 1M 1.06x, 2M 1.09x, 4M 1.12x, 16M 1.24x). At the cap (16M int64 =
# 128 MB) the fill alone is ~28ms of the ~55ms call, so parallelising it is the single largest safe win on
# this sequential kernel. Below the gate the fill stays serial -> bit-identical, zero regression.
_FAC_PAR_MEMSET_MIN = 500_000


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


@njit(cache=True, parallel=True)
def _combine_factorize_njit(joint: np.ndarray, c: np.ndarray, mult: int) -> tuple:
    """Fused ``factorize(joint + c*mult)`` in ONE pass, no temporaries.

    Equivalent to ``_factorize_dense_njit(joint + c*mult)`` but folds the
    multiply-add into the factorize walk -- avoids the two numpy temp arrays
    (``c*mult`` and the sum) the `_renumber_joint` per-column step allocated, and
    walks the data once instead of three times. First-seen dense ids, so the
    induced partition + nclasses match the numpy form exactly (bit-identical).

    The first-seen WALK stays serial (data-dependent on the running ``seen``/``nc`` state -- see the
    iter16 GPU bench-note below). The only parallel part is the large dense ``seen`` initialisation, prange-
    filled when ``kmax+1 >= _FAC_PAR_MEMSET_MIN`` (gate keeps small buffers serial -> bit-identical, no spin-
    up tax). Result is bit-identical to :func:`_combine_factorize_serial_njit` for every input (parity-tested).

    bench-note (iter16, 2026-06-23, resident-GPU /loop): NOT routed to GPU. The dense renumber is a
    first-seen sequential scan -- each output id depends on the running ``seen`` table + ``nc`` counter, a
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
    ``a[i]*(max_b+1)+b[i]`` -- the same fast array-counting trick as ``_factorize_dense_njit``,
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
    resulting ids are densely numbered 0..nclasses-1 -- this is what makes
    multivariate Z trackable: even with d=8 support cols * 10 bins each
    (10**8 cartesian space) the actual occupied bins are <= n_samples, so
    we never allocate the cartesian space.

    Per-fold renumbering uses the njit hash-factorize (first-seen dense ids)
    instead of ``np.unique`` -- see :func:`_factorize_dense_njit`.
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


def _entropy_from_classes(classes: np.ndarray) -> tuple[float, int]:
    """``H = -sum p_i log p_i`` from an integer class array (natural log).

    Returns ``(H_plugin, n_nonempty_cells)``. The cell count is used by
    Miller-Madow bias correction in :func:`_cmi_from_binned` -- plug-in
    MLE entropy has positive bias O((K-1)/(2n)); subtracting the same
    quantity from CMI cancels at first order.

    Delegates to the njit kernel after ensuring a contiguous int64 array (the
    kernel indexes a ``counts`` buffer by class id, so non-negative dense ids
    are required -- guaranteed by the binned / ``_renumber_joint`` callers).
    """
    if classes.size == 0:
        return 0.0, 0
    classes = np.ascontiguousarray(classes, dtype=np.int64)
    H, k = _entropy_from_classes_njit(classes)
    return float(H), int(k)


def _cmi_from_binned(
    x: np.ndarray, y: np.ndarray, z_joint: Optional[np.ndarray], return_cards: bool = False,
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
    # GPU route (2026-06-25): same partition-entropy-via-cp.unique offload as cmi_from_binned_fixed_yz,
    # covering the general CMI callers (conditional-perm null, candidate scoring). value-order densify ->
    # same partition -> same CMI; selection-identical. Gated (STRICT / MLFRAME_CMI_GPU), default CPU.
    if _cmi_gpu_enabled():
        try:
            return _cmi_from_binned_cupy(x, y, z_joint, return_cards=return_cards)
        except Exception:
            pass
    x_i = np.ascontiguousarray(x, dtype=np.int64)
    y_i = np.ascontiguousarray(y, dtype=np.int64)
    n = float(max(1, x_i.size))
    if z_joint is None or z_joint.size == 0:
        h_x, k_x = _entropy_from_classes(x_i)
        h_y, k_y = _entropy_from_classes(y_i)
        xy, _ = _renumber_joint(x_i, y_i)
        h_xy, k_xy = _entropy_from_classes(xy)
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
    xz, _ = _renumber_joint(x_i, z_i)
    yz, _ = _renumber_joint(y_i, z_i)
    xyz, _ = _renumber_joint(x_i, y_i, z_i)
    h_z, k_z = _entropy_from_classes(z_i)
    h_xz, k_xz = _entropy_from_classes(xz)
    h_yz, k_yz = _entropy_from_classes(yz)
    h_xyz, k_xyz = _entropy_from_classes(xyz)
    cmi_plugin = h_xz + h_yz - h_z - h_xyz
    # Plug-in CMI = H(XZ) + H(YZ) - H(Z) - H(XYZ). Each plug-in entropy
    # is biased low by (K-1)/(2n). The CMI bias from combining them
    # (with signs +H_xz +H_yz -H_z -H_xyz, where each contributes
    # -(K-1)/(2n) to the plug-in vs true entropy) is:
    #   E[CMI_plugin] - CMI_true
    #   = -((k_xz-1) + (k_yz-1) - (k_z-1) - (k_xyz-1))/(2n)
    #   = (k_xyz + k_z - k_xz - k_yz)/(2n).
    # On noise frames k_xyz - k_xz dominates (XYZ has many empty cells
    # filled by noise) so plug-in CMI is biased UP -- subtract the
    # bias to denoise.
    cmi_bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
    cmi = max(0.0, cmi_plugin - cmi_bias)
    return (cmi, (int(k_z), int(k_xz), int(k_yz), int(k_xyz))) if return_cards else cmi


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
    xy, _ = _renumber_joint(x_i, y_i)
    h_xy, k_xy = _entropy_from_classes(xy)
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
    result -- pure wasted work. This returns the invariant block once; pair with
    :func:`cmi_from_binned_fixed_yz` for the per-permutation evaluation.

    Returns ``(y_i, z_i, h_yz, h_z, k_yz, k_z, n)`` where ``y_i`` / ``z_i`` are
    contiguous int64 views reused by every permutation.
    """
    y_i = np.ascontiguousarray(y, dtype=np.int64).ravel()
    z_i = np.ascontiguousarray(z_joint, dtype=np.int64).ravel()
    n = float(max(1, y_i.size))
    yz, _ = _renumber_joint(y_i, z_i)
    h_z, k_z = _entropy_from_classes(z_i)
    h_yz, k_yz = _entropy_from_classes(yz)
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
) -> float:
    """``CMI(X; Y | Z)`` for a fresh ``x`` reusing the y/z-invariant terms from
    :func:`precompute_cmi_yz_terms`. Computes only the x-dependent ``xz`` / ``xyz``
    renumberings + their entropies; bit-identical to :func:`_cmi_from_binned` on
    the same inputs (it is the same arithmetic with the y/z block factored out)."""
    # GPU route (2026-06-25): the xz / xyz joint ENTROPIES are partition statistics -- cp.unique(flat_key,
    # return_counts) densifies on the device (sort+unique, value-order labels) and the counts give the SAME
    # partition -> the SAME entropy -> the SAME CMI (only fp reduction order differs ~1e-15; selection
    # identical). Routes the dominant mi_greedy CMI compute (combine_factorize + entropy) onto the GPU under
    # MLFRAME_FE_GPU_STRICT / the KTC gate, instead of the host njit renumber+entropy. Falls back to CPU on
    # any cupy error.
    if _cmi_gpu_enabled():
        try:
            return _cmi_from_binned_fixed_yz_cupy(x, y_i, z_i, h_yz, h_z, k_yz, k_z, n)
        except Exception:
            pass
    x_i = np.ascontiguousarray(x, dtype=np.int64).ravel()
    xz, _ = _renumber_joint(x_i, z_i)
    xyz, _ = _renumber_joint(x_i, y_i, z_i)
    h_xz, k_xz = _entropy_from_classes(xz)
    h_xyz, k_xyz = _entropy_from_classes(xyz)
    cmi_plugin = h_xz + h_yz - h_z - h_xyz
    cmi_bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
    return max(0.0, cmi_plugin - cmi_bias)


def _cmi_gpu_enabled() -> bool:
    """Route the mi_greedy CMI entropies to the GPU when STRICT-GPU is on (or a future KTC gate). Default
    OFF -> host path, byte-identical. STRICT_GPU=1 forces it (the user's "make GPU actually carry the FE
    compute" knob: most FE families were CPU-only, this puts the dominant CMI on the device)."""
    import os as _os
    if _os.environ.get("MLFRAME_CMI_GPU", "") == "1":
        return True
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        return bool(fe_gpu_strict_enabled())
    except Exception:
        return False


# Fused entropy + occupied-cell reduction over a bincount histogram (launch-reduction, 2026-06-25). The
# entropy tail ``c[c>0]; p=c*inv_n; -(p*log p).sum()`` plus the ``c.shape[0]`` count expanded to a
# boolean-mask getitem + astype + multiply + log + sum (~5 cuLaunchKernel). Two cupy ReductionKernels
# (each ONE launch, same cuLaunchKernel driver API -> genuine count reduction) fold it: ENT_RK maps each
# count to ``(c*inv_n)*log(c*inv_n)`` (0 when c==0) and reduces, NNZ_RK counts occupied cells. The c>0
# guard inside the map removes the separate boolean filter. Same plug-in entropy math -> selection-equiv.
_ENT_RK = None
_NNZ_RK = None

# FUSED entropy + occupied-cell in ONE RawKernel (launch-reduction, 2026-06-25). _ent_from_counts ran TWO
# cupy ReductionKernels (ENT_RK for sum xlog x, NNZ_RK for occupied cells) -- two cuLaunchKernel per call,
# and after the per-candidate CMI scoring it was the measured #1 launch source (477). One grid-stride kernel
# now reduces both: each thread accumulates (sum xlogx, nnz) over its strided cells, a block reduction in
# shared memory folds the block, and one atomicAdd per block adds into a 2-slot output (out[0]=sum xlogx,
# out[1]=nnz). out is cp.zeros(2) -- a cudaMemsetAsync, NOT a cuLaunchKernel -- so the call is ONE launch.
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
        out = cp.zeros(2, dtype=cp.float64)             # cudaMemsetAsync, not a cuLaunchKernel
        threads = 256
        blocks = min(1024, max(1, (M + threads - 1) // threads))
        _get_ent_nnz_kernel(cp)((blocks,), (threads,), (c, float(inv_n), np.int64(M), out))
        h_k = cp.asnumpy(out)
        return float(-h_k[0]), int(round(h_k[1]))
    except Exception:                                   # noqa: BLE001
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


# (id(host), size, endpoints) -> max(dev_codes)+1 cardinality. y is a fit-constant and z_support a
# round-constant across the per-candidate CMI calls, so their dev .max() (a reduction + scalar D2H) recurs
# identically; memoize it (keyed like _YZ_CARD_CACHE: endpoints guard id() reuse after GC). x varies -> never
# cached. Selection-exact (same integer cardinality). Module-level -> never on a pickled instance.
_CARD_MAX_CACHE: dict = {}


def _cached_card(host_arr, dev_codes) -> int:
    if dev_codes.size == 0:
        return 1
    ha = np.asarray(host_arr).ravel()
    key = (id(host_arr), int(ha.size), int(ha[0]), int(ha[-1]))
    v = _CARD_MAX_CACHE.get(key)
    if v is None:
        v = int(dev_codes.max()) + 1
        if len(_CARD_MAX_CACHE) > 64:
            _CARD_MAX_CACHE.clear()
        _CARD_MAX_CACHE[key] = v
    return v

# bench-attempt-rejected (2026-06-26): caching the y/z device codes (a _cached_dev resident-operand cache) to
# skip re-uploading the fit-constant y / round-constant z H2D per candidate saved only ~61 MB / ~10 ms on the
# F2 300k STRICT wall (below the 0.5% ship bar) -- nsys shows the redundancy gate is overhead/orchestration-
# bound (GPU idle ~90%), NOT operand-H2D-bound (only ~118 operand uploads; the 1790 cudaMemcpyAsync are
# per-kernel scalar D2H, not operand re-uploads). Not worth a DATA cache's stale id()-reuse collision risk for
# ~10 ms. The real redundancy is the DOUBLE card computation, eliminated by the precomp_cards reuse below.


def _cmi_from_binned_cupy(x, y, z_joint, return_cards: bool = False):
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

    dx = cp.asarray(np.ascontiguousarray(x, dtype=np.int64).ravel())
    dy = cp.asarray(np.ascontiguousarray(y, dtype=np.int64).ravel())
    n = float(max(1, int(dx.size)))
    inv_n = 1.0 / n

    from ._fe_batched_mi import joint_entropy_gpu

    def _entc(codes, cards):
        # FUSED histogram + plug-in entropy in ONE launch (shared-mem hist + entropy reduce) when the joint
        # fits shared, else the two-kernel path. Same partition counts -> selection-equivalent.
        return joint_entropy_gpu(codes, cards, inv_n)

    Kx = (int(dx.max()) + 1) if dx.size else 1
    ky = _cached_card(y, dy)               # y is a fit-constant -> its cardinality is cached
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
    dz = cp.asarray(np.ascontiguousarray(z_joint, dtype=np.int64).ravel())
    kz = _cached_card(z_joint, dz)         # z_support is round-constant -> cardinality cached
    # FOUR joint entropies (z, xz, yz, xyz) in ONE launch when the (x,y,z) joint fits shared -- the #1
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


_YZ_CARD_CACHE: dict = {}   # (y,z)-identity -> (k_z, k_yz, ky, kz); invariant across a round's candidates


def joint_cardinalities_cupy(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[int, int, int, int]:
    """Occupied-cell counts (k_z, k_xz, k_yz, k_xyz) for the analytic CMI-null df, via device cp.unique.
    Only the cardinalities (number of distinct joint codes) are needed -> cp.unique(...).size on the
    device replaces the host renumber+entropy. Value-order densify -> SAME occupied-cell count (the df is
    label-invariant). Raises on any cupy error so the caller falls back to the host path."""
    import cupy as cp

    from ._fe_batched_mi import joint_counts_gpu

    dx = cp.asarray(np.ascontiguousarray(x, dtype=np.int64).ravel())
    dy = cp.asarray(np.ascontiguousarray(y, dtype=np.int64).ravel())
    dz = cp.asarray(np.ascontiguousarray(z, dtype=np.int64).ravel())
    Kx = (int(dx.max()) + 1) if dx.size else 1

    from ._fe_batched_mi import joint_nnz_gpu

    def _nc(codes, cards):
        # occupied-cell count fused into the histogram pass (atomicAdd 0->1 trick) -> ONE launch, vs
        # joint_counts_gpu + _nnz_from_counts (two). Integer cardinality -> identical df.
        return joint_nnz_gpu(codes, cards)

    # CROSS-CALL CACHE of the (y,z)-only cardinalities (launch-reduction). In the analytic-null df the gate
    # scores MANY candidates against the SAME (y target, z support) within a greedy round, so k_z / k_yz
    # (and ky / kz) are INVARIANT across those per-candidate calls -- only k_xz / k_xyz depend on the
    # candidate x. Recomputing k_z + k_yz per candidate was 2 of the 4 occupied-cell histograms each call.
    # Memoize them keyed by the (y,z) host-array identity + size + endpoints (stable within a round, distinct
    # across rounds; endpoints guard against id() reuse after GC). Same cardinalities -> df unchanged.
    ya = np.asarray(y).ravel(); za = np.asarray(z).ravel()
    yzkey = (id(y), id(z), int(ya.size),
             int(ya[0]) if ya.size else 0, int(ya[-1]) if ya.size else 0,
             int(za[0]) if za.size else 0, int(za[-1]) if za.size else 0)
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


def _cmi_from_binned_fixed_yz_cupy(x, y_i, z_i, h_yz, h_z, k_yz, k_z, n) -> float:
    """Device twin of :func:`cmi_from_binned_fixed_yz`: the xz / xyz joint entropies via cp.unique counts.
    Value-order densification -> same partition -> same CMI (selection-identical, fp-order ~1e-15)."""
    import cupy as cp

    from ._fe_batched_mi import joint_entropy_gpu

    dx = cp.asarray(np.ascontiguousarray(x, dtype=np.int64).ravel())
    dy = cp.asarray(np.ascontiguousarray(y_i, dtype=np.int64).ravel())
    dz = cp.asarray(np.ascontiguousarray(z_i, dtype=np.int64).ravel())
    inv_n = 1.0 / float(n)

    def _entc(codes, cards):
        # FUSED histogram + plug-in entropy in ONE launch (shared-mem hist + entropy reduce) when the joint
        # fits shared, else the two-kernel path. Same partition counts -> selection-equivalent.
        return joint_entropy_gpu(codes, cards, inv_n)

    Kx = (int(dx.max()) + 1) if dx.size else 1
    ky = _cached_card(y_i, dy)             # y is a fit-constant -> cardinality cached
    kz = int(k_z) if int(k_z) > 0 else (int(dz.max()) + 1 if dz.size else 1)
    h_xz, k_xz = _entc([dx, dz], [Kx, kz])
    h_xyz, k_xyz = _entc([dx, dy, dz], [Kx, ky, kz])
    cmi_plugin = h_xz + float(h_yz) - float(h_z) - h_xyz
    cmi_bias = (k_xyz + int(k_z) - k_xz - int(k_yz)) / (2.0 * float(n))
    return max(0.0, cmi_plugin - cmi_bias)


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
    y_arr = np.asarray(y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    # Bin y by unique-value remap (y is already class-typed at the call
    # site; this just renumbers to dense 0..K-1).
    _, y_bin = np.unique(y_arr, return_inverse=True)
    y_bin = y_bin.astype(np.int64)

    if X_support is None or X_support.shape[1] == 0:
        z_joint: Optional[np.ndarray] = None
    else:
        sup_bins = [
            _quantile_bin(X_support[c].to_numpy(), nbins=nbins)
            for c in X_support.columns
        ]
        z_joint, _ = _renumber_joint(*sup_bins)

    cand_cols = list(X_cand.columns)
    # BATCHED born-on-device path under STRICT (default OFF -> per-candidate CPU loop, byte-identical):
    # bin all candidates into one (n, K) code matrix and score CMI for EVERY candidate in ONE device
    # workload (batched_cmi_gpu), instead of a per-candidate cp.unique CMI. Parity-pinned selection-equiv.
    if _cmi_gpu_enabled() and len(cand_cols) > 1:
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
                cmis = batched_cmi_gpu(X_codes, y_bin, z_joint, codes_trusted=True)   # host equi-freq binner -> 0-based
            return pd.Series({c: float(cmis[j]) for j, c in enumerate(cand_cols)}, dtype=np.float64)
        except Exception:
            pass  # any cupy error -> exact CPU loop below
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end CMI-greedy feature constructor.

    Pipeline:

    1. Enumerate UNARY candidates over the FULL numeric column pool (NOT a
       top-N seed pool). The whole point of CMI ranking is that columns
       with near-zero marginal ``MI(x; y)`` can still emit transforms that
       carry the signal -- on ``y = sign(x^2 - 1)``, ``MI(x; y) ~= 0``
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
    from ._mi_greedy_fe import (
        generate_mi_greedy_features,
        iter_candidates,
    )
    from ._orthogonal_univariate_fe import _mi_classif_batch

    empty_scores = pd.DataFrame(columns=[
        "engineered_col", "transform", "source_cols",
        "cmi_at_selection", "step",
    ])

    candidates_pool = [
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    if not candidates_pool:
        return X.copy(), empty_scores

    y_arr = np.asarray(y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)

    # 1. Pick the top-N raw cols by marginal MI as the BINARY-pair source
    #    pool (controls the O(N^2 * |BINARY_TRANSFORMS|) explosion).
    #    Unary candidates still enumerate over the full pool below.
    raw_arr = X[candidates_pool].to_numpy(dtype=np.float64)
    raw_mi = _mi_classif_batch(raw_arr, y_arr, nbins=nbins)
    order = np.argsort(-raw_mi)
    binary_seed_cols = [
        candidates_pool[i] for i in order[: int(seed_cols_count)]
    ] if int(seed_cols_count) > 0 else list(candidates_pool)

    # 2. Enumerate candidates. UNARY over the full pool (so transforms on
    #    symmetric / interaction-only cols are never silently dropped),
    #    BINARY only over the seeded subset (pair explosion control).
    cands: list[tuple[tuple[str, ...], str]] = []
    if include_unary:
        cands.extend(iter_candidates(
            X, cols=candidates_pool,
            include_unary=True, include_binary=False,
            include_trig_on_bounded=include_trig_on_bounded,
        ))
    if include_binary:
        cands.extend(iter_candidates(
            X, cols=binary_seed_cols,
            include_unary=False, include_binary=True,
            include_trig_on_bounded=False,
        ))
    engineered, parsed = generate_mi_greedy_features(X, cands)
    if engineered.empty:
        return X.copy(), empty_scores

    # 3. Bin y; start Z EMPTY. Z grows step-by-step from greedy picks
    #    (under the fragmentation cap below). Starting with several raw
    #    cols in Z up front pushes joint Z cardinality past the
    #    chi-squared contingency budget and collapses every candidate's
    #    CMI toward noise -- defeats the purpose of CMI ranking.
    _, y_bin = np.unique(y_arr, return_inverse=True)
    y_bin = y_bin.astype(np.int64)
    n_samples = int(y_bin.size)
    frag_cap = max(2, n_samples // 5)
    z_joint: Optional[np.ndarray] = None
    z_card = 1

    # 4. Bin every engineered candidate up front. Compute a sortable
    #    bin fingerprint (tuple of the sorted unique bin counts) used
    #    below for monotone-equivalence dedup against already-picked
    #    winners -- when Z hits the fragmentation cap (frozen Z),
    #    monotone-equivalent candidates would otherwise tie at the same
    #    plug-in CMI and all get picked.
    cand_names = list(engineered.columns)
    name_to_parsed = dict(zip(cand_names, parsed))
    cand_bins: dict[str, np.ndarray] = {
        name: _quantile_bin(engineered[name].to_numpy(), nbins=nbins)
        for name in cand_names
    }

    def _bin_fingerprint(b: np.ndarray) -> bytes:
        # Bytes of the int64 array -> hashable + cheap. Identical
        # binned arrays (i.e. monotone-equivalent under equi-frequency
        # quantization) collapse to one fingerprint.
        return b.tobytes()

    cand_fp: dict[str, bytes] = {
        name: _bin_fingerprint(cand_bins[name]) for name in cand_names
    }

    # Permutation-based noise-floor for the current Z: shuffle y once,
    # rebin, sample 24 candidates' CMI; take the 95th percentile as
    # the floor. Combined with the user's ``min_cmi_gain`` via max().
    # Avoids the "noise CMI ~ 0.01 with k=4 Z and small n still admits
    # spurious transforms" failure mode that bias correction alone
    # can't fully suppress at finite n. Recomputed when Z grows so the
    # floor scales with the conditioning's fragmentation.
    rng_floor = np.random.default_rng(0xC011)

    def _noise_floor_for_current_z() -> float:
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
            if _cmi_gpu_enabled() and len(sample_names) > 1:
                from ._fe_batched_mi import batched_cmi_gpu

                _Xs = np.empty((int(y_shuf.shape[0]), len(sample_names)), dtype=np.int64)
                for _j, _nm in enumerate(sample_names):
                    _Xs[:, _j] = cand_bins[_nm]
                _zc = z_joint if (z_joint is not None and z_joint.size > 0) else None
                _cmis = np.asarray(batched_cmi_gpu(_Xs, y_shuf, _zc), dtype=np.float64)
                return float(np.quantile(_cmis, 0.95))
        except Exception:
            pass
        cmis_shuf = []
        for nm in sample_names:
            cmis_shuf.append(_cmi_from_binned(cand_bins[nm], y_shuf, z_joint))
        if not cmis_shuf:
            return 0.0
        return float(np.quantile(np.asarray(cmis_shuf), 0.95))

    # 5. Greedy CMI loop.
    winners: list[str] = []
    winner_fps: set[bytes] = set()
    rows: list[dict] = []
    remaining = set(cand_names)
    step = 0
    z_card_at_floor = -1
    cur_floor = 0.0
    while remaining and len(winners) < int(top_k):
        # Recompute the noise floor when Z cardinality changes (i.e. Z
        # was grown since last iter). Step 0 always recomputes.
        if z_card != z_card_at_floor:
            cur_floor = _noise_floor_for_current_z()
            z_card_at_floor = z_card
        effective_floor = max(float(min_cmi_gain), cur_floor)
        best_name = None
        best_cmi = -1.0
        # Hoist the y/z-invariant CMI block out of the per-candidate scan: y_bin
        # and z_joint are fixed across the inner loop, so H(Y,Z) / H(Z) and their
        # renumberings are recomputed-and-discarded once per candidate by the
        # plain ``_cmi_from_binned``. When Z is non-empty, precompute them once
        # per step and score candidates via ``cmi_from_binned_fixed_yz`` (same
        # arithmetic, x-block only). Marginal (empty-Z) step keeps the original
        # path since the hoist helpers assume a present Z.
        _have_z = z_joint is not None and z_joint.size > 0
        if _have_z:
            _y_i, _z_i, _h_yz, _h_z, _k_yz, _k_z, _n = precompute_cmi_yz_terms(
                y_bin, z_joint)
        # The fingerprint-skip set is fixed for this step -> the candidates to score are
        # ``_scan`` (a list over ``remaining`` preserving its iteration order). y_bin / z_joint are fixed
        # across them, so score ALL their CMI in ONE batched_cmi_gpu workload and take the first-max argmax
        # (== the sequential ``cmi > best_cmi`` tie-break). Same MM plug-in CMI -> selection-equivalent;
        # per-candidate loop fallback on any error / GPU-off.
        _scan = [name for name in remaining if cand_fp[name] not in winner_fps]
        _batched_cmis = None
        try:
            if _cmi_gpu_enabled() and len(_scan) > 1:
                from ._fe_batched_mi import batched_cmi_gpu

                _Xc = np.empty((int(y_bin.shape[0]), len(_scan)), dtype=np.int64)
                for _j, _nm in enumerate(_scan):
                    _Xc[:, _j] = cand_bins[_nm]
                _zc = z_joint if _have_z else None
                _batched_cmis = np.asarray(batched_cmi_gpu(_Xc, y_bin, _zc), dtype=np.float64)
        except Exception:
            _batched_cmis = None
        if _batched_cmis is not None:
            _bi = int(np.argmax(_batched_cmis))   # first-max == the sequential cmi > best_cmi tie-break
            best_cmi = float(_batched_cmis[_bi]); best_name = _scan[_bi]
        else:
            for name in _scan:
                if _have_z:
                    cmi = cmi_from_binned_fixed_yz(
                        cand_bins[name], _y_i, _z_i, _h_yz, _h_z, _k_yz, _k_z, _n)
                else:
                    cmi = _cmi_from_binned(cand_bins[name], y_bin, z_joint)
                if cmi > best_cmi:
                    best_cmi = cmi
                    best_name = name
        if best_name is None:
            break
        if best_cmi < effective_floor:
            # No remaining candidate adds enough new info; stop.
            break
        winners.append(best_name)
        winner_fps.add(cand_fp[best_name])
        src_cols, tname = name_to_parsed[best_name]
        rows.append({
            "engineered_col": best_name,
            "transform": tname,
            "source_cols": tuple(src_cols),
            "cmi_at_selection": float(best_cmi),
            "step": step,
        })
        # Fold the winner into the conditioning support so the next CMI
        # measures the gain ON TOP OF this column. Same fragmentation
        # cap as the seed-support build: if folding would push joint Z
        # past ``frag_cap`` cells, freeze Z (the winner still counts as
        # selected, but later CMI continues against the previous Z so
        # downstream candidates are still measurable).
        new_support_bin = cand_bins[best_name]
        if z_joint is None or z_joint.size == 0:
            z_joint = new_support_bin.copy()
            z_card = int(np.unique(z_joint).size)
        else:
            candidate_joint, _ = _renumber_joint(z_joint, new_support_bin)
            cand_card = int(np.unique(candidate_joint).size)
            if cand_card <= frag_cap:
                z_joint = candidate_joint
                z_card = cand_card
            # else: leave z_joint unchanged; subsequent CMI uses prev Z.
        remaining.discard(best_name)
        step += 1

    scores = pd.DataFrame(rows, columns=[
        "engineered_col", "transform", "source_cols",
        "cmi_at_selection", "step",
    ])
    if winners:
        X_aug = pd.concat([X, engineered[winners]], axis=1)
    else:
        X_aug = X.copy()
    return X_aug, scores


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
):
    """Same as :func:`greedy_cmi_fe_construct` but additionally returns a list
    of ``EngineeredRecipe`` objects (one per appended column) so MRMR.transform
    can replay each column on test data without re-running CMI scoring AND
    without referencing y.

    Recipes reuse kind ``"mi_greedy_transform"`` (same as Layer 26) so the
    replay code path is shared infrastructure.
    """
    from ._mi_greedy_fe import _parse_binary_name, _parse_unary_name
    from .engineered_recipes import build_mi_greedy_transform_recipe

    X_aug, scores = greedy_cmi_fe_construct(
        X, y,
        cols=cols, seed_cols_count=seed_cols_count, top_k=top_k,
        include_unary=include_unary, include_binary=include_binary,
        include_trig_on_bounded=include_trig_on_bounded,
        min_cmi_gain=min_cmi_gain, nbins=nbins,
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
            logger.warning(
                "greedy_cmi_fe_construct_with_recipes: cannot parse "
                "engineered column %r back to (transform, source); skipping "
                "recipe.",
                name,
            )
    return X_aug, scores, recipes
