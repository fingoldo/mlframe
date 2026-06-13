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
    from numba import njit
    from numba.core import types as _nb_types
    from numba.typed import Dict as _NbDict
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a hard dep in practice
    _NUMBA_AVAILABLE = False
    _nb_types = None
    _NbDict = None

    def njit(*args, **kwargs):  # no-op fallback so the module imports
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

logger = logging.getLogger(__name__)

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


@njit(cache=True)
def _combine_factorize_njit(joint: np.ndarray, c: np.ndarray, mult: int) -> tuple:
    """Fused ``factorize(joint + c*mult)`` in ONE pass, no temporaries.

    Equivalent to ``_factorize_dense_njit(joint + c*mult)`` but folds the
    multiply-add into the factorize walk -- avoids the two numpy temp arrays
    (``c*mult`` and the sum) the `_renumber_joint` per-column step allocated, and
    walks the data once instead of three times. First-seen dense ids, so the
    induced partition + nclasses match the numpy form exactly (bit-identical)."""
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
    x: np.ndarray, y: np.ndarray, z_joint: Optional[np.ndarray],
) -> float:
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
    """
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
        return max(0.0, mi_plugin - mi_bias)
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
    return max(0.0, cmi_plugin - cmi_bias)


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
    x_i = np.ascontiguousarray(x, dtype=np.int64).ravel()
    xz, _ = _renumber_joint(x_i, z_i)
    xyz, _ = _renumber_joint(x_i, y_i, z_i)
    h_xz, k_xz = _entropy_from_classes(xz)
    h_xyz, k_xyz = _entropy_from_classes(xyz)
    cmi_plugin = h_xz + h_yz - h_z - h_xyz
    cmi_bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
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

    out = {}
    for c in X_cand.columns:
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
        for name in remaining:
            # Skip candidates that are monotone-equivalent (bin-pattern-
            # identical) to an already-picked winner. Important when Z
            # is frozen at the fragmentation cap: identical-bin
            # candidates carry identical CMI and would all be picked
            # otherwise.
            if cand_fp[name] in winner_fps:
                continue
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
