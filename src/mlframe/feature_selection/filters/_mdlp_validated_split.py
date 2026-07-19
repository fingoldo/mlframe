"""Significance-gated variant of Fayyad-Irani MDLP splitting (2026-07-19).

PRODUCTION STATUS (2026-07-19, user decision -- accuracy over speed per project convention):
``mdlp_bin_edges`` in ``supervised_binning.py`` now calls into this module's
``_mdlp_recurse_validated`` by DEFAULT (``fast_mode=False``). The classic in-sample-MDL +
depth-cap path documented below stays available as the explicit ``fast_mode=True`` opt-out for
callers where wall-time matters more than the measured accuracy win (20-80x cheaper per column).
See ``mdlp_bin_edges``'s own docstring for the user-facing knob and the accuracy numbers.

METHODOLOGY NOTE (verified against the actual implementation, both branches below): both the
analytic and the permutation-null accept-test are computed ENTIRELY IN-SAMPLE -- the same node
rows are used to (a) search for the best cut point and (b) judge whether its gain is
"unlikely under the null that the split indicator is independent of y," on that SAME sample.
This is a genuine null-hypothesis significance test (it bounds the node-level false-positive
rate, including the max-over-candidates selection effect, via the Bonferroni correction below)
-- which is a real improvement over the raw MDL heuristic bound (not a significance test at
all). But it is NOT equivalent to true out-of-sample validation (fit the cut on one fold, check
the gain holds on a DIFFERENT held-out fold) -- a split can pass an in-sample significance test
and still not generalize if the significance test's own assumptions are violated (e.g. non-
i.i.d. rows) or if the "null" itself is mis-specified for the true data-generating process. See
``mdlp_bin_edges_oos_validated`` below for a genuine cross-validated variant, added specifically
to check whether true OOS validation catches anything the in-sample significance test misses.

Original research-prototype framing follows (still accurate for the design rationale):

Context: ``mdlp_bin_edges`` (``supervised_binning.py``) accepts/rejects each candidate split via the
classic Fayyad-Irani 1993 MDL threshold test (``_mdlp_pass_threshold_njit``), computed ENTIRELY
IN-SAMPLE -- the same rows are used to search for the split and to judge whether it clears the MDL
bound. That is a heuristic complexity-penalty approximation, not a genuine held-out/generalization
check. A sibling fix (same date, ``supervised_binning.py`` / ``_adaptive_nbins.py``) found a concrete
symptom: once a continuous target is quantile-discretized into pseudo-classes (``max_y_classes``, to
stop the ``3.0**n_classes`` MDL-threshold overflow that used to silently reject every split), an
UNCAPPED recursion depth accepts many more splits whose apparent gain does not generalize -- measured
20x feature-count blowup and 22% WORSE held-out RMSE on the real wellbore target vs the pre-fix
quantile-fallback baseline. The landed fix is a blunt ``max_depth <= ceil(log2(max_y_classes))`` cap --
correct as a guardrail, but it is an arbitrary ratio tied to one dataset's cardinality, and it does not
touch the root cause: the accept-test itself has no mechanism to detect a split whose gain is only
noise.

This module is a RESEARCH PROTOTYPE (not wired into the production ``mdlp_bin_edges`` path) that
replaces the in-sample MDL threshold with a statistical-significance gate on the SAME candidate gain,
reusing existing project infrastructure rather than inventing a new one:

  * Each MDLP candidate split partitions a node's rows into a BINARY left/right indicator ``S``. The
    gain the existing scan already computes, ``H(y) - [n_l/n * H(y|S=left) + n_r/n * H(y|S=right)]``,
    IS the mutual information ``I(S; y)`` on that node's rows (2 categories x ``n_classes_full`` in
    ``y``) -- an ordinary discrete MI, computed in nats via ``math.log`` exactly like the rest of
    this module.
  * ``_analytic_mi_null.py`` already establishes (and the codebase already relies on, for the MRMR
    redundancy/relevance permutation gate) that the classic G-test/likelihood-ratio identity
    ``2*N*MI ~ chi2(df=(Bx-1)*(By-1))`` under independence gives an ANALYTIC permutation-null
    p-value, valid once cells are not too sparse (``analytic_null_applicable``). For an MDLP split,
    ``Bx=2`` (binary indicator) and ``By=n_classes_full`` (labels present in the node), so
    ``df = n_classes_full - 1`` and the existing ``analytic_mi_null(gain, n, 2, n_classes_full)``
    helper is DIRECTLY reusable, unmodified, for every split at every recursion node.
  * When the node is too small / cells too sparse for the analytic approximation (``analytic_null_
    applicable`` returns False -- always true near ``min_split_size`` at the bottom of the recursion,
    where cheap-node-size means a literal permutation loop is affordable anyway), fall back to an
    actual permutation null: shuffle the node's ``y`` ``n_permutations`` times, recompute the SAME
    best-gain statistic via the existing ``_mdlp_best_split_njit`` kernel each time, and accept the
    observed split only if it beats the ``(1 - alpha)`` quantile of the null distribution. This
    mirrors (in spirit, not machinery -- no GPU residency needed at this data scale) the analytic-vs-
    permutation dual-path design already used by ``_analytic_mi_null.py`` / ``_fe_cmi_perm_null_gpu.py``
    for the MRMR relevance/redundancy gates.

Literature context (see module docstring in the research report for full citations): Fayyad & Irani
1993 (MDLP) is an MDL-cost heuristic, NOT a significance test. ChiMerge (Kerber 1992) and its
extension Chi2 (Liu & Setiono 1995) are the well-established alternative family that DOES gate
supervised-discretization merges/splits by a chi-square (equivalently G-test) significance test on
the same kind of binary-partition x class contingency table used here -- so "significance-gated
binary-split acceptance" is an established, decades-old idea in the discretization literature. What
is NOT established (novel-to-this-codebase) is grafting that chi-square/permutation gate onto
Fayyad-Irani's specific recursive-binary-split SEARCH procedure (candidate scan restricted to class-
boundary midpoints, MDL-style recursion bookkeeping) rather than ChiMerge's bottom-up interval-merge
procedure. No published "MDLP + permutation test" hybrid was found to cite directly; this module is
that combination, evaluated empirically below rather than asserted from a citation.

Cost note: naive per-candidate-split permutation testing (shuffle once per candidate cut point, not
just the winning split) would be far too expensive -- MDLP already restricts candidates to class-
boundary midpoints (Fayyad-Irani's own theorem) and only tests the WINNING gain per node, so this
design pays at most one significance test per recursion node (analytic: O(1); permutation-fallback:
O(n_permutations * node scan cost), and only at small, cheap nodes) -- not one per candidate cut.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._analytic_mi_null import analytic_mi_null, analytic_null_applicable
from .supervised_binning import _entropy_from_counts_njit, _mdlp_best_split_njit


def _dedupe_xy(x: np.ndarray, y_i: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Collapse exact-duplicate ``(x, y)`` rows to one occurrence each (first-seen order preserved).

    BUG FOUND AND FIXED (2026-07-19, duplicate-row robustness probe): exact duplicate rows inflate
    the raw row count both the analytic chi-square null (``df/(2N)`` bias floor, ``2*N*MI`` G-stat)
    and the permutation-fallback null use as ``N``/the shuffle size, WITHOUT adding independent
    evidence -- but that alone is not the mechanism that broke on pure-noise data at this node
    scale (n well under the analytic-null floor, so the permutation fallback ran). The real
    mechanism: after sorting by ``x``, a duplicated row sits immediately adjacent to its twin with
    an IDENTICAL ``y`` by construction, which the observed best-gain scan exploits as a genuine
    (if spurious, injection-only) zero-within-cluster-variance boundary. A permutation null built by
    shuffling ``y`` across all rows (duplicates included) breaks that exact adjacency pairing in
    every null draw, so the null never reproduces the artifact the observed statistic benefits from
    -- the observed gain then clears an otherwise-valid significance test almost every time.
    Measured on 2000 iid noise rows (x, y independent): 0% duplication -> 1 bin (correct); 10% ->
    6 bins; 50% -> 110 bins; 90% -> 139 bins. Deduplicating ``(x, y)`` pairs once, before the
    recursion starts, removes the artifact at the source (re-verified: 1 bin at every duplication
    rate above) without touching the significance-test machinery itself -- a genuinely-informative
    duplicate observation (real sensor re-reads, resampled rows) collapses to the same single
    unique ``(x, y)`` pair a first-occurrence-only sample would have produced, so this is lossless
    for the recursion's actual inputs, not a tolerance loosening.
    """
    order = np.lexsort((y_i, x))
    x_s, y_s = x[order], y_i[order]
    if x_s.size <= 1:
        return x, y_i
    keep = np.empty(x_s.size, dtype=bool)
    keep[0] = True
    keep[1:] = (x_s[1:] != x_s[:-1]) | (y_s[1:] != y_s[:-1])
    if keep.all():
        return x, y_i
    return x_s[keep], y_s[keep]


@njit(nogil=True, cache=True)
def _count_candidates_njit(x_sorted: np.ndarray, y_sorted: np.ndarray, min_split_size: int) -> int:
    """Count class-boundary candidate cut points a node's scan actually considers (mirrors the skip
    conditions inside ``_mdlp_best_split_njit`` exactly, without doing the entropy work) -- needed for
    the Bonferroni multiplicity correction on the analytic significance path."""
    n = x_sorted.shape[0]
    count = 0
    for i in range(n - 1):
        n_l = i + 1
        n_r = n - n_l
        if n_l < min_split_size or n_r < min_split_size:
            continue
        if y_sorted[i] == y_sorted[i + 1]:
            continue
        if x_sorted[i] == x_sorted[i + 1]:
            continue
        count += 1
    return count


@njit(nogil=True, cache=True)
def _permutation_null_gain_njit(x_sorted: np.ndarray, y_compact: np.ndarray, n_classes: int, min_split_size: int, n_permutations: int, seed: int) -> np.ndarray:
    """Shuffle ``y_compact`` ``n_permutations`` times and recompute the best-split gain each time via the
    SAME production kernel (``_mdlp_best_split_njit``) the observed split used. ``x_sorted`` stays fixed
    (only the label assignment is permuted, matching the standard permutation-null construction: the
    marginal distributions of x and y are preserved, only their pairing is randomized). Returns the
    ``(n_permutations,)`` array of null gains (best_gain, possibly -1.0 when no split at all clears
    ``min_split_size``/candidate constraints -- treated as a zero-strength null draw by the caller).
    """
    np.random.seed(seed)
    n = y_compact.shape[0]
    null_gains = np.empty(n_permutations, dtype=np.float64)
    y_perm = y_compact.copy()
    for p in range(n_permutations):
        # Fisher-Yates on a scratch copy -- np.random.shuffle is not always njit-supported across numba
        # versions for arbitrary dtypes, so implement it directly (int64 labels only).
        for i in range(n - 1, 0, -1):
            j = int(np.random.randint(0, i + 1))
            tmp = y_perm[i]
            y_perm[i] = y_perm[j]
            y_perm[j] = tmp
        _, gain, _, _, _ = _mdlp_best_split_njit(x_sorted, y_perm, n_classes, min_split_size)
        null_gains[p] = gain if gain > 0.0 else 0.0
    return null_gains


def _permutation_prefilter_reject(gain: float, n: int, n_classes_full: int) -> bool:
    """Cheap O(1) reject-only shortcut for the permutation-fallback branch, to skip the
    ``n_permutations``-cost shuffle loop (the confirmed cost driver -- 20-80x per column, per the
    original prototype report) when the observed gain is so weak it cannot possibly clear the
    permutation null's acceptance quantile.

    Uses ``analytic_mi_null``'s Miller-Madow ``null_mean = df/(2N)`` -- valid as an absolute bias
    floor for a SINGLE fixed comparison regardless of whether the dense-cell chi-square p-value
    approximation itself applies (same reasoning already used for the OOS variant's absolute floor
    above). The permutation branch's actual null is the STRICTER max-over-``n_candidates``
    statistic, whose mean and quantiles are never below the single-comparison null's (a max over
    >=1 draws from the same distribution stochastically dominates one draw) -- so failing to clear
    even the easier single-comparison floor means the harder max-of-many floor is certainly not
    cleared either. This can only ever turn a permutation-branch REJECT into an early reject
    (never an accept), so it cannot inflate the false-accept rate -- verified to produce IDENTICAL
    accept/reject decisions to the unfiltered path across the adversarial + robustness scenario
    suite in ``_benchmarks/bench_mdlp_prefilter_hybrid.py`` (0 mismatches over 36 scenario x seed
    combinations at n in {1500, 20000}).

    MEASURED IMPACT (honest, not the hoped-for fix): this is safe but NOT the answer to the 20-80x
    permutation-fallback cost problem. Warm-JIT, median-of-3, interleaved-order A/B (see the bench
    module) measured only ~1-7% wall-time reduction, not an order of magnitude -- because the
    observed gain reaching this branch is already the MAX over every class-boundary candidate at
    the node, and that max-selection effect alone usually pushes it well above the single-
    comparison bias floor even under pure noise, so the reject-shortcut fires rarely in practice.
    A real fix for the cost driver still needs the GPU-resident batched-permutation treatment
    recommended in the original prototype report (batch all node-level shuffles into one device-
    resident kernel call, mirroring ``_fe_cmi_perm_null_gpu.py``'s MRMR redundancy-gate design) --
    unimplemented, left as the actual next step. Kept here anyway (never silently reverted) because
    it is unconditionally safe and a genuine, if small, net win.
    """
    if gain <= 0.0:
        return True
    null_mean, _ = analytic_mi_null(float(gain), int(n), 2, int(n_classes_full))
    return gain <= null_mean


def _split_significant(
    gain: float,
    n: int,
    n_classes_full: int,
    x_sorted: np.ndarray,
    y_compact: np.ndarray,
    min_split_size: int,
    alpha: float,
    n_permutations: int,
    seed: int,
    n_candidates: int,
    force_permutation: bool,
) -> "tuple[bool, float, str]":
    """Return ``(accept, p_value, path)`` -- ``path`` is ``'analytic'`` or ``'permutation'`` (diagnostic only).

    ``df = n_classes_full - 1`` because the split partitions rows into 2 groups (``Bx=2``), so
    ``(Bx-1)*(By-1) == n_classes_full - 1``, exactly what ``analytic_mi_null`` expects as
    ``n_bins_x=2, n_bins_y=n_classes_full``.

    SELECTIVE-INFERENCE CORRECTION (found empirically during this prototype's own A/B, not assumed):
    the observed ``gain`` is the MAX over every class-boundary candidate scanned at this node, not a
    single candidate's MI -- ``analytic_mi_null``'s chi-square null is only valid for a SINGLE fixed
    comparison. Feeding it the max-of-many statistic without correction massively inflates the false-
    accept rate (measured: on pure-noise synthetic data, uncorrected produced MORE accepted splits than
    plain unsupervised quantile binning, i.e. worse than doing nothing). Bonferroni-correcting the
    analytic branch by ``n_candidates`` (the number of class-boundary cut points actually scanned at
    this node) restores validity cheaply (still O(1), no extra passes) -- the permutation branch does
    NOT need this since shuffling+re-searching the SAME max-of-candidates statistic each draw already
    builds the correct null of "best split of random data," by construction.

    ``n`` here is safe to use directly as the analytic branch's effective sample size: callers
    dedupe exact ``(x, y)`` duplicates once at the entry point (``_dedupe_xy``, called from both
    ``mdlp_bin_edges_validated`` and the production ``mdlp_bin_edges`` default path) before the
    recursion starts, so no node in the tree ever sees duplicate-row-inflated ``n`` -- see
    ``_dedupe_xy``'s docstring for the duplicate-row robustness bug this closes at the source.
    """
    if not force_permutation and analytic_null_applicable(n, 2, n_classes_full):
        _, p_value_raw = analytic_mi_null(float(gain), int(n), 2, int(n_classes_full))
        p_value = min(1.0, p_value_raw * max(1, int(n_candidates)))
        return (p_value < alpha), p_value, "analytic"
    if _permutation_prefilter_reject(gain, n, n_classes_full):
        return False, 1.0, "permutation_prefiltered"
    null_gains = _permutation_null_gain_njit(x_sorted, y_compact, int(n_classes_full), int(min_split_size), int(n_permutations), int(seed))
    null_gains.sort()
    q_idx = min(math.ceil((1.0 - alpha) * n_permutations) - 1, n_permutations - 1)
    q_idx = max(0, q_idx)
    threshold_gain = float(null_gains[q_idx])
    # Empirical p-value: fraction of null draws >= observed (add-one-smoothed, standard permutation-test estimator).
    p_value = float((np.sum(null_gains >= gain) + 1.0) / (n_permutations + 1.0))
    return (gain > threshold_gain), p_value, "permutation"


def _mdlp_recurse_validated(
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    depth: int,
    min_split_size: int,
    max_depth: int,
    alpha: float,
    n_permutations: int,
    seed: int,
    bonferroni: bool,
    counts_parent: "np.ndarray | None" = None,
    tree_wide_alpha: "float | None" = None,
) -> None:
    """Same recursion shape as ``_mdlp_recurse_njit`` (candidate search unchanged), but the accept test
    is swapped for the significance gate in ``_split_significant`` instead of ``_mdlp_pass_threshold_njit``.

    ``tree_wide_alpha``, when not ``None``, overrides ``bonferroni``'s depth-decay with a single fixed
    alpha shared by every node in the tree (see ``mdlp_bin_edges_validated``'s ``tree_wide_bonferroni``
    for how it's derived) -- mutually exclusive with the depth-decay mode, never both applied at once.
    """
    n = len(x)
    if n < 2 * min_split_size or depth >= max_depth:
        return
    if counts_parent is None:
        present = np.unique(y)
    else:
        present = np.flatnonzero(counts_parent)
    n_classes_full = present.size
    if n_classes_full <= 1:
        return
    if present[0] != 0 or present[-1] != n_classes_full - 1:
        y_compact = np.searchsorted(present, y).astype(np.int64)
    else:
        y_compact = y
    best_idx, best_gain, _h_full, _n_l, _n_r = _mdlp_best_split_njit(x, y_compact, int(n_classes_full), int(min_split_size))
    if best_idx < 0 or best_gain <= 0.0:
        return
    best_split = 0.5 * (x[best_idx] + x[best_idx + 1])
    # Depth-adjusted alpha (Bonferroni over the exponentially-growing candidate-node count per level) is
    # OPTIONAL (default off, benched both ways below) -- classic ChiMerge/Chi2 use a single fixed
    # significance level per decision, not a multiplicity correction across the whole recursion tree.
    # This is SEPARATE from (and stacks with) the mandatory per-node candidate-count Bonferroni inside
    # ``_split_significant`` -- that one corrects for the max-over-cut-points selection effect within a
    # single node and is not optional (see that function's docstring for the empirical justification).
    if tree_wide_alpha is not None:
        _alpha = tree_wide_alpha
    else:
        _alpha = alpha / (2.0**depth) if bonferroni else alpha
    n_candidates = _count_candidates_njit(x, y_compact, min_split_size)
    accept, _p_value, _path = _split_significant(
        best_gain, n, n_classes_full, x, y_compact, min_split_size, _alpha, n_permutations,
        seed + depth * 7919 + int(best_idx), n_candidates, force_permutation=False,
    )
    if not accept:
        return
    left_mask_idx = best_idx + 1
    y_left = y_compact[:left_mask_idx]
    y_right = y_compact[left_mask_idx:]
    counts_left_dense = np.bincount(y_left, minlength=int(n_classes_full)).astype(np.int64)
    counts_right_dense = np.bincount(y_right, minlength=int(n_classes_full)).astype(np.int64)
    splits.append(float(best_split))
    _mdlp_recurse_validated(
        x[:left_mask_idx], y_compact[:left_mask_idx], splits, depth + 1, min_split_size, max_depth,
        alpha, n_permutations, seed, bonferroni, counts_left_dense, tree_wide_alpha,
    )
    _mdlp_recurse_validated(
        x[left_mask_idx:], y_compact[left_mask_idx:], splits, depth + 1, min_split_size, max_depth,
        alpha, n_permutations, seed, bonferroni, counts_right_dense, tree_wide_alpha,
    )


def mdlp_bin_edges_validated(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_split_size: int = 5,
    max_depth: int = 8,
    max_y_classes: int = 64,
    alpha: float = 0.05,
    n_permutations: int = 30,
    bonferroni: bool = False,
    tree_wide_bonferroni: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Significance-gated MDLP (research prototype). Same y-quantization / NaN-handling / dtype
    contract as ``mdlp_bin_edges`` (deliberately duplicated rather than imported, to keep this
    prototype fully separate from the in-flight production file per this session's file-ownership
    convention); only the accept-test inside the recursion differs. Returns sorted edges including
    ``-inf``/``+inf`` sentinels, same as ``mdlp_bin_edges``.

    Args:
        bonferroni: Depth-decay correction ``alpha / 2**depth`` -- assumes a perfectly balanced
            binary tree (exactly ``2**depth`` nodes at depth ``d``), which the data-dependent
            recursion here rarely produces exactly, so it over- or under-corrects depending on
            actual tree shape. Mutually exclusive with ``tree_wide_bonferroni`` (that one wins if
            both are set).
        tree_wide_bonferroni: Data-independent, provably-conservative alternative: divides ``alpha``
            once by ``max(1, n // min_split_size)``, the maximum possible number of internal nodes
            ANY binary tree over ``n`` rows can have when every leaf needs at least ``min_split_size``
            rows (a tree with ``L`` leaves has exactly ``L - 1`` internal nodes, and ``L <=
            n // min_split_size``) -- true regardless of how the actual (data-dependent) recursion
            shape turns out, so this alpha is valid for every node without a two-pass count of the
            real tree. Strictly more conservative than the true tree-wide FWER bound (uses the size
            bound rather than the exact node count actually visited), by design -- see
            ``_benchmarks/bench_mdlp_adversarial_suite.py``'s multi-comparisons-defeat sweep for the
            measured empirical tree-wide false-discovery rate under this mode vs depth-decay vs off.
    """
    x = np.asarray(x).ravel()
    _y_arr = np.asarray(y).ravel()
    if _y_arr.dtype.kind in ("O", "U", "S") or _y_arr.dtype.name in ("category", "string", "object"):
        try:
            import pandas as _pd
            _y_arr, _ = _pd.factorize(_y_arr, sort=True)
        except Exception:
            _uniq, _y_arr = np.unique(_y_arr, return_inverse=True)
    else:
        _y_finite = _y_arr[np.isfinite(_y_arr)] if _y_arr.dtype.kind == "f" else _y_arr
        if _y_finite.size and int(np.unique(_y_finite).size) > int(max_y_classes):
            _q = np.linspace(0.0, 1.0, int(max_y_classes) + 1)[1:-1]
            _y_edges = np.unique(np.quantile(_y_finite, _q))
            _y_arr = np.searchsorted(_y_edges, _y_arr, side="right")
    y_i = _y_arr.astype(np.int64)
    if len(x) != len(y_i):
        raise ValueError(f"len(x)={len(x)} != len(y)={len(y_i)}")
    _finite_mask = np.isfinite(x)
    if not _finite_mask.all():
        x = x[_finite_mask]
        y_i = y_i[_finite_mask]
    if x.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    x, y_i = _dedupe_xy(x, y_i)
    sorter = np.argsort(x)
    x_sorted = np.ascontiguousarray(x[sorter].astype(np.float64))
    y_sorted = np.ascontiguousarray(y_i[sorter])
    splits: list = []
    tree_wide_alpha = float(alpha) / max(1, x_sorted.size // int(min_split_size)) if tree_wide_bonferroni else None
    _mdlp_recurse_validated(
        x_sorted, y_sorted, splits, 0, int(min_split_size), int(max_depth), float(alpha), int(n_permutations), int(seed), bool(bonferroni),
        tree_wide_alpha=tree_wide_alpha,
    )
    splits.sort()
    edges = np.concatenate([[-np.inf], np.asarray(splits, dtype=np.float64), [np.inf]])
    return edges


# =============================================================================
# Genuine out-of-sample (held-out fold) validated splitting -- additional variant,
# NOT the default (see ``mdlp_bin_edges_oos_validated`` docstring for why).
# =============================================================================


def _holdout_gain(x_holdout: np.ndarray, y_holdout_c: np.ndarray, best_split: float, n_classes_full: int, min_split_size: int):
    """Compute the SAME weighted-entropy-reduction statistic as the training scan, but on an
    independent held-out row set, at the FIXED cut point the training portion already chose (no
    re-search here -- that is what makes this a genuine OOS check rather than another in-sample
    max-over-candidates search). Rows whose label was never seen in the training node (``y_holdout_c
    == -1``, stamped by the caller) are dropped -- they carry no information about whether train's
    class-space cut generalizes. Returns ``(gain, n_l, n_r)``; ``gain`` is ``-1.0`` if either side
    is smaller than ``min_split_size`` after dropping unseen-label rows (can't confirm generalization
    on too few held-out rows -> caller treats this as a reject, the conservative choice).
    """
    keep = y_holdout_c >= 0
    if not np.any(keep):
        return -1.0, 0, 0
    xh = x_holdout[keep]
    yh = y_holdout_c[keep]
    left = xh <= best_split
    n_l, n_r = int(left.sum()), int((~left).sum())
    if n_l < min_split_size or n_r < min_split_size:
        return -1.0, n_l, n_r
    counts_total = np.bincount(yh, minlength=n_classes_full).astype(np.int64)
    counts_left = np.bincount(yh[left], minlength=n_classes_full).astype(np.int64)
    counts_right = counts_total - counts_left
    n = n_l + n_r
    h_full = float(_entropy_from_counts_njit(counts_total, n))
    h_left = float(_entropy_from_counts_njit(counts_left, n_l))
    h_right = float(_entropy_from_counts_njit(counts_right, n_r))
    h_split = (n_l * h_left + n_r * h_right) / n
    return h_full - h_split, n_l, n_r


def _mdlp_recurse_oos_validated(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
    splits: list,
    depth: int,
    min_split_size: int,
    max_depth: int,
    oos_tolerance: float,
    counts_parent: "np.ndarray | None" = None,
    present_parent: "np.ndarray | None" = None,
) -> None:
    """A candidate split is found on ``x_train``/``y_train`` exactly as the in-sample path does
    (single best-gain search via ``_mdlp_best_split_njit``), but is only ACCEPTED if the SAME cut
    point, re-evaluated on the independent ``x_holdout``/``y_holdout`` at that node, still shows a
    gain that is both positive and at least ``oos_tolerance`` fraction of the training gain -- i.e.
    the split's apparent value must hold up on data it was not chosen from, not merely look
    significant on the data that produced it. Both train and holdout subsets are partitioned by the
    accepted cut and recursed on independently (holdout rows never influence the NEXT node's search
    either -- generalization is checked fresh at every node, mirroring nested/repeated-holdout CV
    rather than a single train/test split re-used for the whole tree).

    Deliberately implemented as a single train/holdout split rather than k-fold: MDLP's recursion
    already multiplies cost by the number of nodes visited, and repeating the whole recursive search
    k times (or even just re-searching per fold at every node) would multiply the ALREADY 20-80x
    validated-split cost by another factor of k for comparatively little benefit ("fit once, check
    generalization once" is already a meaningfully different bar than the in-sample tests above);
    if this variant proves worth it, k-fold is a natural but separately-costed follow-up.
    """
    n = len(x_train)
    if n < 2 * min_split_size or depth >= max_depth:
        return
    if present_parent is None:
        present = np.unique(y_train)
    else:
        present = present_parent if counts_parent is None else np.flatnonzero(counts_parent)
    n_classes_full = present.size
    if n_classes_full <= 1:
        return
    if present[0] != 0 or present[-1] != n_classes_full - 1:
        y_train_c = np.searchsorted(present, y_train).astype(np.int64)
    else:
        y_train_c = y_train
    best_idx, best_gain, _h_full, _n_l, _n_r = _mdlp_best_split_njit(x_train, y_train_c, int(n_classes_full), int(min_split_size))
    if best_idx < 0 or best_gain <= 0.0:
        return
    best_split = 0.5 * (x_train[best_idx] + x_train[best_idx + 1])

    # Map holdout labels into the SAME class space as train; labels train never saw are stamped -1
    # (dropped by ``_holdout_gain`` -- they carry no evidence about whether train's cut generalizes).
    idx = np.searchsorted(present, y_holdout)
    idx_clipped = np.clip(idx, 0, n_classes_full - 1)
    match = present[idx_clipped] == y_holdout
    y_holdout_c = np.where(match, idx_clipped, -1)

    oos_gain, oos_nl, oos_nr = _holdout_gain(x_holdout, y_holdout_c, best_split, int(n_classes_full), int(min_split_size))
    if oos_gain <= 0.0:
        return
    # BUG FOUND AND FIXED DURING THIS PROTOTYPE'S OWN A/B (2026-07-19): a relative-only bar
    # (``oos_gain >= oos_tolerance * best_gain``) has NO absolute floor, so as recursion deepens
    # and ``best_gain`` shrinks towards the noise scale, an arbitrarily tiny (and therefore easy-
    # to-pass-by-chance) holdout gain clears the bar -- measured to produce MORE splits (~50+) on
    # a 2-true-breakpoint synthetic than the in-sample significance-gated default (which correctly
    # found exactly 2). The analytic Miller-Madow null-mean bias ``(Bx-1)(By-1)/(2*N_holdout)`` --
    # ``analytic_mi_null``'s first return value -- is a valid absolute noise floor REGARDLESS of
    # whether the dense-cell chi-square approximation itself is applicable (the bias term alone
    # does not need the same sparsity/large-n safe condition its p-value does), so requiring
    # BOTH the relative bar and this absolute floor closes the hole without reintroducing a
    # multiple-comparison correction (only ONE fixed cut is being confirmed here, not searched).
    n_oos = oos_nl + oos_nr
    _null_mean, _ = analytic_mi_null(oos_gain, n_oos, 2, int(n_classes_full))
    if oos_gain <= _null_mean or oos_gain < oos_tolerance * best_gain:
        return  # split does not generalize to the held-out fold -> reject regardless of in-sample gain

    left_mask_idx = best_idx + 1
    y_train_left = y_train_c[:left_mask_idx]
    y_train_right = y_train_c[left_mask_idx:]
    counts_left_dense = np.bincount(y_train_left, minlength=int(n_classes_full)).astype(np.int64)
    counts_right_dense = np.bincount(y_train_right, minlength=int(n_classes_full)).astype(np.int64)
    holdout_left = x_holdout <= best_split
    splits.append(float(best_split))
    _mdlp_recurse_oos_validated(
        x_train[:left_mask_idx], y_train_left, x_holdout[holdout_left], y_holdout[holdout_left], splits,
        depth + 1, min_split_size, max_depth, oos_tolerance, counts_left_dense, present,
    )
    _mdlp_recurse_oos_validated(
        x_train[left_mask_idx:], y_train_right, x_holdout[~holdout_left], y_holdout[~holdout_left], splits,
        depth + 1, min_split_size, max_depth, oos_tolerance, counts_right_dense, present,
    )


def mdlp_bin_edges_oos_validated(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_split_size: int = 5,
    max_depth: int = 8,
    max_y_classes: int = 64,
    oos_tolerance: float = 0.3,
    holdout_frac: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Genuine out-of-sample-validated MDLP variant: every candidate split is chosen on a TRAIN
    fold and only accepted if its gain, re-computed on an independent HOLDOUT fold at the SAME cut
    point, is positive and within ``oos_tolerance`` of the training gain. Added specifically to
    check whether true OOS validation catches anything the in-sample significance-gated default
    (``mdlp_bin_edges``/``_mdlp_recurse_validated``) misses -- see the module docstring's
    methodology note and ``_benchmarks/bench_mdlp_validated_split_suite.py`` for the A/B.

    NOT wired into ``mdlp_bin_edges`` -- this is a research variant, evaluated alongside the
    in-sample significance-gated default rather than replacing it (see the bench suite's findings
    for whether it earns production wiring). Same y-quantization / NaN-handling contract as
    ``mdlp_bin_edges`` (duplicated, not imported, for the same file-ownership reason as
    ``mdlp_bin_edges_validated`` above).

    Args:
        oos_tolerance: Minimum fraction of the training gain the SAME cut must reproduce on the
            held-out fold to be accepted. ``0.3`` is a forgiving generalization bar (noise alone
            would rarely reproduce even 30% of an in-sample-optimized gain); raise for a stricter
            criterion.
        holdout_frac: Fraction of rows set aside as the single held-out fold. ``0.3`` balances
            leaving enough rows in each fold as the recursion narrows to small nodes.
    """
    x = np.asarray(x).ravel()
    _y_arr = np.asarray(y).ravel()
    if _y_arr.dtype.kind in ("O", "U", "S") or _y_arr.dtype.name in ("category", "string", "object"):
        try:
            import pandas as _pd
            _y_arr, _ = _pd.factorize(_y_arr, sort=True)
        except Exception:
            _uniq, _y_arr = np.unique(_y_arr, return_inverse=True)
    else:
        _y_finite = _y_arr[np.isfinite(_y_arr)] if _y_arr.dtype.kind == "f" else _y_arr
        if _y_finite.size and int(np.unique(_y_finite).size) > int(max_y_classes):
            _q = np.linspace(0.0, 1.0, int(max_y_classes) + 1)[1:-1]
            _y_edges = np.unique(np.quantile(_y_finite, _q))
            _y_arr = np.searchsorted(_y_edges, _y_arr, side="right")
    y_i = _y_arr.astype(np.int64)
    if len(x) != len(y_i):
        raise ValueError(f"len(x)={len(x)} != len(y)={len(y_i)}")
    _finite_mask = np.isfinite(x)
    if not _finite_mask.all():
        x = x[_finite_mask]
        y_i = y_i[_finite_mask]
    if x.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n = x.size
    perm = rng.permutation(n)
    n_holdout = max(1, round(holdout_frac * n))
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    x_train_raw, y_train_raw = x[train_idx], y_i[train_idx]
    x_holdout, y_holdout = x[holdout_idx], y_i[holdout_idx]

    sorter = np.argsort(x_train_raw)
    x_train_sorted = np.ascontiguousarray(x_train_raw[sorter].astype(np.float64))
    y_train_sorted = np.ascontiguousarray(y_train_raw[sorter])

    splits: list = []
    _mdlp_recurse_oos_validated(
        x_train_sorted, y_train_sorted, np.asarray(x_holdout, dtype=np.float64), y_holdout,
        splits, 0, int(min_split_size), int(max_depth), float(oos_tolerance),
    )
    splits.sort()
    edges = np.concatenate([[-np.inf], np.asarray(splits, dtype=np.float64), [np.inf]])
    return edges
