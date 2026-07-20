"""Supervised discretisation methods (no leak when fit on train, applied to val).

Two backends:

* ``mdlp_bin_edges`` -- Fayyad-Irani 1993 entropy-based recursive splits. No ``n_bins`` hyperparameter; splits are chosen via the MDL
  principle. Simple pure-numpy implementation.
* ``optimal_bin_edges`` -- thin wrapper around ``optbinning.OptimalBinning`` (already a project dep). Production-grade, supports
  monotonic constraints + IV-based feature pre-selection.

Both produce ``bin_edges`` that the downstream ``np.searchsorted`` / ``np.digitize`` chain can consume identically to the unsupervised
paths in ``discretization.py``.

Leak-safe usage pattern::

    edges = mdlp_bin_edges(X_train[:, j], y_train)        # fit on train
    binned_train = np.searchsorted(edges[1:-1], X_train[:, j])
    binned_val   = np.searchsorted(edges[1:-1], X_val[:, j])

The helper does not call ``y`` on the val rows, so passing the same edges to both train and val is leak-safe.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


def mdlp_bin_edges(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_split_size: int = 5,
    max_depth: int = 8,
    backend: str = "njit",
    scaled_min_split: bool = False,
    max_y_classes: int = 64,
    fast_mode: bool = False,
    alpha: float = 0.05,
    n_permutations: int = 30,
    bonferroni: bool = False,
    validated_seed: int = 0,
) -> np.ndarray:
    """Fayyad-Irani MDLP discretisation. Returns sorted bin edges (includes ``-inf`` / ``+inf`` sentinels).

    Algorithm (recursive):
    1. Sort ``x``, with target ``y`` aligned.
    2. Find candidate split point that maximally reduces conditional entropy ``H(y | x <= split) + H(y | x > split)``.
    3. Accept the split via a per-node significance gate (DEFAULT) or the classic Fayyad-Irani
       1993 MDL threshold test (``fast_mode=True``) -- see ``fast_mode`` below.
    4. Recurse on each half.

    Args:
        x: 1-D continuous feature.
        y: 1-D class labels (int, string, or continuous -- see ``max_y_classes``).
        min_split_size: Absolute floor on samples per child node. Default 5
            mirrors pre-fix behaviour. Pass via ``scaled_min_split=True`` to
            additionally scale to ``max(min_split_size, int(0.02 * N))``.
        max_depth: Recursion cap; splitting stops earlier in practice.
        backend: ``'python'`` (legacy pure-numpy recursion; ~96 ms / col at n=2000),
            ``'njit'`` (audit recommendation; ports the candidate-scan +
            entropy hot loop into ``_mdlp_best_split`` njit kernel - targets
            10-30x total speedup, recursion bookkeeping stays Python). Only
            consulted when ``fast_mode=True``; the default validated-splitting
            path always uses its own njit kernel regardless of ``backend``.
        scaled_min_split: Audit recommendation - lift ``min_split_size`` floor
            to ``max(5, int(0.02 * N))`` so the constraint scales with N.
            Default ``False`` preserves the bench baseline.
        max_y_classes: cap on distinct ``y`` values before MDLP treats them as class
            labels. 2026-07-19 wellbore-50k profiling fix: a regression target (e.g.
            depth-like ``TVT``, tens of thousands of near-unique float values) passed
            straight through used to be silently ``.astype(np.int64)``-truncated into
            thousands of spurious "classes" -- one per truncated value. Two compounding
            bugs resulted: (a) ``3.0**n_classes_full`` in the MDL threshold test
            overflows to ``inf`` for n_classes_full above ~650, so ``best_gain/log2 >
            threshold`` is NEVER true -- every split is silently rejected at the root,
            producing empty edges (caller falls back to plain quantile binning anyway);
            (b) the O(n_classes) entropy/count arrays in ``_mdlp_best_split_njit`` are
            rebuilt at every class-boundary candidate along the O(n) scan, so a column
            with ~2500 spurious classes measured ~190x slower than the same column with
            a real few-class target (1.05s vs ~5.6ms) -- the dominant new hotspot in the
            wellbore-50k profile (``categorize_dataset`` ~50s of a 432.9s fit, of which
            isolated few-class MDLP itself is only ~2.9s). Above ``max_y_classes``
            distinct values, ``y`` is quantile-discretized into ``max_y_classes``
            pseudo-classes before MDLP runs -- restores both a bounded-cost recursion
            and an actually-meaningful (non-empty) supervised split, since MDL entropy
            over ~64 pseudo-classes is a sane classification-style signal instead of
            per-row noise. Below the cap (real classification targets: binary, few-way,
            small ordinal) behaviour is unchanged bit-for-bit -- this is the y-analog of
            the existing high-cardinality-x safety caps elsewhere in this module. When
            the quantization branch engages, ``max_depth`` is additionally capped to
            ``ceil(log2(max_y_classes))`` in ``fast_mode`` ONLY (see ``fast_mode`` below)
            -- an uncapped depth measured WORSE held-out RMSE on the real wellbore target
            (0.5553 vs 0.4333 baseline) while costing 3.6x more wall time in that classic
            path, i.e. the extra x-resolution overfits pseudo-class boundaries rather
            than adding real signal.
        fast_mode: ``False`` (DEFAULT, 2026-07-19 user decision -- accuracy over speed
            per project convention): route splitting through
            ``_mdlp_validated_split._mdlp_recurse_validated``, which gates every
            candidate split by a statistical-significance test (an analytic chi-square/
            G-test null reusing ``_analytic_mi_null.analytic_mi_null`` when the node is
            large/dense enough, else an actual permutation-null loop) INSTEAD OF the
            in-sample Fayyad-Irani MDL threshold + fixed depth cap. This is NOT a free
            upgrade: it measured 20-80x slower per column than ``fast_mode=True`` in
            isolated A/B (see ``_mdlp_validated_split.py`` and
            ``_benchmarks/bench_mdlp_validated_split_ab.py`` /
            ``bench_mdlp_validated_split_suite.py``), because the permutation-null
            fallback re-runs the full candidate scan ``n_permutations`` times per
            small/sparse node. Measured accuracy win on the real wellbore target: GR
            450.4 vs 491.1 RMSE, GR_diff_5 576.0 vs 585.8 RMSE (both vs the
            ``fast_mode=True`` depth-capped path); pure-noise synthetic data confirmed
            it does NOT over-split (matches the capped baseline's 1-bin collapse). Pass
            ``fast_mode=True`` (reachable from
            ``MRMR(nbins_strategy_kwargs={"mdlp_fast_mode": True})`` or
            ``train_mlframe_models_suite(mrmr_kwargs={"nbins_strategy_kwargs":
            {"mdlp_fast_mode": True}})``) to opt back into the cheap depth-capped
            classic path for a specific run where wall-time matters more than the
            measured accuracy win. A GPU-resident batched permutation-null (mirroring
            ``_fe_cmi_perm_null_gpu.py``'s design for the MRMR redundancy gate) is a
            filed, NOT-YET-DONE follow-up that would likely close most of the 20-80x
            gap; it does not gate this default flip.
        alpha: Significance level for the validated-split accept test (default path
            only). Default ``0.05``.
        n_permutations: Permutation-null draws for the small/sparse-node fallback
            (default path only). Default ``30``.
        bonferroni: Extra depth-wise ``alpha / 2**depth`` correction on top of the
            (always-applied) per-node candidate-count correction (default path only).
            Default ``False`` -- classic ChiMerge/Chi2 use one fixed significance level
            per decision, not a whole-tree multiplicity correction; benched both ways,
            no consistent accuracy difference observed, left off to match that convention.
        validated_seed: RNG seed for the permutation-null fallback (default path only).
    """
    x = np.asarray(x).ravel()
    # 2026-05-30 Wave 9.1 fix (loop iter 50): handle non-numeric y
    # dtypes (string / object / pandas Categorical / pandas StringDtype).
    # Pre-fix the raw ``.astype(np.int64)`` cast crashed with
    # ``ValueError: invalid literal for int() with base 10: 'yes'`` on
    # any classifier user passing string labels (the standard sklearn
    # convention) or pandas Categorical (LabelEncoder / .astype('category')
    # output). Error pointed into MDLP internals, not into y dtype, so
    # caller debugging was hard. MDLP only needs class-IDENTITY (label
    # equality at split-purity computation), so factorize is sufficient
    # and order-preserving for already-integer-encoded inputs.
    _y_arr = np.asarray(y).ravel()
    if _y_arr.dtype.kind in ("O", "U", "S") or _y_arr.dtype.name in (
        "category", "string", "object",
    ):
        try:
            import pandas as _pd_iter50
            _y_arr, _ = _pd_iter50.factorize(_y_arr, sort=True)
        except Exception:
            # Fallback: numpy-only label encode via unique.
            _uniq, _y_arr = np.unique(_y_arr, return_inverse=True)
    else:
        # Continuous / high-cardinality numeric y (typically a regression target):
        # quantile-discretize into a bounded number of pseudo-classes BEFORE the
        # int64 truncation below, which would otherwise fabricate one spurious
        # "class" per distinct truncated value (see ``max_y_classes`` docstring).
        _y_finite = _y_arr[np.isfinite(_y_arr)] if _y_arr.dtype.kind == "f" else _y_arr
        if _y_finite.size and int(np.unique(_y_finite).size) > int(max_y_classes):
            _q = np.linspace(0.0, 1.0, int(max_y_classes) + 1)[1:-1]
            _y_edges = np.unique(np.quantile(_y_finite, _q))
            _y_arr = np.searchsorted(_y_edges, _y_arr, side="right")
            # 2026-07-19: this blunt depth cap now applies ONLY when ``fast_mode=True``.
            # The DEFAULT path (``fast_mode=False``) replaces the depth heuristic with a
            # per-split significance gate (``_mdlp_recurse_validated``) that decides, per
            # COLUMN and per NODE, whether the extra x-resolution reflects real signal
            # instead of applying one fixed ratio to every column regardless of its
            # actual signal depth -- measured to both reject pure-noise splits (matches
            # the capped baseline, 1 bin) AND recover real signal the uniform cap
            # truncates (real wellbore GR: 450.4 vs 491.1 RMSE; GR_diff_5: 576.0 vs
            # 585.8 RMSE -- see ``_mdlp_validated_split.py`` module docstring).
            if fast_mode:
                max_depth = min(int(max_depth), max(1, math.ceil(math.log2(int(max_y_classes)))))
    if len(x) != len(_y_arr):
        raise ValueError(f"len(x)={len(x)} != len(y)={len(_y_arr)}")
    if scaled_min_split:
        min_split_size = max(int(min_split_size), int(0.02 * len(x)))

    # 2026-05-30 Wave 9.1 fix (loop iter 48): drop NaN rows BEFORE
    # sorting. Pre-fix ``np.argsort(x)`` placed NaN at the tail; the
    # njit recursion then sliced ``x[best_idx+1:]`` and could include
    # the NaN tail, poisoning subsequent ``x[best_idx] + x[best_idx+1]``
    # midpoints and emitting literal NaN values into the ``splits``
    # list. Final ``edges = [-inf, ...sorted_splits..., +inf]`` then
    # contained NaN in the middle, violating the
    # ``np.searchsorted``-relies-on-sorted invariant downstream.
    # python backend and njit backend disagreed silently on NaN-bearing
    # inputs - docstring promises identical semantics. Filter NaN out
    # of both inputs once at the entry point (single-source-of-truth).
    #
    # mrmr_audit_2026-07-20 B-13: this mask used to check ONLY ``x``. When ``y`` is a continuous
    # target with too FEW distinct finite values to trigger the quantile-rebucketing branch above
    # (so ``_y_arr`` still holds raw floats here), a NaN in ``y`` survived this filter and then
    # ``.astype(np.int64)`` below turned it into a platform-defined garbage class label (typically
    # INT64_MIN) instead of being dropped or raising -- a phantom "class" silently polluting every
    # entropy computation downstream. Fold ``y``'s finiteness into the SAME mask (only meaningful
    # when ``_y_arr`` is still float; the factorize/quantise branches above never leave a NaN in an
    # already-integer ``_y_arr``).
    _finite_mask = np.isfinite(x)
    if _y_arr.dtype.kind == "f":
        _finite_mask &= np.isfinite(_y_arr)
    if not _finite_mask.all():
        x = x[_finite_mask]
        _y_arr = _y_arr[_finite_mask]
    y = _y_arr.astype(np.int64)
    if x.size == 0:
        # All-NaN / all-non-finite input: no splits computable.
        return np.array([-np.inf, np.inf], dtype=np.float64)

    sorter = np.argsort(x)
    x_sorted = np.ascontiguousarray(x[sorter].astype(np.float64))
    y_sorted = np.ascontiguousarray(y[sorter])

    splits: list[float] = []
    if fast_mode:
        # Explicit speed opt-out: the pre-2026-07-19 classic Fayyad-Irani accept test
        # (in-sample MDL threshold + the depth-cap guardrail above), unchanged.
        if backend == "njit":
            _mdlp_recurse_njit(x_sorted, y_sorted, splits, depth=0, min_split_size=int(min_split_size), max_depth=int(max_depth))
        else:
            _mdlp_recurse(x_sorted, y_sorted, splits, depth=0, min_split_size=int(min_split_size), max_depth=int(max_depth))
    else:
        # DEFAULT (2026-07-19, user decision -- accuracy over speed per project convention):
        # significance-gated splitting (``_mdlp_validated_split.py``). Lazy import to avoid a
        # module-load-order circular import (that module imports ``_mdlp_best_split_njit`` /
        # ``_entropy_from_counts_njit`` from HERE at ITS top level; by the time this function
        # runs, this module has finished loading, so the import is safe -- doing it at this
        # module's own top level would not be).
        from ._mdlp_validated_split import _dedupe_xy, _mdlp_recurse_validated

        # Exact-duplicate (x, y) rows must be collapsed BEFORE the significance-gated recursion --
        # see ``_dedupe_xy``'s docstring for the measured over-splitting artifact this prevents
        # (duplicated rows sit x-adjacent with an identical y, which a permutation null built by
        # shuffling y never reproduces, so the observed gain spuriously clears the test).
        x_dedup, y_dedup = _dedupe_xy(x_sorted, y_sorted)
        _mdlp_recurse_validated(
            x_dedup, y_dedup, splits, 0, int(min_split_size), int(max_depth),
            float(alpha), int(n_permutations), int(validated_seed), bool(bonferroni),
        )
    splits.sort()
    edges = np.concatenate([[-np.inf], np.asarray(splits, dtype=np.float64), [np.inf]])
    return edges


def _entropy_from_labels(y: np.ndarray) -> float:
    """Shannon entropy of label distribution in nats."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


# =============================================================================
# njit-accelerated hot loop (audit recommendation - 10-30x speedup target)
# =============================================================================


@njit(nogil=True, cache=True)
def _entropy_from_counts_njit(counts: np.ndarray, n: int) -> float:
    """Shannon entropy in nats from already-tabulated label counts."""
    if n <= 0:
        return 0.0
    h = 0.0
    n_f = float(n)
    for k in range(counts.shape[0]):
        c = counts[k]
        if c > 0:
            p = c / n_f
            h -= p * math.log(p)
    return h


@njit(nogil=True, cache=True)
def _mdlp_best_split_njit(x_sorted: np.ndarray, y_sorted: np.ndarray, n_classes: int, min_split_size: int):
    """Find Fayyad-Irani best split on already-sorted (x, y).

    Single forward pass over class-boundary candidates using running label
    counts; entropy at each candidate computed in O(K_y) via cumulative
    counts. Total cost O(N + |candidates| * K_y) vs Python's O(|candidates| * N * K_y).

    Returns: (best_idx, best_gain, h_full, n_left, n_right). ``best_idx`` is
    the array index such that the split point is ``(x[idx] + x[idx+1]) / 2``;
    ``-1`` if no acceptable split found.
    """
    n = x_sorted.shape[0]
    if n < 2 * min_split_size:
        return -1, -1.0, 0.0, 0, 0
    counts_total = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        counts_total[y_sorted[i]] += 1
    h_full = _entropy_from_counts_njit(counts_total, n)
    if h_full <= 0.0:
        return -1, -1.0, h_full, 0, 0
    # Forward scan: maintain running counts of left side.
    counts_left = np.zeros(n_classes, dtype=np.int64)
    counts_right = counts_total.copy()
    best_gain = -1e300
    best_idx = -1
    best_nl = 0
    best_nr = 0
    for i in range(n - 1):
        y_i = y_sorted[i]
        counts_left[y_i] += 1
        counts_right[y_i] -= 1
        n_l = i + 1
        n_r = n - n_l
        if n_l < min_split_size or n_r < min_split_size:
            continue
        # Candidate split only at class boundaries (Fayyad-Irani 1993 theorem).
        if y_sorted[i] == y_sorted[i + 1]:
            continue
        # Also skip if x values are equal (can't split between identical x).
        if x_sorted[i] == x_sorted[i + 1]:
            continue
        h_l = _entropy_from_counts_njit(counts_left, n_l)
        h_r = _entropy_from_counts_njit(counts_right, n_r)
        h_split = (n_l * h_l + n_r * h_r) / n
        gain = h_full - h_split
        if gain > best_gain:
            best_gain = gain
            best_idx = i
            best_nl = n_l
            best_nr = n_r
    return best_idx, best_gain, h_full, best_nl, best_nr


@njit(nogil=True, cache=True)
def _mdlp_pass_threshold_njit(
    best_gain: float, h_full: float, n: int, n_classes_full: int, n_classes_left: int, n_classes_right: int, h_left: float, h_right: float
) -> bool:
    """Fayyad-Irani 1993 MDL acceptance test.

    Returns True iff the split's information gain exceeds the MDL threshold.
    """
    if n <= 1:
        return False
    delta_arg = (3.0**n_classes_full) - 2.0
    if delta_arg <= 0.0:
        return False
    log2 = math.log(2.0)
    delta = (math.log(delta_arg) / log2) - (n_classes_full * h_full - n_classes_left * h_left - n_classes_right * h_right) / log2
    threshold = (math.log(float(n - 1)) / log2 + delta) / n
    return (best_gain / log2) > threshold


def _mdlp_recurse_njit(
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    depth: int,
    min_split_size: int,
    max_depth: int,
    counts_parent: "np.ndarray | None" = None,
) -> None:
    """njit-backed MDLP recursion. Same semantics as ``_mdlp_recurse``; just
    routes the hot loop through ``_mdlp_best_split_njit`` + entropy njit kernels.

    ``counts_parent``: the dense bincount of THIS subset the parent already built for its MDL test
    (``None`` at the root). The present class labels are its non-zero bins, so at every deeper node the
    class set + count come from ``np.flatnonzero`` (O(K)) instead of an ``np.unique`` re-sort of the
    O(m) subset -- bit-identical (``flatnonzero(bincount) == np.unique`` for the parent-compacted labels).
    """
    n = len(x)
    if n < 2 * min_split_size or depth >= max_depth:
        return
    # Class labels present in this subset (root: sort-unique y; deeper: the parent's bincount non-zeros).
    if counts_parent is None:
        present = np.unique(y)
    else:
        present = np.flatnonzero(counts_parent)
    n_classes_full = present.size
    if n_classes_full <= 1:
        return
    # Map y to dense [0, K) for the njit kernel.
    if present[0] != 0 or present[-1] != n_classes_full - 1:
        y_compact = np.searchsorted(present, y).astype(np.int64)
    else:
        y_compact = y
    best_idx, best_gain, h_full, n_l, n_r = _mdlp_best_split_njit(x, y_compact, int(n_classes_full), int(min_split_size))
    if best_idx < 0 or best_gain <= 0.0:
        return
    best_split = 0.5 * (x[best_idx] + x[best_idx + 1])
    # Compute left/right entropies for the MDL test (njit recompute is cheap).
    # bench-attempt-rejected (2026-07-05): fusing counts_left/right + h_l/h_r OUT of _mdlp_best_split_njit
    # (snapshot at the winning boundary, return them) to skip these np.bincount x2 + entropy x2 measured
    # 0.81x SLOWER -- the per-best-update counts.copy() (O(K), fires on every gain improvement in the scan)
    # + returning two njit arrays cost more than the two vectorised-C bincounts they remove. Kept.
    left_mask_idx = best_idx + 1  # x[:left_mask_idx] is left, x[left_mask_idx:] is right
    y_left = y_compact[:left_mask_idx]
    y_right = y_compact[left_mask_idx:]
    counts_left_dense = np.bincount(y_left, minlength=int(n_classes_full)).astype(np.int64)
    counts_right_dense = np.bincount(y_right, minlength=int(n_classes_full)).astype(np.int64)
    # Distinct-class count == number of non-zero bins in the bincount just built, so derive it from
    # ``counts`` (an O(n_classes) scan, K ~ 2-20) instead of a separate ``np.unique`` sort of the
    # O(m) subset per recursion node. Bit-identical: y is compacted to dense [0, K) so every present
    # class occupies exactly one bin.
    n_classes_left = int(np.count_nonzero(counts_left_dense))
    n_classes_right = int(np.count_nonzero(counts_right_dense))
    h_left = float(_entropy_from_counts_njit(counts_left_dense, int(n_l)))
    h_right = float(_entropy_from_counts_njit(counts_right_dense, int(n_r)))
    if not _mdlp_pass_threshold_njit(float(best_gain), float(h_full), int(n), int(n_classes_full), int(n_classes_left), int(n_classes_right), h_left, h_right):
        return
    splits.append(float(best_split))
    _mdlp_recurse_njit(x[:left_mask_idx], y_compact[:left_mask_idx], splits, depth + 1, min_split_size, max_depth, counts_left_dense)
    _mdlp_recurse_njit(x[left_mask_idx:], y_compact[left_mask_idx:], splits, depth + 1, min_split_size, max_depth, counts_right_dense)


# =============================================================================
# Legacy pure-Python recursion (kept for A/B testing the njit port)
# =============================================================================


def _mdlp_recurse(
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    depth: int,
    min_split_size: int,
    max_depth: int,
) -> None:
    """Legacy pure-Python MDLP recursive split search, kept only as an A/B reference for validating ``_mdlp_recurse_njit``'s bit-identity; not on the production path."""
    n = len(x)
    if n < 2 * min_split_size or depth >= max_depth:
        return
    if len(np.unique(y)) <= 1:
        return  # already pure

    # Find candidate splits at class-boundary midpoints.
    boundary_idx = np.where(y[:-1] != y[1:])[0]
    if len(boundary_idx) == 0:
        return
    candidates = (x[boundary_idx] + x[boundary_idx + 1]) / 2.0
    candidates = np.unique(candidates)

    h_full = _entropy_from_labels(y)
    best_gain = -np.inf
    best_split = None
    best_left_idx = None

    for split in candidates:
        left_mask = x <= split
        if left_mask.sum() < min_split_size or (~left_mask).sum() < min_split_size:
            continue
        y_left = y[left_mask]
        y_right = y[~left_mask]
        h_left = _entropy_from_labels(y_left)
        h_right = _entropy_from_labels(y_right)
        n_l = len(y_left)
        n_r = len(y_right)
        h_split = (n_l * h_left + n_r * h_right) / n
        gain = h_full - h_split
        if gain > best_gain:
            best_gain = gain
            best_split = split
            best_left_idx = left_mask

    if best_split is None or best_gain <= 0:
        return
    assert best_left_idx is not None  # set together with best_split whenever gain > best_gain (initial -inf)

    # MDL stopping criterion (Fayyad-Irani 1993).
    n_classes_full = len(np.unique(y))
    n_classes_left = len(np.unique(y[best_left_idx]))
    n_classes_right = len(np.unique(y[~best_left_idx]))
    delta = np.log2(3**n_classes_full - 2) - (
        n_classes_full * h_full - n_classes_left * _entropy_from_labels(y[best_left_idx]) - n_classes_right * _entropy_from_labels(y[~best_left_idx])
    ) / np.log(2.0)
    threshold = (np.log2(n - 1) + delta) / n
    if best_gain / np.log(2.0) <= threshold:
        return  # MDL says stop

    splits.append(float(best_split))
    _mdlp_recurse(x[best_left_idx], y[best_left_idx], splits, depth + 1, min_split_size, max_depth)
    _mdlp_recurse(x[~best_left_idx], y[~best_left_idx], splits, depth + 1, min_split_size, max_depth)


def optimal_bin_edges(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_n_bins: int = 10,
    monotonic_trend: str = "auto",
) -> np.ndarray:
    """Wrapper around ``optbinning.OptimalBinning``. Returns sorted bin edges with ``-inf / +inf`` sentinels.

    Parameters
    ----------
    monotonic_trend
        ``"auto"`` lets optbinning pick. Pass ``"ascending"`` / ``"descending"`` / ``None`` to override.

    Notes
    -----
    Optbinning is an existing project dep. Pricier than MDLP (~0.5s per column on n=10000) but produces monotonic bins which downstream
    GBM models prefer.

    **Compatibility note**: optbinning < 0.21 calls the sklearn API ``check_array(force_all_finite=...)`` which was removed in
    sklearn 1.6 (renamed to ``ensure_all_finite``). The project now requires ``optbinning>=0.21`` (uses ``ensure_all_finite``),
    so this combination works out of the box. If you somehow run optbinning 0.20.x against sklearn 1.6+ and hit
    ``TypeError: check_array() got an unexpected keyword argument 'force_all_finite'``, upgrade optbinning (``pip install 'optbinning>=0.21'``).
    """
    try:
        from optbinning import OptimalBinning
    except ImportError as e:
        raise ImportError("optimal_bin_edges requires the `optbinning` package. " "Install via `pip install optbinning`.") from e

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ob = OptimalBinning(
        name="x",
        dtype="numerical",
        max_n_bins=max_n_bins,
        monotonic_trend=monotonic_trend,
    )
    ob.fit(x, y)
    inner = ob.splits
    edges = np.concatenate([[-np.inf], np.asarray(inner, dtype=np.float64), [np.inf]])
    return edges


def apply_bin_edges(
    x: np.ndarray,
    edges: np.ndarray,
    dtype: type | None = None,
) -> np.ndarray:
    """Apply pre-fit bin edges to discretise an array.

    Leak-safe: edges are computed once on train and used on both train + val without re-fitting.

    2026-05-30 Wave 9.1 fix (loop iter 47):
    1. Auto-pick dtype based on the actual bin count so int8's 128-bin
       ceiling never silently wraps. Pre-fix default ``dtype=np.int8``
       silently overflowed once ``len(edges) >= 128`` (multi-quantile /
       cross-feature joint binning routinely produces > 128 edges),
       returning monotonic-looking but wrong-class codes. Pass an
       explicit narrower dtype to opt back into the legacy behaviour
       (now warns on overflow).
    2. Route NaN inputs to a dedicated ``-1`` sentinel (or out-of-range
       slot depending on caller) instead of silently aliasing them with
       the highest finite bin. ``np.searchsorted(edges, NaN,
       side='right')`` returns ``len(edges)`` (= top bin index after
       slicing) - NaN rows silently merged with the largest valid bin.
    """
    n_codes = int(np.asarray(edges).size) - 1  # bins, not edges
    n_codes = max(1, n_codes)
    if dtype is None:
        # Auto-pick the smallest int dtype that holds [0, n_codes].
        # Include +1 headroom for an optional NaN sentinel at n_codes.
        _max_code = n_codes
        if _max_code < 127:
            dtype = np.int8
        elif _max_code < 32767:
            dtype = np.int16
        else:
            dtype = np.int32
    else:
        # Caller forced a dtype - validate it can hold n_codes.
        try:
            _info_max = int(np.iinfo(np.dtype(dtype)).max)  # type: ignore[type-var]  # caller may pass a non-integer dtype; guarded by the except below
        except (TypeError, ValueError):
            _info_max = None
        if _info_max is not None and n_codes > _info_max:
            raise ValueError(
                f"apply_bin_edges: n_codes={n_codes} exceeds caller-"
                f"forced dtype {dtype}'s range (max {_info_max}); "
                f"silent overflow would produce wrong class codes. "
                f"Pass a wider dtype or omit dtype= to auto-pick."
            )
    arr = np.asarray(x, dtype=np.float64)
    _nan_mask = np.isnan(arr) if arr.dtype.kind == "f" else None
    codes: np.ndarray = np.searchsorted(edges[1:-1], arr, side="right").astype(dtype)
    if _nan_mask is not None and _nan_mask.any():
        # NaN rows get a dedicated sentinel at n_codes (one past max
        # real bin). Caller's downstream code needs to treat this
        # sentinel as NaN; the alternative (silent aliasing) was
        # documented behaviour but produced wrong MI / WoE.
        codes[_nan_mask] = n_codes
    return codes
