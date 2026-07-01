"""Multi-dimensional weak-slice finder.

Goes beyond the 1-2-feature weak-segment heatmap (``error_analysis.weak_segment_heatmap``):
it enumerates feature-value SLICES of 1-3 binned features and ranks them by how badly the
model underperforms there, weighted by how much data the slice covers. The headline output is
a ranked table of the worst slices plus a horizontal bar panel (worst slice on top, each bar
annotated with its support), so a reviewer sees "the model is worst on age in [60,75] AND
region=NW, covering 4% of rows, mean error 3.1x global" at a glance.

Ranking score per slice::

    score = (slice_mean_error - global_mean_error) * sqrt(support_fraction)

The ``sqrt`` support weight (vs raw support) keeps a small-but-catastrophic slice competitive
with a large-but-mildly-worse one -- a slice that is 5x worse over 1% of rows is as actionable
as one 1.5x worse over 9%. Slices below ``min_support_fraction`` are dropped as noise.

EFFICIENCY: features are quantile-binned ONCE into an (n, n_features) int8 code matrix. Every
candidate slice's mean error + support comes from a single ``np.bincount`` over the combined
linear bin code with ``weights=per_row_error`` -- aggregate-first, no nested python loop over
rows. The combinatorial search is capped: all singletons + all 2-feature pairs by default,
3-way slices only among the top ``three_way_top_features`` most-error-discriminating features,
and the whole pair/triple enumeration is bounded by ``max_combos`` (what gets capped is logged).
The spec carries only the worst-K aggregated rows, never length-n arrays.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.charts.error_analysis import (
    _per_row_error, _resolve_feature_matrix, _top_split_features,
)
from mlframe.reporting.spec import BarPanelSpec, FigureSpec

logger = logging.getLogger(__name__)

try:
    import numba

    @numba.njit(cache=True, fastmath=False)
    def _fused_sum_count(flat: np.ndarray, err: np.ndarray, ncells: int):
        sums = np.zeros(ncells, dtype=np.float64)
        counts = np.zeros(ncells, dtype=np.float64)
        for i in range(flat.shape[0]):
            c = flat[i]
            sums[c] += err[i]
            counts[c] += 1.0
        return sums, counts

    @numba.njit(cache=True, fastmath=False)
    def _fused_sum_count_2col(c0: np.ndarray, c1: np.ndarray, stride0: int, err: np.ndarray, ncells: int):
        # Arity-2 fast path (the dominant slice-finder regime: thousands of feature pairs). Fuses the mixed-radix
        # flatten ``c0*stride0 + c1`` into the same row pass as the per-cell (sum_error, count) reduction, so the
        # length-n ``flat`` int64 array is never materialised. The two code columns are passed pre-gathered and
        # contiguous (1D, cache-friendly) by the caller, so this avoids the strided 2D ``codes[i, feat]`` access that
        # regressed a fully-fused variant. Accumulation is row order -> float64 sums bit-identical to the bincount path.
        sums = np.zeros(ncells, dtype=np.float64)
        counts = np.zeros(ncells, dtype=np.float64)
        for i in range(c0.shape[0]):
            c = c0[i] * stride0 + c1[i]
            sums[c] += err[i]
            counts[c] += 1.0
        return sums, counts

    @numba.njit(parallel=True, cache=True, fastmath=False)
    def _batched_pair_sum_count(codes_t: np.ndarray, err: np.ndarray, f0_arr: np.ndarray,
                                f1_arr: np.ndarray, stride0_arr: np.ndarray, max_cells: int):
        # One parallel pass over ALL feature pairs (the dominant slice-finder regime: thousands of pairs, each an
        # independent O(n) reduction that the serial loop ran one-at-a-time on a single core -> "CPU not loaded").
        # ``codes_t`` is the transposed (p, n) C-contiguous code matrix so ``codes_t[f]`` is a contiguous row -> the
        # inner i-loop is cache-friendly. Each pair writes only its own output row, so prange is race-free. The cell
        # flatten ``c0*stride0 + c1`` and row-order accumulation are identical to ``_fused_sum_count_2col``, so the
        # per-pair sums/counts are bit-identical to the serial path.
        npairs = f0_arr.shape[0]
        n = err.shape[0]
        sums = np.zeros((npairs, max_cells), dtype=np.float64)
        counts = np.zeros((npairs, max_cells), dtype=np.float64)
        for pidx in numba.prange(npairs):
            row0 = codes_t[f0_arr[pidx]]
            row1 = codes_t[f1_arr[pidx]]
            s0 = stride0_arr[pidx]
            for i in range(n):
                c = row0[i] * s0 + row1[i]
                sums[pidx, c] += err[i]
                counts[pidx, c] += 1.0
        return sums, counts

    _HAS_NUMBA_SLICE = True
except Exception:  # numba unavailable: fall back to the two-bincount numpy path.
    _HAS_NUMBA_SLICE = False

    def _fused_sum_count(flat, err, ncells):  # type: ignore[misc]
        counts = np.bincount(flat, minlength=ncells).astype(np.float64)
        sums = np.bincount(flat, weights=err, minlength=ncells)
        return sums, counts

    def _fused_sum_count_2col(c0, c1, stride0, err, ncells):  # type: ignore[misc]
        flat = c0 * stride0 + c1
        counts = np.bincount(flat, minlength=ncells).astype(np.float64)
        sums = np.bincount(flat, weights=err, minlength=ncells)
        return sums, counts

# Quantile bins per feature: 4 keeps each 1-feature slice coarse (quartiles) so a slice stays a readable, actionable
# region rather than a noisy single-value cell; the 2-3-way product of these stays small enough to bincount cheaply.
DEFAULT_NBINS: int = 4
# A slice covering less than this fraction of rows is statistical noise, not an actionable weak region.
DEFAULT_MIN_SUPPORT_FRACTION: float = 0.01
# How many worst slices to surface in the table + bar panel.
DEFAULT_TOP_K: int = 7
# Pair / triple enumeration is bounded by this; beyond it lower-arity feature pairs are kept and the rest are dropped
# (logged). At the default 4 bins a 2-way slice grid is 16 cells, so 5000 combos covers ~300 feature pairs.
DEFAULT_MAX_COMBOS: int = 5_000
# 3-way slices only among this many top error-discriminating features (3-way over all features explodes combinatorially).
DEFAULT_THREE_WAY_TOP_FEATURES: int = 6
# Cap on the transient transposed code-matrix copy used to batch the pair search across cores. Above it, fall back to
# the serial per-pair path rather than risk doubling RAM on a huge diagnostic sample (the code matrix is int64 n x p).
_SLICE_TRANSPOSE_MAX_BYTES: int = 4 * 1024 ** 3


@dataclass(frozen=True)
class SliceFinderResult:
    """Ranked worst-slice table + the bar panel + the global reference + what the search capped.

    ``table`` columns: ``features`` (tuple of feature names), ``bounds`` (human-readable per-feature range/value),
    ``mean_error``, ``support`` (row count), ``support_fraction``, ``error_ratio`` (slice / global), ``score``.
    ``worst_slice`` is the top row's (features, bounds, mean_error, support). ``capped`` lists any search reductions
    (e.g. "3-way restricted to top 6 features", "pair enumeration truncated at 5000 combos").
    """

    figure: FigureSpec
    table: Any  # pandas.DataFrame
    global_error: float
    worst_slice: Tuple[Tuple[str, ...], str, float, int]
    capped: Tuple[str, ...] = field(default_factory=tuple)


def _bin_matrix(mat: np.ndarray, nbins: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Quantile-bin every column once into an (n, n_features) int64 code matrix + the per-feature edge arrays.

    Each column's edges are equal-frequency quantiles (deduped); a constant / degenerate column collapses to a single
    bin (code 0). Codes are computed with one vectorised ``np.searchsorted`` per column -- no per-row python work.
    Codes are int64 (not int16) so the per-combo mixed-radix flatten reuses them without a length-n re-cast per slice.
    Stored column-major (Fortran order) so each ``codes[:, j]`` column slice is already contiguous: the arity-2 hot path
    gathers two columns per candidate pair via ``ascontiguousarray`` over thousands of pairs, and a C-order layout would
    copy ``n`` int64 per gather (2 copies x N_pairs). F-order makes those gathers a zero-copy view (~7x on the per-combo
    aggregate at n=100k); bit-identical (same values, layout-only change).
    """
    n, p = mat.shape
    codes = np.zeros((n, p), dtype=np.int64, order="F")
    all_edges: List[np.ndarray] = []
    for j in range(p):
        col = mat[:, j]
        nan_mask = ~np.isfinite(col)
        finite = col[~nan_mask]
        if finite.size == 0:
            all_edges.append(np.array([0.0, 1.0]))
            continue
        edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, nbins + 1)))
        all_edges.append(edges)
        if edges.size < 2:
            continue  # constant column -> all code 0
        # searchsorted over interior edges; non-finite rows land in bin 0 (kept, not dropped). A cheap masked write of
        # the lowest edge into the NaN positions avoids np.nan_to_num's allocate + isposinf/isneginf scan (16% of wall).
        if nan_mask.any():
            col = col.copy()
            col[nan_mask] = edges[0]
        c = np.searchsorted(edges[1:-1], col, side="right")
        np.clip(c, 0, edges.size - 2, out=c)
        codes[:, j] = c
    return codes, all_edges


def _bin_label(edges: np.ndarray, b: int) -> str:
    """Human-readable bound for bin ``b`` of a feature with the given quantile edges."""
    if edges.size < 2:
        return "const"
    b = int(np.clip(b, 0, edges.size - 2))
    return f"[{edges[b]:.3g}..{edges[b + 1]:.3g}]"


def _aggregate_combo(
    codes: np.ndarray,
    err: np.ndarray,
    feat_idx: Tuple[int, ...],
    nbins_per: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-slice (sum_error, count, flat_cell_id) for one feature combination via a single bincount pair.

    The combined linear cell id is ``sum_k code[:,f_k] * stride_k`` (mixed-radix over the per-feature bin counts).
    ``np.bincount`` with ``weights=err`` gives per-cell error sums; a plain ``bincount`` gives per-cell counts.
    One O(n) pass per combination -- no python loop over rows.
    """
    m = len(feat_idx)
    strides = np.ones(m, dtype=np.int64)
    for k in range(m - 1, 0, -1):
        strides[k - 1] = strides[k] * nbins_per[k]
    ncells = 1
    for k in range(m):
        ncells *= int(nbins_per[k])
    if m == 2:
        # Arity-2 fast path (thousands of feature pairs dominate the search): the flatten is folded into the njit
        # reduction, so no length-n ``flat`` array is allocated. The two code columns are gathered once into
        # contiguous 1D arrays (cache-friendly) and passed straight in.
        c0 = np.ascontiguousarray(codes[:, feat_idx[0]])
        c1 = np.ascontiguousarray(codes[:, feat_idx[1]])
        sums, counts = _fused_sum_count_2col(c0, c1, int(strides[0]), err, ncells)
        return sums, counts, strides
    # Mixed-radix flatten, accumulated in place. codes is int64 (see _bin_matrix) so no per-combo re-cast; the last
    # feature has stride 1 (radix LSB) so it is added directly without a multiply.
    flat = codes[:, feat_idx[0]] * strides[0]
    for k in range(1, m):
        col = codes[:, feat_idx[k]]
        if strides[k] == 1:
            flat += col
        else:
            flat += col * strides[k]
    # Single fused row-order pass accumulates per-cell error sum + count together, replacing the two
    # separate O(n) ``np.bincount`` walks (+ the int->float counts copy). Accumulation order is row
    # order in both paths, so the float64 sums are bit-identical (verified across adversarial trials).
    sums, counts = _fused_sum_count(flat, err, ncells)
    return sums, counts, strides


def find_weak_slices(
    X: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: str = "regression",
    feature_names: Optional[Sequence[str]] = None,
    nbins: int = DEFAULT_NBINS,
    max_arity: int = 2,
    top_k: int = DEFAULT_TOP_K,
    min_support_fraction: float = DEFAULT_MIN_SUPPORT_FRACTION,
    max_combos: int = DEFAULT_MAX_COMBOS,
    three_way_top_features: int = DEFAULT_THREE_WAY_TOP_FEATURES,
    title: str = "Worst feature-value slices (error degradation x support)",
    seed: int = 0,
) -> SliceFinderResult:
    """Search 1-``max_arity``-feature binned slices for the worst (degradation x support) regions.

    Features are quantile-binned once; every candidate slice's mean error + support is one ``np.bincount`` over the
    combined bin code. Singletons and 2-feature pairs are enumerated fully (bounded by ``max_combos``); 3-way slices
    are restricted to the ``three_way_top_features`` most error-discriminating features (tree-ranked, reusing the
    weak-segment ranker). Everything the search capped is recorded in ``capped`` and logged.
    """
    import pandas as pd

    err = _per_row_error(y_true, y_pred, task=task)
    mat, names = _resolve_feature_matrix(X, feature_names)
    finite = np.isfinite(err)
    if not np.all(finite):
        err = err[finite]
        mat = mat[finite]
    err = np.ascontiguousarray(err)  # hoisted out of the per-combo loop; the njit reduction kernels need it C-contig.
    n, p = mat.shape
    global_error = float(err.mean()) if err.size else float("nan")
    min_support = max(1, int(np.ceil(min_support_fraction * max(n, 1))))

    capped: List[str] = []
    if n == 0 or p == 0:
        empty = pd.DataFrame(columns=["features", "bounds", "mean_error", "support",
                                       "support_fraction", "error_ratio", "score"])
        bar = BarPanelSpec(categories=("(no data)",), values=np.array([0.0]),
                           title=title + " (no usable data)", orientation="horizontal")
        return SliceFinderResult(FigureSpec(panels=((bar,),), figsize=(8.0, 5.0)),
                                 empty, global_error, ((), "", float("nan"), 0), ())

    codes, all_edges = _bin_matrix(mat, nbins)
    nbins_per = [max(1, all_edges[j].size - 1) for j in range(p)]

    # Build the combination list with caps.
    combos: List[Tuple[int, ...]] = [(j,) for j in range(p)]
    pairs = list(itertools.combinations(range(p), 2))
    if max_arity >= 2:
        if len(combos) + len(pairs) > max_combos:
            keep = max(0, max_combos - len(combos))
            capped.append(f"pair enumeration truncated at {max_combos} combos ({len(pairs)} pairs, kept {keep})")
            pairs = pairs[:keep]
        combos.extend(pairs)
    if max_arity >= 3 and p > 3:
        top = _top_split_features(mat, err, names, max_depth=3, n_features=min(three_way_top_features, p), seed=seed)
        if len(top) < p:
            capped.append(f"3-way restricted to top {len(top)} error-discriminating features")
        triples = list(itertools.combinations(sorted(top), 3))
        room = max_combos - len(combos)
        if len(triples) > room:
            capped.append(f"3-way enumeration truncated to {max(room, 0)} of {len(triples)} triples")
            triples = triples[:max(room, 0)]
        combos.extend(triples)

    # Batch the arity-2 pair aggregations (the dominant ~thousands of combos) into one prange-parallel pass; singles
    # and triples (far fewer) stay on the serial per-combo path. Bit-identical to the serial path by construction.
    _pair_agg: dict = {}
    if _HAS_NUMBA_SLICE:
        _pairs_only = [c for c in combos if len(c) == 2]
        if _pairs_only and codes.nbytes <= _SLICE_TRANSPOSE_MAX_BYTES:
            _f0 = np.array([c[0] for c in _pairs_only], dtype=np.int64)
            _f1 = np.array([c[1] for c in _pairs_only], dtype=np.int64)
            _s0 = np.array([nbins_per[c[1]] for c in _pairs_only], dtype=np.int64)
            _ncells = np.array([nbins_per[c[0]] * nbins_per[c[1]] for c in _pairs_only], dtype=np.int64)
            _max_cells = int(_ncells.max()) if _ncells.size else 1
            _codes_t = np.ascontiguousarray(codes.T)
            _bsums, _bcounts = _batched_pair_sum_count(_codes_t, err, _f0, _f1, _s0, _max_cells)
            for _k2, _c in enumerate(_pairs_only):
                _nc = int(_ncells[_k2])
                _strides2 = np.array([int(_s0[_k2]), 1], dtype=np.int64)
                _pair_agg[_c] = (_bsums[_k2, :_nc].copy(), _bcounts[_k2, :_nc].copy(), _strides2)

    # Aggregate every combo; collect slices above the support floor. Only the cheap numeric records (mean / support /
    # score) plus the minimal slice IDENTITY (the combo's feature indices + the decoded per-feature bin row) are kept
    # here -- the expensive human-readable ``bounds`` / ``features`` labels (thousands of ``_bin_label`` calls + f-string
    # formatting + " & ".join) are deferred to AFTER the top_k selection below, so labels are built for only the ~top_k
    # displayed rows instead of every candidate cell across all combos (the caller discards all but top_k).
    rec_combo: List[Tuple[int, ...]] = []
    rec_bins: List[Tuple[int, ...]] = []
    rec_mean: List[float] = []
    rec_support: List[int] = []
    rec_score: List[float] = []
    for combo in combos:
        _cached = _pair_agg.get(combo)
        if _cached is not None:
            sums, counts, strides = _cached
        else:
            sums, counts, strides = _aggregate_combo(codes, err, combo, [nbins_per[f] for f in combo])
        valid = counts >= min_support
        if not valid.any():
            continue
        means = np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), 0.0)
        degradation = means - global_error
        support_frac = counts / n
        scores = degradation * np.sqrt(support_frac)
        cell_ids = np.flatnonzero(valid & (degradation > 0.0))
        if cell_ids.size == 0:
            continue
        # Decode every selected mixed-radix cell id at once: bins[:,k] = (rem // stride_k), rem %= stride_k.
        # Same integer recurrence as the per-cell Python loop (np.unravel_index arithmetic), bit-identical
        # by construction, but vectorized over all selected cells of the combo instead of a python row loop.
        m = len(combo)
        rem = cell_ids.astype(np.int64, copy=True)
        decoded = np.empty((cell_ids.size, m), dtype=np.int64)
        for k in range(m):
            sk = int(strides[k])
            decoded[:, k] = rem // sk
            rem %= sk
        decoded_rows = decoded.tolist()
        for ci, cid in enumerate(cell_ids):
            rec_combo.append(combo)
            rec_bins.append(tuple(decoded_rows[ci]))
            rec_mean.append(float(means[cid]))
            rec_support.append(int(counts[cid]))
            rec_score.append(float(scores[cid]))

    if not rec_score:
        empty = pd.DataFrame(columns=["features", "bounds", "mean_error", "support",
                                       "support_fraction", "error_ratio", "score"])
        bar = BarPanelSpec(categories=("(no weak slice)",), values=np.array([0.0]),
                           title=title + f"\n(no slice worse than global={global_error:.3g})",
                           orientation="horizontal")
        for msg in capped:
            logger.info("slice_finder cap: %s", msg)
        return SliceFinderResult(FigureSpec(panels=((bar,),), figsize=(8.0, 5.0)),
                                 empty, global_error, ((), "", float("nan"), 0), tuple(capped))

    # Display order is worst-ERROR-first: rank the surfaced slices by mean error descending (stable mergesort so equal
    # errors keep their score-built order), then take the top_k. The candidate pool itself is still built by the
    # degradation x support score above -- only the displayed ordering is by error.
    order = np.argsort(np.asarray(rec_mean), kind="stable")[::-1][:top_k]
    # Build the expensive bounds / features labels ONLY for the displayed top_k rows (see the deferral note above).
    def _features_for(i: int) -> Tuple[str, ...]:
        return tuple(names[f] for f in rec_combo[i])

    def _bounds_for(i: int) -> str:
        combo_i = rec_combo[i]
        bins_i = rec_bins[i]
        return " & ".join(
            f"{names[combo_i[k]]} {_bin_label(all_edges[combo_i[k]], int(bins_i[k]))}"
            for k in range(len(combo_i))
        )

    rec_features = {i: _features_for(i) for i in order}
    rec_bounds = {i: _bounds_for(i) for i in order}
    table = pd.DataFrame({
        "features": [rec_features[i] for i in order],
        "bounds": [rec_bounds[i] for i in order],
        "mean_error": [rec_mean[i] for i in order],
        "support": [rec_support[i] for i in order],
        "support_fraction": [rec_support[i] / n for i in order],
        "error_ratio": [rec_mean[i] / global_error if global_error > 0 else float("inf") for i in order],
        "score": [rec_score[i] for i in order],
    })
    table.index = np.arange(1, len(order) + 1)
    table.index.name = "rank"

    for msg in capped:
        logger.info("slice_finder cap: %s", msg)

    # Horizontal bars, worst on top: bar length = mean error, annotated label carries the support fraction + ratio.
    cats = tuple(
        f"{table['bounds'].iloc[i]}  (n={int(table['support'].iloc[i]):_}, {table['error_ratio'].iloc[i]:.2g}x)"
        for i in range(len(table))
    )
    vals = table["mean_error"].to_numpy()
    # ``table`` is already worst-first; the renderer inverts the y-axis for horizontal bars so the first category
    # lands on TOP, so the worst slice reads at the top to match the "worst-on-top" title (no pre-reverse here).
    bar = BarPanelSpec(
        categories=cats,
        values=vals,
        title=title + f"\n(global mean error = {global_error:.3g}; worst-on-top, label = support n + error ratio)",
        xlabel="Slice mean error",
        ylabel="slice",
        orientation="horizontal",
        colors=("crimson",),
        hline=(global_error, "black", f"global = {global_error:.3g}"),
    )
    fig = FigureSpec(suptitle="", panels=((bar,),), figsize=(10.0, max(5.0, 0.5 * len(table) + 2.0)))
    top_row = table.iloc[0]
    worst_slice = (tuple(top_row["features"]), str(top_row["bounds"]),
                   float(top_row["mean_error"]), int(top_row["support"]))
    return SliceFinderResult(fig, table, global_error, worst_slice, tuple(capped))


__all__ = [
    "SliceFinderResult",
    "find_weak_slices",
    "DEFAULT_NBINS",
    "DEFAULT_MIN_SUPPORT_FRACTION",
    "DEFAULT_TOP_K",
    "DEFAULT_MAX_COMBOS",
    "DEFAULT_THREE_WAY_TOP_FEATURES",
]
