"""Per-column edge computation behind ``_adaptive_nbins.per_feature_edges``.

``compute_col_edges`` is the pure (no cache I/O, no shared state, hence thread-safe) computation for ONE column: the low-cardinality midpoint shortcut,
the method's edge builder, the collapsed-supervised and sparse-dominance fallbacks, and the empty-edges guardrail. The builders themselves live in
``_adaptive_nbins``; they are reached through that module at call time, so a patch of one of them there is seen here, and the two modules have no
import cycle (``per_feature_edges`` imports this one lazily).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._binning_spec import SUPERVISED_SCHEMES
from mlframe.feature_selection.filters.discretization.shared import quantiles_of_sorted, sorted_unique_with_counts
from mlframe.feature_selection.filters.discretization.shared import unique_with_counts

logger = logging.getLogger("mlframe.feature_selection.filters._adaptive_nbins")

__all__ = ["compute_col_edges", "clear_edge_memo"]

_SUPERVISED_METHODS = SUPERVISED_SCHEMES


def _an() -> Any:
    """The ``_adaptive_nbins`` module, resolved at call time."""
    from . import _adaptive_nbins

    return _adaptive_nbins


def _is_empty(edges: Any) -> bool:
    """``None`` or a zero-size array."""
    return edges is None or (hasattr(edges, "size") and edges.size == 0)


def _max_depth(kwargs: dict) -> int:
    """The MDLP depth: an explicit ``max_depth`` wins, else it is derived from the ``max_adaptive_nbins`` ceiling (``log2``)."""
    return int(kwargs.get("max_depth", max(1, int(math.log2(kwargs.get("max_adaptive_nbins", _an().MAX_ADAPTIVE_NBINS))))))


def _fayyad_irani(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """MDLP (Fayyad-Irani) edges with the depth, split-size, backend and validated-split settings read from ``kwargs``."""
    assert y is not None  # needs_y guard raises for this method when y is None
    return _an().edges_fayyad_irani(
        col, y,
        # MDLP's own leaf count is bounded by 2**max_depth - deriving the default from the
        # same max_adaptive_nbins ceiling used by knuth/bayesian_blocks/freedman_diaconis
        # (instead of a hardcoded 8) keeps "how many bins can one column produce" answered
        # the same way across every strategy. An explicit max_depth still wins.
        max_depth=_max_depth(kwargs),
        min_split_size=kwargs.get("min_split_size", 5),
        # Match edges_fayyad_irani / mdlp_bin_edges default ('njit').
        # The legacy 'python' default here re-introduced the
        # 1566 s / 1700 s @500 k regression that the iter570 fix
        # otherwise resolved - this kwargs.get path is the
        # production caller from categorize_dataset, so the
        # default flip is the actual gating change.
        backend=kwargs.get("mdlp_backend", "njit"),
        scaled_min_split=kwargs.get("mdlp_scaled_min_split", False),
        max_y_classes=kwargs.get("mdlp_max_y_classes", 64),
        # Validated (significance-gated) splitting is now the DEFAULT
        # (accuracy over speed per project convention - see supervised_binning.py's
        # mdlp_bin_edges docstring for the full A/B). Pass mdlp_fast_mode=True (e.g.
        # MRMR(binning_kwargs={"mdlp_fast_mode": True})) to opt back into the
        # cheap depth-capped classic path for a specific run.
        fast_mode=kwargs.get("mdlp_fast_mode", False),
        alpha=kwargs.get("mdlp_alpha", 0.05),
        n_permutations=kwargs.get("mdlp_n_permutations", 15),
        bonferroni=kwargs.get("mdlp_bonferroni", False),
        validated_seed=kwargs.get("mdlp_validated_seed", 0),
        y_pseudo_classes=kwargs.get("mdlp_y_pseudo_classes", 16),
        presorted=kwargs.get("_presorted"),
    )


def _fayyad_irani_validated(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Fayyad-Irani edges with a held-out validation split deciding which splits are kept."""
    assert y is not None  # needs_y guard raises for this method when y is None
    return _an().edges_fayyad_irani_validated(
        col, y,
        val_frac=kwargs.get("mdlp_val_frac", 0.3),
        max_depth=_max_depth(kwargs),
        min_split_size=kwargs.get("min_split_size", 5),
        val_min_split_size=kwargs.get("mdlp_val_min_split_size", 5),
        random_state=kwargs.get("random_state", 0),
    )


def _mah(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Supervised ``mah`` edges seeded from ``mah_initial_k`` intervals."""
    assert y is not None  # needs_y guard raises for this method when y is None
    return _an().edges_mah(col, y, initial_k=int(kwargs.get("mah_initial_k", 16)))


def _optimal_joint(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Edges for the bin count chosen by cross-validated joint MI over the candidate counts."""
    assert y is not None  # needs_y guard raises for this method when y is None
    return _an().edges_optimal_joint(
        col, y,
        candidates=kwargs.get("candidates", (4, 8, 16, 32)),
        n_splits=kwargs.get("n_splits", 3),
        base=base,
        random_state=kwargs.get("random_state", 0),
        max_y_classes=kwargs.get("optimal_joint_max_y_classes", 64),
        plan_cache=kwargs.get("optimal_joint_plan_cache"),
    )


def _partition(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Edges of the ``quantile`` / ``uniform`` schemes with the bin count of ``count_rule`` capped at ``nbins_cap``."""
    return _an().edges_partition(
        col, kwargs["partition"], kwargs["count_rule"], cap=kwargs.get("nbins_cap", _an().DEFAULT_NBINS_CAP), fixed_nbins=kwargs.get("fixed_nbins", 10), alpha=kwargs.get("qs_alpha", 0.30),
        facts=kwargs.get("_unique_facts"),
    )


def _bayesian_blocks(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Bayesian-blocks edges with the ``p0`` false-alarm rate and the subsample threshold."""
    return _an().edges_bayesian_blocks(
        col,
        p0=kwargs.get("p0", 0.05),
        edge_placement=kwargs.get("bb_edge_placement", "start"),
        subsample_threshold=kwargs.get("bb_subsample_threshold", _an()._BB_DEFAULT_SUBSAMPLE_THRESHOLD),
        m_max_cap=kwargs.get("bb_m_max_cap", kwargs.get("max_adaptive_nbins", _an().MAX_ADAPTIVE_NBINS)),
    )


_EDGE_BUILDERS: dict[str, Callable[[np.ndarray, Optional[np.ndarray], str, dict], Any]] = {
    "partition": _partition,
    "bayesian_blocks": _bayesian_blocks,
    "mdlp": _fayyad_irani,
    "mdlp_validated": _fayyad_irani_validated,
    "mah": _mah,
    "optimal_joint": _optimal_joint,
}


def _quantile_edges_of_sorted(xs: np.ndarray, n_bins: int) -> np.ndarray:
    """``_edges_from_quantiles`` of the finite values ``xs`` already sorted ascending: the same inner edges (``n_bins - 1`` linear quantiles, de-duplicated), read off the sort instead of a partition."""
    full_edges = np.unique(quantiles_of_sorted(xs, np.linspace(0.0, 100.0, n_bins + 1) / 100.0))
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    return np.asarray(full_edges[1:-1], dtype=np.float64)  # Intentional float64: bin-edge output contract


def _collapsed_supervised_edges(
    edges: Any, col: np.ndarray, finite: np.ndarray, base: str, kwargs: dict, facts_key: Any = None, uniq: Optional[np.ndarray] = None, uniq_counts: Optional[np.ndarray] = None,
    presorted: "tuple[np.ndarray, np.ndarray] | None" = None,
) -> Any:
    """Edges of a supervised method after the collapse handling.

    When a supervised binning method (MDLP / Mah / optimal_joint /
    fayyad_irani) returns zero inner edges - meaning the feature
    was collapsed to a single bin because individually it has no
    MI with y - the joint MI on any tuple containing this feature
    is identically 0 (1-cell joint). This silently DESTROYS
    synergy detection for XOR-family targets (y = sign(x1*x2),
    boolean conjunctions, etc.) where individual components are
    independent of y but their joint perfectly predicts y.
    Fall back to UNSUPERVISED binning (the requested ``base``)
    for collapsed columns so synergy tuples still have signal at
    the joint level. The single-feature MDLP signal is already
    gone (it returned no splits), so the unsupervised fallback
    can only improve detection power, never hurt.
    """
    if _is_empty(edges):
        _fallback_nb = int(kwargs.get("collapsed_fallback_nbins", 5))
        if base == "quantile":
            if presorted is not None and _fallback_nb >= 2:
                return _quantile_edges_of_sorted(presorted[0], _fallback_nb)
            return _memoised_fallback("quantile", col, _fallback_nb, uniq, lambda: _an()._edges_from_quantiles(col, _fallback_nb))
        return _memoised_fallback("uniform", col, _fallback_nb, uniq, lambda: _an()._edges_from_uniform(col, _fallback_nb))
    from ._supervised_collapse_refine import refine_near_collapsed_supervised_edges

    fallback_nbins = int(kwargs.get("collapsed_fallback_nbins", 5))
    cache = kwargs.get("optimal_joint_plan_cache")
    if cache is not None and facts_key is not None and hasattr(edges, "tobytes"):
        # The refinement reads the column and the edges only; a null that permutes the target meets the same few edge sets of a column again and again.
        refined = cache.memo(("refine", facts_key, edges.tobytes(), base, fallback_nbins), lambda: refine_near_collapsed_supervised_edges(edges, finite, base, fallback_nbins, uniq, uniq_counts))
        return np.array(refined, copy=True) if hasattr(refined, "size") else refined
    return refine_near_collapsed_supervised_edges(edges, finite, base, fallback_nbins, uniq, uniq_counts)


def _sparse_dominance_edges(edges: Any, finite: np.ndarray, uniq: np.ndarray, uniq_counts: np.ndarray, kwargs: dict) -> Any:
    """SPARSE-AWARE secondary fallback.

    For TF-IDF / one-hot / bag-of-words style columns (>50% mass at a single
    value, e.g. zero for sparse tokens) the unsupervised quantile
    fallback ALSO collapses: every quantile lands at the dominant
    value, np.unique dedups to 1-2 edges, and the resulting 1-bin
    column produces MI=0 with y. This silently kills sparse-token
    signal (Layer 20 finding: binning_scheme='mdlp' default + 95%-
    zero token columns -> screening returns fallback_used_=True
    with support=['tok_0']). Detect the sparse-dominance pattern
    explicitly and split into a separate-bin for the dominant
    value + quantile bins on the non-dominant subset.
    """
    if not (finite.size >= 4 and (edges is None or (hasattr(edges, "size") and edges.size <= 1))):
        return edges
    _vals_sp, _counts_sp = uniq, uniq_counts
    if _vals_sp.size < 2:
        return edges
    _max_idx = int(_counts_sp.argmax())
    _dom_frac = float(_counts_sp[_max_idx]) / float(finite.size)
    if _dom_frac <= 0.5:
        return edges
    _dom_val = float(_vals_sp[_max_idx])
    _non_dom_mask = finite != _dom_val
    _non_dom = finite[_non_dom_mask]
    _sparse_nb = int(kwargs.get("sparse_separate_fallback_nbins", 4))
    if _non_dom.size >= _sparse_nb:
        _qs_levels = np.linspace(0.0, 1.0, _sparse_nb + 1)[1:-1]
        _sub = np.quantile(_non_dom, _qs_levels)
        _sub = np.unique(_sub)
    else:
        _sub = np.unique(_non_dom)
    # Boundary between dominant value and non-dominant range.
    _non_dom_min = float(_non_dom.min())
    _non_dom_max = float(_non_dom.max())
    if _dom_val <= _non_dom_min:
        _boundary = 0.5 * (_dom_val + _non_dom_min)
        _new_edges = np.concatenate([[_boundary], _sub])
    elif _dom_val >= _non_dom_max:
        _boundary = 0.5 * (_non_dom_max + _dom_val)
        _new_edges = np.concatenate([_sub, [_boundary]])
    else:
        _lower_dom = float(finite[finite < _dom_val].max())
        _upper_dom = float(finite[finite > _dom_val].min())
        _new_edges = np.concatenate([
            _sub[_sub < _dom_val],
            [0.5 * (_lower_dom + _dom_val), 0.5 * (_dom_val + _upper_dom)],
            _sub[_sub > _dom_val],
        ])
    return np.unique(_new_edges)


_EDGE_MEMO: "OrderedDict" = OrderedDict()
_EDGE_MEMO_MAX = 4096
_EDGE_MEMO_LOCK = threading.Lock()
_SIMPLE_KWARG_TYPES = (int, float, str, bool, type(None))


def clear_edge_memo() -> None:
    """Forget the memoised edges (the table is content-keyed, so this only frees memory)."""
    with _EDGE_MEMO_LOCK:
        _EDGE_MEMO.clear()


def _edge_memo_key(col: np.ndarray, method_resolved: str, base: str, kwargs: dict, y: Optional[np.ndarray], low_card_cap: int) -> "Optional[tuple]":
    """Content key of one edge computation, or ``None`` when it must not be memoised (a keyword argument that is not a plain value, such as the optimal-joint plan cache, which carries state)."""
    items = []
    for k in sorted(kwargs):
        v = kwargs[k]
        if not isinstance(v, _SIMPLE_KWARG_TYPES):
            return None
        items.append((k, v))
    from mlframe.feature_selection.filters._fe_resident_operands import operand_signature

    y_sig = None if y is None or method_resolved not in _SUPERVISED_METHODS else operand_signature(np.ascontiguousarray(y))
    return (operand_signature(np.ascontiguousarray(col)), y_sig, method_resolved, str(base), int(low_card_cap), tuple(items))


# DISTINCT-VALUE FACTS of a column by content: the honest null of a supervised strategy asks for the edges of the same column under every permutation of the target, and the sorted distinct values
# and counts depend on the column alone (a radix sort of the column, a tenth of the time of a mah-family fit). Bounded; a miss recomputes.
_UNIQUE_FACTS: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_UNIQUE_FACTS_MAX = 256
_UNIQUE_FACTS_MAX_BYTES = 256 * 1024 * 1024  # the entries are 12 bytes a row of a continuous column: 256 of them were 1.5 GB at 500000 rows
_UNIQUE_FACTS_LOCK = threading.Lock()
_UNIQUE_FACTS_MIN_ROWS = 2048


def _unique_facts(finite: np.ndarray, col_sig: Optional[tuple]) -> "tuple[np.ndarray, np.ndarray]":
    """``unique_with_counts(finite)``, served from the content-keyed memo when the signature of the column is known and the column is long enough to be worth keeping."""
    if col_sig is None or finite.size < _UNIQUE_FACTS_MIN_ROWS:
        return unique_with_counts(finite)
    with _UNIQUE_FACTS_LOCK:
        hit = _UNIQUE_FACTS.get(col_sig)
        if hit is not None:
            _UNIQUE_FACTS.move_to_end(col_sig)
            return hit
    facts = unique_with_counts(finite)
    for arr in facts:
        arr.setflags(write=False)  # shared between calls: a caller that wrote into them would corrupt the next column's edges
    with _UNIQUE_FACTS_LOCK:
        _UNIQUE_FACTS[col_sig] = facts
        _held = sum(int(a.nbytes) + int(b.nbytes) for a, b in _UNIQUE_FACTS.values())
        while len(_UNIQUE_FACTS) > 1 and (len(_UNIQUE_FACTS) > _UNIQUE_FACTS_MAX or _held > _UNIQUE_FACTS_MAX_BYTES):
            _old_u, _old_c = _UNIQUE_FACTS.popitem(last=False)[1]  # evict-ok: a miss recomputes the facts
            _held -= int(_old_u.nbytes) + int(_old_c.nbytes)
    return facts


# The unsupervised fallback edges of a collapsed column are a function of the column alone; a null that re-bins a column without signal under every permutation meets the collapse each time
# (a 20000-row column: the percentile alone was 40 % of the samples of a mah fit). Keyed by the distinct-values object of the column (held by the entry, so its id is not recycled while cached).
_FALLBACK_EDGES: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_FALLBACK_EDGES_MAX = 256
_FALLBACK_EDGES_LOCK = threading.Lock()


def _memoised_fallback(kind: str, col: np.ndarray, n_bins: int, uniq: Optional[np.ndarray], compute: Callable[[], np.ndarray]) -> np.ndarray:
    """``compute()`` (the fallback edges of ``col`` for ``n_bins``), served from the memo when the column's distinct-values array is known; returned arrays are copies."""
    if uniq is None or col.size < _UNIQUE_FACTS_MIN_ROWS:
        return compute()
    key = (id(uniq), kind, int(n_bins))
    with _FALLBACK_EDGES_LOCK:
        hit = _FALLBACK_EDGES.get(key)
        if hit is not None and hit[0] is uniq:
            _FALLBACK_EDGES.move_to_end(key)
            _memo_edges: np.ndarray = hit[1]
            return np.array(_memo_edges, copy=True)
    edges = compute()
    with _FALLBACK_EDGES_LOCK:
        _FALLBACK_EDGES[key] = (uniq, edges.copy())
        while len(_FALLBACK_EDGES) > _FALLBACK_EDGES_MAX:
            _FALLBACK_EDGES.popitem(last=False)  # evict-ok: a miss recomputes the edges
    return edges


def compute_col_edges(col: np.ndarray, method_resolved: str, base: str, kwargs: dict, y: Optional[np.ndarray], low_card_cap: int) -> tuple[Any, bool]:
    """``_compute_col_edges_uncached`` through a content-keyed memo: the same column, labels and settings give the same edges, so a warm-up pass over the raw columns (run while the feature
    engineering cascades work) serves the real discretisation without computing them again. Returned arrays are copies."""
    key = _edge_memo_key(col, method_resolved, base, kwargs, y, low_card_cap)
    if key is not None:
        with _EDGE_MEMO_LOCK:
            hit = _EDGE_MEMO.get(key)
            if hit is not None:
                _EDGE_MEMO.move_to_end(key)
        if hit is not None:
            return (hit[0].copy() if isinstance(hit[0], np.ndarray) else hit[0]), hit[1]
    edges, low = _compute_col_edges_uncached(col, method_resolved, base, kwargs, y, low_card_cap, None if key is None else key[0])
    if key is not None:
        with _EDGE_MEMO_LOCK:
            _EDGE_MEMO[key] = (edges.copy() if isinstance(edges, np.ndarray) else edges, low)
            while len(_EDGE_MEMO) > _EDGE_MEMO_MAX:
                _EDGE_MEMO.popitem(last=False)  # evict-ok: a miss recomputes the edges
    return edges, low


def _mdlp_labels(y: np.ndarray, kwargs: dict) -> np.ndarray:
    """The labels the MDLP search encodes ``y`` to (``_encode_mdlp_target`` is remembered for the last target, so this costs a hash of ``y``)."""
    from .supervised_binning import _encode_mdlp_target

    y_enc, _depth = _encode_mdlp_target(y, kwargs.get("mdlp_max_y_classes", 64), kwargs.get("mdlp_fast_mode", False), kwargs.get("mdlp_y_pseudo_classes", 16), _max_depth(kwargs))
    return np.asarray(y_enc)


class EdgePresort:
    """The device presort of the edge search for the columns of one call: it sorts a block of the columns that will be computed (not the ones the content memo already holds) with their labels, on the device."""

    def __init__(self, labels: np.ndarray, method_resolved: str, base: str, kwargs: dict, y: Optional[np.ndarray], low_card_cap: int) -> None:
        self.labels, self._memo_args = labels, (method_resolved, base, kwargs, y, low_card_cap)

    def block(self, columns: "list[np.ndarray]") -> list:
        """One :class:`_mdlp_gpu_presort.Presorted` (or ``None``) per column; a column whose edges the memo holds is not sorted."""
        from ._mdlp_gpu_presort import presort_block

        wanted = []
        for i, col in enumerate(columns):
            key = _edge_memo_key(col, *self._memo_args)
            with _EDGE_MEMO_LOCK:
                held = key is not None and key in _EDGE_MEMO
            if not held:
                wanted.append(i)
        out: list = [None] * len(columns)
        if wanted:
            for i, result in zip(wanted, presort_block([columns[i] for i in wanted], self.labels)):
                out[i] = result
        return out


def edge_presort_for(method_resolved: str, base: str, kwargs: dict, y: Optional[np.ndarray], n_rows: int, low_card_cap: int, workers: int = 8) -> "EdgePresort | None":
    """The device presort of the edge search of this call, or ``None`` when the host path serves it: the MDLP search of a supervised target on a frame the device takes (``workers`` edge-search workers share the host passes)."""
    if method_resolved != "mdlp" or y is None:
        return None
    from ._mdlp_gpu_presort import applicable

    labels = _mdlp_labels(y, kwargs)
    if labels.shape[0] != n_rows or not applicable(n_rows, labels, workers):
        return None
    return EdgePresort(labels, method_resolved, base, kwargs, y, low_card_cap)


def _presort_for_mdlp(col: np.ndarray, finite: np.ndarray, finite_mask: np.ndarray, method_resolved: str, y: Optional[np.ndarray], kwargs: dict) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(x_sorted, y_sorted)`` of the finite rows of ``col`` for the MDLP (``fayyad_irani``) search, or ``None`` when the radix sort does not apply (small or non-float input, a non-integer
    label vector) and the search sorts for itself. The labels are the ones ``mdlp_bin_edges`` encodes ``y`` to, so the search gives the edges it gives unsorted."""
    if method_resolved != "mdlp" or y is None or finite.size == 0:
        return None
    from ._mdlp_gpu_presort import supplied_presort
    from .discretization.shared import sort_xy

    device = supplied_presort(col)  # sorted on the device with its block (EdgePresort): the same arrays the host sort returns
    if device is not None:
        return device
    y_enc = _mdlp_labels(y, kwargs)
    if y_enc.shape[0] != col.shape[0]:
        return None  # the search raises its own length error
    return sort_xy(finite, y_enc[finite_mask])


def _compute_col_edges_uncached(col: np.ndarray, method_resolved: str, base: str, kwargs: dict, y: Optional[np.ndarray], low_card_cap: int, col_sig: Optional[tuple] = None) -> tuple[Any, bool]:
    """Pure per-column edge computation (no cache I/O). Returns ``(edges, was_low_card)``.

    Identical math to the historical serial loop body; factored out so the heavy path can run under a thread pool. Touches no shared state ->
    thread-safe.

    If a column has few unique finite values (e.g. binary target, small categorical, ordinal already pre-encoded), quantile-based binning collapses to
    1-bin because ``_edges_from_quantiles`` returns empty edges after ``np.unique`` dedup. Such columns get midpoint-edges between consecutive unique
    values - this preserves the column's natural cardinality without going through quantile / supervised logic that doesn't apply.
    """
    from ._mdlp_gpu_presort import SuppliedPresort

    _facts_cache = kwargs.get("optimal_joint_plan_cache")
    _facts_key = None
    if _facts_cache is not None:
        # The finite values and their unique counts depend on the column only; a null that permutes the target bins the same columns for every permutation.
        _facts_key, _finite, _uniq, _uniq_counts = _facts_cache.column_facts(col)
        _presorted = None
    else:
        _finite_mask = np.isfinite(col)
        _finite = col[_finite_mask]
        _presorted = _presort_for_mdlp(col, _finite, _finite_mask, method_resolved, y, kwargs)
        # return_counts=True up front so the rare sparse-dominance fallback below (which needs the
        # per-value counts) reuses THIS array instead of re-running np.unique a second time.
        if _presorted is not None:
            # the column was sorted once for the MDLP search: its distinct values and counts are a pass over that sort, not a second radix sort of the same values (or, from the device, the runs it read off its sort)
            if isinstance(_presorted, SuppliedPresort):
                _uniq, _uniq_counts = _presorted.uniq, _presorted.counts
            else:
                _uniq, _uniq_counts = sorted_unique_with_counts(_presorted[0])
                _uniq = _uniq.astype(_finite.dtype, copy=False)
        elif _finite.size > 0:
            _uniq, _uniq_counts = _unique_facts(_finite, col_sig)
        else:
            _uniq = np.empty(0, dtype=_crit_np_dtype())
            _uniq_counts = np.empty(0, dtype=np.int64)
    if 1 < _uniq.size <= low_card_cap:
        # Use midpoints between consecutive uniques as edges so
        # each unique value lands in its own bin.
        return 0.5 * (_uniq[:-1] + _uniq[1:]), True
    builder = _EDGE_BUILDERS.get(method_resolved)
    if builder is None:
        raise NotImplementedError(method_resolved)
    edges = builder(col, y, base, {**kwargs, "_unique_facts": (_uniq, _uniq_counts)} if method_resolved == "partition" else (kwargs if _presorted is None else {**kwargs, "_presorted": _presorted}))
    if method_resolved in _SUPERVISED_METHODS:
        edges = _collapsed_supervised_edges(edges, col, _finite, base, kwargs, _facts_key, _uniq, _uniq_counts, _presorted)
    edges = _sparse_dominance_edges(edges, _finite, _uniq, _uniq_counts, kwargs)
    # Systemic silent-degenerate-fallback guardrail: EVERY method funnels through this
    # single return point, so this is the ONE place that can catch "binning method returned empty/
    # near-empty edges despite the column having real variance" for ALL strategies (qs/mah/sturges/fd/
    # knuth/blocks/fayyad_irani/optimal_joint), not just the three with a dedicated collapse-fallback
    # above. This is exactly the bug class an MDLP overflow (3.0**n_classes -> inf, acceptance check
    # always False, empty edges returned with no signal) slipped through undetected - a column with
    # >1 distinct finite value that still ends up with 0 usable edges collapses to a single degenerate
    # bin (all rows get the same code) with NO observable signal anywhere. Log so it is diagnosable
    # from a production run's logs alone, without per-strategy vigilance.
    if _is_empty(edges) and _uniq.size > 1:
        logger.warning(
            "per_feature_edges: method=%r produced EMPTY bin edges for a column with %d distinct finite "
            "values (real variance) -- this column will silently collapse to a single degenerate bin, "
            "destroying its MI signal. If this is unexpected, investigate the binning method on this "
            "column's distribution (extreme skew/cardinality/scale can silently degrade some strategies).",
            method_resolved, int(_uniq.size),
        )
    return edges, False
