"""Per-column edge computation behind ``_adaptive_nbins.per_feature_edges``.

``compute_col_edges`` is the pure (no cache I/O, no shared state, hence thread-safe) computation for ONE column: the low-cardinality midpoint shortcut,
the method's edge builder, the collapsed-supervised and sparse-dominance fallbacks, and the empty-edges guardrail. The builders themselves live in
``_adaptive_nbins``; they are reached through that module at call time, so a patch of one of them there is seen here, and the two modules have no
import cycle (``per_feature_edges`` imports this one lazily).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters._adaptive_nbins")

__all__ = ["compute_col_edges"]

_SUPERVISED_METHODS = ("fayyad_irani", "fayyad_irani_validated", "optimal_joint", "mah")


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
        # MRMR(nbins_strategy_kwargs={"mdlp_fast_mode": True})) to opt back into the
        # cheap depth-capped classic path for a specific run.
        fast_mode=kwargs.get("mdlp_fast_mode", False),
        alpha=kwargs.get("mdlp_alpha", 0.05),
        n_permutations=kwargs.get("mdlp_n_permutations", 15),
        bonferroni=kwargs.get("mdlp_bonferroni", False),
        validated_seed=kwargs.get("mdlp_validated_seed", 0),
        y_pseudo_classes=kwargs.get("mdlp_y_pseudo_classes", 16),
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
    )


def _sturges(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Sturges-rule edges on the requested ``base`` (quantile or uniform)."""
    return _an().edges_sturges(col, base=base)


def _freedman_diaconis(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Freedman-Diaconis-rule edges on the requested ``base``, capped at ``max_adaptive_nbins`` bins."""
    return _an().edges_freedman_diaconis(col, base=base, max_bins=kwargs.get("max_adaptive_nbins", _an().MAX_ADAPTIVE_NBINS))


def _qs(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Quantile-spacing edges with the ``qs_alpha`` smoothing."""
    return _an().edges_qs(col, alpha=kwargs.get("qs_alpha", 0.30))


def _knuth(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Knuth's optimal-bin-count edges."""
    return _an().edges_knuth(
        col,
        edge_type=kwargs.get("knuth_edge_type", "uniform"),
        m_max_cap=kwargs.get("knuth_m_max_cap", kwargs.get("max_adaptive_nbins", _an().MAX_ADAPTIVE_NBINS)),
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


def _uniform(col: np.ndarray, y: Optional[np.ndarray], base: str, kwargs: dict) -> Any:
    """Uniform-width edges with the Freedman-Diaconis bin count (capped at ``max_adaptive_nbins``)."""
    mod = _an()
    return mod.edges_uniform(col, n_bins=mod.freedman_diaconis_nbins(col, max_bins=kwargs.get("max_adaptive_nbins", mod.MAX_ADAPTIVE_NBINS)))


_EDGE_BUILDERS: dict[str, Callable[[np.ndarray, Optional[np.ndarray], str, dict], Any]] = {
    "sturges": _sturges,
    "freedman_diaconis": _freedman_diaconis,
    "qs": _qs,
    "knuth": _knuth,
    "bayesian_blocks": _bayesian_blocks,
    "fayyad_irani": _fayyad_irani,
    "fayyad_irani_validated": _fayyad_irani_validated,
    "uniform": _uniform,
    "mah": _mah,
    "optimal_joint": _optimal_joint,
}


def _collapsed_supervised_edges(edges: Any, col: np.ndarray, finite: np.ndarray, base: str, kwargs: dict) -> Any:
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
            return _an()._edges_from_quantiles(col, _fallback_nb)
        return _an()._edges_from_uniform(col, _fallback_nb)
    from ._supervised_collapse_refine import refine_near_collapsed_supervised_edges

    return refine_near_collapsed_supervised_edges(edges, finite, base, int(kwargs.get("collapsed_fallback_nbins", 5)))


def _sparse_dominance_edges(edges: Any, finite: np.ndarray, uniq: np.ndarray, uniq_counts: np.ndarray, kwargs: dict) -> Any:
    """SPARSE-AWARE secondary fallback.

    For TF-IDF / one-hot / bag-of-words style columns (>50% mass at a single
    value, e.g. zero for sparse tokens) the unsupervised quantile
    fallback ALSO collapses: every quantile lands at the dominant
    value, np.unique dedups to 1-2 edges, and the resulting 1-bin
    column produces MI=0 with y. This silently kills sparse-token
    signal (Layer 20 finding: nbins_strategy='mdlp' default + 95%-
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


def compute_col_edges(col: np.ndarray, method_resolved: str, base: str, kwargs: dict, y: Optional[np.ndarray], low_card_cap: int) -> tuple[Any, bool]:
    """Pure per-column edge computation (no cache I/O). Returns ``(edges, was_low_card)``.

    Identical math to the historical serial loop body; factored out so the heavy path can run under a thread pool. Touches no shared state ->
    thread-safe.

    If a column has few unique finite values (e.g. binary target, small categorical, ordinal already pre-encoded), quantile-based binning collapses to
    1-bin because ``_edges_from_quantiles`` returns empty edges after ``np.unique`` dedup. Such columns get midpoint-edges between consecutive unique
    values - this preserves the column's natural cardinality without going through quantile / supervised logic that doesn't apply.
    """
    _finite = col[np.isfinite(col)]
    # return_counts=True up front so the rare sparse-dominance fallback below (which needs the
    # per-value counts) reuses THIS array instead of re-running np.unique a second time.
    if _finite.size > 0:
        _uniq, _uniq_counts = np.unique(_finite, return_counts=True)
    else:
        _uniq = np.empty(0, dtype=np.float64)
        _uniq_counts = np.empty(0, dtype=np.int64)
    if 1 < _uniq.size <= low_card_cap:
        # Use midpoints between consecutive uniques as edges so
        # each unique value lands in its own bin.
        return 0.5 * (_uniq[:-1] + _uniq[1:]), True
    builder = _EDGE_BUILDERS.get(method_resolved)
    if builder is None:
        raise NotImplementedError(method_resolved)
    edges = builder(col, y, base, kwargs)
    if method_resolved in _SUPERVISED_METHODS:
        edges = _collapsed_supervised_edges(edges, col, _finite, base, kwargs)
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
