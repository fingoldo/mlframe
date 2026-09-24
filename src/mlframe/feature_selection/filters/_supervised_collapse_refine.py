"""Refinement of NEAR-collapsed supervised bin edges (MDLP / Mah / optimal_joint).

A supervised binner that returns NO inner edge already falls back to unsupervised bins in
``_adaptive_nbins`` so a zero-marginal operand keeps its joint (synergy) signal. The same failure happens
when the binner returns a split that isolates a sliver of the rows: e.g. the significance-gated MDLP
accepts a spurious tail cut on a pure-synergy column at its ~5% per-column false-positive rate
(``x3`` of the wide-synergy fixture: one cut leaving 10 of 2000 rows on one side). The column is then
almost a single bin, so every joint MI containing it collapses toward zero and a genuine
``sign(x3*x4)`` interaction drops out of the pair screen (joint MI 0.0037 vs 0.09 on proper bins).

Rule: when one bin holds more than ``1 - 1/(2*fallback_nbins)`` of the rows, the supervised partition
carries less information than a single unsupervised fallback cut, so treat it as collapsed and ADD the
fallback edges. The supervised cuts are kept (a genuine tail split still separates its rows); only the
missing resolution is restored. Columns whose supervised bins are reasonably balanced are untouched.
"""
from __future__ import annotations

from typing import Optional, cast, overload

import numpy as np


def _fallback_edges(finite: np.ndarray, base: str, n_bins: int) -> np.ndarray:
    """Unsupervised inner edges matching ``_adaptive_nbins``'s collapsed-column fallback."""
    if n_bins < 2 or finite.size == 0:
        return np.array([], dtype=np.float64)
    if base == "quantile":
        full = np.percentile(finite, np.linspace(0.0, 100.0, n_bins + 1))
    else:
        full = np.linspace(float(finite.min()), float(finite.max()), n_bins + 1)
    return cast(np.ndarray, np.unique(full[1:-1]))


@overload
def refine_near_collapsed_supervised_edges(edges: np.ndarray, finite: np.ndarray, base: str, fallback_nbins: int) -> np.ndarray:
    """Given edges, the result is always an array: refinement only ever adds cuts."""


@overload
def refine_near_collapsed_supervised_edges(edges: None, finite: np.ndarray, base: str, fallback_nbins: int) -> None:
    """Given no edges, there is nothing to refine and the result stays ``None``."""


def refine_near_collapsed_supervised_edges(edges: Optional[np.ndarray], finite: np.ndarray, base: str, fallback_nbins: int) -> Optional[np.ndarray]:
    """Return ``edges`` unchanged unless one bin dominates (see module docstring), else ``edges`` united with the
    unsupervised fallback edges. ``finite`` is the column's finite values; ``edges`` are inner cuts."""
    if edges is None or not hasattr(edges, "size") or edges.size == 0 or finite.size == 0 or fallback_nbins < 2:
        return edges
    inner = np.asarray(edges, dtype=np.float64)
    counts = np.bincount(np.searchsorted(inner, finite, side="right"), minlength=inner.size + 1)
    if counts.max() <= (1.0 - 1.0 / (2.0 * fallback_nbins)) * finite.size:
        return edges
    extra = _fallback_edges(np.asarray(finite, dtype=np.float64), base, fallback_nbins)
    if extra.size == 0:
        return edges
    return cast(np.ndarray, np.unique(np.concatenate([inner, extra])))
