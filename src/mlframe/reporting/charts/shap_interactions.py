"""SHAP feature-PAIR interaction summary (exotic, heavily capped).

Beyond the main-effect beeswarm/dependence (``shap_panels``), this ranks the strongest feature-PAIR
interactions by mean |SHAP interaction value| and renders a top-pairs bar + a feature x feature
interaction-strength heatmap (off-diagonal).

COST: ``shap.TreeExplainer.shap_interaction_values`` is O(F^2) per row (a full feature x feature matrix
per sample), far more expensive than plain SHAP values -- so the row sample is capped HARD at
``max_rows`` (default 2000). ``max_rows`` is THE cost lever: wall scales ~linearly with it. Only
TreeExplainer computes interaction values cheaply; non-tree models are skipped (we do NOT run a
KernelExplainer interaction approximation -- it is prohibitively slow and not what this diagnostic is for).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # plt-using paths are guarded; matplotlib-less envs skip plotting
    plt = None  # type: ignore[assignment]

from mlframe.reporting.charts._sampling import subsample_preserving_extremes
from mlframe.reporting.charts.shap_panels import (
    _close_figs,
    _matplotlib_formats,
    _as_frame_and_names,
    _row_subset,
    _score_proxy,
    is_tree_model,
)

logger = logging.getLogger(__name__)

# Interaction values are O(F^2) per row -- much heavier than plain SHAP, so the row cap is small.
DEFAULT_MAX_ROWS: int = 2_000
DEFAULT_TOP_PAIRS: int = 10
_K_EXTREMES: int = 100


@dataclass
class ShapInteractionResult:
    """Outcome of :func:`shap_interaction_summary`.

    ``pair_names`` / ``pair_strength`` are the top off-diagonal pairs ranked by mean |interaction|
    (descending, parallel). ``matrix`` is the full feature x feature mean |interaction| matrix (the
    heatmap source). ``skipped`` is a reason string when nothing ran (non-tree, <2 features, etc.).
    """

    figures: List[Any] = field(default_factory=list)
    paths: List[str] = field(default_factory=list)
    pair_names: List[str] = field(default_factory=list)
    pair_strength: np.ndarray = field(default_factory=lambda: np.empty(0))
    matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    skipped: Optional[str] = None


def _mean_abs_interaction(model: Any, X_sample: Any) -> np.ndarray:
    """Mean over rows of |interaction matrix|; returns a (F, F) array. ONE TreeExplainer pass."""
    import shap

    explainer = shap.TreeExplainer(model)
    iv = explainer.shap_interaction_values(X_sample)
    arr = np.asarray(getattr(iv, "values", iv), dtype=np.float64)
    if arr.ndim == 4:  # (rows, F, F, classes) under the new API for a multiclass/binary model
        arr = arr[..., -1]
    if isinstance(iv, list):  # legacy list-of-(rows,F,F) per class
        arr = np.asarray(iv[-1], dtype=np.float64)
    return np.abs(arr).mean(axis=0)


def _rank_pairs(mat: np.ndarray, names: Sequence[str], top_pairs: int) -> Tuple[List[Tuple[int, int]], List[str], np.ndarray]:
    """Rank off-diagonal (i<j) pairs by ``mat`` strength, descending. Off-diagonal only (diag = main effect)."""
    f = mat.shape[0]
    iu, ju = np.triu_indices(f, k=1)
    strengths = mat[iu, ju]
    order = np.argsort(strengths)[::-1][: max(int(top_pairs), 1)]
    pairs = [(int(iu[k]), int(ju[k])) for k in order]
    pnames = [f"{names[i]} x {names[j]}" for i, j in pairs]
    return pairs, pnames, strengths[order]


def shap_interaction_summary(
    model: Any,
    X: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    top_pairs: int = DEFAULT_TOP_PAIRS,
    plot_file: Optional[str] = None,
    plot_outputs: Optional[str] = None,
    seed: int = 0,
) -> ShapInteractionResult:
    """Rank the strongest feature-PAIR SHAP interactions; render a top-pairs bar + interaction heatmap.

    Tree models ONLY (``shap.TreeExplainer.shap_interaction_values``). Rows are subsampled to
    ``max_rows`` (the cost lever -- interaction values are O(F^2) per row) BEFORE any SHAP work,
    stratified to keep the high-|score-proxy| tail. Degenerate inputs (<2 features, non-tree, empty)
    return a result with ``skipped`` set and no figures.
    """
    import shap  # noqa: F401  required dep; let ImportError surface to the caller

    carrier, vals, names = _as_frame_and_names(X, feature_names)
    n, f = vals.shape
    if n == 0 or f < 2:
        return ShapInteractionResult(skipped="need >=2 features and >=1 row for pair interactions")
    if not is_tree_model(model):
        return ShapInteractionResult(skipped="non-tree model; interaction values need TreeExplainer (KernelExplainer interactions are too slow)")

    cap = min(int(max_rows), n)
    proxy = _score_proxy(model, carrier, n)
    idx = subsample_preserving_extremes(
        np.arange(n), sample_size=cap, extreme_values=proxy, k_extremes=_K_EXTREMES, rng=seed,
    )
    X_sample = _row_subset(carrier, idx)

    mat = _mean_abs_interaction(model, X_sample)
    pairs, pair_names, pair_strength = _rank_pairs(mat, names, top_pairs)

    if plt is None:
        return ShapInteractionResult(pair_names=pair_names, pair_strength=pair_strength, matrix=mat, skipped="matplotlib unavailable")

    figures: List[Any] = []
    paths: List[str] = []
    figs_before = set(plt.get_fignums())
    try:
        fig_bar = _render_top_pairs_bar(pair_names, pair_strength)
        figures.append(fig_bar)
        if plot_file:
            paths.extend(_save_figure(fig_bar, _base_for(plot_file, "interaction_top_pairs"), plot_outputs))

        fig_hm = _render_heatmap(mat, names)
        figures.append(fig_hm)
        if plot_file:
            paths.extend(_save_figure(fig_hm, _base_for(plot_file, "interaction_heatmap"), plot_outputs))
    finally:
        leaked = [plt.figure(num) for num in plt.get_fignums() if num not in figs_before]
        _close_figs(leaked or figures)

    return ShapInteractionResult(figures=figures, paths=paths, pair_names=pair_names, pair_strength=pair_strength, matrix=mat)


def _render_top_pairs_bar(pair_names: Sequence[str], strengths: np.ndarray) -> Any:
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.45 * len(pair_names) + 1.0)))
    y = np.arange(len(pair_names))[::-1]  # strongest on top
    ax.barh(y, strengths, color="#3b528b")
    ax.set_yticks(y)
    ax.set_yticklabels(list(pair_names))
    ax.set_xlabel("mean |SHAP interaction value|")
    ax.set_title("Top feature-pair interactions")
    fig.tight_layout()
    return fig


def _render_heatmap(mat: np.ndarray, names: Sequence[str]) -> Any:
    # Zero the diagonal so the colour scale is driven by interaction (off-diagonal), not main effects.
    off = mat.copy()
    np.fill_diagonal(off, 0.0)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * len(names) + 2.0),) * 2)
    im = ax.imshow(off, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(list(names), rotation=90, fontsize=8)
    ax.set_yticklabels(list(names), fontsize=8)
    ax.set_title("Interaction strength (off-diagonal)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean |interaction|")
    fig.tight_layout()
    return fig


def _save_figure(fig: Any, base: str, plot_outputs: Optional[str]) -> List[str]:
    """Save ``fig`` to ``base`` honouring matplotlib format(s) in ``plot_outputs`` (raster/vector only)."""
    formats = _matplotlib_formats(plot_outputs)
    root, ext = os.path.splitext(base)
    if ext:  # explicit extension on base path wins over the DSL
        formats = [ext.lstrip(".").lower()]
    written: List[str] = []
    for fmt in formats:
        path = f"{root}.{fmt}"
        try:
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        except Exception as save_err:
            logger.warning("SHAP interaction savefig failed for %s: %s", path, save_err)
    return written


def _base_for(plot_file: str, suffix: str) -> str:
    root, ext = os.path.splitext(plot_file)
    return f"{root}_{suffix}{ext}"


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TOP_PAIRS",
    "ShapInteractionResult",
    "shap_interaction_summary",
]
