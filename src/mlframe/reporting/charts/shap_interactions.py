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

from dataclasses import dataclass, field
import logging

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # plt-using paths are guarded; matplotlib-less envs skip plotting
    plt = None  # type: ignore[assignment]

from mlframe.reporting.charts._sampling import subsample_preserving_extremes
from mlframe.reporting.charts._layout import base_for as _base_for
from mlframe.reporting.charts.shap_panels import (
    _close_figs,
    _save_figure,
    _as_frame_and_names,
    _row_subset,
    _score_proxy,
    is_tree_model,
)

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


logger = logging.getLogger(__name__)


def _skipped_interactions(reason: str, plot_file: Optional[str], plot_outputs: Optional[str]) -> "ShapInteractionResult":
    """A skipped ``ShapInteractionResult`` that also WRITES the reason to the requested path, so the report shows it."""
    paths: List[str] = []
    if plot_file and plt is not None:
        try:
            fig = plt.figure(figsize=(9.0, 2.2))
            fig.text(0.5, 0.5, "SHAP interaction panels not produced:" + chr(10) + reason, ha="center", va="center", fontsize=10, wrap=True)
            paths = _save_figure(fig, _base_for(plot_file, "interaction_skipped"), plot_outputs)
            plt.close(fig)
        except Exception as exc:  # a notice must never be the thing that breaks a report
            logger.debug("shap-interaction skip-notice write failed (%s: %s)", type(exc).__name__, exc)
    return ShapInteractionResult(paths=paths, skipped=reason)


def _mean_abs_interaction(model: Any, X_sample: Any) -> np.ndarray:
    """Mean over rows of |interaction matrix|; returns a (F, F) array. ONE TreeExplainer pass."""
    import shap

    explainer = shap.TreeExplainer(model)
    iv = explainer.shap_interaction_values(X_sample)
    if isinstance(iv, list):  # legacy list-of-(rows,F,F) per class
        arr = np.asarray(iv[-1], dtype=np.float64)
    else:
        arr = np.asarray(getattr(iv, "values", iv), dtype=np.float64)
        if arr.ndim == 4:  # (rows, F, F, classes) under the new API for a multiclass/binary model
            arr = arr[..., -1]
    return np.asarray(np.abs(arr).mean(axis=0))


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
        return _skipped_interactions(
            f"pair interactions need >= 2 features and >= 1 row; got {n:,} rows and {f} features",
            plot_file, plot_outputs,
        )
    if not is_tree_model(model):
        return _skipped_interactions(
            "non-tree model; interaction values need TreeExplainer (KernelExplainer interactions are too slow)",
            plot_file, plot_outputs,
        )

    cap = min(int(max_rows), n)
    proxy = _score_proxy(model, carrier, n)
    idx = subsample_preserving_extremes(
        np.arange(n), sample_size=cap, extreme_values=proxy, k_extremes=_K_EXTREMES, rng=seed,
    )
    X_sample = _row_subset(carrier, idx)

    mat = _mean_abs_interaction(model, X_sample)
    _pairs, pair_names, pair_strength = _rank_pairs(mat, names, top_pairs)

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
    """Horizontal bar chart of the top feature-pair interaction strengths, strongest pair on top."""
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.45 * len(pair_names) + 1.0)))
    y = np.arange(len(pair_names))[::-1]  # strongest on top
    ax.barh(y, strengths, color="#3b528b")
    ax.set_yticks(y)
    ax.set_yticklabels(list(pair_names))
    ax.set_xlabel("mean |SHAP interaction value|")
    ax.set_ylabel("feature pair")
    _top = f" -- strongest: {pair_names[0]} ({strengths[0]:.4g})" if len(pair_names) else ""
    ax.set_title("Top feature-pair interactions" + _top)
    fig.tight_layout()
    return fig


_HEATMAP_MAX_FEATURES: int = 40


def _render_heatmap(mat: np.ndarray, names: Sequence[str]) -> Any:
    """Feature x feature interaction-strength heatmap, diagonal zeroed so the colour scale is driven by off-diagonal interaction, not main effects."""
    # Zero the diagonal so the colour scale is driven by interaction (off-diagonal), not main effects.
    off = mat.copy()
    np.fill_diagonal(off, 0.0)
    title_suffix = ""
    if len(names) > _HEATMAP_MAX_FEATURES:
        # 0.5 inch per feature is unbounded: 200 features asks matplotlib for a 102-inch square (over 100 megapixels)
        # whose labels are unreadable anyway. Show the strongest interactors and say so rather than emitting a wall.
        keep = np.argsort(off.max(axis=1))[::-1][:_HEATMAP_MAX_FEATURES]
        keep.sort()
        off = off[np.ix_(keep, keep)]
        title_suffix = f" (top {_HEATMAP_MAX_FEATURES} of {len(names)} features by peak interaction)"
        names = [names[i] for i in keep]
    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * len(names) + 2.0),) * 2)
    im = ax.imshow(off, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(list(names), rotation=90, fontsize=8)
    ax.set_yticklabels(list(names), fontsize=8)
    ax.set_title("Interaction strength (off-diagonal)" + title_suffix)
    ax.set_xlabel("feature")
    ax.set_ylabel("feature")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean |SHAP interaction value| (off-diagonal)")
    fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TOP_PAIRS",
    "ShapInteractionResult",
    "shap_interaction_summary",
]
