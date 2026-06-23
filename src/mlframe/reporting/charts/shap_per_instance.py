"""Per-instance SHAP attribution for the TOP-K worst (most-confident-wrong) predictions.

Global SHAP (beeswarm / dependence / interactions) explains the model in aggregate; it cannot answer
"WHY was THIS costly case predicted the way it was". This panel ranks instances by error severity, picks
the K most-confident-wrong, and renders a per-instance signed-SHAP attribution so a reviewer sees which
features drove each expensive error.

Compute gate (``is_tree_model``): only tree / GBDT models qualify -- ``shap.TreeExplainer`` is exact and
fast (cost scales with tree size + the K explained rows, not n). Non-tree models would need the per-row
KernelExplainer, far too slow for a niche panel, so they are SKIPPED with an annotation (no fallback).

Error severity:
* binary (a 1-D score in [0,1]): confidence-wrong magnitude ``|y_true - y_score|`` -- a confident-wrong
  positive (true=0, score~1) or negative (true=1, score~0) sits at the top; ties broken by raw |residual|.
* otherwise (regression / generic score): raw ``|y_true - y_score|``.

The explainer is built ONCE on a bounded background sample (``max_explain_rows`` cap) and SHAP values are
computed ONCE for the K worst rows; one horizontal signed-SHAP bar per instance, annotated with the
predicted prob, the true label, and the top contributing features.
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

from mlframe.reporting.charts.shap_panels import (
    _as_frame_and_names,
    _close_figs,
    _matplotlib_formats,
    _row_subset,
    _safe,
    _shap_values_2d,
    is_tree_model,
)

logger = logging.getLogger(__name__)

# Default number of worst-error instances to explain; past ~6 the small-multiples grid gets unreadable.
DEFAULT_K: int = 4
# Background rows the TreeExplainer is built on (for a stable base value); capped so the panel stays bounded.
DEFAULT_MAX_EXPLAIN_ROWS: int = 2000
# Top contributing features annotated / shown per instance bar (the rest are folded into "other").
_TOP_FEATURES_PER_INSTANCE: int = 8


@dataclass
class ShapPerInstanceResult:
    """Outcome of :func:`shap_worst_errors_explanation`.

    ``figure`` is the (still-open unless closed) small-multiples figure. ``paths`` are files written.
    ``worst_idx`` are the original-row indices of the explained instances (worst error first).
    ``severities`` parallels ``worst_idx``. ``contributions`` is a list of (feature_name, signed_shap)
    lists per instance (sorted by |shap| desc). ``skipped`` is a reason string when nothing ran.
    """

    figure: Any
    paths: List[str]
    worst_idx: np.ndarray
    severities: np.ndarray
    contributions: List[List[Tuple[str, float]]] = field(default_factory=list)
    n_background: int = 0
    skipped: Optional[str] = None


def _error_severity(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Per-row error severity; for a binary score in [0,1] this is the confidence-wrong magnitude.

    Binary: ``|y_true - y_score|`` puts confident-wrong rows (true=0/score~1 or true=1/score~0) at the top.
    Otherwise: raw ``|y_true - y_score|`` (regression / generic score).
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    return np.abs(yt - ys)


def _is_binary_score(y_true: np.ndarray, y_score: np.ndarray) -> bool:
    """True iff y_true is {0,1} labels and y_score looks like a probability in [0,1]."""
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    finite = ys[np.isfinite(ys)]
    label_ok = np.all(np.isin(yt[~np_isnan(yt)], (0, 1))) if yt.size else False
    score_ok = finite.size > 0 and float(finite.min()) >= -1e-9 and float(finite.max()) <= 1.0 + 1e-9
    return bool(label_ok and score_ok)


def np_isnan(a: np.ndarray) -> np.ndarray:
    """NaN mask tolerant of integer / object label arrays (which ``np.isnan`` rejects)."""
    arr = np.asarray(a)
    if arr.dtype.kind in "fc":
        return np.isnan(arr)
    return np.zeros(arr.shape, dtype=bool)


def shap_worst_errors_explanation(
    model: Any,
    X: Any,
    y_true: Any,
    y_score: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    k: int = DEFAULT_K,
    max_explain_rows: int = DEFAULT_MAX_EXPLAIN_ROWS,
    plot_file: Optional[str] = None,
    plot_outputs: Optional[str] = None,
    seed: int = 0,
) -> ShapPerInstanceResult:
    """Per-instance SHAP attribution for the K most-confident-wrong predictions (tree models only).

    Ranks rows by error severity (binary: ``|y_true - y_score|``; else raw |residual|), takes the K
    worst, builds ONE ``shap.TreeExplainer`` on a background capped to ``max_explain_rows``, computes
    SHAP values ONCE for the K rows, and draws a small-multiples of signed-SHAP horizontal bars (one per
    instance, annotated with predicted prob / true label / top drivers). Non-tree models are SKIPPED
    (per-row KernelExplainer is too slow); a perfectly-separated input (no errors) is annotated.

    Returns a :class:`ShapPerInstanceResult`; degenerate / skipped input returns ``skipped`` set.
    """
    import shap  # required dep; let ImportError surface to the caller

    carrier, vals, names = _as_frame_and_names(X, feature_names)
    n = vals.shape[0]
    if n == 0 or vals.shape[1] == 0:
        return _skip("empty input")

    if not is_tree_model(model):
        return _skip("non-tree model; per-instance KernelExplainer is too slow -- skipped")

    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    if yt.shape[0] != n or ys.shape[0] != n:
        return _skip("y_true / y_score length mismatch with X")

    severity = _error_severity(yt, ys)
    binary = _is_binary_score(yt, ys)
    # A "no misclassification" panel is informative on its own: annotate rather than silently emit nothing.
    no_errors = binary and float(np.nanmax(severity)) < 0.5

    k_eff = max(int(k), 1)
    order = np.argsort(severity)[::-1]  # worst first
    worst_idx = order[: min(k_eff, n)]
    severities = severity[worst_idx]

    explain_idx = _background_index(order, n, max_explain_rows, worst_idx, seed)
    X_explain = _row_subset(carrier, explain_idx)
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_explain, check_additivity=False)
    shap_mat = _shap_values_2d(sv)

    pos = {int(orig): row for row, orig in enumerate(explain_idx)}
    contributions: List[List[Tuple[str, float]]] = []
    for orig in worst_idx:
        row = pos[int(orig)]
        sh = shap_mat[row]
        ranked = sorted(zip(names, sh), key=lambda t: abs(t[1]), reverse=True)
        contributions.append([(nm, float(v)) for nm, v in ranked])

    if plt is None:
        return ShapPerInstanceResult(None, [], worst_idx, severities, contributions, skipped="matplotlib unavailable")

    fig, paths = _render(
        worst_idx, severities, contributions, yt, ys, binary, no_errors, plot_file, plot_outputs,
    )
    return ShapPerInstanceResult(fig, paths, worst_idx, severities, contributions, n_background=len(explain_idx))


def _skip(reason: str) -> ShapPerInstanceResult:
    return ShapPerInstanceResult(None, [], np.empty(0, dtype=int), np.empty(0), [], skipped=reason)


def _background_index(order: np.ndarray, n: int, max_rows: int, worst_idx: np.ndarray, seed: int) -> np.ndarray:
    """Bounded background row index: the K worst rows (always) plus a random fill to ``max_rows``.

    The K worst rows MUST be in the explained set (we need their per-row SHAP); the rest are a random
    background so the TreeExplainer base value reflects the population, not just the tail.
    """
    cap = min(max(int(max_rows), len(worst_idx)), n)
    keep = set(int(i) for i in worst_idx)
    rng = np.random.default_rng(seed)
    pool = np.array([i for i in order if int(i) not in keep], dtype=int)
    rng.shuffle(pool)
    fill = pool[: max(cap - len(keep), 0)]
    idx = np.concatenate([np.asarray(worst_idx, dtype=int), fill]) if fill.size else np.asarray(worst_idx, dtype=int)
    return idx


def _render(
    worst_idx: np.ndarray,
    severities: np.ndarray,
    contributions: List[List[Tuple[str, float]]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    binary: bool,
    no_errors: bool,
    plot_file: Optional[str],
    plot_outputs: Optional[str],
) -> Tuple[Any, List[str]]:
    """Small-multiples: one signed-SHAP horizontal bar per worst-error instance."""
    k = len(worst_idx)
    ncol = min(k, 2) if k > 1 else 1
    nrow = int(np.ceil(k / ncol))
    figs_before = set(plt.get_fignums())
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 3.4 * nrow), squeeze=False)
    try:
        flat = axes.ravel()

        for ax in flat[k:]:
            ax.axis("off")

        for i, orig in enumerate(worst_idx):
            ax = flat[i]
            contrib = contributions[i][:_TOP_FEATURES_PER_INSTANCE]
            labels = [nm for nm, _ in contrib][::-1]
            vals = [v for _, v in contrib][::-1]
            colors = ["#d62728" if v >= 0 else "#1f77b4" for v in vals]
            ax.barh(range(len(vals)), vals, color=colors)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0.0, color="0.4", lw=0.8)
            ax.set_xlabel("signed SHAP (impact on model output)", fontsize=8)
            yt = y_true[int(orig)]
            ys = float(y_score[int(orig)])
            if binary:
                title = f"row {int(orig)}: pred prob={ys:.3f}  true={int(yt)}  |err|={severities[i]:.3f}"
            else:
                title = f"row {int(orig)}: pred={ys:.4g}  true={yt:.4g}  |err|={severities[i]:.4g}"
            ax.set_title(title, fontsize=9)

        if no_errors:
            fig.suptitle("Per-instance SHAP -- worst predictions (NOTE: no misclassifications; showing largest residuals)", fontsize=11)
        else:
            fig.suptitle("Per-instance SHAP attribution -- top-K most-confident-wrong predictions", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        paths: List[str] = []
        if plot_file:
            paths = _save(fig, plot_file, plot_outputs)
        return fig, paths
    finally:
        # The figure is rendered to disk, not displayed; leaving it open leaks it into matplotlib's global registry on
        # every call. Mirror shap_panels: close every figure opened since entry (incl. the saved one and any mid-flow leak).
        leaked = [plt.figure(num) for num in plt.get_fignums() if num not in figs_before]
        _close_figs(leaked)


def _save(fig: Any, plot_file: str, plot_outputs: Optional[str]) -> List[str]:
    """Save the figure honouring the matplotlib raster/vector formats in ``plot_outputs`` (PNG default)."""
    formats = _matplotlib_formats(plot_outputs)
    root, ext = os.path.splitext(plot_file)
    if ext:
        formats = [ext.lstrip(".").lower()]
    written: List[str] = []
    for fmt in formats:
        path = f"{root}.{fmt}"
        try:
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        except Exception as save_err:
            logger.warning("SHAP per-instance savefig failed for %s: %s", path, save_err)
    return written


__all__ = [
    "DEFAULT_K",
    "DEFAULT_MAX_EXPLAIN_ROWS",
    "ShapPerInstanceResult",
    "shap_worst_errors_explanation",
]
