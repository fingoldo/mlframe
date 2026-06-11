"""SHAP post-fit diagnostic panels (compute-gated).

Two matplotlib-native panels off ONE explainer + ONE SHAP-value computation:

* the beeswarm (summary) -- every feature's |SHAP| distribution, ranked top-down by mean |SHAP|
  so the model's headline drivers sit at the top;
* a dependence scatter per top-K feature -- how that feature's value maps to its SHAP contribution,
  revealing the learned monotone / kink / interaction shape.

Compute gate (``is_tree_model``): tree models get ``shap.TreeExplainer`` which is exact and fast
(polynomial in tree size, independent of background size) -> a default-on diagnostic candidate.
Non-tree models fall back to ``shap.KernelExplainer``, whose cost is O(n_background * 2^features-ish)
sampling -- SLOW -- so it is OPT-IN (``allow_kernel=True``) and runs on a much smaller row cap.

Efficiency contract:
* rows are subsampled to ``max_rows`` BEFORE any SHAP work, STRATIFIED to keep the high-|error| / tail
  rows (``subsample_preserving_extremes`` on a residual / score proxy) so the tail the headline metrics
  quote is never silently dropped;
* the explainer is built ONCE and ``shap_values`` computed ONCE -- reused for the beeswarm and EVERY
  dependence plot (never one explainer / recompute per feature);
* KernelExplainer additionally caps the background to ``kernel_background`` rows (k-means summary) and
  the explained rows to ``kernel_max_rows`` so the opt-in path stays bounded.

Lifecycle mirrors the confidence-analysis beeswarm: snapshot ``plt.get_fignums()`` before each shap
plot, savefig, then ``_close_unless_interactive`` every figure shap opened so none leak in the registry.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # plt-using paths are guarded; matplotlib-less envs skip plotting
    plt = None  # type: ignore[assignment]

from mlframe.reporting.charts._sampling import subsample_preserving_extremes

logger = logging.getLogger(__name__)

# Default row cap for the FAST (tree) path: 20k rows give a stable beeswarm / dependence shape; the
# TreeExplainer cost is dominated by tree traversal per row, so this cap is the wall-time lever.
DEFAULT_MAX_ROWS: int = 20_000
# Number of top-|SHAP| features to draw a dependence scatter for. The beeswarm shows all; dependence
# plots past ~6 become an unreadable wall of small panels.
DEFAULT_TOP_K: int = 6
# Tail rows force-kept by the stratified subsample (the high-|score-proxy| rows the metrics quote).
_K_EXTREMES: int = 200
# KernelExplainer is SLOW; the opt-in path caps both the background summary and the explained rows hard.
KERNEL_BACKGROUND: int = 100
KERNEL_MAX_ROWS: int = 500

# Substrings in a model's class MRO that mark a tree / GBDT model TreeExplainer handles exactly+fast.
_TREE_MARKERS: Tuple[str, ...] = (
    "randomforest", "extratrees", "decisiontree", "gradientboosting", "histgradientboosting",
    "xgb", "lgbm", "lightgbm", "catboost", "isolationforest", "baggingclassifier", "baggingregressor",
)


@dataclass
class ShapPanelsResult:
    """Outcome of :func:`shap_summary_and_dependence`.

    ``figures`` are the (still-open unless closed) matplotlib figures in draw order: beeswarm first,
    then one dependence figure per top-K feature. ``paths`` are the files written (parallel to the
    plot kinds). ``top_features`` / ``mean_abs_shap`` rank the features by mean |SHAP| (descending).
    ``explainer_kind`` is ``"tree"`` / ``"kernel"``. ``skipped`` is a reason string when nothing ran.
    """

    figures: List[Any]
    paths: List[str]
    top_features: List[str]
    mean_abs_shap: np.ndarray
    explainer_kind: str
    skipped: Optional[str] = None


def is_tree_model(model: Any) -> bool:
    """True iff ``model`` is a tree / GBDT estimator ``shap.TreeExplainer`` handles exactly and fast.

    Drives the default gate: tree -> cheap exact explainer (default-on candidate); non-tree -> the
    slow sampling KernelExplainer (opt-in). Detection walks the class MRO names (so wrapped / subclassed
    estimators are caught) and also honours an explicit ``_is_tree`` marker an adapter may set.
    """
    if model is None:
        return False
    if bool(getattr(model, "_is_tree", False)):
        return True
    for klass in type(model).__mro__:
        name = klass.__name__.lower()
        if any(marker in name for marker in _TREE_MARKERS):
            return True
    return False


def _as_frame_and_names(X: Any, feature_names: Optional[Sequence[str]]) -> Tuple[Any, np.ndarray, List[str]]:
    """Return ``(carrier_for_shap, values_2d, names)``.

    ``carrier_for_shap`` keeps the original frame flavour where it has columns (shap indexes
    column-by-column and labels axes from it); a bare ndarray carrier is returned as-is. ``values_2d``
    is the float view used only for the residual proxy + dependence x-values, never a frame copy.
    """
    if hasattr(X, "columns") and not isinstance(X, np.ndarray):
        names = [str(c) for c in X.columns]
        vals = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return X, np.asarray(vals, dtype=np.float64), (list(feature_names) if feature_names is not None else names)
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_cols = arr.shape[1]
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_cols)]
    return arr, np.asarray(arr, dtype=np.float64), names


def _score_proxy(model: Any, carrier: Any, n: int) -> Optional[np.ndarray]:
    """A per-row magnitude proxy so the subsample keeps the tail rows (large |score| / |residual|).

    Uses the model's own output (decision_function / predict_proba positive / predict) centred to its
    median, so the extreme-magnitude rows -- the ones a beeswarm tail and the headline metrics care
    about -- are force-kept. Returns ``None`` when no usable output (then the subsample is plain random).
    """
    for attr in ("decision_function", "predict_proba", "predict"):
        fn = getattr(model, attr, None)
        if not callable(fn):
            continue
        try:
            out = np.asarray(fn(carrier))
        except Exception:
            continue
        if out.ndim == 2:
            out = out[:, -1] if out.shape[1] >= 2 else out.ravel()
        out = np.asarray(out, dtype=np.float64).ravel()
        if out.shape[0] != n or not np.any(np.isfinite(out)):
            continue
        med = float(np.nanmedian(out[np.isfinite(out)]))
        return out - med
    return None


def _row_subset(carrier: Any, idx: np.ndarray) -> Any:
    """Row-subset the carrier in its native format (no whole-frame copy)."""
    if isinstance(carrier, np.ndarray):
        return carrier[idx]
    if hasattr(carrier, "iloc"):  # pandas
        return carrier.iloc[idx]
    try:  # polars
        return carrier[idx]
    except Exception:
        return np.asarray(carrier)[idx]


def _shap_values_2d(sv: Any) -> np.ndarray:
    """Extract the 2-D (n_rows, n_features) SHAP matrix from a shap Explanation / array / list.

    Binary classifiers return a 3-D (rows, features, classes) Explanation under the new API (or a
    list-of-2 under the legacy one); take the positive class. Regression / single-output is already 2-D.
    """
    values = getattr(sv, "values", sv)
    if isinstance(values, list):
        values = values[-1] if len(values) > 1 else values[0]
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _save_figure(fig: Any, base: str, plot_outputs: Optional[str]) -> List[str]:
    """Save ``fig`` to ``base`` honouring the format(s) in ``plot_outputs`` (matplotlib raster/vector only).

    ``plot_outputs`` is the project plot-output DSL (e.g. ``"matplotlib[png]"``); shap panels are
    matplotlib-native, so only matplotlib raster/vector formats are emitted (plotly clauses are ignored
    -- there is no plotly beeswarm). When ``plot_outputs`` is empty a single ``.png`` is written.
    """
    formats = _matplotlib_formats(plot_outputs)
    written: List[str] = []
    root, ext = os.path.splitext(base)
    if ext:  # explicit extension on base path wins, regardless of the DSL
        formats = [ext.lstrip(".").lower()]
        root = root
    for fmt in formats:
        path = f"{root}.{fmt}"
        try:
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        except Exception as save_err:
            logger.warning("SHAP panel savefig failed for %s: %s", path, save_err)
    return written


def _matplotlib_formats(plot_outputs: Optional[str]) -> List[str]:
    """Matplotlib raster/vector formats requested by the DSL; defaults to ``["png"]``."""
    if not plot_outputs:
        return ["png"]
    try:
        from mlframe.reporting.output import parse_plot_output_dsl
        spec = parse_plot_output_dsl(plot_outputs)
    except Exception:
        return ["png"]
    fmts: List[str] = []
    for backend, formats in spec.backends:
        if str(backend).lower() != "matplotlib":
            continue
        for f in sorted(formats):  # frozenset -> deterministic order
            fl = str(f).lower()
            if fl not in fmts:
                fmts.append(fl)
    return fmts or ["png"]


def _close_figs(figs: List[Any]) -> None:
    """Close every shap-opened figure unless inside an interactive kernel (mirror confidence-analysis)."""
    if plt is None or not figs:
        return
    try:
        from mlframe.metrics import show_plots_unless_agg
        from mlframe.metrics.calibration import _close_unless_interactive
        was_shown = show_plots_unless_agg()
        _close_unless_interactive(figs, was_shown=was_shown)
    except Exception:
        for fig in figs:
            try:
                plt.close(fig)
            except Exception:
                pass


def shap_summary_and_dependence(
    model: Any,
    X: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    top_k: int = DEFAULT_TOP_K,
    plot_file: Optional[str] = None,
    plot_outputs: Optional[str] = None,
    allow_kernel: bool = False,
    kernel_background: int = KERNEL_BACKGROUND,
    kernel_max_rows: int = KERNEL_MAX_ROWS,
    seed: int = 0,
) -> ShapPanelsResult:
    """Beeswarm + top-K dependence plots off ONE explainer / ONE SHAP-value computation.

    Tree models use the exact fast ``shap.TreeExplainer`` (default-on candidate). Non-tree models need
    the slow sampling ``shap.KernelExplainer`` and run only when ``allow_kernel=True`` (and on the much
    smaller ``kernel_max_rows`` / ``kernel_background`` caps). Rows are subsampled to ``max_rows`` first,
    stratified to keep the high-|score-proxy| tail. Returns a :class:`ShapPanelsResult`; a degenerate /
    skipped input returns a result with ``skipped`` set and empty figures (best-effort diagnostic).

    ``plot_file`` is the base path (extension optional); ``plot_outputs`` selects matplotlib format(s).
    """
    import shap  # required dep; let ImportError surface to the caller

    carrier, vals, names = _as_frame_and_names(X, feature_names)
    n = vals.shape[0]
    if n == 0 or vals.shape[1] == 0:
        return ShapPanelsResult([], [], [], np.empty(0), "none", skipped="empty input")

    tree = is_tree_model(model)
    if not tree and not allow_kernel:
        return ShapPanelsResult(
            [], [], [], np.empty(0), "none",
            skipped="non-tree model; KernelExplainer is slow -- pass allow_kernel=True to opt in",
        )

    cap = max_rows if tree else min(max_rows, kernel_max_rows)
    proxy = _score_proxy(model, carrier, n)
    idx = subsample_preserving_extremes(
        np.arange(n), sample_size=min(cap, n), extreme_values=proxy, k_extremes=_K_EXTREMES, rng=seed,
    )
    X_sample = _row_subset(carrier, idx)
    vals_sample = vals[idx]

    if tree:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample, check_additivity=False)
        explainer_kind = "tree"
    else:
        bg = shap.sample(X_sample, min(kernel_background, len(idx)), random_state=seed)
        predict = getattr(model, "predict_proba", None) or getattr(model, "predict")
        explainer = shap.KernelExplainer(predict, bg)
        shap_values = explainer.shap_values(X_sample)
        explainer_kind = "kernel"

    shap_mat = _shap_values_2d(shap_values)
    mean_abs = np.abs(shap_mat).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_names = [names[i] if i < len(names) else f"f{i}" for i in order[: max(int(top_k), 1)]]

    if plt is None:
        return ShapPanelsResult([], [], top_names, mean_abs, explainer_kind, skipped="matplotlib unavailable")

    figures: List[Any] = []
    paths: List[str] = []

    figs_before = set(plt.get_fignums())
    try:
        # Beeswarm SUMMARY from the 2-D per-sample SHAP matrix (positive class for a classifier). Passing the raw
        # Explanation makes shap render a per-class / interaction grid instead of the canonical dot beeswarm.
        shap.summary_plot(
            shap_mat, vals_sample, feature_names=names, plot_type="dot",
            max_display=max(int(top_k), 1), show=False,
        )
        beeswarm = plt.gcf()
        # Widen so the "SHAP value (impact on model output)" x-axis label is not clipped by the colorbar.
        beeswarm.set_size_inches(9.0, max(4.0, 0.5 * min(len(names), max(int(top_k), 1)) + 2.0))
        figures.append(beeswarm)
        if plot_file:
            paths.extend(_save_figure(beeswarm, _base_for(plot_file, "shap_beeswarm"), plot_outputs))

        for rank, col in enumerate(order[: max(int(top_k), 1)]):
            before = set(plt.get_fignums())
            shap.dependence_plot(int(col), shap_mat, vals_sample, feature_names=names, interaction_index=None, show=False)
            new = [plt.figure(num) for num in plt.get_fignums() if num not in before]
            dep_fig = new[-1] if new else plt.gcf()
            figures.append(dep_fig)
            if plot_file:
                paths.extend(_save_figure(dep_fig, _base_for(plot_file, f"shap_dependence_{rank}_{_safe(top_names[rank])}"), plot_outputs))
    finally:
        # Close EVERY figure shap opened (not just the ones we tracked) so a mid-flow error never leaks.
        leaked = [plt.figure(num) for num in plt.get_fignums() if num not in figs_before]
        _close_figs(leaked or figures)

    return ShapPanelsResult(figures, paths, top_names, mean_abs, explainer_kind)


def _base_for(plot_file: str, suffix: str) -> str:
    """Compose a per-panel base path: ``<root>_<suffix><ext>`` so each panel writes a distinct file."""
    root, ext = os.path.splitext(plot_file)
    return f"{root}_{suffix}{ext}"


def _safe(name: str) -> str:
    """Filename-safe feature name (alnum / underscore / dash)."""
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name))[:48]


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TOP_K",
    "KERNEL_BACKGROUND",
    "KERNEL_MAX_ROWS",
    "ShapPanelsResult",
    "is_tree_model",
    "shap_summary_and_dependence",
]
