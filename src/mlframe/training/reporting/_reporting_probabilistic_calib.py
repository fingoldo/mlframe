"""report_probabilistic_model_perf -- moved out of _reporting.py.

Wave 97 (2026-05-21): the ~520-line ``report_probabilistic_model_perf``
function lives here so its parent module stays below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the symbol is
re-exported from ``_reporting`` so existing
``from mlframe.training._reporting import report_probabilistic_model_perf``
imports continue to work.

The function lazy-imports helpers from ``_reporting`` (``_canonical_multilabel_y``,
``_maybe_display``) inside the body to avoid the circular load with
that module's own top-level imports.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]


# Wave 97 (2026-05-21): _canonical_multilabel_y / _maybe_display + the
# DEFAULT_* constants all live in ``_reporting``; that module imports us
# from its bottom (after the helpers + constants are bound at module top),
# so by the time Python resolves these names ``_reporting`` is partially
# loaded and the symbols are already there. No circular-load failure,
# AND a single source of truth (no constant duplication across siblings).

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


from ._reporting_probabilistic import _slugify_class


def _render_fairness_calibration(
    *,
    subgroups: dict[str, Any],
    subset_index: np.ndarray | None,
    y_true: np.ndarray | pd.Series,
    pos_score: np.ndarray,
    plot_file: str,
    plot_outputs: str | None,
    metrics: dict[int | str, Any] | None,
) -> None:
    """Render a per-subgroup calibration-fairness figure per group feature (binary positive-class score).

    For each fairness group feature, slice the per-row group labels for THIS split (``bins.loc[subset_index]``) and
    compose the per-group reliability overlay + per-group ECE bar + max-min disparity gap. Failures are logged and
    skipped per feature so one bad group feature never aborts the report.
    """
    from mlframe.reporting.charts.fairness_calibration import (
        compose_fairness_calibration_figure,
        compute_subgroup_ece_disparity,
    )

    yt = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    _dsl = plot_outputs if plot_outputs else "matplotlib[png]"
    disparities: dict[str, Any] = {}

    for group_name, group_params in subgroups.items():
        if group_name in ("**ORDER**", "**RANDOM**"):
            continue  # robustness pseudo-groups, not sensitive features
        try:
            bins = group_params.get("bins") if isinstance(group_params, dict) else None
            if bins is None:
                continue
            if subset_index is not None and hasattr(bins, "loc"):
                bins = bins.loc[subset_index]
            labels = bins.to_numpy() if hasattr(bins, "to_numpy") else np.asarray(bins)
            if labels.shape[0] != yt.shape[0]:
                continue
            disparity = compute_subgroup_ece_disparity(yt, pos_score, labels)
            disparities[group_name] = disparity
            spec = compose_fairness_calibration_figure(
                yt, pos_score, labels, title=f"Calibration fairness by {group_name}",
            )
            _slug = _slugify_class(str(group_name))
            base_path = f"{plot_file}_faircal_{_slug}"
            from mlframe.reporting.output import parse_plot_output_dsl
            from mlframe.reporting.renderers import render_and_save
            render_and_save(spec, parse_plot_output_dsl(_dsl), base_path)
        except Exception as e:
            logger.debug("fairness_calibration chart for %r skipped: %s", group_name, e)

    if metrics is not None and disparities:
        metrics.update(dict(fairness_calibration_disparity=disparities))


def _top_importance_features(model: Any, columns: Sequence[str], top_k: int) -> list[str]:
    """Top-k feature names by the model's ``feature_importances_``, aligned to ``columns``. Empty when unavailable."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    imp = np.asarray(imp, dtype=np.float64).ravel()
    cols = list(columns)
    if imp.shape[0] != len(cols) or not np.isfinite(imp).any():
        return []
    order = np.argsort(imp)[::-1][:top_k]
    return [cols[i] for i in order]


def _render_calibration_by_feature(
    *,
    df: pd.DataFrame,
    columns: Sequence[str],
    model: Any,
    y_true: np.ndarray | pd.Series,
    pos_score: np.ndarray,
    plot_file: str,
    plot_outputs: str | None,
    metrics: dict[int | str, Any] | None,
) -> None:
    """Render per-feature calibration (reliability + ECE by feature quantile bin) for the top-importance features.

    The top-1..2 features by ``model.feature_importances_`` are pulled from ``df`` as continuous columns; each yields
    a small-multiples reliability figure + an ECE-vs-feature-bin line + the max-min "calibration heterogeneity"
    metric. Non-numeric / missing columns and degenerate features are skipped per feature so one bad column never
    aborts the report.
    """
    from mlframe.reporting.charts.calibration_by_feature import (
        compose_calibration_by_feature_figure,
        compute_calibration_by_feature_heterogeneity,
    )

    feats = _top_importance_features(model, columns, top_k=2)
    if not feats or not hasattr(df, "__getitem__"):
        return
    yt = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    _dsl = plot_outputs if plot_outputs else "matplotlib[png]"
    heterogeneity: dict[str, Any] = {}

    for fname in feats:
        try:
            col = df[fname]
            fv = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            fv = np.asarray(fv, dtype=np.float64).ravel()  # NaN for non-numeric; dropped downstream
            if fv.shape[0] != yt.shape[0]:
                continue
            het = compute_calibration_by_feature_heterogeneity(yt, pos_score, fv)
            heterogeneity[str(fname)] = het
            spec = compose_calibration_by_feature_figure(yt, pos_score, fv, feature_name=str(fname))
            _slug = _slugify_class(str(fname))
            base_path = f"{plot_file}_calibfeat_{_slug}"
            from mlframe.reporting.output import parse_plot_output_dsl
            from mlframe.reporting.renderers import render_and_save
            render_and_save(spec, parse_plot_output_dsl(_dsl), base_path)
        except Exception as e:
            logger.debug("calibration_by_feature chart for %r skipped: %s", fname, e)

    if metrics is not None and heterogeneity:
        metrics.update(dict(calibration_by_feature_heterogeneity=heterogeneity))


def _render_calibration_heatmap_2d(
    *,
    df: pd.DataFrame,
    columns: Sequence[str],
    model: Any,
    y_true: np.ndarray | pd.Series,
    pos_score: np.ndarray,
    plot_file: str,
    plot_outputs: str | None,
    metrics: dict[int | str, Any] | None,
) -> None:
    """Render a 2D calibration-ECE heatmap over the quantile grid of the top-2-importance feature pair (binary target).

    Both features are quantile-binned; per cell ``|mean(score) - mean(true)|`` is shown on an RdYlGn_r grid, with the
    worst cell + its location as the headline -- a localized over/under-confidence pocket the pooled / 1D views hide.
    Needs >=2 distinct top-importance features; degenerate columns are skipped without aborting the report.
    """
    from mlframe.reporting.charts.calibration_heatmap_2d import (
        compose_calibration_heatmap_2d_figure,
        compute_calibration_heatmap_2d,
    )

    feats = _top_importance_features(model, columns, top_k=2)
    if len(feats) < 2 or not hasattr(df, "__getitem__"):
        return
    fx_name, fy_name = feats[0], feats[1]
    yt = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    try:
        cx, cy = df[fx_name], df[fy_name]
        fx = np.asarray(cx.to_numpy() if hasattr(cx, "to_numpy") else cx, dtype=np.float64).ravel()
        fy = np.asarray(cy.to_numpy() if hasattr(cy, "to_numpy") else cy, dtype=np.float64).ravel()
        if fx.shape[0] != yt.shape[0] or fy.shape[0] != yt.shape[0]:
            return
        res = compute_calibration_heatmap_2d(yt, pos_score, fx, fy)
        if metrics is not None and res.get("worst_cell") is not None:
            metrics.update(dict(calibration_heatmap_2d={
                "worst_ece": res["worst_ece"], "worst_cell": res["worst_cell"],
                "median_cell_ece": res["median_cell_ece"], "traffic_light": res["traffic_light"],
                "feat_x": str(fx_name), "feat_y": str(fy_name),
            }))
        spec = compose_calibration_heatmap_2d_figure(
            yt, pos_score, fx, fy, feat_x_name=str(fx_name), feat_y_name=str(fy_name),
        )
        _dsl = plot_outputs if plot_outputs else "matplotlib[png]"
        base_path = f"{plot_file}_calib2d_{_slugify_class(str(fx_name))}_{_slugify_class(str(fy_name))}"
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
        render_and_save(spec, parse_plot_output_dsl(_dsl), base_path)
    except Exception as e:
        logger.debug("calibration_heatmap_2d chart skipped: %s", e)
