"""2-D partial-dependence SURFACE for the top interacting feature pair (filled contour + colorbar).

A 1-D PDP averages the partner feature away, so non-additive interaction structure (a saddle / twist where
the response to feat_x depends on feat_y) is invisible in it. This module renders the model's average predicted
response over a (feat_x, feat_y) grid as a filled contour with overlaid contour lines, making the joint
response-surface SHAPE readable.

Complementary to the SHAP interaction ranking (it ranks pair STRENGTH); this SHOWS the surface shape.

Cost: the surface is ``grid`` predict calls over a (sample * grid) tiled block (one per outer-grid value), reusing
``compute_pdp_2d`` from ``pdp_ice`` -- so the work is ``grid`` predictions, independent of n, with sample/grid caps.

The interaction is quantified as the non-additive residual: subtract the additive (row-mean + col-mean - global)
reconstruction from the surface; a large residual RMS means a non-separable interaction, a small one means the
surface is (approximately) additive f(x)+g(y).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from mlframe.reporting.charts.pdp_ice import (
    DEFAULT_PDP_GRID, DEFAULT_PDP_SAMPLE, _as_2d, _feat_label, compute_pdp_2d,
)

Feature = Union[int, str]


def interaction_residual(surface: np.ndarray) -> dict:
    """Non-additive residual of a 2-D PDP grid: surface minus its additive (row + col - global) reconstruction.

    The additive part is the best separable f(x)+g(y) fit (two-way ANOVA main effects); whatever remains is pure
    interaction. Returns ``residual`` (same shape), ``residual_rms`` (its RMS), ``surface_std`` (spread of the
    surface), and ``residual_ratio`` = residual_rms / surface_std (interaction strength relative to total variation;
    ~0 for an additive surface, large for a strong twist/saddle). A constant surface yields ratio 0.
    """
    s = np.asarray(surface, dtype=np.float64)
    g = s.mean()
    row = s.mean(axis=1, keepdims=True)
    col = s.mean(axis=0, keepdims=True)
    additive = row + col - g  # two-way ANOVA main-effect reconstruction
    residual = s - additive
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    surface_std = float(s.std())
    ratio = residual_rms / surface_std if surface_std > 1e-12 else 0.0
    return {"residual": residual, "residual_rms": residual_rms, "surface_std": surface_std, "residual_ratio": ratio}


def _default_pair(
    model: Any, X: Any, names: Optional[List[str]], n_cols: int
) -> Optional[Tuple[Feature, Feature]]:
    """Pick the pair: top SHAP interaction pair when available (tree model + shap), else top-2 importance features.

    Importance falls back to ``feature_importances_`` / ``coef_`` magnitude, else the first two columns. Returns
    ``None`` only when there are fewer than two columns.
    """
    if n_cols < 2:
        return None
    label = lambda i: (names[i] if names is not None and i < len(names) else i)
    try:
        from mlframe.reporting.charts.shap_interactions import shap_interaction_summary

        res = shap_interaction_summary(model, X, plot_outputs=None, top_pairs=1)
        if res.skipped is None and res.matrix.size and res.matrix.shape[0] >= 2:
            mat = np.asarray(res.matrix, dtype=np.float64)
            iu = np.triu_indices(mat.shape[0], k=1)
            best = int(np.argmax(mat[iu]))
            return label(int(iu[0][best])), label(int(iu[1][best]))
    except Exception:
        pass

    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            imp = np.abs(np.asarray(coef, dtype=np.float64)).reshape(-1) if np.ndim(coef) == 1 else np.abs(np.asarray(coef)).sum(axis=0)
    if imp is not None and np.size(imp) >= 2:
        top2 = np.argsort(np.asarray(imp, dtype=np.float64))[::-1][:2]
        return label(int(top2[0])), label(int(top2[1]))
    return label(0), label(1)


def compose_pdp_2d_figure(
    model: Any,
    X: Any,
    feat_x: Optional[Feature] = None,
    feat_y: Optional[Feature] = None,
    *,
    feat_x_name: Optional[str] = None,
    feat_y_name: Optional[str] = None,
    grid: int = DEFAULT_PDP_GRID,
    sample_rows: int = DEFAULT_PDP_SAMPLE,
    seed: int = 0,
    suptitle: str = "2-D partial dependence",
    figsize: Tuple[float, float] = (7.5, 6.0),
    dpi: int = 110,
) -> Any:
    """Filled-contour 2-D partial-dependence surface for one feature pair; returns a matplotlib Figure.

    When ``feat_x`` / ``feat_y`` are omitted the pair defaults to the top SHAP-interaction pair (tree model + shap
    available) else the top-2 importance features. The surface is the model's average predicted response over the
    (feat_x, feat_y) quantile grid, rendered as a filled contour with overlaid iso-response contour lines and a
    colorbar (predicted prob for a binary classifier, else the predicted value). The non-additive interaction
    residual ratio is annotated in the title so a separable surface (ratio ~0) is distinguishable from a twist.

    Edge-safe: a constant feature collapses its grid to a single point -> an annotated note instead of a degenerate
    contour; a categorical/discrete feature is treated as a discrete grid by the underlying sweep.
    """
    import matplotlib.pyplot as plt

    _, _, names = _as_2d(X)
    n_cols = len(names) if names is not None else (np.asarray(X).reshape(len(X), -1).shape[1])

    if feat_x is None or feat_y is None:
        pair = _default_pair(model, X, names, n_cols)
        if pair is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.axis("off")
            ax.text(0.5, 0.5, "2-D PDP needs >=2 features", ha="center", va="center", fontsize=12)
            return fig
        feat_x = pair[0] if feat_x is None else feat_x
        feat_y = pair[1] if feat_y is None else feat_y

    res = compute_pdp_2d(model, X, (feat_x, feat_y), grid=grid, sample=sample_rows, seed=seed)
    i_x, i_y = res["feature_index"]
    gx, gy, surface = res["grid0"], res["grid1"], res["surface"]
    lab_x = feat_x_name or _feat_label(feat_x, names, i_x)
    lab_y = feat_y_name or _feat_label(feat_y, names, i_y)
    cbar_label = "predicted P(y=1)" if res["kind"] == "proba" else "predicted value"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if gx.shape[0] < 2 or gy.shape[0] < 2:
        const = lab_x if gx.shape[0] < 2 else lab_y
        ax.axis("off")
        ax.text(0.5, 0.5, f"2-D PDP undefined:\n'{const}' is constant (single grid point)", ha="center", va="center", fontsize=12)
        fig.suptitle(suptitle)
        return fig

    # surface rows index feat_x (gx), cols index feat_y (gy); plot feat_x on the x-axis -> transpose to (gy, gx).
    Z = surface.T
    GX, GY = np.meshgrid(gx, gy)
    levels = 14
    cf = ax.contourf(GX, GY, Z, levels=levels, cmap="viridis")
    cs = ax.contour(GX, GY, Z, levels=levels, colors="white", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)

    metr = interaction_residual(surface)
    ax.set_xlabel(lab_x)
    ax.set_ylabel(lab_y)
    ax.set_title(f"{lab_x} x {lab_y}  (interaction residual ratio = {metr['residual_ratio']:.2f})")
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


__all__ = ["compose_pdp_2d_figure", "interaction_residual"]
