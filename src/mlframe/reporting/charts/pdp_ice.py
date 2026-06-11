"""Partial-dependence (PDP) + individual-conditional-expectation (ICE) panels.

PDP answers "how does the model's prediction move as one feature sweeps its range, marginalising the rest";
ICE shows the same sweep per row so heterogeneous / interaction effects (curves that fan out or cross) are
visible rather than averaged away. A monotone-increasing PDP with sign-correct slope is the headline read; a
flat PDP means the model ignores that feature.

Builders:
- ``compute_pdp(model, X, feature, ...)``   -- one-feature sweep -> grid, PDP mean, ICE matrix (+ optional c-ICE).
- ``compute_pdp_2d(model, X, (f0, f1), ...)`` -- two-feature interaction surface over a grid x grid.
- ``pdp_panel(...)``                          -- LinePanelSpec: faint ICE lines + bold PDP mean.
- ``pdp_2d_panel(...)``                       -- HeatmapPanelSpec for the interaction surface.
- ``compose_pdp_figure(model, X, features, ...)`` -- a grid of the top-N caller-ranked features.

Efficiency contract (the prediction call is the only cost that scales with data):
- rows are subsampled to ``sample`` (default 2000) BEFORE any prediction -- ICE needs at most a few hundred
  legible curves and the PDP mean converges far below 2000 rows;
- every grid point is ONE predict call over the whole (sample) row block (the feature column is broadcast to
  the grid value), never a per-row predict -- so the total prediction work is ``grid`` calls, independent of n;
- the 2-D surface is ``g0`` predict calls over the (sample * g1) tiled block (one call per outer-grid value),
  i.e. ``g0`` predictions, never ``g0*g1`` per-cell predictions;
- only the (grid, sample) ICE matrix lives in the spec, never a length-n array.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid, pack_panels
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec, PanelSpec,
)

# Row cap for ICE / PDP. The PDP mean converges far below this and a legible ICE plot needs only a few hundred
# curves; capping here makes the per-grid predict cost independent of n.
DEFAULT_PDP_SAMPLE: int = 2_000
# Default sweep resolution for a continuous feature (quantile grid).
DEFAULT_PDP_GRID: int = 20
# Above this many distinct ICE curves the per-row lines become an unreadable blob, so only this many are drawn
# (a uniform subsample of the computed ICE rows); the PDP mean is still over ALL sampled rows.
ICE_CURVE_DRAW_CAP: int = 200
# A feature with at most this many distinct values is treated as discrete (grid = its categories) rather than
# continuous (quantile grid). Mirrors the low-cardinality cat heuristic used elsewhere in the suite.
DISCRETE_MAX_UNIQUE: int = 12


def _predict_fn(model: Any) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Pick the model's per-row scalar output: ``predict_proba``[:, 1] for a binary classifier, else ``predict``.

    Returns ``(fn, kind)`` where ``kind`` is "proba" / "predict" for the panel y-label. For a multiclass
    ``predict_proba`` (>2 columns) the positive read is ambiguous, so we fall back to ``predict`` (the class /
    value the model would output) rather than guessing a class column.
    """
    proba = getattr(model, "predict_proba", None)
    if callable(proba):
        def fn(arr: np.ndarray) -> np.ndarray:
            p = np.asarray(proba(arr))
            if p.ndim == 2 and p.shape[1] == 2:
                return p[:, 1]
            if p.ndim == 1:
                return p
            return None  # multiclass / unexpected shape -> caller falls back to predict
        # Probe shape once on a tiny slice is avoided (a predict call has side-effect cost); instead detect at call
        # time and signal multiclass via a None return that the wrapper below converts to a predict fallback.
        return fn, "proba"
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError("model must expose predict_proba or predict")
    return (lambda arr: np.asarray(predict(arr)).ravel()), "predict"


def _scalar_predict(model: Any) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Wrap ``_predict_fn`` so a multiclass-proba None falls back to ``predict`` transparently."""
    fn, kind = _predict_fn(model)
    predict = getattr(model, "predict", None)

    def call(arr: np.ndarray) -> np.ndarray:
        out = fn(arr)
        if out is None:
            if not callable(predict):
                raise TypeError("multiclass predict_proba but no predict to fall back on")
            return np.asarray(predict(arr)).ravel()
        return np.asarray(out, dtype=np.float64).ravel()

    return call, ("predict" if kind == "predict" else "proba")


def _as_2d(X: Any) -> Tuple[np.ndarray, Any, Optional[List[str]]]:
    """Return ``(values_2d, carrier, feature_names)``.

    ``carrier`` is the original frame type so a column can be substituted in the model's expected input format
    (a pandas / polars model is fed a frame of the same flavour, not a bare ndarray). For an ndarray input the
    carrier is the ndarray itself. Feature names come from pandas / polars columns when present.
    """
    if hasattr(X, "columns") and not isinstance(X, np.ndarray):
        names = [str(c) for c in X.columns]
        vals = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return np.asarray(vals, dtype=np.float64), X, names
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, arr, None


def _resolve_feature_index(feature: Union[int, str], names: Optional[List[str]], n_cols: int) -> int:
    if isinstance(feature, (int, np.integer)):
        idx = int(feature)
        if not (0 <= idx < n_cols):
            raise IndexError(f"feature index {idx} out of range for {n_cols} columns")
        return idx
    if names is None:
        raise ValueError(f"feature name {feature!r} given but X has no column names")
    if feature not in names:
        raise ValueError(f"feature {feature!r} not in columns {names}")
    return names.index(feature)


def _feature_grid(col: np.ndarray, grid: int) -> Tuple[np.ndarray, bool]:
    """Sweep grid for one feature: its distinct values when discrete, else an equal-frequency quantile grid.

    Returns ``(grid_values, is_discrete)``. A quantile grid (vs equal-width) keeps points where the data is so a
    skewed feature is swept where it actually varies instead of across an empty tail.
    """
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return np.array([0.0]), True
    uniq = np.unique(finite)
    if uniq.size <= max(2, min(int(grid), DISCRETE_MAX_UNIQUE)) and uniq.size <= DISCRETE_MAX_UNIQUE:
        return uniq.astype(np.float64), True
    qs = np.linspace(0.0, 1.0, int(grid))
    g = np.unique(np.quantile(finite, qs))
    return g.astype(np.float64), False


def _subsample_idx(n: int, sample: int, seed: int) -> np.ndarray:
    if n <= sample:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size=sample, replace=False))


def _substitute_column(carrier: Any, base_vals: np.ndarray, col_idx: int, value: float) -> Any:
    """Return a model-input block (carrier's flavour) with column ``col_idx`` set to ``value`` for every row.

    ``base_vals`` is the (sample, n_cols) ndarray of the subsampled rows. For an ndarray carrier we build the
    substituted block directly; for a pandas / polars carrier we hand back a frame of the same type so the model's
    feature-name / dtype expectations hold. The substitution copies only the (sample, n_cols) block, never the
    caller's full frame.
    """
    block = base_vals.copy()
    block[:, col_idx] = value
    return _wrap_like(carrier, block)


def _wrap_like(carrier: Any, block: np.ndarray) -> Any:
    """Wrap a (sample, n_cols) ndarray as the carrier's frame type (pandas / polars / ndarray)."""
    mod = type(carrier).__module__
    if mod.startswith("pandas"):
        import pandas as pd
        return pd.DataFrame(block, columns=list(carrier.columns))
    if mod.startswith("polars"):
        import polars as pl
        return pl.DataFrame(block, schema=list(carrier.columns))
    return block


def compute_pdp(
    model: Any,
    X: Any,
    feature: Union[int, str],
    *,
    grid: int = DEFAULT_PDP_GRID,
    sample: int = DEFAULT_PDP_SAMPLE,
    ice: bool = True,
    centered: bool = False,
    seed: int = 0,
) -> dict:
    """One-feature partial dependence + ICE.

    Subsamples to ``sample`` rows, then for each grid value sets the feature column to that value across all
    sampled rows and predicts ONCE (vectorized over rows) -- ``grid`` predict calls total, independent of n.

    Returns a dict:
        ``grid``      : (G,) sweep values (quantile grid for continuous, categories for discrete)
        ``pdp``       : (G,) mean prediction at each grid value (the PDP, mean over all sampled rows)
        ``ice``       : (n_draw, G) per-row prediction curves (subsampled to ICE_CURVE_DRAW_CAP for drawing) or None
        ``ice_centered`` : (n_draw, G) c-ICE (each row's curve minus its value at the first grid point) or None
        ``is_discrete`` : bool
        ``kind``      : "proba" / "predict" (model output read)
        ``feature_index`` : resolved column index
    """
    vals, carrier, names = _as_2d(X)
    n, n_cols = vals.shape
    col_idx = _resolve_feature_index(feature, names, n_cols)
    predict, kind = _scalar_predict(model)

    idx = _subsample_idx(n, sample, seed)
    base = vals[idx]  # (m, n_cols) -- the only length<=sample copy we hold
    grid_vals, is_discrete = _feature_grid(vals[:, col_idx], grid)
    g = grid_vals.shape[0]
    m = base.shape[0]

    # ice_full[k] = predictions of all m rows with the feature pinned to grid_vals[k]; one predict per grid value.
    ice_full = np.empty((g, m), dtype=np.float64)
    for k in range(g):
        block = _substitute_column(carrier, base, col_idx, float(grid_vals[k]))
        ice_full[k] = predict(block)

    pdp = ice_full.mean(axis=1)  # PDP mean over ALL sampled rows (not the drawn subset)

    ice_curves = None
    ice_centered = None
    if ice:
        ice_mat = ice_full.T  # (m, g)
        if m > ICE_CURVE_DRAW_CAP:
            draw = np.sort(np.random.default_rng(seed + 1).choice(m, size=ICE_CURVE_DRAW_CAP, replace=False))
            ice_mat = ice_mat[draw]
        ice_curves = ice_mat
        if centered:
            ice_centered = ice_mat - ice_mat[:, :1]

    return {
        "grid": grid_vals,
        "pdp": pdp,
        "ice": ice_curves,
        "ice_centered": ice_centered,
        "is_discrete": bool(is_discrete),
        "kind": kind,
        "feature_index": col_idx,
    }


def compute_pdp_2d(
    model: Any,
    X: Any,
    features: Tuple[Union[int, str], Union[int, str]],
    *,
    grid: int = DEFAULT_PDP_GRID,
    sample: int = DEFAULT_PDP_SAMPLE,
    seed: int = 0,
) -> dict:
    """Two-feature partial-dependence interaction surface over a grid0 x grid1 mesh.

    For each value of the outer feature we substitute it across a (sample * g1)-row block that tiles every inner
    grid value over the sampled rows, predict ONCE, and average per inner-grid value -- so the surface costs
    ``g0`` predictions (not ``g0 * g1`` per-cell predictions). Returns ``grid0`` (rows), ``grid1`` (cols),
    ``surface`` (g0 x g1 mean predictions), ``kind``.
    """
    vals, carrier, names = _as_2d(X)
    n, n_cols = vals.shape
    i0 = _resolve_feature_index(features[0], names, n_cols)
    i1 = _resolve_feature_index(features[1], names, n_cols)
    predict, kind = _scalar_predict(model)

    idx = _subsample_idx(n, sample, seed)
    base = vals[idx]
    m = base.shape[0]
    grid0, _ = _feature_grid(vals[:, i0], grid)
    grid1, _ = _feature_grid(vals[:, i1], grid)
    g0, g1 = grid0.shape[0], grid1.shape[0]

    # Tile the m sampled rows g1 times; inner feature column is set to grid1 repeated per row-block. One predict per
    # outer-grid value over the (m*g1) tiled block, then mean per inner-grid value -> the g0 x g1 surface.
    tiled = np.repeat(base, g1, axis=0)  # (m*g1, n_cols), row r,j at index r*g1 + j
    inner_col = np.tile(grid1, m)        # (m*g1,)
    tiled[:, i1] = inner_col
    surface = np.empty((g0, g1), dtype=np.float64)
    for a in range(g0):
        tiled[:, i0] = float(grid0[a])
        preds = predict(_wrap_like(carrier, tiled)).reshape(m, g1)
        surface[a] = preds.mean(axis=0)

    return {"grid0": grid0, "grid1": grid1, "surface": surface, "kind": kind,
            "feature_index": (i0, i1)}


def _feat_label(feature: Union[int, str], names: Optional[List[str]], idx: int) -> str:
    if isinstance(feature, str):
        return feature
    if names is not None and 0 <= idx < len(names):
        return names[idx]
    return f"f{idx}"


def pdp_panel(
    model: Any,
    X: Any,
    feature: Union[int, str],
    *,
    grid: int = DEFAULT_PDP_GRID,
    sample: int = DEFAULT_PDP_SAMPLE,
    ice: bool = True,
    centered: bool = False,
    seed: int = 0,
) -> PanelSpec:
    """LinePanelSpec for one feature: faint per-row ICE curves under the bold PDP mean.

    When ``centered`` is set the ICE curves are c-ICE (each anchored to 0 at the first grid point) so pure
    interaction shape is comparable across rows of different baselines. A degenerate single-point grid (constant
    feature) returns an AnnotationPanelSpec.
    """
    _, _, names = _as_2d(X)
    res = compute_pdp(model, X, feature, grid=grid, sample=sample, ice=ice, centered=centered, seed=seed)
    label = _feat_label(feature, names, res["feature_index"])
    gv = res["grid"]
    if gv.shape[0] < 2:
        return AnnotationPanelSpec(text=f"PDP undefined for '{label}'\n(feature is constant)", title=f"PDP: {label}")

    ylab = "predicted P(y=1)" if res["kind"] == "proba" else "prediction"
    ice_draw = res["ice_centered"] if (centered and res["ice_centered"] is not None) else res["ice"]
    pdp_curve = res["pdp"] - res["pdp"][0] if (centered and ice_draw is not None) else res["pdp"]

    series: List[np.ndarray] = []
    styles: List[str] = []
    colors: List[str] = []
    labels: List[str] = []
    if ice_draw is not None:
        for row in ice_draw:
            series.append(row)
            styles.append("-")
            colors.append("#9ecae1")  # faint blue ICE
            labels.append("")
    series.append(pdp_curve)
    styles.append("-")
    colors.append("#08519c")  # bold dark-blue PDP mean
    labels.append("PDP (mean)" + (" [centered]" if centered else ""))

    style_for_discrete = "lines+markers" if res["is_discrete"] else None
    if style_for_discrete is not None:
        styles = [style_for_discrete if s == "-" else s for s in styles]

    return LinePanelSpec(
        x=gv,
        y=tuple(series),
        series_labels=tuple(labels),
        title=f"PDP / ICE: {label}",
        xlabel=label,
        ylabel=ylab + (" (centered)" if centered else ""),
        line_styles=tuple(styles),
        colors=tuple(colors),
    )


def pdp_2d_panel(
    model: Any,
    X: Any,
    features: Tuple[Union[int, str], Union[int, str]],
    *,
    grid: int = DEFAULT_PDP_GRID,
    sample: int = DEFAULT_PDP_SAMPLE,
    seed: int = 0,
) -> PanelSpec:
    """HeatmapPanelSpec of the two-feature partial-dependence interaction surface (rows = f0, cols = f1)."""
    _, _, names = _as_2d(X)
    res = compute_pdp_2d(model, X, features, grid=grid, sample=sample, seed=seed)
    i0, i1 = res["feature_index"]
    lab0 = _feat_label(features[0], names, i0)
    lab1 = _feat_label(features[1], names, i1)
    grid0, grid1, surface = res["grid0"], res["grid1"], res["surface"]
    cbar = "P(y=1)" if res["kind"] == "proba" else "prediction"
    return HeatmapPanelSpec(
        matrix=surface,
        row_labels=tuple(f"{v:.3g}" for v in grid0),
        col_labels=tuple(f"{v:.3g}" for v in grid1),
        title=f"2-D PDP: {lab0} x {lab1}",
        xlabel=lab1,
        ylabel=lab0,
        colormap="viridis",
        colorbar_label=cbar,
    )


def compose_pdp_figure(
    model: Any,
    X: Any,
    features: Sequence[Union[int, str]],
    *,
    grid: int = DEFAULT_PDP_GRID,
    sample: int = DEFAULT_PDP_SAMPLE,
    ice: bool = True,
    centered: bool = False,
    interaction_pair: Optional[Tuple[Union[int, str], Union[int, str]]] = None,
    suptitle: str = "Partial dependence / ICE",
    max_cols: int = 2,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
    seed: int = 0,
) -> FigureSpec:
    """Grid of one-feature PDP/ICE panels for the caller-ranked ``features`` (top-N first).

    ``features`` is consumed in order -- the caller passes its importance-ranked top-N. When ``interaction_pair``
    is given, a 2-D PDP heatmap for that pair is appended as the final panel. Self-contained composer (no token
    template); each panel is one ``pdp_panel`` call.
    """
    if not features:
        return FigureSpec(suptitle=suptitle, panels=((AnnotationPanelSpec(text="compose_pdp_figure: no features"),),), figsize=(8.0, 3.0))
    panels: List[PanelSpec] = [
        pdp_panel(model, X, f, grid=grid, sample=sample, ice=ice, centered=centered, seed=seed) for f in features
    ]
    if interaction_pair is not None:
        panels.append(pdp_2d_panel(model, X, interaction_pair, grid=grid, sample=sample, seed=seed))
    packed = pack_panels(panels, max_cols=max_cols)
    n_rows = len(packed)
    n_cols = max_cols if packed else 0
    return FigureSpec(
        suptitle=suptitle,
        panels=packed,
        figsize=figsize_for_grid(n_rows, n_cols, cell_width=cell_width, cell_height=cell_height),
    )


__all__ = [
    "DEFAULT_PDP_SAMPLE",
    "DEFAULT_PDP_GRID",
    "ICE_CURVE_DRAW_CAP",
    "DISCRETE_MAX_UNIQUE",
    "compute_pdp",
    "compute_pdp_2d",
    "pdp_panel",
    "pdp_2d_panel",
    "compose_pdp_figure",
]
