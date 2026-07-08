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


def _predict_fn(model: Any) -> Tuple[Callable[[np.ndarray], Optional[np.ndarray]], str]:
    """Pick the model's per-row scalar output: ``predict_proba``[:, 1] for a binary classifier, else ``predict``.

    Returns ``(fn, kind)`` where ``kind`` is "proba" / "predict" for the panel y-label. For a multiclass
    ``predict_proba`` (>2 columns) the positive read is ambiguous, so we fall back to ``predict`` (the class /
    value the model would output) rather than guessing a class column.
    """
    proba = getattr(model, "predict_proba", None)
    if callable(proba):
        def fn(arr: np.ndarray) -> Optional[np.ndarray]:
            # A bound ``predict_proba`` is not proof the model is a classifier: mlframe's PartialFitESWrapper always defines the method and only raises at CALL time when wrapping a regressor (no predict_proba / decision_function underneath). Treat that raise like the multiclass case -> return None so _scalar_predict falls back to predict, instead of failing the whole PDP/ICE diagnostic.
            try:
                p = np.asarray(proba(arr))
            except (AttributeError, NotImplementedError, TypeError):
                return None
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
        return _coerce_float_2d(vals), X, names
    arr = _coerce_float_2d(np.asarray(X))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, arr, None


def _coerce_float_2d(vals: np.ndarray) -> np.ndarray:
    """Best-effort 2-D float64 view of a possibly mixed / string / categorical value matrix, used ONLY for the PDP grid
    construction + ICE x-values (the model is always fed the native ``carrier`` frame, never this view). A whole-frame
    ``astype(float64)`` blew up with "could not convert string to float" whenever ANY feature column was string /
    categorical -- taking down the ENTIRE PDP figure (all numeric features' panels included) via the one upfront cast,
    even though the categorical column is usually not one of the drawn top-K features. Numeric columns pass through; a
    non-numeric column is label-encoded (``pd.factorize``) to category codes so the grid still has usable spread."""
    vals = np.asarray(vals)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    if vals.dtype.kind in "fiub":
        return vals.astype(np.float64)
    import pandas as pd
    out = np.empty(vals.shape, dtype=np.float64)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        try:
            out[:, j] = col.astype(np.float64)
        except (ValueError, TypeError):
            codes, _ = pd.factorize(pd.Series(col).astype("string"), use_na_sentinel=True)
            out[:, j] = np.where(codes < 0, np.nan, codes).astype(np.float64)
    return out


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


def _native_row_subset(carrier: Any, idx: np.ndarray) -> Any:
    """Row-subset the carrier in its NATIVE dtypes (category columns stay category), never a whole-frame copy.

    The prediction block must preserve the dtypes the model was trained on -- a LightGBM / CatBoost categorical model
    rejects a float-coerced frame ("categorical_feature do not match" / "could not convert string to float"). So the
    grid sweep substitutes into this native subsample, NOT the float ``vals`` view (which is only for grid / ICE-x).
    """
    if isinstance(carrier, np.ndarray):
        return carrier[idx]
    if hasattr(carrier, "iloc"):  # pandas
        return carrier.iloc[idx]
    if hasattr(carrier, "__getitem__") and type(carrier).__module__.startswith("polars"):
        return carrier[idx]
    return np.asarray(carrier)[idx]


def _carrier_with_categoricals(carrier: Any) -> Any:
    """Cast a pandas carrier's object/string columns to 'category' (new frame via assign -- untouched blocks reused,
    caller frame not mutated) so a categorical model can predict on it. Non-pandas / already-numeric-or-category
    carriers are returned unchanged."""
    try:
        import pandas as pd
    except ImportError:
        return carrier
    if not isinstance(carrier, pd.DataFrame):
        return carrier
    obj_cols = [c for c in carrier.columns if not (carrier[c].dtype.kind in "iufb" or isinstance(carrier[c].dtype, pd.CategoricalDtype))]
    return carrier.assign(**{c: carrier[c].astype("category") for c in obj_cols}) if obj_cols else carrier


def _categorical_grid(carrier: Any, col_name: Optional[str]) -> Tuple[Optional[list], Any]:
    """If ``col_name`` is a categorical column of the (pandas / polars) ``carrier``, return ``(category_labels,
    dtype)`` so the sweep can iterate the NATIVE categories and substitute native labels; else ``(None, None)``.

    Sweeping a numeric grid value into a categorical column produces an invalid model input (CatBoost:
    "cat_features must be integer or string ... =0.0" -- an outright error at Pool build for a string-category
    column, and a native-predict hang for an int-coded one). The labels come straight from the carrier's own
    category set (no float-code round-trip), so the substituted value is always a value the model saw at fit time.
    """
    if col_name is None or isinstance(carrier, np.ndarray):
        return None, None
    try:
        import pandas as pd
        if isinstance(carrier, pd.DataFrame):
            if col_name in carrier.columns and isinstance(carrier[col_name].dtype, pd.CategoricalDtype):
                return list(carrier[col_name].cat.categories), carrier[col_name].dtype
            return None, None
    except ImportError:
        pass
    if type(carrier).__module__.startswith("polars"):
        import polars as pl
        dt = carrier.schema.get(col_name) if hasattr(carrier, "schema") else None
        is_cat = dt is not None and (dt == pl.Categorical or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum)))
        if is_cat and dt is not None:
            labels = carrier[col_name].cat.get_categories().to_list() if dt == pl.Categorical else list(dt.categories)
            return labels, dt
    return None, None


def _substitute_column(carrier_sample: Any, base_vals: Optional[np.ndarray], col_idx: int, value: Any,
                       col_name: Optional[str] = None, categorical_dtype: Any = None) -> Any:
    """Return a model-input block with column ``col_idx`` set to ``value`` for every row.

    For a pandas / polars ``carrier_sample`` (already the native-dtype subsampled rows), set the swept column to
    ``value`` while PRESERVING every other column's dtype (so categorical models predict) -- via ``assign`` /
    ``with_columns`` on the small (sample, n_cols) subsample, never the caller's full frame. When
    ``categorical_dtype`` is supplied the swept column is itself categorical: ``value`` is a native category label
    and is assigned back as that categorical dtype (never a bare float, which breaks categorical model predict).
    For an ndarray carrier the float ``base_vals`` block path is exact and kept.
    """
    if isinstance(carrier_sample, np.ndarray):
        assert base_vals is not None
        block = base_vals.copy()
        block[:, col_idx] = value
        return block
    if hasattr(carrier_sample, "assign"):  # pandas subsample
        import pandas as pd
        name = col_name if col_name is not None else list(carrier_sample.columns)[col_idx]
        if categorical_dtype is not None:
            arr = ([value] * len(carrier_sample)) if np.ndim(value) == 0 else list(value)
            return carrier_sample.assign(**{name: pd.Categorical(arr, dtype=categorical_dtype)})
        return carrier_sample.assign(**{name: value})
    mod = type(carrier_sample).__module__
    if mod.startswith("polars"):  # polars subsample
        import polars as pl
        name = col_name if col_name is not None else carrier_sample.columns[col_idx]
        if categorical_dtype is not None:
            expr = (pl.lit(value) if np.ndim(value) == 0 else pl.Series(name, list(value))).cast(categorical_dtype)
            return carrier_sample.with_columns(expr.alias(name))
        return carrier_sample.with_columns(pl.lit(value).alias(name))
    assert base_vals is not None
    block = base_vals.copy()
    block[:, col_idx] = value
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
    carrier = _carrier_with_categoricals(carrier)  # so categorical models can predict on the substituted block
    n, n_cols = vals.shape
    col_idx = _resolve_feature_index(feature, names, n_cols)
    predict, kind = _scalar_predict(model)

    idx = _subsample_idx(n, sample, seed)
    base = vals[idx]  # (m, n_cols) float view -- grid + ICE-x only
    carrier_sample = _native_row_subset(carrier, idx)  # native-dtype block the model actually predicts on
    _col_name = names[col_idx] if names is not None else None
    _cat_labels, _cat_dtype = _categorical_grid(carrier, _col_name)
    if _cat_labels is not None:
        # Categorical feature: sweep its native categories (display axis = category codes 0..k-1), substituting the
        # native label so the model receives valid categorical input rather than a dtype-breaking float grid value.
        grid_vals = np.arange(len(_cat_labels), dtype=np.float64)
        is_discrete = True
    else:
        grid_vals, is_discrete = _feature_grid(vals[:, col_idx], grid)
    g = grid_vals.shape[0]
    m = base.shape[0]

    # ice_full[k] = predictions of all m rows with the feature pinned to grid_vals[k]; one predict per grid value.
    ice_full = np.empty((g, m), dtype=np.float64)
    for k in range(g):
        if _cat_labels is not None:
            block = _substitute_column(carrier_sample, base, col_idx, _cat_labels[k], col_name=_col_name, categorical_dtype=_cat_dtype)
        else:
            block = _substitute_column(carrier_sample, base, col_idx, float(grid_vals[k]), col_name=_col_name)
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
    carrier = _carrier_with_categoricals(carrier)  # categorical models must predict on native (not float) dtypes
    n, n_cols = vals.shape
    i0 = _resolve_feature_index(features[0], names, n_cols)
    i1 = _resolve_feature_index(features[1], names, n_cols)
    predict, kind = _scalar_predict(model)

    idx = _subsample_idx(n, sample, seed)
    base = vals[idx]
    m = base.shape[0]
    name0 = names[i0] if names is not None else None
    name1 = names[i1] if names is not None else None
    # Categorical dims sweep their native categories (display axis = codes 0..k-1) and substitute native labels, so a
    # categorical model never receives a dtype-breaking float grid value (CatBoost errors / native-predict hangs).
    _cat0_labels, _cat0_dtype = _categorical_grid(carrier, name0)
    _cat1_labels, _cat1_dtype = _categorical_grid(carrier, name1)
    grid0 = np.arange(len(_cat0_labels), dtype=np.float64) if _cat0_labels is not None else _feature_grid(vals[:, i0], grid)[0]
    grid1 = np.arange(len(_cat1_labels), dtype=np.float64) if _cat1_labels is not None else _feature_grid(vals[:, i1], grid)[0]
    g0, g1 = grid0.shape[0], grid1.shape[0]

    # Tile the m sampled rows g1 times; inner feature column is set to grid1 repeated per row-block. One predict per
    # outer-grid value over the (m*g1) tiled block, then mean per inner-grid value -> the g0 x g1 surface. The tiling
    # is done on the NATIVE carrier (preserving category dtype so categorical models predict), falling back to the
    # float block for a bare-ndarray carrier.
    inner_col = np.tile(grid1, m)  # (m*g1,)
    inner_values = np.array([_cat1_labels[int(round(c))] for c in inner_col], dtype=object) if _cat1_labels is not None else inner_col
    surface = np.empty((g0, g1), dtype=np.float64)
    if isinstance(carrier, np.ndarray):
        tiled = np.repeat(base, g1, axis=0)
        tiled[:, i1] = inner_col
        for a in range(g0):
            tiled[:, i0] = float(grid0[a])
            surface[a] = predict(tiled).reshape(m, g1).mean(axis=0)
    else:
        carrier_sample = _native_row_subset(carrier, idx)
        tiled_native = _native_row_subset(carrier_sample, np.repeat(np.arange(m), g1))
        for a in range(g0):
            block = _substitute_column(tiled_native, None, i1, inner_values, col_name=name1, categorical_dtype=_cat1_dtype)
            outer_val = _cat0_labels[a] if _cat0_labels is not None else float(grid0[a])
            block = _substitute_column(block, None, i0, outer_val, col_name=name0, categorical_dtype=_cat0_dtype)
            surface[a] = np.asarray(predict(block)).reshape(m, g1).mean(axis=0)

    return {"grid0": grid0, "grid1": grid1, "surface": surface, "kind": kind, "feature_index": (i0, i1)}


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
    cell_height: float = 5.0,
    seed: int = 0,
) -> FigureSpec:
    """Grid of one-feature PDP/ICE panels for the caller-ranked ``features`` (top-N first).

    ``features`` is consumed in order -- the caller passes its importance-ranked top-N. When ``interaction_pair``
    is given, a 2-D PDP heatmap for that pair is appended as the final panel. Self-contained composer (no token
    template); each panel is one ``pdp_panel`` call.
    """
    if not features:
        return FigureSpec(suptitle=suptitle, panels=((AnnotationPanelSpec(text="compose_pdp_figure: no features"),),), figsize=(8.0, 3.0))
    panels: List[PanelSpec] = [pdp_panel(model, X, f, grid=grid, sample=sample, ice=ice, centered=centered, seed=seed) for f in features]
    # Auto-conclusion in the suptitle: a feature whose PDP mean barely moves has ~no marginal effect on
    # the prediction (the model isn't using it on average -- e.g. the two flat panels the operator
    # spotted). Measure each 1-D panel's PDP-mean range (last series = bold mean) and flag those under
    # 5% of the largest feature's range, so the reader sees the verdict without eyeballing every panel.
    _ranges: list[tuple[str, float]] = []
    for _pnl in panels[: len(features)]:
        _ys = getattr(_pnl, "y", None)
        if isinstance(_ys, tuple) and _ys:
            _mean = np.asarray(_ys[-1], dtype=float)
            if _mean.size and np.isfinite(_mean).any():
                _ranges.append((getattr(_pnl, "xlabel", "?"), float(np.nanmax(_mean) - np.nanmin(_mean))))
    if _ranges:
        _max_rng = max((r for _, r in _ranges), default=0.0)
        _flat = [name for name, r in _ranges if _max_rng > 0 and r < 0.05 * _max_rng]
        if _flat:
            suptitle = f"{suptitle}\nFlat PDP -- ~no marginal effect (range < 5% of top feature): " f"{', '.join(_flat[:6])}"
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
