"""Partial-dependence (PDP) + individual-conditional-expectation (ICE) panels.

PDP answers "how does the model's prediction move as one feature sweeps its range, marginalising the rest";
ICE shows the same sweep per row so heterogeneous / interaction effects (curves that fan out or cross) are
visible rather than averaged away. A monotone-increasing PDP with sign-correct slope is the headline read; a
flat PDP means the model ignores that feature.

Builders:
- ``compute_pdp(model, X, feature, ...)``  -- one-feature sweep -> grid, PDP mean, ICE matrix (+ optional c-ICE).
- ``compute_pdp_2d(model, X, (f0, f1), ...)`` -- two-feature interaction surface over a grid x grid.
- ``pdp_panel(...)``  -- LinePanelSpec: faint ICE lines + bold PDP mean.
- ``pdp_2d_panel(...)``  -- HeatmapPanelSpec for the interaction surface.
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

import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from ._pdp_carrier import (
    _carrier_with_categoricals,
    _categorical_grid,
    _model_text_feature_names,  # noqa: F401 -- re-export: a meta-test reaches for it through this module
    _substitute_column,
)
import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid, pack_panels
from mlframe.reporting.charts._coerce_shared import coerce_float_2d as _coerce_float_2d
from mlframe.reporting.charts._catboost_guards import catboost_pool_rebuild_risk
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec, PanelSpec,
)

logger = logging.getLogger(__name__)

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
            """Positive-class probability column, or ``None`` when the shape/error signals a fallback to ``predict``."""
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
    return (lambda arr: _scalar_from_predict_output(predict(arr))), "predict"


def _scalar_from_predict_output(out: Any) -> np.ndarray:
    """Reduce a raw ``predict`` output to one scalar per row.

    A bare ``.ravel()`` silently flattens an (n_rows, n_classes) multiclass proba-like output into an
    n_rows*n_classes 1-D array -- caught via a fuzz combo where a multiclass wrapper's ``predict`` returns per-class
    scores, breaking every downstream ``ice_full[k] = ...`` assignment with a shape-mismatch
    (e.g. (5925,) into (1975,)). 2-D output is reduced to the predicted class index (argmax) instead, matching
    the same "the class the model would output" semantic the multiclass proba fallback already documents.
    """
    arr = np.asarray(out)
    if arr.ndim == 2:
        return arr[:, 0].astype(np.float64) if arr.shape[1] == 1 else arr.argmax(axis=1).astype(np.float64)
    return arr.ravel().astype(np.float64)


def _scalar_predict(model: Any) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Wrap ``_predict_fn`` so a multiclass-proba None falls back to ``predict`` transparently."""
    fn, kind = _predict_fn(model)
    predict = getattr(model, "predict", None)

    def call(arr: np.ndarray) -> np.ndarray:
        """Scalar per-row prediction: the proba column when ``fn`` resolves one, else ``predict`` on ``arr``."""
        out = fn(arr)
        if out is None:
            if not callable(predict):
                raise TypeError("multiclass predict_proba but no predict to fall back on")
            return _scalar_from_predict_output(predict(arr))
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


def _resolve_feature_index(feature: Union[int, str], names: Optional[List[str]], n_cols: int) -> int:
    """Resolve a ``feature`` (positional int index or column name) to a validated column index."""
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
    """Sorted row indices for a size-``sample`` random subsample (all ``n`` rows if ``n<=sample``); sorted so
    downstream native row-subsetting preserves original row order."""
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


# Row cap for the batched grid predict (_predict_grid_batched): bounds the transient (grid*sample, n_cols)
# stacked block so a pathological caller-supplied grid/sample can't blow up memory on a very wide frame.
# 200k matches the row-cap convention used elsewhere in this reporting package (e.g. adversarial_validation's
# ADV_MAX_ROWS_PER_SIDE) -- comfortably above the default grid=20 x sample=2000=40_000.
_PDP_BATCH_MAX_ROWS = 200_000


def _predict_grid_batched(
    predict: Callable[[Any], np.ndarray],
    carrier_sample: Any,
    base: np.ndarray,
    col_idx: int,
    grid_vals: np.ndarray,
    cat_labels: Optional[list],
    m: int,
    g: int,
    col_name: Optional[str],
    categorical_dtype: Any,
) -> Optional[np.ndarray]:
    """Predict all ``g`` grid steps in ONE call instead of ``g`` separate ones.

    Stacks ``g`` row-major copies of the ``m``-row sample into one ``(g*m, n_cols)`` block (row-block ``k``
    gets the swept column pinned to grid step ``k``, matching the per-step loop's semantics exactly), predicts
    ONCE, and reshapes to ``(g, m)``. Purely a fusion of the existing per-step substitution logic -- same
    values fed to the same stateless ``predict`` callable, so the result is bit-identical to the per-step
    loop; only the number of ``predict``/Pool-construction calls changes (g -> 1). This matters most for
    tree-model wrappers (CatBoost/LightGBM) whose ``predict`` pays a fixed per-call Pool/Dataset construction
    cost that dominates at wide (500+ column) frames -- profiled at ~350ms/call fixed overhead, independent
    of row count, so consolidating g calls into 1 removes (g-1)/g of it.

    Falls back to the per-step loop (returns ``None``) when the stacked block would exceed
    ``_PDP_BATCH_MAX_ROWS`` rows, or for a carrier type this fusion does not special-case.
    """
    if g * m > _PDP_BATCH_MAX_ROWS:
        return None
    if cat_labels is not None:
        # A categorical/discrete grid sweep -- NOT batched, ever (falls straight to the per-step loop below).
        # Investigating an intermittent "Windows fatal exception: access violation" (stack rooted
        # in catboost's Pool._init, triggered from a PDP predict call) surfaced on
        # tests/training/test_core.py's TestPolarsNativeFastpath / TestTextAndEmbeddingFeatures classes (a
        # polars carrier, CatBoost fit with cat_features=[...]). IMPORTANT: this crash reproduced via BOTH
        # this batched path AND the original (pre-fix) per-step loop below -- it is a pre-existing bug
        # in compute_pdp's CatBoost/polars predict path, not something this batching optimization introduced,
        # and this restriction is NOT proven to fix it (it only narrows this function's own contribution to
        # the risk surface). It reproduced 2/4 times on an identical repro command and did NOT reproduce on
        # isolated raw-catboost/raw-polars minimal repros outside the full mlframe pipeline -- consistent with
        # a native memory-safety/GC-timing race rather than a deterministic logic bug, and matches the
        # signature of two prior fixed incidents of the same crash class (d0d7fa7de: PDP/ICE crashed CatBoost
        # on a text-feature column; c825c0c8b: SHAP crashed CatBoost with embedding_features) -- both were
        # "downstream code feeds CatBoost a carrier that doesn't match what the model registered at fit time".
        # The exact trigger for THIS incident is not yet pinned down; kept as an open, tracked issue. Skipping
        # batching for categorical sweeps still removes the one path (a large multi-thousand-row repeated-
        # category block predicted in one call) that plausibly amplifies whatever the underlying race is,
        # while keeping the validated numeric-sweep win (test_catboost_trains_on_mixed_dtypes, 214s -> 93s
        # PDP time, which swept mostly numeric top-importance features) fully intact.
        return None
    # categorical_dtype is always None here (the categorical branch above already returned); every remaining
    # path is a purely numeric column substitution.
    repeat_vals = np.repeat(grid_vals, m)
    if isinstance(carrier_sample, np.ndarray):
        big = np.tile(base, (g, 1))
        big[:, col_idx] = repeat_vals
        return np.asarray(predict(big), dtype=np.float64).reshape(g, m)
    import pandas as pd

    if isinstance(carrier_sample, pd.DataFrame):
        big = pd.concat([carrier_sample] * g, ignore_index=True)
        name = col_name if col_name is not None else list(big.columns)[col_idx]
        big[name] = repeat_vals
        return np.asarray(predict(big), dtype=np.float64).reshape(g, m)
    if type(carrier_sample).__module__.startswith("polars"):
        import polars as pl

        # pl.concat defaults to rechunk=False: concatenating the SAME frame g times produces a
        # multi-chunk column whose g chunks all alias the identical underlying Arrow buffer. Every
        # native segfault caught chasing this crash (4 independent CI failures, 3.9/3.11/3.13,
        # TestTextAndEmbeddingFeatures) traced to this exact line -- consistent with CatBoost's
        # native embedding/text-column extraction not being safe against multi-chunk (let alone
        # buffer-aliased) Arrow input, matching two prior fixed incidents of the same class
        # (d0d7fa7de, c825c0c8b: CatBoost fed a carrier that doesn't match what it registered at
        # fit time). rechunk=True materialises ONE contiguous, non-aliased buffer before the
        # column substitution below, at the cost of one extra copy of an already-small
        # (g*sample, n_cols) block -- negligible next to the predict call it feeds.
        big = pl.concat([carrier_sample] * g, rechunk=True)
        name = col_name if col_name is not None else carrier_sample.columns[col_idx]
        expr = pl.Series(name, repeat_vals)
        big = big.with_columns(expr.alias(name))
        return np.asarray(predict(big), dtype=np.float64).reshape(g, m)
    return None  # unrecognised carrier type -> caller falls back to the per-step loop


def _set_column_inplace(block: Any, col_idx: int, value: Any, col_name: Optional[str] = None, categorical_dtype: Any = None) -> Any:
    """Mutate a pandas ``block``'s single column in place instead of taking the ``_substitute_column`` copy path.

    ``DataFrame.assign``/``.copy()`` walks every column's block manager even when only one column actually
    changes (profiled: on a 500+-column frame, half the PDP sweep's wall time was these per-grid-step full-frame
    copies, not the predict calls themselves) -- a direct violation of the project's no-whole-frame-copy-per-step
    convention. ``block`` here is always a private working copy the caller made once before the sweep loop (never
    the caller's original frame), so mutating it in place is safe: each grid step's predict reads the column
    right after this call, and the next step overwrites the same column again, so no restore is needed."""
    import pandas as pd

    name = col_name if col_name is not None else list(block.columns)[col_idx]
    if categorical_dtype is not None:
        arr = ([value] * len(block)) if np.ndim(value) == 0 else list(value)
        block[name] = pd.Categorical(arr, dtype=categorical_dtype)
    else:
        block[name] = value
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
    carrier = _carrier_with_categoricals(carrier, model)  # so categorical models can predict on the substituted block
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

    # ice_full[k] = predictions of all m rows with the feature pinned to grid_vals[k]. Tries the ONE-CALL batched
    # path first (_predict_grid_batched): stacks all g grid steps into one (g*m, n_cols) block and predicts once,
    # eliminating g-1 of the g fixed-per-call Pool/Dataset construction costs a tree-model wrapper's predict()
    # otherwise pays on every grid step (dominant at 500+-column frames -- see that function's docstring).
    ice_full = _predict_grid_batched(predict, carrier_sample, base, col_idx, grid_vals, _cat_labels, m, g, _col_name, _cat_dtype)
    if ice_full is None:
        # Fallback: per-step loop (pathological grid*sample size, or an unrecognised carrier type). A pandas
        # carrier gets ONE private working copy mutated in place per grid step (see _set_column_inplace) --
        # avoids g full-frame `.assign()` copies on a wide model-input frame. ndarray/polars keep the existing
        # per-step _substitute_column path (ndarray's own copy is already O(1)-column; polars' with_columns is
        # already columnar-cheap for a single column, so there is no equivalent whole-frame-copy cost to avoid there).
        import pandas as pd
        _pd_working = carrier_sample.copy() if isinstance(carrier_sample, pd.DataFrame) else None

        ice_full = np.empty((g, m), dtype=np.float64)
        for k in range(g):
            _value = _cat_labels[k] if _cat_labels is not None else float(grid_vals[k])
            if _pd_working is not None:
                block = _set_column_inplace(_pd_working, col_idx, _value, col_name=_col_name, categorical_dtype=_cat_dtype)
            else:
                block = _substitute_column(carrier_sample, base, col_idx, _value, col_name=_col_name, categorical_dtype=_cat_dtype)
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
    carrier = _carrier_with_categoricals(carrier, model)  # categorical models must predict on native (not float) dtypes
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
    inner_values = np.array([_cat1_labels[round(c)] for c in inner_col], dtype=object) if _cat1_labels is not None else inner_col
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
        # i1's column is CONSTANT across the g0 outer loop (only i0 varies per outer step) -- the prior form
        # re-substituted it on every iteration anyway, paying a redundant full-frame copy each time on a pandas
        # carrier. Substitute it once, then (for pandas) mutate i0's column in place per step instead of taking
        # another full-frame `.assign()` copy (see _set_column_inplace) -- halves the per-step copy count and
        # removes the O(n_cols) cost from the remaining ones.
        import pandas as pd
        if isinstance(tiled_native, pd.DataFrame):
            block = tiled_native.copy()
            block = _set_column_inplace(block, i1, inner_values, col_name=name1, categorical_dtype=_cat1_dtype)
            for a in range(g0):
                outer_val = _cat0_labels[a] if _cat0_labels is not None else float(grid0[a])
                block = _set_column_inplace(block, i0, outer_val, col_name=name0, categorical_dtype=_cat0_dtype)
                surface[a] = np.asarray(predict(block)).reshape(m, g1).mean(axis=0)
        else:
            block = _substitute_column(tiled_native, None, i1, inner_values, col_name=name1, categorical_dtype=_cat1_dtype)
            for a in range(g0):
                outer_val = _cat0_labels[a] if _cat0_labels is not None else float(grid0[a])
                surface[a] = np.asarray(
                    predict(_substitute_column(block, None, i0, outer_val, col_name=name0, categorical_dtype=_cat0_dtype))
                ).reshape(m, g1).mean(axis=0)

    return {"grid0": grid0, "grid1": grid1, "surface": surface, "kind": kind, "feature_index": (i0, i1)}


def _feat_label(feature: Union[int, str], names: Optional[List[str]], idx: int) -> str:
    """Display label for a panel title/axis: the given name if ``feature`` is already a string, else the resolved
    column name, else a positional fallback ``f{idx}``."""
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
    risk = catboost_pool_rebuild_risk(model)
    if risk:
        return AnnotationPanelSpec(
            text=f"PDP / ICE not computed: {risk}",
            title="PDP / ICE",
        )
    res = compute_pdp(model, X, feature, grid=grid, sample=sample, ice=ice, centered=centered, seed=seed)
    label = _feat_label(feature, names, res["feature_index"])
    gv = res["grid"]
    if gv.shape[0] < 2:
        return AnnotationPanelSpec(text=f"PDP undefined for '{label}'\n(feature is constant)", title=f"PDP: {label}")

    ylab = "predicted P(y=1)" if res["kind"] == "proba" else "prediction"
    ice_draw = res["ice_centered"] if (centered and res["ice_centered"] is not None) else res["ice"]
    pdp_curve = res["pdp"] - res["pdp"][0] if centered else res["pdp"]

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

    support_note = ""
    try:
        _arr, _, _ = _as_2d(X)
        _vals = np.asarray(_arr[:, res["feature_index"]], dtype=np.float64)
        _vals = _vals[np.isfinite(_vals)]
        if _vals.size and gv.shape[0] > 1:
            _edges = (gv[1:] + gv[:-1]) / 2.0
            _counts = np.bincount(np.searchsorted(_edges, _vals), minlength=gv.shape[0])
            _empty = int(np.sum(_counts == 0))
            _thinnest = int(_counts.min())
            support_note = (f"; grid support: thinnest point holds {_thinnest:,} of {_vals.size:,} rows"
                            + (f", {_empty} of {gv.shape[0]} grid points hold none" if _empty else ""))
    except Exception as exc:  # best-effort enrichment: a caveat must never break the panel
        logger.debug("pdp support note failed (%s: %s)", type(exc).__name__, exc)

    return LinePanelSpec(
        x=gv,
        y=tuple(series),
        series_labels=tuple(labels),
        title=f"PDP / ICE: {label}{support_note}",
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
    risk = catboost_pool_rebuild_risk(model)
    if risk:
        return AnnotationPanelSpec(text=f"2-D PDP not computed: {risk}", title="2-D partial dependence")
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
        cell_hovertext=_pdp_2d_support_text(X, i0, i1, grid0, grid1),
    )


def pdp_2d_support_counts(X: Any, i0: int, i1: int, grid0: np.ndarray, grid1: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
    """``(counts, n_total)`` of rows falling in each 2-D PDP grid cell, or ``None`` when it cannot be computed.

    Each row is assigned to the NEAREST grid value on each axis (midpoints between consecutive grid values are the
    bin edges), which matches how a reader interprets a cell: the region the cell stands for.
    """
    try:
        arr, _, _ = _as_2d(X)
        v0 = np.asarray(arr[:, i0], dtype=np.float64)
        v1 = np.asarray(arr[:, i1], dtype=np.float64)
        finite = np.isfinite(v0) & np.isfinite(v1)
        v0, v1 = v0[finite], v1[finite]
        n_total = int(v0.size)
        if n_total == 0:
            return None
        e0 = (np.asarray(grid0, dtype=np.float64)[1:] + np.asarray(grid0, dtype=np.float64)[:-1]) / 2.0
        e1 = (np.asarray(grid1, dtype=np.float64)[1:] + np.asarray(grid1, dtype=np.float64)[:-1]) / 2.0
        r = np.searchsorted(e0, v0)
        c = np.searchsorted(e1, v1)
        counts = np.zeros((len(grid0), len(grid1)), dtype=np.int64)
        np.add.at(counts, (r, c), 1)
        return counts, n_total
    except Exception as exc:  # best-effort: a support overlay must never break the chart
        logger.debug("pdp_2d support counting failed (%s: %s)", type(exc).__name__, exc)
        return None


def _pdp_2d_support_text(X: Any, i0: int, i1: int, grid0: np.ndarray, grid1: np.ndarray) -> Optional[np.ndarray]:
    """Per-cell ``"N rows (P%)"`` support strings for a 2-D PDP surface, or ``None`` if it can't be computed.

    A partial-dependence surface is evaluated on a REGULAR grid, so it reports a value for every cell --
    including combinations the training data barely contains or never contains at all. Those cells are pure
    model extrapolation and read exactly like well-supported ones. Putting the real row count behind the
    tooltip lets a reader tell "this interaction is real" from "this corner of the grid has 3 rows in it".

    Counting assigns each row to the nearest grid value on each axis (the grid is what the surface was
    evaluated on), which matches how a reader interprets a cell: the region the cell stands for.
    """
    try:
        got = pdp_2d_support_counts(X, i0, i1, grid0, grid1)
        if got is None:
            return None
        counts, n_total = got
        pct = counts / float(n_total) * 100.0
        return np.array(
            [[f"{counts[i, j]:,} rows ({pct[i, j]:.1f}% of {n_total:,})" for j in range(counts.shape[1])] for i in range(counts.shape[0])],
            dtype=object,
        )
    except Exception as exc:  # best-effort enrichment: a tooltip must never break the chart
        logger.debug("pdp_2d support-text computation failed (%s: %s); tooltip falls back to axes+value", type(exc).__name__, exc)
        return None


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
        caption=(
            "How to read: the bold line is the average predicted response as ONE feature is swept with the others "
            "held at their observed values; the faint lines are individual rows. Fanning ICE lines mean the effect "
            "depends on the rest of the row, so the average curve hides as much as it shows. These are model "
            "behaviour, not causal effects, and the sweep manufactures feature combinations the data may never "
            "contain -- treat the far ends of the grid with suspicion."
        ),
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
