"""SHAP post-fit diagnostic panels (compute-gated).

Two matplotlib-native panels off ONE explainer + ONE SHAP-value computation:

* the beeswarm (summary) -- every feature's |SHAP| distribution, ranked top-down by mean |SHAP|
  so the model's headline drivers sit at the top;
* the top-K dependence scatters PACKED INTO A GRID (>= ``DEPENDENCE_GRID_COLS`` panels per figure, not one
  stacked figure per feature) -- each panel shows how that feature's value maps to its SHAP contribution and
  carries an AUTO-INTERPRETATION in its title: the monotone direction (Spearman of value vs SHAP), the SHAP
  impact range, and whether the curve is SMOOTH (a simple directional effect) or STEP/DISCONTINUOUS (a sharp
  learned threshold, with the threshold value marked) or non-monotone (interaction / mixed effect).

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

from mlframe.reporting.charts._layout import figsize_for_grid, pack_panels  # noqa: F401  (pack_panels re-exported for grid composers / parity with pdp_ice)
from mlframe.reporting.charts._sampling import subsample_preserving_extremes

logger = logging.getLogger(__name__)

# Default row cap for the FAST (tree) path. TreeExplainer cost is ~linear in rows (per-row tree traversal), so this cap
# is the wall-time lever; the beeswarm density + the mean-|SHAP| feature ranking SATURATE well below 20k. Set to 8k
# (was 20k): measured on a 15-feature GBDT the top-6 dependence-panel feature set is IDENTICAL at 8k vs 20k and the
# mean-|SHAP| importances match to plotting tolerance, while the explain cost drops ~2.4x (3.78s -> 1.57s). The
# stratified subsample already force-keeps the high-|score| tail rows, so shrinking the random bulk does not drop the
# extreme points the beeswarm exists to show.
DEFAULT_MAX_ROWS: int = 8_000
# Number of top-|SHAP| features to draw a dependence scatter for. The beeswarm shows all; dependence
# plots past ~6 become an unreadable wall of small panels.
DEFAULT_TOP_K: int = 6
# Tail rows force-kept by the stratified subsample (the high-|score-proxy| rows the metrics quote).
_K_EXTREMES: int = 200
# KernelExplainer is SLOW; the opt-in path caps both the background summary and the explained rows hard.
KERNEL_BACKGROUND: int = 100
KERNEL_MAX_ROWS: int = 500

# Dependence panels are packed into a grid (>= this many per figure) instead of one-per-figure stacked
# vertically, so a top_k=6 run is ONE 2x3 figure rather than six stacked plots wasting screen height.
DEPENDENCE_GRID_COLS: int = 2
# A dependence curve is judged DISCONTINUOUS/STEP when the largest local jump in the value-sorted, smoothed
# SHAP curve exceeds this fraction of the curve's overall SHAP range -- i.e. the model has a sharp threshold
# in that feature rather than a gradual effect.
_STEP_JUMP_FRAC: float = 0.33
# Below this many finite (value, shap) points a per-feature interpretation is unreliable, so the panel just
# scatters without a verdict (never raises -- best-effort diagnostic).
_MIN_INTERP_POINTS: int = 20

# Substrings in a model's class MRO that mark a tree / GBDT model TreeExplainer handles exactly+fast.
_TREE_MARKERS: Tuple[str, ...] = (
    "randomforest", "extratrees", "decisiontree", "gradientboosting", "histgradientboosting",
    "xgb", "lgbm", "lightgbm", "catboost", "isolationforest", "baggingclassifier", "baggingregressor",
)


@dataclass
class ShapPanelsResult:
    """Outcome of :func:`shap_summary_and_dependence`.

    ``figures`` are the (still-open unless closed) matplotlib figures in draw order: beeswarm first,
    then the grouped dependence-grid figure(s) (top-K panels packed >= ``DEPENDENCE_GRID_COLS`` per
    figure, each auto-interpreted). ``paths`` are the files written (parallel to the plot kinds). ``top_features`` / ``mean_abs_shap`` rank the features by mean |SHAP| (descending).
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


_CATBOOST_MULTI_OUTPUT_LOSSES: frozenset = frozenset({"multilogloss", "multicrossentropy"})


def _is_multi_output_catboost(model: Any) -> bool:
    """True iff ``model`` is a CatBoost estimator trained with a multi-output leaf-value loss (MultiLogloss /
    MultiCrossEntropy, the multilabel losses) -- see the caller for why shap's CatBoost parser crashes on these.
    Detected via ``get_all_params()['loss_function']`` (present on every fitted CatBoost estimator); any lookup
    failure (non-CatBoost model, unfitted, API drift) is treated as "not the risky case" so this never masks an
    unrelated model type as tree/non-tree.
    """
    get_params = getattr(model, "get_all_params", None)
    if not callable(get_params):
        return False
    try:
        loss = str(get_params().get("loss_function", "")).lower()
    except Exception:
        return False
    return loss in _CATBOOST_MULTI_OUTPUT_LOSSES


def _coerce_float_2d(vals: np.ndarray) -> np.ndarray:
    """Best-effort 2-D float64 view of a (possibly mixed / string / categorical) value matrix.

    Only used for the residual-proxy tail-selection and the per-feature dependence x-values, so it must never crash on a
    non-numeric column: a whole-frame ``astype(float64)`` blew up with ``could not convert string to float`` whenever the
    model was trained with a string/categorical feature, which silently disabled the ENTIRE SHAP panel (the caller's
    broad except swallowed it). Numeric columns pass through; a non-numeric column is label-encoded (``pd.factorize``)
    to category codes so the dependence x-axis still has usable spread instead of taking down the diagnostic.
    """
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
            # factorize returns -1 for missing; map that to NaN so the dependence scatter drops it rather than plotting a
            # spurious -1 category.
            out[:, j] = np.where(codes < 0, np.nan, codes).astype(np.float64)
    return out


def _carrier_with_categoricals(X: Any) -> Any:
    """Return ``X`` with object/string columns cast to pandas 'category' dtype (a new frame via assign -- untouched
    numeric blocks are reused, so no whole-frame copy on a possibly-huge carrier; the caller's frame is not mutated).

    ``shap.TreeExplainer`` runs the model's own ``predict`` internally, and LightGBM/other tree backends reject raw
    object/string feature columns ("could not convert string to float" / "categorical_feature do not match") -- they
    need 'category' dtype (the form they were trained on). Without this the ENTIRE SHAP panel was silently absent
    (dispatcher broad-except) for any model trained with string categoricals (surfaced by the 300k fuzz-profile loop
    on MRMR-engineered / string-cat combos). Non-pandas carriers and already-numeric/category columns are untouched.
    """
    import pandas as pd
    if not isinstance(X, pd.DataFrame):
        return X
    obj_cols = [c for c in X.columns if not (X[c].dtype.kind in "iufb" or isinstance(X[c].dtype, pd.CategoricalDtype))]
    if not obj_cols:
        return X
    return X.assign(**{c: X[c].astype("category") for c in obj_cols})


def _as_frame_and_names(X: Any, feature_names: Optional[Sequence[str]]) -> Tuple[Any, np.ndarray, List[str]]:
    """Return ``(carrier_for_shap, values_2d, names)``.

    ``carrier_for_shap`` keeps the original frame flavour where it has columns (shap indexes
    column-by-column and labels axes from it); a bare ndarray carrier is returned as-is. ``values_2d``
    is the float view used only for the residual proxy + dependence x-values, never a frame copy. Object/string
    columns are cast to 'category' so shap's internal tree-model predict accepts them (see ``_carrier_with_categoricals``).
    """
    if hasattr(X, "columns") and not isinstance(X, np.ndarray):
        names = [str(c) for c in X.columns]
        vals = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return _carrier_with_categoricals(X), _coerce_float_2d(vals), (list(feature_names) if feature_names is not None else names)
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_cols = arr.shape[1]
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_cols)]
    return arr, _coerce_float_2d(arr), names


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
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in shap_panels.py:200: %s", e)
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


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (rank-Pearson) of ``x`` vs ``y``; 0.0 when either side is constant / degenerate.

    Rank-based so it reports the MONOTONE direction of the feature-value -> SHAP relationship regardless of
    curvature -- a smooth saturating curve still reads near +/-1, which is exactly the directional read wanted.
    """
    n = x.shape[0]
    if n < 2:
        return 0.0
    def _ranks(v: np.ndarray) -> np.ndarray:
        """Stable ordinal ranks via single argsort + scatter (bit-identical to argsort(argsort(v, kind=
        "mergesort"), kind="mergesort"), ~1.7-1.9x faster -- the second sort was pure waste)."""
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        return r

    rx = _ranks(x)
    ry = _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:  # one side is constant (all ranks equal after centring)
        return 0.0
    return float(np.sum(rx * ry) / denom)


@dataclass
class _DepInterp:
    """Auto-interpretation of one feature's SHAP-vs-value dependence curve."""

    direction: float  # Spearman corr of feature value vs its SHAP (sign = effect direction)
    shap_range: float  # max - min SHAP for this feature (impact magnitude)
    shape: str  # "smooth (monotone)" / "non-monotone" / "step/discontinuous"
    step_value: Optional[float]  # approximate feature value where the sharpest jump sits (step shape only)
    enough: bool  # False -> too few finite points for a reliable verdict


def _interpret_dependence(feat_vals: np.ndarray, shap_vals: np.ndarray) -> _DepInterp:
    """Classify a feature's SHAP-vs-value curve: monotone direction, impact range, and SMOOTH vs STEP shape.

    Robust by construction (never raises): drops non-finite pairs, skips the verdict below ``_MIN_INTERP_POINTS``.
    Sorts SHAP by feature value and smooths with a small running mean (de-noises the scatter so a real threshold
    stands out from per-point jitter); a STEP/discontinuity is the largest local first-difference of that smoothed
    curve exceeding ``_STEP_JUMP_FRAC`` of the overall SHAP range. A near +/-1 Spearman with NO dominant jump is a
    simple directional effect (smooth monotone); a dominant jump means the model learned a sharp threshold there.
    """
    fv = np.asarray(feat_vals, dtype=np.float64).ravel()
    sv = np.asarray(shap_vals, dtype=np.float64).ravel()
    m = np.isfinite(fv) & np.isfinite(sv)
    fv, sv = fv[m], sv[m]
    n = fv.shape[0]
    rng = float(sv.max() - sv.min()) if n else 0.0
    if n < _MIN_INTERP_POINTS or rng <= 0.0:
        return _DepInterp(direction=0.0, shap_range=rng, shape="insufficient data", step_value=None, enough=False)

    direction = _spearman(fv, sv)
    order = np.argsort(fv, kind="mergesort")
    fv_s, sv_s = fv[order], sv[order]
    # Running-mean smooth so per-point scatter (interaction noise) doesn't masquerade as a discontinuity; window
    # scales with n but stays small so a genuine threshold survives.
    win = max(3, min(15, n // 20))
    kernel = np.ones(win, dtype=np.float64) / win
    smooth = np.convolve(sv_s, kernel, mode="valid")
    if smooth.size < 2:
        smooth = sv_s
    diffs = np.abs(np.diff(smooth))
    jmax = int(np.argmax(diffs)) if diffs.size else 0
    max_jump = float(diffs[jmax]) if diffs.size else 0.0
    jump_frac = max_jump / rng if rng > 0 else 0.0

    if jump_frac >= _STEP_JUMP_FRAC:
        # Map the smoothed-curve jump index back to an approximate feature value (smoothing trims win-1 points).
        pos = min(jmax + win // 2, fv_s.shape[0] - 1)
        return _DepInterp(direction=direction, shap_range=rng, shape="step/discontinuous", step_value=float(fv_s[pos]), enough=True)
    if abs(direction) >= 0.6:
        return _DepInterp(direction=direction, shap_range=rng, shape="smooth (monotone)", step_value=None, enough=True)
    return _DepInterp(direction=direction, shap_range=rng, shape="non-monotone", step_value=None, enough=True)


def _interp_title(name: str, interp: _DepInterp) -> str:
    """Two-line panel title: the feature name, then its plain-language verdict (direction / impact / shape)."""
    if not interp.enough:
        return f"{name}\n(too few finite points -- no verdict)"
    sign = "+" if interp.direction >= 0 else "-"
    arrow = "increasing" if interp.direction >= 0 else "decreasing"
    if interp.shape == "step/discontinuous":
        verdict = f"STEP at x~{interp.step_value:.3g} -- sharp threshold"
    elif interp.shape == "smooth (monotone)":
        verdict = f"smooth monotone ({arrow}) -- simple directional effect"
    else:
        verdict = "non-monotone -- interaction / mixed effect"
    return f"{name}\n{sign}dir rho={interp.direction:+.2f}, impact={interp.shap_range:.3g}; {verdict}"


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
            except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate; nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in shap_panels.py:382: %s", e)
                pass


def _dependence_grid_figs(
    shap_mat: np.ndarray,
    vals_sample: np.ndarray,
    cols: Sequence[int],
    names: Sequence[str],
    top_names: Sequence[str],
    *,
    max_cols: int = DEPENDENCE_GRID_COLS,
    panels_per_fig: Optional[int] = None,
) -> List[Any]:
    """Build grouped dependence-grid figure(s): each panel scatters a feature's value vs its SHAP, auto-annotated.

    Replaces the prior one-figure-per-feature stack with a packed grid (``max_cols`` per row). The per-panel title
    carries the auto-interpretation from :func:`_interpret_dependence` (direction / impact / smooth-vs-step), so the
    reader gets the conclusion without eyeballing every curve. Robust: a feature with too few finite points is still
    drawn (scatter) but labelled "no verdict"; never raises.
    """
    if plt is None or len(cols) == 0:
        return []
    # All requested panels go in ONE figure by default (a top_k of 4-6 is a clean 2x2 / 2x3); callers wanting a
    # hard cap per figure pass ``panels_per_fig``.
    chunk = panels_per_fig if panels_per_fig and panels_per_fig > 0 else len(cols)
    figs: List[Any] = []
    for start in range(0, len(cols), chunk):
        block = list(cols[start : start + chunk])
        ranks = list(range(start, start + len(block)))
        n_panels = len(block)
        n_cols = min(max_cols, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize_for_grid(n_rows, n_cols, cell_width=5.0, cell_height=4.0), squeeze=False)
        flat_axes = axes.ravel()
        for slot, (rank, col) in enumerate(zip(ranks, block)):
            ax = flat_axes[slot]
            feat_vals = np.asarray(vals_sample[:, col], dtype=np.float64).ravel()
            shap_vals = np.asarray(shap_mat[:, col], dtype=np.float64).ravel()
            name = top_names[rank] if rank < len(top_names) else (names[col] if col < len(names) else f"f{col}")
            interp = _interpret_dependence(feat_vals, shap_vals)
            finite = np.isfinite(feat_vals) & np.isfinite(shap_vals)
            ax.scatter(feat_vals[finite], shap_vals[finite], s=8, alpha=0.4, color="#3182bd", edgecolors="none")
            ax.axhline(0.0, color="#969696", lw=0.6, ls="--")
            if interp.enough and interp.shape == "step/discontinuous" and interp.step_value is not None:
                ax.axvline(interp.step_value, color="#de2d26", lw=1.0, ls=":")
            ax.set_title(_interp_title(name, interp), fontsize=8)
            ax.set_xlabel(name, fontsize=8)
            ax.set_ylabel("SHAP value", fontsize=8)
            ax.tick_params(labelsize=7)
        for empty in flat_axes[n_panels:]:  # blank the padding cells in a partial last row
            empty.set_visible(False)
        fig.suptitle("SHAP dependence -- feature value vs SHAP (auto-interpreted)", fontsize=10)
        try:
            fig.tight_layout(rect=(0, 0, 1, 0.97))
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in shap_panels.py:437: %s", e)
            pass
        figs.append(fig)
    return figs


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
    if tree and _is_multi_output_catboost(model):
        # shap's CatBoost tree parser (shap/explainers/_tree.py TreeEnsemble.get_trees) assumes ONE scalar leaf
        # value per tree and rebuilds the binary-index arrays (children_left/children_right/...) via a `2**counter`
        # size formula derived from len(leaf_values). A CatBoost MultiLogloss (multilabel) model's leaf_values are a
        # FLAT list of n_leaves*n_labels entries (multiple values per leaf), so that formula silently computes the
        # wrong node count, producing malformed/wrongly-sized child-index arrays -- SingleTree.__init__ then walks
        # them and reads out of bounds, crashing the WHOLE PROCESS with a native access violation (caught live via a
        # fuzz combo: models=('cb',) target=multilabel_classification -- confirmed deterministic at a fixed seed,
        # and the crash happens before any Python-catchable exception, so this must be caught BEFORE
        # shap.TreeExplainer(model) is ever constructed). Skip rather than risk the crash.
        return ShapPanelsResult(
            [], [], [], np.empty(0), "none",
            skipped="CatBoost multi-output (MultiLogloss) leaf values are not supported by shap's TreeExplainer "
            "CatBoost parser -- constructing it risks a native crash (see shap_panels.py comment)",
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
        try:
            shap_values = explainer(X_sample, check_additivity=False)
        except Exception as _shap_exc:
            # Tree-SHAP on categorical / text / non-pandas frames is fragile across shap + backend versions. shap runs
            # the model's OWN predict internally, so the carrier must be in the exact form each backend expects: shap
            # may float-convert a string-categorical column (ValueError "could not convert string to float"); LightGBM
            # raises on a category-set mismatch ("categorical_feature do not match"); CatBoost raises its own
            # CatBoostError on a polars String / text feature column ("Unsupported data type String for a numerical
            # feature column"). The carrier is cast to 'category' upstream, which fixes the common pandas-categorical
            # cases (validated on LightGBM / XGBoost / CatBoost), but text features and polars-String carriers are
            # genuinely out of SHAP's scope here. Rather than let ANY backend error raise into the dispatcher's broad
            # except (a noisy ERROR + a vanished panel with no reason), degrade to a clean skip carrying the cause --
            # this is a best-effort VISUAL diagnostic, never fatal. Broad ``except Exception`` is deliberate: the
            # backends raise library-specific error types (CatBoostError etc.), not a common base beyond Exception.
            return ShapPanelsResult(
                [], [], [], np.empty(0), "none",
                skipped=f"tree SHAP unavailable for this feature frame ({type(_shap_exc).__name__}: {str(_shap_exc)[:80]})",
            )
        explainer_kind = "tree"
    else:
        bg = shap.sample(X_sample, min(kernel_background, len(idx)), random_state=seed)
        # A bound ``predict_proba`` is not proof of a classifier: mlframe's PartialFitESWrapper always defines the method and raises at CALL time when wrapping a regressor. Probe once on the tiny background sample and fall back to ``predict`` so KernelSHAP works for regression models instead of raising mid-explain.
        predict = getattr(model, "predict_proba", None) or getattr(model, "predict")
        if predict is not getattr(model, "predict", None):
            try:
                predict(bg[:1])
            except (AttributeError, NotImplementedError, TypeError):
                predict = getattr(model, "predict")
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

        # Dependence panels GROUPED into grid figure(s) (>= DEPENDENCE_GRID_COLS per row) instead of one stacked
        # figure per feature, each panel auto-annotated with its monotone direction / impact / smooth-vs-step verdict.
        cols = order[: max(int(top_k), 1)].tolist()
        for fig_rank, dep_fig in enumerate(_dependence_grid_figs(shap_mat, vals_sample, cols, names, top_names)):
            figures.append(dep_fig)
            if plot_file:
                paths.extend(_save_figure(dep_fig, _base_for(plot_file, f"shap_dependence_grid{fig_rank}"), plot_outputs))
    finally:
        # Close EVERY figure shap opened (not just the ones we tracked) so a mid-flow error never leaks.
        leaked = [plt.figure(num) for num in plt.get_fignums() if num not in figs_before]
        _close_figs(leaked or figures)

    return ShapPanelsResult(figures, paths, top_names, mean_abs, explainer_kind)


def _safe(name: str) -> str:
    """Filename-safe feature name (alnum / underscore / dash). Retained for sibling shap_per_instance reuse."""
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name))[:48]


def _base_for(plot_file: str, suffix: str) -> str:
    """Compose a per-panel base path: ``<root>_<suffix><ext>`` so each panel writes a distinct file."""
    root, ext = os.path.splitext(plot_file)
    return f"{root}_{suffix}{ext}"


__all__ = [
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TOP_K",
    "DEPENDENCE_GRID_COLS",
    "KERNEL_BACKGROUND",
    "KERNEL_MAX_ROWS",
    "ShapPanelsResult",
    "is_tree_model",
    "shap_summary_and_dependence",
]
