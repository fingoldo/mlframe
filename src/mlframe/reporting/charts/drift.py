"""Temporal-drift + adversarial-validation diagnostics for time-ordered tabular data.

Four spec builders (each returns a pure-data FigureSpec, no matplotlib/plotly objects):

- ``psi_heatmap``            -- Population Stability Index per feature per time bucket vs a baseline
                               (train slice or rolling): features x time HeatmapPanelSpec with the 0.10 / 0.25
                               triage thresholds. PSI > 0.25 in a feature's later buckets => that feature drifted.
- ``residual_vs_time``       -- regression residual mean +- std per time bin (LinePanelSpec band): bias drift
                               (mean wandering off zero) + variance drift (band widening) over time.
- ``metric_over_time``       -- wraps ``training.evaluation.compute_ml_perf_by_time`` (numpy-fast, byte-identical)
                               into a LinePanelSpec, with per-split / regime shading via vspans.
- ``adversarial_validation`` -- the Kaggle "will my CV transfer" panel: a LightGBM classifier separating
                               train-vs-test (and train-vs-val) rows on a shuffled union; ROC + AUC annotation +
                               top-20 drifting-feature importance bar. AUC ~0.5 => same distribution, AUC >> 0.5 => drift.

All builders are aggregate-first (per-bucket histograms / bincounts), subsample scatters/fits with extremes preserved,
and decimate curves so a 1M-row time-ordered frame stays cheap. New behaviour defaults ON (no opt-in gate).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid, pack_panels
from mlframe.reporting.charts._sampling import subsample_preserving_extremes
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec, PanelSpec,
)

# PSI triage thresholds (DataRobot / H2O / Arize industry standard): < 0.10 stable, 0.10-0.25 moderate shift,
# > 0.25 significant drift. Drawn as marker thresholds on the heatmap colorbar scale.
PSI_MODERATE: float = 0.10
PSI_SIGNIFICANT: float = 0.25
# 10-bin PSI is the canonical choice; baseline bin edges are quantile-based so each baseline bin holds ~10% mass
# (equal-frequency binning makes PSI robust to skewed marginals -- equal-width bins put all mass in one bin on a
# heavy-tailed feature and report 0 drift regardless).
PSI_DEFAULT_BINS: int = 10
# Floor every bucket-bin proportion at this fraction before the log ratio so an empty bucket bin does not blow PSI to
# +inf (the standard PSI epsilon; 1e-4 corresponds to "<1 in 10k" which is below any actionable per-bucket mass).
PSI_EPS: float = 1e-4


def _quantile_edges(baseline: np.ndarray, nbins: int) -> np.ndarray:
    """Equal-frequency bin edges from the baseline distribution.

    Returns ``nbins+1`` strictly-increasing edges with -inf / +inf as the outer edges so any out-of-baseline-range
    value in a later bucket lands in the first / last bin (and thus contributes to PSI) rather than being dropped.
    Degenerate baselines (constant, or fewer distinct values than bins) collapse to as many unique edges as exist.
    """
    finite = baseline[np.isfinite(baseline)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(finite, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([edges[0], edges[0]])
    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _binned_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of finite ``values`` falling in each bin defined by ``edges`` (sums to 1; zeros where empty)."""
    finite = values[np.isfinite(values)]
    nbins = len(edges) - 1
    if finite.size == 0:
        return np.zeros(nbins, dtype=np.float64)
    counts = np.histogram(finite, bins=edges)[0].astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.zeros(nbins, dtype=np.float64)
    return counts / total


def _psi_one(baseline_props: np.ndarray, bucket_props: np.ndarray, eps: float = PSI_EPS) -> float:
    """PSI between a baseline and a bucket proportion vector: sum((b - e) * ln(b / e)) with both floored at eps."""
    e = np.clip(baseline_props, eps, None)
    b = np.clip(bucket_props, eps, None)
    return float(np.sum((b - e) * np.log(b / e)))


def compute_psi_matrix(
    feature_frame: Any,
    timestamps: np.ndarray,
    *,
    baseline_mask: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    n_time_buckets: int = 10,
    nbins: int = PSI_DEFAULT_BINS,
    max_features: int = 40,
) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """PSI per feature (rows) per time bucket (cols) vs a baseline distribution.

    ``feature_frame`` may be a 2-D ndarray, a pandas DataFrame, or a polars DataFrame (columns pulled one at a time as
    ndarrays -- never a whole-frame copy, per the 100GB-frame rule). ``baseline_mask`` selects the reference rows
    (default: the first time bucket, i.e. earliest period == train-like baseline); when given, PSI for every bucket is
    measured against that fixed reference. Time is split into ``n_time_buckets`` equal-count buckets by sorted
    timestamp order so each bucket holds a comparable sample (robust to irregular spacing).

    Aggregate-first: each (feature, bucket) cell is one ``np.histogram`` over that bucket's column slice against the
    baseline's quantile edges -- O(n) per feature, no per-row python. Features are ranked by peak PSI and the top
    ``max_features`` kept so a 500-column frame yields a readable heatmap.

    Returns ``(matrix[n_feat, n_buckets], row_labels, col_labels)``.
    """
    cols, names = _frame_columns(feature_frame, feature_names)
    ts = np.asarray(timestamps)
    n = ts.shape[0]
    if n == 0 or not cols:
        return np.zeros((0, 0), dtype=np.float64), (), ()

    order = np.argsort(ts, kind="stable")
    n_buckets = max(1, min(int(n_time_buckets), n))
    bucket_bounds = np.linspace(0, n, n_buckets + 1).astype(np.int64)
    bucket_of = np.empty(n, dtype=np.int64)
    for b in range(n_buckets):
        bucket_of[order[bucket_bounds[b]:bucket_bounds[b + 1]]] = b

    if baseline_mask is None:
        base_sel = bucket_of == 0
    else:
        base_sel = np.asarray(baseline_mask, dtype=bool)
        if base_sel.shape[0] != n:
            raise ValueError("baseline_mask length must equal the number of rows")

    rows: List[np.ndarray] = []
    peak: List[float] = []
    for col in cols:
        col = np.asarray(col, dtype=np.float64)
        edges = _quantile_edges(col[base_sel], nbins)
        base_props = _binned_proportions(col[base_sel], edges)
        per_bucket = np.empty(n_buckets, dtype=np.float64)
        for b in range(n_buckets):
            per_bucket[b] = _psi_one(base_props, _binned_proportions(col[bucket_of == b], edges))
        rows.append(per_bucket)
        peak.append(float(np.nanmax(per_bucket)) if per_bucket.size else 0.0)

    matrix = np.vstack(rows) if rows else np.zeros((0, n_buckets), dtype=np.float64)
    if matrix.shape[0] > max_features:
        keep = np.argsort(peak)[::-1][:max_features]
        keep = keep[np.argsort(keep)]  # preserve original feature order among the kept set
        matrix = matrix[keep]
        names = tuple(names[i] for i in keep)

    col_labels = tuple(f"t{b}" for b in range(n_buckets))
    return matrix, tuple(names), col_labels


def _frame_columns(
    feature_frame: Any, feature_names: Optional[Sequence[str]]
) -> Tuple[List[np.ndarray], List[str]]:
    """Yield per-column ndarrays + names from ndarray / pandas / polars without copying the whole frame."""
    if hasattr(feature_frame, "columns") and hasattr(feature_frame, "__getitem__") and not isinstance(feature_frame, np.ndarray):
        names = [str(c) for c in feature_frame.columns]
        # polars exposes ``to_numpy`` per Series; pandas ``.values``. Pull one column at a time (narrow ndarray pull).
        cols = []
        for c in feature_frame.columns:
            s = feature_frame[c]
            arr = s.to_numpy() if hasattr(s, "to_numpy") else np.asarray(s)
            cols.append(arr)
        return cols, names
    arr = np.asarray(feature_frame)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(arr.shape[1])]
    return [arr[:, i] for i in range(arr.shape[1])], names


def psi_heatmap(
    feature_frame: Any,
    timestamps: np.ndarray,
    *,
    baseline_mask: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    n_time_buckets: int = 10,
    nbins: int = PSI_DEFAULT_BINS,
    max_features: int = 40,
    title: str = "Feature drift (PSI vs baseline)",
    figsize: Optional[Tuple[float, float]] = None,
) -> FigureSpec:
    """PSI feature x time-bucket drift heatmap (R-12).

    Each cell is the 10-bin PSI of a feature's distribution in that time bucket vs the baseline slice. Color is the raw
    PSI on an RdYlGn_r scale (green = stable, red = drifted); the 0.10 / 0.25 triage thresholds are noted in the title
    and read directly off the colorbar. Aggregate-first per-bucket histograms, so a 1M-row frame is one O(n) pass per
    feature. Returns a single-panel FigureSpec.
    """
    matrix, row_labels, col_labels = compute_psi_matrix(
        feature_frame, timestamps,
        baseline_mask=baseline_mask, feature_names=feature_names,
        n_time_buckets=n_time_buckets, nbins=nbins, max_features=max_features,
    )
    if matrix.size == 0:
        panel: PanelSpec = AnnotationPanelSpec(text="PSI heatmap: no features / rows", title=title)
        return FigureSpec(suptitle="", panels=((panel,),), figsize=figsize or (8.0, 3.0))

    n_feat, n_buckets = matrix.shape
    # cell_text shows the PSI numerically so an operator can read the exact value past the color (red cells matter).
    cell_text = matrix.copy()
    heat = HeatmapPanelSpec(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{title}\n(stable < {PSI_MODERATE:g}; moderate {PSI_MODERATE:g}-{PSI_SIGNIFICANT:g}; drift > {PSI_SIGNIFICANT:g})",
        xlabel="time bucket (earliest -> latest)",
        ylabel="feature",
        colormap="RdYlGn_r",
        cell_text=cell_text,
        text_format=".2f",
        colorbar_label="PSI",
    )
    fs = figsize or (max(8.0, 0.6 * n_buckets + 4.0), max(3.0, 0.32 * n_feat + 1.5))
    return FigureSpec(suptitle="", panels=((heat,),), figsize=fs)


def _time_bucket_edges(ts: np.ndarray, n_buckets: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-count time buckets by sorted timestamp order.

    Returns ``(order, bucket_of, bucket_centers)`` where ``bucket_of[i]`` is row i's bucket index and
    ``bucket_centers`` are the per-bucket mean timestamps (float; used as the x-axis). Equal-count (not equal-width)
    buckets keep each bucket's residual statistics comparably estimated even when timestamps are clustered.
    """
    n = ts.shape[0]
    order = np.argsort(ts, kind="stable")
    nb = max(1, min(int(n_buckets), n))
    bounds = np.linspace(0, n, nb + 1).astype(np.int64)
    bucket_of = np.empty(n, dtype=np.int64)
    centers = np.empty(nb, dtype=np.float64)
    ts_f = ts.astype(np.float64)
    for b in range(nb):
        idx = order[bounds[b]:bounds[b + 1]]
        bucket_of[idx] = b
        centers[b] = float(np.mean(ts_f[idx])) if idx.size else np.nan
    return order, bucket_of, centers


def residual_vs_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    *,
    n_time_buckets: int = 20,
    x_is_time: bool = True,
    title: str = "Regression residual drift over time",
    figsize: Tuple[float, float] = (10.0, 4.0),
) -> FigureSpec:
    """Regression residual mean +- std per time bin (INV-26).

    Residual = y_true - y_pred is bucketed into equal-count time bins; the line is the per-bin mean residual and the
    band is mean +- std. A mean drifting off zero is bias drift (model goes stale); a band that widens over time is
    variance drift (the model's errors grow). A flat zero reference line is overlaid for the eye. Aggregate-first via
    weighted bincount (one O(n) pass for the mean, one for the second moment) -- no per-row python at 1M rows.
    Returns a single-panel FigureSpec.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    ts = np.asarray(timestamps).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp, ts = yt[mask], yp[mask], ts[mask]
    n = yt.size
    if n == 0:
        panel: PanelSpec = AnnotationPanelSpec(text="residual_vs_time: no finite data", title=title)
        return FigureSpec(suptitle="", panels=((panel,),), figsize=figsize)

    resid = yt - yp
    _, bucket_of, centers = _time_bucket_edges(ts, n_time_buckets)
    nb = centers.shape[0]
    counts = np.bincount(bucket_of, minlength=nb).astype(np.float64)
    counts_safe = np.where(counts > 0, counts, 1.0)
    mean = np.bincount(bucket_of, weights=resid, minlength=nb) / counts_safe
    mean_sq = np.bincount(bucket_of, weights=resid * resid, minlength=nb) / counts_safe
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    std = np.sqrt(var)
    empty = counts == 0
    mean[empty] = np.nan
    std[empty] = np.nan

    zero = np.zeros_like(centers)
    line = LinePanelSpec(
        x=centers,
        y=(mean, zero),
        series_labels=("mean residual", "zero"),
        title=title,
        xlabel="time",
        ylabel="residual (y_true - y_pred)",
        line_styles=("lines+markers", "--"),
        colors=("steelblue", "green"),
        x_is_time=x_is_time,
        band=(mean - std, mean + std),
        band_color="steelblue",
        band_label="+/- 1 std",
    )
    return FigureSpec(suptitle="", panels=((line,),), figsize=figsize)


def metric_over_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    *,
    metric: str = "roc_auc",
    freq: str = "D",
    min_samples: int = 100,
    regimes: Optional[Sequence[Tuple[Any, Any, str, str]]] = None,
    regime_alpha: float = 0.12,
    higher_is_better: bool = True,
    title: Optional[str] = None,
    x_is_time: bool = True,
    figsize: Tuple[float, float] = (11.0, 4.0),
    max_vertices: int = 2000,
) -> FigureSpec:
    """Rolling metric per time bucket as a LinePanelSpec with split / regime shading (INV-9).

    Wraps ``training.evaluation.compute_ml_perf_by_time`` (numpy-fast, byte-identical day-divisor path) to compute the
    chosen metric per ``freq`` time bucket, then renders it as a single line with optional shaded ``regimes`` (e.g.
    train / val / test spans, or detected regime changes) via ``vspans``. ``regimes`` is a sequence of
    ``(start, end, color, label)`` where start/end are timestamps (matched onto the bucket x-axis); the label rides in
    the title since vspans are unlabeled. Curves are decimated to ``max_vertices`` so a multi-year daily series stays
    light. Returns a single-panel FigureSpec.
    """
    from mlframe.training.evaluation import compute_ml_perf_by_time

    perf = compute_ml_perf_by_time(y_true, y_pred, timestamps, freq=freq, metric=metric, min_samples=min_samples)
    # Under-populated buckets are kept with a NaN metric (compute_ml_perf_by_time does not drop them); a figure is
    # only meaningful when at least one bucket cleared min_samples and produced a finite metric.
    if perf is None or len(perf) == 0 or metric not in perf.columns or not np.isfinite(perf[metric].to_numpy(dtype=np.float64)).any():
        panel: PanelSpec = AnnotationPanelSpec(
            text=f"metric_over_time: no buckets with >= {min_samples} samples", title=title or metric,
        )
        return FigureSpec(suptitle="", panels=((panel,),), figsize=figsize)

    idx = perf.index
    # Numeric x for the line (nanosecond epoch for timestamps; renderers format ticks via x_is_time). Datetime index
    # converts to int64 ns directly; a non-datetime fallback uses the row ordinal.
    x = idx.values.astype("datetime64[ns]").astype(np.int64).astype(np.float64) if _is_datetime_index(idx) else np.arange(len(idx), dtype=np.float64)
    yvals = perf[metric].to_numpy(dtype=np.float64)

    if x.size > max_vertices:
        keep = np.linspace(0, x.size - 1, max_vertices).astype(np.int64)
        keep = np.unique(keep)
        x, yvals = x[keep], yvals[keep]

    vspans = _regimes_to_vspans(regimes, regime_alpha)
    regime_note = ""
    if regimes:
        regime_note = " | regimes: " + ", ".join(str(r[3]) for r in regimes if len(r) >= 4 and r[3])
    direction = "higher=better" if higher_is_better else "lower=better"
    line = LinePanelSpec(
        x=x,
        y=yvals,
        series_labels=(metric,),
        title=(title or f"{metric} over time ({direction})") + regime_note,
        xlabel="time",
        ylabel=metric,
        line_styles=("lines+markers",),
        colors=("steelblue",),
        x_is_time=x_is_time,
        vspans=vspans,
    )
    return FigureSpec(suptitle="", panels=((line,),), figsize=figsize)


def _is_datetime_index(idx: Any) -> bool:
    """True when a pandas index carries datetime64 values (so we can take .astype('datetime64[ns]')."""
    try:
        return np.issubdtype(np.asarray(idx.values).dtype, np.datetime64)
    except (TypeError, AttributeError):
        return False


def _regimes_to_vspans(
    regimes: Optional[Sequence[Tuple[Any, Any, str, str]]], alpha: float
) -> Optional[Tuple[Tuple[float, float, str, float], ...]]:
    """Convert ``(start, end, color, label)`` regime spans to LinePanelSpec ``vspans`` ``(x0, x1, color, alpha)``.

    start/end are coerced to the same numeric x-scale as the line (datetime -> int64 ns, else float). Labels are
    dropped here (vspans are unlabeled) -- the caller folds them into the title.
    """
    if not regimes:
        return None
    import pandas as pd

    out: List[Tuple[float, float, str, float]] = []
    for span in regimes:
        if len(span) < 3:
            continue
        start, end, color = span[0], span[1], span[2]
        x0 = _coerce_x(start, pd)
        x1 = _coerce_x(end, pd)
        out.append((x0, x1, str(color), float(alpha)))
    return tuple(out) if out else None


def _coerce_x(v: Any, pd: Any) -> float:
    """Coerce a regime boundary to the numeric x-scale: datetime-like -> int64 ns, else float."""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    try:
        return float(pd.Timestamp(v).value)
    except (ValueError, TypeError):
        return float(v)


__all__ = [
    "PSI_MODERATE",
    "PSI_SIGNIFICANT",
    "PSI_DEFAULT_BINS",
    "compute_psi_matrix",
    "psi_heatmap",
    "residual_vs_time",
    "metric_over_time",
]
