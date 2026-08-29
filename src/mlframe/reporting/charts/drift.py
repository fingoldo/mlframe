"""Temporal-drift + adversarial-validation diagnostics for time-ordered tabular data.

Five spec builders (each returns a pure-data FigureSpec, no matplotlib/plotly objects):

- ``psi_heatmap``  -- Population Stability Index per feature per time bucket vs a baseline
                               (train slice or rolling): features x time HeatmapPanelSpec with the 0.10 / 0.25
                               triage thresholds. PSI > 0.25 in a feature's later buckets => that feature drifted.
- ``residual_vs_time``  -- regression residual mean +- std per time bin (LinePanelSpec band): bias drift
                               (mean wandering off zero) + variance drift (band widening) over time.
- ``cusum_residual_drift``  -- two-sided tabular CUSUM of standardized residuals: catches a SUSTAINED mean shift
                               (structural break) earlier than per-bucket residual_vs_time, since a small persistent
                               bias accumulates past the control limit before any single bucket's mean looks abnormal.
- ``metric_over_time``  -- wraps ``training.evaluation.compute_ml_perf_by_time`` (numpy-fast, byte-identical)
                               into a LinePanelSpec, with per-split / regime shading via vspans.
- ``adversarial_validation`` -- the Kaggle "will my CV transfer" panel: a LightGBM classifier separating
                               train-vs-test (and train-vs-val) rows on a shuffled union; ROC + AUC annotation +
                               top-20 drifting-feature importance bar. AUC ~0.5 => same distribution, AUC >> 0.5 => drift.

All builders are aggregate-first (per-bucket histograms / bincounts), subsample scatters/fits with extremes preserved,
and decimate curves so a 1M-row time-ordered frame stays cheap. New behaviour defaults ON (no opt-in gate).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# ``_adversarial_auc_bar`` is imported from HERE by tests/reporting/test_charts_statistical_regressions.py,
# so the carve keeps it reachable at its old home.
from ._drift_shared import _adversarial_auc_bar  # noqa: F401
from ._drift_shared import (
    ADV_MAX_ROWS_PER_SIDE, ADV_N_ESTIMATORS, ADV_TOP_FEATURES, _frame_columns,
)
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec, PanelSpec,
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
# Past this many cells, per-cell numeric annotations overlap into an unreadable smear; the confusion-matrix and
# multiclass heatmaps apply the same ceiling.
_HEATMAP_CELL_TEXT_MAX: int = 400


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
    return np.asarray(counts / total)


def _psi_one(baseline_props: np.ndarray, bucket_props: np.ndarray, eps: float = PSI_EPS) -> float:
    """PSI between a baseline and a bucket proportion vector: sum((b - e) * ln(b / e)) with both floored at eps."""
    e = np.clip(baseline_props, eps, None)
    b = np.clip(bucket_props, eps, None)
    return float(np.sum((b - e) * np.log(b / e)))


def _psi_verdict(matrix, noise_floor: float, row_labels) -> str:
    """One-line pass/fail over the whole PSI grid: how many features drift, and which is worst."""
    per_feature = np.nanmax(matrix, axis=1) if matrix.size else np.empty(0)
    real = per_feature[np.isfinite(per_feature)]
    if real.size == 0:
        return " -- no feature has a computable PSI"
    bar = max(PSI_SIGNIFICANT, noise_floor)
    drifting = int(np.sum(real > bar))
    w = int(np.nanargmax(per_feature))
    worst = f"{row_labels[w]} (peak PSI {per_feature[w]:.2f})" if w < len(row_labels) else f"peak PSI {per_feature[w]:.2f}"
    if drifting == 0:
        return f" -- no feature drifts past {bar:.2f}; worst is {worst}"
    return f" -- {drifting} of {real.size} features drift past {bar:.2f}; worst is {worst}"


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
        bucket_of[order[bucket_bounds[b] : bucket_bounds[b + 1]]] = b

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
        base_vals = col[base_sel]
        edges = _quantile_edges(base_vals, nbins)
        per_bucket = np.empty(n_buckets, dtype=np.float64)
        if edges.size < 3:
            # A baseline with fewer than two distinct finite values has no distribution to compare against: every
            # later value falls in the single [-inf, +inf] bin and PSI is identically 0. Reporting 0 there said
            # "stable" about a feature that was constant during the baseline and exploded afterwards -- the exact
            # case this chart exists to catch. NaN renders blank, which is the honest answer.
            per_bucket[:] = np.nan
        else:
            base_props = _binned_proportions(base_vals, edges)
            # One contiguous sweep over the time-sorted column instead of a full-n boolean mask + copy per bucket:
            # the `order` permutation and `bucket_bounds` above already describe every bucket as a slice.
            col_sorted = col[order]
            for b in range(n_buckets):
                block = col_sorted[bucket_bounds[b] : bucket_bounds[b + 1]]
                per_bucket[b] = _psi_one(base_props, _binned_proportions(block, edges))
        rows.append(per_bucket)
        peak.append(float(np.nanmax(per_bucket)) if per_bucket.size and np.isfinite(per_bucket).any() else 0.0)

    matrix = np.vstack(rows) if rows else np.zeros((0, n_buckets), dtype=np.float64)
    if matrix.shape[0] > max_features:
        keep = np.argsort(peak)[::-1][:max_features]
        keep = keep[np.argsort(keep)]  # preserve original feature order among the kept set
        matrix = matrix[keep]
        names = [names[i] for i in keep]

    # Bucket size belongs on the axis: PSI's null expectation is ~(nbins-1)/n_bucket, so the same cell value means
    # "drifted" at 5000 rows and "pure noise" at 50. Without the count on the label the reader cannot tell which.
    col_labels = tuple(f"t{b}\n(n={int(bucket_bounds[b + 1] - bucket_bounds[b]):,})" for b in range(n_buckets))
    return matrix, tuple(names), col_labels


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
    """PSI feature x time-bucket drift heatmap.

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
    n_rows_total = int(np.asarray(timestamps).shape[0])
    n_per_bucket = max(1, n_rows_total // max(1, n_buckets))
    # PSI is a chi-square-like statistic: under NO drift its expectation is about (bins-1)/n_bucket. At 50 rows per
    # bucket that floor is ~0.18, well past the "moderate" line, so i.i.d. data painted the grid orange.
    psi_noise_floor = (max(1, int(nbins)) - 1) / float(n_per_bucket)
    # cell_text shows the PSI numerically so an operator can read the exact value past the color (red cells matter),
    # but past a few hundred cells the numbers overlap into an unreadable smear -- the same ceiling the confusion
    # matrix and multiclass heatmaps already apply.
    cell_text = matrix.copy() if matrix.size <= _HEATMAP_CELL_TEXT_MAX else None
    suppressed = "" if cell_text is not None else f"; per-cell values hidden above {_HEATMAP_CELL_TEXT_MAX} cells"
    heat = HeatmapPanelSpec(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title=(
            f"{title}{_psi_verdict(matrix, psi_noise_floor, row_labels)}\n(stable < {PSI_MODERATE:g}; moderate {PSI_MODERATE:g}-{PSI_SIGNIFICANT:g}; "
            f"drift > {PSI_SIGNIFICANT:g}; no-drift noise floor at {n_per_bucket:,} rows/bucket "
            f"= {psi_noise_floor:.3f}{suppressed})"
        ),
        xlabel="time bucket (earliest -> latest)",
        ylabel="feature",
        colormap="RdYlGn_r",
        cell_text=cell_text,
        text_format=".2f",
        colorbar_label="PSI",
        # Iso-PSI triage contours: the renderer draws a line only where the heatmap crosses 0.10 / 0.25, so the
        # moderate / significant drift boundaries are visible directly on the grid rather than read off the colorbar.
        threshold_contours=(
            (PSI_MODERATE, "orange", "dash", f"moderate {PSI_MODERATE:g}"),
            (PSI_SIGNIFICANT, "red", "solid", f"significant {PSI_SIGNIFICANT:g}"),
        ),
    )
    fs = figsize or (max(8.0, 0.6 * n_buckets + 4.0), max(3.0, 0.32 * n_feat + 1.5))
    n_flagged = int(np.nansum(matrix > PSI_SIGNIFICANT))
    n_blank = int(np.isnan(matrix).sum())
    caption = (
        f"PSI compares each feature's distribution in a time bucket against its baseline-period distribution using "
        f"{nbins} equal-frequency baseline bins: 0 = identical, > {PSI_MODERATE:g} moderate, "
        f"> {PSI_SIGNIFICANT:g} significant. PSI is INFLATED at small bucket sizes -- its no-drift expectation is "
        f"about (bins-1)/rows-per-bucket, which is {psi_noise_floor:.3f} here, so read any cell below that as noise. "
        f"Column t0 is the baseline compared against itself and is 0 by construction. "
        f"{n_flagged} feature-buckets exceed {PSI_SIGNIFICANT:g}"
        + (f"; {n_blank} cells are blank (baseline constant, so PSI is undefined for that feature)." if n_blank else ".")
    )
    return FigureSpec(suptitle="", panels=((heat,),), figsize=fs, caption=caption)


def _format_x(v: float) -> str:
    """Render an x-axis position as a date when the axis carries epoch nanoseconds, else as a plain number."""
    # Epoch-ns timestamps are the only values reaching this module at ~1e18; anything smaller is a real number.
    if np.isfinite(v) and abs(v) > 1e15:
        import datetime as _dt

        return _dt.datetime.fromtimestamp(v / 1e9, tz=_dt.timezone.utc).strftime("%Y-%m-%d")
    return f"{v:.6g}"


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
        idx = order[bounds[b] : bounds[b + 1]]
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
    """Regression residual mean +- std per time bin.

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
    # Two-pass centred variance, NOT E[x^2]-E[x]^2. Residuals of a price/revenue model sit far from zero relative to
    # their spread, and the raw-moment form then subtracts two nearly-equal large numbers: at centre 1e8 / true std
    # 0.01 it returned per-bucket stds of [5.1, 0, 0, 0, 7.6]. The clip to zero made that silent -- a negative computed
    # variance became a zero-width band, which reads as "this bucket's errors are perfectly consistent".
    centred = resid - mean[bucket_of]
    var = np.bincount(bucket_of, weights=centred * centred, minlength=nb) / counts_safe
    std = np.sqrt(np.clip(var, 0.0, None))
    empty = counts == 0
    mean[empty] = np.nan
    std[empty] = np.nan

    zero = np.zeros_like(centers)
    # The docstring names both failure modes (bias drift and spread drift) but the title was the caller's static
    # string, so the reader had to eyeball the very comparison the per-bucket statistics already answer.
    third = max(1, nb // 3)
    first_ok, last_ok = np.isfinite(mean[:third]), np.isfinite(mean[-third:])
    if first_ok.any() and last_ok.any():
        d_bias = float(np.nanmean(mean[-third:]) - np.nanmean(mean[:third]))
        d_spread = float(np.nanmean(std[-third:]) - np.nanmean(std[:third]))
        drift_note = f" (last third vs first: bias {d_bias:+.3g}, spread {d_spread:+.3g})"
    else:
        drift_note = " (too few populated buckets to compare the ends of the timeline)"
    line = LinePanelSpec(
        x=centers,
        y=(mean, zero),
        series_labels=("mean residual", "zero"),
        title=title + drift_note,
        xlabel="time",
        ylabel="residual (y_true - y_pred)",
        line_styles=("lines+markers", "--"),
        colors=("steelblue", "green"),
        x_is_time=x_is_time,
        band=(mean - std, mean + std),
        band_color="steelblue",
        band_label="+/- 1 std",
    )
    caption = (
        "Line = mean residual (y_true - y_pred) per equal-count time bucket; the band is plus/minus one standard "
        "deviation of the residuals IN that bucket -- the spread of the errors, not a confidence interval on the "
        "mean, so it does not narrow as the bucket grows. The mean drifting off the green zero line is bias drift; "
        f"the band widening is variance drift. Buckets hold about {int(n // max(nb, 1)):,} rows each."
    )
    return FigureSpec(suptitle="", panels=((line,),), figsize=figsize, caption=caption)


# Two-sided tabular CUSUM defaults (Page 1954 / Montgomery SPC): slack k=0.5 sigma is tuned to detect a 1-sigma
# sustained shift fastest. The textbook decision interval is h=5 sigma (ARL_0 ~ 930), but a drift chart often runs
# over many thousands of rows where ARL_0 930 produces nuisance false alarms; h=8 sigma pushes ARL_0 into the tens of
# thousands so a no-drift series stays quiet, at a small cost in detection delay on a true shift. Residuals are
# standardized first so k/h are in sigma units regardless of the residual scale.
CUSUM_SLACK_K: float = 0.5
CUSUM_DECISION_H: float = 8.0
# Target probability that a genuinely drift-free series raises a change-point anywhere along its length. h=8 was
# chosen for "ARL_0 in the tens of thousands", but Siegmund's approximation puts the TWO-SIDED ARL_0 at h=8, k=0.5
# near 9,500, so a 6000-row series false-alarms about 47% of the time -- measured: 3 of 4 pure-noise seeds crossed.
# Series length is what decides whether a fixed h is quiet, so h is solved from the length instead of fixed.
CUSUM_FALSE_ALARM_TARGET: float = 0.05


def cusum_h_for_length(n: int, k: float = CUSUM_SLACK_K, alpha: float = CUSUM_FALSE_ALARM_TARGET) -> float:
    """Decision interval h (in sigma) holding the whole-series false-alarm probability near ``alpha`` at length ``n``.

    Inverts Siegmund's ARL_0 approximation ``(exp(u) - u - 1) / (2k^2)`` with ``u = 2k(h + 1.166)``, targeting a
    two-sided ARL_0 of ``n / alpha``. Never returns less than :data:`CUSUM_DECISION_H`, so this only ever makes the
    chart QUIETER than the previous fixed default -- it cannot cause a real shift to be missed that was caught before.
    """
    if n <= 0 or k <= 0:
        return CUSUM_DECISION_H
    target_one_arm = 2.0 * n / max(alpha, 1e-6)  # two arms each get half the alarm budget
    rhs = 2.0 * k * k * target_one_arm
    u = np.log(rhs + 1.0)
    for _ in range(60):  # exp(u) - u - 1 = rhs, converging from below
        u = np.log(rhs + u + 1.0)
    return float(max(CUSUM_DECISION_H, u / (2.0 * k) - 1.166))
# Robust in-control mean/std from the FIRST in_control_frac of the time-ordered residuals (the period assumed drift-
# free); median + MAD (scaled to sigma by 1.4826) resist the very mean-shift we are trying to detect downstream.
CUSUM_IN_CONTROL_FRAC: float = 0.25
_MAD_TO_SIGMA: float = 1.4826


def _cusum_tabular_loop(z: np.ndarray, k: float, h: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Two-sided tabular CUSUM recurrence over standardized residuals ``z``.

    ``sp_t = max(0, sp_{t-1} + z_t - k)`` accumulates positive drift, ``sm_t = max(0, sm_{t-1} - z_t - k)`` negative
    drift. Returns ``(sp, sm, cross_idx)`` where ``cross_idx`` is the first index at which either arm exceeds ``h``
    (the detected change-point), or -1 if neither crosses. Inherently sequential (each step clips at 0), so this is a
    single O(n) pass -- njit-compiled when numba is present, plain-Python-loop fallback otherwise.
    """
    n = z.shape[0]
    sp = np.empty(n, dtype=np.float64)
    sm = np.empty(n, dtype=np.float64)
    prev_p = 0.0
    prev_m = 0.0
    cross = -1
    for i in range(n):
        cur_p = prev_p + z[i] - k
        if cur_p < 0.0:
            cur_p = 0.0
        cur_m = prev_m - z[i] - k
        if cur_m < 0.0:
            cur_m = 0.0
        sp[i] = cur_p
        sm[i] = cur_m
        if cross < 0 and (cur_p > h or cur_m > h):
            cross = i
        prev_p = cur_p
        prev_m = cur_m
    return sp, sm, cross


if _NUMBA_AVAILABLE:
    _cusum_tabular = njit(cache=True)(_cusum_tabular_loop)
else:
    _cusum_tabular = _cusum_tabular_loop


def cusum_residual_drift(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    *,
    slack_k: float = CUSUM_SLACK_K,
    decision_h: Optional[float] = None,
    in_control_frac: float = CUSUM_IN_CONTROL_FRAC,
    max_vertices: int = 2000,
    x_is_time: bool = True,
    title: str = "Residual drift CUSUM (sustained mean shift)",
    figsize: Tuple[float, float] = (11.0, 4.0),
) -> FigureSpec:
    """Two-sided tabular CUSUM of standardized regression residuals -- detects a SUSTAINED mean shift.

    Residual = y_true - y_pred is standardized by a robust in-control mean/std (median + MAD over the first
    ``in_control_frac`` of the time-ordered rows, the period assumed drift-free), then run through the classic
    two-sided tabular CUSUM with slack ``k`` and decision interval ``h`` (defaults h=8/k=0.5: detects a sustained
    1-sigma shift fast while keeping false alarms rare over long series). Each arm accumulates one-sided drift and
    resets to 0 in-control, so a small persistent bias
    accumulates past ``h`` and trips a change-point alarm long before any single bucket's mean looks abnormal -- the
    edge over per-bucket ``residual_vs_time``. The detected change-point (first arm crossing) is drawn as a vline and a
    shaded post-change span; ``+h`` is an hline control limit.

    O(n): one robust-stat pass + one cumulative CUSUM recurrence (njit). The crossing is computed on the FULL n; only
    the PLOTTED curve is decimated to ``max_vertices`` (the crossing marker stays at its true x). ``timestamps`` orders
    the residuals (index order when None). Edge-degenerate input (n small, all-equal residuals, all-NaN) ->
    AnnotationPanelSpec. Returns a single-panel FigureSpec.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    if timestamps is not None:
        ts = np.asarray(timestamps).ravel()
        # The prior `else mask` branch for non-numeric timestamps was a no-op (`mask &= mask`), so a
        # datetime64 NaT timestamp was never filtered and could land at an arbitrary sorted position in the
        # CUSUM's time-ordered residual sequence (np.argsort sorts NaT to the end, silently shifting every
        # subsequent row's time-order). Filter NaT explicitly for datetime64; other non-numeric dtypes fall
        # through unfiltered (unchanged from before -- no known non-finite sentinel for them).
        if np.issubdtype(ts.dtype, np.number):
            mask &= np.isfinite(ts.astype(np.float64))
        elif np.issubdtype(ts.dtype, np.datetime64):
            mask &= ~np.isnat(ts)
    yt, yp = yt[mask], yp[mask]
    n = yt.size
    if n < 8:
        ann: PanelSpec = AnnotationPanelSpec(text=f"cusum_residual_drift: need >= 8 finite rows (got {n})", title=title)
        return FigureSpec(suptitle="", panels=((ann,),), figsize=figsize)

    if timestamps is not None:
        ts = np.asarray(timestamps).ravel()[mask]
        order = np.argsort(ts, kind="stable")
        resid = (yt - yp)[order]
        x_full = ts[order].astype(np.float64)
    else:
        resid = yt - yp
        x_full = np.arange(n, dtype=np.float64)
        x_is_time = False

    n_ic = max(4, round(in_control_frac * n))
    ic = resid[:n_ic]
    center = float(np.median(ic))
    mad = float(np.median(np.abs(ic - center)))
    sigma = mad * _MAD_TO_SIGMA
    if not np.isfinite(sigma) or sigma <= 0.0:
        # MAD collapses on a near-constant in-control window; fall back to the full-series std (then a tiny floor) so a
        # shift that begins right after the in-control window is still standardized on a real scale, not divided by 0.
        sigma = float(np.std(resid))
    if not np.isfinite(sigma) or sigma <= 0.0:
        ann = AnnotationPanelSpec(text="cusum_residual_drift: residuals are constant (no variation to standardize)", title=title)
        return FigureSpec(suptitle="", panels=((ann,),), figsize=figsize)

    z = (resid - center) / sigma
    # Solve h from the series length unless the caller pinned one, so the whole-series false-alarm probability stays
    # near CUSUM_FALSE_ALARM_TARGET instead of growing with n.
    decision_h = cusum_h_for_length(n, float(slack_k)) if decision_h is None else float(decision_h)
    sp, sm, cross = _cusum_tabular(z, float(slack_k), float(decision_h))
    # Signed CUSUM for a readable single curve: positive arm minus negative arm. Both arms are >= 0 and only one is
    # active at a time once drift sets in, so the difference shows direction (up-shift positive, down-shift negative).
    stat = sp - sm

    x_plot, sp_p, sm_p, stat_p = x_full, sp, sm, stat
    if n > max_vertices:
        keep = np.unique(np.linspace(0, n - 1, max_vertices).astype(np.int64))
        if cross >= 0 and cross not in keep:
            keep = np.unique(np.concatenate([keep, np.array([cross], dtype=np.int64)]))
        x_plot, sp_p, sm_p, stat_p = x_full[keep], sp[keep], sm[keep], stat[keep]

    hi = np.full_like(x_plot, float(decision_h))
    lo = np.full_like(x_plot, -float(decision_h))
    series = (stat_p, sp_p, sm_p, hi, lo)
    labels = ("signed CUSUM (S+ - S-)", "S+ (up-shift)", "S- (down-shift)", f"+h ({decision_h:g} sigma)", "-h")
    styles = ("-", "-", "-", "--", "--")
    colors = ("steelblue", "darkorange", "purple", "red", "red")

    vlines = None
    vspans = None
    if cross >= 0:
        cx = float(x_full[cross])
        direction = "up" if sp[cross] > sm[cross] else "down"
        vlines = ((cx, "red", f"change-point @ {direction}-shift"),)
        vspans = ((cx, float(x_full[-1]), "red", 0.07, "post-change"),)
        # "ordered-row 4173" is an internal index nobody can act on; report WHEN it happened.
        subtitle = f"\nchange-point detected at {_format_x(cx)} ({direction}-shift, ordered row {cross:,} of {n:,});" f" h={decision_h:g}, k={slack_k:g} sigma"
    else:
        subtitle = f"\nno sustained shift detected (CUSUM stayed within +/-{decision_h:g} sigma); k={slack_k:g}"
    # The in-control window is an ASSUMPTION, not a measurement: if the model was already drifting there, sigma is
    # inflated and no alarm can ever fire. Stating the span is what lets a reader check that assumption.
    subtitle += f"; in-control baseline = first {in_control_frac:.0%}" f" ({_format_x(float(x_full[0]))} .. {_format_x(float(x_full[n_ic - 1]))})"

    line = LinePanelSpec(
        x=x_plot,
        y=series,
        series_labels=labels,
        title=title + subtitle,
        xlabel="time" if x_is_time else "ordered row",
        ylabel="CUSUM statistic (sigma)",
        line_styles=styles,
        colors=colors,
        x_is_time=x_is_time,
        vlines=vlines,
        vspans=vspans,
    )
    caption = (
        "Two-sided tabular CUSUM of residuals standardised by a robust median/MAD estimated from the FIRST part of "
        "the series, which is ASSUMED drift-free -- if the model was already drifting there, sigma is inflated and "
        "no alarm will ever fire. S+ accumulates sustained over-prediction, S- sustained under-prediction; either "
        "crossing the control limit signals a structural break that a per-bucket mean chart is too noisy to catch, "
        "because a small persistent bias accumulates long before any single bucket's mean looks abnormal."
    )
    return FigureSpec(suptitle="", panels=((line,),), figsize=figsize, caption=caption)


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
    """Rolling metric per time bucket as a LinePanelSpec with split / regime shading.

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
    direction = "higher=better" if higher_is_better else "lower=better"
    line = LinePanelSpec(
        x=x,
        y=yvals,
        series_labels=(metric,),
        title=title or f"{metric} over time ({direction})",
        xlabel="time",
        ylabel=metric,
        line_styles=("lines+markers",),
        colors=("steelblue",),
        x_is_time=x_is_time,
        vspans=vspans,
    )
    caption = (
        f"One point per '{freq}' time bucket, computed only on buckets holding at least {min_samples} rows -- a "
        "bucket below that is left blank rather than plotted as a low score. A step DOWN that persists across "
        "several buckets is staleness (the world moved); a single low bucket is usually just sample size. Shaded "
        "spans mark the split / regime boundaries passed in by the caller."
    )
    return FigureSpec(suptitle="", panels=((line,),), figsize=figsize, caption=caption)


def _is_datetime_index(idx: Any) -> bool:
    """True when a pandas index carries datetime64 values (so we can take .astype('datetime64[ns]')."""
    try:
        return np.issubdtype(np.asarray(idx.values).dtype, np.datetime64)
    except (TypeError, AttributeError):
        return False


def _regimes_to_vspans(regimes: Optional[Sequence[Tuple[Any, Any, str, str]]], alpha: float) -> Optional[Tuple[Tuple[Any, ...], ...]]:
    """Convert ``(start, end, color, label)`` regime spans to LinePanelSpec ``vspans``.

    start/end are coerced to the same numeric x-scale as the line (datetime -> int64 ns, else float). A non-empty
    label emits a 5-tuple ``(x0, x1, color, alpha, label)`` so the renderer adds a legend proxy per regime; an empty
    label stays the 4-tuple ``(x0, x1, color, alpha)``.
    """
    if not regimes:
        return None
    import pandas as pd

    out: List[Tuple[Any, ...]] = []
    for span in regimes:
        if len(span) < 3:
            continue
        start, end, color = span[0], span[1], span[2]
        label = str(span[3]) if len(span) >= 4 and span[3] else ""
        x0 = _coerce_x(start, pd)
        x1 = _coerce_x(end, pd)
        if label:
            out.append((x0, x1, str(color), float(alpha), label))
        else:
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


# Re-exported so every existing ``from ...drift import adversarial_auc`` keeps working. Imported at the BOTTOM
# because the sibling imports this module's frame helpers and ADV_* constants back.
from ._drift_adversarial import adversarial_auc, adversarial_validation

__all__ = [
    "PSI_MODERATE",
    "PSI_SIGNIFICANT",
    "PSI_DEFAULT_BINS",
    "CUSUM_SLACK_K",
    "CUSUM_DECISION_H",
    "CUSUM_IN_CONTROL_FRAC",
    "ADV_MAX_ROWS_PER_SIDE",
    "ADV_TOP_FEATURES",
    "ADV_N_ESTIMATORS",
    "compute_psi_matrix",
    "psi_heatmap",
    "residual_vs_time",
    "cusum_residual_drift",
    "metric_over_time",
    "adversarial_auc",
    "adversarial_validation",
]
