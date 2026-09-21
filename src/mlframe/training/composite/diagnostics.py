"""Visualisation helpers for composite-target discovery output.

Eight plot helpers + Markdown rendering integration. All use matplotlib
non-interactively (``Agg`` backend safe), return the ``Figure`` so the
caller can ``.savefig()`` or ``.show()`` as they prefer. Lazy-imports
matplotlib so this module's import cost stays zero when the helpers
aren't called.

Plots
-----

- :func:`plot_target_distribution`: histogram of ``y`` overlaid with
  the histogram of one composite ``T`` value array. Visual sanity-
  check for "did the transform actually shift / re-shape the
  distribution as expected".
- :func:`plot_qq`: Q-Q plot of ``T`` against the standard normal.
  Reveals whether ``logratio`` actually normalised the heavy-tail
  target.
- :func:`plot_linear_fit`: scatter of ``y`` vs ``base`` with the
  fitted ``alpha * base + beta`` line and the in-sample R^2.
  Justifies ``linear_residual`` to a stakeholder in one image.
- :func:`plot_mi_gain_with_jitter`: bar chart of per-spec ``mi_gain``
  with Gaussian-jitter error bars (NOT a bootstrap CI). Reveals which
  gains are robustly ranked vs sensitive to small noise.
  :func:`plot_mi_gain_with_ci` is a deprecated alias.
- :func:`plot_per_fold_tiny_rmse`: boxplot of per-fold tiny CV-RMSE
  per spec, flagging specs whose mean is best-by-a-hair but unstable.
- :func:`plot_per_family_disagreement`: heatmap of Spearman rank-
  correlation between tiny-model families' rerank rankings.
- :func:`plot_alpha_stability`: line plot of fitted ``alpha`` across
  rolling windows; drift signals base->y concept drift.
- :func:`plot_predictions_vs_actual`: side-by-side ``y_pred`` vs
  ``y_true`` scatter per composite with the y=x diagonal.
- :func:`plot_reliability_diagram`: top-label calibration curve for
  ``CompositeClassificationEstimator`` (consumes the
  ``calibration_report`` shape or raw ``y_true`` + ``proba``); annotates ECE.
- :func:`plot_interval_coverage`: empirical coverage + mean width for
  conformal / CQR / Mondrian prediction bands.
- :func:`plot_interval_width_vs_x`: width-vs-feature scatter showing
  adaptive (CQR) vs constant (split-conformal) interval width.

Each helper returns a ``matplotlib.figure.Figure``; the caller is
responsible for ``savefig`` / display / close. The Figure objects do
NOT auto-display under ``Agg``.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _lazy_pyplot():
    """Lazy-import matplotlib.pyplot. Picks the ``Agg`` backend when
    the current backend is interactive (Agg is the headless default
    in CI / scripts; we don't want a stray ``plt.show()`` blocking).

    Raises a clear ``ImportError`` when matplotlib is not installed so a
    caller on a headless box without the plotting extra gets an actionable
    message instead of a bare ``ModuleNotFoundError``."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "matplotlib is required for mlframe composite diagnostic plots; " "install it via `pip install matplotlib` (or `pip install mlframe[all]`)."
        ) from exc
    # Only force Agg when the backend is still the unresolved auto-sentinel; calling get_backend() would itself resolve (and thus pin) a
    # backend, so we must read the raw rcParam to tell "user hasn't picked one yet" from "already configured". A user who ran
    # matplotlib.use(...) keeps their choice.
    try:
        _raw = dict.__getitem__(matplotlib.rcParams, "backend")
        if isinstance(_raw, str) and _raw == matplotlib.rcsetup._auto_backend_sentinel:
            matplotlib.use("Agg")
    except (KeyError, AttributeError):
        pass
    import matplotlib.pyplot as plt
    return plt


def _moments(arr: np.ndarray, cap: int = 100_000) -> tuple[float, float, float, str]:
    """std, skew, excess kurtosis of ``arr`` (on a seeded ``cap``-row subsample for large inputs) plus a subsample note."""
    from scipy.stats import kurtosis, skew

    note = ""
    if arr.size > cap:
        arr = arr[np.random.default_rng(0).choice(arr.size, size=cap, replace=False)]
        note = f" (moments on {cap:,} rows)"
    return float(np.std(arr)), float(skew(arr)), float(kurtosis(arr)), note


def _top_value_share(arr: np.ndarray) -> tuple[float, float]:
    """Most frequent value and its share of rows (a spike such as 70% zeros dominates every other feature of the shape)."""
    vals, counts = np.unique(arr, return_counts=True)
    i = int(np.argmax(counts))
    return float(vals[i]), float(counts[i] / arr.size)


def _asinh_panel(ax, arr: np.ndarray, color: str, name: str, bins: int) -> None:
    """Histogram of ``arr`` on an asinh-scaled x axis over its [0.1%, 99.9%] range, log density, tails counted in the label.

    A linear axis over a heavy-tailed target puts almost every row into the first bar (a production chart spanned 0..90 000
    with one visible bar). asinh is linear near zero and logarithmic in both tails, so zeros, negatives and the tail all
    stay readable; the tick labels are in original units.
    """
    lo, hi = np.quantile(arr, [0.001, 0.999])
    if lo == hi:
        lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    scale = max(float(np.quantile(np.abs(arr - np.median(arr)), 0.5)), 1e-12)
    fwd = lambda v: np.arcsinh(np.asarray(v, dtype=np.float64) / scale)  # noqa: E731
    inv = lambda u: np.sinh(np.asarray(u, dtype=np.float64)) * scale  # noqa: E731
    edges = inv(np.linspace(fwd(lo), fwd(hi), bins + 1))
    inside = (arr >= lo) & (arr <= hi)
    counts, _ = np.histogram(arr[inside], bins=edges)
    ax.bar(edges[:-1], counts / arr.size, width=np.diff(edges), align="edge", color=color, alpha=0.8, edgecolor="none")
    ax.set_xscale("function", functions=(fwd, inv))
    # Decade ticks on both signs, thinned so neighbours are at least ~0.8 asinh units apart (no overlapping labels).
    k0 = int(np.floor(np.log10(scale)))
    mags = [0.0] + [10.0**k for k in range(k0, k0 + 16)]
    cand = sorted({v for m in mags for v in (m, -m) if lo <= v <= hi})
    ticks: list[float] = []
    for v in cand:
        if not ticks or fwd(v) - fwd(ticks[-1]) >= 0.8:
            ticks.append(v)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8)
    ax.set_yscale("log")
    n_lo, n_hi = int((arr < lo).sum()), int((arr > hi).sum())
    ax.set_xlabel(f"{name} value (asinh scale; {n_lo:,} rows below / {n_hi:,} above the shown 0.1-99.9% range)", fontsize=8)
    ax.set_ylabel("share of rows per bin (log)")


def plot_target_distribution(
    y: np.ndarray,
    t: np.ndarray,
    *,
    title: str = "Target distribution: y vs T",
    bins: int = 60,
    figsize: tuple[float, float] = (13, 6.5),
    y_name: str | None = None,
    transform_name: str | None = None,
    base_column: str | None = None,
):
    """Side-by-side distributions of ``y`` and ``T = transform(y, base)`` with a plain-language verdict.

    The question the chart answers: did the transform make the target easier to model -- a less skewed, lighter-tailed,
    smaller-spread T -- or did it barely change the shape? Each panel has its own asinh-scaled x axis (y and T generally
    live on different scales) and a log density axis, so both the spike and the tail are visible. The header states
    std / skew / excess kurtosis before and after, the change, and the most frequent value's share.

    A footer explains what y and T are, since a reader seeing this chart without the discovery docs has no way to know:
    y is the original target, T the substitute target the models train on, built from y and a base column and converted
    back to y after prediction. ``y_name`` / ``transform_name`` / ``base_column`` make that footer concrete (the
    transform's registry description is quoted when available).
    """
    plt = _lazy_pyplot()
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    fy = y_arr[np.isfinite(y_arr)]
    ft = t_arr[np.isfinite(t_arr)]
    fig, (ax_y, ax_t) = plt.subplots(1, 2, figsize=figsize)
    for ax, arr, color, name in ((ax_y, fy, "tab:blue", "y"), (ax_t, ft, "tab:orange", "T")):
        if arr.size == 0:
            ax.text(0.5, 0.5, f"no finite {name}", ha="center", va="center", transform=ax.transAxes)
            continue
        _asinh_panel(ax, arr, color, name, bins)
        top_v, top_share = _top_value_share(arr)
        spike = f"; {top_share:.0%} of rows equal {top_v:.4g}" if top_share >= 0.05 else ""
        ax.set_title(f"{name}  (n={arr.size:,}{spike})", fontsize=9)

    verdict = ""
    if fy.size > 1 and ft.size > 1:
        try:
            sy, ky_skew, ky_kurt, note = _moments(fy)
            st, kt_skew, kt_kurt, _ = _moments(ft)

            def _chg(a: float, b: float) -> str:
                """Relative change from ``a`` to ``b`` as a signed percentage, or ``n/a`` when ``a`` is zero."""
                return f"{(b - a) / abs(a):+.0%}" if a else "n/a"

            tail_better = abs(kt_skew) < 0.5 * abs(ky_skew) and kt_kurt < 0.5 * ky_kurt
            tail_worse = abs(kt_skew) > 1.25 * abs(ky_skew) or kt_kurt > 1.25 * ky_kurt
            verdict = (
                f"std {sy:.3g} -> {st:.3g} ({_chg(sy, st)}),  skew {ky_skew:.2f} -> {kt_skew:.2f},  "
                f"excess kurtosis {ky_kurt:.1f} -> {kt_kurt:.1f}{note}\n"
                + (
                    "Verdict: T is much less skewed / heavy-tailed than y -- the transform makes the target easier for the models."
                    if tail_better
                    else "Verdict: T is MORE skewed or heavy-tailed than y -- the transform makes the shape harder, not easier."
                    if tail_worse
                    else "Verdict: T keeps roughly the shape of y -- any gain must come from the base explaining part of y, "
                    "not from a nicer distribution (judge it by the holdout RMSE, not this chart)."
                )
            )
        except Exception as e:  # nosec B110 - annotation only
            logger.debug("target-distribution verdict failed: %s", e)
    fig.text(0.01, 0.01, _explain_y_and_t(y_name, transform_name, base_column), ha="left", va="bottom", fontsize=8, wrap=True,
             bbox={"facecolor": "#f4f4f4", "edgecolor": "#bbbbbb"})
    fig.suptitle(f"{title}\n{verdict}" if verdict else title, fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


def _explain_y_and_t(y_name: str | None, transform_name: str | None, base_column: str | None) -> str:
    """Plain-language footer: what y is, what T is, and how the models use T."""
    y_label = f"'{y_name}'" if y_name else "the target"
    how = ""
    if transform_name:
        try:
            from .transforms import get_transform

            desc = str(getattr(get_transform(transform_name), "description", "") or "").strip()
            how = f" Here T = {transform_name}(y{', ' + base_column if base_column else ''}): {desc.split('. ')[0].rstrip('.')}."
        except Exception as e:  # nosec B110 - an unknown name just drops the formula line
            logger.debug("transform description lookup failed for %r: %s", transform_name, e)
            how = f" Here T = {transform_name}(y{', ' + base_column if base_column else ''})."
    return (
        f"y = the original target {y_label}, as it is in the data.   "
        f"T = a substitute target the models are trained on INSTEAD of y, computed from y"
        f"{' and the base column ' + repr(base_column) if base_column else ''}; each model's prediction of T is converted "
        f"back to y, so the final forecast is still in y units.{how}\n"
        "Why look: a model fits a compact, symmetric target more easily than a spiky, heavy-tailed one. "
        "If T's panel is narrower and less lopsided than y's, the substitution helps."
    )


def _qq_decimation_indices(n: int, max_points: int = 2000, tail_keep: int = 20) -> np.ndarray:
    """Order-statistic ranks to plot on a QQ scatter: uniform stride over the bulk, plus the first/last ``tail_keep`` ranks kept
    exactly -- tail behaviour is the whole point of a QQ plot, so the extreme order statistics must never be decimated away."""
    if n <= max_points:
        return np.arange(n)
    mid = np.linspace(tail_keep, n - 1 - tail_keep, max_points - 2 * tail_keep).round().astype(np.int64)
    return np.unique(np.concatenate([np.arange(tail_keep), mid, np.arange(n - tail_keep, n)]))


def plot_qq(
    t: np.ndarray,
    *,
    title: str = "Q-Q plot of T vs standard normal",
    figsize: tuple[float, float] = (6, 6),
):
    """Q-Q plot of ``T`` quantiles against the standard normal.

    A perfectly normal ``T`` lands on the y=x line. Heavy-tail T
    bows away from the line at the extremes; light-tail (uniform-ish)
    bows toward the line. Useful for diagnosing whether ``logratio``
    actually normalised the target.

    The scatter is decimated to ~2000 order statistics on large inputs
    (a screen has fewer horizontal pixels) while the extreme ranks are
    always kept; the fit line uses the full order statistics.
    """
    plt = _lazy_pyplot()
    t_arr = np.asarray(t).reshape(-1)
    finite = t_arr[np.isfinite(t_arr)]
    if finite.size < 5:
        # Not enough data; render an empty placeholder so the caller
        # still has a Figure object to handle uniformly.
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"too few finite T values (n={finite.size})", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    from scipy.stats import linregress, norm

    n = finite.size
    osr = np.sort(finite)
    # Filliben order-statistic medians: the exact positions scipy.stats.probplot uses, so the undecimated output is identical to the
    # probplot(plot=ax) call this replaces -- probplot itself was dropped because it scatter-plots all n points (5s+ savefig at 2M).
    pos = (np.arange(1.0, n + 1.0) - 0.3175) / (n + 0.365)
    pos[0] = 1.0 - 0.5 ** (1.0 / n)
    pos[-1] = 0.5 ** (1.0 / n)
    osm = norm.ppf(pos)
    fit = linregress(osm, osr)

    fig, ax = plt.subplots(figsize=figsize)
    idx = _qq_decimation_indices(n)
    ax.plot(osm[idx], osr[idx], "bo")
    ax.plot([osm[0], osm[-1]], [fit.slope * osm[0] + fit.intercept, fit.slope * osm[-1] + fit.intercept], "r-")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered Values")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_linear_fit(
    y: np.ndarray,
    base: np.ndarray,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    title: str = "Linear residual fit",
    figsize: tuple[float, float] = (8, 5),
):
    """Scatter of ``y`` against ``base`` with the fitted
    ``alpha * base + beta`` line overlaid. R^2 annotated in the
    upper-left corner.

    If ``alpha`` / ``beta`` are not supplied, computes them via OLS
    so the helper works for ad-hoc inspection without a
    ``CompositeSpec`` already in hand. When supplied (e.g. from a
    spec's ``fitted_params``), the helper plots THAT line so the
    visual matches what the discovery actually saved.
    """
    plt = _lazy_pyplot()
    y_arr = np.asarray(y).reshape(-1).astype(np.float64)
    base_arr = np.asarray(base).reshape(-1).astype(np.float64)
    finite = np.isfinite(y_arr) & np.isfinite(base_arr)
    y_f = y_arr[finite]
    b_f = base_arr[finite]

    if alpha is None or beta is None:
        if b_f.size < 2 or np.std(b_f) < 1e-12:
            alpha_use, beta_use = 0.0, float(np.mean(y_f)) if y_f.size else 0.0
        else:
            X = np.column_stack([b_f, np.ones(len(b_f))])
            coef, *_ = np.linalg.lstsq(X, y_f, rcond=None)
            alpha_use, beta_use = float(coef[0]), float(coef[1])
    else:
        alpha_use, beta_use = float(alpha), float(beta)

    # R^2 of the line vs y.
    y_pred = alpha_use * b_f + beta_use
    ss_res = float(np.sum((y_f - y_pred) ** 2))
    ss_tot = float(np.sum((y_f - np.mean(y_f)) ** 2)) if y_f.size > 0 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    fig, ax = plt.subplots(figsize=figsize)
    # Subsample to ~5000 points so the scatter doesn't choke on big n.
    if b_f.size > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(b_f.size, size=5000, replace=False)
        ax.scatter(b_f[idx], y_f[idx], alpha=0.3, s=8, color="tab:blue")
    else:
        ax.scatter(b_f, y_f, alpha=0.4, s=10, color="tab:blue")
    # Draw the line over the observed base range.
    if b_f.size > 0:
        x_line = np.linspace(b_f.min(), b_f.max(), 100)
        ax.plot(x_line, alpha_use * x_line + beta_use, color="tab:red", lw=2, label=f"y = {alpha_use:.3g} * base + {beta_use:.3g}")
        ax.legend(loc="lower right")
    ax.set_xlabel("base")
    ax.set_ylabel("y")
    ax.set_title(title)
    text = f"R^2 = {r2:.4f}\nn = {finite.sum()}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10, family="monospace", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"})
    fig.tight_layout()
    return fig


def plot_mi_gain_with_jitter(
    specs: Sequence[dict[str, Any]],
    *,
    n_jitter: int = 200,
    title: str = "MI gain per composite spec (jitter error bars)",
    figsize: tuple[float, float] = (10, 5),
    random_state: int = 42,
    jitter_scale: float = 0.05,
):
    """Bar chart of per-spec ``mi_gain`` with Gaussian-jitter error bars.

    The error bars are NOT a bootstrap CI on the MI estimate. A true bootstrap would
    require resampling the screening sample and recomputing MI per replicate, which
    is infeasible from the spec-dict input alone. Instead each replicate adds
    independent Gaussian noise scaled to ``jitter_scale * |mi_gain|`` (default 5%);
    the resulting 2.5/97.5 percentile band reflects this synthetic noise only and
    should be read as a visual cue for "small gains are not robustly ranked", NOT as
    a statistical uncertainty.

    Parameters
    ----------
    specs
        List of dicts in the format ``CompositeTargetDiscovery.export_specs()`` returns.
    n_jitter
        Number of jitter replicates (controls error-bar smoothness, not statistical power).
    title
        Chart title.
    figsize
        Figure size in inches.
    random_state
        Seed of the jitter noise.
    jitter_scale
        Multiplier on ``|mi_gain|`` for the per-replicate Gaussian noise.

    Returns
    -------
    matplotlib.figure.Figure
        The bar chart (a placeholder figure when ``specs`` is empty).
    """
    plt = _lazy_pyplot()
    if not specs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no specs to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    names = [s.get("name", "?") for s in specs]
    gains = np.array([float(s.get("mi_gain", float("nan"))) for s in specs])

    # Per-spec independent Gaussian jitter at ``jitter_scale * |gain|``. This is a
    # visual heuristic for "small gains are not robustly ranked", NOT a CI on the MI
    # estimate (which would require resampling the screening sample, infeasible from
    # the spec-dict input).
    rng = np.random.default_rng(random_state)
    n = gains.size
    jittered = np.empty((n_jitter, n))
    for b in range(n_jitter):
        jittered[b] = gains + rng.normal(scale=jitter_scale * np.abs(gains), size=n)
    lo = np.nanpercentile(jittered, 2.5, axis=0)
    hi = np.nanpercentile(jittered, 97.5, axis=0)
    err_lo = gains - lo
    err_hi = hi - gains

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n)
    bar_colors = ["tab:green" if g > 0 else "tab:red" for g in gains]
    ax.bar(x, gains, yerr=[err_lo, err_hi], color=bar_colors, alpha=0.7, capsize=4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mi_gain (MI with T - MI with y)")
    ax.set_title(title)
    fig.text(
        0.01, 0.01,
        "Each bar is one composite target that survived discovery (name = target-transform-base column). mi_gain is how "
        "much MORE mutual information the features carry about the substitute target T than about the original y: "
        "positive means T is easier to predict from the features than y is. The absolute values are small by nature; "
        "compare bars with each other, not with 1. Error bars are a visual noise cue (5% jitter), not a confidence "
        "interval. The final keep/drop decision is made on holdout RMSE in y units, not on this number.",
        ha="left", va="bottom", fontsize=8, wrap=True,
        bbox={"facecolor": "#f4f4f4", "edgecolor": "#bbbbbb"},
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return fig


def plot_mi_gain_with_ci(
    specs: Sequence[dict[str, Any]],
    *,
    n_bootstrap: int = 200,
    title: str = "MI gain per composite spec (jitter error bars)",
    figsize: tuple[float, float] = (10, 5),
    random_state: int = 42,
):
    """Deprecated alias. The error bars are Gaussian jitter, not a bootstrap CI;
    use :func:`plot_mi_gain_with_jitter` directly. Kept for back-compat with
    pre-rename callers.
    """
    import warnings as _w
    _w.warn(
        "plot_mi_gain_with_ci is a deprecated alias for plot_mi_gain_with_jitter; "
        "the error bars are Gaussian jitter (not a bootstrap CI). Call "
        "plot_mi_gain_with_jitter(specs, n_jitter=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_mi_gain_with_jitter(
        specs,
        n_jitter=n_bootstrap,
        title=title,
        figsize=figsize,
        random_state=random_state,
    )


def plot_per_fold_tiny_rmse(
    per_fold_rmses: dict[str, Sequence[float]],
    *,
    raw_baseline: float | None = None,
    title: str = "Per-fold tiny CV-RMSE per composite spec",
    figsize: tuple[float, float] = (10, 5),
):
    """Boxplot of per-fold tiny CV-RMSE (y-scale, after inverse) per
    composite spec. Shows the cross-fold variance directly so a spec
    whose mean RMSE is best-by-a-hair but has wide fold spread can be
    flagged as unstable.

    ``per_fold_rmses`` is ``{spec_name: [rmse_fold_0, rmse_fold_1, ...]}``.
    Pass ``raw_baseline`` to overlay the raw-y baseline RMSE as a
    dashed horizontal line; specs whose box sits entirely above this
    line are gate-rejected.
    """
    plt = _lazy_pyplot()
    if not per_fold_rmses:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no per-fold RMSE data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig
    names = list(per_fold_rmses.keys())
    data = [[float(v) for v in per_fold_rmses[n] if np.isfinite(v)] for n in names]
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.6)
    if raw_baseline is not None and np.isfinite(raw_baseline):
        ax.axhline(raw_baseline, color="black", linestyle="--", linewidth=1.0, label=f"raw-y baseline = {raw_baseline:.4f}")
        ax.legend(loc="best", fontsize=9)
    ax.set_ylabel("CV-RMSE on y-scale (after inverse)")
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_family_disagreement(
    per_family_scores: dict[str, Sequence[float]],
    spec_names: Sequence[str],
    *,
    title: str = "Per-family rerank rank-correlation",
    figsize: tuple[float, float] = (6, 5),
):
    """Heatmap of Spearman rank-correlation between tiny-model
    families' rerank rankings.

    ``per_family_scores`` is ``{family: [score_for_spec_0, score_for_spec_1, ...]}``
    aligned to ``spec_names``. When per-family screening is enabled,
    different families (LightGBM / XGBoost / CatBoost / linear) can
    disagree on which composite is best -- this heatmap surfaces how
    aligned they are. High off-diagonal correlation = consensus is
    safe; low = "union" or "borda" aggregation matters.

    ``spec_names`` is not used for axis labels (the heatmap is family x family,
    not spec x spec) but its declared alignment to ``per_family_scores`` IS
    enforced: a length mismatch means the caller built the two arguments from
    different spec sets, which would silently corrupt every correlation in the
    matrix, so it is checked eagerly here instead.
    """
    plt = _lazy_pyplot()
    from scipy.stats import spearmanr
    families = list(per_family_scores.keys())
    if len(families) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "need >= 2 families for disagreement plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig
    # Lower CV-RMSE is better -> rank ascending.
    score_matrix = np.array([list(per_family_scores[f]) for f in families], dtype=np.float64)
    n_specs = score_matrix.shape[1]
    if len(spec_names) != n_specs:
        raise ValueError(f"spec_names has {len(spec_names)} entries but per_family_scores has {n_specs} scores per family; they must be aligned 1:1.")
    if n_specs < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "need >= 2 specs for disagreement plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig
    n_fam = len(families)
    corr = np.zeros((n_fam, n_fam))
    for i in range(n_fam):
        for j in range(n_fam):
            if i == j:
                corr[i, j] = 1.0
            else:
                # spearmanr uses average ranks for ties (redundant specs with equal RMSE), unlike argsort-of-argsort which assigns ties
                # arbitrary distinct ranks; it returns nan when either input has zero variance, preserving the prior NaN guard.
                r = spearmanr(score_matrix[i], score_matrix[j]).statistic
                corr[i, j] = float("nan") if not np.isfinite(r) else float(r)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_fam))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_yticks(range(n_fam))
    ax.set_yticklabels(families)
    # Auto-flip text colour by background luminance: dark text on
    # light cells (corr near 0) was unreadable on white-on-yellow
    # cells (memory note feedback_chart_color_consistency).
    for i in range(n_fam):
        for j in range(n_fam):
            v = corr[i, j]
            text_color = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "NA", ha="center", va="center", color=text_color, fontsize=10)
    fig.colorbar(im, ax=ax, label="Spearman rank corr")
    ax.set_title(title + f" ({n_specs} specs)")
    fig.tight_layout()
    return fig


def plot_alpha_stability(
    alpha_per_window: Sequence[float],
    *,
    window_indices: Sequence[Any] | None = None,
    title: str = "linear_residual alpha stability over windows",
    figsize: tuple[float, float] = (10, 4),
    expected_alpha: float | None = None,
):
    """Line plot of fitted ``alpha`` (linear_residual coefficient)
    across rolling windows of the train data.

    A drift in ``alpha`` over time signals concept drift in the
    base->y relationship. ``alpha_per_window`` is the sequence of
    fitted alphas; ``window_indices`` gives x-axis labels (e.g.
    timestamps or window IDs). ``expected_alpha`` overlays a
    horizontal reference line (e.g. the alpha fitted on the full
    train).
    """
    plt = _lazy_pyplot()
    alpha_arr = np.asarray(alpha_per_window, dtype=np.float64)
    if alpha_arr.size == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no alpha samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig
    if window_indices is not None and len(window_indices) != alpha_arr.size:
        raise ValueError(f"plot_alpha_stability: window_indices ({len(window_indices)}) and " f"alpha_per_window ({alpha_arr.size}) must have equal length.")
    if window_indices is None:
        x = np.arange(alpha_arr.size)
    else:
        x = np.arange(len(window_indices))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, alpha_arr, marker="o", linewidth=1.5, color="tab:blue", label="alpha per window")
    if expected_alpha is not None and np.isfinite(expected_alpha):
        ax.axhline(expected_alpha, color="black", linestyle="--", linewidth=1.0, label=f"reference alpha = {expected_alpha:.4f}")
    if window_indices is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(w) for w in window_indices], rotation=30, ha="right",
            fontsize=8,
        )
    finite = alpha_arr[np.isfinite(alpha_arr)]
    if finite.size:
        ax.text(0.02, 0.98,
                f"mean={finite.mean():.4f}\n"
                f"std ={finite.std():.4f}\n"
                f"range=[{finite.min():.4f}, {finite.max():.4f}]",
                transform=ax.transAxes, va="top", fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.7,
                      "edgecolor": "gray"})
    ax.set_xlabel("window")
    ax.set_ylabel("fitted alpha")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred_per_spec: dict[str, np.ndarray],
    *,
    title: str = "Predictions vs actual per composite",
    figsize: tuple[float, float] = (12, 4),
    sample_n: int = 5000,
    random_state: int = 42,
):
    """Side-by-side scatter ``y_pred`` vs ``y_true`` per composite.

    ``y_pred_per_spec`` is ``{spec_name: y_hat_array}`` where each
    ``y_hat_array`` is in the y-scale (post-inverse). On large
    datasets we subsample to ``sample_n`` rows for plot legibility.
    The diagonal y=x is overlaid for reference; tight clusters along
    it = good predictions.
    """
    plt = _lazy_pyplot()
    n_specs = len(y_pred_per_spec)
    if n_specs == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no predictions to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig
    rng = np.random.default_rng(random_state)
    if y_true.size > sample_n:
        sample_idx = rng.choice(y_true.size, size=sample_n, replace=False)
    else:
        sample_idx = np.arange(y_true.size)
    y_sample = y_true[sample_idx]
    fig, axes = plt.subplots(
        1, n_specs, figsize=(figsize[0], figsize[1]), sharey=True,
    )
    if n_specs == 1:
        axes = [axes]
    lo = float(np.nanmin(y_sample))
    hi = float(np.nanmax(y_sample))
    for ax, (name, y_hat) in zip(axes, y_pred_per_spec.items()):
        y_hat_arr = np.asarray(y_hat)
        if y_hat_arr.size != y_true.size:
            ax.text(0.5, 0.5, f"size mismatch:\n{y_hat_arr.size} vs {y_true.size}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue
        y_hat_sample = y_hat_arr[sample_idx]
        finite = np.isfinite(y_sample) & np.isfinite(y_hat_sample)
        ax.scatter(y_sample[finite], y_hat_sample[finite], s=4, alpha=0.4, color="tab:blue")
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0, label="y = ŷ")
        # RMSE annotation for the rendered sample.
        diff = y_hat_sample[finite] - y_sample[finite]
        rmse = float(np.sqrt(np.mean(diff * diff))) if finite.any() else float("nan")
        ax.text(0.02, 0.98,
                f"RMSE = {rmse:.4f}\nn = {int(finite.sum())}",
                transform=ax.transAxes, va="top",
                fontsize=9, family="monospace",
                bbox={"facecolor": "white", "alpha": 0.7,
                      "edgecolor": "gray"})
        ax.set_xlabel("y_true")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y_pred (post-inverse)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _subsample_idx(n: int, cap: int, random_state: int) -> np.ndarray:
    """Random row indices for subsampling a large input down to ``cap`` rows.
    Returns ``arange(n)`` (no copy of a fresh array beyond the range itself)
    when ``n <= cap`` so small inputs are plotted in full and in order."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=cap, replace=False)


def plot_reliability_diagram(
    y_true: np.ndarray | None = None,
    proba: np.ndarray | None = None,
    *,
    report: dict[str, Any] | None = None,
    n_bins: int = 10,
    title: str = "Reliability diagram (top-label calibration)",
    figsize: tuple[float, float] = (6, 6),
    sample_n: int = 200_000,
    random_state: int = 42,
):
    """Reliability diagram for :class:`CompositeClassificationEstimator` calibration.

    Plots per-bin observed accuracy against mean predicted confidence (the
    standard reliability curve, valid binary and multiclass) plus the y=x
    perfect-calibration diagonal, and annotates the Expected Calibration
    Error (ECE). Bars are sized to bin counts so empty bins are visible.

    Two input shapes are accepted:

    - ``report`` -- the dict returned by
      ``CompositeClassificationEstimator.calibration_report`` (keys
      ``bin_confidence`` / ``bin_accuracy`` / ``bin_count`` / ``ece``).
      Already-binned, so nothing is recomputed.
    - ``y_true`` + ``proba`` -- raw labels and the ``predict_proba`` matrix;
      binned here with the SAME equal-width top-label scheme
      ``calibration_report`` uses. Large inputs are subsampled to
      ``sample_n`` rows before binning (disclosed in the annotation).

    Pass exactly one of (``report``) or (``y_true`` and ``proba``).
    """
    plt = _lazy_pyplot()
    if report is None:
        if y_true is None or proba is None:
            raise ValueError("plot_reliability_diagram: pass either report=... or both " "y_true=... and proba=...")
        report = _bin_top_label_calibration(y_true, proba, n_bins=n_bins, sample_n=sample_n, random_state=random_state)
    bin_conf = np.asarray(report["bin_confidence"], dtype=np.float64)
    bin_acc = np.asarray(report["bin_accuracy"], dtype=np.float64)
    bin_cnt = np.asarray(report.get("bin_count", np.zeros(bin_conf.size)), dtype=np.float64)
    ece = float(report.get("ece", float("nan")))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.0, label="perfect calibration")
    nb = bin_conf.size
    centers = (np.arange(nb) + 0.5) / nb
    valid = np.isfinite(bin_conf) & np.isfinite(bin_acc)
    # Bar widths track bin occupancy so under-populated bins read as thin / absent;
    # the line+markers trace the reliability curve over the populated bins only.
    if bin_cnt.sum() > 0:
        widths = 0.9 / nb * (bin_cnt / bin_cnt.max())
    else:
        widths = np.full(nb, 0.9 / nb)
    ax.bar(centers[valid], bin_acc[valid], width=widths[valid], alpha=0.35, color="tab:blue", label="observed accuracy")
    ax.plot(bin_conf[valid], bin_acc[valid], marker="o", color="tab:red", linewidth=1.5, label="reliability curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.98, 0.02, f"ECE = {ece:.4f}\nbins = {nb}",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=10, family="monospace",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"})
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _bin_top_label_calibration(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_bins: int,
    sample_n: int,
    random_state: int,
) -> dict[str, Any]:
    """Equal-width top-label reliability binning, delegating the shared binning core to
    ``_calibration_binning.top_label_calibration_bins`` (the SAME function
    ``CompositeClassificationEstimator.calibration_report`` calls) so the standalone plotter
    produces the identical curve by construction, not by two independently-maintained
    implementations happening to agree. Subsamples to ``sample_n`` rows."""
    from ._calibration_binning import top_label_calibration_bins

    proba_arr = np.asarray(proba, dtype=np.float64)
    if proba_arr.ndim != 2:
        raise ValueError("plot_reliability_diagram: proba must be a 2D (n, n_classes) array.")
    y_arr = np.asarray(y_true).reshape(-1)
    n = y_arr.shape[0]
    idx = _subsample_idx(n, sample_n, random_state)
    proba_arr = proba_arr[idx]
    y_arr = y_arr[idx]
    # Map argmax column back to a label via the sorted unique labels -- the same class ordering
    # sklearn's ``classes_`` (and the estimator) uses. This caller has no fitted estimator to read
    # classes_ from directly (it only receives raw proba/y_true arrays), so it derives a proxy;
    # falls back to raw column indices when the observed label count doesn't match proba's width
    # (e.g. not every class is present in this particular slice).
    classes = np.unique(y_arr)
    if classes.size != proba_arr.shape[1]:
        classes = np.arange(proba_arr.shape[1])
    return top_label_calibration_bins(y_arr, proba_arr, classes, n_bins=n_bins)


def plot_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    title: str = "Prediction-interval coverage",
    figsize: tuple[float, float] = (8, 5),
    sample_n: int = 5000,
    random_state: int = 42,
):
    """Empirical coverage + mean width for conformal / CQR / Mondrian bands.

    Sorts the rendered sample by ``y_true`` and draws each interval as a
    vertical span (green when it covers ``y_true``, red when it misses),
    overlaying the true value. Annotates the EMPIRICAL coverage (fraction of
    rows with ``lower <= y_true <= upper``) and the mean band width, both
    computed on the FULL input (not just the rendered subsample) so the
    numbers are honest at any scale.
    """
    plt = _lazy_pyplot()
    y_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    lo_arr = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi_arr = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not (y_arr.shape == lo_arr.shape == hi_arr.shape):
        raise ValueError(f"plot_interval_coverage: y_true {y_arr.shape}, lower {lo_arr.shape}, " f"upper {hi_arr.shape} must share the same shape.")
    finite = np.isfinite(y_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
    y_f, lo_f, hi_f = y_arr[finite], lo_arr[finite], hi_arr[finite]
    # Honest stats on the full finite input, before any plot subsampling.
    covered_mask = (y_f >= lo_f) & (y_f <= hi_f)
    coverage = float(covered_mask.mean()) if y_f.size else float("nan")
    mean_width = float(np.mean(hi_f - lo_f)) if y_f.size else float("nan")

    fig, ax = plt.subplots(figsize=figsize)
    if y_f.size == 0:
        ax.text(0.5, 0.5, "no finite interval rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig
    sub = _subsample_idx(y_f.size, sample_n, random_state)
    order = sub[np.argsort(y_f[sub])]
    x = np.arange(order.size)
    cov_sub = covered_mask[order]
    # Two vlines calls (covered / missed) instead of per-row plotting keep the
    # render O(2) matplotlib artists even for the full 5k subsample.
    if cov_sub.any():
        ax.vlines(x[cov_sub], lo_f[order][cov_sub], hi_f[order][cov_sub], color="tab:green", alpha=0.5, linewidth=1.0, label="covered")
    if (~cov_sub).any():
        ax.vlines(x[~cov_sub], lo_f[order][~cov_sub], hi_f[order][~cov_sub], color="tab:red", alpha=0.7, linewidth=1.0, label="missed")
    ax.plot(x, y_f[order], color="black", linewidth=0.8, label="y_true (sorted)")
    ax.set_xlabel("sample (sorted by y_true)")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.98, 0.02,
            f"empirical coverage = {coverage:.4f}\n"
            f"mean width = {mean_width:.4f}\n"
            f"n = {y_f.size}",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=10, family="monospace",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"})
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_interval_width_vs_x(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    title: str = "Interval width vs x (adaptive CQR vs constant split)",
    figsize: tuple[float, float] = (8, 5),
    sample_n: int = 5000,
    random_state: int = 42,
):
    """Scatter of band width ``upper - lower`` against a 1D feature ``x``.

    Visualises whether the interval width ADAPTS to the input: a CQR / Mondrian
    band widens in high-variance regions (sloped / heteroscedastic cloud) while
    a split-conformal band is constant (flat horizontal line). Annotates the
    width coefficient-of-variation -- near zero means constant width, larger
    means adaptive. Large inputs subsample to ``sample_n`` for the scatter; the
    CV is computed on the full finite input.
    """
    plt = _lazy_pyplot()
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    lo_arr = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi_arr = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not (x_arr.shape == lo_arr.shape == hi_arr.shape):
        raise ValueError(f"plot_interval_width_vs_x: x {x_arr.shape}, lower {lo_arr.shape}, " f"upper {hi_arr.shape} must share the same shape.")
    width = hi_arr - lo_arr
    finite = np.isfinite(x_arr) & np.isfinite(width)
    x_f, w_f = x_arr[finite], width[finite]
    mean_w = float(np.mean(w_f)) if w_f.size else float("nan")
    cv = float(np.std(w_f) / mean_w) if w_f.size and abs(mean_w) > 1e-12 else float("nan")

    fig, ax = plt.subplots(figsize=figsize)
    if w_f.size == 0:
        ax.text(0.5, 0.5, "no finite width rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig
    sub = _subsample_idx(w_f.size, sample_n, random_state)
    ax.scatter(x_f[sub], w_f[sub], s=8, alpha=0.4, color="tab:blue", label="width")
    ax.axhline(mean_w, color="tab:red", linestyle="--", linewidth=1.2, label=f"mean width = {mean_w:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("interval width (upper - lower)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    kind = "constant (split-like)" if (np.isfinite(cv) and cv < 0.05) else "adaptive (CQR-like)"
    ax.text(0.02, 0.98,
            f"width CV = {cv:.4f}\n{kind}\nn = {w_f.size}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"})
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_target_distribution",
    "plot_qq",
    "plot_linear_fit",
    "plot_mi_gain_with_jitter",
    "plot_mi_gain_with_ci",
    "plot_per_fold_tiny_rmse",
    "plot_per_family_disagreement",
    "plot_alpha_stability",
    "plot_predictions_vs_actual",
    "plot_reliability_diagram",
    "plot_interval_coverage",
    "plot_interval_width_vs_x",
]
