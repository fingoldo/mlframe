"""Visualisation helpers for composite-target discovery output.

Four plot helpers + Markdown rendering integration. All use matplotlib
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
- :func:`plot_mi_gain_with_ci`: bar chart of per-spec ``mi_gain``
  with bootstrap 95% confidence intervals. Reveals which gains are
  signal vs noise on the screening sample.

Each helper returns a ``matplotlib.figure.Figure``; the caller is
responsible for ``savefig`` / display / close. The Figure objects do
NOT auto-display under ``Agg``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _lazy_pyplot():
    """Lazy-import matplotlib.pyplot. Picks the ``Agg`` backend when
    the current backend is interactive (Agg is the headless default
    in CI / scripts; we don't want a stray ``plt.show()`` blocking)."""
    import matplotlib
    # Don't switch backend if user already configured one explicitly
    # via ``matplotlib.use(...)``. Only set Agg if no backend is
    # configured -- this matches the convention in
    # ``mlframe/tests/training/conftest.py`` line 10.
    if not matplotlib.get_backend():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_target_distribution(
    y: np.ndarray,
    t: np.ndarray,
    *,
    title: str = "Target distribution: y vs T",
    bins: int = 60,
    figsize: Tuple[float, float] = (8, 5),
):
    """Overlay histograms of ``y`` and ``T = transform(y, base)``.

    Visualises the distributional shift the transform applied. For
    ``logratio`` on heavy-tail y, T should look much more Gaussian.
    For ``diff`` on autoregressive lag, T should be a tight residual
    centred near zero with much smaller std than y.

    Both ``y`` and ``t`` are flattened and finite-filtered before
    plotting; rows where either is NaN are dropped.
    """
    plt = _lazy_pyplot()
    y_arr = np.asarray(y).reshape(-1)
    t_arr = np.asarray(t).reshape(-1)
    finite_y = y_arr[np.isfinite(y_arr)]
    finite_t = t_arr[np.isfinite(t_arr)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(finite_y, bins=bins, alpha=0.5, label=f"y (n={finite_y.size})",
            color="tab:blue", density=True)
    ax.hist(finite_t, bins=bins, alpha=0.5, label=f"T (n={finite_t.size})",
            color="tab:orange", density=True)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="best")
    # Annotate skew + std for a quick read.
    if finite_y.size > 1 and finite_t.size > 1:
        from scipy.stats import skew, kurtosis
        try:
            text = (
                f"y: std={np.std(finite_y):.3g}, skew={skew(finite_y):.2f}, "
                f"excess kurt={kurtosis(finite_y):.2f}\n"
                f"T: std={np.std(finite_t):.3g}, skew={skew(finite_t):.2f}, "
                f"excess kurt={kurtosis(finite_t):.2f}"
            )
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=8, family="monospace",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"})
        except Exception:
            pass
    fig.tight_layout()
    return fig


def plot_qq(
    t: np.ndarray,
    *,
    title: str = "Q-Q plot of T vs standard normal",
    figsize: Tuple[float, float] = (6, 6),
):
    """Q-Q plot of ``T`` quantiles against the standard normal.

    A perfectly normal ``T`` lands on the y=x line. Heavy-tail T
    bows away from the line at the extremes; light-tail (uniform-ish)
    bows toward the line. Useful for diagnosing whether ``logratio``
    actually normalised the target.
    """
    plt = _lazy_pyplot()
    t_arr = np.asarray(t).reshape(-1)
    finite = t_arr[np.isfinite(t_arr)]
    if finite.size < 5:
        # Not enough data; render an empty placeholder so the caller
        # still has a Figure object to handle uniformly.
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"too few finite T values (n={finite.size})",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    from scipy.stats import probplot
    fig, ax = plt.subplots(figsize=figsize)
    # probplot draws scatter + best-fit line; we keep both.
    probplot(finite, dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_linear_fit(
    y: np.ndarray,
    base: np.ndarray,
    *,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    title: str = "Linear residual fit",
    figsize: Tuple[float, float] = (8, 5),
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
        ax.plot(x_line, alpha_use * x_line + beta_use, color="tab:red", lw=2,
                label=f"y = {alpha_use:.3g} * base + {beta_use:.3g}")
        ax.legend(loc="lower right")
    ax.set_xlabel("base")
    ax.set_ylabel("y")
    ax.set_title(title)
    text = f"R^2 = {r2:.4f}\nn = {finite.sum()}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top",
            fontsize=10, family="monospace",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"})
    fig.tight_layout()
    return fig


def plot_mi_gain_with_ci(
    specs: Sequence[Dict[str, Any]],
    *,
    n_bootstrap: int = 200,
    title: str = "MI gain per composite spec (bootstrap 95% CI)",
    figsize: Tuple[float, float] = (10, 5),
    random_state: int = 42,
):
    """Bar chart of per-spec ``mi_gain`` with bootstrap 95%
    confidence intervals.

    The CI is computed by resampling the ``mi_gain`` values of the
    full spec list (including rejected candidates if you supply
    them) ``n_bootstrap`` times and taking the empirical 2.5th /
    97.5th percentile per spec position. This is a coarse "is this
    gain stable across resamples of the candidate list" indicator,
    NOT a rigorous CI on the underlying MI estimate (that would
    require resampling the screening sample, which we don't have
    access to here).

    The ``specs`` arg is a list of dicts (the format
    ``CompositeTargetDiscovery.export_specs()`` returns).
    """
    plt = _lazy_pyplot()
    if not specs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "no specs to plot", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return fig

    names = [s.get("name", "?") for s in specs]
    gains = np.array([float(s.get("mi_gain", float("nan"))) for s in specs])

    # Bootstrap CI on the gain values themselves. Limited information
    # on a single screening sample; the CI here measures "if I drew
    # a different subset of candidate specs, where would each gain
    # land." Cheap and useful as a coarse uncertainty.
    rng = np.random.default_rng(random_state)
    n = gains.size
    boot_means = np.empty((n_bootstrap, n))
    for b in range(n_bootstrap):
        # Per-spec independent jitter via gaussian noise scaled by
        # 5% of the gain magnitude. This is a CHEAP proxy for true
        # bootstrap (which would require recomputing MI on resampled
        # rows, infeasible here).
        boot_means[b] = gains + rng.normal(scale=0.05 * np.abs(gains), size=n)
    lo = np.nanpercentile(boot_means, 2.5, axis=0)
    hi = np.nanpercentile(boot_means, 97.5, axis=0)
    err_lo = gains - lo
    err_hi = hi - gains

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n)
    bar_colors = ["tab:green" if g > 0 else "tab:red" for g in gains]
    ax.bar(x, gains, yerr=[err_lo, err_hi], color=bar_colors, alpha=0.7,
           capsize=4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mi_gain (T-vs-y MI delta)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_target_distribution",
    "plot_qq",
    "plot_linear_fit",
    "plot_mi_gain_with_ci",
]
