
from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import re
from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

from os.path import join
from matplotlib import pyplot as plt
import pandas as pd, polars as pl, numpy as np
from sklearn.inspection import permutation_importance

from pyutilz.system import ensure_dir_exists

import shap

# Precompile once; strips anything that could turn ``model_name`` into a path
# traversal, a hidden file, or a Windows-reserved character when interpolated
# into a filename.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ =\[\]@\-]+")


def _sanitize_for_filename(s: str, max_len: int = 120) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", str(s)).strip(" .")
    return cleaned[:max_len] if cleaned else "unnamed"

# *****************************************************************************************************************************************************
# Feature importances
# *****************************************************************************************************************************************************


def show_shap_beeswarm_plot(model: object, df: pd.DataFrame, **kwargs):

    shap.initjs()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)

    shap.plots.beeswarm(shap_values, **kwargs)


def _format_top_fi_for_log(
    sorted_df: pd.DataFrame,
    top_n: int,
    kind: str,
    width: int = 110,
) -> str:
    """Build a compact one-paragraph string of the top-N feature
    importances suitable for ``logger.info``. Format:

      [FI top-N] <kind>: feat_a=0.342, feat_b=0.198, ... (N=15, sum=...)

    Truncates feature names to keep the line within ``width`` chars.
    """
    head = sorted_df.head(top_n)
    if head.empty:
        return f"[FI top-{top_n}] {kind}: (no features)"
    items = [
        f"{str(name)[:40]}={fi:+.4g}"
        for name, fi in zip(head.index, head["fi"].values)
    ]
    total_n = len(sorted_df)
    sum_top = float(head["fi"].sum())
    sum_all = float(sorted_df["fi"].sum())
    share = (sum_top / sum_all * 100.0) if sum_all > 0 else 0.0
    parts = [f"[FI top-{len(items)}] {kind}:"]
    line = parts[-1]
    for item in items:
        candidate = (line + " " + item) if line.endswith(":") \
            else (line + ", " + item)
        if len(candidate) > width:
            parts.append("  " + item)
            line = parts[-1]
        else:
            parts[-1] = candidate
            line = parts[-1]
    parts.append(
        f"  (N_total={total_n}, top_sum={sum_top:.4g}, "
        f"share={share:.1f}%)"
    )
    return "\n".join(parts)


# 2026-05-26 (user request): bumped from 10 to 15 after the 33-new-
# features rollout pushed the informative tail past the top-10 cut.
# Override via FeatureImportanceConfig(num_factors=N) or via the
# log_top_n / n kwargs.
_FI_PLOT_DEFAULT_N: int = 15
_FI_LOG_DEFAULT_TOP_N: int = 15
# 2026-05-12 (user request): cap how many bars-with-FI~0 the chart shows.
# Tree models on a residual target often pin 1-2 features near 1.0 and zero
# everything else; rendering 23 invisible bars wastes vertical real-estate.
# Show at most this many zero-magnitude features in the bar plot, dropped
# from the bottom of the magnitude-sorted list.
_FI_DEFAULT_MAX_ZERO: int = 4

# 2026-05-13 (user request): default FI figsize is intentionally half the
# regression-diagnostic 3-panel chart (DEFAULT_FIGSIZE = (15, 5) in
# evaluation.py). Direct callers outside the suite (tests, notebooks) get the
# compact size too; the suite layer still passes
# ``reporting_config.feature_importance_config.figsize`` explicitly.
_FI_DEFAULT_FIGSIZE: tuple = (7.5, 2.5)


def plot_feature_importance(
    feature_importances: np.ndarray,
    columns: Sequence,
    kind: str,
    n: int = _FI_PLOT_DEFAULT_N,
    figsize: tuple = _FI_DEFAULT_FIGSIZE,
    positive_fi_only: bool = False,
    show_plots: bool = True,
    plot_file: str = "",
    log_top_n: int = _FI_LOG_DEFAULT_TOP_N,
    log_fi: bool = True,
    max_zero_fi_to_plot: int = _FI_DEFAULT_MAX_ZERO,
):
    """Plot + log top-N feature importances.

    Parameters
    ----------
    feature_importances : np.ndarray
        Per-feature importance scores aligned to ``columns``.
    columns : Sequence
        Feature names; ``len(columns)`` may be 0 (uses integer index).
    kind : str
        Display label (e.g. ``"XGBRegressor y"``). Goes into the
        plot title AND the text log line.
    n : int, default 10
        Number of bars to plot. Top-N positive AND bottom-N negative
        (when ``positive_fi_only=False`` and the minimum FI is < 0).
        Default reduced from 25 to 10 (2026-05-12) so plots/logs stay
        scannable; raise via the ``reporting_config.fi_top_n`` knob in
        ``train_mlframe_models_suite``.
    log_top_n : int, default 10
        Number of features included in the text-format log line. Set
        to 0 to disable text logging entirely.
    log_fi : bool, default True
        Master switch for the text log line. Set False to suppress
        even when ``log_top_n > 0`` (e.g. when the caller already
        logs FI via a different path).

    Returns
    -------
    pd.DataFrame
        Sorted (descending) FI as a DataFrame with one column ``fi``,
        indexed by feature name.
    """
    sorted_idx = np.argsort(feature_importances)
    if len(columns) == 0:
        columns = np.arange(len(feature_importances))
    sorted_columns = np.array(columns)[sorted_idx]
    df = pd.Series(data=feature_importances[sorted_idx], index=sorted_columns, name="fi").to_frame().sort_values(by="fi", ascending=False)
    if positive_fi_only:
        df = df[df.fi > 0.0]

    # 2026-05-11: text-log of top-N FI (default ON). Emitted via
    # ``logger.info`` BEFORE the rendering branch so the log line
    # appears even when no plot is produced (e.g. headless / CI runs
    # where the figure short-circuit fires). Suppress with
    # ``log_fi=False`` or ``log_top_n=0``.
    if log_fi and log_top_n > 0 and not df.empty:
        try:
            logger.info(_format_top_fi_for_log(
                sorted_df=df, top_n=log_top_n, kind=kind,
            ))
        except Exception as _log_err:
            logger.debug(
                "FI text-log formatting failed for kind=%r: %s",
                kind, _log_err,
            )

    # 2026-05-09: short-circuit when nothing consumes the plot. Same
    # rule as ``mlframe.metrics.core.show_calibration_plot``: in a script /
    # CI / fuzz run the ``show_plots=True`` default renders a figure
    # for nobody (``plt.show()`` is a no-op on Agg, no ``plot_file``
    # to disk). The FI bar plot is 80-150 ms / call; the suite emits
    # one per (model, train+val+test) so this saves ~300-450 ms per
    # fit on top of cal-plot wins.
    if plot_file == "" and show_plots:
        try:
            _is_interactive_session = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            import sys as _sys
            _is_interactive_session = hasattr(_sys, "ps1")
        if not _is_interactive_session:
            return df  # render-side skipped; the importance series is the data return

    if plot_file or show_plots:
        # 2026-05-12 (user feedback): pick the top-N by ABSOLUTE magnitude in
        # one chart with signed bars, instead of emitting a separate "BOTTOM
        # feature importances" plot whenever the most-negative FI is < 0.
        # On linear models with ~25 features and the old n=20 default, the
        # TOP and BOTTOM views shared 15/20 features and looked like the
        # same chart printed twice. Magnitude-ranked single-plot rendering
        # solves both the duplication AND the title misnomer (a large
        # negative coefficient is high-importance, not "bottom").
        figs = []
        fig_top = plt.figure(figsize=figsize)
        figs.append(fig_top)
        ax = plt.gca()  # visible=True
        # Rank by |FI| descending. Use original signed values for the bar so
        # negative coefficients show as left-bars.
        # Wave 57 (2026-05-20): lexsort with feature-position tiebreaker so tied
        # zero-importance features (common after model pruning) pick the same
        # set across runs and the displayed bar chart stays reproducible.
        _abs_fi = np.abs(feature_importances)
        _abs_order_full = np.lexsort((np.arange(len(_abs_fi)), -_abs_fi))
        _picked = _abs_order_full[:n]
        # 2026-05-12 (user request): drop excess zero-FI bars so the chart
        # stays compact when most features got pruned by the model. ``eps``
        # is scaled to the LARGEST magnitude so we don't accidentally drop
        # legitimately small but nonzero importances (eg gain=1e-6 on a
        # near-orthogonal feature). Keep at most ``max_zero_fi_to_plot`` of
        # the zero-FI bars; if the head already exceeds ``n``, no zeros pass.
        _abs_picked = np.abs(feature_importances[_picked])
        if _abs_picked.size > 0:
            _fi_eps = max(1e-12, float(_abs_picked.max()) * 1e-6)
        else:
            _fi_eps = 1e-12
        _nonzero_mask = _abs_picked > _fi_eps
        _nonzero_ids = _picked[_nonzero_mask]
        _zero_ids = _picked[~_nonzero_mask][:max(0, int(max_zero_fi_to_plot))]
        _abs_order = np.concatenate([_nonzero_ids, _zero_ids])
        # Re-sort the picked subset by signed value so the bars stack
        # cleanly (most-negative at the bottom, most-positive at the top).
        _abs_order = _abs_order[np.argsort(feature_importances[_abs_order])]
        _picked_fi = feature_importances[_abs_order]
        _picked_cols = np.array(columns)[_abs_order]
        # 2026-05-13 (user request): match the perf-chart aesthetic --
        # translucent matplotlib-default blue bars + light-alpha grid +
        # explicit zero reference line. Pre-fix the FI plot used solid
        # opaque bars with no grid, visually clashing with the perf-chart
        # diagnostic above it (alpha=0.3 dots + grid(alpha=0.3)).
        ax.barh(
            range(len(_abs_order)), _picked_fi,
            align="center", alpha=0.7,
            color="tab:blue", edgecolor="tab:blue",
        )
        ax.set(yticks=range(len(_abs_order)), yticklabels=_picked_cols)
        ax.set_title(f"{kind} feature importances", fontsize=11)
        ax.set_xlabel("Importance")
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        if plot_file:
            # bbox_inches="tight" so long ytick labels (feature names) and
            # the title don't get cropped by the default figure bbox.
            fig_top.savefig(plot_file, bbox_inches="tight", pad_inches=0.15)

        if show_plots:
            # 2026-05-11: prefer explicit ``IPython.display.display(fig)``
            # when running inside a Jupyter / IPython kernel. This is
            # robust to the matplotlib backend being Agg (the global
            # mlframe pipeline may have set it for the on-disk rendering
            # path); ``plt.show()`` on Agg prints the "non-GUI backend"
            # warning and renders nothing. ``display(fig)`` works
            # regardless of backend because it hands the figure to the
            # kernel's display-data channel and the inline backend
            # renders it as a PNG payload.
            try:
                # ``__IPYTHON__`` is defined only inside an IPython /
                # Jupyter kernel (not in bare python or in a script
                # that happened to import IPython transitively).
                _in_kernel = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
            except NameError:
                _in_kernel = False
            _displayed_inline = False
            if _in_kernel:
                try:
                    from IPython.display import display as _ipy_display
                    for _fig in figs:
                        _ipy_display(_fig)
                    _displayed_inline = True
                except Exception:
                    # Fall back to plt.show on any import / display error.
                    from mlframe.metrics import show_plots_unless_agg
                    show_plots_unless_agg()
            else:
                from mlframe.metrics import show_plots_unless_agg
                show_plots_unless_agg()
            # 2026-05-26 Jupyter double-render fix: after
            # ``IPython.display.display(fig)`` the kernel has already
            # serialised the figure to PNG / SVG and shipped it to the
            # display channel; the matplotlib backing is now independent.
            # If we LEAVE the figure alive in matplotlib's pyplot
            # registry, the inline backend's end-of-cell auto-flush
            # picks it up and renders it AGAIN, producing the "толпа
            # FI графиков" wave after the composite-ensemble phase
            # finishes its async pipeline. Closing here drops the
            # registry reference without affecting the already-
            # displayed inline image. Out-of-kernel path is handled
            # by the unified ``_close_unless_interactive`` below.
            if _displayed_inline:
                for _fig in figs:
                    try:
                        plt.close(_fig)
                    except Exception:
                        pass
        # Close ALL figs (top + bottom) unless inside an IPython /
        # Jupyter kernel where the inline display has ALREADY captured
        # the rendered figure to the display channel. Previously only
        # the last-assigned fig was closed in the ``not show_plots``
        # branch (top-FI leaked whenever the bottom branch also fired)
        # AND the ``show_plots=True`` path never closed anything --
        # the explicit ``plt.close`` above now covers the Jupyter
        # branch, so this helper is the safety net for the no-show
        # path. 2026-05-09 leak fix; helper unifies the detection
        # across modules.
        from mlframe.metrics.core import _close_unless_interactive
        _close_unless_interactive(figs, was_shown=show_plots)

    return df


def compute_permutation_importances(*sklearn_args, columns: list, **sklearn_kwargs) -> pl.DataFrame:

    result = permutation_importance(*sklearn_args, **sklearn_kwargs)

    # `result` is a Bunch; "importances" is 2D (n_features x n_repeats) and
    # breaks Polars construction on some versions. Keep only the per-feature
    # 1-D arrays that survive conversion, then assemble the frame explicitly.
    frame = {
        key: np.asarray(value)
        for key, value in result.items()
        if key != "importances" and hasattr(value, "__len__") and np.asarray(value).ndim == 1
    }
    frame["feature"] = list(columns)

    return (
        pl.DataFrame(frame)
        .filter(~((pl.col("importances_mean") == 0) & (pl.col("importances_std") == 0)))
        .sort(pl.col("importances_mean") - pl.col("importances_std") * 0.2, descending=True)
    )


def explain_top_feature_importances(
    model: object,
    model_name: str,
    df: pd.DataFrame,
    beeswarm_plot_params: dict = None,
    save_chart: bool = True,
    figsize: tuple = (15, 20),
) -> None:
    if beeswarm_plot_params is None:
        beeswarm_plot_params = dict(max_display=30, group_remaining_features=False)
    fig, ax = plt.subplots(figsize=figsize)
    show_shap_beeswarm_plot(model.model, df, ax=ax, plot_size=None, show=False, **beeswarm_plot_params)
    fi_name = f"{model_name} {type(model.model).__name__} @iter={model.metrics.get('best_iter','')} [{len(model.columns):_}F]"
    _ = ax.set_title(fi_name)
    if save_chart:
        ensure_dir_exists("reports")
        safe_name = _sanitize_for_filename(fi_name)
        fig.savefig(join("reports", f"{safe_name}_shap_beeswarm.png"), bbox_inches="tight", dpi=400)
