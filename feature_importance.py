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


def plot_feature_importance(
    feature_importances: np.ndarray,
    columns: Sequence,
    kind: str,
    n: int = 20,
    figsize: tuple = (12, 6),
    positive_fi_only: bool = False,
    show_plots: bool = True,
    plot_file: str = "",
    log_top_n: int = 20,
    log_fi: bool = True,
):
    """Plot + log top-N feature importances.

    Parameters
    ----------
    feature_importances : np.ndarray
        Per-feature importance scores aligned to ``columns``.
    columns : Sequence
        Feature names; ``len(columns)`` may be 0 (uses integer index).
    kind : str
        Display label (e.g. ``"XGBRegressor TVT"``). Goes into the
        plot title AND the text log line.
    n : int, default 20
        Number of bars to plot. Top-N positive AND bottom-N negative
        (when ``positive_fi_only=False`` and the minimum FI is < 0).
    log_top_n : int, default 20
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
    # rule as ``mlframe.metrics.show_calibration_plot``: in a script /
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
        figs = []
        fig_top = plt.figure(figsize=figsize)
        figs.append(fig_top)
        ax = plt.gca()  # visible=True
        ax.barh(range(len(sorted_idx[-n:])), feature_importances[sorted_idx[-n:]], align="center")
        ax.set(yticks=range(len(sorted_idx[-n:])), yticklabels=sorted_columns[-n:])
        ax.set_title(f"{kind} feature importances")

        if plot_file:
            fig_top.savefig(plot_file)

        if not positive_fi_only and feature_importances[sorted_idx[0]] < 0:
            fig_bot = plt.figure(figsize=figsize)
            figs.append(fig_bot)
            ax = plt.gca()  # visible=True
            ax.barh(range(len(sorted_idx[:n])), feature_importances[sorted_idx[:n]], align="center")
            ax.set(yticks=range(len(sorted_idx[:n])), yticklabels=sorted_columns[:n])
            ax.set_title(f"{kind} BOTTOM feature importances")

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
            if _in_kernel:
                try:
                    from IPython.display import display as _ipy_display
                    for _fig in figs:
                        _ipy_display(_fig)
                except Exception:
                    # Fall back to plt.show on any import / display
                    # error; the warning is acceptable here.
                    plt.ion()
                    plt.show()
            else:
                plt.ion()
                plt.show()
        # Close ALL figs (top + bottom) unless inside an IPython /
        # Jupyter kernel where the inline display already rendered
        # them. Previously only the last-assigned fig was closed in
        # the ``not show_plots`` branch (top-FI leaked whenever the
        # bottom branch also fired) AND the ``show_plots=True`` path
        # never closed anything (suite-default leak per fit). 2026-
        # 05-09 leak fix; helper unifies the detection across modules.
        from mlframe.metrics import _close_unless_interactive
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
