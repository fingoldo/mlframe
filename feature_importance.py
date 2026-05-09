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


def plot_feature_importance(
    feature_importances: np.ndarray,
    columns: Sequence,
    kind: str,
    n: int = 20,
    figsize: tuple = (12, 6),
    positive_fi_only: bool = False,
    show_plots: bool = True,
    plot_file: str = "",
):

    sorted_idx = np.argsort(feature_importances)
    if len(columns) == 0:
        columns = np.arange(len(feature_importances))
    sorted_columns = np.array(columns)[sorted_idx]
    df = pd.Series(data=feature_importances[sorted_idx], index=sorted_columns, name="fi").to_frame().sort_values(by="fi", ascending=False)
    if positive_fi_only:
        df = df[df.fi > 0.0]

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
