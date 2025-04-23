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

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import
import pandas as pd, polars as pl, numpy as np
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

import shap

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

    if plot_file or show_plots:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()  # visible=True
        ax.barh(range(len(sorted_idx[-n:])), feature_importances[sorted_idx[-n:]], align="center")
        ax.set(yticks=range(len(sorted_idx[-n:])), yticklabels=sorted_columns[-n:])
        ax.set_title(f"{kind} feature importances")

        if plot_file:
            fig.savefig(plot_file)

        if not positive_fi_only and feature_importances[0] < 0:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()  # visible=True
            ax.barh(range(len(sorted_idx[:n])), feature_importances[sorted_idx[:n]], align="center")
            ax.set(yticks=range(len(sorted_idx[:n])), yticklabels=sorted_columns[:n])
            ax.set_title(f"{kind} BOTTOM feature importances")

        if show_plots:
            plt.ion()
            plt.show()
        else:
            plt.close(fig)

    return df


def compute_permutation_importances(*sklearn_args, columns: list, **sklearn_kwargs) -> pl.DataFrame:

    result = permutation_importance(*sklearn_args, **sklearn_kwargs)

    return (
        pl.DataFrame({**result, **dict(feature=columns)})
        .drop("importances")
        .filter(~((pl.col("importances_mean") == 0) & (pl.col("importances_std") == 0)))
        .sort(pl.col("importances_mean") - pl.col("importances_std") * 0.2, descending=True)
    )
