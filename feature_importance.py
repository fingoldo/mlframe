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
import pandas as pd, numpy as np
from matplotlib import pyplot as plt

# *****************************************************************************************************************************************************
# Feature importances
# *****************************************************************************************************************************************************

def plot_feature_importance(
    feature_importances: np.ndarray, columns: Sequence, kind: str, n: int = 20, figsize: tuple = (12, 6), show_plots: bool = True, plot_file: str = ""
):

    sorted_idx = np.argsort(feature_importances)[-n:]
    sorted_columns = np.array(columns)[sorted_idx]
    df = pd.Series(data=feature_importances[sorted_idx], index=sorted_columns, name="fi").to_frame().sort_values(by="fi", ascending=False)

    if plot_file or show_plots:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca() # visible=True
        ax.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
        ax.set(yticks=range(len(sorted_idx)), yticklabels=sorted_columns)
        ax.set_title(f"{kind} feature importances")

        if plot_file:
            fig.savefig(plot_file)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    return df