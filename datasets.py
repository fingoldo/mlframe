# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import scipy
from scipy import stats
from scipy.stats import norm
import numpy as np, pandas as pd


# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def I(cond: np.ndarray) -> np.ndarray:
    # Indicator function
    return cond.astype(int)


def get_sapp_dataset(
    loc: float = 0.0,
    scale: float = 9.0,
    distr_name: str = "norm",
    distr_params: tuple = (),
    N: int = 1000,
    add_error: bool = False,
    random_state: int = 42,
    dtype=np.float32,
    binarize: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Used in work
    Subsemble: an ensemble method for combining subset-specific algorithm fits
    Stephanie Sapp, Mark J. van der Laan, and John Canny
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4000126/pdf/nihms-539092.pdf
    """

    np.random.seed(random_state)

    df = pd.DataFrame()
    for i in range(20):
        df[f"X{i+1}"] = getattr(stats, distr_name).rvs(loc, scale, *distr_params, size=N)

    target = df.eval(
        "X1+sin(X2)+log(abs(X3))+X4**2+X5*X6 +@I((X7*X8*X9)<0)+@I(X10>0)+X11*@I(X11>0)+sqrt(abs(X12)) +cos(X13)+2*X14+abs(X15)+@I(X16<-1)+X17*@I(X17<-1)-2*X18-X19*X20"
    )

    if add_error:
        target += getattr(stats, distr_name).rvs(loc, scale, *distr_params, size=N)
    if binarize:
        target = (target > target.mean()).astype(np.int8)
    return df.astype(dtype), target.astype(dtype)


def showcase_pycaret_datasets():

    from pycaret.datasets import get_data

    df = get_data(verbose=False)
    df["# Instances"] = df["# Instances"].astype(np.int32)
    return df.sort_values("# Instances").tail(20).reset_index(drop=True)
