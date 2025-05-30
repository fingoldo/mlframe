"""Computing of exponentially weighted values."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

# ensure_installed("numpy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import numpy as np
from numba import njit


def ewma(x, alpha) -> np.ndarray:
    """
    Returns the exponentially weighted moving average of x.

    >>>alpha = 0.55
    >>>x = np.random.randint(0,30,15)
    >>>df = pd.DataFrame(x, columns=['A'])
    >>>np.allclose(df.ewm(alpha=alpha).mean().values.flatten(),ewma(x=x, alpha=alpha))
    True

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    """
    # Coerce x to an array
    # x = np.array(x)

    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * (1 - alpha)
    p = np.vstack([np.arange(i, i - n, -1) for i in range(n)])

    # Create the weight matrix
    w = np.tril(w0**p, 0)

    # Calculate the ewma
    return np.dot(w, x[:: np.newaxis]) / w.sum(axis=1)


@njit
def compute_p(n: int) -> np.ndarray:
    p = np.empty((n, n), dtype=np.int64)  # Preallocate the output array
    for i in range(n):
        for j in range(n):
            p[i, j] = i - j
    return p


@njit()
def ewma_numba(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Returns the exponentially weighted moving average of x.
    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    """

    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * (1 - alpha)
    p = compute_p(n)

    # Create the weight matrix
    w = np.tril(w0**p, 0).astype(x.dtype)

    # Calculate the ewma
    return np.dot(w, np.ascontiguousarray(x[:: np.newaxis])) / w.sum(axis=1)
