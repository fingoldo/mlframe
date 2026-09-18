"""Dense 0-based class codes for the FE step's MI / CMI scorers."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def dense_class_codes(classes_y: ArrayLike) -> np.ndarray:
    """Map ``classes_y`` to a 1-D array of dense int64 codes 0..K-1, one per row.

    Distinct values stay distinct (no integer cast before ``np.unique``, which would merge fractional labels), and the input is raveled first:
    numpy >= 2 shapes ``return_inverse`` like its input, so an ``(n, 1)`` target would otherwise give ``(n, 1)`` codes.
    """
    _, dense = np.unique(np.asarray(classes_y).ravel(), return_inverse=True)
    return dense.astype(np.int64).ravel()
