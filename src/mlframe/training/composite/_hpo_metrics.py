"""Shared scoring primitives for :mod:`hpo` and :mod:`hpo_ensembling`.

Extracted as a leaf module so both siblings can import ``rmse`` without a
module-level cycle (``hpo`` re-exports ``hpo_ensembling``'s public names at
its bottom; ``hpo_ensembling`` needs a default scorer matching
``optimize_composite``'s own default). Neither sibling imports the other
here, so this file has no internal mlframe dependency to cycle through.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rmse"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default scorer: root-mean-squared error on original ``y`` scale.

    Non-finite predictions (an inverse-transform blow-up on a pathological
    fold) are penalised with ``inf`` so the optimizer steers away from the
    config rather than crashing the whole search.
    """
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    if not np.all(np.isfinite(diff)):
        return float("inf")
    return float(np.sqrt(np.mean(diff * diff)))
