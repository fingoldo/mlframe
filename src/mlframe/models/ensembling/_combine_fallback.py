"""NaN/inf fallback for ``combine_probs``, kept beside ``base.py`` so that module stays under the 1k-line ceiling."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def _finite_member_mean(stacked: np.ndarray, weights_arr: Optional[np.ndarray]) -> np.ndarray:
    """Per-cell mean over the members that are FINITE at that cell, weighted when ``weights_arr`` is given.

    This is the NaN/inf fallback of ``combine_probs``. It must ignore the non-finite members rather than average them
    in: ``np.mean`` / ``np.average`` over a stack containing one NaN member is NaN everywhere, so the documented
    "fallback to arithmetic mean" reproduced the exact value it was supposed to repair. Weights (NNLS / Caruana) are
    honoured and renormalised over the members finite at each cell - an unweighted fallback would revert just those
    rows to an unweighted mean while every other row stayed weighted. A cell no member could predict stays NaN.
    """
    finite = np.isfinite(stacked)
    if weights_arr is not None:
        w = np.asarray(weights_arr, dtype=np.float64).reshape((-1,) + (1,) * (stacked.ndim - 1))
        w_eff = np.where(finite, w, 0.0)
        w_sum = w_eff.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(w_sum > 0, (np.where(finite, stacked, 0.0) * w_eff).sum(axis=0) / w_sum, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stacked, axis=0)
