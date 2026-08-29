"""Row-filtering shared by the binary-scorer builders.

``_finite_binary`` is imported by five modules (binary, the decile table, decision_curve, model_card,
model_comparison, split_comparison), so it lives in a neutral module: having it in ``binary.py`` made every
one of those an importer of the whole panel module, and made the carved decile table import its parent back.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _finite_binary(y_true, y_score) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite (y_true in {0,1}, y_score) pairs as float64 / int8 arrays.

    Non-finite scores and labels outside {0, 1} are dropped (the binary panels are one-vs-rest
    on the positive class), mirroring how the regression panels drop non-finite pairs up front.
    """
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    mask = np.isfinite(ys)
    yt_f = np.asarray(yt, dtype=np.float64)
    label_ok = np.isfinite(yt_f) & ((yt_f == 0.0) | (yt_f == 1.0))
    mask &= label_ok
    # Dropping an off-{0,1} label silently means a multiclass target passed here by mistake yields confident
    # binary curves computed on whatever subset happened to be 0 or 1 -- a plausible-looking chart about a
    # question nobody asked. Say how many went, and name the offending labels.
    n_bad_label = int(np.count_nonzero(np.isfinite(yt_f) & ~label_ok))
    if n_bad_label:
        offenders = np.unique(yt_f[np.isfinite(yt_f) & ~label_ok])[:5].tolist()
        logger.warning(
            "binary charts: dropped %d of %d rows whose label is outside {0, 1} (saw %s). These panels are "
            "one-vs-rest on the positive class; pass a binarised target, or use the multiclass composer.",
            n_bad_label, yt_f.size, offenders,
        )
    return yt_f[mask].astype(np.int8), ys[mask]


__all__ = ["_finite_binary"]
