"""Early stop between the FE stage cascades: one column that already carries (almost) all the information of ``y`` makes the remaining families pointless.

A candidate is dominant when its bias-corrected plug-in MI with the target reaches ``_DOMINANT_FRACTION`` of the target entropy (the residual uncertainty of y given the column is then below the
noise floor a second feature could add to). Small frames and continuous targets with too few distinct values never qualify; ``MLFRAME_FE_DOMINANT_STOP=0`` switches the check off.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ._fe_frame_ops import fe_columns, fe_extract_columns, fe_is_numeric_col
from typing import Any

logger = logging.getLogger(__name__)

_DOMINANT_FRACTION = 0.99
_MIN_ROWS = 5_000
_N_BINS = 64
_Y_BINS = 10
_MAX_Y_LEVELS = 20


def _target_codes(y: np.ndarray) -> "np.ndarray | None":
    """Dense int codes of a one-dimensional target (quantile bins for a continuous one); None when unusable."""
    y = np.asarray(y)
    if y.ndim != 1 or y.dtype.kind not in "biuf":
        return None
    if y.dtype.kind == "f" and not np.isfinite(y).all():
        return None
    levels, codes = np.unique(y, return_inverse=True)
    if len(levels) <= _MAX_Y_LEVELS:
        dense: np.ndarray = codes.astype(np.int32)
        return dense
    edges = np.quantile(y, np.linspace(0, 1, _Y_BINS + 1)[1:-1])
    binned: np.ndarray = np.searchsorted(np.unique(edges), y).astype(np.int32)
    return binned


#: Host bytes of one block of columns scored together (the sanitised float32 matrix of the block).
_BLOCK_BYTES = 256 * 1024 * 1024


def dominant_candidate(X: Any, y: Any, n_rows_min: int = _MIN_ROWS) -> "str | None":
    """Name of a numeric column of ``X`` whose corrected MI with ``y`` reaches ``_DOMINANT_FRACTION`` of H(y), else None."""
    if os.environ.get("MLFRAME_FE_DOMINANT_STOP", "1") == "0":
        return None
    codes = _target_codes(y)
    n = 0 if codes is None else len(codes)
    if codes is None or n < n_rows_min:
        return None
    counts = np.bincount(codes)
    p = counts[counts > 0] / n
    h_y = float(-(p * np.log(p)).sum())
    if h_y <= 0:
        return None
    names = [c for c in fe_columns(X) if fe_is_numeric_col(X, c)]
    if not names:
        return None
    from ._hermite_fe_mi import plugin_mi_classif_batch_dispatch

    # A frame that fits the budget is scored in one call, as before. A larger one goes in blocks of columns: the extracted columns, their sanitised copies and the stacked matrix were
    # three frame-sized arrays at once (1.2 GB for a million rows x 100 columns), and every column's MI is its own.
    width = len(names) if n * len(names) * 4 <= _BLOCK_BYTES else max(1, _BLOCK_BYTES // (n * 4))
    parts = []
    for start in range(0, len(names), width):
        block = names[start : start + width]
        cols = fe_extract_columns(X, block)
        mat = np.column_stack([np.nan_to_num(np.asarray(cols[c], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) for c in block])
        del cols
        # Intentional float64: MI statistic vector from the plug-in estimator
        parts.append(np.asarray(plugin_mi_classif_batch_dispatch(mat, codes, _N_BINS), dtype=np.float64))
        del mat
    mi = np.concatenate(parts)
    bias = (_N_BINS - 1) * (len(p) - 1) / (2.0 * n)
    best = int(np.argmax(mi))
    if mi[best] - bias >= _DOMINANT_FRACTION * h_y:
        logger.info("FE early stop: column %r explains %.4f of the target entropy, remaining stages skipped", names[best], (mi[best] - bias) / h_y)
        return names[best]
    return None
