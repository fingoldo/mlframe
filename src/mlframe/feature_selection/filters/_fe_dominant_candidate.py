"""Early stop between the FE stage cascades: one column that already carries (almost) all the information of ``y`` makes the remaining families pointless.

A candidate is dominant when its bias-corrected plug-in MI with the target reaches ``_DOMINANT_FRACTION`` of the target entropy (the residual uncertainty of y given the column is then below the
noise floor a second feature could add to). Small frames and continuous targets with too few distinct values never qualify; ``MLFRAME_FE_DOMINANT_STOP=0`` switches the check off.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ._fe_frame_ops import fe_columns, fe_extract_columns, fe_is_numeric_col, fe_stack_columns, fe_subsample_to_pandas
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
#: Rows of the evenly spaced sample a long frame is screened on, and the frame length from which the screen is used (below it the sample is no cheaper than the frame).
_SCREEN_ROWS = 65_536
_SCREEN_MIN_ROWS = 4 * _SCREEN_ROWS
#: A column survives the screen when its corrected MI on the sample reaches this share of the sample's H(y). The bar of a dominant column is 0.99 of H(y) on all rows; the plug-in MI of 65536 rows
#: is within a few thousandths of the full-length value, so a column that clears the bar is nowhere near 0.9 on the sample.
_SCREEN_FRACTION = 0.9
#: The screen is used only when the bias of the plug-in MI is a small share of H(y) on the sample (a target with a handful of rare rows in the sample is decided on all rows).
_SCREEN_MIN_ENTROPY_OVER_BIAS = 50.0


def _column_mis(X: Any, names: "list[str]", codes: np.ndarray) -> np.ndarray:
    """Plug-in MI (``_N_BINS`` quantile bins) of the columns ``names`` of ``X`` with the target ``codes``, scored a block of columns at a time."""
    from ._hermite_fe_mi import plugin_mi_classif_batch_dispatch

    n = len(codes)
    # A frame that fits the budget is scored in one call, as before. A larger one goes in blocks of columns: the extracted columns, their sanitised copies and the stacked matrix were
    # three frame-sized arrays at once (1.2 GB for a million rows x 100 columns), and every column's MI is its own.
    width = len(names) if n * len(names) * 4 <= _BLOCK_BYTES else max(1, _BLOCK_BYTES // (n * 4))
    parts = []
    for start in range(0, len(names), width):
        block = names[start : start + width]
        cols = fe_extract_columns(X, block)
        # each column is written into its place once; a NaN or an infinity becomes 0 (``isfinite`` is False for exactly those), and only a column that has one pays the second pass
        # (``nan_to_num`` per column was 2.4 ms at 100000 rows, most of this stage's 0.3 s of matrix building)
        mat = fe_stack_columns([np.asarray(cols[c]) for c in block], np.float32)  # filled in row tiles
        finite = np.isfinite(mat)
        if not bool(finite.all()):
            mat = np.where(finite, mat, np.float32(0.0))
        del cols
        # Intentional float64: MI statistic vector from the plug-in estimator
        parts.append(np.asarray(plugin_mi_classif_batch_dispatch(mat, codes, _N_BINS), dtype=np.float64))
        del mat
    return np.concatenate(parts)


def _entropy_and_bias(codes: np.ndarray) -> "tuple[float, float]":
    """``(H(y), bias of the plug-in MI)`` of the target ``codes`` in nats."""
    counts = np.bincount(codes)
    p = counts[counts > 0] / len(codes)
    return float(-(p * np.log(p)).sum()), (_N_BINS - 1) * (len(p) - 1) / (2.0 * len(codes))


def _screened(X: Any, names: "list[str]", codes: np.ndarray) -> "list[str]":
    """The columns of ``names`` that can still be dominant after a look at an evenly spaced sample of the rows (all of them when the sample cannot tell).

    Scoring every column of a 500000 x 1400 frame at full length stacked 2.8 GB on one thread (21 s in a 500k x 1000 fit, both devices idle); the sample is an eighth of the rows, and the full-length
    scoring that decides is left to the few columns that clear the screen (none, in almost every fit)."""
    n = len(codes)
    if n < _SCREEN_MIN_ROWS:
        return names
    rows = np.linspace(0, n - 1, _SCREEN_ROWS).astype(np.int64)
    sample_codes = codes[rows]
    h_sample, bias_sample = _entropy_and_bias(sample_codes)
    if h_sample <= 0 or h_sample < _SCREEN_MIN_ENTROPY_OVER_BIAS * bias_sample:
        return names
    mi_sample = _column_mis(fe_subsample_to_pandas(X, rows), names, sample_codes)
    keep = np.flatnonzero(mi_sample - bias_sample >= _SCREEN_FRACTION * h_sample)
    return [names[j] for j in keep]


def dominant_candidate(X: Any, y: Any, n_rows_min: int = _MIN_ROWS) -> "str | None":
    """Name of a numeric column of ``X`` whose corrected MI with ``y`` reaches ``_DOMINANT_FRACTION`` of H(y), else None."""
    if os.environ.get("MLFRAME_FE_DOMINANT_STOP", "1") == "0":
        return None
    codes = _target_codes(y)
    n = 0 if codes is None else len(codes)
    if codes is None or n < n_rows_min:
        return None
    h_y, bias = _entropy_and_bias(codes)
    if h_y <= 0:
        return None
    names = [c for c in fe_columns(X) if fe_is_numeric_col(X, c)]
    if names:
        names = _screened(X, names, codes)
    if not names:
        return None
    mi = _column_mis(X, names, codes)
    best = int(np.argmax(mi))
    if mi[best] - bias >= _DOMINANT_FRACTION * h_y:
        logger.info("FE early stop: column %r explains %.4f of the target entropy, remaining stages skipped", names[best], (mi[best] - bias) / h_y)
        return names[best]
    return None
