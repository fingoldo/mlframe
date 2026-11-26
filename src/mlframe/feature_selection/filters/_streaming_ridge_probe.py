"""Held-out standardise-then-ridge probe of a column set, fitted from row-chunked normal equations: the dense (rows, columns) design is never formed.

``_heldout_ridge_r2`` (the downstream no-harm guard of the raw-redundancy drop) used to stack every kept column and every raw column into one double-precision matrix of the full row count, and
``ScaledRidge`` then standardised a second copy of its training rows: at 500000 rows and 1000 raw columns that is 4 GB for the stack, 4 GB for the list the stack was built from and the
standardised copy on top (the 7.6 GB peak of the whole fit sat in this one step, and 1M rows would need twice that). The probe needs only the columns' means and spreads, the (columns, columns)
Gram matrix and the right-hand side of the standardised training rows, and the scores of the held-out rows: all are sums over rows, so they are accumulated over chunks of a few thousand rows.

The arithmetic is that of ``ScaledRidge(alpha).fit(X[:sp], y[:sp])`` followed by ``fast_r2_score(y[sp:], model.predict(X[sp:]))``: values are rounded to the criterion dtype first (as the dense probe did),
non-finite values are zero, the spread is the population standard deviation of the training rows (a column that is constant up to rounding keeps a spread of 1), the ridge system is the centred normal
equations with the penalty ``alpha`` on the diagonal and an unpenalised intercept.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import linalg as _sla

from mlframe.metrics.core import fast_r2_score

#: Bytes of the float64 chunk of the design that is held at a time (the chunk is column-major, so a column is copied in as one contiguous run).
CHUNK_BYTES = 128 * 1024 * 1024
#: Smallest number of training and of held-out rows for which the probe is evaluated.
MIN_SPLIT_ROWS = 50


def _chunk_rows(n_rows: int, n_cols: int) -> int:
    """Rows per chunk: the double-precision chunk of ``n_cols`` columns stays within ``CHUNK_BYTES`` (at least 1024 rows, at most all of them)."""
    return int(max(1024, min(n_rows, CHUNK_BYTES // max(1, n_cols * 8))))


def _load_chunk(columns: Sequence[np.ndarray], r0: int, r1: int, dtype: "np.dtype") -> np.ndarray:
    """Rows ``r0:r1`` of every column as a column-major chunk in double precision, rounded to ``dtype`` first with the non-finite values set to zero."""
    # Intentional float64: the normal equations square the conditioning of the columns, so the chunk is held in double precision
    out = np.empty((r1 - r0, len(columns)), dtype=np.float64, order="F")
    for j, col in enumerate(columns):
        out[:, j] = np.nan_to_num(np.asarray(col[r0:r1], dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _train_moments(columns: Sequence[np.ndarray], sp: int, rows: int, dtype: "np.dtype") -> "tuple[np.ndarray, np.ndarray]":
    """Mean and population standard deviation of the first ``sp`` rows of every column (Chan's merge of the chunk moments; a constant column keeps a spread of 1, as ``StandardScaler`` decides)."""
    p = len(columns)
    count = 0
    mean = np.zeros(p)
    m2 = np.zeros(p)
    for r0 in range(0, sp, rows):
        chunk = _load_chunk(columns, r0, min(sp, r0 + rows), dtype)
        m = chunk.shape[0]
        c_mean = chunk.mean(axis=0)
        c_m2 = np.square(chunk - c_mean).sum(axis=0)
        delta = c_mean - mean
        total = count + m
        mean = mean + delta * (m / total)
        m2 = m2 + c_m2 + np.square(delta) * (count * m / total)
        count = total
    var = m2 / count
    # Intentional float64: the spread test of the standardiser is made in double precision
    eps = np.finfo(np.float64).eps
    constant = var <= count * eps * var + (count * mean * eps) ** 2
    return mean, np.where(constant, 1.0, np.sqrt(var))


def heldout_ridge_r2_of_columns(columns: Sequence[np.ndarray], y: np.ndarray, *, frac: float = 0.7, alpha: float = 1.0, dtype: "np.dtype | type" = np.float32) -> Optional[float]:
    """Held-out R^2 of a standardise-then-ridge fit on the first ``frac`` of the rows, scored on the rest; ``None`` when the split or the fit is degenerate.

    ``columns``: one 1-D array per column, all of the length of ``y``, in any float or integer dtype (read in row chunks, never copied whole). ``dtype``: the precision the values are rounded to before
    they are widened to double (the criterion dtype of the dense probe)."""
    dtype = np.dtype(dtype)
    target = np.nan_to_num(np.asarray(y, dtype=dtype).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    n = int(target.shape[0])
    p = len(columns)
    if p == 0 or any(int(c.shape[0]) != n for c in columns):
        return None
    sp = int(n * frac)
    if sp < MIN_SPLIT_ROWS or (n - sp) < MIN_SPLIT_ROWS or float(np.std(target[:sp])) < 1e-12:
        return None
    rows = _chunk_rows(n, p)
    mean, scale = _train_moments(columns, sp, rows, dtype)
    # Intentional float64: the normal equations square the conditioning of the columns, so they are formed and solved in double precision
    t = target.astype(np.float64)
    t_mean = float(t[:sp].mean())
    gram = np.zeros((p, p))
    rhs = np.zeros(p)
    for r0 in range(0, sp, rows):
        r1 = min(sp, r0 + rows)
        z = (_load_chunk(columns, r0, r1, dtype) - mean) / scale
        gram += z.T @ z
        rhs += z.T @ (t[r0:r1] - t_mean)
    gram[np.diag_indices_from(gram)] += float(alpha)
    try:
        weights = _sla.cho_solve(_sla.cho_factor(gram, lower=True, check_finite=False), rhs, check_finite=False)
    except (np.linalg.LinAlgError, _sla.LinAlgError):  # not positive definite (non-finite sums): the general solver decides
        try:
            weights = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:  # best-effort: the caller leaves its decision unchanged on None
            return None
    if not np.isfinite(weights).all():
        return None
    intercept = t_mean  # the standardised training rows have mean zero up to rounding, so the intercept is the mean of the training target
    predicted = np.empty(n - sp)
    for r0 in range(sp, n, rows):
        r1 = min(n, r0 + rows)
        predicted[r0 - sp : r1 - sp] = ((_load_chunk(columns, r0, r1, dtype) - mean) / scale) @ weights + intercept
    return float(fast_r2_score(target[sp:], predicted))
