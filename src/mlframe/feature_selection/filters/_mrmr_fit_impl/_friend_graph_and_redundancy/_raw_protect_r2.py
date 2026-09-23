"""Held-out R^2 scorer for the raw-feature floor-drop protection (``_group2``)."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# A triangular factor whose smallest |diagonal| falls below this fraction of its largest is treated as rank deficient.
_RCOND = 1e-10


def _well_conditioned(r: np.ndarray) -> bool:
    """True when the triangular factor ``r`` is safely full rank for ``solve_triangular``."""
    d = np.abs(np.diag(r))
    return bool(d.size) and bool(np.all(np.isfinite(d))) and float(d.min()) > _RCOND * float(d.max())


def heldout_r2_scorer(base_mat: np.ndarray | Sequence[np.ndarray], y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray) -> Callable[..., float]:
    """Return ``r2(extra=None)``: the held-out R^2 of a least-squares fit of ``y`` on ``[base | extra]``, trained on ``train_mask`` rows.

    ``base_mat`` is an ``(n, p)`` array or a sequence of ``p`` length-n columns; a sequence is sliced per column, so the full-height design
    never exists (only its train and validation blocks). ``extra`` is one full-length candidate column or an ``(n, m)`` block of columns that only mean something together (the sin/cos legs of one frequency), or ``None`` for the base design alone. The base is QR-factorised once and extended by one
    column per call, an O(n*p) update instead of a fresh solve. That is exact only while the factor is full rank: an unpivoted QR of a
    rank-deficient design (a raw column and its monotone twin, a count and its frequency encoding - both routinely selected together) has
    a near-zero diagonal, ``solve_triangular`` then returns huge coefficients without raising, and the R^2 becomes noise. So the fast path is
    taken only while the factor is well conditioned; otherwise the rank-revealing, minimum-norm ``lstsq`` decides.
    """
    import scipy.linalg as sla

    yv = y[val_mask]
    ss = float(np.sum((yv - yv.mean()) ** 2))
    y_tr = y[train_mask]
    if isinstance(base_mat, np.ndarray):
        base_tr = base_mat[train_mask]
        base_va = base_mat[val_mask]
    else:
        base_tr = np.column_stack([np.asarray(c)[train_mask] for c in base_mat])
        base_va = np.column_stack([np.asarray(c)[val_mask] for c in base_mat])

    def _lstsq_r2(a_tr: np.ndarray, a_va: np.ndarray) -> float:
        """Held-out R^2 from the rank-revealing least-squares solution."""
        coef = np.linalg.lstsq(a_tr, y_tr, rcond=None)[0]
        return 1.0 - float(np.sum((yv - a_va @ coef) ** 2)) / ss

    Q = R = coef_base = None
    try:
        Q, R = sla.qr(base_tr, mode="economic")
        if _well_conditioned(R):
            coef_base = sla.solve_triangular(R, Q.T @ y_tr)
        else:
            Q = R = None
    except Exception as exc:
        logger.debug("mrmr: QR of the raw-protection base design failed; scoring with lstsq: %r", exc, exc_info=True)
        Q = R = None

    def r2(extra=None):
        """Held-out R^2 of ``[base | extra]``, where ``extra`` is one column or an ``(n, m)`` block of columns added together."""
        if ss < 1e-24:
            return 0.0
        if extra is None:
            if coef_base is not None:
                return 1.0 - float(np.sum((yv - base_va @ coef_base) ** 2)) / ss
            return _lstsq_r2(base_tr, base_va)
        extra = np.asarray(extra, dtype=np.float64)
        extra = extra.reshape(-1, 1) if extra.ndim == 1 else extra
        a_va = np.column_stack((base_va, extra[val_mask]))
        if Q is not None:
            try:
                q1, r1 = sla.qr_insert(Q, R, extra[train_mask], Q.shape[1], which="col")
                if _well_conditioned(r1):
                    coef = sla.solve_triangular(r1, q1.T @ y_tr)
                    return 1.0 - float(np.sum((yv - a_va @ coef) ** 2)) / ss
            except Exception as e:
                logger.debug("QR-insert regression probe failed (%s: %s); scoring this candidate with lstsq", type(e).__name__, e)
        # A candidate collinear with the base (or a rank-deficient base) lands here.
        return _lstsq_r2(np.column_stack((base_tr, extra[train_mask])), a_va)

    return r2
