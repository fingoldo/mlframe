"""Held-out R^2 scorer for the raw-feature floor-drop protection (``_group2``)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

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

    # Integer row indices, computed once: ``take`` with them is several times cheaper than boolean-mask indexing of a full-length column, and every call below slices one.
    tr_idx = np.flatnonzero(train_mask)
    va_idx = np.flatnonzero(val_mask)
    yv = y[va_idx]
    ss = float(np.sum((yv - yv.mean()) ** 2))
    y_tr = y[tr_idx]
    if isinstance(base_mat, np.ndarray):
        base_tr = base_mat[tr_idx]
        base_va = base_mat[va_idx]
    else:
        base_tr = np.column_stack([np.asarray(c)[tr_idx] for c in base_mat])
        base_va = np.column_stack([np.asarray(c)[va_idx] for c in base_mat])

    def _lstsq_r2(a_tr: np.ndarray, a_va: np.ndarray) -> float:
        """Held-out R^2 from the rank-revealing least-squares solution."""
        coef = np.linalg.lstsq(a_tr, y_tr, rcond=None)[0]
        return 1.0 - float(np.sum((yv - a_va @ coef) ** 2)) / ss

    Q: Any = None
    R: Any = None
    coef_base: Any = None
    qty: Any = None
    try:
        Q, R = sla.qr(base_tr, mode="economic")
        if _well_conditioned(R):
            qty = Q.T @ y_tr
            coef_base = sla.solve_triangular(R, qty)
        else:
            Q = R = None
    except Exception as exc:
        logger.debug("mrmr: QR of the raw-protection base design failed; scoring with lstsq: %r", exc, exc_info=True)
        Q = R = None

    def _extended_coef(e_tr: np.ndarray):
        """Coefficients ``[beta_base; beta_extra]`` of ``y ~ [base | e]`` from the base QR without copying ``Q``: ``E = Q r + E_perp``, ``E_perp = Q2 R2``, so
        ``R2 beta_e = Q2' y`` and ``R beta_b = Q' y - r beta_e``. ``None`` when the extended factor is ill conditioned (a column collinear with the base)."""
        r_blk = Q.T @ e_tr
        e_perp = e_tr - Q @ r_blk
        q2, r2_blk = sla.qr(e_perp, mode="economic")
        diag = np.concatenate((np.abs(np.diag(R)), np.abs(np.diag(r2_blk))))
        if not (np.all(np.isfinite(diag)) and float(diag.min()) > _RCOND * float(diag.max())):
            return None
        beta_e = sla.solve_triangular(r2_blk, q2.T @ y_tr)
        beta_b = sla.solve_triangular(R, qty - r_blk @ beta_e)
        return np.concatenate((beta_b, beta_e))

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
        extra_tr = extra[tr_idx]
        extra_va = extra[va_idx]
        if Q is not None:
            try:
                coef = _extended_coef(extra_tr)
                if coef is not None:
                    p = base_va.shape[1]
                    pred = base_va @ coef[:p] + extra_va @ coef[p:]
                    return 1.0 - float(np.sum((yv - pred) ** 2)) / ss
            except Exception as e:
                logger.debug("QR-update regression probe failed (%s: %s); scoring this candidate with lstsq", type(e).__name__, e)
        # A candidate collinear with the base (or a rank-deficient base) lands here.
        return _lstsq_r2(np.column_stack((base_tr, extra_tr)), np.column_stack((base_va, extra_va)))

    return r2
