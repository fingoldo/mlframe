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


def _base_block(base_mat: "np.ndarray | Sequence[np.ndarray]", idx: np.ndarray) -> np.ndarray:
    """The rows ``idx`` of the base design in double precision, column-major: a sequence of columns is gathered one column at a time into the finished array, so neither a list of per-column copies nor a
    stack of them exists (at 1000000 rows and 245 columns those were two more copies of a 1 GB block), and LAPACK takes the block without transposing it."""
    # Intentional float64: the least-squares probe design: a float32 design rounds the rank test (about 1e-7 noise against _RCOND), so a collinear column looks well conditioned.
    if isinstance(base_mat, np.ndarray):
        # Intentional float64: the least-squares probe design: a float32 design rounds the rank test (about 1e-7 noise against _RCOND), so a collinear column looks well conditioned.
        return np.asfortranarray(base_mat[idx], dtype=np.float64)
    # Intentional float64: the least-squares probe design: a float32 design rounds the rank test (about 1e-7 noise against _RCOND), so a collinear column looks well conditioned.
    out = np.empty((idx.shape[0], len(base_mat)), dtype=np.float64, order="F")
    for j, col in enumerate(base_mat):
        out[:, j] = np.asarray(col)[idx]
    return out


def _base_qr(base_tr: np.ndarray, y_tr: np.ndarray, *, consume: bool = False) -> "tuple[Any, Any, Any, Any]":
    """``(Q, R, coef_base, qty)`` of the economic QR of the base design when it is well conditioned, else ``(None, None, None, None)`` (the lstsq path then decides).

    ``consume``: the factorisation may overwrite ``base_tr`` (a column-major block in double precision), which saves a second copy of it; the caller must not read it afterwards."""
    import scipy.linalg as sla

    Q: Any = None
    R: Any = None
    coef_base: Any = None
    qty: Any = None
    try:
        Q, R = sla.qr(base_tr, mode="economic", overwrite_a=consume, check_finite=False)
        if _well_conditioned(R):
            qty = Q.T @ y_tr
            coef_base = sla.solve_triangular(R, qty)
        else:
            Q = R = None
    except Exception as exc:
        logger.warning("mrmr: QR of the raw-protection base design failed; scoring with lstsq: %r", exc, exc_info=True)
        Q = R = None
    return Q, R, coef_base, qty


def _extended_coef(Q: Any, R: Any, qty: Any, y_tr: np.ndarray, e_tr: np.ndarray) -> "np.ndarray | None":
    """Coefficients ``[beta_base; beta_extra]`` of ``y ~ [base | e]`` from the base QR without copying ``Q``: ``E = Q r + E_perp``, ``E_perp = Q2 R2``, so
    ``R2 beta_e = Q2' y`` and ``R beta_b = Q' y - r beta_e``. ``None`` when the extended factor is ill conditioned (a column collinear with the base)."""
    import scipy.linalg as sla

    r_blk = Q.T @ e_tr
    e_perp = e_tr - Q @ r_blk
    q2, r2_blk = sla.qr(e_perp, mode="economic")
    diag = np.concatenate((np.abs(np.diag(R)), np.abs(np.diag(r2_blk))))
    if not (np.all(np.isfinite(diag)) and float(diag.min()) > _RCOND * float(diag.max())):
        return None
    beta_e = sla.solve_triangular(r2_blk, q2.T @ y_tr)
    beta_b = sla.solve_triangular(R, qty - r_blk @ beta_e)
    return np.concatenate((beta_b, beta_e))


class _TrainBlock:
    """The train rows of the base design for the paths that need them after the factorisation consumed the first copy (rank deficient base, collinear candidate): built again on the first request."""

    def __init__(self, base_mat: "np.ndarray | Sequence[np.ndarray]", idx: np.ndarray) -> None:
        self._base_mat = base_mat
        self._idx = idx
        self._block: "np.ndarray | None" = None

    def get(self) -> np.ndarray:
        """The block, built on the first call."""
        if self._block is None:
            self._block = _base_block(self._base_mat, self._idx)
        return self._block


def heldout_r2_scorer(base_mat: np.ndarray | Sequence[np.ndarray], y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray) -> Callable[..., float]:
    """Return ``r2(extra=None)``: the held-out R^2 of a least-squares fit of ``y`` on ``[base | extra]``, trained on ``train_mask`` rows.

    ``base_mat`` is an ``(n, p)`` array or a sequence of ``p`` length-n columns; a sequence is sliced per column, so the full-height design
    never exists (only its train and validation blocks). ``extra`` is one full-length candidate column or an ``(n, m)`` block of columns that only mean something together (the sin/cos legs of one frequency), or ``None`` for the base design alone. The base is QR-factorised once and extended by one
    column per call, an O(n*p) update instead of a fresh solve. That is exact only while the factor is full rank: an unpivoted QR of a
    rank-deficient design (a raw column and its monotone twin, a count and its frequency encoding - both routinely selected together) has
    a near-zero diagonal, ``solve_triangular`` then returns huge coefficients without raising, and the R^2 becomes noise. So the fast path is
    taken only while the factor is well conditioned; otherwise the rank-revealing, minimum-norm ``lstsq`` decides.
    """
    # Integer row indices, computed once: ``take`` with them is several times cheaper than boolean-mask indexing of a full-length column, and every call below slices one.
    tr_idx = np.flatnonzero(train_mask)
    va_idx = np.flatnonzero(val_mask)
    yv = y[va_idx]
    ss = float(np.sum((yv - yv.mean()) ** 2))
    y_tr = y[tr_idx]
    base_tr = _base_block(base_mat, tr_idx)
    base_va = _base_block(base_mat, va_idx)
    # The factorisation overwrites the train block: it is rebuilt (``train_again``) only by the rank-deficient and fallback paths, which are rare; the common path keeps Q, R and the validation block.
    train_again = _TrainBlock(base_mat, tr_idx)

    def _lstsq_pred(a_tr: np.ndarray, a_va: np.ndarray) -> np.ndarray:
        """Validation predictions of the rank-revealing least-squares solution."""
        coef = np.linalg.lstsq(a_tr, y_tr, rcond=None)[0]
        pred: np.ndarray = a_va @ coef
        return pred

    Q, R, coef_base, qty = _base_qr(base_tr, y_tr, consume=True)
    del base_tr

    def _pred(extra):
        """Validation predictions of ``[base | extra]`` (``extra`` one column, an ``(n, m)`` block, or ``None`` for the base alone)."""
        if extra is None:
            if coef_base is not None:
                return base_va @ coef_base
            return _lstsq_pred(train_again.get(), base_va)
        extra = np.asarray(extra, dtype=np.float64)  # Intentional float64: rank-deficiency and exact-copy fallbacks need f64 lstsq precision
        extra = extra.reshape(-1, 1) if extra.ndim == 1 else extra
        extra_tr = extra[tr_idx]
        extra_va = extra[va_idx]
        if Q is not None:
            try:
                coef = _extended_coef(Q, R, qty, y_tr, extra_tr)
                if coef is not None:
                    p = base_va.shape[1]
                    return base_va @ coef[:p] + extra_va @ coef[p:]
            except Exception as e:
                logger.debug("QR-update regression probe failed (%s: %s); scoring this candidate with lstsq", type(e).__name__, e)
        # A candidate collinear with the base (or a rank-deficient base) lands here.
        return _lstsq_pred(np.column_stack((train_again.get(), extra_tr)), np.column_stack((base_va, extra_va)))

    def r2(extra=None):
        """Held-out R^2 of ``[base | extra]``, where ``extra`` is one column or an ``(n, m)`` block of columns added together."""
        if ss < 1e-24:
            return 0.0
        return 1.0 - float(np.sum((yv - _pred(extra)) ** 2)) / ss

    def paired_gain(extra) -> tuple:
        """``(gain, se)``: the held-out R^2 gain of adding ``extra`` to the base and its standard error over the validation rows (paired per-row squared-error reduction)."""
        if ss < 1e-24:
            return 0.0, 0.0
        d = (yv - _pred(None)) ** 2 - (yv - _pred(extra)) ** 2
        return float(d.sum()) / ss, float(d.std(ddof=1)) * float(np.sqrt(d.size)) / ss if d.size > 1 else 0.0

    r2.paired_gain = paired_gain  # type: ignore[attr-defined]
    return r2
