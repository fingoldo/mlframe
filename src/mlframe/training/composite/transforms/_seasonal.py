"""``seasonal_residual`` composite transform.

``T = y - seasonal_mean(phase)`` with ``phase = row_index % period``: the fit learns per-phase means of the train target and the forward subtracts each
row's phase mean; the inverse adds it back. ``period`` is either supplied via a fit kwarg or selected on train by held-out error over alternating
cycles among period 1 (no seasonality) and a small candidate grid (``_SEASONAL_PERIOD_CANDIDATES`` capped at ``n // 3`` so every candidate sees at
least ~3 full cycles), taking the smallest period within one standard error of the best.

Index-position phase assumption. Like ``ewma_residual``, the transform is defined on the ROW SEQUENCE: phase is the row's position modulo ``period``
within the batch it is evaluated on, NOT a calendar field. The caller is responsible for chronological, gap-free row order at fit and predict. A
predict batch starts at phase 0 by default (the same stateless-batch convention the EWMA anchor uses); under recurrence continuation the inverse
starts at the phase that follows the train series, and explicit absolute ``row_index`` positions override both. Given the phase, the transform IS pointwise
(no neighbour reads) -- but it is still registered ``recurrent=True``, because that flag's real purpose in ``CompositeTargetEstimator.fit()`` is not
"needs neighbour values", it's "must see the FULL, uncompacted row sequence to compute correct per-row state": this transform's per-row phase is an
ABSOLUTE POSITION in that sequence, and the wrapper's ordinary (non-recurrent) path compacts away domain-violating rows BEFORE calling forward(),
which silently shifts every later row's array position -- and hence its phase -- whenever a row is dropped. That shift is introduced by the wrapper's
OWN internal domain-filter compaction, not by anything visible in the caller's own frame (the caller passed a full, gap-free series), which found this exact position-vs-neighbour distinction get missed by an earlier version
of this comment.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_SEASONAL_PERIOD_CANDIDATES: tuple[int, ...] = (4, 5, 7, 12, 24, 52)
"""Candidate seasonal periods for the fit-time grid search: intra-month week (4/5), week of daily data (7), months (12), hours (24), weeks of year (52)."""


def _seasonal_phase_means(y_f: np.ndarray, period: int, row_index: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Per-phase means of ``y_f`` (NaN-aware) + the residual variance after subtracting them. Empty phases fall back to the global mean.

    ``row_index``: optional absolute row-position array (same length as ``y_f``), used instead of
    ``arange(len(y_f))`` when the phase must reflect the row's TRUE position in some larger original
    sequence -- see ``_seasonal_residual_fit``'s ``row_index`` parameter.
    """
    n = y_f.size
    idx = row_index if row_index is not None else np.arange(n, dtype=np.int64)
    phase = idx % period
    finite = np.isfinite(y_f)
    global_mean = float(np.mean(y_f[finite])) if finite.any() else 0.0
    sums = np.bincount(phase[finite], weights=y_f[finite], minlength=period)
    counts = np.bincount(phase[finite], minlength=period)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)
    resid = y_f[finite] - means[phase[finite]]
    var = float(np.var(resid)) if resid.size else float("inf")
    return np.asarray(means, dtype=np.float64), var


def _seasonal_cv_score(y_f: np.ndarray, period: int, idx: np.ndarray) -> tuple[float, float]:
    """Held-out MSE of per-phase means for ``period``: 2-fold CV over alternating whole cycles (means fit on even cycles score the odd ones and
    vice versa). Returns (mean held-out squared error, its standard error). In-sample variance can only fall as the period grows (a longer period
    has more phase means and 24/52 nest 4/12), so it picked the largest candidate on pure noise and nested multiples over the true period."""
    finite = np.isfinite(y_f)
    yv = y_f[finite]
    ix = idx[finite]
    phase = ix % period
    fold = (ix // period) % 2
    sq = np.empty(yv.size, dtype=np.float64)
    for f in (0, 1):
        tr = fold != f
        te = ~tr
        if not te.any():
            continue
        gm = float(yv[tr].mean()) if tr.any() else float(yv.mean())
        sums = np.bincount(phase[tr], weights=yv[tr], minlength=period)
        counts = np.bincount(phase[tr], minlength=period)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), gm)
        sq[te] = (yv[te] - means[phase[te]]) ** 2
    if sq.size == 0:
        return float("inf"), 0.0
    return float(sq.mean()), float(sq.std() / np.sqrt(sq.size))


def _seasonal_residual_fit(
    y: np.ndarray, base: np.ndarray | None,
    period: int | None = None,
    _finite_mask: np.ndarray | None = None,
    row_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Learn per-phase means. ``period`` may be supplied explicitly (like other transforms accept ``k`` / ``d`` fit kwargs); otherwise it is chosen
    from period 1 (no seasonality, a global de-mean) plus ``_SEASONAL_PERIOD_CANDIDATES`` (capped at n//3) by held-out error over alternating
    cycles, taking the SMALLEST period whose held-out error is within one standard error of the best (fewest phase means that fit as well).

    ``row_index``: optional absolute-position array (same length as ``y``), signature-gated in from
    ``CompositeTargetEstimator.fit()`` (mirroring its ``sample_weight`` gating). ``fit()`` is always called
    on domain-filter-COMPACTED ``y``/``base`` regardless of the ``recurrent`` flag (unlike ``forward()``,
    which the recurrent branch routes through the FULL sequence) -- so without ``row_index``, phase would be
    computed from COMPACTED-array position, silently misaligned whenever any row was dropped between fit's
    input and the row's true position.
    ``None`` (the default, e.g. for a direct/standalone call) falls back to ``arange(len(y))``, unchanged.

    ``n_seen`` (the length of the train row sequence) is stored so a continuation batch can start at its true phase.
    """
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y_f.size
    row_idx = np.asarray(row_index, dtype=np.int64).reshape(-1) if row_index is not None else None
    idx = row_idx if row_idx is not None else np.arange(n, dtype=np.int64)
    n_seen = int(idx.max()) + 1 if idx.size else 0
    if period is not None:
        period = max(1, int(period))
        means, _ = _seasonal_phase_means(y_f, period, row_idx)
        return {"period": period, "phase_means": means, "y_train_mean": float(means.mean()), "n_seen": n_seen}
    candidates = [1] + [p for p in _SEASONAL_PERIOD_CANDIDATES if p <= max(n // 3, 1)]
    scores = {p: _seasonal_cv_score(y_f, p, idx) for p in candidates}
    best = min(candidates, key=lambda p: scores[p][0])
    threshold = scores[best][0] + scores[best][1]
    best_period = min(p for p in candidates if scores[p][0] <= threshold)
    means, _ = _seasonal_phase_means(y_f, best_period, row_idx)
    return {"period": int(best_period), "phase_means": means, "y_train_mean": float(means.mean()), "n_seen": n_seen}


def _seasonal_phase(n_rows: int, params: dict[str, Any], row_index: np.ndarray | None, continuation: bool) -> np.ndarray:
    """Phase of each batch row: ``row_index % period`` when absolute positions are supplied (same coordinates as the train sequence, whose first
    row is 0); else the batch position, offset by the train length under recurrence continuation (inverse only) so a batch that continues the
    train series starts at its true phase instead of phase 0."""
    period = int(params["period"])
    if row_index is not None:
        return np.asarray(row_index, dtype=np.int64).reshape(-1) % period
    offset = int(params.get("n_seen", 0)) if continuation and params.get("recurrence_continuation") else 0
    return (np.arange(n_rows, dtype=np.int64) + offset) % period


def _seasonal_residual_forward(
    y: np.ndarray, base: np.ndarray | None, params: dict[str, Any],
    row_index: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``T = y - phase_means[phase]`` (phase 0 = first row of the batch unless absolute ``row_index`` positions are given)."""
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    means = np.asarray(params["phase_means"], dtype=np.float64)
    return np.asarray(y_f - means[_seasonal_phase(y_f.size, params, row_index, continuation=False)])


def _seasonal_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray | None, params: dict[str, Any],
    row_index: np.ndarray | None = None,
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + phase_means[phase]``; the phase follows the train series under recurrence continuation."""
    t_f = np.asarray(t_hat, dtype=np.float64).reshape(-1)
    means = np.asarray(params["phase_means"], dtype=np.float64)
    return np.asarray(t_f + means[_seasonal_phase(t_f.size, params, row_index, continuation=True)])


def _seasonal_residual_domain(
    y: np.ndarray | None, base: np.ndarray | None,
) -> np.ndarray:
    """Unary y-only domain: finite ``y`` at fit time; all-True (sized off whichever array is present) at predict time."""
    if y is None:
        n = len(base) if base is not None and hasattr(base, "__len__") else 1
        return np.ones(n, dtype=bool)
    return np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))
