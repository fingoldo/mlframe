"""Base-column target-leakage detection for composite-target discovery (train-only).

A composite *base* that is a near-deterministic function of the CURRENT target ``y``
-- ``base == y``, ``base == y + tiny_noise``, ``base == monotone(y)``, or ``base == y``
at the SAME time step (not a genuine lag) -- yields a trivially-perfect-but-useless
composite that collapses in production (the base simply *is* the answer). Such a base
must be rejected before discovery spends compute on it.

:func:`detect_base_target_leakage` screens one (y, base) pair with cheap signals:

* **Near-perfect rank correlation** ``|spearman(y, base)| ~ 1.0`` -- base orders the
  rows exactly like the target (catches any monotone re-encoding, not just linear).
* **Near-zero residual after a monotone fit** -- the base explains essentially ALL of
  the target's variance once a monotone (rank-isotonic) relation is allowed, so what
  is left over is noise, not a real lagged signal component.

These two together are leakage. A *legitimate strong lag* (e.g. yesterday's value of
the same series) is correlated and may even rank-correlate highly, but it is **shifted
in time** -- its near-identity is with ``y`` at a DIFFERENT time index, and against the
current ``y`` it leaves a non-trivial residual. The optional ``time_ordering`` check
distinguishes same-time near-identity (leakage) from a true shift (legitimate): if the
base aligns far better with a time-shifted ``y`` than with the current ``y``, it is a
lag, not leakage.

Returns ``{"is_leaky", "score", "reason"}``. :func:`screen_base_pool` batch-screens a
candidate-base pool and returns the per-candidate verdicts plus the leak-free subset.

Train-only: this runs on the discovery training rows; no held-out / calibration rows
are touched. Cheap O(n log n) per pair (one argsort-based rank + a sort for the
monotone residual); negligible vs the model fits discovery would otherwise spend on a
leaky base.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Default decision thresholds. A base is leaky when its absolute Spearman vs the current
# target is at/above _LEAK_SPEARMAN AND the monotone-fit residual fraction is at/below
# _LEAK_RESIDUAL (base explains essentially all of y). Both must hold: high rank-corr
# alone is a merely-strong feature; only high rank-corr WITH a vanishing residual is
# the trivially-perfect "base IS the target" pattern.
_LEAK_SPEARMAN: float = 0.999
_LEAK_RESIDUAL: float = 1e-3

# A time-shift is judged a genuine lag (NOT leakage) when shifting y by some small lag
# improves the alignment markedly over the same-time alignment: the same-time residual
# fraction must exceed the best-shifted residual fraction by at least this margin. A
# true lag-1 base hugs a shifted y (tiny shifted residual) yet leaves a real same-time
# residual; a leaky same-time base is already perfect at shift 0, so no shift helps.
_LAG_RESIDUAL_MARGIN: float = 0.05
_MAX_LAG_PROBE: int = 5

_MIN_ROWS: int = 8


def _finite_pair(y: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows where BOTH y and base are finite, as float64 1-D arrays."""
    y = np.asarray(y, dtype=np.float64).ravel()
    base = np.asarray(base, dtype=np.float64).ravel()
    if y.shape[0] != base.shape[0]:
        raise ValueError(f"y and base length mismatch: {y.shape[0]} vs {base.shape[0]}")
    mask = np.isfinite(y) & np.isfinite(base)
    return y[mask], base[mask]


def _rank(a: np.ndarray) -> np.ndarray:
    """Average (tie-corrected) ranks of ``a`` -- the basis for Spearman correlation."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    sa = a[order]
    n = a.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)  # average rank over the tie block
        i = j
    return ranks


def _abs_spearman(y: np.ndarray, base: np.ndarray) -> float:
    """``|spearman(y, base)|`` via Pearson on ranks; 0.0 if either side is constant."""
    ry = _rank(y)
    rb = _rank(base)
    ry -= ry.mean()
    rb -= rb.mean()
    vy = float(np.dot(ry, ry))
    vb = float(np.dot(rb, rb))
    if vy <= 0.0 or vb <= 0.0:
        return 0.0
    return float(abs(float(np.dot(ry, rb)) / np.sqrt(vy * vb)))


def _monotone_residual_frac(y: np.ndarray, base: np.ndarray) -> float:
    """Residual variance fraction of y after the best MONOTONE map of base -> y.

    Sorts y by base order and reads off the rank-aligned prediction (a non-parametric
    monotone fit -- isotonic in spirit, using the rank-sorted target as the fitted
    surface). Returns ``var(y - yhat) / var(y)`` in ``[0, 1]``: ~0 means base explains
    essentially ALL of y's order (a re-encoding of the target); ~1 means it explains
    nothing. Captures any monotone relation (linear, log, squared-on-positives, ...),
    not just linear, so ``base == y**3`` reads as fully explanatory.
    """
    n = y.shape[0]
    if n < 2:
        return 1.0
    var_y = float(np.var(y))
    if var_y <= 0.0:
        return 0.0  # constant y is trivially explained
    # Order rows by base; the monotone prediction for each row is the SORTED-by-base
    # target value at the same rank position (best monotone-increasing surface). For a
    # decreasing relation we also try the reversed sort and take the better fit.
    order = np.argsort(base, kind="mergesort")
    y_sorted = y[order]
    pred_inc = np.empty(n, dtype=np.float64)
    pred_inc[order] = _isotonic_pava(y_sorted)
    pred_dec = np.empty(n, dtype=np.float64)
    pred_dec[order] = _isotonic_pava(y_sorted[::-1])[::-1]
    res_inc = float(np.mean((y - pred_inc) ** 2))
    res_dec = float(np.mean((y - pred_dec) ** 2))
    return min(res_inc, res_dec) / var_y


def _isotonic_pava(y_sorted: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators isotonic (non-decreasing) fit of an already-ordered y.

    Standard O(n) PAVA: merge adjacent blocks that violate monotonicity, weighting by
    block size. Gives the least-squares monotone-increasing surface, so a perfectly
    monotone base yields a residual of exactly 0.
    """
    n = y_sorted.shape[0]
    vals = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    starts = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        vals[k] = y_sorted[i]
        weights[k] = 1.0
        starts[k] = i
        k += 1
        while k > 1 and vals[k - 1] < vals[k - 2]:
            w = weights[k - 1] + weights[k - 2]
            vals[k - 2] = (vals[k - 1] * weights[k - 1] + vals[k - 2] * weights[k - 2]) / w
            weights[k - 2] = w
            k -= 1
    out = np.empty(n, dtype=np.float64)
    for b in range(k):
        end = starts[b + 1] if b + 1 < k else n
        out[starts[b] : end] = vals[b]
    return out


def _looks_like_lag(y: np.ndarray, base: np.ndarray, *, max_lag: int) -> bool:
    """True when base aligns markedly better with a TIME-SHIFTED y than with current y.

    Probes small forward shifts of y (base[i] predicting y[i-k]); if the best shifted
    monotone residual undercuts the same-time residual by more than _LAG_RESIDUAL_MARGIN,
    the base's near-identity is with a PAST/FUTURE y -- a genuine lag, not same-time
    leakage. Requires ``time_ordering`` to be meaningful (rows already time-sorted).
    """
    n = y.shape[0]
    same_time = _monotone_residual_frac(y, base)
    best_shift = same_time
    for k in range(1, min(max_lag, n - _MIN_ROWS) + 1):
        # base[k:] is contemporaneous with y[:-k] under a forward shift of k steps.
        shifted = _monotone_residual_frac(y[:-k], base[k:])
        if shifted < best_shift:
            best_shift = shifted
    return (same_time - best_shift) >= _LAG_RESIDUAL_MARGIN


def detect_base_target_leakage(
    y: np.ndarray,
    base: np.ndarray,
    *,
    time_ordering: np.ndarray | Sequence | None = None,
    spearman_threshold: float = _LEAK_SPEARMAN,
    residual_threshold: float = _LEAK_RESIDUAL,
    max_lag_probe: int = _MAX_LAG_PROBE,
) -> dict[str, Any]:
    """Detect whether ``base`` is a near-deterministic function of the CURRENT ``y``.

    Parameters
    ----------
    y, base : 1-D arrays of equal length (train rows only). Non-finite rows in either
        are dropped pairwise before scoring.
    time_ordering : optional 1-D array giving the row time order. When provided, rows
        are sorted by it and a small lag-probe distinguishes a genuine time-shifted lag
        (NOT leaky) from a same-time near-identity (leaky). Pass the timestamp / index
        column the discovery already knows; omit for cross-sectional (non-temporal) data.
    spearman_threshold, residual_threshold : decision cutoffs (see module docstring).
    max_lag_probe : number of forward shifts to probe in the lag check.

    Returns
    -------
    dict with:
        ``is_leaky`` : bool -- True when the base is a trivial re-encoding of y.
        ``score``    : float in [0, 1] -- leakage confidence; ``|spearman| * (1 - resid)``
                       (high only when both rank-corr is near 1 AND residual near 0).
        ``reason``   : short human-readable explanation.
    """
    yf, bf = _finite_pair(y, base)
    n = yf.shape[0]
    if n < _MIN_ROWS:
        return {"is_leaky": False, "score": 0.0, "reason": f"too few finite rows ({n})"}

    if time_ordering is not None:
        to = np.asarray(time_ordering).ravel()
        if to.shape[0] == np.asarray(y).ravel().shape[0]:
            # Re-derive the finite mask on the ORIGINAL arrays so time_ordering aligns.
            y0 = np.asarray(y, dtype=np.float64).ravel()
            b0 = np.asarray(base, dtype=np.float64).ravel()
            m = np.isfinite(y0) & np.isfinite(b0)
            order = np.argsort(to[m], kind="mergesort")
            yf = y0[m][order]
            bf = b0[m][order]

    sp = _abs_spearman(yf, bf)
    resid = _monotone_residual_frac(yf, bf)
    score = sp * max(0.0, 1.0 - resid)

    if sp < spearman_threshold or resid > residual_threshold:
        return {
            "is_leaky": False,
            "score": score,
            "reason": f"not near-identity (|spearman|={sp:.4f}, residual={resid:.4g})",
        }

    # High rank-corr AND vanishing residual: same-time near-identity UNLESS the optional
    # time check shows the near-identity is with a SHIFTED y (a real lag).
    if time_ordering is not None and _looks_like_lag(yf, bf, max_lag=max_lag_probe):
        return {
            "is_leaky": False,
            "score": score,
            "reason": f"genuine time-shifted lag (|spearman|={sp:.4f}, residual={resid:.4g})",
        }

    return {
        "is_leaky": True,
        "score": score,
        "reason": f"near-identity to current target (|spearman|={sp:.4f}, residual={resid:.4g})",
    }


def screen_base_pool(
    y: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    *,
    time_ordering: np.ndarray | Sequence | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Screen a pool of candidate bases for target leakage against ``y`` (train-only).

    Parameters
    ----------
    y : 1-D target array (train rows).
    candidates : mapping ``name -> base_array`` of candidate bases to screen.
    time_ordering, **kwargs : forwarded to :func:`detect_base_target_leakage`.

    Returns
    -------
    dict with:
        ``verdicts`` : ``{name: detect_base_target_leakage(...) result}`` for every name.
        ``leaky``    : list of names flagged leaky.
        ``safe``     : list of names NOT flagged leaky (the leak-free pool to keep).
    """
    verdicts: dict[str, dict[str, Any]] = {}
    leaky: list[str] = []
    safe: list[str] = []
    for name, base in candidates.items():
        v = detect_base_target_leakage(y, base, time_ordering=time_ordering, **kwargs)
        verdicts[name] = v
        (leaky if v["is_leaky"] else safe).append(name)
    return {"verdicts": verdicts, "leaky": leaky, "safe": safe}
