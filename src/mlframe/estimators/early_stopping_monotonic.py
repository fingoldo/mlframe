"""Shared, dependency-free monotonic strict-decline overfitting stopper.

``MonotonicDeclineStopper`` implements a *confident overfitting* early-stop rule that is
COMPLEMENTARY to patience-based early stopping. It tracks a validation ML metric per
iteration and fires once the metric has STRICTLY WORSENED for ``patience`` consecutive
iterations AFTER the global optimum has been reached (a monotone decline run of fixed
length ``patience``).

Semantics (the exact contract every backend reuses), greater-is-better (``mode="max"``)::

    if current > best:   best, streak = current, 0   # new global best resets
    elif current < prev: streak += 1                  # strict decline since prev
    else:                streak = 0                    # plateau / bounce-up resets
    if streak >= patience: STOP
    prev = current

For ``mode="min"`` the comparisons are mirrored (``current < best`` is a new best,
``current > prev`` is a strict decline). A NEW global best, a PLATEAU (equal to the
previous value), or a BOUNCE-UP (better than the previous value, even if not a new
global best) all reset the streak to 0 -- only an uninterrupted run of strictly-worse
steps accumulates. This makes it a high-confidence overfitting signal: the curve has to
be monotonically bending the wrong way, not merely failing to set new records.

Why a separate detector from patience
--------------------------------------
Standard patience stops on "no NEW best for N iters" -- it tolerates a long flat/noisy
tail that never improves but also never strictly declines. This detector stops on "N
strict declines in a row since the best" -- it reacts to a curve that is actively
deteriorating without waiting for the (larger) patience budget. Training stops when
EITHER fires, whichever comes first; ``best_model_`` / ``best_iteration`` still point at
the global best.

Relationship to ``UniversalCallback`` worsening detector
---------------------------------------------------------
``mlframe.training.callbacks._callbacks.UniversalCallback`` already carries a *budget-scaled*
"worsening" detector whose threshold is ``max(max_iter // coeff, min_iters)`` and which
does NOT reset on an exact plateau. This class is the fixed-N, plateau-resets variant the
backends below (EarlyStoppingWrapper / lightning / lgb / xgb) share verbatim so the rule is
identical everywhere. It is intentionally pure-Python and import-free (no numpy/torch) so
every backend can embed it without a dependency hit.

This is a trivial control-flow helper (a handful of float compares per iteration), not a
numeric kernel, so the cProfile / acceleration-ladder convention does not apply.
"""

from __future__ import annotations


class MonotonicDeclineStopper:
    """Fixed-length monotone strict-decline overfitting detector (see module docstring).

    Parameters
    ----------
    patience:
        Number of CONSECUTIVE strictly-worse steps (since the global best) that triggers a
        stop. ``None`` (or ``<= 0``) disables the detector entirely -- ``update`` always
        returns ``False``.
    mode:
        ``"max"`` for greater-is-better metrics (AUC, accuracy, negative-RMSE), ``"min"``
        for lower-is-better metrics (loss, RMSE, error). Drives the sign of every compare.
    """

    __slots__ = ("patience", "mode", "_greater_is_better", "best", "prev", "streak", "stopped")

    def __init__(self, patience: int | None, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = None if (patience is None or int(patience) <= 0) else int(patience)
        self.mode = mode
        self._greater_is_better = mode == "max"
        self.reset()

    def reset(self) -> None:
        """Clear all running state so the stopper can be reused for a fresh fit."""
        self.best: float | None = None
        self.prev: float | None = None
        self.streak: int = 0
        self.stopped: bool = False

    @property
    def enabled(self) -> bool:
        return self.patience is not None

    def update(self, value: float) -> bool:
        """Feed the next iteration's validation metric; return ``True`` once stop fires.

        Stop is latched: once it returns ``True`` it keeps returning ``True`` on later calls
        (so a backend that keeps iterating after ignoring the signal stays stopped). A
        disabled detector (patience None) always returns ``False``.
        """
        if self.patience is None:
            return False
        # Non-finite metric (NaN / inf): treat as a no-op rather than corrupting best/prev
        # or the streak -- the backend's own NaN handling decides what to do with it.
        if value != value or value in (float("inf"), float("-inf")):
            return self.stopped
        cur = float(value)

        if self.best is None:
            # First observation establishes the baseline best + prev; no decline possible yet.
            self.best = cur
            self.prev = cur
            return self.stopped

        is_new_best = (cur > self.best) if self._greater_is_better else (cur < self.best)
        if is_new_best:
            self.best = cur
            self.streak = 0
        else:
            assert self.prev is not None  # always set together with self.best above
            is_strict_decline = (cur < self.prev) if self._greater_is_better else (cur > self.prev)
            if is_strict_decline:
                self.streak += 1
            else:
                # Plateau (== prev) or bounce-up (better than prev but not a new global best).
                self.streak = 0

        self.prev = cur
        if self.streak >= self.patience:
            self.stopped = True
        return self.stopped


__all__ = ["MonotonicDeclineStopper"]
