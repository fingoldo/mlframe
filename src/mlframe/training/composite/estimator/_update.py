"""Streaming-buffer update / inspect methods for ``CompositeTargetEstimator``.

Functions here become bound methods on ``CompositeTargetEstimator`` at the parent's bottom via direct class-attribute assignment.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class _RingBuffer:
    """Fixed-capacity float64 FIFO ring buffer for the streaming-refit window.

    The previous implementation appended new rows to a ``collections.deque`` of
    boxed Python floats (``deque.extend(arr.tolist())``) and rebuilt the whole
    window into a fresh ndarray (``np.asarray(deque)``) on EVERY ``update`` call
    -- O(buffer_n) Python boxing + unboxing per call even when the cheap drift
    z-check immediately short-circuits.

    Here the storage is a single preallocated ``np.empty(capacity)`` array plus a
    head index and a live count, so an append copies only the NEW rows
    (O(new_rows), fully vectorised, no per-element boxing). The drift check reads
    a contiguous FIFO-ordered view materialised into a SECOND preallocated array
    that is reused across calls rather than reallocated each time. Oldest-first
    (FIFO) eviction at capacity matches the old ``deque(maxlen=...)`` semantics
    exactly, including the over-long-batch case (a single append larger than the
    capacity keeps only its last ``capacity`` rows).
    """

    __slots__ = ("_count", "_head", "_store", "_view", "capacity")

    def __init__(self, capacity: int) -> None:
        self.capacity = max(int(capacity), 1)
        # Storage ring + a reusable contiguous-view scratch, both float64 so the
        # drift check never has to allocate or up-cast.
        self._store = np.empty(self.capacity, dtype=np.float64)
        self._view = np.empty(self.capacity, dtype=np.float64)
        self._head = 0  # index of the oldest live row within ``_store``
        self._count = 0  # number of live rows (<= capacity)

    def __len__(self) -> int:
        return self._count

    def append(self, arr: np.ndarray) -> None:
        """Append new rows (FIFO), evicting oldest beyond ``capacity``.

        Copies only the (clipped) incoming rows into the ring -- O(new_rows),
        never O(buffer_n). An incoming batch longer than ``capacity`` keeps only
        its trailing ``capacity`` rows, matching ``deque(maxlen=capacity)``.
        """
        cap = self.capacity
        m = int(arr.size)
        if m == 0:
            return
        if m >= cap:
            # The new batch alone overflows the window: keep only its tail and
            # reset the ring so the contiguous view starts at index 0.
            self._store[:] = arr[m - cap :]
            self._head = 0
            self._count = cap
            return
        # Write ``m`` rows starting at the current tail, wrapping the ring.
        tail = (self._head + self._count) % cap
        first = min(m, cap - tail)
        self._store[tail : tail + first] = arr[:first]
        if first < m:
            self._store[: m - first] = arr[first:]
        new_count = self._count + m
        if new_count > cap:
            # Overwrote the oldest rows -- advance the head past the evicted ones.
            self._head = (self._head + (new_count - cap)) % cap
            self._count = cap
        else:
            self._count = new_count

    def contiguous(self) -> np.ndarray:
        """Return the live window in FIFO (oldest-first) order, contiguous.

        Materialises into the reused ``_view`` scratch (no per-call allocation):
        when the live window does not wrap the ring this is a single slice copy;
        when it wraps it is two slice copies. The result is a length-``count``
        view of ``_view`` -- callers must treat it as read-only / transient.
        """
        cap = self.capacity
        n = self._count
        head = self._head
        end = head + n
        if end <= cap:
            self._view[:n] = self._store[head:end]
        else:
            first = cap - head
            self._view[:first] = self._store[head:]
            self._view[first:n] = self._store[: end - cap]
        return self._view[:n]


def update(self, y_recent: Any, base_recent: Any) -> dict[str, Any]:
    """Streaming-update interface: append new (y, base) observations to a rolling buffer and run a drift check.

    Caller invokes this method on incoming production data; when the buffer fills past ``online_refit_min_buffer_n`` AND the Chow-style z-score crosses ``online_refit_z_threshold``, the wrapper's ``fitted_params_["alpha"]`` / ``["beta"]`` get updated in-place so subsequent predict() calls use the drift-corrected coefficients.

    Only supported for the ``linear_residual`` transform (the only one with closed-form alpha/beta in the fitted params; other transforms have transform-specific params that aren't suitable for streaming refit). For other transforms, raises ``NotImplementedError``.

    Parameters
    ----------
    y_recent, base_recent
        New observation arrays (1-D, equal length). Appended to the rolling buffer; oldest rows evicted (FIFO) once the buffer reaches ``online_refit_buffer_n``.

    Returns
    -------
    info: dict carrying the same fields as ``streaming_alpha_check_and_refit`` (refit / z_score / alpha_buffer / beta_buffer / reason) PLUS ``buffer_n_total`` (current buffer size after the update).
    """
    if not getattr(self, "online_refit_enabled", False):
        raise RuntimeError("CompositeTargetEstimator.update: online_refit_enabled is False. Set it to True in __init__ to enable streaming refit.")
    # linear_residual_robust shares the {alpha, beta} param shape and the
    # forward/inverse of linear_residual, so the closed-form streaming refit
    # applies to it too (the streaming refit recomputes alpha/beta by OLS on
    # the recent buffer -- the robust trim only matters at the initial fit).
    if self.transform_name not in ("linear_residual", "linear_residual_robust"):
        raise NotImplementedError(
            f"streaming alpha refit only supported for 'linear_residual'; got transform_name={self.transform_name!r}. Other transforms have transform-specific params (eps for ratio, mad_eff for logratio, per-bin medians for quantile_residual, etc.) that don't fit the closed-form alpha/beta refit pattern."
        )
    if not hasattr(self, "fitted_params_"):
        raise RuntimeError("CompositeTargetEstimator.update called before fit (no fitted_params_ to refit).")
    # Lazy-init the preallocated ring buffers on first update. The trailing
    # underscore marks runtime-only state that sklearn.clone() drops (cloned
    # estimators start with an empty buffer).
    if not hasattr(self, "_buffer_y_"):
        self._buffer_y_ = _RingBuffer(int(self.online_refit_buffer_n))
        self._buffer_base_ = _RingBuffer(int(self.online_refit_buffer_n))
    y_arr = np.asarray(y_recent, dtype=np.float64).reshape(-1)
    base_arr = np.asarray(base_recent, dtype=np.float64).reshape(-1)
    if y_arr.size != base_arr.size:
        raise ValueError(f"CompositeTargetEstimator.update: y_recent ({y_arr.size} rows) and base_recent ({base_arr.size} rows) must have equal length.")
    # Append only the new rows (O(new_rows)); the heavy whole-buffer unboxing of
    # the old deque path is gone -- the drift check reads a contiguous view.
    self._buffer_y_.append(y_arr)
    self._buffer_base_.append(base_arr)
    buffer_n = len(self._buffer_y_)
    # Lazy import to break the composite_estimator <-> composite_streaming
    # cycle (composite_streaming lazy-imports _linear_residual_fit from
    # composite, which re-exports CompositeTargetEstimator from us).
    from ..streaming import streaming_alpha_check_and_refit
    # Run drift check; the helper handles the buffer-too-small case. The
    # contiguous views are FIFO-ordered (oldest head, newest tail) so the
    # change-point scan sees the dead regime first and the live regime last.
    new_alpha, new_beta, info = streaming_alpha_check_and_refit(
        self._buffer_y_.contiguous(),
        self._buffer_base_.contiguous(),
        current_alpha=float(self.fitted_params_.get("alpha", 0.0)),
        current_beta=float(self.fitted_params_.get("beta", 0.0)),
        z_threshold=float(self.online_refit_z_threshold),
        min_buffer_n=int(self.online_refit_min_buffer_n),
    )
    info["buffer_n_total"] = buffer_n
    if info.get("refit"):
        # Update params in-place. The wrapper's predict() reads these on every call so the next predict will use the drifted alpha / beta.
        self.fitted_params_["alpha"] = new_alpha
        self.fitted_params_["beta"] = new_beta
        # Refresh the y-clip envelope + median + T-clip from the RECENT
        # (drifted) buffer. The alpha/beta-only refit left these at their
        # pre-drift train values, so a drift-corrected prediction that moved
        # into the new regime was clipped back toward the DEAD regime by the
        # stale envelope -- defeating the correction.
        try:
            from . import _y_train_clip_bounds
            from ..transforms import get_transform
            _by = np.asarray(self._buffer_y_.contiguous(), dtype=np.float64)
            _bb = np.asarray(self._buffer_base_.contiguous(), dtype=np.float64)
            _finy = np.isfinite(_by)
            if int(_finy.sum()) >= 10:
                self.fitted_params_["y_train_median"] = float(np.median(_by[_finy]))
                _lo, _hi = _y_train_clip_bounds(_by[_finy])
                self.fitted_params_["y_clip_low"] = _lo
                self.fitted_params_["y_clip_high"] = _hi
                _t = get_transform(self.transform_name).forward(_by, _bb, self.fitted_params_)
                _tf = _t[np.isfinite(_t)]
                if _tf.size >= 10:
                    _med_t = float(np.median(_tf))
                    # RAW (unscaled) MAD, matching fit() / from_fitted_inner()'s identical envelope formula
                    # in _estimator.py -- this site previously applied the extra normal-consistent *1.4826
                    # scale factor those two don't, silently widening the post-drift-refit T-clip envelope
                    # by ~48% for the same underlying spread purely as a side effect of ever calling
                    # .update() with a firing drift correction.
                    _mad_t = float(np.median(np.abs(_tf - _med_t)))
                    if _mad_t > 0:
                        self.fitted_params_["t_clip_low"] = _med_t - 10.0 * _mad_t
                        self.fitted_params_["t_clip_high"] = _med_t + 10.0 * _mad_t
        except Exception as _env_err:
            logger.warning(
                "[CompositeTargetEstimator.update] envelope refresh after drift " "refit failed (%s); kept the pre-drift clip bounds.",
                _env_err,
            )
        logger.info(
            "[CompositeTargetEstimator.update] streaming refit fired (z=%.2f > %.2f). alpha %.4f -> %.4f, beta %.4f -> %.4f. buffer_n=%d",
            info["z_score"], self.online_refit_z_threshold,
            info["alpha_buffer"] if info["alpha_buffer"] is not None else float("nan"),
            new_alpha,
            info["beta_buffer"] if info["beta_buffer"] is not None else float("nan"),
            new_beta, buffer_n,
        )
    return info


def get_buffer_state(self) -> dict[str, Any]:
    """Diagnostic: returns the current rolling-buffer state without exposing the ring-buffer internals to callers.

    Useful for monitoring / unit tests. Returns ``{"buffer_n": int, "buffer_full": bool, "alpha_current": float, "beta_current": float}``.
    """
    buf_n = len(getattr(self, "_buffer_y_", []))
    return {
        "buffer_n": buf_n,
        "buffer_full": buf_n >= int(self.online_refit_buffer_n),
        "alpha_current": float(getattr(self, "fitted_params_", {}).get("alpha", float("nan"))),
        "beta_current": float(getattr(self, "fitted_params_", {}).get("beta", float("nan"))),
    }
