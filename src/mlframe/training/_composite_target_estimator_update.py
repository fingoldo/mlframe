"""Streaming-buffer update / inspect methods for ``CompositeTargetEstimator``.

Carved out of ``_composite_target_estimator.py`` to keep the parent below the 1k-line monolith threshold. Functions here become bound methods on ``CompositeTargetEstimator`` at the parent's bottom via direct class-attribute assignment. Behavioural identity is preserved bit-for-bit.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
        raise RuntimeError(
            "CompositeTargetEstimator.update: online_refit_enabled is False. Set it to True in __init__ to enable streaming refit."
        )
    if self.transform_name not in ("linear_residual",):
        raise NotImplementedError(
            f"streaming alpha refit only supported for 'linear_residual'; got transform_name={self.transform_name!r}. Other transforms have transform-specific params (eps for ratio, mad_eff for logratio, per-bin medians for quantile_residual, etc.) that don't fit the closed-form alpha/beta refit pattern."
        )
    if not hasattr(self, "fitted_params_"):
        raise RuntimeError(
            "CompositeTargetEstimator.update called before fit (no fitted_params_ to refit)."
        )
    # Lazy-init the rolling buffers on first update.
    if not hasattr(self, "_buffer_y_"):
        self._buffer_y_ = deque(maxlen=int(self.online_refit_buffer_n))
        self._buffer_base_ = deque(maxlen=int(self.online_refit_buffer_n))
    y_arr = np.asarray(y_recent, dtype=np.float64).reshape(-1)
    base_arr = np.asarray(base_recent, dtype=np.float64).reshape(-1)
    if y_arr.size != base_arr.size:
        raise ValueError(
            f"CompositeTargetEstimator.update: y_recent ({y_arr.size} rows) and base_recent ({base_arr.size} rows) must have equal length."
        )
    self._buffer_y_.extend(y_arr.tolist())
    self._buffer_base_.extend(base_arr.tolist())
    buffer_n = len(self._buffer_y_)
    # Lazy import to break the composite_estimator <-> composite_streaming
    # cycle (composite_streaming lazy-imports _linear_residual_fit from
    # composite, which re-exports CompositeTargetEstimator from us).
    from .composite_streaming import streaming_alpha_check_and_refit
    # Run drift check; the helper handles the buffer-too-small case.
    new_alpha, new_beta, info = streaming_alpha_check_and_refit(
        np.asarray(self._buffer_y_, dtype=np.float64),
        np.asarray(self._buffer_base_, dtype=np.float64),
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
    """Diagnostic: returns the current rolling-buffer state without exposing the deque internals to callers.

    Useful for monitoring / unit tests. Returns ``{"buffer_n": int, "buffer_full": bool, "alpha_current": float, "beta_current": float}``.
    """
    buf_n = len(getattr(self, "_buffer_y_", []))
    return {
        "buffer_n": buf_n,
        "buffer_full": buf_n >= int(self.online_refit_buffer_n),
        "alpha_current": float(getattr(self, "fitted_params_", {}).get("alpha", float("nan"))),
        "beta_current": float(getattr(self, "fitted_params_", {}).get("beta", float("nan"))),
    }
