"""Streaming alpha check/refit: Chow-style drift detector + refit for linear_residual coefficients. Lazy-imports ``_linear_residual_fit`` from composite.py."""


from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Streaming alpha re-fit (concept-drift guard).
#
# Online maintenance of the ``linear_residual`` alpha when production data drifts away from the training distribution. Two helpers:
#
# 1. ``streaming_alpha_check_and_refit(y_buffer, base_buffer, current_alpha, current_beta, *, z_threshold=3.0, min_buffer_n=200)`` -- Chow-style stability check + refit. Computes a fresh (alpha, beta) on the buffer, compares the alpha to ``current_alpha`` via a z-score (delta normalised by an approximate SE), and returns the refit params iff |z| > threshold. Otherwise returns the unchanged ``(current_alpha, current_beta)`` so the caller can no-op.
#
# 2. ``CompositeTargetEstimator.update(y_recent, base_recent)`` -- rolling-buffer interface for production callers. Each ``update()`` appends new observations (FIFO eviction at ``online_refit_buffer_n``); when the buffer fills and a check fires, the wrapper's ``fitted_params_`` is updated in-place with the new alpha/beta. Disabled by default (``online_refit_enabled=False``); explicit opt-in protects against sklearn ``clone()`` semantics breakage (stateful estimators clone as fresh).
# ----------------------------------------------------------------------

_STREAMING_DEFAULT_Z_THRESHOLD: float = 3.0
_STREAMING_DEFAULT_MIN_BUFFER_N: int = 200


def streaming_alpha_check_and_refit(
    y_buffer: np.ndarray,
    base_buffer: np.ndarray,
    current_alpha: float,
    current_beta: float,
    *,
    z_threshold: float = _STREAMING_DEFAULT_Z_THRESHOLD,
    min_buffer_n: int = _STREAMING_DEFAULT_MIN_BUFFER_N,
) -> tuple[float, float, dict[str, Any]]:
    """Chow-style drift check + optional refit on the recent buffer.

    Parameters
    ----------
    y_buffer, base_buffer
        Recent observation arrays (1-D, same length).
    current_alpha, current_beta
        Currently-deployed coefficients to compare against.
    z_threshold
        |z| above this triggers a refit. Default 3.0 (~99.7% Normal CI).
    min_buffer_n
        Minimum buffer size to run the check. Below this the function returns ``(current_alpha, current_beta, {"refit": False, "reason": "buffer_too_small"})`` without computing anything.

    Returns
    -------
    ``(new_alpha, new_beta, info)`` where ``info`` carries diagnostics:
    - ``refit``: bool -- whether the alpha was updated.
    - ``z_score``: float -- |alpha_buffer - current_alpha| / approx_SE. NaN when refit skipped.
    - ``alpha_buffer`` / ``beta_buffer``: fresh fit on the buffer.
    - ``reason``: str -- "drift_detected" / "no_drift" / "buffer_too_small" / "degenerate_buffer".
    """
    # Lazy-import composite-internal helper to break the import cycle.
    from . import _linear_residual_fit
    y_f = np.asarray(y_buffer, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base_buffer, dtype=np.float64).reshape(-1)
    if y_f.size < min_buffer_n:
        return current_alpha, current_beta, {
            "refit": False, "z_score": float("nan"),
            "alpha_buffer": float("nan"), "beta_buffer": float("nan"),
            "reason": "buffer_too_small",
        }
    finite = np.isfinite(y_f) & np.isfinite(base_f)
    if finite.sum() < min_buffer_n:
        return current_alpha, current_beta, {
            "refit": False, "z_score": float("nan"),
            "alpha_buffer": float("nan"), "beta_buffer": float("nan"),
            "reason": "buffer_too_small",
        }
    y_clean = y_f[finite]
    base_clean = base_f[finite]
    fit_params = _linear_residual_fit(y_clean, base_clean)
    alpha_buf = float(fit_params["alpha"])
    beta_buf = float(fit_params["beta"])
    # Residual-based OLS slope SE. Form: SE(alpha) = sqrt(SSE / (n-2)) /
    # (sqrt(n) * base_std). The naive y_std / (sqrt(n) * base_std)
    # over-inflates SE when the regressor explains most variance.
    base_std = float(np.std(base_clean))
    n_buf = int(finite.sum())
    if base_std < 1e-12:
        return current_alpha, current_beta, {
            "refit": False, "z_score": float("nan"),
            "alpha_buffer": alpha_buf, "beta_buffer": beta_buf,
            "reason": "degenerate_buffer",
        }
    if n_buf > 2:
        residuals = y_clean - (alpha_buf * base_clean + beta_buf)
        sse = float(np.sum(residuals * residuals))
        sigma_resid = float(np.sqrt(max(sse / (n_buf - 2), 0.0)))
    else:
        sigma_resid = float(np.std(y_clean))
    se_alpha = sigma_resid / (np.sqrt(n_buf) * base_std)
    z = abs(alpha_buf - current_alpha) / max(se_alpha, 1e-12)
    if z > z_threshold:
        return alpha_buf, beta_buf, {
            "refit": True, "z_score": float(z),
            "alpha_buffer": alpha_buf, "beta_buffer": beta_buf,
            "reason": "drift_detected",
        }
    return current_alpha, current_beta, {
        "refit": False, "z_score": float(z),
        "alpha_buffer": alpha_buf, "beta_buffer": beta_buf,
        "reason": "no_drift",
    }
