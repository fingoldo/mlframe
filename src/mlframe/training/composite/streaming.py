"""Streaming alpha check/refit: Chow-style drift detector + change-point-aware refit for linear_residual coefficients. Lazy-imports ``_linear_residual_fit`` from composite.py."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Streaming alpha re-fit (concept-drift guard).
#
# Online maintenance of the ``linear_residual`` coefficients when production data drifts away from the training distribution. Two helpers:
#
# 1. ``streaming_alpha_check_and_refit(y_buffer, base_buffer, current_alpha, current_beta, *, z_threshold=3.0, min_buffer_n=200)`` -- Chow-style stability check + refit. Computes a fresh (alpha, beta) on the buffer, compares BOTH the slope alpha and the intercept beta to the deployed coefficients via per-coefficient z-scores (each delta normalised by its OLS standard error), and returns the refit params iff EITHER |z_alpha| > threshold or |z_beta| > threshold. Otherwise returns the unchanged ``(current_alpha, current_beta)`` so the caller can no-op.
#
#    Before fitting, the helper runs a single-change-point scan (two-segment SSE split): a drifting buffer holds pre- AND post-drift rows mixed FIFO, so an OLS over the whole buffer is biased toward the dead (pre-drift) regime. When a significant break is detected the refit (and the SE / z estimates) use ONLY the post-change segment, so the corrected coefficients track the live regime instead of the blend. A homogeneous buffer (no break that beats the F gate) falls back to the full-buffer fit.
#
# 2. ``CompositeTargetEstimator.update(y_recent, base_recent)`` -- rolling-buffer interface for production callers. Each ``update()`` appends new observations (FIFO eviction at ``online_refit_buffer_n``); when the buffer fills and a check fires, the wrapper's ``fitted_params_`` is updated in-place with the new alpha/beta. Disabled by default (``online_refit_enabled=False``); explicit opt-in protects against sklearn ``clone()`` semantics breakage (stateful estimators clone as fresh).
# ----------------------------------------------------------------------

_STREAMING_DEFAULT_Z_THRESHOLD: float = 3.0
_STREAMING_DEFAULT_MIN_BUFFER_N: int = 200
# Minimum rows on EACH side of a candidate change point so both segment OLS
# fits are well-posed (need >2 for a slope+intercept; 30 keeps the SE sane).
_STREAMING_CP_MIN_SEGMENT_N: int = 30
# F-statistic gate for accepting a single change point. A two-segment fit
# always lowers SSE (4 free params vs 2); we only act on a break that beats
# this ratio so noise wiggles never trigger a spurious post-segment refit.
# 12.0 ~= the upper tail of F(2, n-4) for n in the few-hundred range -- a
# conservative gate that admits a genuine regime shift (the SSE drop is then
# large) while rejecting in-regime noise (the drop is marginal). Exposed as a
# parameter so callers can retune.
_STREAMING_CP_F_THRESHOLD: float = 12.0


def _ols_alpha_beta_sse(y: np.ndarray, base: np.ndarray) -> tuple[float, float, float]:
    """Closed-form OLS slope/intercept + residual SSE on a finite 1-D pair.

    Returns ``(alpha, beta, sse)``. ``base`` must already be the finite,
    cleaned regressor (caller masks NaN/inf). For a degenerate (zero-variance)
    base the slope is 0 and beta is the mean of ``y`` (matches
    ``_linear_residual_fit``'s rank-deficient lstsq behaviour).
    """
    n = y.size
    if n < 2:
        b = float(np.mean(y)) if n > 0 else 0.0
        return 0.0, b, 0.0
    bmean = float(np.mean(base))
    ymean = float(np.mean(y))
    base_c = base - bmean
    sxx = float(np.dot(base_c, base_c))
    if sxx < 1e-300:
        resid = y - ymean
        return 0.0, ymean, float(np.dot(resid, resid))
    sxy = float(np.dot(base_c, y - ymean))
    alpha = sxy / sxx
    beta = ymean - alpha * bmean
    resid = y - (alpha * base + beta)
    sse = float(np.dot(resid, resid))
    return alpha, beta, sse


def _detect_change_point(
    y: np.ndarray,
    base: np.ndarray,
    *,
    min_segment_n: int = _STREAMING_CP_MIN_SEGMENT_N,
    f_threshold: float = _STREAMING_CP_F_THRESHOLD,
) -> dict[str, Any]:
    """Single-change-point scan over a FIFO buffer.

    Scans every split ``k`` with at least ``min_segment_n`` rows on each side,
    fits OLS on ``[0:k]`` and ``[k:n]`` independently, and picks the split that
    minimises the combined two-segment SSE. The break is accepted only when the
    Chow F-statistic

        F = ((SSE_full - SSE_split) / q) / (SSE_split / (n - 2q)),  q = 2

    exceeds ``f_threshold`` (a regime shift drops SSE far more than the 2 extra
    free parameters explain; in-regime noise does not).

    Parameters
    ----------
    y, base
        Finite, cleaned 1-D arrays of equal length (caller masks non-finite).
    min_segment_n
        Minimum rows on each side of the split.
    f_threshold
        F gate above which the break is accepted.

    Returns
    -------
    dict with:
    - ``found``: bool -- a significant break was accepted.
    - ``cp_index``: int -- index of the first post-change row (the live segment
      is ``[cp_index:]``); ``-1`` when no break found.
    - ``f_stat``: float -- the Chow F at the best split (NaN when not scannable).
    - ``sse_full`` / ``sse_split``: float -- single- vs two-segment SSE.
    - ``n_post``: int -- rows in the live (post-change) segment.
    """
    n = int(y.size)
    no_break = {
        "found": False, "cp_index": -1, "f_stat": float("nan"),
        "sse_full": float("nan"), "sse_split": float("nan"), "n_post": n,
    }
    if n < 2 * min_segment_n + 1:
        return no_break
    _, _, sse_full = _ols_alpha_beta_sse(y, base)
    if not np.isfinite(sse_full):
        return no_break
    # O(n) split scan via prefix sufficient statistics. Each candidate
    # split's two-segment SSE is computed in O(1) from cumulative sums of
    # base, y, base^2, base*y, y^2 -- the per-segment OLS SSE is
    #   SSE_seg = Syy - Sxy^2 / Sxx
    # with the centred sums Sxx = sum(b^2) - (sum b)^2/m, etc. Numerically
    # equivalent to the per-segment _ols_alpha_beta_sse (same sufficient
    # statistics) up to FP rounding in the cumulative reduction.
    # _ols_alpha_beta_sse is retained for sse_full + as the tested reference.
    cb = np.concatenate(([0.0], np.cumsum(base)))  # prefix sum base
    cy = np.concatenate(([0.0], np.cumsum(y)))  # prefix sum y
    cbb = np.concatenate(([0.0], np.cumsum(base * base)))  # prefix sum base^2
    cby = np.concatenate(([0.0], np.cumsum(base * y)))  # prefix sum base*y
    cyy = np.concatenate(([0.0], np.cumsum(y * y)))  # prefix sum y^2
    ks = np.arange(min_segment_n, n - min_segment_n + 1, dtype=np.int64)

    def _seg_sse(m, sb, sy, sbb, sby, syy):
        """Vectorised per-segment OLS residual SSE from sufficient stats."""
        m = m.astype(np.float64)
        sxx = sbb - (sb * sb) / m
        sxy = sby - (sb * sy) / m
        syy_c = syy - (sy * sy) / m
        # alpha = sxy/sxx; SSE = syy_c - sxy^2/sxx. Degenerate (sxx~0) -> slope
        # 0, SSE = syy_c (mean-only fit), matching _ols_alpha_beta_sse.
        safe = sxx > 1e-300
        sse = np.where(safe, syy_c - np.where(safe, (sxy * sxy), 0.0) / np.where(safe, sxx, 1.0), syy_c)
        return np.maximum(sse, 0.0)

    m_l = ks
    sse_l = _seg_sse(m_l, cb[ks], cy[ks], cbb[ks], cby[ks], cyy[ks])
    m_r = n - ks
    sse_r = _seg_sse(
        m_r, cb[n] - cb[ks], cy[n] - cy[ks], cbb[n] - cbb[ks],
        cby[n] - cby[ks], cyy[n] - cyy[ks],
    )
    sse_split_all = sse_l + sse_r
    best_i = int(np.argmin(sse_split_all))
    best_k = int(ks[best_i])
    best_sse = float(sse_split_all[best_i])
    if best_k < 0 or not np.isfinite(best_sse):
        return no_break
    q = 2  # extra parameters introduced by the split (alpha2, beta2)
    dof_resid = n - 2 * q
    if dof_resid <= 0 or best_sse <= 0.0:
        return no_break
    f_stat = ((sse_full - best_sse) / q) / (best_sse / dof_resid)
    if f_stat <= f_threshold:
        return {
            "found": False, "cp_index": -1, "f_stat": float(f_stat),
            "sse_full": float(sse_full), "sse_split": float(best_sse),
            "n_post": n,
        }
    return {
        "found": True, "cp_index": int(best_k), "f_stat": float(f_stat),
        "sse_full": float(sse_full), "sse_split": float(best_sse),
        "n_post": int(n - best_k),
    }


def streaming_alpha_check_and_refit(
    y_buffer: np.ndarray,
    base_buffer: np.ndarray,
    current_alpha: float,
    current_beta: float,
    *,
    z_threshold: float = _STREAMING_DEFAULT_Z_THRESHOLD,
    min_buffer_n: int = _STREAMING_DEFAULT_MIN_BUFFER_N,
    detect_change_point: bool = True,
    cp_min_segment_n: int = _STREAMING_CP_MIN_SEGMENT_N,
    cp_f_threshold: float = _STREAMING_CP_F_THRESHOLD,
) -> tuple[float, float, dict[str, Any]]:
    """Chow-style drift check + change-point-aware refit on the recent buffer.

    Parameters
    ----------
    y_buffer, base_buffer
        Recent observation arrays (1-D, same length).
    current_alpha, current_beta
        Currently-deployed coefficients to compare against.
    z_threshold
        |z| above this (on EITHER the slope or the intercept) triggers a refit.
        Default 3.0 (~99.7% Normal CI).
    min_buffer_n
        Minimum buffer size to run the check. Below this the function returns
        ``(current_alpha, current_beta, {"refit": False, ...,
        "reason": "buffer_too_small"})`` without computing anything.
    detect_change_point
        When True (default), scan the buffer for a single regime break and,
        if one is found, refit + compute the drift z-scores on the POST-CHANGE
        segment only. A FIFO buffer mixes pre- and post-drift rows, so a
        whole-buffer OLS is biased toward the dead regime; fitting the live
        segment recovers the current coefficients. No break (or too few rows) ->
        full-buffer fit. Set False to force the whole-buffer fit.
    cp_min_segment_n, cp_f_threshold
        Change-point scan knobs (min rows per side; F gate to accept a break).

    Returns
    -------
    ``(new_alpha, new_beta, info)`` where ``info`` carries diagnostics:
    - ``refit``: bool -- whether the coefficients were updated.
    - ``z_score``: float -- max(|z_alpha|, |z_beta|); the drift magnitude
      (NaN when refit skipped before any fit). Reported for both the slope-
      drift and the level-shift case.
    - ``z_alpha`` / ``z_beta``: float -- per-coefficient z-scores.
    - ``alpha_buffer`` / ``beta_buffer``: fresh fit on the (live segment of the)
      buffer.
    - ``change_point``: int -- first post-change row index when a break was
      detected and used, else -1.
    - ``cp_f_stat``: float -- the Chow F at the detected break (NaN if none).
    - ``n_fit``: int -- rows the refit actually used (post-change segment or
      whole buffer).
    - ``reason``: str -- "drift_detected" (slope drift) / "drift_detected_level"
      (intercept/level-shift drift only) / "no_drift" / "buffer_too_small" /
      "degenerate_buffer".
    """
    # Lazy-import composite-internal helper to break the import cycle.
    from . import _linear_residual_fit
    y_f = np.asarray(y_buffer, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base_buffer, dtype=np.float64).reshape(-1)

    def _too_small() -> tuple[float, float, dict[str, Any]]:
        return current_alpha, current_beta, {
            "refit": False, "z_score": float("nan"),
            "z_alpha": float("nan"), "z_beta": float("nan"),
            "alpha_buffer": float("nan"), "beta_buffer": float("nan"),
            "change_point": -1, "cp_f_stat": float("nan"), "n_fit": 0,
            "reason": "buffer_too_small",
        }

    if y_f.size < min_buffer_n:
        return _too_small()
    finite = np.isfinite(y_f) & np.isfinite(base_f)
    if finite.sum() < min_buffer_n:
        return _too_small()
    y_clean = y_f[finite]
    base_clean = base_f[finite]

    # Change-point-aware fit window. A drifting FIFO buffer holds the dead
    # regime in its head and the live regime in its tail; fitting the whole
    # buffer blends them. Restrict the refit window to the live (post-change)
    # segment when a significant single break is detected.
    cp_info: dict[str, Any] = {
        "found": False, "cp_index": -1, "f_stat": float("nan"), "n_post": int(y_clean.size),
    }
    fit_y = y_clean
    fit_base = base_clean
    if detect_change_point:
        cp_info = _detect_change_point(
            y_clean, base_clean,
            min_segment_n=cp_min_segment_n, f_threshold=cp_f_threshold,
        )
        if cp_info["found"] and cp_info["n_post"] >= 2:
            cp = int(cp_info["cp_index"])
            fit_y = y_clean[cp:]
            fit_base = base_clean[cp:]

    fit_params = _linear_residual_fit(fit_y, fit_base)
    alpha_buf = float(fit_params["alpha"])
    beta_buf = float(fit_params["beta"])
    n_fit = int(fit_y.size)

    base_std = float(np.std(fit_base))
    base_mean = float(np.mean(fit_base))
    if base_std < 1e-12:
        return current_alpha, current_beta, {
            "refit": False, "z_score": float("nan"),
            "z_alpha": float("nan"), "z_beta": float("nan"),
            "alpha_buffer": alpha_buf, "beta_buffer": beta_buf,
            "change_point": int(cp_info["cp_index"]) if cp_info["found"] else -1,
            "cp_f_stat": float(cp_info["f_stat"]),
            "n_fit": n_fit,
            "reason": "degenerate_buffer",
        }

    # Residual-based OLS standard errors. sigma_resid = sqrt(SSE / (n-2)) is the
    # residual scale of the fit window. The slope SE is the classic
    # SE(alpha) = sigma_resid / (sqrt(n) * base_std); the intercept SE is
    # SE(beta) = sigma_resid * sqrt(1/n + base_mean^2 / (n * base_var)) -- the
    # textbook OLS intercept SE, which is what lets us detect a pure
    # level-shift (alpha unchanged, beta jumps).
    if n_fit > 2:
        residuals = fit_y - (alpha_buf * fit_base + beta_buf)
        sse = float(np.sum(residuals * residuals))
        sigma_resid = float(np.sqrt(max(sse / (n_fit - 2), 0.0)))
    else:
        sigma_resid = float(np.std(fit_y))
    base_var = base_std * base_std
    se_alpha = sigma_resid / (np.sqrt(n_fit) * base_std)
    se_beta = sigma_resid * np.sqrt(1.0 / n_fit + (base_mean * base_mean) / (n_fit * base_var))
    z_alpha = abs(alpha_buf - current_alpha) / max(se_alpha, 1e-12)
    z_beta = abs(beta_buf - current_beta) / max(se_beta, 1e-12)
    z = max(z_alpha, z_beta)

    cp_used = int(cp_info["cp_index"]) if cp_info["found"] else -1
    cp_f = float(cp_info["f_stat"])
    if z > z_threshold:
        # Distinguish a pure level-shift (only the intercept moved) so callers
        # / monitoring can see that the slope was stable but the level drifted.
        reason = "drift_detected" if z_alpha > z_threshold else "drift_detected_level"
        return alpha_buf, beta_buf, {
            "refit": True, "z_score": float(z),
            "z_alpha": float(z_alpha), "z_beta": float(z_beta),
            "alpha_buffer": alpha_buf, "beta_buffer": beta_buf,
            "change_point": cp_used, "cp_f_stat": cp_f, "n_fit": n_fit,
            "reason": reason,
        }
    return current_alpha, current_beta, {
        "refit": False, "z_score": float(z),
        "z_alpha": float(z_alpha), "z_beta": float(z_beta),
        "alpha_buffer": alpha_buf, "beta_buffer": beta_buf,
        "change_point": cp_used, "cp_f_stat": cp_f, "n_fit": n_fit,
        "reason": "no_drift",
    }
