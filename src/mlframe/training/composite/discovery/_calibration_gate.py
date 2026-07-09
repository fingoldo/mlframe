"""Calibration-aware (overfit-resistant) spec scoring for composite discovery.

A candidate spec can show a high in-sample MI-gain / low screen RMSE yet be a
poorly-CALIBRATED overfit: its held-out (OOF) residuals are systematically
biased (mean != 0) or mis-scaled (their spread differs from the in-fold spread).
A lucky-but-miscalibrated spec generalises worse than a slightly-weaker spec
whose OOF residuals are unbiased and correctly scaled.

This module is a PURE, OPTIONAL ranking signal the rerank caller MAY consult --
it does NOT rewire the discovery loop. Given a spec's OOF (held-out) residuals
plus its in-fold (training) residual spread, it returns a
``CalibrationScore(gain, calibration_penalty, adjusted_score)`` so discovery can
prefer specs that GENERALISE + stay calibrated over lucky-but-miscalibrated ones.

Train-only / held-out OOF: the caller feeds residuals computed on held-out CV
folds (the OOF/test analogue), never on rows the fold's model trained on. The
in-fold spread is the train-side IQR. No leakage, no frame copy -- the function
takes small 1-D residual arrays only.

Default is no-harm: the gate is OFF by default
(``CALIBRATION_GATE_DEFAULT_ENABLED = False``); callers opt in. When disabled the
caller simply does not consult this module and ranking is byte-for-byte unchanged.

Definitions (all on the OOF residuals ``r = y_true - y_pred`` on held-out rows):
  - bias        = |mean(r)| / scale  -- systematic offset, scale-normalised.
  - var_miscal  = |IQR(r_oof) / IQR(r_infold) - 1|  -- spread inflation/deflation
                  of the held-out residuals vs the in-fold residuals. A spec that
                  overfits has tight in-fold residuals but wide OOF residuals
                  (ratio >> 1); var_miscal grows with that gap, in either
                  direction.
  - penalty     = bias_weight*bias + var_weight*var_miscal  (>= 0).
  - adjusted    = gain - penalty_weight * penalty  -- higher is better. A
                  well-calibrated spec keeps ~all its gain; a miscalibrated one
                  is docked in proportion to how badly it mis-bias / mis-scales.

cProfile note: every op here is a handful of numpy reductions on small (n_oof)
residual arrays -- microseconds at screening scale (n_oof <= tiny_model_sample_n).
The dominant discovery cost is the K-fold tiny-model fits that PRODUCE the
residuals, not this scoring; no actionable speedup (profiled trivial).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# No-harm default: the gate is an opt-in ranking signal. With this False the
# caller does not consult the module and ranking is unchanged (see module docs).
CALIBRATION_GATE_DEFAULT_ENABLED: bool = False

# Default mixing weights. ``penalty_weight`` scales the whole penalty against the
# gain units; ``bias_weight`` / ``var_weight`` split the penalty between the two
# miscalibration modes. Conservative defaults so a mild miscalibration never
# overturns a large genuine gain, but a gross one (overfit) does.
_DEFAULT_PENALTY_WEIGHT: float = 1.0
_DEFAULT_BIAS_WEIGHT: float = 1.0
_DEFAULT_VAR_WEIGHT: float = 1.0
_EPS: float = 1e-12


@dataclass(frozen=True)
class CalibrationScore:
    """Result of :func:`calibration_adjusted_score`.

    Attributes:
        gain: the spec's raw generalisation gain (e.g. ``raw_rmse - oof_rmse``,
            or any "higher-is-better" merit the caller already computed).
        calibration_penalty: combined non-negative penalty (bias + variance
            miscalibration); 0.0 for a perfectly calibrated spec.
        adjusted_score: ``gain - penalty_weight * calibration_penalty``; the
            value the caller ranks by (higher wins).
        bias: scale-normalised |mean OOF residual| component.
        var_miscal: |IQR(oof)/IQR(infold) - 1| component (NaN when the in-fold
            spread is unavailable / degenerate -> treated as 0 in the penalty).
    """

    gain: float
    calibration_penalty: float
    adjusted_score: float
    bias: float
    var_miscal: float


def _iqr(r: np.ndarray) -> float:
    """Interquartile spread of finite entries; NaN if < 4 finite points."""
    r = r[np.isfinite(r)]
    if r.size < 4:
        return float("nan")
    q75, q25 = np.percentile(r, [75.0, 25.0])
    return float(q75 - q25)


def calibration_penalty(
    oof_residuals: np.ndarray,
    infold_residuals: Optional[np.ndarray] = None,
    *,
    scale: Optional[float] = None,
    bias_weight: float = _DEFAULT_BIAS_WEIGHT,
    var_weight: float = _DEFAULT_VAR_WEIGHT,
) -> tuple[float, float, float]:
    """Compute the calibration penalty from held-out (OOF) residuals.

    Args:
        oof_residuals: ``y_true - y_pred`` on HELD-OUT (OOF) rows only -- the
            rows the fold's model never trained on. No leakage by construction.
        infold_residuals: training-fold residuals (the in-fold spread reference).
            When None / degenerate, the variance-miscalibration term is skipped
            (penalty uses bias only); the spread of ``oof_residuals`` itself is
            used as the bias normaliser fallback.
        scale: optional explicit normaliser for the bias term (e.g. ``std(y)``);
            defaults to the OOF residual IQR (robust) then std, then 1.0.
        bias_weight / var_weight: per-component mixing weights.

    Returns:
        ``(penalty, bias, var_miscal)`` -- penalty >= 0; ``var_miscal`` is NaN
        when no usable in-fold spread was available (then it does not contribute).
    """
    r = np.asarray(oof_residuals, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan"), float("nan")

    # Robust scale for bias normalisation: prefer caller scale, else OOF IQR,
    # else OOF std, else 1.0. IQR is outlier-robust on heavy-tail residuals.
    if scale is not None and np.isfinite(scale) and scale > _EPS:
        norm = float(scale)
    else:
        oof_iqr = _iqr(r)
        if np.isfinite(oof_iqr) and oof_iqr > _EPS:
            norm = oof_iqr
        else:
            s = float(np.std(r))
            norm = s if s > _EPS else 1.0

    bias = abs(float(np.mean(r))) / norm

    var_miscal = float("nan")
    if infold_residuals is not None:
        inf = np.asarray(infold_residuals, dtype=np.float64).reshape(-1)
        in_iqr = _iqr(inf)
        oof_iqr = _iqr(r)
        if np.isfinite(in_iqr) and in_iqr > _EPS and np.isfinite(oof_iqr):
            var_miscal = abs(oof_iqr / in_iqr - 1.0)

    penalty = bias_weight * bias
    if np.isfinite(var_miscal):
        penalty += var_weight * var_miscal
    return float(penalty), float(bias), float(var_miscal)


def calibration_adjusted_score(
    gain: float,
    oof_residuals: np.ndarray,
    infold_residuals: Optional[np.ndarray] = None,
    *,
    scale: Optional[float] = None,
    penalty_weight: float = _DEFAULT_PENALTY_WEIGHT,
    bias_weight: float = _DEFAULT_BIAS_WEIGHT,
    var_weight: float = _DEFAULT_VAR_WEIGHT,
) -> CalibrationScore:
    """Calibration-adjusted ranking score for a candidate spec.

    ``adjusted = gain - penalty_weight * penalty``, where ``penalty`` rewards
    unbiased + correctly-scaled OOF residuals. Higher ``adjusted_score`` wins.

    A well-calibrated spec (mean OOF residual ~0, OOF spread ~ in-fold spread)
    gets penalty ~0 and keeps its gain. A lucky-but-miscalibrated overfit (low
    in-sample RMSE but biased / inflated OOF residuals) is docked, so a slightly
    weaker but honestly-calibrated spec can outrank it.

    See module docstring for the no-leakage contract: ``oof_residuals`` are
    held-out only. This is a PURE function -- safe to call (or not) per spec; it
    mutates nothing and copies no frame.

    UNIT CAVEAT: ``gain`` is an MI-gain (nats) while ``penalty`` is DIMENSIONLESS -- a sum of a
    scale-normalised bias (``|mean residual| / robust_scale``) and a unit-free spread ratio
    (``|oof_iqr / infold_iqr - 1|``). The subtraction ``gain - penalty_weight * penalty`` is therefore
    only meaningful because ``penalty_weight`` carries the implicit nats-per-penalty-unit conversion: it
    is a TUNED trade-off knob, not a physically commensurate subtraction. The default ``penalty_weight``
    was calibrated so a fully-miscalibrated spec is docked on the order of a typical gain; if you change
    the gain UNIT (e.g. switch the MI estimator's base or normalise gain) you MUST re-tune
    ``penalty_weight`` -- the two terms are not on the same scale by construction.
    """
    penalty, bias, var_miscal = calibration_penalty(
        oof_residuals, infold_residuals,
        scale=scale, bias_weight=bias_weight, var_weight=var_weight,
    )
    if not np.isfinite(penalty):
        # No usable residuals -> no calibration signal; pass gain through
        # unchanged (no-harm) and flag the penalty as NaN for the caller.
        return CalibrationScore(
            gain=float(gain), calibration_penalty=float("nan"),
            adjusted_score=float(gain), bias=float("nan"),
            var_miscal=float("nan"),
        )
    adjusted = float(gain) - penalty_weight * penalty
    return CalibrationScore(
        gain=float(gain), calibration_penalty=penalty,
        adjusted_score=adjusted, bias=bias, var_miscal=var_miscal,
    )
