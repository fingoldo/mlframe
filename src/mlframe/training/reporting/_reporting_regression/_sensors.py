"""Regression-report prediction sensors: envelope clip + collapse detector.

Carved out of the ``report_regression_model_perf`` body to keep the package ``__init__`` below the 1k-line monolith
threshold. Both functions operate on the already-flattened ``preds_arr`` / ``targets_arr`` numpy arrays and the
optional train-y envelope stats; neither mutates the caller's frames.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def apply_prediction_envelope_clip(
    preds_arr: np.ndarray,
    targets_arr: np.ndarray,
    *,
    y_train_min: float | None,
    y_train_max: float | None,
    y_train_std: float | None,
    model_name: Any,
    report_title: str,
) -> np.ndarray:
    """Clip predictions to a sigma window around the train (or eval-fallback) target range BEFORE metrics + chart.

    Applies to ALL regression models. Linear / Ridge / Lasso can extrapolate as catastrophically as unbounded MLPs on
    group-aware splits with strong-AR / heavy-tail targets. Train-y stats are the conceptually correct bound; when the
    suite didn't thread them, an eval-derived bound (k_sigma=10, intentionally generous) is a defensive net against
    truly catastrophic predictions. No-op when no usable envelope can be derived.
    """
    _K_FALLBACK_SIGMA = 10.0
    _env_stats = None
    _envelope_source = "none"
    if (y_train_min is not None and y_train_max is not None
            and y_train_std is not None and y_train_std > 0):
        from ..._prediction_envelope_clip import TrainEnvelopeStats
        _env_stats = TrainEnvelopeStats(
            y_min=float(y_train_min),
            y_max=float(y_train_max),
            y_std=float(y_train_std),
        )
        _envelope_source = "train"
    elif targets_arr.ndim == 1 and targets_arr.size > 0:
        _y_eval = targets_arr[np.isfinite(targets_arr)]
        if _y_eval.size >= 10:
            _y_std = float(_y_eval.std())
            if _y_std > 0:
                from ..._prediction_envelope_clip import TrainEnvelopeStats
                _env_stats = TrainEnvelopeStats(
                    y_min=float(_y_eval.min()),
                    y_max=float(_y_eval.max()),
                    y_std=_y_std,
                )
                _envelope_source = "eval-fallback"
    if _env_stats is not None:
        from ..._prediction_envelope_clip import clip_predictions_to_train_envelope
        preds_arr = clip_predictions_to_train_envelope(
            preds_arr, _env_stats,
            k_sigma=(3.0 if _envelope_source == "train" else _K_FALLBACK_SIGMA),
            model_label=str(model_name) if model_name else "<unknown>",
            split_label=(
                f"{report_title} [{_envelope_source}]" if report_title
                else f"<unknown> [{_envelope_source}]"
            ),
        )
    return preds_arr


def run_collapse_sensor(
    preds_arr: np.ndarray,
    targets_arr: np.ndarray,
    R2: float,
    *,
    model_name: Any,
    y_train_min: float | None = None,
    y_train_max: float | None = None,
    y_train_std: float | None = None,
) -> None:
    """Log a HARD WARNING when predictions look pathological (collapse / extrapolation / mean-shift / out-of-envelope).

    A model whose predictions have std << target std with R^2 < 0 is collapsed (emitting a near-constant value). Other
    branches catch unbounded extrapolation (R^2 << -1 with a far-off worst prediction), systematic mean-shift, and
    predictions far outside the train-y envelope. Skipped on DummyBaseline outputs (intended-constant). Never raises.
    """
    _is_dummy_baseline = "DummyBaseline:" in str(model_name) if model_name else False
    if _is_dummy_baseline:
        return
    try:
        _pred_std = float(np.std(preds_arr)) if preds_arr.size > 1 else 0.0
        _y_std = float(np.std(targets_arr)) if targets_arr.size > 1 else 0.0
        _pred_mean = float(np.mean(preds_arr)) if preds_arr.size else 0.0
        _y_mean = float(np.mean(targets_arr)) if targets_arr.size else 0.0
        _r2 = float(R2)
        _collapse_std = (
            _y_std > 0 and _pred_std < 0.2 * _y_std and _r2 < 0
        )
        try:
            _max_err = float(np.max(np.abs(preds_arr - targets_arr)))
        except Exception:
            _max_err = 0.0
        _collapse_extrapolation = (
            _y_std > 0 and _r2 < -1.0 and _max_err > 5.0 * _y_std
        )
        _collapse_mean_shift = (
            _y_std > 0 and abs(_pred_mean - _y_mean) > 3.0 * _y_std
        )
        # When train-y stats are plumbed, additionally trip when pred falls >3 sigma outside [y_train_min, y_train_max]:
        # catches the case where the in-batch target_std happens tighter than train-y_std and the extrapolation branch misses.
        _collapse_train_envelope = False
        if (y_train_min is not None and y_train_max is not None
                and y_train_std is not None and y_train_std > 0):
            try:
                _pred_min = float(np.min(preds_arr))
                _pred_max = float(np.max(preds_arr))
                _below_lo = (float(y_train_min) - _pred_min) / float(y_train_std)
                _above_hi = (_pred_max - float(y_train_max)) / float(y_train_std)
                if _below_lo > 3.0 or _above_hi > 3.0:
                    _collapse_train_envelope = True
            except Exception:
                pass
        if not (_collapse_std or _collapse_extrapolation
                or _collapse_mean_shift or _collapse_train_envelope):
            return
        # Disambiguate the branch by model name: Ridge / LinearRegression on a group-aware split with feature-distribution
        # shift produces the SAME signature as Identity-MLP collapse, so operators shouldn't be told to blame a neural stack.
        _model_name_s = str(model_name) if model_name else ""
        _is_neural_stack = any(
            tag in _model_name_s for tag in (
                "PytorchLightning", "MLP", "TabularNet", "_TTRWithEvalSetScaling",
            )
        )
        if _collapse_std:
            _branch = "std-collapse"
        elif _collapse_extrapolation:
            _branch = ("linear-extrapolation" if _is_neural_stack else "group-ood-shift")
        elif _collapse_train_envelope:
            _branch = "outside-train-y-envelope"
        else:
            _branch = "mean-shift"
        _hint = (
            "For Identity-MLP / linear-stack: set nlayers=1 or pick a "
            "real nonlinearity (nn.ReLU / nn.GELU); the stacked-Linear "
            "footgun catastrophically extrapolates on unseen-groups "
            "test splits (observed in prod). For MLP+LN_in: try "
            "``mlp_kwargs={'network_params': {'use_layernorm': False}}``. "
            "For tree boosters: check fit_params learning_rate / n_estimators."
            if _is_neural_stack
            else (
            "Likely cause: group-aware split + feature distribution "
            "shift between train and test wells/groups. The linear "
            "model's coefficients fit on train-feature scale, but "
            "test rows have features from a different distribution -- "
            "predictions drift off systematically. Mitigations: "
            "(a) let composite-target discovery propose a residualised "
            "target with bounded variance, (b) use a tree booster "
            "(less sensitive to feature scale shift), (c) verify the "
            "group-aware split assumptions match downstream model "
            "robustness expectations."
            )
        )
        logger.warning(
            "[regression-collapse-sensor:%s] %s: predictions appear pathological -- "
            "pred_std=%.3g (%.1f%% of target_std=%.3g), "
            "pred_mean=%.3g vs target_mean=%.3g, "
            "max|pred-y|=%.3g (%.1fx target_std), R2=%.3g. "
            "%s",
            _branch,
            model_name,
            _pred_std,
            100.0 * _pred_std / max(_y_std, 1e-12),
            _y_std,
            _pred_mean, _y_mean,
            _max_err, _max_err / max(_y_std, 1e-12),
            _r2,
            _hint,
        )
    except Exception as _sensor_err:
        logger.debug(
            "regression-collapse-sensor probe failed (non-fatal): %s", _sensor_err,
        )


__all__ = ["apply_prediction_envelope_clip", "run_collapse_sensor"]
