"""Test-time augmentation: average predictions over slightly perturbed inputs (Workstream B2).

Adds small per-feature Gaussian jitter to the inputs, predicts ``n`` times, and aggregates -- a cheap
robustness boost and an empirical predictive-spread estimate. Regression aggregates by mean (default) or
median (robust to outlier passes under heavy perturbation); classification averages the PROBABILITIES
(arithmetic mean in probability space -- the standard, well-calibrated choice; geometric/logit mean is
sharper but overconfident, so it is not the default). Model-agnostic: takes any ``predict_fn``.

Opt-in (``behavior_config.tta_samples`` wiring is a later step); this module is the tested core.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np


def tta_predict(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    n: int = 16,
    sigma_scale: float = 0.02,
    feature_std: Optional[np.ndarray] = None,
    agg: str = "mean",
    seed: int = 0,
) -> np.ndarray:
    """Average ``predict_fn`` over ``n`` jittered copies of ``X``.

    Jitter sd per feature = ``sigma_scale * feature_std`` (computed from ``X`` when not supplied). ``agg``
    is ``"mean"`` (default; also the correct choice for class probabilities) or ``"median"`` (regression,
    heavy perturbation). ``n<=1`` or ``sigma_scale<=0`` returns the clean prediction unchanged. Works for
    1-D regression output and 2-D ``(rows, classes)`` probabilities alike.
    """
    Xf = np.asarray(X, dtype=np.float64)
    clean = np.asarray(predict_fn(Xf))
    if n <= 1 or sigma_scale <= 0:
        return clean
    if feature_std is None:
        feature_std = Xf.std(axis=0)
    feature_std = np.asarray(feature_std, dtype=np.float64).reshape(1, -1)
    rng = np.random.default_rng(seed)
    preds = [clean]
    for _ in range(n - 1):
        noise = rng.standard_normal(Xf.shape) * (sigma_scale * feature_std)
        preds.append(np.asarray(predict_fn(Xf + noise)))
    stacked = np.stack(preds, axis=0)
    if agg == "median":
        return np.median(stacked, axis=0)
    return np.mean(stacked, axis=0)


def tta_predict_spread(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    n: int = 16,
    sigma_scale: float = 0.02,
    feature_std: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """Per-row std of the prediction across the ``n`` perturbed passes -- an (approximate, uncalibrated) input-sensitivity spread."""
    Xf = np.asarray(X, dtype=np.float64)
    if feature_std is None:
        feature_std = Xf.std(axis=0)
    feature_std = np.asarray(feature_std, dtype=np.float64).reshape(1, -1)
    rng = np.random.default_rng(seed)
    preds = [np.asarray(predict_fn(Xf))]
    for _ in range(max(0, n - 1)):
        noise = rng.standard_normal(Xf.shape) * (sigma_scale * feature_std)
        preds.append(np.asarray(predict_fn(Xf + noise)))
    return np.std(np.stack(preds, axis=0), axis=0)
