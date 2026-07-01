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
    # Per-pass independent RNG streams via SeedSequence.spawn: each jittered
    # augmentation draws statistically independent noise (proper diversity),
    # reproducible under a fixed `seed`, and composable when several TTA
    # ensembles are stacked (distinct parent seeds -> disjoint child streams).
    child_rngs = np.random.default_rng(seed).spawn(n - 1)
    preds = [clean]
    for pass_rng in child_rngs:
        noise = pass_rng.standard_normal(Xf.shape) * (sigma_scale * feature_std)
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
    """Per-row std of the prediction across the ``n`` perturbed passes -- an (approximate, uncalibrated) input-sensitivity spread.

    NOTE: this is the POPULATION standard deviation (numpy ``np.std`` default ``ddof=0``), which is low-biased
    for small ``n`` (divides by ``n``, not ``n-1``). Treat it as a diagnostic spread, not a calibrated sample
    sd; rescale by ``sqrt(n/(n-1))`` if an unbiased sd estimate is required.
    """
    Xf = np.asarray(X, dtype=np.float64)
    if feature_std is None:
        feature_std = Xf.std(axis=0)
    feature_std = np.asarray(feature_std, dtype=np.float64).reshape(1, -1)
    # Per-pass independent RNG streams via SeedSequence.spawn (see tta_predict).
    child_rngs = np.random.default_rng(seed).spawn(max(0, n - 1))
    preds = [np.asarray(predict_fn(Xf))]
    for pass_rng in child_rngs:
        noise = pass_rng.standard_normal(Xf.shape) * (sigma_scale * feature_std)
        preds.append(np.asarray(predict_fn(Xf + noise)))
    return np.std(np.stack(preds, axis=0), axis=0)


def tta_point_mean_spread(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    n: int = 16,
    sigma_scale: float = 0.02,
    feature_std: Optional[np.ndarray] = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the clean point prediction, the TTA mean, and the per-element TTA spread (population std, ddof=0) in ONE streaming pass.

    Fuses what ``predict_fn(X)`` + ``tta_predict(..., agg="mean")`` + ``tta_predict_spread(...)`` would otherwise do in three separate
    sweeps (3 clean passes + 2*(n-1) jittered passes = 2n+1 model calls) into a single sweep of n model calls: one clean pass reused as
    both the point estimate and the first augmentation member, plus n-1 jittered passes accumulated via Welford. The clean pass and the
    per-pass jittered noise streams (``default_rng(seed).spawn(n-1)``) are identical to those of the two standalone functions, so ``mean``
    and ``spread`` match them to floating-point reduction-order tolerance (Welford streaming vs two-pass ``np.mean``/``np.std`` differ by ~1e-9 ULP only).

    Returns ``(point, mean, spread)``; ``mean`` and ``spread`` are over the same n-member population as the standalone helpers. With
    ``n<=1`` or ``sigma_scale<=0`` the mean equals the clean point and the spread is all-zero.

    NOTE: ``spread`` is the POPULATION standard deviation (``ddof=0``, ``sqrt(m2/count)``), low-biased for small ``n``; it is a diagnostic
    spread, not a calibrated sample sd. Rescale by ``sqrt(n/(n-1))`` if an unbiased sd estimate is required.
    """
    Xf = np.asarray(X, dtype=np.float64)
    point = np.asarray(predict_fn(Xf), dtype=np.float64)
    if n <= 1 or sigma_scale <= 0:
        return point, point.copy(), np.zeros_like(point)
    if feature_std is None:
        feature_std = Xf.std(axis=0)
    feature_std = np.asarray(feature_std, dtype=np.float64).reshape(1, -1)
    # Per-pass independent RNG streams via SeedSequence.spawn (see tta_predict);
    # identical spawn scheme to tta_predict / tta_predict_spread, so mean and
    # spread match those standalone helpers member-for-member.
    child_rngs = np.random.default_rng(seed).spawn(n - 1)
    # Welford accumulators seeded with the clean pass as member #1.
    count = 1
    mean = point.astype(np.float64).copy()
    m2 = np.zeros_like(mean)
    for pass_rng in child_rngs:
        noise = pass_rng.standard_normal(Xf.shape) * (sigma_scale * feature_std)
        x = np.asarray(predict_fn(Xf + noise), dtype=np.float64)
        count += 1
        delta = x - mean
        mean += delta / count
        m2 += delta * (x - mean)
    spread = np.sqrt(m2 / count)  # ddof=0 to match np.std default
    return point, mean, spread
