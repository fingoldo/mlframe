"""Generic post-predict envelope clip for regression models.

Phase contract: after a regression model produces predictions on
val / test (or any out-of-sample split), clamp the predictions to a
bounded envelope around the train-target range BEFORE downstream
metric computation, reporting, or ensemble integration.

Motivation: any regression model with unbounded output capacity
(linear / ridge / lasso / MLP / Identity-MLP / linear-leaf trees with
extrapolation flags) can produce predictions hundreds of sigma outside
the train range on group-aware splits with strongly-autoregressive or
heavy-tail targets. Catastrophic predictions poison:
  * the chart (pred range axis blows out -> scatter visually useless);
  * the reported RMSE / MaxError (one row at 10^6 sigma dominates);
  * downstream ensemble stacking (NNLS weights driven by spurious
    components).

Documented prod incidents that drove this:
  * 2026-05-22: Identity-MLP R^2=-326 on a group-aware regression
    test split.
  * 2026-05-24: MLP pred_std=58 vs target_std=645, R^2=-286.
  * 2026-05-26 (a): MLP pred range [-50k, +250k] on target in
    [10500, 12800], MaxError=781k, R^2=-2624.
  * 2026-05-26 (b): Ridge on a composite (Yeo-Johnson residual)
    target produced y-scale predictions in [-400k, +50k] for target
    in similar range, MaxError=1.4M, R^2=-6934.

The clip is a SAFETY NET, not a learning fix: it bounds the damage but
the underlying model still extrapolates badly. The y-scale wrap pass
on composite targets has its OWN clip inside ``CompositeTargetEstimator``;
this module is the GENERIC clip applied to raw-target regression
predictions (and to wrapper outputs when the wrapper itself did not
already clip).

Default policy: clip ALL regression predictions to
``[y_train_min - K*std, y_train_max + K*std]`` where K defaults to 3
(matches the 3-sigma envelope used by other defensive layers like the
TTR predict clip on MLP). Opt out via env
``MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP=1`` or per-call
``apply_clip=False``.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


_ENV_DISABLE = "MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP"


class TrainEnvelopeStats(NamedTuple):
    """Stats needed to define the clip envelope.

    Populated once from the train target at the suite phase that already
    computes other train-side aggregates; passed through to the predict
    paths via the existing ``y_train_min`` / ``y_train_max`` /
    ``y_train_std`` kwargs on ``report_regression_model_perf``.
    """

    y_min: float
    y_max: float
    y_std: float


def compute_train_envelope_stats(y_train: Any) -> Optional[TrainEnvelopeStats]:
    """Reduce ``y_train`` to (min, max, std) skipping non-finite rows.

    Returns ``None`` when y_train is degenerate (too few finite rows,
    zero variance). Callers that get ``None`` should skip the clip
    entirely (no envelope = no clip).
    """
    try:
        arr = np.asarray(y_train, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size < 10:
            return None
        y_std = float(finite.std())
        if y_std <= 0:
            return None
        return TrainEnvelopeStats(
            y_min=float(finite.min()),
            y_max=float(finite.max()),
            y_std=y_std,
        )
    except Exception as exc:
        logger.debug("compute_train_envelope_stats failed: %s", exc)
        return None


def clip_predictions_to_train_envelope(
    preds: Any,
    stats: Optional[TrainEnvelopeStats],
    *,
    k_sigma: float = 3.0,
    model_label: str = "<unknown>",
    split_label: str = "<unknown>",
    apply_clip: bool = True,
) -> np.ndarray:
    """Return ``preds`` clipped to ``[y_min - k*std, y_max + k*std]``.

    No-op (returns ``preds`` unchanged as ndarray) when:
      * ``stats`` is None (degenerate train target).
      * ``apply_clip`` is False.
      * ``MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP`` env var is set.

    Logs a WARNING with the row counts above / below the envelope
    when the clip actually fires. Idempotent: in-envelope predictions
    pass through bit-exact.
    """
    arr = np.asarray(preds, dtype=np.float64)
    if not apply_clip or stats is None or os.environ.get(_ENV_DISABLE):
        return arr
    low = stats.y_min - k_sigma * stats.y_std
    high = stats.y_max + k_sigma * stats.y_std
    # iter433: math.isfinite on Python floats is 7.5x faster than
    # np.isfinite for scalars (1us -> 0.13us). low/high are floats
    # from stats arithmetic; the array-mask uses below still use np.
    if not (math.isfinite(low) and math.isfinite(high)):
        return arr
    n_low = int(np.sum(arr < low))
    n_high = int(np.sum(arr > high))
    if n_low == 0 and n_high == 0:
        return arr
    logger.warning(
        "[prediction-envelope-clip] %s on %s: %d row(s) below %.4g and "
        "%d row(s) above %.4g (envelope = [y_train_min - %.1f*std, "
        "y_train_max + %.1f*std]). Clipping. Disable via env "
        "%s=1.",
        model_label, split_label,
        n_low, low, n_high, high,
        k_sigma, k_sigma, _ENV_DISABLE,
    )
    return np.clip(arr, low, high)


__all__ = [
    "TrainEnvelopeStats",
    "compute_train_envelope_stats",
    "clip_predictions_to_train_envelope",
]
