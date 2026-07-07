"""Robust cross-member aggregation for the FLOAT (regression / quantile) suite ensemble.

The classification path blends member probabilities via :func:`combine_probs` (mean / geomean / RRF /
... flavours chosen at finalize time). The float-prediction path -- regression and quantile heads whose
members expose ``predict`` not ``predict_proba`` -- historically averaged members with a raw
``np.mean(axis=0)``. That is optimal only when every member is an unbiased estimator: a single outlier
fold / seed / degenerate-but-surviving config drags the mean arbitrarily far, since the mean has zero
breakdown point. Bench (``training/_benchmarks/bench_ensemble_mean_vs_median_agg.py``, 5 scenarios x 3
seeds) shows raw mean RMSE 1.5-8x worse than a robust aggregate whenever 1-2 of K members are corrupted,
while costing only ~6% in the all-clean regime.

:func:`robust_float_ensemble` (``flavour="robust"``) drops, per output column, members whose deviation from
the column median exceeds ``mad_factor`` scaled-MADs, then averages the survivors. It is OPT-IN, not the
production default: the MAD gate over-fires on normal fold spread at small K and no factor recovers a clean
default. ``bench_mad_factor_sweep`` (factors 3.5..8.0, clean K=3/5/8 + outlier 1-2-of-K, 5 seeds) shows the
smallest factor with <=1% clean cost would need K>=8, while K=3 clean cost stays ~14% at 3.5 / ~7.5% at 8.0
and raising the factor erodes protection (8.0 -> 1.25x min). No factor reaches <=1% clean cost AND >=2x
protection, so the production resolver keeps ``flavour="mean"`` and robustness is enabled per-model via the
``float_ensemble_flavour`` metadata key. ``flavour="median"`` is the full-breakdown analogue.
"""

from __future__ import annotations

import numpy as np

FloatEnsembleFlavour = ("robust", "mean", "median")

DEFAULT_MAD_FACTOR: float = 3.5


def robust_float_ensemble(stacked: np.ndarray, *, mad_factor: float = DEFAULT_MAD_FACTOR) -> np.ndarray:
    """Per-column robust mean: average members whose |dev from median| <= ``mad_factor`` * scaled-MAD.

    ``stacked`` is ``(K, ...)`` -- K member predictions along axis 0. Columns where the scaled MAD is 0
    (members agree, or only one member) fall back to the plain mean, so a clean ensemble is bit-close to
    ``stacked.mean(axis=0)`` (identical up to the keep-mask being all-True). Returns shape ``stacked[0]``.
    """
    if stacked.shape[0] < 3:
        return np.asarray(stacked.mean(axis=0))
    med = np.median(stacked, axis=0)
    abs_dev = np.abs(stacked - med)
    scaled_mad = 1.4826 * np.median(abs_dev, axis=0)
    thresh = np.where(scaled_mad > 0, mad_factor * scaled_mad, np.inf)
    keep = abs_dev <= thresh
    keep_counts = keep.sum(axis=0)
    keep_counts = np.where(keep_counts == 0, stacked.shape[0], keep_counts)
    masked_sum = np.where(keep, stacked, 0.0).sum(axis=0)
    return np.asarray(masked_sum / keep_counts)


def combine_float_predictions(stacked: np.ndarray, *, flavour: str = "robust", mad_factor: float = DEFAULT_MAD_FACTOR) -> np.ndarray:
    """Aggregate stacked float member predictions ``(K, ...)`` into one array.

    ``flavour``: ``"robust"`` (default, MAD-gated mean), ``"mean"`` (legacy raw mean), ``"median"``.
    """
    if flavour == "mean":
        return np.asarray(stacked.mean(axis=0))
    if flavour == "median":
        return np.asarray(np.median(stacked, axis=0))
    if flavour == "robust":
        return robust_float_ensemble(stacked, mad_factor=mad_factor)
    raise ValueError(f"combine_float_predictions: unknown flavour={flavour!r}, expected one of {FloatEnsembleFlavour}")
