"""Module-level, picklable stand-ins for the metric closures ``get_training_configs`` used to build.

``get_training_configs`` defined its calibration metrics as nested functions and attached them to model params
(``XGB_CALIB_CLASSIF["eval_metric"]``, ``CB_CALIB_CLASSIF``'s ``ICE(metric=...)``, the LightGBM adapter). A local
function cannot be pickled, so every ``save_mlframe_model`` call failed over to ``dill``:

    save_mlframe_model: pickle rejected the payload (AttributeError: Can't pickle local object
    'get_training_configs.<locals>.integral_calibration_error'; offending attribute(s): model)

dill is slower on every save, and it serialises BYTECODE -- a model written under one interpreter or one mlframe
version may not load under another, which matters precisely when training and serving are separate environments.

These classes carry the same configuration as instance attributes and compute exactly the same numbers; the
metric maths is untouched. Being module-level with plain-data state, they pickle by reference like any other
class instance. ``functools.partial`` was not an option: xgboost inspects the callable it is handed, which is
what the original code's "partial won't work with xgboost" note refers to; a callable OBJECT presents the
signature of its ``__call__`` and is accepted.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class IntegralCalibrationError:
    """``compute_probabilistic_multiclass_error`` bound to one suite's calibration configuration."""
    # xgboost reads ``eval_metric.__name__`` to label the metric and raises AttributeError without it -- the
    # same reason the original code noted that functools.partial "won't work with xgboost". A class attribute is enough: the type keeps its own __name__.
    __name__ = "integral_calibration_error"

    __slots__ = (
        "brier_loss_weight",
        "mae_weight",
        "method",
        "min_roc_auc",
        "nbins",
        "pr_auc_weight",
        "roc_auc_penalty",
        "roc_auc_weight",
        "std_weight",
        "use_weighted_calibration",
        "weight_by_class_npositives",
    )

    def __init__(
        self,
        *,
        method: Any,
        mae_weight: float,
        std_weight: float,
        brier_loss_weight: float,
        roc_auc_weight: float,
        pr_auc_weight: float,
        min_roc_auc: float,
        roc_auc_penalty: float,
        use_weighted_calibration: bool,
        weight_by_class_npositives: bool,
        nbins: int,
    ) -> None:
        """Store the configuration the metric is evaluated under."""
        self.method = method
        self.mae_weight = mae_weight
        self.std_weight = std_weight
        self.brier_loss_weight = brier_loss_weight
        self.roc_auc_weight = roc_auc_weight
        self.pr_auc_weight = pr_auc_weight
        self.min_roc_auc = min_roc_auc
        self.roc_auc_penalty = roc_auc_penalty
        self.use_weighted_calibration = use_weighted_calibration
        self.weight_by_class_npositives = weight_by_class_npositives
        self.nbins = nbins

    def _kwargs(self) -> dict:
        """The configuration as the keyword arguments the underlying metric takes."""
        return {
            "method": self.method,
            "mae_weight": self.mae_weight,
            "std_weight": self.std_weight,
            "brier_loss_weight": self.brier_loss_weight,
            "roc_auc_weight": self.roc_auc_weight,
            "pr_auc_weight": self.pr_auc_weight,
            "min_roc_auc": self.min_roc_auc,
            "roc_auc_penalty": self.roc_auc_penalty,
            "use_weighted_calibration": self.use_weighted_calibration,
            "weight_by_class_npositives": self.weight_by_class_npositives,
            "nbins": self.nbins,
        }

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, verbose: bool = False) -> float:
        """Integral calibration error for probabilistic predictions; lower is better."""
        from mlframe.metrics.core import compute_probabilistic_multiclass_error

        err = compute_probabilistic_multiclass_error(y_true=y_true, y_score=y_score, verbose=verbose, **self._kwargs())
        if verbose:
            logger.debug("integral_calibration_error=%s (n=%d)", err, len(y_true))
        return float(err)

    # ``__getstate__``/``__setstate__`` are spelled out because ``__slots__`` classes have no ``__dict__`` for
    # pickle's default protocol-2 path to copy.
    def __getstate__(self) -> dict:
        """State as a plain dict, so a slotted instance pickles."""
        return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        """Restore from the dict produced by ``__getstate__``."""
        for name, value in state.items():
            setattr(self, name, value)

    def __repr__(self) -> str:
        """Readable in a params dump."""
        return f"IntegralCalibrationError(method={self.method!r}, nbins={self.nbins})"


class PositionalIntegralCalibrationError(IntegralCalibrationError):
    """Positional-args variant for feature-selection / HPT callers that pass ``y_true``/``y_score`` positionally.

    It also defaults ``verbose`` to True, matching the closure it replaces.
    """

    __slots__ = ()

    def __call__(self, *args, verbose: bool = True, **kwargs):
        """Same metric, called the way the FS / HPT code calls it: y_true / y_score arrive positionally."""
        from mlframe.metrics.core import compute_probabilistic_multiclass_error

        return compute_probabilistic_multiclass_error(
            *args,
            **kwargs,  # type: ignore[misc]  # the FS / HPT callers pass y_true / y_score positionally, never by these names
            verbose=verbose,
            **self._kwargs(),
        )


class SubgroupAveragedMetric:
    """``robust_mlperf_metric`` applied to a metric across fixed subgroups."""
    # xgboost reads ``eval_metric.__name__`` to label the metric and raises AttributeError without it -- the
    # same reason the original code noted that functools.partial "won't work with xgboost". Wrappers keep the inner metric name so logs stay comparable.
    __name__ = "integral_calibration_error"

    __slots__ = ("higher_is_better", "metric", "subgroups")

    def __init__(self, metric: Callable, subgroups: Any, *, higher_is_better: bool = False) -> None:
        """Bind the inner metric and the subgroup definition."""
        self.metric = metric
        self.subgroups = subgroups
        self.higher_is_better = higher_is_better

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):
        """Average the inner metric across the subgroups."""
        from mlframe.metrics.core import robust_mlperf_metric

        return robust_mlperf_metric(
            y_true,
            y_score,
            *args,
            metric=self.metric,
            higher_is_better=self.higher_is_better,
            subgroups=self.subgroups,
            **kwargs,  # type: ignore[misc]  # the xgboost / lgbm eval callback never passes these names in kwargs
        )

    def __getstate__(self) -> dict:
        """State as a plain dict, so a slotted instance pickles."""
        return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        """Restore from the dict produced by ``__getstate__``."""
        for name, value in state.items():
            setattr(self, name, value)


class RobustTimeSplitMetric:
    """A metric evaluated on consecutive time splits and combined as ``mean +/- std * std_coeff``.

    Falls back to the full-data metric when there is not enough data, or not enough valid splits, for the
    robustness estimate to mean anything.
    """
    # xgboost reads ``eval_metric.__name__`` to label the metric and raises AttributeError without it -- the
    # same reason the original code noted that functools.partial "won't work with xgboost". Wrappers keep the inner metric name so logs stay comparable.
    __name__ = "integral_calibration_error"

    __slots__ = ("ensure_enough_classes", "greater_is_better", "metric_fn", "min_samples_per_split", "num_splits", "std_coeff", "verbose")

    def __init__(
        self,
        metric_fn: Callable,
        num_splits: int,
        std_coeff: float,
        greater_is_better: bool,
        min_samples_per_split: int = 100,
        ensure_enough_classes: bool = False,
        verbose: int = 0,
    ) -> None:
        """Bind the inner metric and the split policy."""
        self.metric_fn = metric_fn
        self.num_splits = num_splits
        self.std_coeff = std_coeff
        self.greater_is_better = greater_is_better
        self.min_samples_per_split = min_samples_per_split
        self.ensure_enough_classes = ensure_enough_classes
        self.verbose = verbose

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, *args, **kwargs):
        """Mean of the per-split metric, penalised by its spread."""
        n = len(y_true)
        if n < self.min_samples_per_split:
            if self.verbose:
                logger.info("RobustTimeSplitMetric: n=%s < min_samples_per_split=%s, using full data", n, self.min_samples_per_split)
            return self.metric_fn(y_true, y_score, *args, **kwargs)

        actual_splits = min(self.num_splits, n // self.min_samples_per_split)
        if actual_splits <= 1:
            if self.verbose:
                logger.info("RobustTimeSplitMetric: actual_splits=%s <= 1, using full data", actual_splits)
            return self.metric_fn(y_true, y_score, *args, **kwargs)

        split_size = n // actual_splits
        values: list = []
        for i in range(actual_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < actual_splits - 1 else n
            y_true_split = y_true[start_idx:end_idx]
            y_score_split = y_score[start_idx:end_idx]
            if len(y_true_split) < self.min_samples_per_split:
                if self.verbose:
                    logger.info("RobustTimeSplitMetric: split %s skipped, len=%d < %d", i, len(y_true_split), self.min_samples_per_split)
                continue
            if self.ensure_enough_classes and len(np.unique(y_true_split)) < 2:
                if self.verbose:
                    logger.info("RobustTimeSplitMetric: split %s skipped, single class in y_true", i)
                continue
            val = self.metric_fn(y_true_split, y_score_split, *args, **kwargs)
            if not np.isnan(val):
                values.append(val)

        if not values:
            if self.verbose:
                logger.info("RobustTimeSplitMetric: no valid splits, using full data")
            return self.metric_fn(y_true, y_score, *args, **kwargs)

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        # Penalise variance in whichever direction is "worse" for this metric.
        return mean_val - std_val * self.std_coeff if self.greater_is_better else mean_val + std_val * self.std_coeff

    def __getstate__(self) -> dict:
        """State as a plain dict, so a slotted instance pickles."""
        return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        """Restore from the dict produced by ``__getstate__``."""
        for name, value in state.items():
            setattr(self, name, value)


class LightGBMMetricAdapter:
    """Adapts a plain metric to LightGBM's custom-metric contract: ``(name, value, higher_is_better)``."""
    # xgboost reads ``eval_metric.__name__`` to label the metric and raises AttributeError without it -- the
    # same reason the original code noted that functools.partial "won't work with xgboost". LightGBM names the metric from the returned tuple, but keep it consistent.
    __name__ = "lgbm_integral_calibration_error"

    __slots__ = ("higher_is_better", "metric", "metric_name")

    def __init__(self, metric: Callable, metric_name: str = "integral_calibration_error", higher_is_better: bool = False) -> None:
        """Bind the metric and the name LightGBM will print."""
        self.metric = metric
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

    def __call__(self, y_true, y_score):
        """LightGBM's expected 3-tuple."""
        return self.metric_name, self.metric(y_true, y_score), self.higher_is_better

    def __getstate__(self) -> dict:
        """State as a plain dict, so a slotted instance pickles."""
        return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        """Restore from the dict produced by ``__getstate__``."""
        for name, value in state.items():
            setattr(self, name, value)


def build_robust_ts_metric(
    metric_fn: Callable,
    num_splits: int,
    std_coeff: float,
    greater_is_better: bool,
    min_samples_per_split: int = 100,
    ensure_enough_classes: bool = False,
    verbose: int = 0,
) -> Callable:
    """Backwards-compatible constructor mirroring the old ``make_robust_ts_metric`` factory."""
    return RobustTimeSplitMetric(
        metric_fn=metric_fn, num_splits=num_splits, std_coeff=std_coeff, greater_is_better=greater_is_better,
        min_samples_per_split=min_samples_per_split, ensure_enough_classes=ensure_enough_classes, verbose=verbose,
    )


__all__ = [
    "IntegralCalibrationError",
    "LightGBMMetricAdapter",
    "PositionalIntegralCalibrationError",
    "RobustTimeSplitMetric",
    "SubgroupAveragedMetric",
    "build_robust_ts_metric",
]
