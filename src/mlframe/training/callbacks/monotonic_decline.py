"""Monotonic strict-decline overfitting-stop callbacks for LightGBM and XGBoost.

Both wrap the shared, dependency-free ``MonotonicDeclineStopper`` so the rule is byte-identical
to the EarlyStoppingWrapper / lightning paths: stop once the monitored eval metric strictly
worsens for ``patience`` consecutive boosting rounds since the global best (a confident
overfitting signal). COMPLEMENTARY to each booster's native ``early_stopping_rounds`` patience --
whichever fires first wins; the native best-iteration bookkeeping still selects the global best.

Direction is auto-derived per metric via ``metrics_registry.metric_name_higher_is_better`` (with a
``mode=`` override for unknown metric names), so AUC-style (max) and loss/error-style (min) metrics
both work without per-call wiring.

CatBoost
--------
CatBoost does NOT host a Python per-iteration callback that can both observe the val metric AND
veto continuation through the public sklearn ``fit`` API (its callback hook is limited and the
od_type / overfitting-detector machinery is native C++). Implementing this for CatBoost would need
a native-API change, so it is intentionally NOT provided here -- mlframe's CatBoost path keeps its
native overfitting detector + the budget-scaled worsening detector already in
``UniversalCallback`` (``training.callbacks._callbacks``). See the project report for the verdict.
"""
from __future__ import annotations

import logging
from typing import Optional

from mlframe.estimators.early_stopping_monotonic import MonotonicDeclineStopper

logger = logging.getLogger(__name__)


def _resolve_mode(metric_name: str, mode: Optional[str]) -> str:
    """Pick the optimization direction for a metric name (explicit ``mode`` wins)."""
    if mode in ("min", "max"):
        return mode
    from ..metrics_registry import metric_name_higher_is_better

    direction = metric_name_higher_is_better(metric_name)
    if direction is True:
        return "max"
    if direction is False:
        return "min"
    # Unknown metric: default to min (the safe boosting-loss assumption) but warn so the caller
    # can register the metric or pass mode= explicitly.
    logger.warning(
        "monotonic-decline callback: unknown direction for metric=%r; defaulting to mode='min'. "
        "Pass mode= explicitly or register the metric in metrics_registry.", metric_name,
    )
    return "min"


class LGBMonotonicDeclineStop:
    """LightGBM callback: stop on a fixed-N monotone strict-decline run of the monitored eval metric.

    Reads ``env.evaluation_result_list`` each round, picks the value for ``monitor_dataset`` /
    ``monitor_metric`` (defaults: first registered valid set, first metric), feeds it to the shared
    stopper, and raises ``lgb.callback.EarlyStopException`` (the native stop mechanism) once the
    streak fires -- so LightGBM rolls the booster back to the recorded best iteration.
    """

    # LightGBM orders callbacks by ``order``; run after metric eval (mirrors native early_stopping).
    order = 30
    before_iteration = False

    def __init__(self, patience: Optional[int] = 3, monitor_dataset: Optional[str] = None,
                 monitor_metric: Optional[str] = None, mode: Optional[str] = None) -> None:
        self.patience = patience
        self.monitor_dataset = monitor_dataset
        self.monitor_metric = monitor_metric
        self._mode = mode
        self._stopper: Optional[MonotonicDeclineStopper] = None
        self._best_iter = 0
        self._best_value: Optional[float] = None

    def __call__(self, env) -> None:
        import lightgbm as lgb

        results = env.evaluation_result_list
        if not results:
            return
        # Each entry: (dataset_name, metric_name, value, is_higher_better[, stdv]).
        ds = self.monitor_dataset or results[0][0]
        mt = self.monitor_metric or results[0][1]
        value = None
        for entry in results:
            if entry[0] == ds and entry[1] == mt:
                value = float(entry[2])
                break
        if value is None:
            return
        if self._stopper is None:
            self._stopper = MonotonicDeclineStopper(self.patience, mode=_resolve_mode(mt, self._mode))
            if not self._stopper.enabled:
                return
            self._best_value = value
        # Track best iteration so the EarlyStopException rolls back to it.
        is_better = (
            self._best_value is None
            or (value > self._best_value if self._stopper.mode == "max" else value < self._best_value)
        )
        if is_better:
            self._best_value = value
            self._best_iter = env.iteration
        if self._stopper.update(value):
            logger.info(
                "[lgb monotonic-decline] stopping at iteration %d: %s/%s strictly worsened for %d "
                "consecutive rounds since best @%d.", env.iteration, ds, mt, self._stopper.streak,
                self._best_iter,
            )
            raise lgb.callback.EarlyStopException(self._best_iter, [(ds, mt, self._best_value, self._stopper.mode == "max")])


def _make_xgb_monotonic_callback(patience: Optional[int] = 3, monitor_dataset: Optional[str] = None,
                                 monitor_metric: Optional[str] = None, mode: Optional[str] = None):
    """Build an XGBoost ``TrainingCallback`` subclass instance for the monotone strict-decline stop.

    Factory (not a module-level class) so ``xgboost`` is imported lazily -- the module stays
    importable when xgboost is absent. Returns ``None`` when xgboost is unavailable or patience
    disables the detector.
    """
    if patience is None or int(patience) <= 0:
        return None
    try:
        import xgboost as xgb
    except ImportError:
        return None

    class _XGBMonotonicDeclineStop(xgb.callback.TrainingCallback):
        _is_mlframe_monotonic_decline = True  # sentinel for de-dup at the shim wiring site

        def __init__(self) -> None:
            super().__init__()
            self._stopper: Optional[MonotonicDeclineStopper] = None
            self._mode = mode
            self._monitor_dataset = monitor_dataset
            self._monitor_metric = monitor_metric

        def after_iteration(self, model, epoch, evals_log) -> bool:
            if not evals_log:
                return False
            ds = self._monitor_dataset or next(iter(evals_log))
            ds_log = evals_log.get(ds)
            if not ds_log:
                return False
            mt = self._monitor_metric or next(iter(ds_log))
            series = ds_log.get(mt)
            if not series:
                return False
            value = float(series[-1])
            if self._stopper is None:
                self._stopper = MonotonicDeclineStopper(patience, mode=_resolve_mode(mt, self._mode))
                if not self._stopper.enabled:
                    return False
            if self._stopper.update(value):
                logger.info(
                    "[xgb monotonic-decline] stopping at iteration %d: %s/%s strictly worsened for %d "
                    "consecutive rounds since best.", epoch, ds, mt, self._stopper.streak,
                )
                return True  # XGBoost stops training; best_iteration bookkeeping already tracked natively.
            return False

    return _XGBMonotonicDeclineStop()


__all__ = ["LGBMonotonicDeclineStop", "_make_xgb_monotonic_callback", "_resolve_mode"]
