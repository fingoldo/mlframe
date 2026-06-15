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
Modern CatBoost (verified against catboost 1.2.10) DOES host a usable Python per-iteration callback:
``CatBoostClassifier/Regressor.fit(..., callbacks=[obj])`` where ``obj.after_iteration(info)`` returns
``False`` to STOP and ``True`` to continue. ``info.metrics`` exposes the per-iteration eval scores keyed
by dataset (``"learn"`` / ``"validation"``) then metric name. ``CBMonotonicDeclineStop`` wraps the same
shared ``MonotonicDeclineStopper`` so CatBoost stops on the byte-identical rule as lgb / xgb / mlp. When
the installed CatBoost build lacks ``callbacks=`` support (probed at wiring time), the detector is skipped
and CatBoost's native overfitting detector (``od_wait``) plus the budget-scaled worsening detector in
``UniversalCallback`` remain the stop signal.
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


def catboost_callbacks_supported() -> bool:
    """Runtime probe: does the installed CatBoost accept a Python ``callbacks=`` arg in ``fit``?

    Probed on the ``fit`` signature rather than a version-string compare so a future build that adds or
    removes the hook is detected directly. Returns False when catboost is absent. Mirrors the
    capability-gate style of ``lgb_dataset_reuse_capable`` / ``xgb_dmatrix_reuse_capable``.
    """
    try:
        import catboost
        import inspect
    except ImportError:
        return False
    try:
        return "callbacks" in inspect.signature(catboost.CatBoostClassifier.fit).parameters
    except (TypeError, ValueError):
        return False


class CBMonotonicDeclineStop:
    """CatBoost callback: stop on a fixed-N monotone strict-decline run of the monitored eval metric.

    CatBoost calls ``after_iteration(info)`` once per boosting iteration with ``info.metrics`` shaped
    ``{dataset_name: {metric_name: [values...]}}`` (dataset keys ``"learn"`` / ``"validation"``). We read
    the monitored value (defaults: ``"validation"`` set, first metric), feed it to the shared stopper, and
    return ``False`` to stop once the streak fires -- CatBoost then ends training, keeping its native
    best-iteration bookkeeping (``best_iteration_`` via ``od_wait``/``use_best_model``) for model selection.
    Returning ``True`` continues training; a disabled detector always returns ``True``.
    """

    def __init__(self, patience: Optional[int] = 3, monitor_dataset: Optional[str] = None,
                 monitor_metric: Optional[str] = None, mode: Optional[str] = None) -> None:
        self.patience = patience
        self.monitor_dataset = monitor_dataset
        self.monitor_metric = monitor_metric
        self._mode = mode
        self._stopper: Optional[MonotonicDeclineStopper] = None

    def after_iteration(self, info) -> bool:
        metrics = getattr(info, "metrics", None)
        if not metrics:
            return True
        ds = self.monitor_dataset if self.monitor_dataset in metrics else (
            "validation" if "validation" in metrics else next(iter(metrics))
        )
        ds_metrics = metrics.get(ds)
        if not ds_metrics:
            return True
        mt = self.monitor_metric if (self.monitor_metric in ds_metrics) else next(iter(ds_metrics))
        series = ds_metrics.get(mt)
        if not series:
            return True
        value = float(series[-1])
        if self._stopper is None:
            self._stopper = MonotonicDeclineStopper(self.patience, mode=_resolve_mode(mt, self._mode))
            if not self._stopper.enabled:
                return True
        if self._stopper.update(value):
            logger.info(
                "[cb monotonic-decline] stopping: %s/%s strictly worsened for %d consecutive iters "
                "since best (confident overfitting).", ds, mt, self._stopper.streak,
            )
            return False  # CatBoost stops; best-iteration bookkeeping handled natively.
        return True


__all__ = [
    "LGBMonotonicDeclineStop",
    "_make_xgb_monotonic_callback",
    "CBMonotonicDeclineStop",
    "catboost_callbacks_supported",
    "_resolve_mode",
]
