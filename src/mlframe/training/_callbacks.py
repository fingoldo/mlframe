"""Callback classes for Universal/LightGBM/XGBoost/CatBoost."""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import TrainingCallback

from pyutilz.pythonlib import get_parent_func_args, store_params_in_object
from pyutilz.system import get_own_memory_usage

logger = logging.getLogger(__name__)

class UniversalCallback:
    def __init__(
        self,
        time_budget_mins: float | None = None,
        reporting_interval_mins: float | None = 1.0,
        patience: int | None = None,
        min_delta: float = 0.0,
        monitor_dataset: str | None = None,
        monitor_metric: str | None = None,
        mode: str | None = None,
        stop_flag: Callable[[], bool] | None = None,
        ndigits: int = 6,
        verbose: int = 1,
    ) -> None:

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

        self.start_time = None
        self.best_metric = None
        self.first_iteration = True
        self.iterations_since_improvement = 0
        self.metric_history: dict[str, dict[str, list[float]]] = {}
        self.stop_flag = stop_flag if stop_flag is not None else lambda: False

        # Call super().__init__() to ensure proper MRO chain initialization.
        # For XGBoostCallback(UniversalCallback, TrainingCallback), this calls
        # TrainingCallback.__init__(), which is required by XGBoost >= 2.x ABC checks.
        super().__init__()

        if self.verbose > 0:
            logger.info(
                "UniversalCallback initialized with params: "
                "time_budget_mins=%s, patience=%s, min_delta=%s, "
                "monitor_dataset=%s, monitor_metric=%s, mode=%s",
                time_budget_mins, patience, min_delta,
                monitor_dataset, monitor_metric, mode,
            )

    def on_start(self) -> None:
        self.start_time = timer()
        if self.verbose > 0:
            self.last_reporting_ts = self.start_time
            logger.info("Training started. Timer initiated. RAM usage %.1fGB.", get_own_memory_usage())

    def update_history(self, metrics_dict: dict[str, dict[str, float]]) -> None:
        for dataset in metrics_dict:
            if dataset not in self.metric_history:
                self.metric_history[dataset] = {}
            for metric, value in metrics_dict[dataset].items():
                self.metric_history[dataset].setdefault(metric, []).append(value)
        if self.verbose > 1:
            logger.debug("Updated metric history: %s", metrics_dict)

    def derive_mode(self, metric_name: str) -> str:
        known_metric_modes = {
            "auc": "max",
            "accuracy": "max",
            "acc": "max",
            "f1": "max",
            "map": "max",
            "ndcg": "max",
            "ice": "min",
            "mae": "min",
            "mse": "min",
            "mape": "min",
            "rmse": "min",
            "logloss": "min",
            "error": "min",
            "loss": "min",
        }

        name = metric_name.lower()
        for key, default_mode in known_metric_modes.items():
            if key == name:
                return default_mode
        if "score" in name or "auc" in name or "accuracy" in name:
            return "max"
        elif "loss" in name or "error" in name:
            return "min"
        elif name.endswith("e"):
            return "min"
        else:
            logger.warning(f"Unsure about correct optimization mode for metric={name}, using min for now.")
            return "min"  # fallback default

    def set_default_monitor_metric(self, metrics_dict: dict[str, dict[str, float]]) -> None:
        if self.monitor_dataset not in metrics_dict:
            raise ValueError(f"Monitor dataset '{self.monitor_dataset}' not found in metrics.")
        available_metrics = list(metrics_dict[self.monitor_dataset].keys())
        logger.info("available_metrics=%s", available_metrics)
        for preferred in ["ICE", "integral_calibration_error", "auc", "AUC"]:
            if preferred in available_metrics:
                self.monitor_metric = preferred
                break
        else:
            self.monitor_metric = available_metrics[0]
        self.mode = self.derive_mode(self.monitor_metric)
        if self.verbose > 0:
            logger.info("Auto-selected monitor_metric: %s, mode: %s", self.monitor_metric, self.mode)

    def _get_state(self, current_value: float) -> str:
        return f"iter={self.iter:_}, {self.monitor_dataset} {self.monitor_metric}: current={current_value:.{self.ndigits}f}, best={self.best_metric:.{self.ndigits}f} @{self.best_iter:_}. RAM usage {get_own_memory_usage():.1f}GB."

    def should_stop(self) -> bool:
        cur_ts = timer()
        if self.time_budget_mins is not None and self.start_time is not None:

            elapsed = cur_ts - self.start_time
            if elapsed > self.time_budget_mins * 60:
                if self.verbose > 0:
                    logger.info("Stopping early due to time budget exceeded (%.2f sec).", elapsed)
                return True

        if self.stop_flag():
            if self.verbose > 0:
                logger.info("Stopping early due to external stop flag.")
            return True

        if self.monitor_dataset in self.metric_history and self.monitor_metric in self.metric_history[self.monitor_dataset]:
            history = self.metric_history[self.monitor_dataset][self.monitor_metric]
            if history:
                current_value = history[-1]
                if self.best_metric is None:
                    self.iter = 0
                    self.best_iter = self.iter
                    self.best_metric = current_value
                    self.iterations_since_improvement = 0
                    if self.verbose > 0:
                        logger.info("Initial metric value: %.*f", self.ndigits, current_value)
                        self.last_reporting_ts = cur_ts
                else:
                    self.iter += 1
                    improved = (self.mode == "min" and current_value < self.best_metric - self.min_delta) or (
                        self.mode == "max" and current_value > self.best_metric + self.min_delta
                    )
                    # Pre-compute reporting condition (used in both branches)
                    should_report = self.verbose > 0 and (
                        not self.reporting_interval_mins or (cur_ts - self.last_reporting_ts) >= self.reporting_interval_mins * 60
                    )
                    if improved:
                        self.best_iter = self.iter
                        self.best_metric = current_value
                        self.iterations_since_improvement = 0
                    else:
                        self.iterations_since_improvement += 1
                    if should_report:
                        logger.info(self._get_state(current_value=current_value))
                        self.last_reporting_ts = cur_ts
                    if self.patience is not None and self.iterations_since_improvement >= self.patience:
                        if self.verbose > 0:
                            logger.info(
                                "Stopping early due to no improvement for %d iterations. %s",
                                self.iterations_since_improvement, self._get_state(current_value=current_value),
                            )
                            self.last_reporting_ts = cur_ts
                        return True
        return False


class LightGBMCallback(UniversalCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "valid_0"

    def __call__(self, env: lgb.callback.CallbackEnv) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False

        metrics_dict = {}
        for dataset, metric, value, _ in env.evaluation_result_list:
            metrics_dict.setdefault(dataset, {})[metric] = value
        self.update_history(metrics_dict)

        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)
        if self.should_stop():
            if hasattr(self, "best_iter"):
                best_iter = self.best_iter
            else:
                best_iter = 0
            raise lgb.callback.EarlyStopException(best_iter, [(dataset, metric, self.best_metric, False)])


class XGBoostCallback(UniversalCallback, TrainingCallback):
    # XGBoost >= 2.x CallbackContainer rejects callbacks that aren't isinstance
    # of TrainingCallback. The MRO `(UniversalCallback, TrainingCallback)` lets
    # `super().__init__()` inside UniversalCallback chain into
    # TrainingCallback.__init__() without us having to call it explicitly.
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "validation_0"

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict[str, list[float]]]) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False
        metrics_dict = {dataset: {metric: values[-1] for metric, values in metric_dict.items()} for dataset, metric_dict in evals_log.items()}

        self.update_history(metrics_dict)

        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)

        if self.should_stop():
            if hasattr(self, "best_iter"):
                best_iter = self.best_iter
            else:
                best_iter = 0
            model.set_attr(best_score=self.best_metric, best_iteration=best_iter)
            return True


class CatBoostCallback(UniversalCallback):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.monitor_dataset = self.monitor_dataset or "validation"

    def after_iteration(self, info: Any) -> bool:
        if self.first_iteration:
            self.on_start()
            self.first_iteration = False

        metrics_dict = {dataset: {metric: values[-1] for metric, values in metric_dict.items()} for dataset, metric_dict in info.metrics.items()}
        self.update_history(metrics_dict)

        if self.monitor_metric is None:
            self.set_default_monitor_metric(metrics_dict)
        return not self.should_stop()


__all__ = [
    "UniversalCallback",
    "LightGBMCallback",
    "XGBoostCallback",
    "CatBoostCallback",
]
