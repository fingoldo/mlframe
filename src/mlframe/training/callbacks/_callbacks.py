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
        # Slice-stable ES knobs. When ``slice_k > 0`` the callback aggregates per-shard scores
        # over the first ``slice_k`` registered eval datasets OTHER than ``monitor_dataset``,
        # then applies the patience / min_delta logic to the aggregate instead of to the single
        # monitor_dataset value. ``slice_k=0`` (default) keeps the legacy single-val path
        # bit-identical. ``slice_dataset_names`` is optional: when provided, those exact names
        # are used; when None, the callback picks the first ``slice_k`` non-monitor dataset keys
        # in insertion order at first iteration (works for LGB ``valid_*``, XGB ``validation_*``,
        # CB ``validation_*`` without booster-specific glue).
        slice_k: int = 0,
        slice_dataset_names: list[str] | None = None,
        slice_aggregate_mode: str = "t_lcb",
        slice_aggregate_alpha: float = 1.0,
        slice_aggregate_confidence: float = 0.9,
        slice_aggregate_quantile_level: float = 0.9,
        slice_correlation_inflation: float = 1.5,
        slice_min_delta_in_se: float | None = None,
        slice_persist_history: bool = False,
        # When ``slice_diagnostic_only=True``, the callback still walks the per-shard history
        # and populates ``slice_shard_score_history`` so the Pareto-plot artefact has data to
        # render, but the stop decision keeps reading the single full-val metric (no aggregator
        # math, no patience bump). Used for shipping per-shard diagnostics without changing the
        # ES behaviour. Ignored when ``slice_k == 0``.
        slice_diagnostic_only: bool = False,
        # Curve-shape ES detector (see TrainingBehaviorConfig docstring). Forward-looking
        # complement to patience: when a strict-monotone-worsening run since the best iter
        # has lasted ``max(max_iter // worsening_coeff, worsening_min_iters)`` iterations,
        # stop. ``worsening_max_iter`` is the booster's iteration budget; when None we
        # fall back to ``worsening_min_iters`` as the threshold (conservative).
        worsening_enabled: bool = True,
        worsening_coeff: int = 5,
        worsening_min_iters: int = 5,
        worsening_max_iter: int | None = None,
    ) -> None:

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

        self.start_time = None
        self.best_metric = None
        self.first_iteration = True
        self.iterations_since_improvement = 0
        self.metric_history: dict[str, dict[str, list[float]]] = {}
        self.stop_flag = stop_flag if stop_flag is not None else lambda: False
        # Slice-stable bookkeeping: aggregated score at every iteration, per-shard table for the
        # Pareto plot (only kept when ``slice_persist_history=True`` to keep the no-slice path's
        # memory profile unchanged).
        self.slice_aggregate_history: list[float] = []
        self.slice_shard_score_history: list[list[float]] = []
        self.slice_resolved_dataset_names: list[str] | None = None

        # Curve-shape detector state: count of consecutive iterations since the best that
        # produced a STRICTLY worse value than their immediate predecessor (no improvement
        # over the previous iter in the is_greater_better direction). Resets to 0 the moment
        # any successor improves on its predecessor, even if it's not a new best.
        self._worsening_streak_len: int = 0
        self._worsening_last_value: float | None = None
        self._worsening_stopped: bool = False
        # Tracks whether the booster ran to (almost) its full iteration budget without our
        # ES ever firing. When True at finalize time, the val curve was still improving at
        # the budget cap -- val essentially didn't get used as a stop signal, so the
        # operator could plausibly retrain on train+val with the same (or doubled) budget.
        # See ``maybe_warn_max_iter_hit()`` for the diagnostic + TODO log message.
        # TODO(train+val refit): when best_iter >= max_iter-1, val was effectively a
        # held-out scorer with no stopping role. Two follow-ups for a future change:
        #   1. Refit on train+val at the SAME max_iter (most common pattern, low risk,
        #      see classical "refit on full data after CV" / GridSearchCV.refit=True).
        #   2. partial_fit-on-val: continue the existing fitted booster on the val rows
        #      for a few extra iters (preserves the trained state, no train-val refit
        #      needed). Requires booster.partial_fit / xgb_train(xgb_model=...) support.
        # Neither is implemented yet -- this comment is the placeholder for that work.

        # Patience auto-bump compensates for the larger variance of the dispersion-penalised
        # aggregate; helper lives next to the shard builder so the two stay in lock-step.
        # Skip the bump when slice mode is diagnostic-only (single-val drives ES, no penalty
        # to compensate for).
        if slice_k and patience is not None and not slice_diagnostic_only:
            from .._slice_helpers import effective_patience as _eff_pat
            bumped = _eff_pat(int(patience), int(slice_k))
            if bumped != patience and self.verbose > 0:
                logger.info(
                    "UniversalCallback: slice-stable ES patience auto-bumped from %d to %d (K=%d)",
                    patience, bumped, slice_k,
                )
            self.patience = bumped

        # Call super().__init__() to ensure proper MRO chain initialization.
        # For XGBoostCallback(UniversalCallback, TrainingCallback), this calls
        # TrainingCallback.__init__(), which is required by XGBoost >= 2.x ABC checks.
        super().__init__()

        if self.verbose > 0:
            logger.info(
                "UniversalCallback initialized with params: "
                "time_budget_mins=%s, patience=%s, min_delta=%s, "
                "monitor_dataset=%s, monitor_metric=%s, mode=%s, "
                "slice_k=%s, slice_aggregate_mode=%s",
                time_budget_mins, self.patience, min_delta,
                monitor_dataset, monitor_metric, mode,
                slice_k, slice_aggregate_mode if slice_k else None,
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
        """Wave 20 fix: delegate to ``metric_name_higher_is_better`` instead
        of the prior heuristic ladder that ended in ``endswith("e") -> "min"``.

        The old fallback silently classified custom metric names (``gini``,
        ``kappa``, ``r2``, ``accuracy_score``, ``pr_auc``, etc.) as ``"min"``
        because they didn't match the hard-coded substring rules. ``should_stop()``
        then early-stopped at the WORST iteration. This is a P0 (drives actual
        training, not just reporting).

        Unknown metric names now WARN and default to ``"min"`` only as a
        deliberate fallback -- the caller can override via the configured
        ``mode=`` kwarg if they pass a custom metric the registry doesn't
        know about.
        """
        from ..metrics_registry import metric_name_higher_is_better
        direction = metric_name_higher_is_better(metric_name)
        if direction is True:
            return "max"
        if direction is False:
            return "min"
        # Genuinely unknown -- WARN loudly with names of both built-in tables
        # so the operator either registers the metric or passes mode= explicitly.
        logger.warning(
            "derive_mode: cannot determine optimization direction for metric=%r. "
            "Register it via mlframe.training.metrics_registry.register_metric "
            "or pass mode='min'/'max' explicitly. Falling back to 'min' which "
            "may cause early-stop at the WORST iteration if the metric is "
            "actually higher-is-better.", metric_name,
        )
        return "min"  # explicit-fallback default

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

    def _resolve_slice_dataset_names(self) -> list[str]:
        """Decide which dataset keys feed the slice aggregate.

        Caller may have specified explicit names; otherwise we pick the first ``slice_k`` keys
        OTHER than ``monitor_dataset`` from ``metric_history`` in insertion order (which equals
        registration order on Python 3.7+ dicts). Works uniformly for LGB ``valid_1..K``, XGB
        ``validation_1..K``, and CB ``validation_1..K`` without booster-specific name plumbing.
        """
        if self.slice_resolved_dataset_names is not None:
            return self.slice_resolved_dataset_names
        if self.slice_dataset_names:
            self.slice_resolved_dataset_names = list(self.slice_dataset_names)
            return self.slice_resolved_dataset_names
        other = [name for name in self.metric_history.keys() if name != self.monitor_dataset]
        self.slice_resolved_dataset_names = other[: int(self.slice_k)]
        return self.slice_resolved_dataset_names

    def _compute_slice_aggregate(self) -> float | None:
        """Per-iteration aggregated score from the K shards, or None when shards aren't ready yet.

        Returns ``None`` (skip aggregation this iteration) when at least one shard hasn't pushed
        a fresh value into history -- this is the CB ``metric_period > 1`` guard: phantom values
        on intermediate iterations would give std=0 and a spurious "stable" signal.
        """
        from .._cv_aggregation import aggregate_fold_scores

        names = self._resolve_slice_dataset_names()
        if len(names) < 2:
            return None
        # Use monitor_dataset's history length as the canonical iteration count.
        monitor_hist = self.metric_history.get(self.monitor_dataset, {}).get(self.monitor_metric, [])
        n_iters = len(monitor_hist)
        if n_iters == 0:
            return None
        shard_values: list[float] = []
        for name in names:
            shard_hist = self.metric_history.get(name, {}).get(self.monitor_metric, [])
            if len(shard_hist) < n_iters:
                # CB metric_period>1 or per-iter eval not run on this shard -> skip aggregation
                # this round; we'll try again next call.
                return None
            shard_values.append(float(shard_hist[-1]))
        if self.slice_persist_history:
            self.slice_shard_score_history.append(list(shard_values))
        direction = "min" if self.mode == "min" else "max"
        agg = aggregate_fold_scores(
            shard_values,
            mode=self.slice_aggregate_mode,  # type: ignore[arg-type]
            direction=direction,
            alpha=self.slice_aggregate_alpha,
            confidence=self.slice_aggregate_confidence,
            quantile_level=self.slice_aggregate_quantile_level,
            correlation_inflation=self.slice_correlation_inflation,
        )
        self.slice_aggregate_history.append(agg)
        return agg

    def maybe_warn_max_iter_hit(self) -> None:
        """Log a WARN when the booster ran to its full budget without ES firing.

        Indicates the val curve was still improving at the cap -- val essentially served
        only as a held-out scorer with no stopping role. Two operator-facing follow-ups
        (see TODO in __init__): refit on train+val at the same max_iter, OR partial_fit
        the existing booster on val rows for a few more iters. Neither auto-applies; the
        warning surfaces the diagnostic so the operator can decide.
        """
        if self.worsening_max_iter is None or self.worsening_max_iter <= 0:
            return
        if not hasattr(self, "best_iter") or self.best_iter is None:
            return
        if self._worsening_stopped:
            return  # we stopped via curve-shape detector; budget wasn't exhausted
        # ``best_iter`` is 0-indexed; "hit the budget" means within 1 of the cap.
        if int(self.best_iter) >= int(self.worsening_max_iter) - 1:
            logger.warning(
                "best_iter=%d hit the iteration budget max_iter=%d. The val curve was still "
                "improving at the cap, so val essentially didn't serve as a stop signal. "
                "Consider: (1) raising max_iter and retraining; (2) refitting on train+val "
                "at the same (or doubled) max_iter; or (3) partial_fit-on-val to extend "
                "the booster on the val rows. None of these auto-apply -- see TODO in "
                "UniversalCallback.__init__.",
                int(self.best_iter), int(self.worsening_max_iter),
            )

    def _worsening_threshold(self) -> int:
        """Resolve the worsening-streak length threshold for the curve-shape ES trigger.

        The threshold scales with the booster's iteration budget so a 1000-iter run tolerates
        a longer worsening tail (200 iters at coeff=5) than a 30-iter run (5 iters via the
        min-iters floor). Returns ``worsening_min_iters`` when no budget is known.
        """
        if self.worsening_max_iter is None or self.worsening_max_iter <= 0:
            return int(self.worsening_min_iters)
        return max(int(self.worsening_max_iter) // int(self.worsening_coeff),
                    int(self.worsening_min_iters))

    def _update_worsening_streak(self, current_value: float, improved: bool) -> bool:
        """Track strict-monotone-worsening run length. Returns True when stop is triggered.

        The streak counts post-best iterations where each successor is NOT strictly better
        than its immediate predecessor (in the is_greater_better direction). A new global
        best resets the streak. A non-best value that nevertheless improves over its
        immediate predecessor also resets (the curve has "bent back" -- not monotone worsening).
        Equal-to-prev does NOT reset -- the user spec is "no successor BETTER than predecessor".
        """
        if not self.worsening_enabled:
            return False
        cur = float(current_value)
        if self._worsening_last_value is None:
            # First observation: no predecessor to compare against. Initialize and wait.
            self._worsening_last_value = cur
            return False
        prev = self._worsening_last_value
        self._worsening_last_value = cur
        improved_over_prev = (cur < prev) if self.mode == "min" else (cur > prev)
        if improved or improved_over_prev:
            self._worsening_streak_len = 0
            return False
        self._worsening_streak_len += 1
        if self._worsening_streak_len >= self._worsening_threshold():
            self._worsening_stopped = True
            return True
        return False

    def _effective_min_delta(self, shard_values: list[float] | None) -> float:
        """When ``slice_min_delta_in_se`` is set, scale the per-iteration absolute min_delta by
        ``slice_min_delta_in_se * SE(shard_values)``. Keeps abs-threshold semantics otherwise.
        """
        if self.slice_min_delta_in_se is None or not shard_values or len(shard_values) < 2:
            return self.min_delta
        import math
        std = float(np.std(shard_values, ddof=1)) * float(self.slice_correlation_inflation)
        se = std / math.sqrt(len(shard_values))
        return float(self.slice_min_delta_in_se) * se

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
                # Slice-stable ES path: aggregate per-shard scores before applying patience/min_delta.
                # When slice_k>0 AND the aggregate is unavailable this iter (CB metric_period>1
                # phantom or shards not yet registered) we SKIP the decision rather than
                # contaminate best_metric with the single-val number on a different scale.
                # In ``diagnostic_only`` mode we still call _compute_slice_aggregate so the
                # per-shard history populates for the Pareto artefact, but the stop decision
                # reads single-val (legacy path).
                slice_value: float | None = None
                slice_shards: list[float] | None = None
                if int(self.slice_k) > 0:
                    slice_value = self._compute_slice_aggregate()
                    if slice_value is None and not self.slice_diagnostic_only:
                        return False
                    if self.slice_persist_history and self.slice_shard_score_history:
                        slice_shards = self.slice_shard_score_history[-1]
                if self.slice_diagnostic_only:
                    current_value = history[-1]
                    effective_delta = self.min_delta
                else:
                    current_value = slice_value if slice_value is not None else history[-1]
                    effective_delta = self._effective_min_delta(slice_shards) if slice_shards else self.min_delta
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
                    improved = (self.mode == "min" and current_value < self.best_metric - effective_delta) or (
                        self.mode == "max" and current_value > self.best_metric + effective_delta
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
                    # Curve-shape detector: forward-looking complement to patience. Triggers
                    # when each successor since the best has STRICTLY failed to improve over
                    # its predecessor for ``_worsening_threshold()`` iterations -- the val
                    # curve is monotonically bending wrong and there's no point burning the
                    # remaining "uncashed" iterations to confirm patience.
                    if self._update_worsening_streak(current_value, improved):
                        if self.verbose > 0:
                            logger.info(
                                "Stopping early via curve-shape detector: strict-worsening "
                                "streak of %d iters since best @%d. %s",
                                self._worsening_streak_len, self.best_iter,
                                self._get_state(current_value=current_value),
                            )
                            self.last_reporting_ts = cur_ts
                        return True
                    # Last-iteration check: when the booster is at its iteration cap and we're
                    # NOT stopping early, log the diagnostic so the operator can decide whether
                    # to raise the budget or refit on train+val.
                    if (self.worsening_max_iter is not None and
                            self.iter >= int(self.worsening_max_iter) - 1):
                        self.maybe_warn_max_iter_hit()
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
