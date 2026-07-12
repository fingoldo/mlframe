"""``TimeBudgetEnsemble``: wall-clock-budget-aware ensemble inference ("fire until the clock runs out").

Source: 3rd_riiid-answer-correctness-prediction.md's "Blindfolded Gunslinger" -- ensemble inference that
runs as many models as the time budget allows, degrading gracefully to fewer models under a hard
inference-time constraint, rather than a fixed static ensemble size (which either wastes budget when
running ahead of schedule or blows the SLA when running behind).

Models are ranked by priority (best-first): given a per-request time budget, run models in priority order,
checking elapsed time after each one, and stop as soon as continuing would risk exceeding the budget
(estimated from the running average per-model cost so far). Predictions from whichever models actually ran
are averaged -- always at least the top-priority model runs, even under a budget too tight for a second.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence

import numpy as np


class TimeBudgetEnsemble:
    """Wall-clock-budget-aware ensemble predictor.

    Parameters
    ----------
    models
        Estimators ranked BEST-FIRST (highest priority first) -- each exposes ``predict`` (regression) or
        ``predict_proba`` (classification, selected via ``use_predict_proba``).
    time_budget_seconds
        Wall-clock budget for the WHOLE ``predict``/``predict_proba`` call.
    use_predict_proba
        If True, call ``predict_proba`` on each model (classification) and average column-wise; if False,
        call ``predict`` (regression) and average.
    min_models
        Always run at least this many models regardless of budget (default 1 -- never return an empty
        ensemble).
    value_per_ms
        OPT-IN. Per-model priority score (typically holdout-metric-lift / measured-latency-ms), same length
        and order as ``models``. When provided, ``self.models`` is re-sorted DESCENDING by this score before
        any prediction happens -- so a caller-supplied static order (e.g. cheap-first) can no longer starve
        a slower-but-far-more-accurate model out of a tight budget. When ``None`` (the default), the
        caller-supplied order in ``models`` is used verbatim -- behavior is bit-identical to before this
        parameter existed. See ``compute_value_per_ms`` to derive scores from a holdout benchmark.
    """

    def __init__(
        self,
        models: Sequence[Any],
        time_budget_seconds: float,
        use_predict_proba: bool = False,
        min_models: int = 1,
        value_per_ms: Optional[Sequence[float]] = None,
    ) -> None:
        if len(models) == 0:
            raise ValueError("models must be non-empty")
        if value_per_ms is not None and len(value_per_ms) != len(models):
            raise ValueError(f"value_per_ms must have the same length as models, got {len(value_per_ms)} vs {len(models)}")

        if value_per_ms is None:
            self.models: List[Any] = list(models)
        else:
            # Stable sort: ties keep the caller's original relative order rather than being shuffled.
            order = sorted(range(len(models)), key=lambda i: value_per_ms[i], reverse=True)
            self.models = [models[i] for i in order]

        self.value_per_ms = None if value_per_ms is None else list(value_per_ms)
        self.time_budget_seconds = float(time_budget_seconds)
        self.use_predict_proba = use_predict_proba
        self.min_models = max(1, min_models)
        self.last_n_models_used_: Optional[int] = None
        self.last_model_costs_: List[float] = []

    @staticmethod
    def compute_value_per_ms(metric_lift: Sequence[float], latency_seconds: Sequence[float]) -> List[float]:
        """Derive a per-model priority score from a holdout-metric benchmark: lift-per-millisecond.

        ``metric_lift`` is each model's improvement over some common reference (e.g. AUC/accuracy gain, or
        error REDUCTION so higher is always better) measured on a holdout set; ``latency_seconds`` is that
        model's measured single-call inference latency. A cheap-but-mediocre model can have a high lift/cost
        ratio only if its lift is genuinely competitive -- a merely-fast model with near-zero lift scores low
        and drops in priority, letting a slower-but-much-more-accurate model fire first instead.
        """
        if len(metric_lift) != len(latency_seconds):
            raise ValueError(f"metric_lift and latency_seconds must have equal length, got {len(metric_lift)} vs {len(latency_seconds)}")
        ms = np.asarray(latency_seconds, dtype=np.float64) * 1000.0
        if np.any(ms <= 0):
            raise ValueError("latency_seconds must be strictly positive")
        return list(np.asarray(metric_lift, dtype=np.float64) / ms)

    def _predict_one(self, model: Any, X: Any) -> np.ndarray:
        fn = model.predict_proba if self.use_predict_proba else model.predict
        return np.asarray(fn(X))

    def predict(self, X: Any) -> np.ndarray:
        """Run models in priority order until the time budget is exhausted, average whichever ran.

        After each model, the ELAPSED time plus the RUNNING AVERAGE per-model cost so far is compared
        against the budget -- stopping before starting a model expected to blow the deadline, rather than
        stopping only after the deadline has already passed (the whole point of graceful degradation is to
        never miss the SLA on the request that triggered the cutoff).
        """
        start = time.perf_counter()
        predictions: List[np.ndarray] = []
        costs: List[float] = []
        cost_sum = 0.0

        for i, model in enumerate(self.models):
            elapsed = time.perf_counter() - start
            if i >= self.min_models and costs:
                # A running sum avoids recomputing np.mean(costs) (numpy dispatch overhead) on every
                # iteration -- costs is a small Python list, so plain arithmetic is both simpler and faster.
                expected_next_cost = cost_sum / len(costs)
                if elapsed + expected_next_cost > self.time_budget_seconds:
                    break

            t0 = time.perf_counter()
            predictions.append(self._predict_one(model, X))
            cost = time.perf_counter() - t0
            costs.append(cost)
            cost_sum += cost

        self.last_n_models_used_ = len(predictions)
        self.last_model_costs_ = costs
        return np.asarray(np.mean(predictions, axis=0))

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.use_predict_proba:
            raise AttributeError("predict_proba is only available when use_predict_proba=True")
        return self.predict(X)


__all__ = ["TimeBudgetEnsemble"]
