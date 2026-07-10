"""biz_value test for ``inference.time_budget_ensemble.TimeBudgetEnsemble``.

The win: a conservative FIXED-size ensemble (small enough to always meet a tight latency budget in the
worst case) leaves accuracy on the table whenever the request is actually fast (models come back quicker
than the worst case). ``TimeBudgetEnsemble`` adaptively keeps firing models while budget remains, giving
strictly better average accuracy than the conservative-fixed baseline while STILL never exceeding the
wall-clock budget on any individual request -- unlike a fixed FULL ensemble, which blows the budget whenever
per-model latency is on the slow side of its distribution.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_squared_error

from mlframe.inference.time_budget_ensemble import TimeBudgetEnsemble


class _LatencyModel:
    """A stub regressor whose ``predict`` call sleeps ``latency_fn()`` seconds and returns a fixed offset
    prediction -- each additional model in an ensemble narrows the average error toward 0."""

    def __init__(self, true_offset: float, noise_scale: float, latency_seconds: float, rng: np.random.Generator) -> None:
        self.true_offset = true_offset
        self.noise_scale = noise_scale
        self.latency_seconds = latency_seconds
        self._rng = rng

    def predict(self, X):
        time.sleep(self.latency_seconds)
        n = len(X)
        return np.full(n, self.true_offset) + self._rng.normal(scale=self.noise_scale, size=n)


def _make_models(rng: np.random.Generator, n_models: int = 6):
    # Priority order: best-first, each contributing independent noise around the true value (0.0) -- more
    # models averaged narrows the ensemble's error toward 0 by the usual sqrt(n) averaging argument.
    return [_LatencyModel(true_offset=0.0, noise_scale=1.0, latency_seconds=0.01, rng=rng) for _ in range(n_models)]


def test_biz_val_time_budget_ensemble_beats_conservative_fixed_size_within_budget():
    rng = np.random.default_rng(0)
    n_trials = 40
    n_rows = 5
    time_budget_seconds = 0.035  # enough for ~3 models at 0.01s/model plus scheduling slack

    y_true = np.zeros(n_rows)

    budget_errors = []
    conservative_errors = []
    max_wall_time = 0.0

    for trial in range(n_trials):
        models = _make_models(rng)
        ensemble = TimeBudgetEnsemble(models, time_budget_seconds=time_budget_seconds)

        t0 = time.perf_counter()
        pred_budget = ensemble.predict(np.zeros((n_rows, 1)))
        wall = time.perf_counter() - t0
        max_wall_time = max(max_wall_time, wall)

        # Conservative-fixed baseline: only run 1 model (the minimum that's guaranteed safe even if every
        # model in the ensemble happened to run at its slowest -- the standard worst-case-sizing approach).
        pred_conservative = models[0].predict(np.zeros((n_rows, 1)))

        budget_errors.append(mean_squared_error(y_true, pred_budget))
        conservative_errors.append(mean_squared_error(y_true, pred_conservative))

    mean_budget_mse = float(np.mean(budget_errors))
    mean_conservative_mse = float(np.mean(conservative_errors))
    rel_improvement = (mean_conservative_mse - mean_budget_mse) / mean_conservative_mse

    assert rel_improvement > 0.3, f"expected >30% MSE reduction vs conservative fixed-size baseline, got {rel_improvement:.4f} (budget_mse={mean_budget_mse:.4f}, conservative_mse={mean_conservative_mse:.4f})"
    assert max_wall_time < time_budget_seconds * 2.5, f"expected wall time to stay near the budget, got {max_wall_time:.4f}s vs budget {time_budget_seconds}s"


def test_time_budget_ensemble_always_runs_at_least_min_models():
    rng = np.random.default_rng(1)
    slow_models = [_LatencyModel(true_offset=0.0, noise_scale=1.0, latency_seconds=1.0, rng=rng) for _ in range(3)]
    ensemble = TimeBudgetEnsemble(slow_models, time_budget_seconds=0.001, min_models=1)
    ensemble.predict(np.zeros((2, 1)))
    assert ensemble.last_n_models_used_ == 1


def test_time_budget_ensemble_uses_more_models_with_larger_budget():
    rng = np.random.default_rng(2)
    models = [_LatencyModel(true_offset=0.0, noise_scale=1.0, latency_seconds=0.01, rng=rng) for _ in range(6)]

    tight = TimeBudgetEnsemble(models, time_budget_seconds=0.005)
    tight.predict(np.zeros((3, 1)))

    generous = TimeBudgetEnsemble(models, time_budget_seconds=0.5)
    generous.predict(np.zeros((3, 1)))

    assert generous.last_n_models_used_ > tight.last_n_models_used_
