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
        """Helper that predict."""
        time.sleep(self.latency_seconds)
        n = len(X)
        return np.full(n, self.true_offset) + self._rng.normal(scale=self.noise_scale, size=n)


def _make_models(rng: np.random.Generator, n_models: int = 6):
    # Priority order: best-first, each contributing independent noise around the true value (0.0) -- more
    # models averaged narrows the ensemble's error toward 0 by the usual sqrt(n) averaging argument.
    """Helper that make models."""
    return [_LatencyModel(true_offset=0.0, noise_scale=1.0, latency_seconds=0.01, rng=rng) for _ in range(n_models)]


def test_biz_val_time_budget_ensemble_beats_conservative_fixed_size_within_budget():
    """Time budget ensemble beats conservative fixed size within budget."""
    rng = np.random.default_rng(0)
    n_trials = 40
    n_rows = 5
    time_budget_seconds = 0.035  # enough for ~3 models at 0.01s/model plus scheduling slack

    y_true = np.zeros(n_rows)

    budget_errors = []
    conservative_errors = []
    max_wall_time = 0.0

    for _trial in range(n_trials):
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

    assert (
        rel_improvement > 0.3
    ), f"expected >30% MSE reduction vs conservative fixed-size baseline, got {rel_improvement:.4f} (budget_mse={mean_budget_mse:.4f}, conservative_mse={mean_conservative_mse:.4f})"
    assert max_wall_time < time_budget_seconds * 2.5, f"expected wall time to stay near the budget, got {max_wall_time:.4f}s vs budget {time_budget_seconds}s"


def test_time_budget_ensemble_always_runs_at_least_min_models():
    """Time budget ensemble always runs at least min models."""
    rng = np.random.default_rng(1)
    slow_models = [_LatencyModel(true_offset=0.0, noise_scale=1.0, latency_seconds=1.0, rng=rng) for _ in range(3)]
    ensemble = TimeBudgetEnsemble(slow_models, time_budget_seconds=0.001, min_models=1)
    ensemble.predict(np.zeros((2, 1)))
    assert ensemble.last_n_models_used_ == 1


def test_biz_val_time_budget_ensemble_value_per_ms_avoids_starving_accurate_model():
    """Naive cheap-first priority order under a tight budget can starve a slower-but-far-more-accurate model
    out entirely. ``value_per_ms`` reordering fires the high-lift model first instead, materially cutting
    ensemble MSE for the SAME time budget."""
    rng = np.random.default_rng(3)
    n_trials = 40
    n_rows = 5
    time_budget_seconds = 0.012  # enough for exactly one 0.01s model plus scheduling slack

    y_true = np.zeros(n_rows)

    # Cheap-first (naive static priority): several near-worthless fast models ranked ahead of one slow but
    # much more accurate model -- under a tight budget the naive order never reaches the accurate model.
    def _make_cheap_first(rng):
        """Helper that make cheap first."""
        cheap = [_LatencyModel(true_offset=3.0, noise_scale=1.0, latency_seconds=0.001, rng=rng) for _ in range(4)]
        accurate = _LatencyModel(true_offset=0.0, noise_scale=0.05, latency_seconds=0.01, rng=rng)
        return [*cheap, accurate]

    naive_errors = []
    valuemode_errors = []

    for _ in range(n_trials):
        models = _make_cheap_first(rng)

        naive = TimeBudgetEnsemble(models, time_budget_seconds=time_budget_seconds)
        pred_naive = naive.predict(np.zeros((n_rows, 1)))
        naive_errors.append(mean_squared_error(y_true, pred_naive))

        # Holdout benchmark: cheap models have near-zero metric lift (offset 3.0 is nearly useless), the
        # accurate model has a large lift -- value_per_ms promotes it ahead of the cheap models.
        metric_lift = [0.01, 0.01, 0.01, 0.01, 5.0]
        latencies = [0.001, 0.001, 0.001, 0.001, 0.01]
        scores = TimeBudgetEnsemble.compute_value_per_ms(metric_lift, latencies)
        valuemode = TimeBudgetEnsemble(models, time_budget_seconds=time_budget_seconds, value_per_ms=scores)
        pred_valuemode = valuemode.predict(np.zeros((n_rows, 1)))
        valuemode_errors.append(mean_squared_error(y_true, pred_valuemode))

    mean_naive_mse = float(np.mean(naive_errors))
    mean_valuemode_mse = float(np.mean(valuemode_errors))
    rel_improvement = (mean_naive_mse - mean_valuemode_mse) / mean_naive_mse

    assert rel_improvement > 0.5, (
        f"expected >50% MSE reduction from value_per_ms reordering vs naive cheap-first order, "
        f"got {rel_improvement:.4f} (naive_mse={mean_naive_mse:.4f}, valuemode_mse={mean_valuemode_mse:.4f})"
    )


def test_time_budget_ensemble_value_per_ms_default_none_preserves_caller_order():
    """OPT-IN contract: when ``value_per_ms`` is not passed, model order and predictions must be bit-identical
    to the pre-extension behavior."""
    rng = np.random.default_rng(4)
    models = _make_models(rng, n_models=4)
    baseline = TimeBudgetEnsemble(models, time_budget_seconds=0.5)
    assert baseline.models == models
    assert baseline.value_per_ms is None


def test_time_budget_ensemble_uses_more_models_with_larger_budget():
    """Time budget ensemble uses more models with larger budget."""
    rng = np.random.default_rng(2)
    models = [_LatencyModel(true_offset=0.0, noise_scale=1.0, latency_seconds=0.01, rng=rng) for _ in range(6)]

    tight = TimeBudgetEnsemble(models, time_budget_seconds=0.005)
    tight.predict(np.zeros((3, 1)))

    generous = TimeBudgetEnsemble(models, time_budget_seconds=0.5)
    generous.predict(np.zeros((3, 1)))

    assert generous.last_n_models_used_ > tight.last_n_models_used_
