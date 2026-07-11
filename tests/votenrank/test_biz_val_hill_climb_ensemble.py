"""biz_value + unit tests for ``votenrank.hill_climb_ensemble``.

The win: on a pool of 20 OOF prediction arrays where 5 are genuinely good (low-noise) and 15 are mediocre
(high-noise), greedy hill-climbing with replacement should converge to an ensemble that beats BOTH (a) a
naive equal-weight average over all 20 candidates (diluted by the 15 noisy ones) and (b) the single best
model alone (diversity among the good models still helps) — quantifying the two failure modes a real
ensembling pipeline needs to avoid.
"""
from __future__ import annotations

import numpy as np

from mlframe.votenrank.hill_climb import hill_climb_ensemble


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _make_model_pool(n_samples: int, n_good: int, n_bad: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n_samples) * 3.0
    good_preds = [y_true + 0.3 * rng.standard_normal(n_samples) for _ in range(n_good)]
    bad_preds = [y_true + 2.5 * rng.standard_normal(n_samples) for _ in range(n_bad)]
    return y_true, good_preds + bad_preds


def test_hill_climb_ensemble_single_model_pool_returns_that_model():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    preds = [y_true + 0.1]
    result = hill_climb_ensemble(preds, y_true, _rmse, maximize=False, max_iterations=10)
    assert result["selected_indices"] == [0] * len(result["selected_indices"])
    assert np.allclose(result["weights"], [1.0])


def test_hill_climb_ensemble_stops_when_no_candidate_improves():
    # two IDENTICAL models -- after the first pick, adding the other changes nothing (averaging an
    # identical array with itself is a no-op), so the search should stop quickly, not loop to max_iterations.
    y_true = np.array([1.0, 2.0, 3.0])
    preds = [y_true + 0.5, y_true + 0.5]
    result = hill_climb_ensemble(preds, y_true, _rmse, maximize=False, max_iterations=50, tol=1e-12)
    assert len(result["selected_indices"]) < 50


def test_hill_climb_ensemble_weights_sum_to_one():
    y_true, preds = _make_model_pool(200, n_good=3, n_bad=2, seed=0)
    result = hill_climb_ensemble(preds, y_true, _rmse, maximize=False, max_iterations=30)
    assert np.isclose(result["weights"].sum(), 1.0)


def test_hill_climb_ensemble_empty_pool_raises():
    import pytest

    with pytest.raises(ValueError):
        hill_climb_ensemble([], np.array([1.0]), _rmse)


def test_biz_val_hill_climb_beats_equal_weight_average_and_single_best_model():
    y_true, preds = _make_model_pool(n_samples=3000, n_good=5, n_bad=15, seed=42)

    result = hill_climb_ensemble(preds, y_true, _rmse, maximize=False, max_iterations=60, tol=1e-6)
    hill_climb_rmse = result["score"]

    equal_weight_pred = np.mean(preds, axis=0)
    equal_weight_rmse = _rmse(y_true, equal_weight_pred)

    single_scores = [_rmse(y_true, p) for p in preds]
    single_best_rmse = min(single_scores)

    assert hill_climb_rmse < equal_weight_rmse, (
        f"hill-climbing should beat naive equal-weight averaging over the full noisy pool: "
        f"hill_climb={hill_climb_rmse:.4f} equal_weight={equal_weight_rmse:.4f}"
    )
    assert hill_climb_rmse < single_best_rmse, (
        f"hill-climbing should beat the single best model alone (diversity among the good models still helps): "
        f"hill_climb={hill_climb_rmse:.4f} single_best={single_best_rmse:.4f}"
    )

    # the selected pool should be dominated by the 5 good models (indices 0-4), not the 15 noisy ones.
    good_weight = float(result["weights"][:5].sum())
    assert good_weight > 0.8, f"hill-climbing should overwhelmingly favor the good models, got good_weight={good_weight:.2f}"


def test_biz_val_hill_climb_ensemble_bagging_generalizes_better_than_single_path():
    # Small, noisy OOF set (n=40) with many candidate models (30) -- greedy hill-climbing on such a small
    # set can chase sampling noise in the OOF metric_fn evaluation itself and pick a path that looks great on
    # THAT small set but doesn't generalize to a genuinely separate held-out set drawn from the same models.
    n_oof = 40
    n_holdout = 4000
    n_models = 30
    rng = np.random.default_rng(7)

    # each model has a fixed per-model bias/noise-scale "skill" that is consistent across OOF and holdout,
    # but the OOF set is small enough that per-sample noise realizations let mediocre models look
    # spuriously good by chance on just those 40 points.
    skills = rng.uniform(0.3, 3.0, size=n_models)

    y_oof = rng.standard_normal(n_oof) * 3.0
    y_holdout = rng.standard_normal(n_holdout) * 3.0

    oof_preds = [y_oof + skills[m] * rng.standard_normal(n_oof) for m in range(n_models)]
    holdout_preds = [y_holdout + skills[m] * rng.standard_normal(n_holdout) for m in range(n_models)]

    single_result = hill_climb_ensemble(oof_preds, y_oof, _rmse, maximize=False, max_iterations=60, tol=1e-6)
    single_oof_rmse = single_result["score"]

    bagged_result = hill_climb_ensemble(
        oof_preds,
        y_oof,
        _rmse,
        maximize=False,
        max_iterations=60,
        tol=1e-6,
        n_bags=25,
        randomize_start=True,
        randomize_order=True,
        random_state=123,
    )

    def _holdout_rmse(weights: np.ndarray) -> float:
        blended = np.zeros(n_holdout)
        for idx, w in enumerate(weights):
            blended += w * holdout_preds[idx]
        return _rmse(y_holdout, blended)

    single_holdout_rmse = _holdout_rmse(single_result["weights"])
    bagged_holdout_rmse = _holdout_rmse(bagged_result["weights"])

    # the win we're proving: single-path greedy looks at least as good (often better) on the tiny OOF set it
    # was fit against, but the bagged/averaged-weights variant generalizes better (lower RMSE) on the
    # genuinely separate held-out set -- a real, numeric overfitting-reduction effect, not a placeholder.
    assert bagged_holdout_rmse < single_holdout_rmse * 0.95, (
        f"bagged hill-climb should generalize meaningfully better to held-out data: "
        f"bagged_holdout={bagged_holdout_rmse:.4f} single_holdout={single_holdout_rmse:.4f} "
        f"(single_oof={single_oof_rmse:.4f})"
    )
