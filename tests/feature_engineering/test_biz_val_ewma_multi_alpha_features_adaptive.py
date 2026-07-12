"""biz_value test for the opt-in ``adaptive_alpha_grid`` mode of ``ewma_multi_alpha_features``.

Rationale: the source trick (4th_santander-product-recommendation.md) hand-picks one fixed alpha set shared
by every entity. Real populations mix low-volatility entities (a slowly-drifting persistent level, best
tracked by a SMALL alpha that averages over more history) with high-volatility entities (a level that jumps
around fast, best tracked by a LARGE alpha that reacts quickly) -- no single fixed alpha (or small shared
set) is simultaneously optimal for both. ``adaptive_alpha_grid`` walk-forward-validates a grid PER ENTITY on
that entity's own history and should therefore one-step-ahead-predict better than the best fixed alpha
shared across the whole population.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.ewma_multi_alpha_features import ewma_multi_alpha_features


def _make_mixed_volatility_entities(n_entities: int, hist_len: int, seed: int):
    """Half the entities have a slowly-drifting level (needs small alpha), half a fast-jumping one (needs large alpha)."""
    rng = np.random.default_rng(seed)
    values_list, group_list = [], []
    for e in range(n_entities):
        is_high_vol = e % 2 == 1
        level_step_std = 0.9 if is_high_vol else 0.03
        level = 0.0
        levels = np.empty(hist_len)
        for t in range(hist_len):
            level += rng.normal(0.0, level_step_std)
            levels[t] = level
        obs = levels + rng.normal(0.0, 0.5, size=hist_len)
        values_list.append(obs)
        group_list.append(np.full(hist_len, e))
    return values_list, np.concatenate(group_list)


def _one_step_ahead_mse(ewma_col: np.ndarray, values: np.ndarray, group_ids: np.ndarray, tail_frac: float) -> float:
    """MSE of using yesterday's smoothed value to predict today's raw value, on each entity's held-out tail."""
    sq_errors = []
    for e in np.unique(group_ids):
        mask = group_ids == e
        v = values[mask]
        s = ewma_col[mask]
        length = v.shape[0]
        tail = max(3, int(length * tail_frac))
        tail = min(tail, length - 1)
        for i in range(length - tail, length):
            pred = s[i - 1]
            sq_errors.append((v[i] - pred) ** 2)
    return float(np.mean(sq_errors))


def test_biz_val_ewma_multi_alpha_features_adaptive_beats_best_fixed_alpha():
    values_list, group_ids = _make_mixed_volatility_entities(n_entities=200, hist_len=60, seed=7)
    values = np.concatenate(values_list)

    fixed_grid = [0.03, 0.1, 0.9]
    res = ewma_multi_alpha_features(values, group_ids, alphas=fixed_grid, adaptive_alpha_grid=fixed_grid, adaptive_val_frac=0.3, adaptive_min_val_points=4)

    mse_fixed_by_alpha = {a: _one_step_ahead_mse(res[f"ewma_alpha_{float(a)}"], values, group_ids, tail_frac=0.3) for a in fixed_grid}
    best_fixed_mse = min(mse_fixed_by_alpha.values())
    adaptive_mse = _one_step_ahead_mse(res["ewma_adaptive"], values, group_ids, tail_frac=0.3)

    assert adaptive_mse < best_fixed_mse * 0.95, (
        f"expected per-entity adaptive alpha to beat the single best fixed alpha shared across a mixed-volatility "
        f"population by >=5% MSE, got adaptive={adaptive_mse:.4f} best_fixed={best_fixed_mse:.4f} ({mse_fixed_by_alpha})"
    )


def test_ewma_multi_alpha_features_adaptive_omitted_is_bit_identical_to_fixed_only():
    rng = np.random.default_rng(3)
    values = rng.normal(size=200)
    groups = rng.integers(0, 10, size=200)

    baseline = ewma_multi_alpha_features(values, groups, alphas=[0.5, 0.1])
    with_adaptive_param_unset = ewma_multi_alpha_features(values, groups, alphas=[0.5, 0.1], adaptive_alpha_grid=None)

    assert set(baseline.keys()) == set(with_adaptive_param_unset.keys())
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], with_adaptive_param_unset[key])


def test_ewma_multi_alpha_features_adaptive_rejects_empty_grid():
    import pytest

    with pytest.raises(ValueError):
        ewma_multi_alpha_features(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]), adaptive_alpha_grid=[])
