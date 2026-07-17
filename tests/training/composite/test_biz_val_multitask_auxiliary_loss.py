"""biz_value test for ``training.composite.MultiTaskAuxiliaryLossRegressor``.

The win: a primary target has a sharp local "spike" near a boundary region (x near 0), the exact "weird
spikes near event boundaries" failure mode the LANL Earthquake Prediction 1st place's auxiliary losses were
built to fix. A single-task network trained only on the primary MSE loss has no extra signal steering its
shared representation toward the boundary region specifically. Adding an auxiliary binary "near boundary"
classification head and an auxiliary "distance to boundary" regression head, trained JOINTLY (shared trunk,
weighted-sum backpropagated loss), should recover the spike region's true values materially better -- even
if it's not necessarily better everywhere (the technique is a targeted fix for the boundary failure mode,
not a universal accuracy improvement, which is exactly what's tested here).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from mlframe.training.composite import MultiTaskAuxiliaryLossRegressor


def _make_boundary_spike_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = np.sin(x) + 3.0 * np.exp(-(x**2) / 0.02) + rng.normal(scale=0.05, size=n)
    aux_binary = (np.abs(x) < 0.2).astype(np.float32)
    aux_regression = np.abs(x)
    X = x.reshape(-1, 1).astype(np.float32)
    return X, y, aux_binary, aux_regression, x


def test_biz_val_multitask_auxiliary_loss_beats_single_task_near_boundary():
    X, y, aux_binary, aux_regression, x = _make_boundary_spike_dataset(n=2000, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:1400], perm[1400:]

    single_task = MultiTaskAuxiliaryLossRegressor(hidden_sizes=(32, 16), n_epochs=400, lr=0.01, random_state=0)
    single_task.fit(X[train_idx], y[train_idx])
    pred_single = single_task.predict(X[test_idx])

    multi_task = MultiTaskAuxiliaryLossRegressor(hidden_sizes=(32, 16), aux_task_weight=0.3, n_epochs=400, lr=0.01, random_state=0)
    multi_task.fit(X[train_idx], y[train_idx], y_aux_binary=aux_binary[train_idx], y_aux_regression=aux_regression[train_idx])
    pred_multi = multi_task.predict(X[test_idx])

    boundary_mask = np.abs(x[test_idx]) < 0.2
    assert boundary_mask.sum() >= 10, "synthetic didn't produce enough boundary-region test rows"

    mse_single_boundary = mean_squared_error(y[test_idx][boundary_mask], pred_single[boundary_mask])
    mse_multi_boundary = mean_squared_error(y[test_idx][boundary_mask], pred_multi[boundary_mask])
    improvement = 1.0 - mse_multi_boundary / mse_single_boundary
    assert improvement > 0.1, (
        f"expected >10% MSE reduction near the boundary region, got {improvement:.4f} (single={mse_single_boundary:.4f}, multi={mse_multi_boundary:.4f})"
    )


def test_multitask_auxiliary_loss_works_with_only_binary_aux_head():
    X, y, aux_binary, _, _ = _make_boundary_spike_dataset(n=300, seed=2)
    model = MultiTaskAuxiliaryLossRegressor(hidden_sizes=(8,), n_epochs=20, random_state=0)
    model.fit(X, y, y_aux_binary=aux_binary)
    pred = model.predict(X)
    assert pred.shape == (300,)
    assert model.aux_binary_head_ is not None
    assert model.aux_regression_head_ is None


def test_multitask_auxiliary_loss_records_decreasing_training_loss():
    X, y, aux_binary, aux_regression, _ = _make_boundary_spike_dataset(n=200, seed=3)
    model = MultiTaskAuxiliaryLossRegressor(hidden_sizes=(8,), n_epochs=50, lr=0.05, random_state=0)
    model.fit(X, y, y_aux_binary=aux_binary, y_aux_regression=aux_regression)
    assert len(model.train_losses_) == 50
    assert model.train_losses_[-1] < model.train_losses_[0]
