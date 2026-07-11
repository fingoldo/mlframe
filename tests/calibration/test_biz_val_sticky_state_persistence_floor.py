"""biz_value test for ``calibration.sticky_state_persistence_floor``.

Source: dd_2nd_nasa-airport-config.md -- "Minimum Configuration Support ... a learned parameter enforcing a
minimum predicted-probability floor for the currently active configuration ... 'one of the most important
aspects of our final submission.'" A per-step classifier can transiently flicker to a wrong class on a single
noisy row even when the true state is stable and unchanged on either side; flooring the active (previous
step's) class's probability should smooth those isolated flickers away, recovering the true persistent state.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from mlframe.calibration.sticky_state_persistence_floor import apply_sticky_state_persistence_floor


def _make_sticky_state_with_flicker(n: int, n_classes: int, n_flicker: int, seed: int):
    rng = np.random.default_rng(seed)
    true_state = np.zeros(n, dtype=int)
    cur = 0
    for i in range(1, n):
        if rng.random() < 0.01:
            cur = rng.integers(0, n_classes)
        true_state[i] = cur

    probs = np.zeros((n, n_classes))
    for i in range(n):
        row = np.full(n_classes, 0.05)
        row[true_state[i]] = 0.9
        probs[i] = row

    flicker_rows = rng.choice(n, size=n_flicker, replace=False)
    for i in flicker_rows:
        wrong = (true_state[i] + 1) % n_classes
        row = np.full(n_classes, 0.1)
        row[wrong] = 0.7
        probs[i] = row

    return true_state, probs


def _simulate_sequential(probs: np.ndarray, floor: float) -> np.ndarray:
    n = probs.shape[0]
    preds = np.zeros(n, dtype=int)
    preds[0] = np.argmax(probs[0])
    for i in range(1, n):
        floored_row = apply_sticky_state_persistence_floor(probs[i : i + 1], np.array([preds[i - 1]]), floor)
        preds[i] = np.argmax(floored_row[0])
    return preds


def test_biz_val_persistence_floor_smooths_isolated_flicker():
    true_state, probs = _make_sticky_state_with_flicker(n=2000, n_classes=3, n_flicker=150, seed=0)

    acc_raw = accuracy_score(true_state, _simulate_sequential(probs, floor=0.0))
    acc_floored = accuracy_score(true_state, _simulate_sequential(probs, floor=0.45))

    assert acc_floored > acc_raw + 0.05, f"expected the persistence floor to improve accuracy on isolated flicker by >=0.05, got floored={acc_floored:.4f} raw={acc_raw:.4f}"


def test_apply_sticky_state_persistence_floor_preserves_row_sums():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=50)
    active = rng.integers(0, 4, size=50)
    floored = apply_sticky_state_persistence_floor(probs, active, floor=0.6)
    np.testing.assert_allclose(floored.sum(axis=1), 1.0)


def test_apply_sticky_state_persistence_floor_leaves_already_dominant_rows_unchanged():
    probs = np.array([[0.9, 0.05, 0.05]])
    active = np.array([0])
    floored = apply_sticky_state_persistence_floor(probs, active, floor=0.6)
    np.testing.assert_allclose(floored, probs)
