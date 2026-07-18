"""biz_value test for ``calibration.sticky_state_persistence_floor``.

Source: dd_2nd_nasa-airport-config.md -- "Minimum Configuration Support ... a learned parameter enforcing a
minimum predicted-probability floor for the currently active configuration ... 'one of the most important
aspects of our final submission.'" A per-step classifier can transiently flicker to a wrong class on a single
noisy row even when the true state is stable and unchanged on either side; flooring the active (previous
step's) class's probability should smooth those isolated flickers away, recovering the true persistent state.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from mlframe.calibration.sticky_state_persistence_floor import (
    apply_sticky_state_persistence_floor,
    optimize_persistence_floor,
    optimize_persistence_floor_per_class,
)


def _make_sticky_state_with_flicker(n: int, n_classes: int, n_flicker: int, seed: int):
    """Helper that make sticky state with flicker."""
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
    """Helper that simulate sequential."""
    n = probs.shape[0]
    preds = np.zeros(n, dtype=int)
    preds[0] = np.argmax(probs[0])
    for i in range(1, n):
        floored_row = apply_sticky_state_persistence_floor(probs[i : i + 1], np.array([preds[i - 1]]), floor)
        preds[i] = np.argmax(floored_row[0])
    return preds


def test_biz_val_persistence_floor_smooths_isolated_flicker():
    """Persistence floor smooths isolated flicker."""
    true_state, probs = _make_sticky_state_with_flicker(n=2000, n_classes=3, n_flicker=150, seed=0)

    acc_raw = accuracy_score(true_state, _simulate_sequential(probs, floor=0.0))
    acc_floored = accuracy_score(true_state, _simulate_sequential(probs, floor=0.45))

    assert (
        acc_floored > acc_raw + 0.05
    ), f"expected the persistence floor to improve accuracy on isolated flicker by >=0.05, got floored={acc_floored:.4f} raw={acc_raw:.4f}"


def test_apply_sticky_state_persistence_floor_preserves_row_sums():
    """Apply sticky state persistence floor preserves row sums."""
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=50)
    active = rng.integers(0, 4, size=50)
    floored = apply_sticky_state_persistence_floor(probs, active, floor=0.6)
    np.testing.assert_allclose(floored.sum(axis=1), 1.0)


def test_apply_sticky_state_persistence_floor_leaves_already_dominant_rows_unchanged():
    """Apply sticky state persistence floor leaves already dominant rows unchanged."""
    probs = np.array([[0.9, 0.05, 0.05]])
    active = np.array([0])
    floored = apply_sticky_state_persistence_floor(probs, active, floor=0.6)
    np.testing.assert_allclose(floored, probs)


def test_apply_sticky_state_persistence_floor_scalar_and_uniform_vector_are_bit_identical():
    """A ``(k,)`` floor vector with every entry equal to the same scalar must reproduce the original
    scalar-floor code path bit-for-bit -- the per-class vector mode is opt-in and must not silently change
    default (uniform) behavior."""
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(5), size=500)
    active = rng.integers(0, 5, size=500)
    scalar_out = apply_sticky_state_persistence_floor(probs, active, floor=0.4)
    vector_out = apply_sticky_state_persistence_floor(probs, active, floor=np.full(5, 0.4))
    np.testing.assert_array_equal(scalar_out, vector_out)


def _make_sticky_and_volatile_class_mix(n: int, seed: int):
    """Two-class sequence: class 0 is highly persistent (rare spontaneous transitions, 97%+ of steps), class 1
    is volatile (transitions often once entered). Both classes get the same isolated-flicker noise rate, so a
    single global floor tuned for overall accuracy is a genuine compromise between what each class needs."""
    rng = np.random.default_rng(seed)
    switch_prob = {0: 0.01, 1: 0.4}
    true_state = np.zeros(n, dtype=int)
    cur = 0
    for i in range(1, n):
        if rng.random() < switch_prob[cur]:
            cur = 1 - cur
        true_state[i] = cur

    probs = np.zeros((n, 2))
    for i in range(n):
        row = np.array([0.25, 0.25])
        row[true_state[i]] = 0.75
        probs[i] = row

    flicker_rows = rng.choice(n, size=int(n * 0.05), replace=False)
    for i in flicker_rows:
        wrong = 1 - true_state[i]
        row = np.array([0.3, 0.3])
        row[wrong] = 0.7
        probs[i] = row

    # "active_class" is the previously-confirmed true state, i.e. the persistence-floor context signal --
    # not recursively dependent on the model's own prior prediction, so optimize_persistence_floor(_per_class)
    # can be applied directly per row.
    active = np.empty(n, dtype=int)
    active[0] = true_state[0]
    active[1:] = true_state[:-1]
    return true_state, probs, active


def test_biz_val_optimize_persistence_floor_per_class_beats_global_scalar_compromise():
    """Optimize persistence floor per class beats global scalar compromise."""
    true_state, probs, active = _make_sticky_and_volatile_class_mix(n=8000, seed=0)

    # balanced accuracy weights both classes' recall equally regardless of the sticky class's ~40x higher
    # prevalence -- this is what exposes the compromise: a plain accuracy metric would let the scalar sweep
    # ignore the rare volatile class almost entirely.
    metric_fn = lambda yt, yp: float(balanced_accuracy_score(yt, yp))

    scalar_result = optimize_persistence_floor(probs, active, true_state, metric_fn, n_thresholds=30)
    scalar_floored = apply_sticky_state_persistence_floor(probs, active, scalar_result["threshold"])
    acc_scalar = balanced_accuracy_score(true_state, np.argmax(scalar_floored, axis=1))

    per_class_result = optimize_persistence_floor_per_class(probs, active, true_state, metric_fn, n_thresholds=30)
    per_class_floored = apply_sticky_state_persistence_floor(probs, active, per_class_result["floor"])
    acc_per_class = balanced_accuracy_score(true_state, np.argmax(per_class_floored, axis=1))

    assert (
        per_class_result["floor"][0] != per_class_result["floor"][1]
    ), f"expected the two classes to genuinely need different floors, got {per_class_result['floor']}"
    assert acc_per_class > acc_scalar + 0.012, (
        f"expected per-class floor tuning to beat the best single global scalar floor by >=0.012 balanced "
        f"accuracy, got per_class={acc_per_class:.4f} scalar={acc_scalar:.4f}"
    )
