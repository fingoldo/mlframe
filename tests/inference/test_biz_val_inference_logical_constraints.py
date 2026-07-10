"""biz_value + unit tests for ``inference.logical_constraints.apply_logical_constraints``.

The win: on a multi-label problem with a known implication rule (child never true without parent) that noisy
model predictions sometimes violate, swapping the violating pair (a) drives the constraint-violation rate to
exactly 0 and (b) improves (never worsens) log-loss against ground truth, since the swap only fires on rows
where the raw prediction contradicts KNOWN structure.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss

from mlframe.inference.logical_constraints import apply_logical_constraints, discover_logical_constraints


def _make_implication_data(n: int, seed: int, noise: float = 0.35):
    """child implies parent in ground truth: y_child=1 => y_parent=1 always. Noisy predictions sometimes
    violate that (pred_child > pred_parent) even though it never happens in ground truth."""
    rng = np.random.default_rng(seed)
    y_parent = (rng.random(n) < 0.5).astype(np.float64)
    y_child = y_parent * (rng.random(n) < 0.4).astype(np.float64)
    pred_parent = np.clip(y_parent + noise * rng.standard_normal(n), 0.01, 0.99)
    pred_child = np.clip(y_child + noise * rng.standard_normal(n), 0.01, 0.99)
    preds = np.column_stack([pred_child, pred_parent])
    y = np.column_stack([y_child, y_parent])
    return y, preds


def test_apply_logical_constraints_swaps_only_violating_rows():
    preds = np.array([[0.8, 0.3], [0.2, 0.9], [0.5, 0.5]])
    out = apply_logical_constraints(preds, rules=[(0, 1)])
    assert np.allclose(out[0], [0.3, 0.8])  # violated (child > parent) -> swapped
    assert np.allclose(out[1], [0.2, 0.9])  # not violated -> unchanged
    assert np.allclose(out[2], [0.5, 0.5])  # tie is not a violation -> unchanged


def test_apply_logical_constraints_preserves_marginal_value_set():
    rng = np.random.default_rng(0)
    preds = rng.uniform(0, 1, size=(200, 2))
    out = apply_logical_constraints(preds, rules=[(0, 1)])
    assert np.allclose(np.sort(out.ravel()), np.sort(preds.ravel()))


def test_apply_logical_constraints_out_of_bounds_rule_raises():
    preds = np.zeros((5, 2))
    with pytest.raises(ValueError):
        apply_logical_constraints(preds, rules=[(0, 5)])


def test_apply_logical_constraints_identical_indices_raises():
    preds = np.zeros((5, 2))
    with pytest.raises(ValueError):
        apply_logical_constraints(preds, rules=[(1, 1)])


def test_apply_logical_constraints_does_not_mutate_input():
    preds = np.array([[0.9, 0.1]])
    original = preds.copy()
    apply_logical_constraints(preds, rules=[(0, 1)])
    assert np.array_equal(preds, original)


def test_biz_val_apply_logical_constraints_zeroes_violation_rate_and_improves_log_loss():
    y, preds = _make_implication_data(5000, seed=3)
    violated_before = (preds[:, 0] > preds[:, 1]).mean()
    assert violated_before > 0.05, f"sanity: synthetic noise should produce real violations, got rate={violated_before}"

    out = apply_logical_constraints(preds, rules=[(0, 1)])
    violated_after = (out[:, 0] > out[:, 1]).mean()
    assert violated_after == 0.0

    ll_before = log_loss(y.ravel(), preds.ravel(), labels=[0.0, 1.0])
    ll_after = log_loss(y.ravel(), out.ravel(), labels=[0.0, 1.0])
    assert ll_after < ll_before, f"constraint enforcement should not worsen log-loss: before={ll_before:.4f} after={ll_after:.4f}"


def test_discover_logical_constraints_finds_true_implication():
    y, _ = _make_implication_data(3000, seed=5, noise=0.0)  # noise=0: labels only, exact implication holds
    rules = discover_logical_constraints(y, min_child_support=5)
    assert (0, 1) in rules  # child (col 0) implies parent (col 1) by construction


def test_discover_logical_constraints_no_self_pairs():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 3)) < 0.5).astype(np.float64)
    rules = discover_logical_constraints(y, min_child_support=1)
    assert all(c != p for c, p in rules)


def test_discover_logical_constraints_respects_min_support():
    y = np.zeros((100, 2))
    y[:2, 0] = 1.0  # only 2 positives for column 0
    y[:2, 1] = 1.0  # implication holds trivially but support is too low
    rules = discover_logical_constraints(y, min_child_support=10)
    assert (0, 1) not in rules


def test_discover_logical_constraints_requires_2d_input():
    with pytest.raises(ValueError):
        discover_logical_constraints(np.array([1.0, 0.0, 1.0]))


def test_discover_logical_constraints_fewer_than_two_labels_returns_empty():
    y = np.ones((10, 1))
    assert discover_logical_constraints(y) == []


def test_biz_val_discover_then_apply_end_to_end_reduces_violations_on_held_out_predictions():
    """The realistic workflow: discover rules from TRAINING labels, then apply them to fresh, noisy
    PREDICTIONS on a held-out set generated from the same generative process — proving the discovered rule
    generalizes and the combined discover+apply pipeline concretely reduces domain-inconsistent predictions."""
    y_train, _ = _make_implication_data(4000, seed=10, noise=0.0)
    rules = discover_logical_constraints(y_train, min_child_support=5)
    assert rules, "expected at least one discovered rule on data with a true implication"

    y_test, preds_test = _make_implication_data(2000, seed=11, noise=0.3)
    violated_before = (preds_test[:, 0] > preds_test[:, 1]).mean()
    fixed = apply_logical_constraints(preds_test, rules=rules)
    violated_after = (fixed[:, 0] > fixed[:, 1]).mean()

    assert violated_before > 0.05
    assert violated_after == 0.0
