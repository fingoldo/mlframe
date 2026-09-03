"""Regression tests for the shared LightGBM focal-loss gradient/hessian."""

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._focal_loss_shared import make_focal_objective


class _FakeDataset:
    """Minimal stand-in for a LightGBM Dataset, exposing only get_label()."""

    def __init__(self, labels: np.ndarray) -> None:
        """Store the labels array."""
        self._labels = labels

    def get_label(self) -> np.ndarray:
        """Return the stored labels."""
        return self._labels


def _focal_loss(preds: np.ndarray, labels: np.ndarray, gamma: float) -> np.ndarray:
    """Reference (non-gradient) binary focal loss, used for finite-difference checks."""
    p = 1.0 / (1.0 + np.exp(-preds))
    pt = labels * p + (1.0 - labels) * (1.0 - p)
    pt = np.clip(pt, 1e-9, 1.0 - 1e-9)
    return -((1.0 - pt) ** gamma) * np.log(pt)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 3.0])
def test_focal_objective_grad_hess_match_finite_differences(gamma: float) -> None:
    """Grad/hess from make_focal_objective must match a finite-difference reference."""
    rng = np.random.default_rng(0)
    preds = rng.uniform(-6.0, 6.0, 200)
    labels = rng.integers(0, 2, 200).astype(float)
    objective = make_focal_objective(gamma=gamma)
    grad, hess = objective(preds, _FakeDataset(labels))

    eps = 1e-5
    loss_plus = _focal_loss(preds + eps, labels, gamma)
    loss_minus = _focal_loss(preds - eps, labels, gamma)
    loss_mid = _focal_loss(preds, labels, gamma)
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    fd_hess = (loss_plus - 2 * loss_mid + loss_minus) / (eps**2)

    np.testing.assert_allclose(grad, fd_grad, atol=2e-3, rtol=2e-3)
    # hess is floored at 1e-6 for LightGBM's Newton step, so only compare where the
    # unfloored finite-difference curvature is itself comfortably positive.
    keep = fd_hess > 1e-3
    np.testing.assert_allclose(hess[keep], fd_hess[keep], atol=2e-2, rtol=2e-2)


def test_focal_objective_reduces_loss_over_gradient_steps() -> None:
    """A Newton step using the objective's (grad, hess) must reduce the focal loss."""
    # A correct (grad, hess) pair must let a Newton step on a random logit init reduce
    # the focal loss towards the true label; the pre-fix formula moved preds the wrong
    # direction for a large share of rows.
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 500).astype(float)
    preds = rng.uniform(-2.0, 2.0, 500)
    objective = make_focal_objective(gamma=2.0)

    loss_before = _focal_loss(preds, labels, 2.0).mean()
    grad, hess = objective(preds, _FakeDataset(labels))
    preds_after = preds - grad / hess
    loss_after = _focal_loss(preds_after, labels, 2.0).mean()
    assert loss_after < loss_before
