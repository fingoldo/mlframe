"""The batched per-class ICE kernels must return what looping ``fast_ice_only`` per class returns.

Both batched kernels took the bin CENTRE as the bin's predicted probability while the serial reference takes the bin's
MEAN prediction, and the docstring called the two "bit-exact". On a Beta(2,5) bed at nbins=10 they returned -0.0917 and
-0.1477: the ICE that drives CatBoost/LightGBM early stopping was not the ICE printed in the calibration report.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._ice_kernel import (
    _ICE_KWARGS_TUPLE,
    _batch_per_class_ice_kernel,
    _batch_per_class_ice_kernel_serial,
)
from mlframe.metrics.core import fast_ice_only

_KWARG_NAMES = (
    "use_weights", "mae_weight", "std_weight", "brier_loss_weight",
    "roc_auc_weight", "pr_auc_weight", "min_roc_auc", "roc_auc_penalty", "coverage_weight",
)


def _bed(n: int, seed: int, skewed: bool):
    """(y_true_NK, y_pred_NK, desc_idx) for a 2-class problem; ``skewed`` puts the mass off the bin centres."""
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 5, n) if skewed else rng.random(n)
    y = (rng.random(n) < p).astype(np.int8)
    y_true = np.ascontiguousarray(np.stack([1 - y, y], axis=1).astype(np.int8))
    y_pred = np.ascontiguousarray(np.stack([1.0 - p, p], axis=1))
    desc_idx = np.ascontiguousarray(np.argsort(-y_pred, axis=0).astype(np.int64))
    return y_true, y_pred, desc_idx


def _reference(y_true, y_pred, kwargs):
    """The serial gold path: ``fast_ice_only`` once per class, on the uniform bins the batched kernels implement.

    ``binning_strategy`` is pinned here because the kernels only know uniform bins; picking quantile bins for a rare
    positive class is the caller's decision, made in ``ICE.evaluate`` before it chooses the batched path at all.
    """
    return np.array([
        fast_ice_only(np.ascontiguousarray(y_true[:, k]), np.ascontiguousarray(y_pred[:, k]), binning_strategy="uniform", **kwargs)
        for k in range(y_true.shape[1])
    ])


@pytest.mark.parametrize("n, nbins, skewed", [(5_000, 10, True), (20_000, 100, True), (3_000, 20, False)])
def test_batched_kernels_match_fast_ice_only(n, nbins, skewed):
    y_true, y_pred, desc_idx = _bed(n, seed=0, skewed=skewed)
    args = list(_ICE_KWARGS_TUPLE)
    args[0] = nbins
    kwargs = dict(zip(("nbins", *_KWARG_NAMES), args))
    expected = _reference(y_true, y_pred, kwargs)
    for kernel in (_batch_per_class_ice_kernel, _batch_per_class_ice_kernel_serial):
        got = np.asarray(kernel(y_true, y_pred, desc_idx, *args))
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_a_skewed_bin_is_not_scored_at_its_centre():
    """The regression itself: with all mass at one end of every bin, centre and mean differ, so a kernel using the
    centre cannot match. Pins that the batched path tracks the mean."""
    n = 4_000
    rng = np.random.default_rng(3)
    p = np.clip(rng.beta(1.2, 12.0, n), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < p).astype(np.int8)
    y_true = np.ascontiguousarray(np.stack([1 - y, y], axis=1).astype(np.int8))
    y_pred = np.ascontiguousarray(np.stack([1.0 - p, p], axis=1))
    desc_idx = np.ascontiguousarray(np.argsort(-y_pred, axis=0).astype(np.int64))
    args = list(_ICE_KWARGS_TUPLE)
    args[0] = 10
    kwargs = dict(zip(("nbins", *_KWARG_NAMES), args))
    got = np.asarray(_batch_per_class_ice_kernel(y_true, y_pred, desc_idx, *args))
    np.testing.assert_allclose(got, _reference(y_true, y_pred, kwargs), rtol=0, atol=0)
