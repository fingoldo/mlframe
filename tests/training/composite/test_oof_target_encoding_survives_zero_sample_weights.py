"""Target-encoding keeps its out-of-fold training forward when some sample weights are zero."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.transforms import get_transform
from mlframe.training.composite.transforms._call_gateway import call_transform


def _bed():
    """200 rows over 5 categories with a category-dependent level, 10 of them zero-weighted."""
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 5, 400)
    y = 10.0 * groups + rng.normal(0, 3, 400)
    w = np.ones(400)
    w[:10] = 0.0
    return y, groups, w


def test_train_forward_is_out_of_fold_with_zero_weights():
    """The fit's fingerprint covers the whole batch, so forward(y_train) is encoded out of fold, not against the full-train encoding that contains each row's own y."""
    y, groups, w = _bed()
    transform = get_transform("target_encoding_residual")
    params = call_transform(transform, "fit", y, np.zeros_like(y), groups=groups, sample_weight=w)
    assert "train_fingerprint" in params
    t_train = call_transform(transform, "forward", y, np.zeros_like(y), params, groups=groups)
    enc_full = np.array([params["encoding"][str(g)] for g in groups])
    assert not np.allclose(t_train, y - enc_full), "the train T equals the in-sample encoding residual: the out-of-fold path was not taken"


def test_unweighted_fit_is_unchanged():
    """Without zero weights the same out-of-fold forward applies."""
    y, groups, _w = _bed()
    transform = get_transform("target_encoding_residual")
    params = call_transform(transform, "fit", y, np.zeros_like(y), groups=groups)
    t_train = call_transform(transform, "forward", y, np.zeros_like(y), params, groups=groups)
    enc_full = np.array([params["encoding"][str(g)] for g in groups])
    assert not np.allclose(t_train, y - enc_full)
