"""Regression: plug-in MI njit/CUDA kernels must be memory-safe for non-dense (negative) integer labels.

Pre-fix, ``_plugin_mi_classif_njit`` sized the class axis on ``max(y)+1`` and indexed the histogram with the raw
label. A ``y`` containing negative values (a binned continuous target shifted below 0, e.g. ``-10..6``) indexed
``hist_y[-10]`` into a length-7 array -> out-of-bounds read -> Windows native access violation (0xC0000005) that
killed the whole process (the sklearn-fallback ``except`` cannot catch a native AV). Surfaced by fuzz combo c0024.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import (
    _plugin_mi_classif_batch_njit,
    _plugin_mi_classif_njit,
)


def _negative_label_inputs():
    rng = np.random.default_rng(24)
    n = 744
    y = rng.integers(-10, 7, size=n).astype(np.int64)  # non-dense, negative labels
    x = rng.standard_normal(n).astype(np.float64)
    x_const = np.full(n, 3.0, dtype=np.float64)  # degenerate single-value column (c0024 had nuniq==1)
    return x, x_const, y


def test_plugin_mi_classif_njit_negative_labels_no_av():
    x, _, y = _negative_label_inputs()
    mi = _plugin_mi_classif_njit(np.ascontiguousarray(x), np.ascontiguousarray(y), 12)
    assert np.isfinite(mi) and mi >= 0.0


def test_plugin_mi_batch_njit_negative_labels_no_av():
    x, x_const, y = _negative_label_inputs()
    X = np.ascontiguousarray(np.column_stack([x, x_const]))
    mis = _plugin_mi_classif_batch_njit(X, np.ascontiguousarray(y), 12)
    assert mis.shape == (2,)
    assert np.isfinite(mis).all() and (mis >= 0.0).all()


def test_plugin_mi_invariant_to_label_shift():
    """MI is relabel-invariant: shifting y by a constant must not change the result."""
    x, _, y = _negative_label_inputs()
    xc = np.ascontiguousarray(x)
    mi_neg = _plugin_mi_classif_njit(xc, np.ascontiguousarray(y), 12)
    mi_shift = _plugin_mi_classif_njit(xc, np.ascontiguousarray(y - y.min()), 12)
    assert mi_neg == pytest.approx(mi_shift, rel=1e-12)
