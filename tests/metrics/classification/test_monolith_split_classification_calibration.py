"""Sensor: ``_classification_extras.py`` calibration carve into ``_classification_calibration.py``.

Verifies parent + ``metrics.core`` re-export identity AND calls into the moved
bodies (Hosmer-Lemeshow + Accuracy Ratio), forcing the njit compile.
"""

from __future__ import annotations

import numpy as np


def test_calibration_reexport_identity():
    from mlframe.metrics.classification import _classification_calibration as sib
    from mlframe.metrics.classification import _classification_extras as parent
    from mlframe.metrics import core

    for nm in ("hosmer_lemeshow_test", "accuracy_ratio"):
        assert getattr(parent, nm) is getattr(sib, nm)
        assert getattr(core, nm) is getattr(sib, nm)


def test_hosmer_lemeshow_body_callable():
    from mlframe.metrics.classification._classification_calibration import hosmer_lemeshow_test

    rng = np.random.default_rng(0)
    ys = rng.random(500)
    yt = (rng.random(500) < ys).astype(int)
    h, p, dof = hosmer_lemeshow_test(yt, ys, n_groups=10)
    assert np.isfinite(h) and np.isfinite(p) and dof == 8


def test_accuracy_ratio_body_matches_gini():
    from sklearn.metrics import roc_auc_score

    from mlframe.metrics.classification._classification_calibration import accuracy_ratio

    rng = np.random.default_rng(1)
    ys = rng.random(500)
    yt = (rng.random(500) < ys).astype(int)
    ar = accuracy_ratio(yt, ys)
    assert abs(ar - (2 * roc_auc_score(yt, ys) - 1)) < 0.02


def test_accuracy_ratio_single_class_returns_nan():
    from mlframe.metrics.classification._classification_calibration import accuracy_ratio

    assert np.isnan(accuracy_ratio(np.zeros(10, dtype=int), np.random.random(10)))
