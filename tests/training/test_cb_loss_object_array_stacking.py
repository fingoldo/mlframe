"""Regression: the CatBoost loss auto-config helpers must stack a 1-D OBJECT-array
target (each element a row-vector, as the fuzz object-carrier produces) into 2-D and
configure the right multi-dim loss.

Guards the ``np.array(<object-array>.tolist())`` stacking optimization in
``_ensure_cb_mtr_loss`` / ``_ensure_cb_multilabel_loss`` -- the object-array branch
that the usual 2-D-array tests skip (they never enter the ``dtype == object`` path).
"""
from __future__ import annotations

import numpy as np
import pytest


def _as_object_rows(rows_2d: np.ndarray) -> np.ndarray:
    """Wrap a 2-D array as a 1-D object array of per-row vectors."""
    n = rows_2d.shape[0]
    a = np.empty(n, dtype=object)
    for i in range(n):
        a[i] = rows_2d[i]
    return a


def test_multilabel_loss_set_from_object_array_target():
    cb = pytest.importorskip("catboost")
    from mlframe.training._training_loop import _ensure_cb_multilabel_loss

    model = cb.CatBoostClassifier(iterations=2, verbose=0)  # no loss_function wired
    Y = (np.random.default_rng(0).random((60, 3)) < 0.4).astype(np.int64)  # multilabel
    _ensure_cb_multilabel_loss(model, _as_object_rows(Y))
    assert model.get_params().get("loss_function") == "MultiLogloss", (
        "object-array multilabel target must be stacked to 2-D and configured MultiLogloss"
    )


def test_mtr_loss_set_from_object_array_target():
    cb = pytest.importorskip("catboost")
    from mlframe.training._training_loop import _ensure_cb_mtr_loss

    model = cb.CatBoostRegressor(iterations=2, verbose=0)  # no loss_function wired
    Y = np.random.default_rng(1).normal(size=(60, 3))  # 2-D continuous multi-target
    _ensure_cb_mtr_loss(model, _as_object_rows(Y))
    assert model.get_params().get("loss_function") == "MultiRMSE", (
        "object-array multi-target continuous must be stacked to 2-D and configured MultiRMSE"
    )


def test_object_array_stack_matches_np_stack_reference():
    """The optimization's core invariant: np.array(obj.tolist()) equals the prior
    np.stack listcomp for uniform-width rows (int and float)."""
    rng = np.random.default_rng(7)
    for rows in (
        (rng.random((100, 4)) < 0.4).astype(np.int64),
        rng.normal(size=(100, 4)),
    ):
        obj = _as_object_rows(rows)
        ref = np.stack([np.asarray(c) for c in obj], axis=0)
        fast = np.array(obj.tolist())
        assert np.array_equal(ref, fast) and ref.dtype == fast.dtype
