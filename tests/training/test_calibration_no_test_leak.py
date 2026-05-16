"""Tests for Fix 2: post_calibrate_model never fits on test rows.

The pre-fix code sliced ``test_probs[:calib_set_size]`` and fit the meta-calibrator on it -- the calibrator
ingested test-set rows, tuned to them, and the residual test report was no longer an honest holdout (selection bias
via tuning surface). The fix:

1. Drops the test-slice fit path entirely; binary path now requires either ``calib_probs+calib_target`` (OOF-train
   probs preferred) OR ``model.oof_probs`` stamped by the trainer.
2. Adds an explicit guard: if ``calib_idx`` is passed AND any row in it overlaps with ``test_idx``, raise
   ``ValueError("calibration must not touch test_idx rows; got <N> test rows in calibrator input")`` before any
   ``.fit(...)`` call.

These tests cover both branches of the guard plus the clean-OOF happy path.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class _StubMetaModel:
    """Minimal stand-in for the CatBoost meta-calibrator: records the (X, y) it was fit on so the test can assert
    the fit input never includes test rows."""

    def __init__(self):
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X)
        self.fit_y = np.asarray(y)
        return self

    def predict_proba(self, X):
        # Return a non-degenerate (N, 2) array so report_model_perf doesn't crash on shape.
        X_arr = np.asarray(X).reshape(-1, 1)
        # Map raw prob -> identity sigmoid (close to the input). Sufficient for shape contracts.
        p1 = np.clip(X_arr[:, 0], 1e-6, 1 - 1e-6)
        return np.stack([1.0 - p1, p1], axis=1)


def _make_configs_stub():
    """post_calibrate_model touches configs.integral_calibration_error; supply a trivial scalar metric stand-in.

    Real ICE signature accepted by ``report_probabilistic_model_perf`` is ``(y_true=..., y_score=...)`` (keyword-only
    on the call site at _reporting.py:905). The stub must accept both positional and keyword forms.
    """
    def _metric(*args, **kwargs):
        return 0.0
    return SimpleNamespace(integral_calibration_error=_metric)


def test_overlapping_calib_idx_and_test_idx_raises_before_fit():
    """When calib_idx shares any row with test_idx, the guard fires BEFORE meta_model.fit can be called."""
    from mlframe.training.evaluation import post_calibrate_model

    rng = np.random.default_rng(0)
    n_total = 60
    test_idx = np.arange(40, 60)            # rows 40..59
    val_idx = np.arange(20, 40)
    calib_idx = np.array([39, 40, 41, 42])  # rows 40, 41, 42 overlap with test -> 3 leaking rows

    test_probs = rng.uniform(size=(20, 2))
    val_probs = rng.uniform(size=(20, 2))
    test_preds = (test_probs[:, 1] > 0.5).astype(int)
    val_preds = (val_probs[:, 1] > 0.5).astype(int)
    target_series = pd.Series(rng.integers(0, 2, size=n_total))

    meta_model = _StubMetaModel()
    original_model = (
        SimpleNamespace(),  # model -- not used by the guard
        test_preds,
        test_probs,
        val_preds,
        val_probs,
        ["c0", "c1"],
        None,
        {},
    )

    with pytest.raises(ValueError, match="calibration must not touch test_idx rows"):
        post_calibrate_model(
            original_model=original_model,
            target_series=target_series,
            target_label_encoder=None,
            val_idx=val_idx,
            test_idx=test_idx,
            configs=_make_configs_stub(),
            meta_model=meta_model,
            calib_idx=calib_idx,
            # Provide calib_probs/calib_target so the function can otherwise proceed; the guard fires first regardless.
            calib_probs=rng.uniform(size=(4, 2)),
            calib_target=rng.integers(0, 2, size=4),
        )

    # Guard fires BEFORE meta_model.fit is invoked: fit_X must still be its sentinel None.
    assert meta_model.fit_X is None, "meta_model.fit was called despite overlapping calib_idx + test_idx"


def test_clean_oof_calib_fits_without_assertion():
    """With a disjoint calib_idx + valid (calib_probs, calib_target), the calibrator fits and never sees test rows."""
    from mlframe.training.evaluation import post_calibrate_model

    rng = np.random.default_rng(1)
    n_total = 100
    test_idx = np.arange(80, 100)
    val_idx = np.arange(60, 80)
    calib_idx = np.arange(0, 30)  # disjoint from test_idx

    test_probs = rng.uniform(size=(20, 2))
    val_probs = rng.uniform(size=(20, 2))
    test_preds = (test_probs[:, 1] > 0.5).astype(int)
    val_preds = (val_probs[:, 1] > 0.5).astype(int)
    target_series = pd.Series(rng.integers(0, 2, size=n_total))

    # OOF-train style: M calibration rows, (M, 2) probs + (M,) targets.
    calib_probs = rng.uniform(size=(30, 2))
    calib_target = rng.integers(0, 2, size=30)

    meta_model = _StubMetaModel()
    original_model = (
        SimpleNamespace(),
        test_preds,
        test_probs,
        val_preds,
        val_probs,
        ["c0", "c1"],
        None,
        {},
    )

    result = post_calibrate_model(
        original_model=original_model,
        target_series=target_series,
        target_label_encoder=None,
        val_idx=val_idx,
        test_idx=test_idx,
        configs=_make_configs_stub(),
        meta_model=meta_model,
        calib_idx=calib_idx,
        calib_probs=calib_probs,
        calib_target=calib_target,
    )

    assert meta_model.fit_X is not None, "calibrator should have been fit"
    assert meta_model.fit_X.shape[0] == calib_probs.shape[0]
    # Fit input row count equals the calibration source -- no test-slice rows leaked in.
    assert meta_model.fit_X.shape[0] != test_probs.shape[0]
    # Returned tuple shape is preserved.
    assert len(result) == 8
