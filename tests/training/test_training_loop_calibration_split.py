"""Wave 94 (2026-05-21): split _training_loop.py (1014 lines)
into _training_loop.py (now 779 lines) + new _calibration_models.py
(274 lines). The post-hoc calibration wrappers
(_SigmoidAdapter, _PostHocCalibratedModel,
_PerClassIsotonicCalibrator, _PostHocMultiCalibratedModel,
_maybe_apply_posthoc_calibration) moved to the sibling file; the
original re-exports them so existing imports keep working.
"""
from __future__ import annotations

from pathlib import Path


def test_calibration_symbols_still_importable_from_facade() -> None:
    from mlframe.training._training_loop import (
        _SigmoidAdapter,
        _PostHocCalibratedModel,
        _PerClassIsotonicCalibrator,
        _PostHocMultiCalibratedModel,
        _maybe_apply_posthoc_calibration,
    )
    for sym in (
        _SigmoidAdapter,
        _PostHocCalibratedModel,
        _PerClassIsotonicCalibrator,
        _PostHocMultiCalibratedModel,
        _maybe_apply_posthoc_calibration,
    ):
        assert sym is not None


def test_training_loop_ops_still_importable() -> None:
    from mlframe.training._training_loop import (
        _ensure_cb_multilabel_loss,
        _handle_oom_error,
        _train_model_with_fallback,
        _ensure_xgb_classification_objective,
        _maybe_wrap_for_2d_target,
    )
    for fn in (
        _ensure_cb_multilabel_loss,
        _handle_oom_error,
        _train_model_with_fallback,
        _ensure_xgb_classification_objective,
        _maybe_wrap_for_2d_target,
    ):
        assert callable(fn), fn


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "training"
    facade = root / "_training_loop.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"_training_loop.py is {n} lines, still over the 1k threshold"


def test_calibration_module_owns_the_moved_symbols() -> None:
    """Identity: facade and sibling module expose the SAME object."""
    from mlframe.training import _training_loop, _calibration_models
    for name in (
        "_SigmoidAdapter",
        "_PostHocCalibratedModel",
        "_PerClassIsotonicCalibrator",
        "_PostHocMultiCalibratedModel",
        "_maybe_apply_posthoc_calibration",
    ):
        assert getattr(_training_loop, name) is getattr(_calibration_models, name), name


def test_post_hoc_calibrated_model_predict_proba_round_trip() -> None:
    """Functional smoke: wrap a fake base + a fake isotonic calibrator and
    verify the wrapper's predict_proba runs end-to-end."""
    import numpy as np
    from mlframe.training._training_loop import _PostHocCalibratedModel

    class _FakeBase:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.3, 0.7]])

    class _FakeCalibrator:
        def predict(self, p):
            return np.clip(np.asarray(p) * 0.5, 0.0, 1.0)

    wrapped = _PostHocCalibratedModel(_FakeBase(), _FakeCalibrator())
    out = wrapped.predict_proba(np.empty((2, 1)))
    assert out.shape == (2, 2)
    # Class-1 column scaled by 0.5: 0.2 -> 0.1, 0.7 -> 0.35.
    np.testing.assert_allclose(out[:, 1], [0.1, 0.35])
    # Class-0 column = 1 - class-1.
    np.testing.assert_allclose(out[:, 0], [0.9, 0.65])
