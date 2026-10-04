"""Finalize calibration steps: threshold units, shrinkage consistency, and NaN-safe OOF confidence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from sklearn.isotonic import IsotonicRegression

from mlframe.calibration.confidence_shrinkage import apply_confidence_shrinkage, compute_oof_confidence
from mlframe.training._calibration_models import _PostHocCalibratedModel
from mlframe.training.core._phase_finalize_calibration import (
    _apply_confidence_shrinkage_to_regression,
    _optimize_decision_threshold_on_calib_slice,
)


def _calib_entry(wrapped: bool):
    """A binary entry whose raw probabilities are compressed into [0.1, 0.4] while isotonic maps them onto the full range."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    raw = 0.1 + 0.3 * np.clip(0.5 * y + rng.normal(0.25, 0.2, size=400), 0, 1)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    model = _PostHocCalibratedModel(object(), iso) if wrapped else object()
    return SimpleNamespace(model=model, calib_probs=np.column_stack([1 - raw, raw]), calib_target=y, model_name="m"), iso


def _threshold_ctx(entry):
    """Minimal context with the threshold optimizer switched on."""
    return SimpleNamespace(
        behavior_config=SimpleNamespace(auto_optimize_threshold=True, threshold_optimizer_kwargs=None),
        models={"BINARY_CLASSIFICATION": {"t": [entry]}}, metadata={}, verbose=0,
    )


def test_threshold_is_stored_in_units_of_the_shipped_calibrated_model():
    """With an isotonic-wrapped model the stored threshold lies in the calibrated range and the report names its scale; raw models say raw."""
    entry, _ = _calib_entry(wrapped=True)
    ctx = _threshold_ctx(entry)
    _optimize_decision_threshold_on_calib_slice(ctx)
    rep = next(iter(ctx.metadata["decision_threshold"].values()))
    assert rep["probability_scale"] == "posthoc_calibrated_proba"
    assert 0.0 <= rep["best_threshold"] <= 1.0
    entry_raw, _ = _calib_entry(wrapped=False)
    ctx_raw = _threshold_ctx(entry_raw)
    _optimize_decision_threshold_on_calib_slice(ctx_raw)
    assert next(iter(ctx_raw.metadata["decision_threshold"].values()))["probability_scale"] == "raw_base_proba"


def test_calibrated_threshold_separates_shipped_probabilities_better_than_raw_unit_threshold():
    """Applying the stored threshold to the isotonic-calibrated probabilities reproduces the optimised balanced accuracy on the calib slice (a raw-unit threshold would not)."""
    from sklearn.metrics import balanced_accuracy_score

    entry, iso = _calib_entry(wrapped=True)
    ctx = _threshold_ctx(entry)
    _optimize_decision_threshold_on_calib_slice(ctx)
    rep = next(iter(ctx.metadata["decision_threshold"].values()))
    shipped = np.clip(iso.predict(entry.calib_probs[:, 1]), 0, 1)
    got = balanced_accuracy_score(entry.calib_target, shipped >= rep["best_threshold"])
    assert abs(got - rep["best_score"]) < 0.02


def _shrink_ctx():
    """Context with one regression entry carrying OOF preds, targets, val and test preds."""
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, size=n).astype(float)
    entry = SimpleNamespace(
        test_probs=None, oof_preds=0.3 + 0.2 * y + rng.normal(0, 0.1, n), train_target=y,
        test_preds=np.linspace(0.1, 0.9, 50), val_preds=np.linspace(0.2, 0.8, 40), model_name="r",
    )
    ctx = SimpleNamespace(
        regression_calibration_config=SimpleNamespace(apply_confidence_shrinkage=True, confidence_shrinkage_kwargs={"min_confidence": 1.0, "max_confidence": 3.0}),
        models={"REGRESSION": {"t": [entry]}}, metadata={}, verbose=0,
    )
    return ctx, entry


def test_shrinkage_tags_pre_shrinkage_metrics_and_treats_val_like_test():
    """Shrinkage moves val_preds with the same confidence as test_preds, keeps the originals, and tags the metrics as pre-shrinkage."""
    ctx, entry = _shrink_ctx()
    test_before, val_before = entry.test_preds.copy(), entry.val_preds.copy()
    _apply_confidence_shrinkage_to_regression(ctx)
    info = next(iter(ctx.metadata["confidence_shrinkage"].values()))
    assert info["metrics_are_pre_shrinkage"] is True
    np.testing.assert_array_equal(entry.test_preds_pre_shrinkage, test_before)
    np.testing.assert_array_equal(entry.val_preds_pre_shrinkage, val_before)
    conf = info["confidence"]
    kw = {"min_confidence": 1.0, "max_confidence": 3.0}
    np.testing.assert_allclose(entry.val_preds, apply_confidence_shrinkage({"k": val_before}, {"k": conf}, **kw)["k"])
    assert not np.allclose(entry.val_preds, val_before)


def test_oof_confidence_ignores_nan_warmup_rows():
    """NaN warm-up rows no longer turn the confidence (and every shrunk prediction) into NaN."""
    o = np.array([np.nan, np.nan, 0.9, 0.2, 0.8, 0.1])
    y = np.array([1, 0, 1, 0, 1, 0])
    conf = compute_oof_confidence(o, y)
    assert np.isfinite(conf)
    assert conf == compute_oof_confidence(o[2:], y[2:])
    seg = compute_oof_confidence(o, y, segment_ids=np.array([0, 0, 1, 1, 1, 1]))
    assert all(np.isfinite(v) for v in seg.values())
