"""The threshold stability report must binarize held-out folds through ``apply_decision_threshold``.

Before the wiring, ``_threshold_stability_report`` inlined ``(y_proba >= t).astype(int)``, so the public
binarizer had no production caller and the held-out score could drift from it unnoticed.
"""

from __future__ import annotations

import numpy as np

import mlframe.calibration.threshold_optimizer as threshold_optimizer
from mlframe.calibration.threshold_optimizer import _threshold_stability_report


def test_threshold_stability_report_routes_heldout_binarization_through_apply_decision_threshold(monkeypatch):
    """Each held-out fold is binarized through apply_decision_threshold."""
    rng = np.random.default_rng(0)
    n = 400
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    y_proba = np.clip(0.3 * y_true + rng.normal(0.35, 0.2, n), 0.0, 1.0)
    thresholds = np.linspace(0.05, 0.95, 19)

    calls = []
    real = threshold_optimizer.apply_decision_threshold

    def spy(proba, threshold, *args, **kwargs):
        """Record the threshold and delegate to the real binarizer."""
        calls.append(float(threshold))
        return real(proba, threshold, *args, **kwargs)

    monkeypatch.setattr(threshold_optimizer, "apply_decision_threshold", spy)

    def f1_like(y, pred):
        """F1 of binary predictions."""
        tp = float(np.sum((y == 1) & (pred == 1)))
        fp = float(np.sum((y == 0) & (pred == 1)))
        fn = float(np.sum((y == 1) & (pred == 0)))
        return 2 * tp / max(2 * tp + fp + fn, 1e-12)

    report = _threshold_stability_report(y_true, y_proba, thresholds, f1_like, n_splits=5, seed=0, stability_cv_threshold=0.5)

    assert len(calls) == 5, f"expected one held-out binarization per fold, got {len(calls)}"
    assert np.isfinite(report["heldout_score_mean"])
