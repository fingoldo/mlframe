"""Fold-to-fold dispersion of thresholds and rescale factors is the sample std (tracker 2026-11-02 wave 13, FORM13-2, FORM13-3)."""

import numpy as np
import pytest

from mlframe.calibration import asymmetric_rescale as ar
from mlframe.calibration import threshold_optimizer as to


def test_threshold_stability_cv_uses_ddof1(monkeypatch):
    """Fold thresholds .30 .35 .40 .45 .50: CV 0.198 with the sample std (the ddof=0 value 0.177 passed a 0.19 gate)."""
    values = iter([0.30, 0.35, 0.40, 0.45, 0.50])
    monkeypatch.setattr(to, "_best_threshold_for", lambda *a, **k: (next(values), 0.0))
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    out = to._threshold_stability_report(y, p, np.linspace(0, 1, 11), lambda a, b: 0.0, 5, 0, 0.19)
    assert out["cv"] == pytest.approx(np.std([0.30, 0.35, 0.40, 0.45, 0.50], ddof=1) / 0.40)
    assert out["is_stable"] is False


def test_rescale_factor_cv_uses_ddof1(monkeypatch):
    """Fold factors .9 1.0 1.1: CV 0.100 with the sample std, not 0.0816."""
    values = iter([0.9, 1.0, 1.1])
    monkeypatch.setattr(ar, "fit_asymmetric_rescale", lambda *a, **k: {"factor": next(values)})
    rng = np.random.default_rng(0)
    y = rng.random(60)
    out = ar.cross_validate_asymmetric_rescale(y, y + 0.1, lambda a, b: 0.0, n_folds=3)
    assert out["factor_cv"] == pytest.approx(0.1)
