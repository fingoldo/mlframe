"""Regression tests for public-API consistency fixes in calibration.

Covers:
  API15 -- pick_best_calibrator implements the secondary-ECE diagnostic from probs/y.
  API31 -- BinaryPostCalibrator sets classes_/n_features_in_ on fit + exposes predict_proba.
  API32 -- show_classifier_calibration lets unexpected errors propagate (narrowed except).
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- API15
def test_api15_pick_best_calibrator_reports_secondary_ece_when_probs_y_given():
    from mlframe.calibration.policy import pick_best_calibrator

    rng = np.random.default_rng(0)
    n = 400
    oof_y = rng.integers(0, 2, size=n)
    oof_p = np.clip(0.5 + 0.3 * (oof_y - 0.5) + rng.normal(0, 0.1, n), 0.01, 0.99)
    sec_y = rng.integers(0, 2, size=200)
    sec_p = np.clip(0.5 + 0.3 * (sec_y - 0.5) + rng.normal(0, 0.1, 200), 0.01, 0.99)

    res = pick_best_calibrator(
        probs=sec_p,
        y=sec_y,
        oof_probs=oof_p,
        oof_y=oof_y,
        selection="same_oof",
        n_bootstrap=50,
    )
    assert "secondary_ece" in res
    assert res["secondary_ece"] is not None
    assert np.isfinite(res["secondary_ece"])
    assert res["secondary_ece"] >= 0.0


def test_api15_secondary_ece_none_when_probs_y_omitted():
    from mlframe.calibration.policy import pick_best_calibrator

    rng = np.random.default_rng(1)
    n = 300
    oof_y = rng.integers(0, 2, size=n)
    oof_p = np.clip(0.5 + 0.3 * (oof_y - 0.5) + rng.normal(0, 0.1, n), 0.01, 0.99)

    res = pick_best_calibrator(
        probs=None,
        y=None,
        oof_probs=oof_p,
        oof_y=oof_y,
        selection="same_oof",
        n_bootstrap=50,
    )
    assert res["secondary_ece"] is None


# --------------------------------------------------------------------------- API31
class _IdentityCalibrator:
    """Minimal calibrator with fit/transform that returns probs unchanged."""

    def fit(self, p, y):
        self._fitted = True
        return self

    def transform(self, p):
        return np.asarray(p, dtype=np.float64)


def test_api31_binary_postcalibrator_sets_sklearn_attrs_and_predict_proba():
    from mlframe.calibration.post import BinaryPostCalibrator

    cal = BinaryPostCalibrator(calibrator=_IdentityCalibrator())
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    target = np.array([0, 1, 0, 1])

    cal.fit(probs, target)

    # sklearn ClassifierMixin tooling expects these after fit.
    assert hasattr(cal, "classes_")
    np.testing.assert_array_equal(cal.classes_, np.array([0, 1]))
    assert hasattr(cal, "n_features_in_")
    assert cal.n_features_in_ == 2

    # predict_proba alias routes to postcalibrate_probs and returns a 2D matrix.
    out = cal.predict_proba(probs)
    assert out.ndim == 2 and out.shape[1] == 2
    np.testing.assert_allclose(out, cal.postcalibrate_probs(probs))


# --------------------------------------------------------------------------- API32
def test_api32_show_classifier_calibration_propagates_unexpected_error(monkeypatch):
    import mlframe.calibration.quality as quality

    def _boom(*args, **kwargs):
        raise KeyError("unexpected internal bug")

    monkeypatch.setattr(quality, "estimate_calibration_quality_binned", _boom)

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

    # KeyError is NOT in the narrowed (ValueError, ZeroDivisionError, IndexError) set -> must propagate.
    with pytest.raises(KeyError, match="unexpected internal bug"):
        quality.show_classifier_calibration(y_true, y_pred, title="t", nintervals=1, skip_plotting=True)


def test_api32_show_classifier_calibration_swallows_expected_valueerror(monkeypatch):
    import mlframe.calibration.quality as quality

    def _boom(*args, **kwargs):
        raise ValueError("expected data-shape issue")

    monkeypatch.setattr(quality, "estimate_calibration_quality_binned", _boom)

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])

    # ValueError is expected -> logged + returns None (no propagation).
    out = quality.show_classifier_calibration(y_true, y_pred, title="t", nintervals=1, skip_plotting=True)
    assert out is None
