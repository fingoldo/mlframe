"""Regression test for P0-2 (see audit/mlframe_audit_2026_07/calibration.md): neither the fit nor the
postcalibrate_probs call in compare_postcalibrators's per-calibrator loop was wrapped in try/except.
get_postcalibrators builds a zoo of ~20+ third-party calibrators (netcal/dirichletcal/pycalib), several
of which are documented elsewhere as unstable; if any ONE calibrator's fit/predict raised, the whole
comparison call died and every already-fit calibrator's results were discarded, not just the failing
one. Fixed by wrapping each candidate's fit/predict in its own try/except, logging the failure, and
continuing with the remaining candidates -- the failure is reported (not silently dropped) via the new
``failed_calibrators`` return value.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np


def _synth(seed: int = 0, n: int = 300):
    """Builds seeded synthetic test data; returns ``(probs, target)``."""
    rng = np.random.default_rng(seed)
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = (p1 + rng.normal(0, 0.1, n) > 0.5).astype(int)
    return probs, target


def test_compare_postcalibrators_one_bad_calibrator_does_not_kill_the_rest():
    """A calibrator that raises during fit must not abort the whole comparison loop."""
    from mlframe.calibration.post import compare_postcalibrators, named_calibrator
    from sklearn.isotonic import IsotonicRegression

    class _BoomCalibrator:
        """Groups tests covering BoomCalibrator."""
        def fit(self, X, y):
            """Always raises ``RuntimeError('boom: simulated third-party calibrator failure')``."""
            raise RuntimeError("boom: simulated third-party calibrator failure")

        def transform(self, X):
            """Always raises ``RuntimeError('unreachable')``."""
            raise RuntimeError("unreachable")

    probs, target = _synth(n=300)
    fake_calibrators = [
        named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn"),
        named_calibrator(_BoomCalibrator(), name="Boom", lib="test", transform_method_name="transform"),
    ]

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        metrics_df, calibrators, failed = compare_postcalibrators(
            model_name="m",
            columns=["y"],
            calib_probs=probs,
            calib_target=target,
            oos_probs=None,
            oos_target=None,
            calib_type="calib",
            include_patterns=["sklearn", "test"],
        )

    assert metrics_df is not None, "pre-fix: one raising calibrator propagated the exception and the whole call died"
    assert "sklearn.Iso" in metrics_df.index, "the good calibrator's results must survive a sibling's failure"
    assert "sklearn.Iso" in calibrators, "the good calibrator must still be returned as a fitted object"
    assert "test.Boom" in failed, "the failing calibrator must be explicitly reported, not silently dropped"
    assert "test.Boom" not in metrics_df.index
    assert "test.Boom" not in calibrators
