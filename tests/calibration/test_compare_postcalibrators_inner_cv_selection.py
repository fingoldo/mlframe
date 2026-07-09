"""Regression test for P0-1 (see audit/mlframe_audit_2026_07/calibration.md): compare_postcalibrators
fit every candidate calibrator and then self-evaluated it on the SAME rows it was fit on (the sole
in-repo caller, train_postcalibrators, always passes oos_probs=None). This is the same "same_oof"
selection-optimism bug class policy.py::pick_best_calibrator already diagnosed and fixed via
selection="inner_cv" -- a flexible calibrator can drive its self-eval score toward "perfect" purely by
memorising the data it saw, without generalising at all. Fixed by porting the same inner-CV held-out
evaluation approach: each candidate is fit on a fold's complement and scored on the held-out fold.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np


def _synth(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = (p1 + rng.normal(0, 0.1, n) > 0.5).astype(int)
    return probs, target


class _MemorizingCalibrator:
    """A degenerate wrapped calibrator whose 'prediction' is the exact fit-time value.

    Simulates a maximally-flexible calibrator (Isotonic on tiny bins, etc.) that can drive its
    in-sample (self-eval) score to a perfect result purely by memorising the fit data, but which
    does NOT generalise to rows it was not fit on (it just returns 0.5 for unseen probability values).
    """

    def fit(self, X, y):
        X1 = np.asarray(X)[:, 1] if np.asarray(X).ndim == 2 else np.asarray(X)
        self._memo = dict(zip(np.round(X1, 12).tolist(), np.asarray(y, dtype=float).tolist()))
        return self

    def transform(self, X):
        X1 = np.asarray(X)[:, 1] if np.asarray(X).ndim == 2 else np.asarray(X)
        return np.array([self._memo.get(round(float(x), 12), 0.5) for x in X1])


def test_compare_postcalibrators_inner_cv_is_not_optimistic_like_self_eval():
    """A memorizing calibrator scores perfectly under self_eval (it just replays the fit-time target)
    but must NOT do so under the default inner_cv selection, since inner_cv scores it on held-out rows
    it never saw during that fold's fit."""
    from mlframe.calibration.post import compare_postcalibrators, named_calibrator

    probs, target = _synth(n=400)
    fake_calibrators = [named_calibrator(_MemorizingCalibrator(), name="Memo", lib="test", transform_method_name="transform")]

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        metrics_self, _, failed_self = compare_postcalibrators(
            model_name="m", columns=["y"], calib_probs=probs, calib_target=target,
            oos_probs=None, oos_target=None, calib_type="calib", include_patterns=["test"],
            selection="self_eval",
        )
        metrics_cv, _, failed_cv = compare_postcalibrators(
            model_name="m", columns=["y"], calib_probs=probs, calib_target=target,
            oos_probs=None, oos_target=None, calib_type="calib", include_patterns=["test"],
            selection="inner_cv",
        )

    assert failed_self == {} and failed_cv == {}
    ice_self = metrics_self.loc["test.Memo", "ice"]
    ice_cv = metrics_cv.loc["test.Memo", "ice"]
    # The memorizing calibrator's self-eval "ice" (a calibration-error style metric) is near-perfect
    # (it just replays the training target); inner-CV held-out evaluation must NOT be as optimistic,
    # since the fold-complement fit never saw the held-out rows' exact probability values.
    assert ice_cv > ice_self + 1e-6, (
        f"inner_cv held-out score ({ice_cv}) should be materially worse than the optimistic self_eval "
        f"score ({ice_self}) for a calibrator that only memorises fit-time rows -- pre-fix, "
        f"compare_postcalibrators always self-evaluated (oos_probs=None is the only real call path), "
        "so this distinction did not exist."
    )
