"""Audit 2026-11-20 finding 26-F31: a multiclass calibration whose labels cannot be aligned to the columns leaves every column identity.

With labels {1, 2} and three probability columns the column-to-label map is unknown. The old code calibrated column k against label k
(positional), which is the wrong class; each column now has no calibrator, as the comment in the code requires.
"""

import numpy as np

from mlframe.training._calibration_models import _PerClassIsotonicCalibrator
from mlframe.training.configs import TargetTypes


def test_misaligned_labels_leave_every_column_identity():
    """No column is calibrated when the labels do not match the columns: every calibrator is None."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1.0, 1.0, 1.0], size=60)
    y = np.where(rng.random(60) < 0.5, 1, 2)
    model = _PerClassIsotonicCalibrator.fit(probs, y, target_type=TargetTypes.MULTICLASS_CLASSIFICATION, classes=None)
    assert all(calibrator is None for calibrator in model.calibrators.values())
