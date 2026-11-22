"""Cluster 28 regression: string class labels (object dtype) are not treated as multilabel (audit 2026-11-20, finding 28-F2)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training import evaluation
from mlframe.training.configs import TargetTypes
from mlframe.training.trainer import _PerClassIsotonicCalibrator


class _Stop(Exception):
    """Raised by the patched calibrator to end the call once the target type has been captured."""


def test_string_multiclass_labels_are_multiclass(monkeypatch):
    """Object-dtype 1-D string labels must be MULTICLASS, not MULTILABEL (28-F2)."""
    captured = {}

    def _fake_fit(cls, probs, y, target_type, *a, **kw):
        """Record the target type the branch chose, then stop the call."""
        captured["target_type"] = target_type
        raise _Stop

    monkeypatch.setattr(_PerClassIsotonicCalibrator, "fit", classmethod(_fake_fit))
    labels = pd.Series(np.array(["a", "b", "c", "a", "b", "c"], dtype=object))
    probs = np.full((6, 3), 1 / 3)
    with pytest.raises(_Stop):
        evaluation._post_calibrate_multi_output(
            model=object(),
            test_preds=np.zeros(6),
            test_probs=probs,
            val_preds=np.zeros(6),
            val_probs=probs,
            columns=["a", "b", "c"],
            pre_pipeline=None,
            metrics={},
            target_series=labels,
            test_idx=np.arange(6),
            calib_probs=probs,
            calib_target=np.array([0, 1, 2, 0, 1, 2]),
        )
    assert captured["target_type"] == TargetTypes.MULTICLASS_CLASSIFICATION
