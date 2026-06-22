"""Regression: BinaryPostCalibrator dispatched on the substring
`"VennAbersCalibrator" in type(self.calibrator).__name__`, so a SUBCLASS of VennAbersCalibrator
whose class name does not contain that exact substring took the wrong (generic-calibrator)
branch. Dispatch now uses isinstance against the imported class."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("venn_abers")

from venn_abers import VennAbersCalibrator

from mlframe.calibration.post import BinaryPostCalibrator


class MyVennAbers(VennAbersCalibrator):
    """Subclass whose __name__ ('MyVennAbers') lacks the 'VennAbersCalibrator' substring."""


def test_subclass_of_venn_abers_takes_venn_abers_branch():
    cal = MyVennAbers()
    assert BinaryPostCalibrator._is_venn_abers(cal) is True

    pc = BinaryPostCalibrator(calibrator=cal)
    calib_probs = np.linspace(0.05, 0.95, 40)
    calib_target = (calib_probs > 0.5).astype(int)
    # The VennAbers branch stores p_cal / y_cal rather than calling a fit method. Pre-fix the
    # substring check missed the subclass and tried getattr(cal, 'fit')(...), a different path.
    pc.fit(calib_probs, calib_target)
    assert hasattr(pc, "p_cal") and hasattr(pc, "y_cal")
    np.testing.assert_array_equal(pc.p_cal, calib_probs)


def test_plain_object_not_treated_as_venn_abers():
    class NotACalibrator:
        pass

    assert BinaryPostCalibrator._is_venn_abers(NotACalibrator()) is False
