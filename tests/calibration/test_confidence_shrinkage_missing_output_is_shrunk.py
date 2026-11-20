"""A prediction with no confidence is fully shrunk, even when there are no confidences at all (audit 2026-11-20, finding 22-F4)."""
import numpy as np

from mlframe.calibration.confidence_shrinkage import apply_confidence_shrinkage


def test_no_confidences_shrinks_every_output_to_the_neutral_value():
    """With an empty confidence map the early return gave raw predictions; the documented missing-output rule shrinks them fully."""
    preds = {"a": np.array([0.9, 0.1])}
    out = apply_confidence_shrinkage(preds, {}, neutral_value=0.5)
    assert np.allclose(out["a"], 0.5)


def test_present_confidence_still_shrinks_by_its_weight():
    """A confidence at the maximum keeps the prediction intact."""
    preds = {"a": np.array([0.9, 0.1])}
    out = apply_confidence_shrinkage(preds, {"a": 1.0}, neutral_value=0.5, min_confidence=0.0, max_confidence=1.0)
    assert np.allclose(out["a"], [0.9, 0.1])
