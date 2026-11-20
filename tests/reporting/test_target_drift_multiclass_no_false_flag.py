"""A multiclass target gets no class-1 drift flag (audit 2026-11-20, finding 20-F5)."""
import numpy as np

from mlframe.reporting.charts._error_analysis_splits import _target_drift_verdict


def test_multiclass_drift_is_not_flagged_by_the_binary_test():
    """Three classes with a shift in the mix: the binary proportion test used to clip its SE to 0 and flag nothing, or a pooled rate above 1.
    The verdict must say the test does not apply, with no WARNING."""
    rng = np.random.default_rng(0)
    train = rng.integers(0, 3, 3000).astype(float)
    test = rng.integers(0, 3, 3000).astype(float)
    verdict = _target_drift_verdict({"train": train, "test": test}, train_key="train", task="classification")
    assert "does not apply" in verdict
    assert "WARNING" not in verdict
