"""A logloss or accuracy ceiling on a binary target is rejected at spec time, not silently ignored (audit 2026-11-20, finding 04-F2)."""
import pytest
from pydantic import ValidationError

from mlframe.data.datasets.spec import CeilingTarget, LinkSpec, TargetSpec


@pytest.mark.parametrize("metric", ["logloss", "accuracy"])
def test_binary_target_rejects_a_non_auc_ceiling(metric):
    """The generator bisects only on AUC for binary targets, so these ceilings used to be dropped without a word."""
    with pytest.raises(ValidationError, match="not implemented"):
        TargetSpec(name="y", kind="binary", link=LinkSpec(coefficients={"x1": 1.0}), calibrate_to=CeilingTarget(metric=metric, value=0.7))


def test_binary_target_accepts_auc_ceiling():
    """The supported AUC ceiling still builds."""
    spec = TargetSpec(name="y", kind="binary", link=LinkSpec(coefficients={"x1": 1.0}), calibrate_to=CeilingTarget(metric="auc", value=0.8))
    assert spec.calibrate_to.metric == "auc"
