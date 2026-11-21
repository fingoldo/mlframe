"""Audit 2026-11-20 finding 04-F16: the shortfall caveat fires when the calibration gap exceeds the spec's own ceiling tolerance."""

import pytest

from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios._mixed_types import id_trap_spec


@pytest.mark.parametrize("tolerance", [0.0001, 0.5])
def test_caveat_is_present_exactly_when_the_gap_exceeds_the_declared_tolerance(tolerance):
    """The shortfall caveat is present iff |achieved - requested| exceeds the target's own tolerance, for a tiny and a huge tolerance."""
    spec = id_trap_spec(seed=0)
    target = spec.targets[0].model_copy(update={"calibrate_to": spec.targets[0].calibrate_to.model_copy(update={"tolerance": tolerance})})
    spec = spec.model_copy(update={"targets": (target,)})
    result = generate(spec)
    requested, achieved = result.calibration["requested"], result.calibration["achieved_auc"]
    gap = abs(float(achieved) - float(requested))
    has_caveat = any("was not reachable" in c for c in result.truth.caveats)
    assert has_caveat == (gap > tolerance)
