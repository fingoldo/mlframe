"""Every arm's cost must be MEASURED, because the axis is only as good as its worst-covered arm.

`n_model_fits` is the pre-registered primary cost axis, chosen because it is deterministic while
wall-clock on this host is not. It is only meaningful if every arm reports one. An arm whose fits nobody
counted records `None` -- deliberately, since unmeasured is not zero -- but a cost table with holes in it
ranks the arms that were measured against the arms that were not, and a free-looking arm might simply be
an uninstrumented one.

Measured across the whole roster: every arm currently reports a COUNTED figure, from `variance-sort` at
zero fits to `shap-proxied` at a hundred and fifty. This test pins that, so a newly added arm whose
estimator class the counter cannot see fails here rather than appearing in the atlas as cheap.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._arms import FITS_SOURCE_DECLARED, FITS_SOURCE_NOT_MEASURED, build_arm_roster
from mlframe.feature_selection._benchmarks.fs_hybrid._scm_beds import build_scm_bed

#: Deliberately tiny. This test is about whether a number EXISTS and where it came from, never about its
#: size, so the bed only has to be big enough for every arm to complete on it.
ROWS = 400
BED = "linear_k5_p50"


@pytest.fixture(scope="module")
def arm_costs() -> Dict[str, Dict[str, Any]]:
    """Run every roster arm once and collect what each reported about its own cost."""
    frame, labels, _truth = build_scm_bed(BED, seed=0, n_samples=ROWS)
    out: Dict[str, Dict[str, Any]] = {}
    for name, factory in build_arm_roster(int(frame.shape[1]), random_state=0).items():
        result = factory().run(frame, np.asarray(labels))
        out[name] = {
            "fits": result.n_model_fits,
            "source": result.provenance.get("n_model_fits_source"),
            "declared": result.provenance.get("n_model_fits_declared"),
            "counted": result.provenance.get("n_model_fits_counted"),
        }
    return out


@pytest.mark.slow
def test_no_arm_reports_its_cost_as_unmeasured(arm_costs: Dict[str, Dict[str, Any]]) -> None:
    """An uninstrumented arm looks free, and free is the most flattering thing a cost table can say."""
    unmeasured: List[str] = [name for name, record in arm_costs.items() if record["source"] == FITS_SOURCE_NOT_MEASURED]

    assert not unmeasured, f"these arms report no measurable fit count, so the cost axis has holes exactly where they sit: {unmeasured}"


@pytest.mark.slow
def test_every_arm_reports_a_non_negative_integer_fit_count(arm_costs: Dict[str, Dict[str, Any]]) -> None:
    """`None` means unmeasured and is caught above; anything else must be a usable count."""
    assert arm_costs.items()
    for name, record in arm_costs.items():
        assert record["fits"] is not None, f"{name} reports no fit count at all"
        assert int(record["fits"]) >= 0, f"{name} reports a negative fit count: {record['fits']}"


@pytest.mark.slow
def test_the_cost_axis_still_separates_the_cheap_arms_from_the_expensive_ones(arm_costs: Dict[str, Dict[str, Any]]) -> None:
    """A cost axis on which every arm scores the same has stopped measuring anything.

    The separation is the reason this axis is primary. The marginal filters fit no model at all and the
    wrapper arms fit dozens; if a change collapsed that, the atlas's whole cost section would be reporting
    the downstream panel, which every arm pays identically.
    """
    counts = [int(record["fits"]) for record in arm_costs.values() if record["fits"] is not None]

    assert min(counts) == 0, "no arm is free, which means the counter is attributing the shared panel to the arms"
    assert max(counts) >= 10, f"the most expensive arm reports only {max(counts)} fits, so the axis no longer separates wrappers from filters"


@pytest.mark.slow
def test_a_declared_count_that_disagrees_with_the_counted_one_is_still_recorded(arm_costs: Dict[str, Dict[str, Any]]) -> None:
    """Both numbers survive into provenance, so a disagreement is auditable rather than silently resolved.

    They do disagree, for real reasons. `boruta-shap` declares one fit per trial and costs two.
    `lars-order` declares one and costs none, because a LARS path is not a model fit. Keeping both means a
    reader can see which figure the table used and why.

    The counted figure wins, with one exception written into `_resolve_fit_count`: a counted ZERO against a
    positive declaration. That is the CatBoost arms, whose selection runs in native code the counter cannot
    see, so the zero measures the counter's blindness rather than the arm's cost -- publishing it would put
    the most expensive arms at the top of the cost table. The declared figure is used there, and the source
    field says so.
    """
    disagreements = {name: record for name, record in arm_costs.items() if record["declared"] is not None and record["counted"] is not None and record["declared"] != record["counted"]}

    assert arm_costs.items()
    assert disagreements, "the roster has no declared-vs-counted disagreement, so this contract is untested"
    for name, record in disagreements.items():
        if record["counted"] == 0 and record["declared"]:
            assert record["fits"] == record["declared"], f"{name} counted zero against a declared {record['declared']} and must publish the declaration"
            assert record["source"] == FITS_SOURCE_DECLARED, f"{name} used its declared count but reports the source as {record['source']!r}"
            continue
        assert record["fits"] == record["counted"], f"{name} publishes {record['fits']} while counting {record['counted']}: the counted figure is the measured one and must win"
