"""A dropped spec's report reason names the stage that actually dropped it (DSC-23, PMT-27a).

``report()`` used to print one fixed sentence for every spec that passed the MI gate and was missing from the final set:
"top_k_after_mi trim / alpha-drift / linear_residual->diff collapse / tiny-model rerank / multi-base dedup". That list
names none of the gates that drop the most specs today - the y-scale holdout gate, the honest RMSE gate, the honest-OOF
floor, structural fragility - all of which record their verdict per spec in the rejection ledger. The reason is now read
from there, so a new gate is covered as soon as it writes a ledger row.
"""

from __future__ import annotations

import pytest

from mlframe.training.composite.discovery._fit import _reason_from_ledger
from mlframe.training.composite.discovery._rejection_ledger import RejectStage, ledger_append, ledger_init

_STAGES = sorted(v for k, v in vars(RejectStage).items() if not k.startswith("_") and isinstance(v, str))


class _Disc:
    """A bare object standing in for the discovery instance the ledger functions attach to."""


@pytest.mark.parametrize("stage", _STAGES)
def test_every_ledger_stage_reaches_the_report_reason(stage):
    """For each stage in the canonical vocabulary, a rejection at that stage is what the report reason names."""
    disc = _Disc()
    ledger_init(disc)
    ledger_append(disc, spec_name="s1", stage=stage, reason="collapsed on the holdout")
    reason = _reason_from_ledger(disc, "s1")
    assert stage in reason and "collapsed on the holdout" in reason


def test_the_latest_stage_wins_and_other_specs_are_not_read():
    """A spec rejected after an earlier advisory row reports the last stage, and another spec's rows are ignored."""
    disc = _Disc()
    ledger_init(disc)
    ledger_append(disc, spec_name="s1", stage=RejectStage.ALPHA_DRIFT, reason="drifting alpha")
    ledger_append(disc, spec_name="s2", stage=RejectStage.AUTO_BASE_DEDUP, reason="duplicate base")
    ledger_append(disc, spec_name="s1", stage=RejectStage.YSCALE_HOLDOUT, reason="negative R2 on the group holdout")
    assert "yscale_holdout" in _reason_from_ledger(disc, "s1")
    assert "duplicate base" not in _reason_from_ledger(disc, "s1")


def test_a_spec_with_no_ledger_row_names_only_the_gates_that_record_none():
    """The fallback sentence covers the two filters that keep no per-spec verdict, and claims nothing beyond them."""
    disc = _Disc()
    ledger_init(disc)
    reason = _reason_from_ledger(disc, "s1")
    assert "top_k_after_mi" in reason and "multi-base dedup" in reason
    for stage in ("yscale_holdout", "honest_rmse", "structural_fragility", "alpha-drift", "tiny-model rerank"):
        assert stage not in reason, f"the fallback must not blame '{stage}', which records its own verdict"
