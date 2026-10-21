"""The information-based selection compaction must not drop a hinge leg that the linear-usability protection re-added.

A monotone leg of a selected source adds no binned joint information (data-processing inequality), so ``prune_redundant`` removed every leg on seed 5 of the hinge scenario
(``y = relu(x0 - 0.5)``) although the held-out linear gain of the legs was 0.04 to 0.08.
"""

from tests.feature_selection.mrmr.biz_val import _fe_family_survival_common as common
from tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_fe_family_survival import FAMILIES


def test_relu_leg_survives_the_compaction_when_the_raw_source_is_selected():
    """The hinge scenario on seed 5 ends with a relu leg among the engineered recipes."""
    family = next(f for f in FAMILIES if f.name == "hinge")
    fit = common.fit_family(family, 5)
    assert any("__relu_" in str(r.name) for r in fit.model._engineered_recipes_)
