"""A binned numeric aggregate that lifts a held-out linear fit survives the MI screen that drops it behind its group column.

Each cell statistic is a function of the group column's bins, so by the data-processing inequality the screen finds no information in it beyond the group column; on seed 0 of the
binned-aggregate scenario (``aux ~ N(0, sigma(g))``, ``y = sigma(g) + noise``, with ``sigma`` a V in ``g``) only ``g`` was selected, which a linear model cannot use.
"""

from tests.feature_selection.mrmr.biz_val import _fe_family_survival_common as common
from tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_fe_family_survival import FAMILIES


def test_the_cell_spread_column_is_selected_on_the_seed_where_the_screen_dropped_it():
    """The V-shaped spread of ``aux`` per bin of ``g`` is in the selected output on seed 0."""
    family = next(f for f in FAMILIES if f.name == "binned_numeric_agg")
    fit = common.fit_family(family, 0)
    assert any(str(n).startswith("binagg_std(aux|") for n in fit.model.get_feature_names_out())
