"""Audit 2026-11-20 finding 26-F13: every opt-in registry entry resolves, and the fit/apply halves are reachable by name.

A caller that resolves an opt-in function by name must get a callable for each registered name, from the module the registry names.
The test resolves every entry and checks the fit/apply pairs the audit listed.
"""

import pytest

from mlframe.preprocessing.opt_in_slow import OPT_IN_SLOW, resolve_opt_in_slow


@pytest.mark.parametrize("name", sorted(OPT_IN_SLOW))
def test_every_registered_name_resolves_to_a_callable(name):
    """resolve_opt_in_slow returns a callable for each registered name when the caller enables it."""
    assert callable(resolve_opt_in_slow(name, [name]))


def test_fit_apply_halves_are_registered():
    """The fit and apply functions the audit listed are all reachable by name."""
    for name in (
        "fit_outlier_cap_or_missing",
        "apply_outlier_cap_or_missing",
        "fit_rare_category_collapse",
        "apply_rare_category_collapse",
        "fit_missing_indicator_imputation",
        "apply_missing_indicator_imputation",
        "fit_regime_conditioned_median",
        "apply_regime_conditioned_median_fill",
        "fit_outlier_bounds",
        "UnseenCategoryImputer",
    ):
        assert name in OPT_IN_SLOW
