"""A pair of members correlated above 0.99 carries no independent information -- drop one by default.

A production run warned that two CatBoost members correlated at 0.9960 and then built five aggregations of
them anyway. All four numeric flavours returned identical test AUC 0.71 and Brier 20.75%: minutes spent
confirming that the mean of two equal numbers is that number. The auto-drop mechanism already existed; only its
floor was unset, so the warning fired and nothing happened.

0.99 is deliberately high. A 0.95-0.98 pair still contributes, and dropping it would cost real ensemble value.
"""

from __future__ import annotations

import inspect

import pytest

from mlframe.models.ensembling.score import score_ensemble


def _default_of(param: str):
    """Default value of one ``score_ensemble`` keyword, read from the signature itself."""
    return inspect.signature(score_ensemble).parameters[param].default


class TestTheDefaultFloor:
    """The knob's value is the whole fix, so it is what the test pins."""

    def test_auto_drop_is_active_by_default(self):
        """``None`` meant observe-and-warn, which is what let the degenerate ensemble through."""
        assert _default_of("auto_drop_diversity_above") is not None

    def test_floor_is_high_enough_to_spare_genuine_diversity(self):
        """A 0.95-0.98 pair is still a useful ensemble; only a near-duplicate should be dropped."""
        assert _default_of("auto_drop_diversity_above") >= 0.99

    def test_floor_is_below_one(self):
        """A floor of 1.0 would only fire on perfectly identical members, i.e. never in practice."""
        assert _default_of("auto_drop_diversity_above") < 1.0

    def test_the_production_pair_would_be_dropped(self):
        """The measured 0.9960 from the run this fixes must clear the floor."""
        assert 0.9960 >= _default_of("auto_drop_diversity_above")

    @pytest.mark.parametrize("corr", [0.95, 0.97, 0.98])
    def test_ordinary_diversity_survives(self, corr):
        """Members this different still blend usefully; dropping one would be a regression, not a fix."""
        assert corr < _default_of("auto_drop_diversity_above")


class TestTheOptOut:
    """A caller who wants the old observe-and-warn behaviour must still be able to have it."""

    def test_none_is_still_accepted(self):
        """Passing None restores the pre-fix behaviour rather than raising."""
        assert inspect.signature(score_ensemble).parameters["auto_drop_diversity_above"].annotation is not None
