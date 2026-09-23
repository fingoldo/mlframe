"""The hybrid-orth scorer default is a deliberate choice, and the two places that declare it agree.

``recommend_default_scorer`` returns the Layer-83 bake-off winner, CMIM, and the shipped default is ``plug_in``. Adopting the recommendation
as the default was tried and reverted: it failed 13 of this repo's own hybrid-orth business-value tests, which all pass on ``plug_in``, because
CMIM routes the univariate basis-selection stage differently and the Fourier / XOR / cross columns those contracts require stop being emitted.
The bake-off measured downstream AUC on seven datasets; the business-value suite measures whether the FE families produce the columns they
exist to produce. They disagree, and until a benchmark reconciles them the shipped default is the one that keeps the contracts.

So this does not pin the default TO the recommendation. It pins the two declarations of the default to each other, which is a real invariant
that can silently break, and keeps the recommendation honest as a value callers can actually set.
"""

from __future__ import annotations

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import HybridOrthScorersConfig
from mlframe.feature_selection.filters.mrmr._mrmr_param_constants import _VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS


def test_the_config_dataclass_default_matches_the_constructor():
    """The dataclass mirror of the knob must not drift from the constructor default."""
    assert HybridOrthScorersConfig().default_scorer == MRMR()._ctor_defaults()["fe_hybrid_orth_default_scorer"]


def test_both_the_default_and_the_recommendation_are_scorers_the_package_accepts():
    """Whatever either names, it has to be a value the validator allows, so a caller can actually set it."""
    assert MRMR()._ctor_defaults()["fe_hybrid_orth_default_scorer"] in _VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS
    assert MRMR.recommend_default_scorer() in _VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS


def test_the_divergence_from_the_recommendation_is_recorded_where_it_is_decided():
    """If the default ever stops matching the recommendation, the reason must be written down next to the recommendation.

    This is the guard against the state the audit found: a method whose whole purpose is to name the right value, a default that does not use
    it, and nothing anywhere saying which is right or why.
    """
    default = MRMR()._ctor_defaults()["fe_hybrid_orth_default_scorer"]
    recommended = MRMR.recommend_default_scorer()
    if default == recommended:
        return  # nothing to explain
    doc = (MRMR.recommend_default_scorer.__doc__ or "").lower()
    assert "bench-attempt-rejected" in doc, "the default diverges from the recommendation with no recorded reason on the recommendation itself"
    assert default in doc, f"the reason does not name the shipped default {default!r}"
