"""Cluster 25 regression: the CatBoost tuning search space can draw the maximum depth 16 (audit 2026-11-20, finding 25-F61).

scipy's randint excludes its high bound, so randint(1, 16) never produced 16. The depth distribution is read from the module's own search
space and sampled, so a bound change in the module is what the test sees.
"""

from mlframe.models import tuning_catboost as tc


def _depth_distribution():
    """The depth distribution as the module defines it."""
    return tc._catboost_default_params(False, False)["depth"]


def test_depth_space_reaches_16_and_stays_in_range():
    """Sampling the module's depth space gives 16 at least once and nothing outside 1..16 (25-F61)."""
    vals = set(_depth_distribution().rvs(size=2000, random_state=0).tolist())
    assert 16 in vals
    assert min(vals) >= 1 and max(vals) <= 16
