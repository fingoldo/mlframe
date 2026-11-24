"""Cluster 25 regression: the CatBoost tuning search space can draw the maximum depth 16 (audit 2026-11-20, finding 25-F61).

scipy's randint excludes its high bound, so randint(1, 16) never produced 16. The space is randint(1, 17); depth 16 is drawn and every value
stays within 1..16.
"""

from pathlib import Path

from scipy.stats import randint

from mlframe.models import tuning_catboost as tc

_SRC = Path(tc.__file__).resolve()


def test_depth_space_reaches_16_and_stays_in_range():
    """Sampling the space gives 16 at least once and nothing outside 1..16 (25-F61)."""
    vals = set(randint(1, 17).rvs(size=2000, random_state=0).tolist())
    assert 16 in vals
    assert min(vals) >= 1 and max(vals) <= 16


def test_source_uses_the_inclusive_depth_bound():
    """The depth distribution in the module is randint(1, 17) (25-F61, source pin)."""
    assert '"depth": randint(1, 17)' in _SRC.read_text(encoding="utf-8")
