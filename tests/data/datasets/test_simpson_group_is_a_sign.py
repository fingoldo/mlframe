"""The Simpson bed's group is a fair sign, so its interaction reverses sign between subgroups (audit 2026-11-20, finding 04-F12)."""
import numpy as np

from mlframe.data.datasets._columns import draw_family
from mlframe.data.datasets.scenarios._structure import simpson_reversal_spec


def test_rademacher_family_draws_plus_or_minus_one():
    """Values are exactly -1 or +1 with a mean near zero."""
    rng = np.random.default_rng(0)
    values = draw_family("rademacher", {}, 10_000, rng)
    assert set(np.unique(values)) == {-1.0, 1.0}
    assert abs(values.mean()) < 0.05


def test_simpson_bed_declares_a_signed_group():
    """The group column is the sign family, not a 0/1 indicator whose interaction would not reverse."""
    spec = simpson_reversal_spec()
    group = next(f for f in spec.features if f.name == "group")
    assert group.family == "rademacher"
