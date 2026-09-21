"""Config fields that currently do nothing must say so when a user sets them.

``force_inject_diff_on_top_ablation_pct`` (implementation pending) and ``structural_fragility_max_amplification_ratio``
(superseded by the scale-invariant between/total ratio) were accepted, documented as effective, and silently ignored.
Setting either away from its default now warns; the defaults stay silent.
"""

from __future__ import annotations

import warnings

import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig


def test_the_defaults_are_silent():
    """A config that leaves both fields alone must not warn about them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CompositeTargetDiscoveryConfig(enabled=True)
    assert not [w for w in caught if "has no effect" in str(w.message)]


@pytest.mark.parametrize(
    ("field", "value"),
    [("force_inject_diff_on_top_ablation_pct", 50.0), ("structural_fragility_max_amplification_ratio", 0.9)],
)
def test_setting_an_inert_field_warns(field, value):
    """A non-default value on a field nothing reads must produce a warning naming the field."""
    with pytest.warns(UserWarning, match=field):
        CompositeTargetDiscoveryConfig(enabled=True, **{field: value})
