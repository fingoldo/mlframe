"""Coverage for _per_group_discovery.route_spec_column_by_group, previously untested.

The dedicated per-group-discovery test file (test_per_group_discovery.py) exercises
``CompositeTargetDiscovery.fit``'s per-group discovery machinery (``specs_by_group_``
population, fallback, leakage isolation) but never calls the row-routing counterpart this
module provides for PREDICT time: resolving each row's spec of a given name from its OWN
group, falling back to the global spec when the row's group has no per-group discovery
result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery._per_group_discovery import route_spec_column_by_group
from mlframe.training.configs import CompositeTargetDiscoveryConfig

pytestmark = pytest.mark.sklearn_matrix


def _make_panel(n_per_group: int = 900, seed: int = 0) -> pd.DataFrame:
    """3 large groups with different true DGPs (so they discover different top specs, and the
    GLOBAL fit across all of them still finds a usable compromise spec) plus one small group below
    ``per_group_min_rows`` to exercise the fallback-to-global routing path. Mirrors
    test_per_group_discovery.py's ``_make_panel`` -- a 2-group-only panel (no noise group) left
    ``specs_`` empty (the mixed-DGP global fit found nothing passing its gates), so the 3rd group is
    load-bearing for a non-empty global fallback target, not just cosmetic."""
    rng = np.random.default_rng(seed)

    def _group(n, kind, group_id):
        """Build one group's rows for the given DGP kind ('additive' / 'multiplicative' / 'noise')."""
        base_1 = rng.normal(loc=20.0, scale=4.0, size=n)
        base_2 = rng.normal(loc=15.0, scale=3.0, size=n)
        noise_extra = rng.normal(size=n)
        if kind == "additive":
            y = base_1 + 0.5 * noise_extra + rng.normal(scale=0.3, size=n)
        elif kind == "multiplicative":
            y = base_2 * np.exp(0.05 * noise_extra) + rng.normal(scale=0.2, size=n)
        else:  # noise
            y = rng.normal(loc=10.0, scale=5.0, size=n)
        return pd.DataFrame({"base_1": base_1, "base_2": base_2, "x_extra": noise_extra, "group_id": group_id, "y": y})

    df = pd.concat(
        [
            _group(n_per_group, "additive", "A"),
            _group(n_per_group, "multiplicative", "B"),
            _group(n_per_group, "noise", "C"),
            _group(120, "additive", "small"),
        ],
        ignore_index=True,
    )
    return df


def _fit_discovery() -> CompositeTargetDiscovery:
    """Fit a real per-group CompositeTargetDiscovery on the synthetic panel above."""
    df = _make_panel()
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        base_candidates=["base_1", "base_2"],
        transforms=["diff", "additive_residual", "ratio", "logratio", "linear_residual"],
        screening="mi",
        honest_rmse_gate_enabled=False,
        yscale_holdout_gate_enabled=False,
        structural_fragility_gate_enabled=False,
        multi_base_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        honest_holdout_frac=0.2,
        random_state=0,
        per_group_discovery_enabled=True,
        per_group_column="group_id",
        per_group_min_rows=500,
    )
    disc = CompositeTargetDiscovery(cfg)
    train_idx = np.arange(len(df))
    disc.fit(df, "y", ["base_1", "base_2", "x_extra"], train_idx)
    return disc


@pytest.fixture(scope="module")
def fitted_discovery():
    """Module-scoped fixture: one real per-group discovery fit, reused across the tests below."""
    return _fit_discovery()


@pytest.fixture(scope="module")
def panel_df():
    """The same synthetic panel the fitted_discovery fixture was fit on (same seed)."""
    return _make_panel()


def test_route_uses_own_group_spec_when_present(fitted_discovery, panel_df):
    """A row in a group WITH its own per-group spec list gets that group's spec applied, not the global one."""
    disc = fitted_discovery
    assert set(disc.specs_by_group_.keys()) == {"A", "B", "C"}, sorted(disc.specs_by_group_.keys())
    top_a_name = disc.specs_by_group_["A"][0].name

    routed_a = route_spec_column_by_group(disc, panel_df, top_a_name)
    assert routed_a.shape[0] == len(panel_df)

    # Group A rows: finite values wherever spec A's own domain_check passes.
    a_mask = (panel_df["group_id"] == "A").to_numpy()
    assert np.isfinite(routed_a[a_mask]).any(), "group A's own top spec should produce finite T values on its own rows"

    # If B's top spec has a DIFFERENT name than A's top spec and B has no entry under that name,
    # routing B's rows for A's spec name falls through to NaN (no matching spec for B, and A's spec
    # name is not necessarily in the global specs_ either) or to the global spec if it happens to be
    # there -- either way this must not raise and must stay shape-correct.
    assert not np.isnan(routed_a).all()


def test_route_falls_back_to_global_for_group_without_per_group_specs(fitted_discovery, panel_df):
    """The 'small' group (below per_group_min_rows) has no entry in specs_by_group_ -- routing for a
    GLOBAL spec name must fall back to the global spec and produce finite values for that group too."""
    disc = fitted_discovery
    assert "small" not in disc.specs_by_group_
    assert disc.specs_, "global fallback target must be non-empty"
    global_name = disc.specs_[0].name

    routed = route_spec_column_by_group(disc, panel_df, global_name)
    small_mask = (panel_df["group_id"] == "small").to_numpy()
    # The global spec was fit on the whole training set (mixed DGPs), so its domain_check should still
    # pass on at least some 'small'-group rows (same feature ranges as group A, which it was partly fit on).
    assert np.isfinite(routed[small_mask]).any(), "small group must route to the global spec, not stay all-NaN"


def test_route_unknown_spec_name_returns_all_nan(fitted_discovery, panel_df):
    """A spec_name that exists in NEITHER any group's spec list NOR the global specs_ yields all-NaN,
    matching CompositeTargetDiscovery.iter_transform's convention for an unmatched spec."""
    disc = fitted_discovery
    routed = route_spec_column_by_group(disc, panel_df, "definitely_not_a_real_spec_name")
    assert routed.shape[0] == len(panel_df)
    assert np.isnan(routed).all()


def test_route_requires_per_group_column_configured():
    """Calling route_spec_column_by_group on a discovery fit WITHOUT per_group_column set raises ValueError."""
    df = _make_panel(n_per_group=50, seed=1)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        base_candidates=["base_1"],
        transforms=["diff"],
        screening="mi",
        honest_rmse_gate_enabled=False,
        yscale_holdout_gate_enabled=False,
        structural_fragility_gate_enabled=False,
        multi_base_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        random_state=0,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, "y", ["base_1", "base_2", "x_extra"], np.arange(len(df)))
    with pytest.raises(ValueError, match="per_group_column"):
        route_spec_column_by_group(disc, df, "whatever")
