"""Suite-level end-to-end tests for the DEFAULT-ON cluster-medoid reduction
(audit integration-defaults-3). Unlike TestCompositeXRFECV (which constructs a
BARE RFECV), these drive the selectors through the REGISTRY -- the actual
default-ON path that wraps RFECV / BorutaShap in GroupAwareMRMR -- and run the
full selector -> composite-discovery integration. Covers the gap: BorutaShap had
no suite-level test, and neither selector had its REGISTRY-wrapped form exercised
end-to-end. The fixtures carry a correlated cluster so the medoid reduction
genuinely engages (not just the no-reduction guard bypass).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.sklearn_matrix

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _corr_cluster_frame(n=1500, seed=0, binary=False):
    """A standalone strong ``base`` + a 5-member correlated cluster (reflections
    of a latent that also drives y) + noise. The cluster collapses to one medoid
    under the default reduction; ``base`` stays its own unit."""
    rng = np.random.default_rng(seed)
    base = rng.normal(10, 3, n)
    latent = rng.normal(size=n)
    cols = {"base": base}
    for i in range(5):
        cols[f"c{i}"] = latent + 0.1 * rng.normal(size=n)  # tight cluster
    for i in range(4):
        cols[f"noise{i}"] = rng.normal(size=n)
    df = pd.DataFrame(cols)
    signal = 0.95 * (base - base.mean()) + 1.6 * latent + rng.normal(scale=0.1, size=n)
    df["target"] = (signal > np.median(signal)).astype(int) if binary else (0.95 * base + 1.6 * latent + rng.normal(scale=0.1, size=n))
    return df, "target", [c for c in df.columns if c != "target"]


class TestRegistryClusterReducedSelectorsEndToEnd:
    """The registry returns cluster-reduced (GroupAwareMRMR-wrapped) selectors by
    default; verify each fits, exposes a consistent selection surface, actually
    reduces on the correlated cluster, and feeds composite discovery cleanly."""

    def _assert_wrapped_and_consistent(self, sel, X):
        """Assert wrapped and consistent."""
        from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR

        assert isinstance(sel, GroupAwareMRMR), "registry must default to the cluster-reduced wrap"
        assert sel.reduced_ is True and sel.reduction_ > 0.0, "medoid reduction must engage on the correlated cluster"
        names = list(sel.get_feature_names_out())
        assert len(names) == len(sel.support_) >= 1
        assert set(names).issubset(set(X.columns))
        # expand=True: if any cluster member is selected, ALL members are kept.
        cluster = {f"c{i}" for i in range(5)}
        kept_cluster = cluster & set(names)
        assert kept_cluster in (set(), cluster), f"cluster expansion must be all-or-nothing; got {sorted(kept_cluster)}"
        return names

    def test_registry_rfecv_wrap_then_composite_regression(self):
        """Registry rfecv wrap then composite regression."""
        from sklearn.ensemble import RandomForestRegressor
        from mlframe.feature_selection import registry

        df, target_col, feature_cols = _corr_cluster_frame()
        train_idx = np.arange(int(0.8 * len(df)))
        X = df[feature_cols].iloc[train_idx]
        y = df[target_col].iloc[train_idx]

        sel = registry.get("RFECV").instantiate(
            estimator=RandomForestRegressor(n_estimators=20, random_state=42),
            cv=3,
            max_refits=2,
            verbose=0,
            optimizer_plotting="No",
            random_state=42,
        )
        sel.fit(X, y)
        names = self._assert_wrapped_and_consistent(sel, X)
        assert "base" in names, "RFECV-wrap should keep the strong standalone base feature"

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            screening="mi",
            mi_sample_n=800,
            eps_mi_gain=-1.0,
            top_k_after_mi=4,
            require_beats_raw_baseline=False,
            base_candidates=["base"],
            transforms=["diff", "linear_residual"],
            random_state=0,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col=target_col, feature_cols=names, train_idx=train_idx)
        assert len(disc.specs_) >= 1, f"composite discovery yielded no specs from {names}"
        assert disc.specs_[0].base_column in names

    def test_registry_borutashap_wrap_then_composite_binary(self):
        """Registry borutashap wrap then composite binary."""
        from mlframe.feature_selection import registry

        df, target_col, feature_cols = _corr_cluster_frame(binary=True)
        train_idx = np.arange(int(0.8 * len(df)))
        X = df[feature_cols].iloc[train_idx]
        y = df[target_col].iloc[train_idx]

        sel = registry.get("BorutaShap").instantiate(
            importance_measure="gini",
            classification=True,
            n_trials=15,
            verbose=False,
            random_state=0,
        )
        sel.fit(X, y)
        names = self._assert_wrapped_and_consistent(sel, X)
        # accepted (suite BorutaShap report contract) must equal the expanded set.
        assert set(sel.accepted) == set(names)

        # Composite discovery is regression-only but must not crash on the
        # BorutaShap-selected subset of a binary target (treats {0,1} as continuous).
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            screening="mi",
            mi_sample_n=800,
            eps_mi_gain=-100.0,
            top_k_after_mi=4,
            require_beats_raw_baseline=False,
            base_candidates=["base"],
            transforms=["diff"],
            random_state=0,
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col=target_col, feature_cols=names, train_idx=train_idx)
        assert isinstance(disc.specs_, list)
