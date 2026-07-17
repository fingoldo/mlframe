"""DCD consolidation: Layer 48 biz_value: HIERARCHICAL POST-HOC CLUSTERING over DCD anchors.

Consolidated verbatim from test_biz_value_mrmr_layer48.py (per audit finding test_code_quality-16).
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hierarchical_two_super_three_sub(n: int = 2000, seed: int = 0):
    """Two super-clusters, each containing three sub-clusters.

    Super-cluster S1 driven by ``meta_1``: three sub-latents each spawn
    a tight pair of features. Each sub-pair tracks its own latent
    closely (within-pair SU very high -> DCD merges into one cluster).
    The three sub-latents share ``meta_1`` plus distinct private noise,
    so cross-sub-pair SU is moderate (-> super-cluster at L48).

    Same shape for S2 driven by ``meta_2`` which is independent of meta_1.

    Plus a few independent noise filler columns.
    """
    rng = np.random.default_rng(int(seed))
    meta_1 = rng.standard_normal(n)
    meta_2 = rng.standard_normal(n)
    # S1 sub-latents: each shares 70% meta_1 + 30% private noise. So
    # cross-sub pair SU is moderate (driven by the shared meta), high
    # enough to clear super_tau=0.4 but well below the within-sub-pair
    # tightness.
    s1_a_latent = meta_1 + 0.7 * rng.standard_normal(n)
    s1_b_latent = meta_1 + 0.7 * rng.standard_normal(n)
    s1_c_latent = meta_1 + 0.7 * rng.standard_normal(n)
    s2_a_latent = meta_2 + 0.7 * rng.standard_normal(n)
    s2_b_latent = meta_2 + 0.7 * rng.standard_normal(n)
    s2_c_latent = meta_2 + 0.7 * rng.standard_normal(n)

    X = pd.DataFrame(
        {
            # Three tight S1 sub-pairs (each pair within-SU very high).
            "s1_a1": s1_a_latent + 0.01 * rng.standard_normal(n),
            "s1_a2": s1_a_latent + 0.01 * rng.standard_normal(n),
            "s1_b1": s1_b_latent + 0.01 * rng.standard_normal(n),
            "s1_b2": s1_b_latent + 0.01 * rng.standard_normal(n),
            "s1_c1": s1_c_latent + 0.01 * rng.standard_normal(n),
            "s1_c2": s1_c_latent + 0.01 * rng.standard_normal(n),
            # Three tight S2 sub-pairs.
            "s2_a1": s2_a_latent + 0.01 * rng.standard_normal(n),
            "s2_a2": s2_a_latent + 0.01 * rng.standard_normal(n),
            "s2_b1": s2_b_latent + 0.01 * rng.standard_normal(n),
            "s2_b2": s2_b_latent + 0.01 * rng.standard_normal(n),
            "s2_c1": s2_c_latent + 0.01 * rng.standard_normal(n),
            "s2_c2": s2_c_latent + 0.01 * rng.standard_normal(n),
            # Filler noise.
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        }
    )
    y = pd.Series((meta_1 + meta_2 + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _flat_independent_data(n: int = 1500, seed: int = 1):
    """All features are mutually independent. The L48 hierarchy
    analyser should return {} -- no super-cluster ties to surface.
    """
    rng = np.random.default_rng(int(seed))
    X = pd.DataFrame({f"f_{i}": rng.standard_normal(n) for i in range(12)})
    y = pd.Series((X["f_0"] + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# 1. Unit tests: build_cluster_hierarchy direct contract
# ---------------------------------------------------------------------------


class TestLayer48_BuilderUnit:
    def test_none_summary_returns_empty(self):
        """``build_cluster_hierarchy(None, X)`` returns ``{}``."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )

        X, _ = _hierarchical_two_super_three_sub(n=400, seed=2)
        h = build_cluster_hierarchy(None, X)
        assert h == {}

    def test_empty_anchors_returns_empty(self):
        """Empty ``cluster_anchors_names`` -> ``{}``."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )

        X, _ = _hierarchical_two_super_three_sub(n=400, seed=3)
        h = build_cluster_hierarchy(
            {"cluster_anchors_names": {}},
            X,
        )
        assert h == {}

    def test_single_anchor_returns_empty(self):
        """One anchor can't form a super-cluster."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )

        X, _ = _hierarchical_two_super_three_sub(n=400, seed=4)
        h = build_cluster_hierarchy(
            {"cluster_anchors_names": {"s1_a1": ["s1_a2"]}},
            X,
        )
        assert h == {}

    def test_missing_anchors_in_X_filtered(self):
        """Anchors not resolvable against ``X.columns`` are dropped
        from the hierarchy (engineered ``_dcd_pc1_*`` aggregates can't
        be scored against the raw matrix). Two resolvable anchors that
        don't clear super_tau -> still ``{}``."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )

        X, _ = _flat_independent_data(n=600, seed=5)
        summary = {
            "cluster_anchors_names": {
                "f_0": ["f_1"],
                "f_2": ["f_3"],
                "_dcd_pc1_phantom": ["irrelevant"],
            }
        }
        h = build_cluster_hierarchy(summary, X, super_tau=0.5)
        # f_0 and f_2 are mutually independent -> no super-cluster forms.
        # The phantom anchor is silently dropped (not resolvable).
        assert h == {}

    def test_pickle_returns_value_pickleable(self):
        """The returned hierarchy is pickleable (dicts of str/list)."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )

        X, _ = _hierarchical_two_super_three_sub(n=400, seed=6)
        h = build_cluster_hierarchy(
            {"cluster_anchors_names": {"s1_a1": [], "s2_a1": []}},
            X,
        )
        # Always pickle the result, regardless of whether merges happened.
        blob = pickle.dumps(h)
        h2 = pickle.loads(blob)
        assert h == h2


# ---------------------------------------------------------------------------
# 2. Two-super x three-sub: hierarchy actually surfaces super-clusters
# ---------------------------------------------------------------------------


class TestLayer48_TwoLevelHierarchy:
    def test_C1_super_clusters_detected_on_synthetic(self):
        """C1: with a low ``super_tau``, the analyser merges anchors
        that share a meta-latent into one super-cluster, even though
        DCD's greedy single-anchor rule kept them in separate clusters."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )

        X, _ = _hierarchical_two_super_three_sub(n=2000, seed=10)
        # Synthetic anchors: pick the first column of each S1/S2 sub-pair.
        # On real fits the DCD anchors would emerge from greedy selection;
        # the unit-test surrogate skips that step and asserts the analyser
        # behaviour directly.
        anchors_names = {
            "s1_a1": ["s1_a2"],
            "s1_b1": ["s1_b2"],
            "s1_c1": ["s1_c2"],
            "s2_a1": ["s2_a2"],
            "s2_b1": ["s2_b2"],
            "s2_c1": ["s2_c2"],
        }
        summary = {"cluster_anchors_names": anchors_names}
        # super_tau low enough to capture the moderate meta-latent ties
        # but high enough to keep S1 and S2 anchors separate. Calibrated
        # against the fixture: cross-sub-pair SU within an S is around
        # 0.10-0.25 (driven by meta_1 / meta_2 sharing); cross-S SU is
        # near zero (meta_1 _|_ meta_2). Pick super_tau=0.05 so the
        # within-S super-clusters form but the cross-S ones don't.
        h = build_cluster_hierarchy(summary, X, super_tau=0.05, max_levels=3)
        assert isinstance(h, dict)
        assert 1 in h, f"Expected at least one level-1 super-cluster on hierarchical data; got hierarchy={h}"
        level1 = h[1]
        # At least one super-cluster must contain >= 2 S1 anchors (or
        # >= 2 S2 anchors); the meta-latent tie is what L48 surfaces.
        s1_anchors = {"s1_a1", "s1_b1", "s1_c1"}
        s2_anchors = {"s2_a1", "s2_b1", "s2_c1"}
        found_s1_super = False
        found_s2_super = False
        for super_anchor, sub_anchors in level1.items():
            group = {super_anchor} | set(sub_anchors)
            if len(group & s1_anchors) >= 2:
                found_s1_super = True
            if len(group & s2_anchors) >= 2:
                found_s2_super = True
        assert found_s1_super or found_s2_super, f"Expected a super-cluster grouping >= 2 S1 anchors OR >= 2 S2 anchors; got level1={level1}"

    def test_C2_flat_data_no_super_structure(self):
        """C2: flat / independent data -> hierarchy is empty."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )

        X, _ = _flat_independent_data(n=1500, seed=11)
        # Even with several phantom anchors, no super tie exists in
        # independent data.
        summary = {
            "cluster_anchors_names": {
                "f_0": [],
                "f_1": [],
                "f_2": [],
                "f_3": [],
                "f_4": [],
            }
        }
        h = build_cluster_hierarchy(summary, X, super_tau=0.5, max_levels=3)
        assert h == {}, f"Independent data must yield empty hierarchy; got {h}"

    def test_C2b_high_super_tau_suppresses_merges(self):
        """C2: even on hierarchical data, raising ``super_tau`` close to
        1.0 suppresses every merge (no pair clears the bar)."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )

        X, _ = _hierarchical_two_super_three_sub(n=1500, seed=12)
        summary = {
            "cluster_anchors_names": {
                "s1_a1": ["s1_a2"],
                "s1_b1": ["s1_b2"],
                "s2_a1": ["s2_a2"],
                "s2_b1": ["s2_b2"],
            }
        }
        h = build_cluster_hierarchy(summary, X, super_tau=0.99, max_levels=3)
        assert h == {}


# ---------------------------------------------------------------------------
# 3. End-to-end MRMR.fit -> cluster_hierarchy_ accessor
# ---------------------------------------------------------------------------


class TestLayer48_FitIntegration:
    def test_cluster_hierarchy_attr_present_when_dcd_disabled(self):
        """DCD disabled -> ``cluster_hierarchy_`` is ``None`` (mirrors
        the ``cluster_members_`` semantics)."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _hierarchical_two_super_three_sub(n=500, seed=20)
        m = MRMR(dcd_enable=False, verbose=0, random_seed=0).fit(X, y)
        assert hasattr(m, "cluster_hierarchy_")
        assert m.cluster_hierarchy_ is None

    def test_cluster_hierarchy_attr_present_when_dcd_enabled(self):
        """DCD enabled -> ``cluster_hierarchy_`` is a dict (possibly empty
        if DCD discovered fewer than 2 anchors / no super-ties found)."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _hierarchical_two_super_three_sub(n=1500, seed=21)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            full_npermutations=5,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "cluster_hierarchy_")
        assert isinstance(m.cluster_hierarchy_, dict)

    def test_cluster_hierarchy_pickle_round_trip(self):
        """C3: ``cluster_hierarchy_`` survives ``pickle.loads(dumps(m))``."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _hierarchical_two_super_three_sub(n=1200, seed=22)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            full_npermutations=3,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        assert m2.cluster_hierarchy_ == m.cluster_hierarchy_

    def test_cluster_hierarchy_deterministic(self):
        """C6: two identical fits yield identical hierarchies."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _hierarchical_two_super_three_sub(n=1200, seed=23)
        m1 = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            full_npermutations=3,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        m2 = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            full_npermutations=3,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m1.cluster_hierarchy_ == m2.cluster_hierarchy_


# ---------------------------------------------------------------------------
# 4. Regression on Layers 41 + 47
# ---------------------------------------------------------------------------


class TestLayer48_RegressionL41L47:
    def test_l41_cluster_members_still_populated(self):
        """L41 contract preserved: ``cluster_members_`` reports the
        per-anchor membership map; the new L48 hierarchy_ does not
        displace it."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _hierarchical_two_super_three_sub(n=1500, seed=30)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            full_npermutations=3,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "cluster_members_")
        assert hasattr(m, "cluster_hierarchy_")
        # Both attrs coexist.
        assert m.cluster_members_ is not None
        assert isinstance(m.cluster_hierarchy_, dict)

    def test_l47_tau_calibration_compose(self):
        """L47 ``dcd_tau_cluster='auto'`` composes with L48 analyser."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _hierarchical_two_super_three_sub(n=1500, seed=31)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=3,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert isinstance(m.cluster_hierarchy_, dict)
        # L47 tau_calibration key intact under L48 wiring.
        assert "tau_calibration" in m.dcd_
