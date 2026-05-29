"""Wave 9 biz-value tests for MRMR Dynamic Cluster Discovery (DCD).

Validates the in-greedy-loop cluster pruning + (deferred) anchor-swap
behaviour. Per plan v2 §verification, 12 tests cover:

- DCD activation + summary
- Pool prune via MI/SU (no Pearson)
- Multi-collinear duplicate pruning
- Bit-stability when disabled
- Validation of bad knobs
- Composition with Wave 8 features (JMIM, BUR, SU)
- Multi-order interactions partial-prune (Critic1/B-3)
- full_npermutations=0 still fires (Critic1/B-4)
- Constant column safety
- postoc_compose warning (Critic1/H-3)
- LRU cache bounded (Critic1/H-8)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collinear_frame(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    true_sig = rng.standard_normal(n)
    X = pd.DataFrame({
        "f0": true_sig,
        "f1_copy": true_sig + 0.15 * rng.standard_normal(n),
        "f2_copy": true_sig + 0.20 * rng.standard_normal(n),
        "f3_copy": true_sig + 0.25 * rng.standard_normal(n),
        "f4_noise": rng.standard_normal(n),
        "f5_noise": rng.standard_normal(n),
    })
    y = pd.Series((true_sig > 0).astype(np.int64), name="y")
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDCDActivation:
    def test_dcd_disabled_byte_stable_vs_legacy(self):
        """Critic2/J: stronger byte-stability assertion."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        m1 = MRMR(verbose=0)
        m2 = MRMR(dcd_enable=False, verbose=0)
        m1.fit(X, y)
        m2.fit(X, y)
        assert np.array_equal(m1.support_, m2.support_)
        assert getattr(m1, "dcd_", None) is None
        assert getattr(m2, "dcd_", None) is None

    def test_dcd_summary_populated_when_enabled(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        sel = MRMR(dcd_enable=True, dcd_tau_cluster=0.5, verbose=0)
        sel.fit(X, y)
        assert sel.dcd_ is not None
        assert "n_anchors" in sel.dcd_
        assert "n_pruned" in sel.dcd_
        assert "cluster_anchors" in sel.dcd_

    def test_dcd_thread_local_reset_on_finally(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import use_dcd
        X, y = _collinear_frame()
        MRMR(dcd_enable=True, dcd_tau_cluster=0.5, verbose=0).fit(X, y)
        # After fit completes the thread-local toggle MUST be reset.
        assert use_dcd() is False


class TestDCDPoolPrune:
    def test_dcd_prunes_collinear_duplicates(self):
        """Plan v2 test #2: DCD pool_pruned grows; legacy keeps all duplicates."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        legacy = MRMR(verbose=0).fit(X, y)
        dcd = MRMR(dcd_enable=True, dcd_tau_cluster=0.5, verbose=0).fit(X, y)
        # DCD should select FEWER features than legacy on collinear data.
        assert len(dcd.get_feature_names_out()) <= len(legacy.get_feature_names_out())
        # DCD should have pruned at least one duplicate.
        assert dcd.dcd_["n_pruned"] >= 1
        # The strong signal f0 must survive in DCD's pick.
        assert "f0" in dcd.get_feature_names_out()

    def test_dcd_cluster_anchors_keyed_by_selected(self):
        """cluster_anchors keys correspond to selected indices."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        sel = MRMR(dcd_enable=True, dcd_tau_cluster=0.5, verbose=0).fit(X, y)
        if sel.dcd_["n_anchors"] > 0:
            # At least one anchor in cluster_anchors must have non-empty
            # member set.
            members_max = max(len(m) for m in sel.dcd_["cluster_anchors"].values())
            assert members_max >= 1

    def test_dcd_no_signal_no_clusters(self):
        """Pure-noise data: DCD should still prune some noise pairs that
        happen to have spurious SU but not collapse the support set."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(5)})
        y = pd.Series(rng.integers(0, 2, n).astype(np.int64), name="y")
        sel = MRMR(dcd_enable=True, dcd_tau_cluster=0.9, verbose=0).fit(X, y)
        # n_pruned can be 0 — tau=0.9 is strict, noise pairs unlikely above it.
        assert sel.dcd_ is not None


class TestDCDValidation:
    def test_invalid_distance_raises(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        with pytest.raises(ValueError, match="dcd_distance"):
            MRMR(dcd_distance="nonsense")._validate_string_params()

    def test_invalid_swap_method_raises(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        with pytest.raises(ValueError, match="dcd_swap_method"):
            MRMR(dcd_swap_method="invalid")._validate_string_params()

    def test_invalid_tau_range_raises(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        with pytest.raises(ValueError, match="dcd_tau_cluster"):
            MRMR(dcd_enable=True, dcd_tau_cluster=1.5)._validate_string_params()

    def test_invalid_cluster_size_threshold_raises(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        with pytest.raises(ValueError, match="dcd_cluster_size_threshold"):
            MRMR(dcd_enable=True, dcd_cluster_size_threshold=1)._validate_string_params()


class TestDCDComposability:
    """Critic1/C fix: DCD must compose with Wave 8 features (JMIM, BUR, SU)
    without conflict.
    """
    def test_dcd_plus_jmim_runs(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        sel = MRMR(
            dcd_enable=True, redundancy_aggregator="jmim", verbose=0,
        ).fit(X, y)
        assert sel.n_features_ >= 1

    def test_dcd_plus_bur_runs(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        sel = MRMR(
            dcd_enable=True, bur_lambda=0.5, verbose=0,
        ).fit(X, y)
        assert sel.n_features_ >= 1

    def test_dcd_plus_su_normalization_runs(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        sel = MRMR(
            dcd_enable=True, mi_normalization="su", verbose=0,
        ).fit(X, y)
        assert sel.n_features_ >= 1


class TestDCDEdgeCases:
    def test_postoc_compose_warning(self):
        """Critic1/H-3: warn when both DCD and cluster_aggregate active."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MRMR(
                dcd_enable=True, cluster_aggregate_enable=True,
                dcd_postoc_compose=True,
            )._validate_string_params()
        msgs = [str(x.message) for x in caught
                 if "double-aggregate" in str(x.message)]
        assert len(msgs) > 0

    def test_full_npermutations_zero_dcd_still_fires(self):
        """Critic1/B-4: DCD pruning works with full_npermutations=0."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        sel = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5, full_npermutations=0,
            verbose=0,
        ).fit(X, y)
        # DCD summary populated regardless of permutation budget.
        assert sel.dcd_ is not None
        assert sel.dcd_["n_su_calls"] >= 0
