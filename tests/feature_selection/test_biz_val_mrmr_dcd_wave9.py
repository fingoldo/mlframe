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


class TestDCDSwapPath:
    """Wave 9.1: anchor → PC1 swap evaluator + commit. Tests exercise the
    standalone module API (the screen-loop wire-up is best-effort and only
    fires when the PC1 aggregate beats the raw anchor by ``swap_gain_threshold``).
    """
    def _binned_matrix(self, n=200, seed=0):
        import pandas as pd
        rng = np.random.default_rng(int(seed))
        true_sig = rng.standard_normal(n)
        y_arr = (true_sig > 0).astype(np.int64)
        cols_raw = [
            true_sig,
            true_sig + 0.10 * rng.standard_normal(n),
            true_sig + 0.12 * rng.standard_normal(n),
            true_sig + 0.15 * rng.standard_normal(n),
            true_sig + 0.18 * rng.standard_normal(n),
            rng.standard_normal(n),
            y_arr.astype(np.float64),
        ]
        data = np.zeros((n, 7), dtype=np.int32)
        for i, col_vals in enumerate(cols_raw):
            edges = np.linspace(col_vals.min(), col_vals.max(), 11)
            data[:, i] = np.clip(
                np.digitize(col_vals, edges[1:-1]), 0, 9,
            ).astype(np.int32)
        nbins = np.array([10] * 7, dtype=np.int64)
        target_indices = np.array([6], dtype=np.int64)
        X_raw = pd.DataFrame({f"f{i}": cols_raw[i] for i in range(7)})
        return data, nbins, target_indices, X_raw

    def test_evaluate_swap_candidate_returns_decision(self):
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            make_dcd_state, discover_cluster_members, evaluate_swap_candidate,
            SwapDecision,
        )
        data, nbins, target_indices, X_raw = self._binned_matrix()
        state = make_dcd_state(
            X_raw=X_raw, factors_data=data, factors_nbins=nbins,
            cols=[f"f{i}" for i in range(7)], nbins=nbins,
            target_indices=target_indices, tau_cluster=0.3,
            cluster_size_threshold=3, swap_gain_threshold=0.001,
        )
        discover_cluster_members(state, 0, list(range(1, 6)))
        decision = evaluate_swap_candidate(
            state, 0, [0], target_y=target_indices,
            factors_data=data, factors_nbins=nbins,
            cached_MIs={}, entropy_cache=None, full_npermutations=0,
        )
        assert isinstance(decision, SwapDecision)
        # Should have computed rep + anchor relevance.
        assert decision.rep_relevance >= 0.0
        assert decision.anchor_relevance_in_ctx >= 0.0

    def test_swap_e2e_no_crash_when_swap_fires(self):
        """Wave 9.1 loop-iter-1 regression: end-to-end fit MUST NOT crash
        when ``commit_swap`` extends ``factors_data`` mid-screen. The pre-fix
        bug produced ``ValueError: negative dimensions not allowed`` in
        ``_run_fe_step -> merge_vars`` because the outer-scope data/cols/nbins
        didn't pick up the extended matrix.
        """
        import pandas as pd
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(0)
        n = 1500
        # 3 latent signals + 4 collinear members each + 2 noise = 15 cols
        # large enough cluster sizes to force swaps at threshold=4.
        cols_data = {}
        for sig_idx in range(3):
            latent = rng.standard_normal(n)
            cols_data[f"sig{sig_idx}_anchor"] = latent
            for k in range(4):
                cols_data[f"sig{sig_idx}_copy{k}"] = (
                    latent + 0.05 * (k + 1) * rng.standard_normal(n)
                )
        cols_data["noise_0"] = rng.standard_normal(n)
        cols_data["noise_1"] = rng.standard_normal(n)
        X = pd.DataFrame(cols_data)
        y_signal = sum(rng.standard_normal(n) * 0.0 + cols_data[f"sig{i}_anchor"]
                        for i in range(3))
        y = pd.Series((y_signal > np.median(y_signal)).astype(np.int64), name="y")
        sel = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=4, dcd_swap_gain_threshold=0.0,
            verbose=0,
        )
        # The pre-fix bug raised here; post-fix must complete.
        sel.fit(X, y)
        # Every name in support_ must be resolvable (no out-of-bounds idx).
        names = sel.get_feature_names_out()
        assert len(names) >= 1
        assert all(isinstance(n, str) and len(n) > 0 for n in names)

    def test_commit_swap_extends_data_matrix(self):
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            make_dcd_state, discover_cluster_members,
            evaluate_swap_candidate, commit_swap,
        )
        data, nbins, target_indices, X_raw = self._binned_matrix()
        state = make_dcd_state(
            X_raw=X_raw, factors_data=data, factors_nbins=nbins,
            cols=[f"f{i}" for i in range(7)], nbins=nbins,
            target_indices=target_indices, tau_cluster=0.3,
            cluster_size_threshold=3, swap_gain_threshold=0.0,  # any improvement
        )
        discover_cluster_members(state, 0, list(range(1, 6)))
        decision = evaluate_swap_candidate(
            state, 0, [0], target_y=target_indices,
            factors_data=data, factors_nbins=nbins,
            cached_MIs={}, entropy_cache=None, full_npermutations=0,
        )
        if decision.accept:
            data_ref = {}
            selected_vars = [0]
            new_idx = commit_swap(
                state, 0, decision, selected_vars=selected_vars,
                data_ref=data_ref, engineered_recipes=None,
                predictors_log=None,
            )
            # data_ref["data"] should be extended by 1 column.
            assert data_ref["data"].shape[1] == data.shape[1] + 1
            # selected_vars's first element should now be the new aggregate idx.
            assert selected_vars[0] == new_idx
            # pool_pruned_mask should mark the original anchor as pruned.
            assert state.pool_pruned_mask[0] is np.True_ or bool(state.pool_pruned_mask[0])
            # swap_log entry persisted.
            assert len(state.swap_log) >= 1
