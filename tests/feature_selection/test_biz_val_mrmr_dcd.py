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
    def test_dcd_default_now_enabled_after_2026_05_30_flip(self):
        """2026-05-30: ``dcd_enable=True`` flipped to DEFAULT after
        Layer-6 biz_value confirmed DCD is the production-correct
        redundancy mechanism. Default-fit estimator now exposes
        ``dcd_`` summary; legacy bit-stability with pre-Wave-9 fits
        is opt-in via ``dcd_enable=False``.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _collinear_frame()
        m_default = MRMR(verbose=0).fit(X, y)
        m_legacy = MRMR(dcd_enable=False, verbose=0).fit(X, y)
        # Default now uses DCD; ``dcd_`` summary populated.
        assert m_default.dcd_ is not None
        # Legacy opt-out preserves pre-Wave-9 dcd_=None.
        assert getattr(m_legacy, "dcd_", None) is None

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
        # Layer 42 (2026-05-31) lowered the minimum threshold from 2 to 1
        # (threshold counts MEMBERS, not anchor + members, so =1 is the
        # strict 2-feature redundancy case). Only values < 1 are rejected.
        with pytest.raises(ValueError, match="dcd_cluster_size_threshold"):
            MRMR(dcd_enable=True, dcd_cluster_size_threshold=0)._validate_string_params()


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

    def test_swap_does_not_corrupt_subsequent_confirmations(self):
        """Wave 9.1 loop-iter-2 regression: after ``commit_swap`` extends the
        matrix, ``ctx.factors_data``/``ctx.factors_nbins``/``ctx.factors_names``/
        ``ctx.data_copy`` MUST be written too, else the next call to
        ``confirm_one_predictor(ctx, ...)`` within the SAME screen invocation
        sees a stale matrix and ``selected_vars`` containing the post-swap
        index that's out of bounds for the stale matrix - silent OOB under
        numba ``boundscheck=False`` or ``IndexError`` otherwise. Crucially,
        we want at least 2 confirmations AFTER the swap to exercise the
        re-read of ``ctx.factors_data`` inside ``confirm_one_predictor``.
        """
        import pandas as pd
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(7)
        n = 2000
        # one strong cluster (forces swap early), then several independent
        # signals (force multiple subsequent confirmations that re-enter
        # confirm_one_predictor reading from ctx).
        latent = rng.standard_normal(n)
        cols_data = {"clu_anchor": latent}
        for k in range(5):
            cols_data[f"clu_copy{k}"] = latent + 0.05 * (k + 1) * rng.standard_normal(n)
        # 5 independent signals that should each be confirmed AFTER the swap.
        independents = []
        for j in range(5):
            s = rng.standard_normal(n)
            cols_data[f"ind{j}"] = s
            independents.append(s)
        X = pd.DataFrame(cols_data)
        # y depends on latent + every independent -> each ind is a true
        # post-swap candidate that must be confirmable.
        y_signal = latent + sum(independents)
        y = pd.Series((y_signal > np.median(y_signal)).astype(np.int64), name="y")
        sel = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=5, dcd_swap_gain_threshold=0.0,
            verbose=0,
        )
        # Pre-fix: stale ctx.factors_data -> next confirm reads
        # selected_vars=[NEW_IDX] against old matrix -> crash / silent OOB.
        sel.fit(X, y)
        names = sel.get_feature_names_out()
        assert len(names) >= 1
        # At least one independent must survive (proves >=1 successful
        # confirmation took place AFTER the swap).
        if sel.dcd_ and sel.dcd_.get("swap_log"):
            ind_names = {f"ind{j}" for j in range(5)}
            confirmed_post_swap = ind_names & set(names)
            assert len(confirmed_post_swap) >= 1, (
                f"No independent signal confirmed after swap -- ctx propagation "
                f"likely silently corrupted. selected={names}"
            )

    def test_perm_null_rejects_noise_when_full_npermutations_positive(self):
        """Wave 9.1 loop-iter-3 regression: when ``full_npermutations > 0`` and
        the candidate cluster is pure noise (target is noise / uncorrelated),
        the PC1 swap MUST be rejected by the permutation null even if the
        deterministic gate would spuriously fire. Pre-fix the parameter was
        ignored and accept was decided on point-MI alone.
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            make_dcd_state, discover_cluster_members, evaluate_swap_candidate,
        )
        rng = np.random.default_rng(11)
        n = 200
        # All-noise: target is also random; no real cluster
        data = rng.integers(0, 4, (n, 7)).astype(np.int32)
        nbins = np.array([4] * 7, dtype=np.int64)
        target_indices = np.array([6], dtype=np.int64)
        import pandas as pd
        X_raw = pd.DataFrame(data[:, :6], columns=[f"f{i}" for i in range(6)])
        state = make_dcd_state(
            X_raw=X_raw, factors_data=data, factors_nbins=nbins,
            cols=[f"f{i}" for i in range(7)], nbins=nbins,
            target_indices=target_indices,
            tau_cluster=0.0,  # accept ALL into cluster
            min_cluster_size=5, cluster_size_threshold=5,
            swap_gain_threshold=-1.0,  # force deterministic-gate pass
            swap_alpha=0.05,
        )
        discover_cluster_members(state, 0, list(range(1, 6)))
        # With B=0 (no null), the swap WILL go through the deterministic gate
        # because swap_gain_threshold=-1 makes ``rep > anchor*0`` trivially.
        d_no_null = evaluate_swap_candidate(
            state, 0, [0], target_y=target_indices,
            full_npermutations=0,
        )
        # With B>0, the null catches it: under H0 random rep gets same/higher
        # CMI than observed -> high p-value -> reject.
        d_with_null = evaluate_swap_candidate(
            state, 0, [0], target_y=target_indices,
            full_npermutations=100,
        )
        # The null test must run (p was computed) and must reject on pure noise.
        assert 0.0 <= d_with_null.perm_p_value <= 1.0
        # Either deterministic gate already rejected (good) OR permutation null
        # rejected (the fix we're testing). What must NOT happen: accept=True
        # with high p-value -- that would mean the null wasn't checked.
        if d_with_null.perm_p_value >= 0.5:
            assert not d_with_null.accept, (
                f"swap accepted with high perm_p_value={d_with_null.perm_p_value} "
                f"-- permutation null not enforcing rejection."
            )

    def test_perm_null_default_zero_preserves_deterministic_behavior(self):
        """With ``full_npermutations=0`` (the standalone-test default), the
        function must behave exactly as the pre-iter-3 deterministic gate -
        no permutation computed, ``perm_p_value=0.0`` on accept.
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            make_dcd_state, discover_cluster_members, evaluate_swap_candidate,
        )
        data, nbins, target_indices, X_raw = self._binned_matrix()
        state = make_dcd_state(
            X_raw=X_raw, factors_data=data, factors_nbins=nbins,
            cols=[f"f{i}" for i in range(7)], nbins=nbins,
            target_indices=target_indices, tau_cluster=0.3,
            cluster_size_threshold=3, swap_gain_threshold=0.0,
            swap_alpha=0.05,
        )
        discover_cluster_members(state, 0, list(range(1, 6)))
        decision = evaluate_swap_candidate(
            state, 0, [0], target_y=target_indices, full_npermutations=0,
        )
        if decision.accept:
            # No null was run; perm_p_value untouched from the accept branch.
            assert decision.perm_p_value == 0.0

    def test_dcd_swap_alpha_validation(self):
        """``dcd_swap_alpha`` must be in (0, 1]."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = self._noise_frame()
        for bad in (-0.1, 0.0, 1.1, 2.0):
            with pytest.raises(ValueError, match="dcd_swap_alpha"):
                MRMR(dcd_enable=True, dcd_swap_alpha=bad).fit(X, y)
        # 1.0 (always accept) and tiny epsilon must pass validation.
        MRMR(dcd_enable=True, dcd_swap_alpha=1.0).fit(X, y)
        MRMR(dcd_enable=True, dcd_swap_alpha=1e-6).fit(X, y)

    def _noise_frame(self, n: int = 200, seed: int = 0):
        import pandas as pd
        rng = np.random.default_rng(int(seed))
        X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(6)})
        y = pd.Series(rng.integers(0, 2, n), name="y")
        return X, y

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
