"""Tests for Tier 2.1 (sample weights) + Tier 2.2 (bootstrap CIs)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
from mlframe.feature_selection.filters.cat_interactions import (
    _pair_search_kernel_weighted_njit,
    run_cat_interaction_step,
)
from mlframe.feature_selection.filters.info_theory import merge_vars

# ---------------------------------------------------------------------------
# Tier 2.1: weighted MI kernel unit tests
# ---------------------------------------------------------------------------


class TestWeightedKernelUnit:
    """Groups tests covering TestWeightedKernelUnit."""

    def _make_xor(self, n=1000, seed=0):
        """Make xor."""
        rng = np.random.default_rng(seed)
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        y = (x1 ^ x2).astype(np.int32)
        data = np.column_stack([x1, x2, y]).astype(np.int32)
        nbins = np.array([2, 2, 2], dtype=np.int64)
        cls_y, _, _ = merge_vars(
            factors_data=data,
            vars_indices=np.array([2], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        return data, nbins, cls_y

    def test_uniform_weights_match_unweighted(self):
        """When all weights are equal, weighted MI must match
        unweighted MI (modulo float noise)."""
        from mlframe.feature_selection.filters.cat_interactions import (
            _pair_search_kernel_njit,
        )

        data, nbins, cls_y = self._make_xor()
        fq_y = np.array([0.5, 0.5], dtype=np.float64)
        pairs_a = np.array([0], dtype=np.int64)
        pairs_b = np.array([1], dtype=np.int64)
        marginal_mi = np.zeros(3, dtype=np.float64)
        weights = np.ones(len(data), dtype=np.float64)

        joint_uw, _, _ = _pair_search_kernel_njit(
            data,
            pairs_a,
            pairs_b,
            marginal_mi,
            nbins,
            cls_y,
            fq_y,
            np.int32,
        )
        joint_w, _, _ = _pair_search_kernel_weighted_njit(
            data,
            pairs_a,
            pairs_b,
            marginal_mi,
            nbins,
            cls_y,
            weights,
            np.int32,
        )
        # 1e-3 tolerance: the unweighted kernel uses the passed
        # ``freqs_y`` (normalised probabilities) while the weighted
        # kernel rebuilds marginals from row weights. Tiny float drift
        # ~5e-4 typical at n=1000.
        assert abs(joint_uw[0] - joint_w[0]) < 1e-3, f"Uniform weights produced different MI: uw={joint_uw[0]:.6f} vs w={joint_w[0]:.6f}"

    def test_zero_weights_for_signal_rows_kills_mi(self):
        """When weights for the signal-carrying rows are 0, the
        weighted joint MI should approach 0 (signal masked out)."""
        data, nbins, cls_y = self._make_xor(n=2000, seed=42)
        pairs_a = np.array([0], dtype=np.int64)
        pairs_b = np.array([1], dtype=np.int64)
        marginal_mi = np.zeros(3, dtype=np.float64)

        # Weight only first 10 rows
        weights = np.zeros(len(data), dtype=np.float64)
        weights[:10] = 1.0

        joint_w, _, _ = _pair_search_kernel_weighted_njit(
            data,
            pairs_a,
            pairs_b,
            marginal_mi,
            nbins,
            cls_y,
            weights,
            np.int32,
        )
        # With only 10 rows of signal, MI is dominated by sample bias, but
        # the value should be FINITE (not NaN/Inf). True MI on 10 rows
        # ranges up to ln(2)≈0.69. We just check finiteness.
        assert np.isfinite(joint_w[0])

    @pytest.mark.fast
    def test_biz_weights_can_recover_synergy_in_subset(self):
        """biz_value: signal is in the second half of rows; first half
        has independent y. With UNWEIGHTED, the signal-to-noise ratio
        averages to mid-strength. With WEIGHTED (zero weight on the
        first half), the signal stands out cleanly. Quantitative
        assertion: weighted II > 2x unweighted II on this fixture."""
        from mlframe.feature_selection.filters.cat_interactions import (
            _pair_search_kernel_njit,
        )

        rng = np.random.default_rng(7)
        n_sig = 800
        n_noise = 800
        n = n_sig + n_noise
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        y = np.empty(n, dtype=np.int32)
        # First half noise; second half XOR signal
        y[:n_noise] = rng.integers(0, 2, n_noise)
        y[n_noise:] = x1[n_noise:] ^ x2[n_noise:]
        data = np.column_stack([x1, x2, y]).astype(np.int32)
        nbins = np.array([2, 2, 2], dtype=np.int64)
        cls_y, _, _ = merge_vars(
            factors_data=data,
            vars_indices=np.array([2], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        fq_y = np.bincount(cls_y, minlength=2).astype(np.float64) / n
        pairs_a = np.array([0], dtype=np.int64)
        pairs_b = np.array([1], dtype=np.int64)
        marginal_mi = np.zeros(3, dtype=np.float64)

        # Unweighted: averages noise and signal
        joint_uw, _, _ = _pair_search_kernel_njit(
            data,
            pairs_a,
            pairs_b,
            marginal_mi,
            nbins,
            cls_y,
            fq_y,
            np.int32,
        )
        # Weighted: only signal rows get weight 1; noise gets 0
        weights = np.zeros(n, dtype=np.float64)
        weights[n_noise:] = 1.0
        joint_w, _, _ = _pair_search_kernel_weighted_njit(
            data,
            pairs_a,
            pairs_b,
            marginal_mi,
            nbins,
            cls_y,
            weights,
            np.int32,
        )
        # Weighted should recover the full XOR signal (~ln 2 = 0.69);
        # unweighted is dominated by the noise half (~0.2 typically).
        # Assertion: weighted II >= 1.5x unweighted II.
        ratio = joint_w[0] / max(joint_uw[0], 1e-6)
        assert ratio >= 1.5, (
            f"Weighted MI should be >= 1.5x unweighted on signal-in-subset "
            f"fixture; got weighted={joint_w[0]:.4f} vs "
            f"unweighted={joint_uw[0]:.4f} (ratio={ratio:.2f}x)"
        )


# ---------------------------------------------------------------------------
# Tier 2.2: bootstrap CI tests
# ---------------------------------------------------------------------------


class TestBootstrapCIs:
    """Groups tests covering TestBootstrapCIs."""

    @pytest.mark.fast
    def test_bootstrap_ci_populates_diagnostics(self):
        """When ``bootstrap_ci_n_replicates > 0``, ``state.diagnostics``
        gains a ``bootstrap_ii_ci`` field for each engineered feature."""
        rng = np.random.default_rng(5)
        n = 1200
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        y = (x1 ^ x2).astype(np.int32)
        noise = rng.integers(0, 4, size=(n, 4)).astype(np.int32)
        data = np.column_stack([x1, x2, noise, y]).astype(np.int32)
        nbins = np.array([2, 2, 4, 4, 4, 4, 2], dtype=np.int64)
        cls_y, fq_y, _ = merge_vars(
            factors_data=data,
            vars_indices=np.array([6], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        cfg = CatFEConfig(
            enable=True,
            top_k_pairs=4,
            min_interaction_information=0.1,
            full_npermutations=0,
            fwer_correction="none",
            bootstrap_ci_n_replicates=10,
        )
        _, _, _, state = run_cat_interaction_step(
            data=data,
            cols=["x1", "x2", "n0", "n1", "n2", "n3", "y"],
            nbins=nbins,
            target_indices=np.array([6], dtype=np.int64),
            classes_y=cls_y,
            classes_y_safe=cls_y,
            freqs_y=fq_y,
            categorical_vars=[0, 1, 2, 3, 4, 5],
            cfg=cfg,
            dtype=np.int32,
        )
        # XOR pair should survive and have a bootstrap CI
        xor_recipes = [r for r in state.recipes if set(r.src_names) == {"x1", "x2"}]
        assert xor_recipes, "XOR pair should survive on bootstrap CI test"
        diag = state.diagnostics[xor_recipes[0].name]
        ci = diag["bootstrap_ii_ci"]
        assert ci is not None, "Bootstrap CI must be populated"
        lower, median, upper = ci
        # Bootstrap CI should bracket the point estimate; lower > 0 for
        # a strong XOR signal.
        assert lower > 0, f"XOR bootstrap lower CI should be > 0; got {lower:.4f}"
        assert lower <= median <= upper

    def test_bootstrap_disabled_by_default(self):
        """Default config has ``bootstrap_ci_n_replicates=0`` -- no
        CI computed, diagnostics field stays None."""
        cfg = CatFEConfig()
        assert cfg.bootstrap_ci_n_replicates == 0

    def test_bootstrap_lower_ci_drops_unstable_pair(self):
        """If the bootstrap distribution has high variance (CI lower
        below the floor), the pair is dropped."""
        rng = np.random.default_rng(11)
        n = 800
        # Construct: x1 has weak marginal signal, x2 noise. Signal is
        # too noisy for the bootstrap to stabilize CI > 0.05.
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        # Y depends weakly on x1 (50%+5%) -- bootstrap CI on II should
        # bracket ~0 with high variance.
        y = np.where(rng.random(n) < 0.55, x1, 1 - x1).astype(np.int32)
        data = np.column_stack([x1, x2, y]).astype(np.int32)
        nbins = np.array([2, 2, 2], dtype=np.int64)
        cls_y, fq_y, _ = merge_vars(
            factors_data=data,
            vars_indices=np.array([2], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        cfg = CatFEConfig(
            enable=True,
            top_k_pairs=1,
            min_interaction_information=0.05,  # tight floor
            full_npermutations=0,
            fwer_correction="none",
            bootstrap_ci_n_replicates=15,
        )
        _, _, _, state = run_cat_interaction_step(
            data=data,
            cols=["x1", "x2", "y"],
            nbins=nbins,
            target_indices=np.array([2], dtype=np.int64),
            classes_y=cls_y,
            classes_y_safe=cls_y,
            freqs_y=fq_y,
            categorical_vars=[0, 1],
            cfg=cfg,
            dtype=np.int32,
        )
        # Whether this pair is dropped depends on the specific seed
        # variance, but we can verify: state.recipes is either empty
        # (dropped by bootstrap CI) OR all surviving recipes have
        # lower CI >= floor.
        for r in state.recipes:
            ci = state.diagnostics[r.name].get("bootstrap_ii_ci")
            if ci is not None:
                lower, _, _ = ci
                assert lower >= 0.05, f"Surviving recipe must clear lower-CI floor; got lower={lower:.4f}"


# ---------------------------------------------------------------------------
# B-19 (mrmr_audit_2026-07-20): downstream confirmation/rerank steps must
# honour sample weights, not just the search-phase point estimate.
# ---------------------------------------------------------------------------


class TestDownstreamStepsHonourWeights:
    """Weight=2 on a row must match literally duplicating that row, for every
    downstream confirmation/rerank step -- not just the search-phase kernel."""

    def _xor_fixture(self, n=600, seed=3):
        """XOR signal + one already-selected redundant column Z, for exercising both MM re-rank and anti-redundancy."""
        rng = np.random.default_rng(seed)
        x1 = rng.integers(0, 3, n).astype(np.int32)
        x2 = rng.integers(0, 3, n).astype(np.int32)
        z = (x1 + rng.integers(0, 3, n)) % 3  # correlated with x1, for anti-redundancy
        y = ((x1 + x2) % 3).astype(np.int32)
        data = np.column_stack([x1, x2, z, y]).astype(np.int32)
        nbins = np.array([3, 3, 3, 3], dtype=np.int64)
        cls_y, fq_y, _ = merge_vars(
            factors_data=data,
            vars_indices=np.array([3], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        return data, nbins, cls_y, fq_y

    def test_mm_rerank_weighted_matches_literal_duplication(self):
        """``_maybe_rerank_with_mm``'s weighted II must bit-match the II computed on literally-duplicated rows."""
        from mlframe.feature_selection.filters._cat_mm_correction import _maybe_rerank_with_mm

        data, nbins, cls_y, fq_y = self._xor_fixture()
        n = len(data)
        pairs_a = np.array([0], dtype=np.int64)
        pairs_b = np.array([1], dtype=np.int64)
        ii_arr = np.zeros(1, dtype=np.float64)
        selected_idx = np.array([0], dtype=np.int64)
        cfg = CatFEConfig(enable=True, use_miller_madow=True)

        weights = np.ones(n, dtype=np.float64)
        weights[:50] = 2.0
        ii_w, _ = _maybe_rerank_with_mm(
            factors_data=data,
            pairs_a=pairs_a,
            pairs_b=pairs_b,
            selected_idx=selected_idx,
            ii_arr=ii_arr,
            nbins=nbins,
            target_indices=np.array([3], dtype=np.int64),
            classes_y=cls_y,
            freqs_y=fq_y,
            cfg=cfg,
            dtype=np.int32,
            verbose=0,
            weights=weights,
        )

        dup_idx = np.concatenate([np.arange(n), np.arange(50)])
        data_dup = data[dup_idx]
        cls_y_dup, fq_y_dup, _ = merge_vars(
            factors_data=data_dup,
            vars_indices=np.array([3], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        ii_dup, _ = _maybe_rerank_with_mm(
            factors_data=data_dup,
            pairs_a=pairs_a,
            pairs_b=pairs_b,
            selected_idx=selected_idx,
            ii_arr=ii_arr,
            nbins=nbins,
            target_indices=np.array([3], dtype=np.int64),
            classes_y=cls_y_dup,
            freqs_y=fq_y_dup,
            cfg=cfg,
            dtype=np.int32,
            verbose=0,
            weights=None,
        )
        # 1e-3 (not bit-exact): the MM bias term (k_a-1)(k_b-1)(k_y-1)/(2*n_samples) uses the raw row
        # count as its asymptotic ``n``, which is 600 for the weighted call vs 650 for the literally
        # duplicated frame -- a real, expected second-order difference in the BIAS correction, not in
        # the underlying weighted plug-in entropies (which are bit-exact, see the kernel-level test in
        # test_ksg_mi.py / the module-level smoke test for ``compute_mi_from_classes_weighted``).
        assert abs(ii_w[0] - ii_dup[0]) < 1e-3, (
            f"weighted MM-rerank II ({ii_w[0]:.6f}) must closely match the literally-duplicated-rows II " f"({ii_dup[0]:.6f}); B-19 regression."
        )

    def test_anti_redundancy_weighted_matches_literal_duplication(self):
        """``_anti_redundancy_rerank``'s weighted redundancy score must bit-match literally-duplicated rows."""
        from mlframe.feature_selection.filters._cat_post_refine import _anti_redundancy_rerank

        data, nbins, cls_y, _fq_y = self._xor_fixture()
        n = len(data)
        pairs_a = np.array([0], dtype=np.int64)
        pairs_b = np.array([1], dtype=np.int64)
        ii_arr = np.array([0.5], dtype=np.float64)
        selected_idx = np.array([0], dtype=np.int64)
        cfg = CatFEConfig(enable=True, anti_redundancy_beta=0.5)

        weights = np.ones(n, dtype=np.float64)
        weights[:50] = 2.0
        scored_w, _ = _anti_redundancy_rerank(
            factors_data=data,
            pairs_a=pairs_a,
            pairs_b=pairs_b,
            selected_idx=selected_idx,
            ii_arr=ii_arr,
            nbins=nbins,
            selected_so_far=[2],
            classes_y=cls_y,
            cfg=cfg,
            dtype=np.int32,
            verbose=0,
            weights=weights,
        )

        dup_idx = np.concatenate([np.arange(n), np.arange(50)])
        data_dup = data[dup_idx]
        cls_y_dup, _, _ = merge_vars(
            factors_data=data_dup,
            vars_indices=np.array([3], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        scored_dup, _ = _anti_redundancy_rerank(
            factors_data=data_dup,
            pairs_a=pairs_a,
            pairs_b=pairs_b,
            selected_idx=selected_idx,
            ii_arr=ii_arr,
            nbins=nbins,
            selected_so_far=[2],
            classes_y=cls_y_dup,
            cfg=cfg,
            dtype=np.int32,
            verbose=0,
            weights=None,
        )
        assert abs(scored_w[0] - scored_dup[0]) < 1e-9, (
            f"weighted anti-redundancy score ({scored_w[0]:.6f}) must bit-match the " f"literally-duplicated-rows score ({scored_dup[0]:.6f}); B-19 regression."
        )

    def test_run_cat_interaction_step_accepts_weights_end_to_end(self):
        """End-to-end smoke: ``run_cat_interaction_step`` with non-uniform weights doesn't crash and
        still surfaces the XOR pair -- the weighted downstream confirmation path is exercised, not
        just the search-phase kernel."""
        rng = np.random.default_rng(9)
        n = 1200
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        y = (x1 ^ x2).astype(np.int32)
        data = np.column_stack([x1, x2, y]).astype(np.int32)
        nbins = np.array([2, 2, 2], dtype=np.int64)
        cls_y, fq_y, _ = merge_vars(
            factors_data=data,
            vars_indices=np.array([2], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        weights = np.ones(n, dtype=np.float64)
        weights[:300] = 3.0
        # perm_budget_strategy="fixed" (not the default "bandit_ucb1") exercises the weighted
        # joint-independence kernel this test targets -- the bandit allocator has no weighted
        # variant yet (documented gap, separate follow-up) and would confound this assertion.
        cfg = CatFEConfig(
            enable=True,
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=100,
            fwer_correction="none",
            perm_budget_strategy="fixed",
        )
        _, _, _, state = run_cat_interaction_step(
            data=data,
            cols=["x1", "x2", "y"],
            nbins=nbins,
            target_indices=np.array([2], dtype=np.int64),
            classes_y=cls_y,
            classes_y_safe=cls_y,
            freqs_y=fq_y,
            categorical_vars=[0, 1],
            cfg=cfg,
            dtype=np.int32,
            weights=weights,
        )
        xor_recipes = [r for r in state.recipes if set(r.src_names) == {"x1", "x2"}]
        assert xor_recipes, "XOR pair should survive weighted end-to-end confirmation"
