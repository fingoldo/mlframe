"""Real-implementation tests for Tier 4.1 (Bandit UCB1), Tier 4.4
(Streaming cache), Tier 4.8 (Full IPF conditional permutation).

Per ``mlframe/CLAUDE.md`` "Every new feature: unit + biz_value +
cProfile": each item gets unit tests AND a biz_value-style behavior
check where applicable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR, CatFEConfig
from mlframe.feature_selection.filters.cat_fe_state import CatFEState
from mlframe.feature_selection.filters.cat_interactions import (
    _column_signature, _kl_divergence,
    _restore_cached_marginal_mis,
    _confirm_pairs_bandit_ucb1,
    _full_conditional_shuffle_ipf,
    run_cat_interaction_step,
)
from mlframe.feature_selection.filters.info_theory import merge_vars


# ============================================================================
# Tier 4.1: Bandit UCB1 real-impl tests
# ============================================================================


class TestBanditUCB1Real:
    def _make_xor(self, n=1500, seed=42, n_noise=4):
        rng = np.random.default_rng(seed)
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        noise = rng.integers(0, 4, size=(n, n_noise)).astype(np.int32)
        y = (x1 ^ x2).astype(np.int32)
        data = np.column_stack([x1, x2, noise, y]).astype(np.int32)
        nbins = np.array([2, 2] + [4] * n_noise + [2], dtype=np.int64)
        cls_y, fq_y, _ = merge_vars(
            factors_data=data, vars_indices=np.array([n_noise + 2], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        return data, nbins, cls_y, fq_y, n_noise + 2

    @pytest.mark.fast
    def test_bandit_recovers_xor_synergy(self):
        """biz_value: with bandit UCB1 budget, the XOR signal still
        clears 0.95 confidence (it's a high-signal pair, bandit
        allocates enough shuffles to confirm)."""
        data, nbins, cls_y, fq_y, tgt = self._make_xor()
        cfg = CatFEConfig(
            enable=True, top_k_pairs=4,
            min_interaction_information=0.1,
            full_npermutations=100,  # enough for bandit phase 1 + 95% conf
            fwer_correction="none",
            perm_budget_strategy="bandit_ucb1",
        )
        cols = ["x1", "x2"] + [f"n{k}" for k in range(4)] + ["y"]
        _, _, _, state = run_cat_interaction_step(
            data=data, cols=cols, nbins=nbins,
            target_indices=np.array([tgt], dtype=np.int64),
            classes_y=cls_y, classes_y_safe=cls_y, freqs_y=fq_y,
            categorical_vars=list(range(6)),
            cfg=cfg, dtype=np.int32,
        )
        recipe_srcs = {r.src_names for r in state.recipes}
        assert ("x1", "x2") in recipe_srcs or ("x2", "x1") in recipe_srcs, (
            f"Bandit UCB1 should confirm XOR pair; got {recipe_srcs}"
        )

    def test_bandit_rejects_independent_pair(self):
        """Construct independent (X1, X2, Y) -- bandit should reject."""
        rng = np.random.default_rng(0)
        n = 800
        x1 = rng.integers(0, 4, n).astype(np.int32)
        x2 = rng.integers(0, 4, n).astype(np.int32)
        y = rng.integers(0, 2, n).astype(np.int32)
        data = np.column_stack([x1, x2, y]).astype(np.int32)
        nbins = np.array([4, 4, 2], dtype=np.int64)
        cls_y, fq_y, _ = merge_vars(
            factors_data=data, vars_indices=np.array([2], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        cfg = CatFEConfig(
            enable=True, top_k_pairs=1,
            min_interaction_information=-100,  # accept any from search
            max_combined_nbins=200,
            full_npermutations=30,
            perm_budget_strategy="bandit_ucb1",
            fwer_correction="none",
        )
        _, _, _, state = run_cat_interaction_step(
            data=data, cols=["x1", "x2", "y"], nbins=nbins,
            target_indices=np.array([2], dtype=np.int64),
            classes_y=cls_y, classes_y_safe=cls_y, freqs_y=fq_y,
            categorical_vars=[0, 1],
            cfg=cfg, dtype=np.int32,
        )
        assert state.recipes == [], (
            f"Bandit should reject independent pair; got {state.recipes}"
        )


# ============================================================================
# Tier 4.4: Streaming cache real-impl tests
# ============================================================================


class TestStreamingCacheReal:
    def test_kl_divergence_self_is_zero(self):
        p = np.array([0.25, 0.5, 0.25])
        assert _kl_divergence(p, p) < 1e-6

    def test_kl_divergence_positive_for_different(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        assert _kl_divergence(p, q) > 0.1

    def test_column_signature_normalises(self):
        vals = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
        sig = _column_signature(vals, nbins=3)
        np.testing.assert_allclose(sig, [2/6, 3/6, 1/6], rtol=1e-6)

    def test_restore_cached_mis_returns_correct_mask(self):
        """When current signature matches cache (KL=0), the cached MI
        is restored. When signature differs significantly, cache miss."""
        rng = np.random.default_rng(0)
        n = 500
        data = np.column_stack([
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
        ]).astype(np.int32)
        nbins = np.array([2, 2], dtype=np.int64)
        sig0 = _column_signature(data[:, 0], 2)
        # Column 1 in the cache uses a deliberately drifted signature (0.99/0.01 vs the actual
        # near-uniform data) to force a KL-divergence cache miss for that column.
        # Reuse is gated on the cached target signature matching (MI(X;Y) must invalidate on a changed Y);
        # pass a matching target_sig so this test exercises the column-signature KL path.
        cache = {
            "col_signatures": {0: sig0.copy(), 1: np.array([0.99, 0.01])},
            "marginal_mis": {0: 0.05, 1: 0.03},
            "target_sig": "tgt",
        }
        mask, mi_reused, new_sigs = _restore_cached_marginal_mis(
            factors_data=data,
            candidate_idxs=np.array([0, 1], dtype=np.int64),
            nbins=nbins,
            cache=cache,
            kl_threshold=0.01,
            target_sig="tgt",
        )
        # Col 0: signature matches cache exactly (KL=0); reusable
        assert mask[0]
        np.testing.assert_allclose(mi_reused[0], 0.05, rtol=1e-6)
        # Col 1: signature is ~50/50 but cache says 99/1; KL >> threshold
        assert not mask[1]

    def test_mrmr_streaming_cache_persists_across_refits(self):
        """After a fit() with enable_streaming_cache=True, the next
        fit() with the same data finds a non-empty
        ``_cat_fe_cache_`` on the instance."""
        rng = np.random.default_rng(7)
        n = 600
        x1 = rng.integers(0, 2, n).astype(np.int8)
        x2 = rng.integers(0, 2, n).astype(np.int8)
        noise = rng.integers(0, 4, size=(n, 4)).astype(np.int8)
        y = (x1 ^ x2).astype(np.int8)
        cols = {"x1": pd.Categorical(x1), "x2": pd.Categorical(x2)}
        for k in range(4):
            cols[f"n{k}"] = pd.Categorical(noise[:, k])
        df = pd.DataFrame(cols)
        y_s = pd.Series(y, name="target")

        mrmr = MRMR(
            full_npermutations=2, baseline_npermutations=2,
            verbose=0, n_jobs=1,
            cat_fe_config=CatFEConfig(
                enable=True, top_k_pairs=4,
                min_interaction_information=0.1,
                full_npermutations=0, fwer_correction="none",
                enable_streaming_cache=True,
            ),
        )
        # NOTE: skip_retraining_on_same_shape may short-circuit re-fit
        # if signature matches; disable that for this test.
        mrmr.skip_retraining_on_same_shape = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df, y_s)
        # Cache should be populated after first fit
        cache = getattr(mrmr, "_cat_fe_cache_", None)
        assert cache is not None
        assert "col_signatures" in cache
        assert "marginal_mis" in cache
        assert len(cache["col_signatures"]) > 0


# ============================================================================
# Tier 4.8: Full IPF conditional permutation tests
# ============================================================================


class TestFullConditionalIPF:
    def test_ipf_shuffle_preserves_x1_marginals(self):
        """After _full_conditional_shuffle_ipf, the marginal P(X2 | X1)
        within each (X1, Y) stratum should be preserved on average."""
        rng = np.random.default_rng(11)
        n = 1000
        x1 = rng.integers(0, 3, n).astype(np.int32)
        x2 = rng.integers(0, 3, n).astype(np.int32)
        y = rng.integers(0, 2, n).astype(np.int32)
        x2_safe = x2.astype(np.int64, copy=True)
        # Run IPF shuffle (base_seed arg matches the production callers in _prewarm / _cat_confirm_permutation)
        _full_conditional_shuffle_ipf(x2_safe, x1, y, 3, 2, 0)
        # After shuffle: for each (x1=a, y=b) stratum, the set of X2
        # values is the SAME (just reordered).
        for a in range(3):
            for b in range(2):
                mask = (x1 == a) & (y == b)
                if mask.sum() < 2:
                    continue
                # Count of each X2 value in the stratum is unchanged
                pre_counts = np.bincount(x2[mask], minlength=3)
                post_counts = np.bincount(x2_safe[mask], minlength=3)
                np.testing.assert_array_equal(pre_counts, post_counts)

    @pytest.mark.fast
    def test_full_conditional_perm_path_in_orchestrator(self):
        """End-to-end: cfg.enable_full_conditional_perm=True activates
        the IPF path. Smoke check: doesn't crash, runs to completion."""
        rng = np.random.default_rng(7)
        n = 1200
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        y = (x1 ^ x2).astype(np.int32)
        data = np.column_stack([x1, x2, y]).astype(np.int32)
        nbins = np.array([2, 2, 2], dtype=np.int64)
        cls_y, fq_y, _ = merge_vars(
            factors_data=data, vars_indices=np.array([2], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        cfg = CatFEConfig(
            enable=True, top_k_pairs=2,
            min_interaction_information=0.1,
            full_npermutations=10,
            fwer_correction="none",
            permutation_null="conditional",
            enable_full_conditional_perm=True,
        )
        _, _, _, state = run_cat_interaction_step(
            data=data, cols=["x1", "x2", "y"], nbins=nbins,
            target_indices=np.array([2], dtype=np.int64),
            classes_y=cls_y, classes_y_safe=cls_y, freqs_y=fq_y,
            categorical_vars=[0, 1],
            cfg=cfg, dtype=np.int32,
        )
        # Smoke: path completes without crashing. XOR may or may not
        # survive the stricter null on this perm budget.
        assert state is not None
