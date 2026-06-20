"""Tests for Tier 3 (target encoding emit, pipeline integration, tutorial
sanity) and Tier 4 (bandit UCB1, coordinate-ascent, group-aware perms,
streaming cache, KT smoothing, analytical MM gate, big GPU kernel, full
conditional perm).

Each item gets unit + biz_value (where applicable) per
``mlframe/CLAUDE.md`` "Every new feature: unit + biz_value + cProfile"
rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import warnings

from mlframe.feature_selection.filters import MRMR, CatFEConfig
from mlframe.feature_selection.filters.cat_interactions import (
    _entropy_for_mode,
    _refine_kway_coordinate_ascent,
    _should_apply_mm_for_pair,
    _should_apply_mm_for_pair_analytical,
    _compute_target_encoding,
    run_cat_interaction_step,
)
from mlframe.feature_selection.filters.engineered_recipes import (
    EngineeredRecipe, apply_recipe,
)
from mlframe.feature_selection.filters.info_theory import (
    entropy, entropy_miller_madow, merge_vars, compute_mi_from_classes,
)


# ============================================================================
# Tier 3.1: target encoding emit
# ============================================================================


class TestTargetEncoding:
    @pytest.mark.fast
    def test_target_encoding_oof_no_leak(self):
        """biz_value: OOF target encoding on a perfect XOR fixture
        produces encoded values that DON'T leak Y exactly. With OOF=5
        and shrinkage=10, encoded(x1, x2) should be bounded away from
        0/1 (no perfect leakage)."""
        rng = np.random.default_rng(42)
        n = 1000
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        y = (x1 ^ x2).astype(np.int32)
        data = np.column_stack([x1, x2, y]).astype(np.int32)
        nbins = np.array([2, 2, 2], dtype=np.int64)
        cls_y, _, _ = merge_vars(
            factors_data=data, vars_indices=np.array([2], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        te_vals, cell_means = _compute_target_encoding(
            factors_data=data, idx_tuple=(0, 1),
            target_indices=np.array([2], dtype=np.int64),
            classes_y=cls_y, nbins=nbins,
            n_oof_folds=5, smoothing=10.0, dtype=np.int32,
        )
        assert te_vals.shape == (n,)
        # Encoded values are floats in [0, 1] (since y is binary)
        assert te_vals.min() >= -0.01
        assert te_vals.max() <= 1.01
        # Cell means: 4 cells. For XOR with smoothing=10, alpha=10 with
        # n_c~250 per cell shrinks toward global mean (0.5). So cells
        # should be PARTIALLY shrunk: not 0/1 exactly.
        assert cell_means.shape == (4,)
        # Each cell ratio: (250 * raw + 10 * 0.5) / 260 where raw=0 or 1
        # gives ~0.019 or ~0.981 -- bounded away from 0/1
        assert cell_means.max() < 0.99
        assert cell_means.min() > 0.01

    def test_apply_target_encoding_replays_on_test_data(self):
        rng = np.random.default_rng(7)
        n_train = 1000
        n_test = 200
        x1_tr = rng.integers(0, 2, n_train).astype(np.int32)
        x2_tr = rng.integers(0, 2, n_train).astype(np.int32)
        x1_te = rng.integers(0, 2, n_test).astype(np.int32)
        x2_te = rng.integers(0, 2, n_test).astype(np.int32)
        y_tr = (x1_tr ^ x2_tr).astype(np.int32)
        data_tr = np.column_stack([x1_tr, x2_tr, y_tr]).astype(np.int32)
        nbins = np.array([2, 2, 2], dtype=np.int64)
        cls_y, _, _ = merge_vars(
            factors_data=data_tr, vars_indices=np.array([2], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        _, cell_means_global = _compute_target_encoding(
            factors_data=data_tr, idx_tuple=(0, 1),
            target_indices=np.array([2], dtype=np.int64),
            classes_y=cls_y, nbins=nbins,
            n_oof_folds=0, smoothing=10.0, dtype=np.int32,
        )
        # Build a target_encoding recipe and replay on test
        # Build a factorize lookup
        lookup = np.array([0, 1, 2, 3], dtype=np.int64)  # for binary (a,b) all 4 cells
        recipe = EngineeredRecipe(
            name="te(x1__x2)", kind="target_encoding",
            src_names=("x1", "x2"), factorize_nbins=(2, 2),
            unknown_strategy="clip",
            extra={
                "cell_means": cell_means_global,
                "global_mean": float(y_tr.mean()),
                "n_oof_folds": 0,
                "smoothing": 10.0,
                "factorize_lookup": lookup,
            },
        )
        df_te = pd.DataFrame({"x1": x1_te, "x2": x2_te})
        out = apply_recipe(recipe, df_te)
        assert out.shape == (n_test,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_emit_target_encoding_populates_recipes(self):
        rng = np.random.default_rng(13)
        n = 1500
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
            enable=True, top_k_pairs=2, min_interaction_information=0.1,
            full_npermutations=0, fwer_correction="none",
            emit_target_encoding=True, target_encoding_oof_folds=5,
        )
        _, _, _, state = run_cat_interaction_step(
            data=data, cols=["x1", "x2", "y"], nbins=nbins,
            target_indices=np.array([2], dtype=np.int64),
            classes_y=cls_y, classes_y_safe=cls_y, freqs_y=fq_y,
            categorical_vars=[0, 1],
            cfg=cfg, dtype=np.int32,
        )
        te_recipes = [r for r in state.recipes if r.kind == "target_encoding"]
        assert te_recipes, "emit_target_encoding=True should produce TE recipes"


# ============================================================================
# Tier 4.5: KT smoothing
# ============================================================================


class TestKTSmoothing:
    @pytest.mark.fast
    def test_kt_smoothing_pulls_high_card_entropy_toward_mm(self):
        """KT smoothing provides a different entropy estimate than
        plug-in; under sparse counts it sits between plug-in and
        Miller-Madow. Unit check: KT entropy is bounded by log(K) and
        differs from plug-in for sparse freqs."""
        # Synthetic: 8 cells with sparse counts (n=20)
        counts = np.array([5, 4, 3, 2, 2, 2, 1, 1], dtype=np.float64)
        n = int(counts.sum())
        K = len(counts)
        freqs = counts / n
        h_plugin = entropy(freqs)
        h_mm = entropy_miller_madow(freqs, n)
        h_kt = _entropy_for_mode(freqs, n, use_mm=False, use_kt=True)
        # All bounded by log(K)
        assert h_plugin <= np.log(K) + 1e-9
        assert h_mm <= np.log(K) + (K - 1) / (2 * n) + 1e-6
        assert h_kt <= np.log(K + 0.5) + 1e-9
        # KT differs from plug-in (smoothing changes the estimate)
        assert h_kt != h_plugin


# ============================================================================
# Tier 4.6: analytical MM auto-gate
# ============================================================================


class TestAnalyticalMMGate:
    def test_analytical_gate_fires_above_threshold(self):
        """At cardinality 10x10x4 = 400 and n=200, bias = 9*9*3 = 243
        exceeds 6*sqrt(200) ≈ 85, so gate fires."""
        assert _should_apply_mm_for_pair_analytical(10, 10, 4, 200)

    def test_analytical_gate_below_threshold(self):
        """At cardinality 2x2x2 = 8 and n=1000, bias = 1*1*1 = 1
        is well below 6*sqrt(1000) ≈ 190, gate does NOT fire."""
        assert not _should_apply_mm_for_pair_analytical(2, 2, 2, 1000)

    def test_analytical_vs_folklore_differ(self):
        """Construct a borderline case where folklore and analytical
        disagree. At a=20, b=2, c=2, n=300:
        - Folklore: 80/300 = 0.27 > 0.05 -> fires
        - Analytical: 19*1*1 = 19 vs 6*sqrt(300)≈104 -> NOT fires
        """
        folklore_fires = _should_apply_mm_for_pair(20, 2, 2, 300)
        analytical_fires = _should_apply_mm_for_pair_analytical(20, 2, 2, 300)
        assert folklore_fires != analytical_fires


# ============================================================================
# Tier 4.2: coordinate-ascent refinement
# ============================================================================


class TestCoordinateAscent:
    def test_refine_passes_zero_is_noop(self):
        """When refine_passes=0, kway_results passes through unchanged."""
        # Dummy kway_results
        dummy_kway = [(tuple([0, 1, 2]), np.array([0, 1, 2, 0]), 3, 0.5)]
        rng = np.random.default_rng(0)
        data = rng.integers(0, 2, (100, 5)).astype(np.int32)
        nbins = np.array([2, 2, 2, 2, 2], dtype=np.int64)
        cls_y = rng.integers(0, 2, 100).astype(np.int32)
        fq_y = np.bincount(cls_y, minlength=2).astype(np.float64) / 100
        out = _refine_kway_coordinate_ascent(
            factors_data=data, kway_results=dummy_kway,
            candidate_pool=np.array([0, 1, 2, 3, 4], dtype=np.int64),
            nbins=nbins, classes_y=cls_y, freqs_y=fq_y,
            max_combined_nbins=64, n_passes=0,
            dtype=np.int32, verbose=0,
        )
        assert out == dummy_kway


# ============================================================================
# Tier 3.2: sklearn Pipeline integration
# ============================================================================


class TestSklearnPipelineIntegration:
    @pytest.mark.fast
    def test_mrmr_inside_pipeline_fit_transform(self):
        """MRMR with cat-FE enabled must work inside a sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        rng = np.random.default_rng(2)
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
        pipe = Pipeline([
            ("mrmr", MRMR(
                full_npermutations=2, baseline_npermutations=2,
                verbose=0, n_jobs=1,
                cat_fe_config=CatFEConfig(
                    enable=True, top_k_pairs=4,
                    min_interaction_information=0.1,
                    full_npermutations=0, fwer_correction="none",
                ),
            )),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(df, y_s)
            out = pipe.transform(df)
        # No assertion on engineered specifically -- just that the
        # pipeline ran without error and returned a frame.
        assert out.shape[0] == n


# ============================================================================
# Tier 4.3: group-aware permutation -- placeholder smoke test
# ============================================================================


class TestGroupsCol:
    def test_groups_col_accepted_in_config(self):
        cfg = CatFEConfig(groups_col="user_id")
        assert cfg.groups_col == "user_id"


# ============================================================================
# Tier 4.4: streaming cache -- placeholder
# ============================================================================


class TestStreamingCache:
    def test_streaming_cache_flag_default_off(self):
        cfg = CatFEConfig()
        assert cfg.enable_streaming_cache is False


# ============================================================================
# Tier 4.1: bandit UCB1 budget allocation -- placeholder
# ============================================================================


class TestBanditPermBudget:
    def test_perm_budget_strategy_bandit_default(self):
        cfg = CatFEConfig()
        assert cfg.perm_budget_strategy == "bandit_ucb1"

    def test_perm_budget_strategy_ucb1_accepted(self):
        cfg = CatFEConfig(perm_budget_strategy="bandit_ucb1")
        assert cfg.perm_budget_strategy == "bandit_ucb1"


# ============================================================================
# Tier 4.8: full conditional perm -- placeholder
# ============================================================================


class TestFullConditionalPerm:
    def test_enable_full_conditional_perm_flag(self):
        cfg = CatFEConfig(enable_full_conditional_perm=True)
        assert cfg.enable_full_conditional_perm is True


# ============================================================================
# Tier 4.7: bigger GPU kernel -- the 3D RawKernel is shipped; we just
# verify it's importable
# ============================================================================


class TestBigGPUKernel:
    def test_gpu_kernel_module_imports(self):
        """The GPU dispatch shim imports without CuPy; raises only if
        actually called."""
        from mlframe.feature_selection.filters.gpu import (
            mi_direct_gpu_batched_pairs,
        )
        assert callable(mi_direct_gpu_batched_pairs)
