"""Coverage tests for ``mlframe.feature_selection.filters.cat_interactions``.

Each test exercises a single CatFEConfig knob or helper-function branch to
push coverage of the orchestrator file past 50%. Tests use small fixtures
(n=200-500) with fixed seeds and ``full_npermutations`` kept very small so
the suite stays under the 600s timeout.

Coverage strategy:
- Knob-per-test on ``run_cat_interaction_step`` for every branch in
  ``CatFEConfig`` (MM, KT, select_on, permutation_null, perm_budget_strategy,
  mht corrections, K-fold stability, anti-redundancy, k-way expansion,
  marginal floor, include_numeric proxy via cardinality, max_combined_nbins,
  bootstrap, subsample, target-encoding, streaming cache, refine passes).
- Direct calls into helper functions (``_column_signature``,
  ``_kl_divergence``, ``_restore_cached_marginal_mis``,
  ``_select_candidate_indices``, ``_select_top_k_pairs``,
  ``_should_apply_mm_for_pair``, ``_should_apply_mm_for_pair_analytical``,
  ``_apply_fwer_correction``, ``_entropy_for_mode``,
  ``_pair_search_kernel_weighted_njit``, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig, CatFEState
from mlframe.feature_selection.filters.cat_interactions import (
    _anti_redundancy_rerank,
    _apply_fwer_correction,
    _build_factorize_lookup,
    _column_signature,
    _compute_pair_ii_mm,
    _confirm_pairs_via_permutation,
    _entropy_for_mode,
    _greedy_expand_one_seed,
    _kfold_stability_filter,
    _kl_divergence,
    _marginal_screen_njit,
    _maybe_rerank_with_mm,
    _pair_search_kernel_njit,
    _pair_search_kernel_weighted_njit,
    _perm_kernel_dispatch_use_gpu,
    _restore_cached_marginal_mis,
    _select_candidate_indices,
    _select_top_k_pairs,
    _should_apply_mm_for_pair,
    _should_apply_mm_for_pair_analytical,
    resolve_max_combined_nbins,
    resolve_min_interaction_information,
    run_cat_interaction_step,
)
from mlframe.feature_selection.filters._cat_interactions_step import _cat_fe_auto_wants_gpu
from mlframe.feature_selection.filters.info_theory import merge_vars

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_xor_fixture(n: int = 400, seed: int = 7, extra_noise_cols: int = 4):
    """Build a small XOR dataset: y = x1 XOR x2 plus noise columns.

    Returns dict mirroring the existing test_cat_interactions.py
    ``xor_fixture`` shape.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, n).astype(np.int32)
    x2 = rng.integers(0, 2, n).astype(np.int32)
    noise = rng.integers(0, 3, size=(n, extra_noise_cols)).astype(np.int32)
    y = (x1 ^ x2).astype(np.int32)

    data = np.column_stack([x1, x2, noise, y]).astype(np.int32)
    nbins = np.array([2, 2] + [3] * extra_noise_cols + [2], dtype=np.int64)
    cols = ["x1", "x2"] + [f"n{i}" for i in range(extra_noise_cols)] + ["y"]
    target_idx = data.shape[1] - 1

    classes_y, freqs_y, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([target_idx], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    return {
        "data": data,
        "cols": cols,
        "nbins": nbins,
        "classes_y": classes_y,
        "freqs_y": freqs_y,
        "target_indices": np.array([target_idx], dtype=np.int64),
        "categorical_vars": list(range(data.shape[1] - 1)),
    }


@pytest.fixture
def xor_small():
    """n=400 XOR fixture used by every orchestrator test below."""
    return _make_xor_fixture(n=400, seed=7, extra_noise_cols=4)


@pytest.fixture
def xor_medium():
    """n=500 XOR fixture for slightly heavier tests."""
    return _make_xor_fixture(n=500, seed=11, extra_noise_cols=3)


def _run(fx, cfg, **extras):
    """Call ``run_cat_interaction_step`` with the standard fixture mapping."""
    return run_cat_interaction_step(
        data=fx["data"],
        cols=fx["cols"],
        nbins=fx["nbins"],
        target_indices=fx["target_indices"],
        classes_y=fx["classes_y"],
        classes_y_safe=fx["classes_y"].copy(),
        freqs_y=fx["freqs_y"],
        categorical_vars=fx["categorical_vars"],
        cfg=cfg,
        dtype=np.int32,
        verbose=0,
        **extras,
    )


# ---------------------------------------------------------------------------
# 1. Helper-function unit tests (cheap, no orchestrator)
# ---------------------------------------------------------------------------


class TestColumnSignatureAndKL:
    """Test group: column-signature construction and KL-divergence reuse gate."""
    def test_column_signature_normalises_to_probability(self):
        """Column signature normalises to probability."""
        vals = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        sig = _column_signature(vals, nbins=3)
        np.testing.assert_allclose(sig, [0.4, 0.4, 0.2])
        assert sig.sum() == pytest.approx(1.0)

    def test_column_signature_empty_handles_division(self):
        """Column signature empty handles division."""
        vals = np.array([], dtype=np.int32)
        sig = _column_signature(vals, nbins=2)
        assert sig.shape == (2,)
        # No data: ``max(len(values), 1)`` defends against div-by-zero
        assert np.all(sig == 0.0)

    def test_kl_divergence_identical_is_zero(self):
        """Kl divergence identical is zero."""
        p = np.array([0.25, 0.75])
        assert _kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_kl_divergence_known_value(self):
        """Kl divergence known value."""
        # KL(p || q) with eps-smoothed inputs; sanity-check it's positive
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        assert _kl_divergence(p, q) > 0.5


class TestRestoreCachedMarginalMIs:
    """Test group: restoring cached marginal MIs when the target distribution is stable."""
    def test_reuses_when_distribution_stable(self):
        """Reuses when distribution stable."""
        rng = np.random.default_rng(0)
        n = 300
        data = rng.integers(0, 3, size=(n, 3)).astype(np.int32)
        nbins = np.array([3, 3, 3], dtype=np.int64)
        candidate_idxs = np.array([0, 1], dtype=np.int64)
        # Build a fake cache: signature exactly matches current data
        cached_sig_0 = _column_signature(data[:, 0], 3)
        cached_sig_1 = _column_signature(data[:, 1], 3)
        # Reuse is gated on the cached target signature matching (MI(X;Y) invalidates on a changed Y);
        # pass a matching target_sig so this stable-distribution case exercises the reuse path.
        cache = {
            "col_signatures": {0: cached_sig_0, 1: cached_sig_1},
            "marginal_mis": {0: 0.123, 1: 0.456},
            "target_sig": "tgt",
        }
        reuse_mask, mi_reused, new_sigs = _restore_cached_marginal_mis(
            factors_data=data,
            candidate_idxs=candidate_idxs,
            nbins=nbins,
            cache=cache,
            kl_threshold=0.5,
            target_sig="tgt",
        )
        assert reuse_mask.all()
        assert mi_reused[0] == pytest.approx(0.123)
        assert mi_reused[1] == pytest.approx(0.456)
        assert 0 in new_sigs and 1 in new_sigs

    def test_no_reuse_when_column_not_in_cache(self):
        """No reuse when column not in cache."""
        data = np.zeros((100, 2), dtype=np.int32)
        data[:, 1] = 1
        nbins = np.array([2, 2], dtype=np.int64)
        cache = {"col_signatures": {}, "marginal_mis": {}}
        reuse_mask, mi_reused, _ = _restore_cached_marginal_mis(
            factors_data=data,
            candidate_idxs=np.array([0, 1], dtype=np.int64),
            nbins=nbins,
            cache=cache,
            kl_threshold=0.5,
        )
        assert not reuse_mask.any()
        assert np.isnan(mi_reused).all()

    def test_no_reuse_when_kl_exceeds_threshold(self):
        """No reuse when kl exceeds threshold."""
        rng = np.random.default_rng(1)
        n = 200
        data = rng.integers(0, 3, size=(n, 1)).astype(np.int32)
        nbins = np.array([3], dtype=np.int64)
        # Cache the WRONG signature
        wrong_sig = np.array([0.99, 0.005, 0.005])
        cache = {
            "col_signatures": {0: wrong_sig},
            "marginal_mis": {0: 0.999},
        }
        reuse_mask, _, _ = _restore_cached_marginal_mis(
            factors_data=data,
            candidate_idxs=np.array([0], dtype=np.int64),
            nbins=nbins,
            cache=cache,
            kl_threshold=0.001,  # very tight -> KL exceeds it
        )
        assert not reuse_mask[0]


class TestResolveDefaults:
    """Test Resolve Defaults."""
    def test_max_combined_explicit_clamped(self):
        """Max combined explicit clamped."""
        cfg = CatFEConfig(max_combined_nbins=10**9)
        assert resolve_max_combined_nbins(cfg, n_samples=1000) == 10**7

    def test_max_combined_none_paninski(self):
        """Max combined none paninski."""
        cfg = CatFEConfig(max_combined_nbins=None)
        assert resolve_max_combined_nbins(cfg, n_samples=3000) == int(3000 * 0.05 / 3) + 1
        assert resolve_max_combined_nbins(cfg, n_samples=5) == 4

    def test_max_combined_explicit_small(self):
        """Max combined explicit small."""
        cfg = CatFEConfig(max_combined_nbins=20)
        assert resolve_max_combined_nbins(cfg, n_samples=10_000) == 20

    def test_min_ii_none(self):
        """Min ii none."""
        cfg = CatFEConfig(min_interaction_information=None)
        # n=10000 -> -3/100 = -0.03
        assert resolve_min_interaction_information(cfg, n_samples=10000) == pytest.approx(-0.03)
        # n=1 floor for sqrt domain
        assert resolve_min_interaction_information(cfg, n_samples=0) == pytest.approx(-3.0)

    def test_min_ii_explicit_negative(self):
        """Min ii explicit negative."""
        cfg = CatFEConfig(min_interaction_information=-0.5)
        assert resolve_min_interaction_information(cfg, n_samples=100) == -0.5

    def test_min_ii_explicit_positive(self):
        """Min ii explicit positive."""
        cfg = CatFEConfig(min_interaction_information=0.1)
        assert resolve_min_interaction_information(cfg, n_samples=100) == 0.1


class TestSelectCandidateIndices:
    """Test Select Candidate Indices."""
    def test_drop_singleton(self):
        """Drop singleton."""
        cfg = CatFEConfig()
        state = CatFEState()
        kept = _select_candidate_indices(
            nbins=np.array([1, 3, 4], dtype=np.int64),
            categorical_vars=[0, 1, 2],
            cfg=cfg, state=state, n_samples=500,
        )
        assert kept == [1, 2]
        assert state.dropped_singleton_nbins == [0]

    def test_skip_on_high_cardinality_by_default(self):
        """Skip on high cardinality by default."""
        cfg = CatFEConfig()
        state = CatFEState()
        with pytest.warns(UserWarning, match="(?i)skipping"):
            kept = _select_candidate_indices(
                nbins=np.array([2, 500], dtype=np.int64),
                categorical_vars=[0, 1],
                cfg=cfg, state=state, n_samples=400,
            )
        assert kept == [0]
        assert (1, 500) in state.high_cardinality_warnings

    def test_raise_on_high_cardinality_when_opted_in(self):
        """Raise on high cardinality when opted in."""
        cfg = CatFEConfig(on_high_cardinality="raise")
        state = CatFEState()
        with pytest.raises(ValueError, match="(?i)high-cardinality"):
            _select_candidate_indices(
                nbins=np.array([2, 500], dtype=np.int64),
                categorical_vars=[0, 1],
                cfg=cfg, state=state, n_samples=400,
            )

    def test_all_singletons_returns_empty(self):
        """All singletons returns empty."""
        cfg = CatFEConfig()
        state = CatFEState()
        kept = _select_candidate_indices(
            nbins=np.array([1, 1, 1], dtype=np.int64),
            categorical_vars=[0, 1, 2],
            cfg=cfg, state=state, n_samples=500,
        )
        assert kept == []


class TestSelectTopKPairs:
    """Test Select Top K Pairs."""
    def test_synergy_selects_positive_ii(self):
        """Synergy selects positive ii."""
        ii = np.array([0.5, -0.2, 0.3, -0.1, 0.4])
        cfg = CatFEConfig(top_k_pairs=2, min_interaction_information=0.0)
        out = _select_top_k_pairs(
            ii_arr=ii,
            pairs_a=np.arange(5),
            pairs_b=np.arange(5),
            cfg=cfg, n_samples=1000,
        )
        # Top 2 positives: indices [0, 4] (ii 0.5, 0.4)
        assert set(out.tolist()) == {0, 4}

    def test_redundancy_selects_negative_ii(self):
        """Redundancy selects negative ii."""
        ii = np.array([0.5, -0.3, 0.1, -0.4])
        cfg = CatFEConfig(top_k_pairs=2, min_interaction_information=-0.05, select_on="redundancy")
        out = _select_top_k_pairs(
            ii_arr=ii,
            pairs_a=np.arange(4),
            pairs_b=np.arange(4),
            cfg=cfg, n_samples=1000,
        )
        assert set(out.tolist()) == {1, 3}

    def test_absolute_selects_largest_magnitude(self):
        """Absolute selects largest magnitude."""
        ii = np.array([0.5, -0.6, 0.05, -0.1])
        cfg = CatFEConfig(top_k_pairs=2, min_interaction_information=0.0, select_on="absolute")
        out = _select_top_k_pairs(
            ii_arr=ii,
            pairs_a=np.arange(4),
            pairs_b=np.arange(4),
            cfg=cfg, n_samples=1000,
        )
        assert set(out.tolist()) == {0, 1}

    def test_no_eligible_returns_empty(self):
        """No eligible returns empty."""
        ii = np.array([-0.5, -0.3])
        cfg = CatFEConfig(top_k_pairs=2, min_interaction_information=0.1)
        out = _select_top_k_pairs(
            ii_arr=ii,
            pairs_a=np.arange(2),
            pairs_b=np.arange(2),
            cfg=cfg, n_samples=1000,
        )
        assert out.size == 0

    def test_unknown_select_on_raises(self):
        """Unknown select on raises."""
        ii = np.array([0.5, 0.1])
        cfg = CatFEConfig()
        # Bypass __post_init__: mutate after construction
        object.__setattr__(cfg, "select_on", "bogus")
        with pytest.raises(ValueError, match="select_on"):
            _select_top_k_pairs(
                ii_arr=ii, pairs_a=np.arange(2), pairs_b=np.arange(2),
                cfg=cfg, n_samples=1000,
            )

    def test_argpartition_path_when_many_eligible(self):
        """Argpartition path when many eligible."""
        ii = np.linspace(0.01, 0.5, 20)
        cfg = CatFEConfig(top_k_pairs=5, min_interaction_information=0.0)
        out = _select_top_k_pairs(
            ii_arr=ii, pairs_a=np.arange(20), pairs_b=np.arange(20),
            cfg=cfg, n_samples=1000,
        )
        assert len(out) == 5
        # Top 5 should be the largest values (indices 15..19)
        assert set(out.tolist()) == {15, 16, 17, 18, 19}


class TestEntropyForMode:
    """Test Entropy For Mode."""
    def test_plain_plugin(self):
        """Plain plugin."""
        freqs = np.array([0.5, 0.5])
        h = _entropy_for_mode(freqs, n_samples=100, use_mm=False)
        assert h == pytest.approx(np.log(2), abs=1e-6)

    def test_miller_madow_path(self):
        """Miller madow path."""
        freqs = np.array([0.5, 0.5])
        h_pl = _entropy_for_mode(freqs, n_samples=100, use_mm=False)
        h_mm = _entropy_for_mode(freqs, n_samples=100, use_mm=True)
        # MM applies bias correction, should produce different value
        assert h_mm != h_pl

    def test_kt_smoothing_path(self):
        """Kt smoothing path."""
        freqs = np.array([0.5, 0.5])
        h_kt = _entropy_for_mode(freqs, n_samples=100, use_mm=False, use_kt=True)
        # KT smoothing finite output
        assert np.isfinite(h_kt)
        assert h_kt > 0


class TestShouldApplyMM:
    """Test group: the Miller-Madow bias-correction gate."""
    def test_folklore_gate_threshold(self):
        """Folklore gate threshold."""
        # Joint cardinality / n > 0.05 -> True
        assert _should_apply_mm_for_pair(10, 10, 2, 100) is True
        # Below threshold -> False
        assert _should_apply_mm_for_pair(2, 2, 2, 10000) is False

    def test_analytical_gate(self):
        """Analytical gate."""
        # (a-1)(b-1)(c-1) >= 6*sqrt(n)
        # Large cardinality, small n -> True
        assert _should_apply_mm_for_pair_analytical(20, 20, 5, 100) is True
        # Small cardinality, large n -> False
        assert _should_apply_mm_for_pair_analytical(2, 2, 2, 10000) is False


class TestApplyFwerCorrection:
    """Test Apply Fwer Correction."""
    def test_none_passthrough(self):
        """None passthrough."""
        conf = {(0, 1): 0.99, (0, 2): 0.5}
        cfg = CatFEConfig(fwer_correction="none")
        out = _apply_fwer_correction(conf, cfg, n_search_pairs=10)
        assert out == conf

    def test_empty_passthrough(self):
        """Empty passthrough."""
        cfg = CatFEConfig(fwer_correction="bonferroni")
        out = _apply_fwer_correction({}, cfg, n_search_pairs=10)
        assert out == {}

    def test_bonferroni_inflates_p(self):
        """Bonferroni inflates p."""
        # Confidence 0.99 -> p=0.01. With m=10 -> corrected p=0.1 -> conf=0.9
        cfg = CatFEConfig(fwer_correction="bonferroni")
        out = _apply_fwer_correction({(0, 1): 0.99}, cfg, n_search_pairs=10)
        assert out[(0, 1)] == pytest.approx(0.9, abs=1e-6)

    def test_bonferroni_clamps_at_1(self):
        """Bonferroni clamps at 1."""
        # Confidence 0.5 -> p=0.5; with m=10 -> p*m=5.0 clamps to 1 -> conf=0
        cfg = CatFEConfig(fwer_correction="bonferroni")
        out = _apply_fwer_correction({(0, 1): 0.5}, cfg, n_search_pairs=10)
        assert out[(0, 1)] == pytest.approx(0.0)

    def test_bh_fdr(self):
        """Bh fdr."""
        # Two ordered p-values; BH adjusts step-up
        cfg = CatFEConfig(fwer_correction="bh_fdr")
        out = _apply_fwer_correction(
            {(0, 1): 0.99, (0, 2): 0.95, (0, 3): 0.5},
            cfg, n_search_pairs=5,
        )
        # All three keys present, valid confidences
        assert set(out.keys()) == {(0, 1), (0, 2), (0, 3)}
        for v in out.values():
            assert 0.0 <= v <= 1.0

    def test_westfall_young_fallback(self):
        """Westfall young fallback."""
        cfg = CatFEConfig(fwer_correction="westfall_young")
        out = _apply_fwer_correction({(0, 1): 0.95}, cfg, n_search_pairs=4)
        assert (0, 1) in out

    def test_unknown_correction_raises(self):
        """Unknown correction raises."""
        cfg = CatFEConfig()
        object.__setattr__(cfg, "fwer_correction", "bogus")
        with pytest.raises(ValueError, match="fwer_correction"):
            _apply_fwer_correction({(0, 1): 0.9}, cfg, n_search_pairs=4)


class TestPermKernelDispatch:
    """Test Perm Kernel Dispatch."""
    def test_backend_cpu_returns_false(self):
        """Backend cpu returns false."""
        # backend="cpu" must never reach GPU
        assert _perm_kernel_dispatch_use_gpu(n_samples=10_000_000, n_perms=100, backend="cpu") is False

    def test_backend_auto_small_n_returns_false(self):
        """Backend auto small n returns false."""
        assert _perm_kernel_dispatch_use_gpu(n_samples=1000, n_perms=100, backend="auto") is False

    def test_backend_gpu_without_cupy(self):
        """Backend gpu without cupy."""
        # If cupy is missing, this returns False; if present, returns True.
        # Either way the call must not raise.
        out = _perm_kernel_dispatch_use_gpu(n_samples=100, n_perms=10, backend="gpu")
        assert out in (True, False)


class TestCatFEAutoBackendWorkGate:
    """Regression for the 2026-07-16 fix: ``cfg.backend="auto"`` used to require BOTH
    ``n_cols_eff >= 200`` AND ``n_samples >= 500_000`` as two INDEPENDENT thresholds, so a wide-but-
    under-500k-row fit could never qualify for GPU no matter how many columns it had. Fixed to a
    work-PRODUCT check against the SAME combined magnitude (200*500_000) plus a p-floor of 64."""

    def test_old_rule_shape_still_qualifies(self):
        """Old rule shape still qualifies."""
        # The exact old-rule boundary (200 cols, 500_000 rows) must still qualify under the new rule
        # (work = 100M, exactly at the floor) -- the fix must never NARROW what already qualified.
        assert _cat_fe_auto_wants_gpu(n_samples=500_000, n_cols_eff=200) is True

    def test_wide_under_500k_rows_now_qualifies(self):
        """Wide under 500k rows now qualifies."""
        # The bug this fixes: n_cols_eff=1000 far exceeds the old col floor, and n_samples=200_000 was
        # PREVIOUSLY DISQUALIFYING purely for being under 500_000, despite work=200M >> the 100M floor.
        assert _cat_fe_auto_wants_gpu(n_samples=200_000, n_cols_eff=1000) is True

    def test_tall_under_200_cols_now_qualifies(self):
        """Tall under 200 cols now qualifies."""
        # Symmetric case: n_cols_eff=100 was PREVIOUSLY DISQUALIFYING purely for being under 200,
        # despite n_samples=2_000_000 giving work=200M >> the 100M floor.
        assert _cat_fe_auto_wants_gpu(n_samples=2_000_000, n_cols_eff=100) is True

    def test_degenerate_tiny_p_declines_despite_huge_work(self):
        """Degenerate tiny p declines despite huge work."""
        # A huge-n/tiny-p shape clears the work product but has no real column-batching parallelism
        # for the GPU kernel to amortize over -- the p-floor must still decline it.
        assert _cat_fe_auto_wants_gpu(n_samples=1_000_000_000, n_cols_eff=1) is False

    def test_genuinely_small_shape_declines(self):
        """Genuinely small shape declines."""
        assert _cat_fe_auto_wants_gpu(n_samples=1000, n_cols_eff=10) is False

    def test_wellbore_100k_shape_still_declines(self):
        """Wellbore 100k shape still declines."""
        # Documented in the fix's docstring: this specific production shape sits under the 100M floor
        # (work ~49.7M) even after the fix -- pinned here so a future recalibration is a deliberate,
        # visible change rather than an accidental side effect.
        assert _cat_fe_auto_wants_gpu(n_samples=99_401, n_cols_eff=500) is False


# ---------------------------------------------------------------------------
# 2. Direct test of _compute_pair_ii_mm
# ---------------------------------------------------------------------------


class TestComputePairIIMM:
    """Test group: pairwise interaction-information with Miller-Madow correction."""
    def test_runs_with_mm_disabled(self, xor_small):
        """Direct MM-disabled path: should return finite II close to ln(2)
        for the XOR pair."""
        from mlframe.feature_selection.filters.info_theory import entropy
        h_y = entropy(xor_small["freqs_y"])
        ii = _compute_pair_ii_mm(
            factors_data=xor_small["data"],
            idx_a=0, idx_b=1,
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            h_y=h_y,
            use_mm=False,
            dtype=np.int32,
        )
        assert np.isfinite(ii)
        assert ii > 0.3  # strong synergy

    def test_runs_with_mm_enabled(self, xor_small):
        """Runs with mm enabled."""
        from mlframe.feature_selection.filters.info_theory import entropy
        h_y = entropy(xor_small["freqs_y"])
        ii = _compute_pair_ii_mm(
            factors_data=xor_small["data"],
            idx_a=0, idx_b=1,
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            h_y=h_y,
            use_mm=True,
            dtype=np.int32,
        )
        assert np.isfinite(ii)


# ---------------------------------------------------------------------------
# 3. Maybe-rerank-with-MM branches
# ---------------------------------------------------------------------------


class TestMaybeRerankWithMM:
    """Test group: optional Miller-Madow reranking of selected pairs."""
    def _make_inputs(self, xor):
        """Make inputs."""
        candidate_idxs = np.array([0, 1, 2, 3], dtype=np.int64)
        marginal_mi = _marginal_screen_njit(
            factors_data=xor["data"],
            candidate_idxs=candidate_idxs,
            nbins=xor["nbins"],
            classes_y=xor["classes_y"],
            freqs_y=xor["freqs_y"],
            dtype=np.int32,
        )
        marginal_full = np.full(xor["data"].shape[1], np.nan)
        for k, idx in enumerate(candidate_idxs):
            marginal_full[int(idx)] = marginal_mi[k]
        pairs_a = np.array([0, 0, 2], dtype=np.int64)
        pairs_b = np.array([1, 2, 3], dtype=np.int64)
        _, ii_arr, _ = _pair_search_kernel_njit(
            factors_data=xor["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            marginal_mi=marginal_full,
            nbins=xor["nbins"],
            classes_y=xor["classes_y"],
            freqs_y=xor["freqs_y"],
            dtype=np.int32,
        )
        return pairs_a, pairs_b, ii_arr, marginal_full

    def test_disabled_returns_unchanged(self, xor_small):
        """Disabled returns unchanged."""
        pairs_a, pairs_b, ii_arr, _ = self._make_inputs(xor_small)
        selected = np.array([0, 1], dtype=np.int64)
        cfg = CatFEConfig(use_miller_madow=False)
        ii_out, sel_out = _maybe_rerank_with_mm(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected, ii_arr=ii_arr,
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        np.testing.assert_array_equal(ii_out, ii_arr)
        np.testing.assert_array_equal(sel_out, selected)

    def test_empty_selected_returns_unchanged(self, xor_small):
        """Empty selected returns unchanged."""
        pairs_a, pairs_b, ii_arr, _ = self._make_inputs(xor_small)
        cfg = CatFEConfig(use_miller_madow=True)
        selected = np.array([], dtype=np.int64)
        ii_out, sel_out = _maybe_rerank_with_mm(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected, ii_arr=ii_arr,
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        assert sel_out.size == 0

    def test_force_true_applies_mm(self, xor_small):
        """Force true applies mm."""
        pairs_a, pairs_b, ii_arr, _ = self._make_inputs(xor_small)
        selected = np.array([0, 1, 2], dtype=np.int64)
        cfg = CatFEConfig(use_miller_madow=True)
        ii_out, sel_out = _maybe_rerank_with_mm(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected, ii_arr=ii_arr.copy(),
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        # Some IIs should be re-scored
        assert np.isfinite(ii_out).all()
        assert sel_out.size == selected.size

    def test_auto_gate_no_pair_fires(self, xor_small):
        """When use_miller_madow=None and no pair satisfies the analytical
        threshold (all small-cardinality on n=400), return inputs unchanged."""
        pairs_a, pairs_b, ii_arr, _ = self._make_inputs(xor_small)
        selected = np.array([0, 1], dtype=np.int64)
        cfg = CatFEConfig(use_miller_madow=None)
        ii_out, sel_out = _maybe_rerank_with_mm(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected, ii_arr=ii_arr,
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        # Should be no-op since binary cards don't trigger the gate
        np.testing.assert_array_equal(ii_out, ii_arr)

    @pytest.mark.parametrize("select_on", ["synergy", "redundancy", "absolute"])
    def test_select_on_resorts(self, xor_small, select_on):
        """Select on resorts."""
        pairs_a, pairs_b, ii_arr, _ = self._make_inputs(xor_small)
        selected = np.array([0, 1, 2], dtype=np.int64)
        cfg = CatFEConfig(use_miller_madow=True, select_on=select_on)
        ii_out, sel_out = _maybe_rerank_with_mm(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected, ii_arr=ii_arr.copy(),
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        assert sel_out.size == selected.size


# ---------------------------------------------------------------------------
# 4. Confirm-pairs-via-permutation early-exit
# ---------------------------------------------------------------------------


class TestConfirmPermutation:
    """Test Confirm Permutation."""
    def test_no_perms_returns_empty_confidence(self, xor_small):
        """No perms returns empty confidence."""
        cfg = CatFEConfig(full_npermutations=0)
        selected = np.array([0, 1], dtype=np.int64)
        ii = np.array([0.5, 0.3])
        out_sel, conf = _confirm_pairs_via_permutation(
            factors_data=xor_small["data"],
            pairs_a=np.array([0, 0], dtype=np.int64),
            pairs_b=np.array([1, 2], dtype=np.int64),
            selected_idx=selected, ii_arr=ii,
            nbins=xor_small["nbins"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            cfg=cfg, n_search_pairs=10,
            dtype=np.int32, verbose=0,
        )
        np.testing.assert_array_equal(out_sel, selected)
        assert conf == {}

    def test_empty_selected_returns_empty(self, xor_small):
        """Empty selected returns empty."""
        cfg = CatFEConfig(full_npermutations=5)
        out_sel, conf = _confirm_pairs_via_permutation(
            factors_data=xor_small["data"],
            pairs_a=np.array([0, 0], dtype=np.int64),
            pairs_b=np.array([1, 2], dtype=np.int64),
            selected_idx=np.array([], dtype=np.int64),
            ii_arr=np.array([], dtype=np.float64),
            nbins=xor_small["nbins"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            cfg=cfg, n_search_pairs=10,
            dtype=np.int32, verbose=0,
        )
        assert out_sel.size == 0
        assert conf == {}


# ---------------------------------------------------------------------------
# 5. K-fold stability filter direct tests
# ---------------------------------------------------------------------------


class TestKfoldStability:
    """Test Kfold Stability."""
    def test_disabled_passthrough(self, xor_small):
        """Disabled passthrough."""
        cfg = CatFEConfig(n_folds_stability=0)
        selected = np.array([0], dtype=np.int64)
        out_sel, fold_d = _kfold_stability_filter(
            factors_data=xor_small["data"],
            pairs_a=np.array([0], dtype=np.int64),
            pairs_b=np.array([1], dtype=np.int64),
            selected_idx=selected,
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        np.testing.assert_array_equal(out_sel, selected)
        assert fold_d == {}

    def test_empty_selected_passthrough(self, xor_small):
        """Empty selected passthrough."""
        cfg = CatFEConfig(n_folds_stability=2)
        out_sel, fold_d = _kfold_stability_filter(
            factors_data=xor_small["data"],
            pairs_a=np.array([0], dtype=np.int64),
            pairs_b=np.array([1], dtype=np.int64),
            selected_idx=np.array([], dtype=np.int64),
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        assert out_sel.size == 0

    def test_xor_survives_stability(self, xor_small):
        """XOR signal should clear K-fold stability."""
        cfg = CatFEConfig(
            n_folds_stability=2,
            min_fold_prevalence=0.5,
            min_interaction_information=0.05,
            min_n_samples=50,
        )
        selected = np.array([0], dtype=np.int64)
        out_sel, fold_d = _kfold_stability_filter(
            factors_data=xor_small["data"],
            pairs_a=np.array([0], dtype=np.int64),
            pairs_b=np.array([1], dtype=np.int64),
            selected_idx=selected,
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        # XOR is reliably synergistic; should survive
        assert out_sel.size == 1
        assert (0, 1) in fold_d


# ---------------------------------------------------------------------------
# 6. Anti-redundancy direct tests
# ---------------------------------------------------------------------------


class TestAntiRedundancy:
    """Test Anti Redundancy."""
    def test_beta_zero_passthrough(self, xor_small):
        """Beta zero passthrough."""
        cfg = CatFEConfig(anti_redundancy_beta=0.0)
        selected = np.array([0], dtype=np.int64)
        ii = np.array([0.5])
        ii_out, sel_out = _anti_redundancy_rerank(
            factors_data=xor_small["data"],
            pairs_a=np.array([0], dtype=np.int64),
            pairs_b=np.array([1], dtype=np.int64),
            selected_idx=selected, ii_arr=ii,
            nbins=xor_small["nbins"],
            selected_so_far=[2],
            classes_y=xor_small["classes_y"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        np.testing.assert_array_equal(ii_out, ii)
        np.testing.assert_array_equal(sel_out, selected)

    def test_empty_selected_so_far_passthrough(self, xor_small):
        """Empty selected so far passthrough."""
        cfg = CatFEConfig(anti_redundancy_beta=0.5)
        selected = np.array([0], dtype=np.int64)
        ii = np.array([0.5])
        ii_out, sel_out = _anti_redundancy_rerank(
            factors_data=xor_small["data"],
            pairs_a=np.array([0], dtype=np.int64),
            pairs_b=np.array([1], dtype=np.int64),
            selected_idx=selected, ii_arr=ii,
            nbins=xor_small["nbins"],
            selected_so_far=[],
            classes_y=xor_small["classes_y"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        np.testing.assert_array_equal(ii_out, ii)

    @pytest.mark.parametrize("select_on", ["synergy", "redundancy", "absolute"])
    def test_select_on_variants(self, xor_small, select_on):
        """Select on variants."""
        cfg = CatFEConfig(anti_redundancy_beta=0.3, select_on=select_on)
        # Two pairs to give re-ranking something to do
        marginal_mi = np.zeros(xor_small["data"].shape[1])
        pairs_a = np.array([0, 0], dtype=np.int64)
        pairs_b = np.array([1, 2], dtype=np.int64)
        _, ii_arr, _ = _pair_search_kernel_njit(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            marginal_mi=marginal_mi,
            nbins=xor_small["nbins"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            dtype=np.int32,
        )
        selected = np.array([0, 1], dtype=np.int64)
        ii_out, sel_out = _anti_redundancy_rerank(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            selected_idx=selected, ii_arr=ii_arr.copy(),
            nbins=xor_small["nbins"],
            selected_so_far=[3],  # one already-selected feature
            classes_y=xor_small["classes_y"],
            cfg=cfg, dtype=np.int32, verbose=0,
        )
        assert sel_out.size == selected.size


# ---------------------------------------------------------------------------
# 7. Greedy k-way expansion direct
# ---------------------------------------------------------------------------


class TestGreedyExpand:
    """Test Greedy Expand."""
    def test_no_extension_returns_none(self, xor_small):
        """With max_kway_order=2 we don't extend beyond pairs - returns None."""
        marginal_mi = np.zeros(xor_small["data"].shape[1])
        out = _greedy_expand_one_seed(
            factors_data=xor_small["data"],
            seed_indices=(0, 1),
            candidate_pool=np.array([2, 3], dtype=np.int64),
            nbins=xor_small["nbins"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            marginal_mi=marginal_mi,
            max_combined_nbins=200,
            max_kway_order=2,
            min_inc_ii=-100.0,
            dtype=np.int32,
        )
        assert out is None


# ---------------------------------------------------------------------------
# 8. Pair-search weighted kernel direct
# ---------------------------------------------------------------------------


class TestPairSearchWeighted:
    """Test Pair Search Weighted."""
    def test_uniform_weights_matches_unweighted(self, xor_small):
        """Uniform weights matches unweighted."""
        candidate_idxs = np.array([0, 1, 2, 3], dtype=np.int64)
        marginal_mi = _marginal_screen_njit(
            factors_data=xor_small["data"],
            candidate_idxs=candidate_idxs,
            nbins=xor_small["nbins"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            dtype=np.int32,
        )
        marginal_full = np.full(xor_small["data"].shape[1], np.nan)
        for k, idx in enumerate(candidate_idxs):
            marginal_full[int(idx)] = marginal_mi[k]

        pairs_a = np.array([0], dtype=np.int64)
        pairs_b = np.array([1], dtype=np.int64)
        weights = np.ones(xor_small["data"].shape[0], dtype=np.float64)
        joint_w, ii_w, n_uniq_w = _pair_search_kernel_weighted_njit(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            marginal_mi=marginal_full,
            nbins=xor_small["nbins"],
            classes_y=xor_small["classes_y"],
            weights=weights,
            dtype=np.int32,
        )
        joint_u, ii_u, _ = _pair_search_kernel_njit(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            marginal_mi=marginal_full,
            nbins=xor_small["nbins"],
            classes_y=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            dtype=np.int32,
        )
        # Weighted with uniform weights = unweighted (up to fp precision)
        np.testing.assert_allclose(joint_w[0], joint_u[0], rtol=1e-6)
        np.testing.assert_allclose(ii_w[0], ii_u[0], rtol=1e-6)

    def test_nonuniform_weights_runs(self, xor_small):
        """Nonuniform weights runs."""
        marginal_full = np.zeros(xor_small["data"].shape[1])
        pairs_a = np.array([0], dtype=np.int64)
        pairs_b = np.array([1], dtype=np.int64)
        rng = np.random.default_rng(3)
        weights = rng.random(xor_small["data"].shape[0]) + 0.1
        joint_w, ii_w, n_uniq_w = _pair_search_kernel_weighted_njit(
            factors_data=xor_small["data"],
            pairs_a=pairs_a, pairs_b=pairs_b,
            marginal_mi=marginal_full,
            nbins=xor_small["nbins"],
            classes_y=xor_small["classes_y"],
            weights=weights,
            dtype=np.int32,
        )
        assert np.isfinite(joint_w[0])
        assert n_uniq_w[0] >= 1


# ---------------------------------------------------------------------------
# 9. Build-factorize-lookup direct branches
# ---------------------------------------------------------------------------


class TestBuildFactorizeLookup:
    """Test Build Factorize Lookup."""
    def _make_pair_data(self):
        """Make pair data."""
        # 2x2 cartesian product with renumbering
        data = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]],
            dtype=np.int32,
        )
        classes_pair_post = np.array([0, 1, 2, 3, 0, 3], dtype=np.int32)
        return data, classes_pair_post

    def test_clip_strategy(self):
        """Clip strategy."""
        data, classes = self._make_pair_data()
        lookup, n_eff = _build_factorize_lookup(
            factors_data=data, idx_a=0, idx_b=1, nbins_a=2, nbins_b=2,
            classes_pair_post=classes, unknown_strategy="clip",
        )
        assert lookup.shape == (4,)
        assert n_eff == 4

    def test_sentinel_strategy_no_unseen(self):
        """Sentinel strategy no unseen."""
        data, classes = self._make_pair_data()
        lookup, n_eff = _build_factorize_lookup(
            factors_data=data, idx_a=0, idx_b=1, nbins_a=2, nbins_b=2,
            classes_pair_post=classes, unknown_strategy="sentinel",
        )
        assert lookup.shape == (4,)
        # Full table seen -> n_eff stays at 4
        assert n_eff == 4

    def test_clip_with_unseen_cells(self):
        """Clip with unseen cells."""
        # 2x3 table but only (0,0) and (1,1) seen -> 4 unseen
        data = np.array([[0, 0], [1, 1]], dtype=np.int32)
        classes = np.array([0, 1], dtype=np.int32)
        lookup, n_eff = _build_factorize_lookup(
            factors_data=data, idx_a=0, idx_b=1, nbins_a=2, nbins_b=3,
            classes_pair_post=classes, unknown_strategy="clip",
        )
        # All -1 entries clipped to seen_max=1
        assert (lookup >= 0).all()

    def test_sentinel_with_unseen_cells(self):
        """Sentinel with unseen cells."""
        data = np.array([[0, 0], [1, 1]], dtype=np.int32)
        classes = np.array([0, 1], dtype=np.int32)
        lookup, n_eff = _build_factorize_lookup(
            factors_data=data, idx_a=0, idx_b=1, nbins_a=2, nbins_b=3,
            classes_pair_post=classes, unknown_strategy="sentinel",
        )
        # Unseen -> seen_max + 1 = 2 -> n_eff = 3
        assert n_eff == 3
        assert (lookup == 2).sum() > 0

    def test_raise_strategy_leaves_negative_one(self):
        """Raise strategy leaves negative one."""
        data = np.array([[0, 0], [1, 1]], dtype=np.int32)
        classes = np.array([0, 1], dtype=np.int32)
        lookup, _ = _build_factorize_lookup(
            factors_data=data, idx_a=0, idx_b=1, nbins_a=2, nbins_b=3,
            classes_pair_post=classes, unknown_strategy="raise",
        )
        assert (lookup == -1).sum() > 0

    def test_unknown_strategy_raises(self):
        """Unknown strategy raises."""
        data = np.array([[0, 0], [1, 1]], dtype=np.int32)
        classes = np.array([0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="unknown_strategy"):
            _build_factorize_lookup(
                factors_data=data, idx_a=0, idx_b=1, nbins_a=2, nbins_b=3,
                classes_pair_post=classes, unknown_strategy="bogus",
            )


# ---------------------------------------------------------------------------
# 10. Orchestrator knob-coverage tests (one per major branch)
# ---------------------------------------------------------------------------


class TestOrchestratorKnobs:
    """Test Orchestrator Knobs."""
    @pytest.mark.fast
    def test_baseline_xor_runs(self, xor_small):
        """Baseline xor runs."""
        cfg = CatFEConfig(
            enable=True, top_k_pairs=4,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
        )
        data_out, _, _, state = _run(xor_small, cfg)
        assert data_out.shape[1] >= xor_small["data"].shape[1]
        assert len(state.recipes) >= 1

    def test_disabled_via_min_n(self, xor_small):
        """Disabled via min n."""
        cfg = CatFEConfig(min_n_samples=10_000)
        data_out, cols_out, _, state = _run(xor_small, cfg)
        assert data_out is xor_small["data"]
        assert state.recipes == []

    def test_select_on_redundancy(self, xor_small):
        """Select on redundancy."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=-0.05,
            full_npermutations=0,
            min_n_samples=50,
            select_on="redundancy",
        )
        _, _, _, state = _run(xor_small, cfg)
        # Just verify no exception; state may or may not have recipes
        assert isinstance(state.recipes, list)

    def test_select_on_absolute(self, xor_small):
        """Select on absolute."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.01,
            full_npermutations=0,
            min_n_samples=50,
            select_on="absolute",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_use_miller_madow_true(self, xor_small):
        """Use miller madow true."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            use_miller_madow=True,
        )
        _, _, _, state = _run(xor_small, cfg)
        # XOR should still survive MM correction
        assert len(state.recipes) >= 1

    def test_use_miller_madow_false(self, xor_small):
        """Use miller madow false."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            use_miller_madow=False,
        )
        _, _, _, state = _run(xor_small, cfg)
        assert len(state.recipes) >= 1

    def test_use_miller_madow_auto(self, xor_small):
        """Use miller madow auto."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            use_miller_madow=None,
        )
        _, _, _, state = _run(xor_small, cfg)
        assert len(state.recipes) >= 1

    def test_use_kt_smoothing(self, xor_small):
        """Use kt smoothing."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            use_miller_madow=True,
            use_kt_smoothing=True,
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_permutation_with_full_n_perms(self, xor_small):
        """Permutation with full n perms."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=3,
            min_n_samples=50,
            perm_budget_strategy="fixed",
            backend="cpu",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_permutation_conditional_null(self, xor_small):
        """Permutation conditional null."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=2,
            min_n_samples=50,
            permutation_null="conditional",
            perm_budget_strategy="fixed",
            backend="cpu",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_permutation_conditional_full_ipf(self, xor_small):
        """Permutation conditional full ipf."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=2,
            min_n_samples=50,
            permutation_null="conditional",
            enable_full_conditional_perm=True,
            perm_budget_strategy="fixed",
            backend="cpu",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_perm_budget_bandit_ucb1(self, xor_small):
        """Perm budget bandit ucb1."""
        cfg = CatFEConfig(
            top_k_pairs=3,
            min_interaction_information=0.05,
            full_npermutations=4,
            min_n_samples=50,
            perm_budget_strategy="bandit_ucb1",
            backend="cpu",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    @pytest.mark.parametrize("correction", ["bonferroni", "bh_fdr"])
    def test_fwer_correction_variants(self, xor_small, correction):
        """Fwer correction variants."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=3,
            min_n_samples=50,
            fwer_correction=correction,
            perm_budget_strategy="fixed",
            backend="cpu",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_fwer_westfall_young(self, xor_small):
        """Fwer westfall young."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=3,
            min_n_samples=50,
            fwer_correction="westfall_young",
            perm_budget_strategy="fixed",
            backend="cpu",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_n_folds_stability(self, xor_small):
        """N folds stability."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            n_folds_stability=2,
            min_fold_prevalence=0.5,
        )
        _, _, _, state = _run(xor_small, cfg)
        # state.ii_stability is always populated when n_folds_stability>0
        assert isinstance(state.ii_stability, dict)

    def test_anti_redundancy_beta_with_selected_so_far(self, xor_small):
        """Anti redundancy beta with selected so far."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=-0.5,  # generous floor
            full_npermutations=0,
            min_n_samples=50,
            anti_redundancy_beta=0.5,
        )
        # selected_so_far drives the anti-redundancy branch
        _, _, _, state = _run(xor_small, cfg, selected_so_far=[2, 3])
        assert isinstance(state.recipes, list)

    def test_max_kway_order_3(self, xor_small):
        """Max kway order 3."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            max_kway_order=3,
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_max_kway_with_refine(self, xor_small):
        """Max kway with refine."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            max_kway_order=3,
            refine_passes=1,
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_marginal_floor_prunes_aggressively(self, xor_small):
        """Marginal floor prunes aggressively."""
        # All XOR marginals are ~0 so a floor of 0.5 prunes everything
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            marginal_floor=0.5,
        )
        data_out, _, _, state = _run(xor_small, cfg)
        # No pair candidates left after floor
        assert data_out is xor_small["data"]
        assert state.recipes == []

    def test_max_combined_nbins_too_tight(self, xor_small):
        """Max combined nbins too tight."""
        # Force a tight cap so the 2x2 XOR pair is rejected
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            max_combined_nbins=4,  # min allowed; 2*2 = 4 is at the boundary but rejected (> not >=)
        )
        # Cardinality budget allows nb_prod <= max_combined; 4 <= 4 stays in
        data_out, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_min_ii_unreachable_returns_unchanged(self, xor_small):
        """Min ii unreachable returns unchanged."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=10.0,  # impossibly high
            full_npermutations=0,
            min_n_samples=50,
        )
        data_out, cols_out, _, state = _run(xor_small, cfg)
        assert data_out.shape == xor_small["data"].shape
        assert cols_out == xor_small["cols"]

    def test_bootstrap_ci(self, xor_small):
        """Bootstrap ci."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            bootstrap_ci_n_replicates=2,
            bootstrap_sample_frac=0.5,
            bootstrap_ci_alpha=0.10,
        )
        _, _, _, state = _run(xor_small, cfg)
        # Bootstrap CI block runs whether or not the pair survives
        assert isinstance(state.recipes, list)

    def test_target_encoding_emit(self, xor_small):
        """Target encoding emit."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            emit_target_encoding=True,
            target_encoding_oof_folds=2,
            target_encoding_smoothing=5.0,
        )
        _, _, _, state = _run(xor_small, cfg)
        # TE recipes should be present alongside factorize ones if pair clears floor
        # __target_encoding__ diagnostic key signals the branch ran
        assert isinstance(state.recipes, list)

    def test_target_encoding_oof_zero(self, xor_small):
        """Target encoding oof zero."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            emit_target_encoding=True,
            target_encoding_oof_folds=0,  # naive path
            target_encoding_smoothing=5.0,
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_streaming_cache_first_fit(self, xor_small):
        """Streaming cache first fit."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            enable_streaming_cache=True,
        )
        # First fit: cache is empty
        _, _, _, state = _run(xor_small, cfg, streaming_cache=None)
        assert "col_signatures" in state.streaming_cache_out
        assert "marginal_mis" in state.streaming_cache_out

    def test_streaming_cache_warm_reuse(self, xor_small):
        """Streaming cache warm reuse."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            enable_streaming_cache=True,
            streaming_cache_kl_threshold=10.0,  # very lenient
        )
        # First fit -> warm cache
        _, _, _, state1 = _run(xor_small, cfg, streaming_cache=None)
        # Second fit -> reuse
        _, _, _, state2 = _run(xor_small, cfg, streaming_cache=state1.streaming_cache_out)
        assert isinstance(state2.recipes, list)

    def test_with_uniform_weights(self, xor_small):
        """With uniform weights."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            backend="cpu",
        )
        weights = np.ones(xor_small["data"].shape[0], dtype=np.float64)
        _, _, _, state = _run(xor_small, cfg, weights=weights)
        assert isinstance(state.recipes, list)

    def test_with_nonuniform_weights(self, xor_small):
        """With nonuniform weights."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            backend="cpu",
        )
        rng = np.random.default_rng(13)
        weights = rng.random(xor_small["data"].shape[0]) + 0.1
        _, _, _, state = _run(xor_small, cfg, weights=weights)
        assert isinstance(state.recipes, list)

    def test_unknown_strategy_sentinel(self, xor_small):
        """Unknown strategy sentinel."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            unknown_strategy="sentinel",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_unknown_strategy_raise(self, xor_small):
        """Unknown strategy raise."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            unknown_strategy="raise",
        )
        _, _, _, state = _run(xor_small, cfg)
        assert isinstance(state.recipes, list)

    def test_emit_diagnostics_off(self, xor_small):
        """Emit diagnostics off."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=0,
            min_n_samples=50,
            emit_diagnostics=False,
        )
        _, _, _, state = _run(xor_small, cfg)
        # With diagnostics off, dict stays empty
        assert state.diagnostics == {}

    def test_empty_target_indices_raises(self, xor_small):
        """Empty target indices raises."""
        cfg = CatFEConfig(min_n_samples=50)
        with pytest.raises(ValueError, match="empty target_indices"):
            run_cat_interaction_step(
                data=xor_small["data"],
                cols=xor_small["cols"],
                nbins=xor_small["nbins"],
                target_indices=np.array([], dtype=np.int64),
                classes_y=xor_small["classes_y"],
                classes_y_safe=xor_small["classes_y"],
                freqs_y=xor_small["freqs_y"],
                categorical_vars=xor_small["categorical_vars"],
                cfg=cfg, dtype=np.int32,
            )

    def test_only_one_eligible_column_returns_unchanged(self, xor_small):
        """Only one eligible column returns unchanged."""
        # categorical_vars list with only one valid column -> skip
        cfg = CatFEConfig(min_n_samples=50)
        data_out, cols_out, _, state = run_cat_interaction_step(
            data=xor_small["data"],
            cols=xor_small["cols"],
            nbins=xor_small["nbins"],
            target_indices=xor_small["target_indices"],
            classes_y=xor_small["classes_y"],
            classes_y_safe=xor_small["classes_y"],
            freqs_y=xor_small["freqs_y"],
            categorical_vars=[0],  # only 1 col
            cfg=cfg, dtype=np.int32,
        )
        assert data_out is xor_small["data"]
        assert state.recipes == []


# ---------------------------------------------------------------------------
# 11. Additional orchestrator paths combining multiple knobs
# ---------------------------------------------------------------------------


class TestOrchestratorCombined:
    """Test Orchestrator Combined."""
    @pytest.mark.slow
    def test_perm_with_mm_and_bandit(self, xor_medium):
        """Exercise MM + bandit UCB1 + bonferroni in one shot."""
        cfg = CatFEConfig(
            top_k_pairs=3,
            min_interaction_information=0.05,
            full_npermutations=5,
            min_n_samples=50,
            use_miller_madow=True,
            perm_budget_strategy="bandit_ucb1",
            fwer_correction="bonferroni",
            backend="cpu",
        )
        _, _, _, state = _run(xor_medium, cfg)
        assert isinstance(state.recipes, list)

    @pytest.mark.slow
    def test_perm_subsample_path(self, xor_medium):
        """``permutation_subsample`` exercises the subsample-prep branch."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.05,
            full_npermutations=3,
            min_n_samples=50,
            permutation_subsample=200,  # < n=500
            perm_budget_strategy="fixed",
            backend="cpu",
        )
        _, _, _, state = _run(xor_medium, cfg)
        assert isinstance(state.recipes, list)

    @pytest.mark.slow
    def test_kfold_plus_anti_redundancy(self, xor_medium):
        """Kfold plus anti redundancy."""
        cfg = CatFEConfig(
            top_k_pairs=2,
            min_interaction_information=0.01,
            full_npermutations=0,
            min_n_samples=50,
            n_folds_stability=2,
            min_fold_prevalence=0.5,
            anti_redundancy_beta=0.3,
        )
        _, _, _, state = _run(xor_medium, cfg, selected_so_far=[2])
        assert isinstance(state.recipes, list)

    @pytest.mark.slow
    def test_max_kway_3_with_refine(self, xor_medium):
        """Max kway 3 with refine."""
        cfg = CatFEConfig(
            top_k_pairs=3,
            min_interaction_information=-0.5,
            full_npermutations=0,
            min_n_samples=50,
            max_kway_order=3,
            refine_passes=1,
        )
        _, _, _, state = _run(xor_medium, cfg)
        assert isinstance(state.recipes, list)

    # !coverage-deferred: GPU-only branches (mi_direct_gpu_batched_pairs,
    # _count_nfailed_joint_indep_cupy) require a working CuPy install plus
    # a CUDA device. Tests skip-execution under the standard CI box; covered
    # via integration tests on the GPU runner.

    # !coverage-deferred: group-aware shuffle path (_group_aware_shuffle)
    # is currently unreachable from run_cat_interaction_step -- the
    # ``groups_col`` config knob is plumbed at the MRMR layer but never
    # surfaced into cat_interactions.py. Covered separately by test_group_aware.

    # !coverage-deferred: Westfall-Young full path
    # (_compute_westfall_young_corrected_p) requires
    # ``len(pairs_a) * n_samples * 4 < 500MB`` AND fwer_correction='westfall_young'
    # with significant pair counts; the small fixtures here trigger only the
    # fallback path through _apply_fwer_correction.
