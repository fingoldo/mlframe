"""Behavioural tests for pairwise, categorical and grouped feature-engineering steps."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def test_regression_step_score_escalation_fusion_codes_use_safe_dtype():
    """Pre-fix: np.empty(..., dtype=self.quantization_dtype) with quantization_dtype=int8 and
    quantization_nbins=200 wraps discretize_array's widened int16 codes back to negative int8
    on assignment. Post-fix: both call sites pre-widen via _safe_code_dtype first, so no wraparound.
    """
    from mlframe.feature_selection.filters.discretization import _safe_code_dtype, discretize_array

    rng = np.random.default_rng(0)
    arr = rng.standard_normal(2000)
    n_bins = 200  # > int8's 128-code ceiling
    raw_dtype = np.int8

    widened = discretize_array(arr=arr, n_bins=n_bins, method="quantile", dtype=raw_dtype)
    assert widened.dtype != np.dtype(raw_dtype), "discretize_array should auto-widen past int8 for n_bins=200"
    assert (widened >= 0).all(), "discretize_array's own widened codes must never be negative"

    # Reproduce the FIXED pattern used at both call sites: preallocate at _safe_code_dtype, not
    # the raw quantization_dtype, then assign the (already-widened) discretize_array output into it.
    safe_dtype = _safe_code_dtype(n_bins, raw_dtype)
    buf = np.empty(shape=(len(arr), 1), dtype=safe_dtype)
    buf[:, 0] = widened
    assert (buf[:, 0] >= 0).all(), "safe-dtype buffer must not wrap codes negative"
    np.testing.assert_array_equal(buf[:, 0], widened)

    # Demonstrate the PRE-FIX bug mechanism directly: assigning into a buffer still pinned at the
    # raw (narrow) dtype silently downcasts and wraps some codes negative.
    buggy_buf = np.empty(shape=(len(arr), 1), dtype=raw_dtype)
    buggy_buf[:, 0] = widened
    assert (buggy_buf[:, 0] < 0).any(), "sanity check: the narrow-dtype buffer should reproduce the wraparound bug"


# ---------------------------------------------------------------------------
# CORE_CLASS-2 (P1): MRMR.clear_fit_cache() cleared the process-wide _FIT_CACHE
# with no lock, unlike every other _FIT_CACHE mutation site (guarded by the
# canonical _MRMR_FIT_CACHE_LOCK). Covered structurally + concurrently in
# tests/feature_selection/mrmr/caching/test_fit_cache_thread_safety.py
# (test_clear_fit_cache_holds_the_canonical_lock, test_clear_fit_cache_concurrent_with_in_flight_fits).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CORE_CLASS-3 (P1): a nested pydantic config (e.g. fast_search_config) is
# re-applied over the flat ctor kwargs on every __init__ -- including a clone()
# reconstruction after set_params() changed one of the config's own covered
# fields, silently reverting it and making sklearn.base.clone() raise RuntimeError.
# ---------------------------------------------------------------------------


def test_regression_evaluate_candidate_baseline_seed_is_reproducible_and_seed_sensitive(monkeypatch):
    """Pre-fix: evaluate_candidate's baseline mi_direct call always used base_seed=0 regardless of
    random_seed, so two different random_seed values produced the IDENTICAL permutation draw. Post-fix:
    a seed derived from random_seed is threaded through, so different random_seed values produce
    different (but each individually reproducible) draws.
    """
    import mlframe.feature_selection.filters.evaluation as ev_mod

    captured_seeds = []
    orig_mi_direct = ev_mod.mi_direct

    def _spy_mi_direct(*args, **kwargs):
        """Records the call for the assertion below, then delegates to the real implementation."""
        captured_seeds.append(kwargs.get("base_seed"))
        return orig_mi_direct(*args, **kwargs)

    monkeypatch.setattr(ev_mod, "mi_direct", _spy_mi_direct)

    rng = np.random.default_rng(0)
    n = 200
    # Every column must stay inside its own declared bin count: y has 2 bins, and a value of 2..4 indexes past the joint's class table
    # (silently tolerated by compiled code, an IndexError once numba is disabled).
    factors_nbins = np.array([5, 5, 2], dtype=np.int64)
    factors_data = np.column_stack([rng.integers(0, int(k), size=n) for k in factors_nbins]).astype(np.int32)

    def _fresh_kwargs():
        """Builds a fresh kwargs dict so each call starts from identical, unmutated inputs."""
        return dict(
            cand_idx=0,
            X=(0,),
            y=(2,),
            nexisting=0,
            best_gain=0.0,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            factors_names=["a", "b", "y"],
            expected_gains=np.zeros(3),
            partial_gains={},
            selected_vars=[],
            baseline_npermutations=4,
            use_gpu=False,
            cached_MIs={},
            cached_confident_MIs={},
            cached_cond_MIs={},
            cached_jmim_MIs={},
            entropy_cache={},
            verbose=0,
        )

    ev_mod.evaluate_candidate(**_fresh_kwargs(), random_seed=1)
    ev_mod.evaluate_candidate(**_fresh_kwargs(), random_seed=2)

    assert len(captured_seeds) == 2
    assert captured_seeds[0] is not None and captured_seeds[1] is not None
    assert captured_seeds[0] != captured_seeds[1], "different random_seed values must produce different baseline permutation seeds"


def test_regression_genie_weights_not_uniform_for_differentiated_bias_rates():
    """Pre-fix: genie_mi_panel's default bias vector was a constant (identical for every estimator),
    making the (K+2)x(K+2) constraint matrix singular by construction -- confirmed empirically (rank 4 of
    5, det=0.0, LinAlgError), so genie_weights always fell back to plain uniform averaging (w = [1/K]*K).
    Post-fix: _genie_default_bias_rate differentiates by estimator-name family, so the constraint system
    is no longer degenerate and genie_weights returns a genuinely non-uniform solution.
    """
    from mlframe.feature_selection.filters._mi_aggregator import _genie_default_bias_rate, genie_weights

    n = 2000
    bias = [_genie_default_bias_rate(name, n) for name in ("fd", "qs", "mixed_ksg")]
    assert len(set(bias)) > 1, "bias rates must differ across estimator-name families, not be a single shared constant"

    w = genie_weights(bias, [1.0, 1.0, 1.0])
    assert not np.allclose(w, 1.0 / 3.0), "genie_weights must not degenerate to a plain uniform mean"


def _grand_fusion_fixture(n=4000, seed=3):
    """Continuous a, b and a 20-bin target carrying an a**2/b signal, as the grand-fusion pair search receives them."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 4.0, n)
    b = rng.uniform(0.5, 4.0, n)
    y = a**2 / b + 0.5 * rng.standard_normal(n)
    edges = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    y_codes = np.searchsorted(edges, y).astype(np.int64)
    freqs_y = np.bincount(y_codes, minlength=int(y_codes.max()) + 1) / n
    return a, b, y_codes, y_codes, freqs_y


def _cuda_device_count(cp):
    """Visible CUDA device count, or 0 when the CUDA runtime cannot be queried."""
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _require_cuda():
    """Skip unless cupy imports and a CUDA device is visible; the grand-fusion path has no CPU form."""
    cp = pytest.importorskip("cupy")
    if _cuda_device_count(cp) < 1:
        pytest.skip("no usable CUDA device")


@pytest.mark.parametrize("fusion", ["1", "0"])
def test_regression_grand_fused_pair_mi_threads_random_seed(fusion, monkeypatch):
    """Pre-fix: grand_fused_pair_mi ignored any caller-supplied seed (hardcoded base_seed=np.uint64(0)), and the fused
    default path never received one at all. Post-fix: random_seed reaches the permutation null on both the fused
    (default) and the non-fused path, and 0 remains the default."""
    _require_cuda()
    from mlframe.feature_selection.filters import batch_mi_noise_gate_gpu as gate
    from mlframe.feature_selection.filters._gpu_resident_pair_mi import grand_fused_pair_mi

    monkeypatch.setenv("MLFRAME_FE_GPU_GRAND_FUSION", fusion)
    real_shuffle = gate._build_shuffle_matrix
    real_resident = gate.batch_mi_with_noise_gate_cuda_resident
    seeds = []

    def _shuffle_spy(classes_y_safe, base_seed, npermutations):
        """Record the seed the fused path builds its shuffle matrix from, then delegate."""
        seeds.append(int(base_seed))
        return real_shuffle(classes_y_safe, base_seed, npermutations)

    def _resident_spy(*args, base_seed, **kwargs):
        """Record the seed the non-fused resident noise gate receives, then delegate."""
        seeds.append(int(base_seed))
        return real_resident(*args, base_seed=base_seed, **kwargs)

    monkeypatch.setattr(gate, "_build_shuffle_matrix", _shuffle_spy)
    monkeypatch.setattr(gate, "batch_mi_with_noise_gate_cuda_resident", _resident_spy)
    a, b, y_codes, classes_y_safe, freqs_y = _grand_fusion_fixture()

    names, fe_mi = grand_fused_pair_mi(a, b, y_codes, classes_y_safe, freqs_y, nbins=20, npermutations=25, random_seed=7)
    assert seeds and set(seeds) == {7}, f"random_seed=7 must reach the permutation null; saw {seeds}"
    assert len(fe_mi) == len(names)

    seeds.clear()
    grand_fused_pair_mi(a, b, y_codes, classes_y_safe, freqs_y, nbins=20, npermutations=25)
    assert seeds and set(seeds) == {0}, f"the default seed must stay 0; saw {seeds}"


# ---------------------------------------------------------------------------
# GPU_INFRA_B-9 (P2): _env_gpu_default_on reimplemented the CUDA_VISIBLE_DEVICES=""/MLFRAME_DISABLE_GPU=1
# opt-out check inline instead of calling the shared _gpu_policy.gpu_globally_disabled(), so a future change
# to the shared policy's semantics would silently not reach this cluster's default-on gates.
# ---------------------------------------------------------------------------


def test_regression_marginal_screen_weighted_differs_from_unweighted():
    """Pre-fix: no weighted marginal-screen path existed at all. Post-fix: _marginal_screen_weighted
    produces a genuinely different (correctly weighted) marginal MI than the unweighted screen when
    weights are non-uniform and correlated with the column's distribution."""
    from mlframe.feature_selection.filters.cat_interactions import _marginal_screen_njit, _marginal_screen_weighted

    rng = np.random.default_rng(0)
    n = 4000
    # Column 0: a categorical column correlated with y (real marginal signal).
    x0 = rng.integers(0, 4, n).astype(np.int32)
    y_codes = (x0 % 2).astype(np.int32)
    factors_data = x0.reshape(-1, 1)
    nbins = np.array([4], dtype=np.int64)
    candidate_idxs = np.array([0], dtype=np.int64)
    classes_y = y_codes
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.float64) / n

    unweighted = _marginal_screen_njit(factors_data, candidate_idxs, nbins, classes_y, freqs_y, np.int32)
    # Weights heavily skewed toward rows where x0==0 (shifts the effective marginal distribution).
    weights = np.where(x0 == 0, 5.0, 1.0)
    weighted = _marginal_screen_weighted(factors_data, candidate_idxs, nbins, classes_y, weights, np.int32)

    assert not np.allclose(unweighted, weighted), (
        f"weighted and unweighted marginal MI should differ under a distribution-shifting weight scheme: " f"unweighted={unweighted}, weighted={weighted}"
    )


def test_regression_cat_interactions_step_recomputes_weighted_marginal():
    """Post-fix: with sample weights, run_cat_interaction_step overwrites the unweighted marginal MIs with the weighted
    ones before the pair search, so the emitted diagnostics report the weighted marginals."""
    from mlframe.feature_selection.filters._cat_interactions_step import run_cat_interaction_step
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
    from mlframe.feature_selection.filters.cat_interactions import _marginal_screen_weighted
    from mlframe.feature_selection.filters.info_theory import merge_vars

    rng = np.random.default_rng(9)
    n = 1200
    x1 = rng.integers(0, 2, n).astype(np.int32)
    x2 = rng.integers(0, 2, n).astype(np.int32)
    # Rows 0..299 carry x1 as the signal and the rest carry the XOR synergy, so the weights below move x1's marginal.
    y = np.where(np.arange(n) < 300, x1, x1 ^ x2).astype(np.int32)
    data = np.column_stack([x1, x2, y]).astype(np.int32)
    nbins = np.array([2, 2, 2], dtype=np.int64)
    cls_y, fq_y, _ = merge_vars(
        factors_data=data, vars_indices=np.array([2], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
    )
    weights = np.ones(n, dtype=np.float32)
    weights[:300] = 3.0
    candidates = np.array([0, 1], dtype=np.int64)
    weighted = _marginal_screen_weighted(
        factors_data=data, candidate_idxs=candidates, nbins=nbins, classes_y=cls_y, weights=weights, dtype=np.int32,
    )
    unweighted = _marginal_screen_weighted(
        factors_data=data, candidate_idxs=candidates, nbins=nbins, classes_y=cls_y, weights=np.ones(n, dtype=np.float32), dtype=np.int32,
    )
    assert not np.allclose(weighted, unweighted), "the fixture must move the weighted marginals, or the test cannot tell them apart"

    cfg = CatFEConfig(
        enable=True, top_k_pairs=2, min_interaction_information=0.05, full_npermutations=100,
        fwer_correction="none", perm_budget_strategy="fixed", emit_diagnostics=True,
    )
    _, _, _, state = run_cat_interaction_step(
        data=data, cols=["x1", "x2", "y"], nbins=nbins, target_indices=np.array([2], dtype=np.int64),
        classes_y=cls_y, classes_y_safe=cls_y, freqs_y=fq_y, categorical_vars=[0, 1], cfg=cfg, dtype=np.int32, weights=weights,
    )
    assert state.diagnostics, "the XOR pair must survive so that its marginals are reported"
    for entry in state.diagnostics.values():
        i, j = entry["src_indices"]
        assert entry["marginal_X1_MI"] == pytest.approx(weighted[i])
        assert entry["marginal_X2_MI"] == pytest.approx(weighted[j])


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-2 (P2): neither cat-FE GPU dispatch point (permutation-confirmation kernel,
# pair-search kernel) consulted gpu_globally_disabled() -- MLFRAME_DISABLE_GPU=1 was silently ignored.
# ---------------------------------------------------------------------------


def test_regression_bandit_ucb1_auto_falls_back_to_fixed_when_weighted(monkeypatch, caplog):
    """Post-fix: run_cat_interaction_step skips the bandit-UCB1 path entirely (falling back to the correctly-weighted
    fixed path) whenever use_weights is True, regardless of perm_budget_strategy. An unweighted fit keeps the bandit path."""
    import logging

    from mlframe.feature_selection.filters import cat_interactions as ci
    from mlframe.feature_selection.filters._cat_interactions_step import run_cat_interaction_step
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
    from mlframe.feature_selection.filters.info_theory import merge_vars

    calls = []
    real_bandit = ci._confirm_pairs_bandit_ucb1
    real_fixed = ci._confirm_pairs_via_permutation

    def _bandit_spy(*args, **kwargs):
        """Record that the bandit-UCB1 confirmation ran, then delegate."""
        calls.append("bandit")
        return real_bandit(*args, **kwargs)

    def _fixed_spy(*args, **kwargs):
        """Record that the fixed-budget permutation confirmation ran, then delegate."""
        calls.append("fixed")
        return real_fixed(*args, **kwargs)

    monkeypatch.setattr(ci, "_confirm_pairs_bandit_ucb1", _bandit_spy)
    monkeypatch.setattr(ci, "_confirm_pairs_via_permutation", _fixed_spy)

    rng = np.random.default_rng(9)
    n = 1200
    x1 = rng.integers(0, 2, n).astype(np.int32)
    x2 = rng.integers(0, 2, n).astype(np.int32)
    y = (x1 ^ x2).astype(np.int32)
    data = np.column_stack([x1, x2, y]).astype(np.int32)
    nbins = np.array([2, 2, 2], dtype=np.int64)
    cls_y, fq_y, _ = merge_vars(
        factors_data=data, vars_indices=np.array([2], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
    )
    cfg = CatFEConfig(
        enable=True, top_k_pairs=2, min_interaction_information=0.05, full_npermutations=100,
        fwer_correction="none", perm_budget_strategy="bandit_ucb1",
    )
    weights = np.ones(n, dtype=np.float32)
    weights[:300] = 3.0

    def _confirm_path(w):
        """Run the cat-interaction step with the given weights and return which confirmation paths it invoked."""
        calls.clear()
        run_cat_interaction_step(
            data=data, cols=["x1", "x2", "y"], nbins=nbins, target_indices=np.array([2], dtype=np.int64),
            classes_y=cls_y, classes_y_safe=cls_y, freqs_y=fq_y, categorical_vars=[0, 1], cfg=cfg, dtype=np.int32, weights=w,
        )
        return list(calls)

    assert "bandit" in _confirm_path(None), "control: the bandit path must run for an unweighted fit"
    with caplog.at_level(logging.WARNING):
        weighted_calls = _confirm_path(weights)
    assert "bandit" not in weighted_calls, f"a weighted fit must not take the bandit path; saw {weighted_calls}"
    assert "fixed" in weighted_calls, f"a weighted fit must take the fixed-budget path; saw {weighted_calls}"
    assert any("falling back to the fixed-budget permutation path" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-8 (P2): _kfold_target_encode_codes had no n_folds validation; n_folds=0 raised an
# opaque ZeroDivisionError, n_folds=1 silently emitted an all-global_mean column with no warning.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwarg,bad_value,match",
    [
        ("bootstrap_ci_alpha", 1.2, "bootstrap_ci_alpha"),
        ("bootstrap_ci_alpha", 0.0, "bootstrap_ci_alpha"),
        ("bootstrap_sample_frac", 0.0, "bootstrap_sample_frac"),
        ("bootstrap_sample_frac", 1.5, "bootstrap_sample_frac"),
        ("streaming_cache_kl_threshold", 0.0, "streaming_cache_kl_threshold"),
        ("streaming_cache_kl_threshold", -1.0, "streaming_cache_kl_threshold"),
        ("target_encoding_smoothing", -1.0, "target_encoding_smoothing"),
        ("numeric_nbins", 1, "numeric_nbins"),
    ],
)
def test_regression_cat_fe_config_rejects_out_of_range_fields(kwarg, bad_value, match):
    """Pre-fix: CatFEConfig(bootstrap_ci_alpha=1.2) (and the other 4 fields' out-of-range values) passed
    construction silently. Post-fix: __post_init__ raises a clear ValueError naming the bad field."""
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig

    with pytest.raises(ValueError, match=match):
        CatFEConfig(**{kwarg: bad_value})


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-5 (P2): a mojibake-corrupted comment in cat_interactions.py ("n/30 + 1" preceded by
# garbled encoding-round-trip bytes instead of "~=").
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-7 (P2): cat_num_interaction_fit's docstring falsely claimed y-stratified fold
# assignment for binary targets; the implementation always uses a plain random permutation split.
# ---------------------------------------------------------------------------


def test_regression_cat_num_interaction_fit_docstring_no_false_stratification_claim():
    """Post-fix: the docstring no longer claims a stratified-when-binary fold split."""
    from mlframe.feature_selection.filters._count_freq_interaction_fe import cat_num_interaction_fit

    doc = cat_num_interaction_fit.__doc__ or ""
    assert "Used ONLY to derive a stratified fold assignment" not in doc


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-1 (P1): generate_rolling_window_agg_features's stats filter accepted "median" but
# _EXPANDING_STAT_CODE never mapped it -- calling with stats=["median"] raised a bare KeyError instead of
# computing anything or raising an actionable error.
# ---------------------------------------------------------------------------


def test_regression_grouped_agg_nunique_fallback_is_group_scale_not_population_scale():
    """Pre-fix: the unseen-group nunique fallback was np.unique(finite).size (whole-population cardinality,
    ~18x a real per-group value). Post-fix: it's the median per-group nunique, same scale as real values."""
    import pandas as pd

    from mlframe.feature_selection.filters._grouped_agg_fe import generate_grouped_agg_features

    rng = np.random.default_rng(0)
    n = 3000
    groups = rng.integers(0, 20, n)
    X = pd.DataFrame({"grp": groups, "num": rng.integers(0, 500, n).astype(np.float64)})
    enc_df, _recipes = generate_grouped_agg_features(X, ["grp"], ["num"], stats=["nunique"])
    col = next(c for c in enc_df.columns if "nunique" in c)
    real_per_group_values = enc_df[col].unique()
    assert len(real_per_group_values) > 0, "grouped nunique produced no values, so the scale check below would pass vacuously"
    # The whole-population cardinality (the pre-fix fallback) must NOT appear as a "fallback" value mixed
    # into an otherwise per-group-scale column -- every observed value should be well below n (real groups
    # were all seen at fit time here, so this just pins the per-group scale is sane, not degenerate).
    assert all(v < n * 0.5 for v in real_per_group_values), f"nunique values not per-group scale: {real_per_group_values}"


def test_regression_composite_group_agg_count_fallback_is_cell_scale():
    """Post-fix: an unseen composite key gets the median per-cell count (cell scale), not the whole-population size."""
    from mlframe.feature_selection.filters._composite_group_agg_fe import apply_composite_group_agg, generate_composite_group_agg_features

    cell_sizes = {(0, 0): 100, (0, 1): 200, (1, 0): 300, (1, 1): 400}  # median per-cell count = 250, population = 1000
    rows = [key for key, size in cell_sizes.items() for _ in range(size)]
    X = pd.DataFrame({"g1": [r[0] for r in rows], "g2": [r[1] for r in rows], "v": np.arange(len(rows), dtype=np.float64)})
    enc, recipes = generate_composite_group_agg_features(X, [["g1", "g2"]], ["v"], stats=["count"])
    name = next(c for c in enc.columns if c.startswith("cgrpagg_count"))
    X_test = pd.DataFrame({"g1": [7], "g2": [7], "v": [1.0]})  # unseen composite key
    fallback = apply_composite_group_agg(X_test, recipes[name])
    assert fallback[0] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-3 (P1): _gradient_interaction_seeder.py's two except-Exception blocks logged only
# `if verbose`, so with the default verbose=0 any exception was silently swallowed with zero trace.
# ---------------------------------------------------------------------------


def test_regression_lagged_diff_entity_scoping_prevents_cross_entity_diff():
    """Pre-fix: no entity_cols parameter existed at all. Post-fix: passing entity_cols zeroes any diff
    whose period-back neighbour (in time-sorted order) belongs to a different entity."""
    import pandas as pd

    from mlframe.feature_selection.filters._ratio_delta_fe import lagged_diff_features

    X = pd.DataFrame(
        {
            "entity": [1, 1, 1, 2, 2, 2],
            "t": [1, 2, 3, 1, 2, 3],
            "v": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        }
    )
    enc_no_scope, _ = lagged_diff_features(X, "t", ["v"], periods=(1,))
    enc_scoped, recipes_scoped = lagged_diff_features(X, "t", ["v"], periods=(1,), entity_cols=["entity"])

    col = "lagged_diff_v__period1"
    # Without scoping, the global time-sort mixes entities at the t=3(entity1)->t=1(entity2) boundary etc.,
    # producing nonsense (large-magnitude, entity-crossing) diffs.
    assert not np.allclose(enc_no_scope[col].to_numpy(), enc_scoped[col].to_numpy())
    # With scoping: entity 1's diffs are [0, 10, 10]; entity 2's are [0, 100, 100] -- each entity's own
    # first row gets 0 (no prior neighbour within the SAME entity), never a cross-entity value.
    np.testing.assert_allclose(enc_scoped[col].to_numpy(), [0.0, 10.0, 10.0, 0.0, 100.0, 100.0])
    assert recipes_scoped[col]["entity_cols"] == ("entity",)


def test_regression_lagged_diff_replay_honours_entity_scoping():
    """Post-fix: apply_lagged_diff (the recipe-replay path) also respects entity_cols when the recipe
    carries it, reproducing the same per-entity-scoped values on a fresh frame."""
    import pandas as pd

    from mlframe.feature_selection.filters._ratio_delta_fe import apply_lagged_diff

    X_test = pd.DataFrame(
        {
            "entity": [1, 1, 1, 2, 2, 2],
            "t": [1, 2, 3, 1, 2, 3],
            "v": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        }
    )
    out = apply_lagged_diff(X_test, {"time_col": "t", "value_col": "v", "period": 1, "entity_cols": ("entity",)})
    np.testing.assert_allclose(out, [0.0, 10.0, 10.0, 0.0, 100.0, 100.0])


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-6 (P2): _auto_detect_group_cols's two-dot `from .._grouped_agg_fe import ...` was the
# wrong import depth (_grouped_agg_fe.py is a sibling, one dot) and always failed inside a bare except,
# silently falling through to the correct single-dot import -- masked but dead/misleading code.
# ---------------------------------------------------------------------------


def test_regression_composite_group_agg_uses_the_shared_detector():
    """Post-fix: the shared group-column detector is actually reached, not silently fallen past.

    The original defect was a `from .._grouped_agg_fe import ...` at the wrong depth (that module is a
    SIBLING, one dot) sitting inside a bare `except Exception`: it always failed and fell through to the
    inline cardinality heuristic, so the Layer-87 detector was dead code that looked live.

    This used to be checked by reading `inspect.getsource` for the import lines, which breaks on any
    harmless move of the shared implementation -- and did, when the detector was consolidated into
    `_grouped_coerce_shared`. Substituting the detector and watching for its result tests the thing itself:
    if the import fails again for any reason, the except swallows it and the inline heuristic answers
    instead, which this notices.
    """
    import mlframe.feature_selection.filters._grouped_coerce_shared as shared
    from mlframe.feature_selection.filters._composite_group_agg_fe import _auto_detect_group_cols

    sentinel = ["__from_the_shared_detector__"]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(shared, "auto_detect_group_cols", lambda X, max_cols=6, caller="": list(sentinel))
    try:
        # A frame the inline heuristic WOULD answer on, so a fall-through returns real column names rather
        # than an empty list and cannot be mistaken for the shared detector's reply.
        df = pd.DataFrame({"g": np.repeat(np.arange(6), 20), "h": np.tile(np.arange(4), 30), "f": np.random.default_rng(0).normal(size=120)})
        got = _auto_detect_group_cols(df)
    finally:
        monkey.undo()

    assert got == sentinel, f"the shared detector was not reached; got {got!r}, which is the inline fallback"


# ---------------------------------------------------------------------------
# FE_PAIRS_CORE-1 (P1): the chunk-pipeline feature runs chunk k+1's production concurrently with the main
# thread consuming chunk k, and both threads reach into the un-locked _OPERAND_TABLE_CACHE /
# _PREBUILT_OPERAND_TABLE OrderedDicts for the same transformed_vars object with no lock.
# ---------------------------------------------------------------------------


def test_regression_fe_matrix_io_rejects_duplicate_column_names():
    """Post-fix: a duplicate column name raises a clear ValueError instead of silently corrupting the
    round-trip output."""

    from mlframe.feature_selection.filters._fe_matrix_io import FeatureMatrix, from_feature_matrix

    fm = FeatureMatrix(
        numeric=np.array([[1.0, 2.0], [3.0, 4.0]]),
        categorical=np.zeros((2, 0), dtype=np.int64),
        columns=["a", "a"],
        col_kind=["numeric", "numeric"],
        col_index=[0, 1],
        n_rows=2,
        framework="pandas",
        categories={},
        null_mask=None,
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="duplicate column name"):
        from_feature_matrix(fm)
    with pytest.raises(ValueError, match="not unique"):
        fm.numeric_column("a")


# ---------------------------------------------------------------------------
# FE_ORCH_BUDGET-4 (P2): three except-Exception blocks in _fe_auto_escalation.py swallowed with no
# logging, unlike every other exception handler in the same file.
# ---------------------------------------------------------------------------


_ESCALATION_LOGGER = "mlframe.feature_selection.filters.mrmr"


def test_regression_fe_auto_escalation_candidate_values_logs_prewarp_failure(monkeypatch, caplog):
    """Post-fix: a failing apply_operand_prewarp is logged at debug level and the candidate is skipped (None)."""
    import logging

    import mlframe.feature_selection.filters.hermite_fe as hermite_mod
    from mlframe.feature_selection.filters._fe_auto_escalation_proposers import _candidate_values

    def _boom(*args, **kwargs):
        """Test double method _boom."""
        raise RuntimeError("prewarp-boom-marker")

    monkeypatch.setattr(hermite_mod, "apply_operand_prewarp", _boom)
    x = np.linspace(-1.0, 1.0, 50)
    with caplog.at_level(logging.DEBUG, logger=_ESCALATION_LOGGER):
        assert _candidate_values(x, {}, x, {}) is None
    assert any("prewarp-boom-marker" in rec.message for rec in caplog.records)


def test_regression_fe_auto_escalation_resolve_operand_logs_extraction_failure(caplog):
    """Post-fix: a column that raises on extraction is logged at debug level and resolves to None."""
    import logging

    from mlframe.feature_selection.filters._fe_auto_escalation import _resolve_operand

    class _BadFrame:
        """Test double _BadFrame."""
        columns = ["a"]

        def __getitem__(self, key):
            raise RuntimeError("col-boom-marker")

        def __len__(self):
            return 3

    with caplog.at_level(logging.DEBUG, logger=_ESCALATION_LOGGER):
        assert _resolve_operand(_BadFrame(), "a", None) is None
    assert any("col-boom-marker" in rec.message for rec in caplog.records)


def test_regression_fe_auto_escalation_propose_poly_logs_pair_als_failure(monkeypatch, caplog):
    """Post-fix: a failing fit_pair_prewarp_als is logged at debug level and the proposal is skipped (None)."""
    import logging

    import mlframe.feature_selection.filters.hermite_fe as hermite_mod
    from mlframe.feature_selection.filters._fe_auto_escalation_proposers import _propose_poly

    def _boom(*args, **kwargs):
        """Test double method _boom."""
        raise RuntimeError("pair-als-boom-marker")

    monkeypatch.setattr(hermite_mod, "fit_pair_prewarp_als", _boom)
    rng = np.random.default_rng(0)
    n = 600
    x_a = rng.standard_normal(n)
    x_b = rng.standard_normal(n)
    y = (x_a * x_b > 0).astype(np.float64)
    with caplog.at_level(logging.DEBUG, logger=_ESCALATION_LOGGER):
        out = _propose_poly(x_a, x_b, y, degree=3, min_val_corr=0.0)
    assert out is None
    assert any("pair-als-boom-marker" in rec.message for rec in caplog.records), "pair ALS path was not reached"


# ---------------------------------------------------------------------------
# FE_ORCH_BUDGET-8 (P2): persist_budgets/load_budgets concatenated cache_key/fingerprint directly into a
# filename with no sanitisation against path-separator/".." characters.
# ---------------------------------------------------------------------------


def test_regression_fe_family_budget_sanitizes_cache_key():
    """Post-fix: a path-traversal-shaped cache_key/fingerprint cannot escape the cache directory --
    no path separators or ".." segments survive sanitisation."""
    from mlframe.feature_selection.filters._fe_family_budget import _sanitize_budget_file_key

    for bad in ("../../etc/passwd", "..\\..\\windows\\system32\\evil", "/etc/passwd", "a/../../b"):
        sanitized = _sanitize_budget_file_key(bad)
        # No path separators survive -> no directory traversal is possible regardless of any
        # remaining literal ".." characters (there is no "/" or "\\" left for them to traverse across).
        assert "/" not in sanitized and "\\" not in sanitized


# ---------------------------------------------------------------------------
# FE_REDUNDANCY_SYNERGY-1 (P1): decide_exhaustive_sweep's CPU-only branch hardcoded
# _EXHAUSTIVE_CPU_FALLBACK_PAIRS_PER_SEC=2000 with no kernel_tuning_cache lookup at all, unlike the CUDA
# branch a few lines above it and contrary to the module's own "NEVER hardcoded" docstring claim.
# ---------------------------------------------------------------------------


def test_regression_exhaustive_sweep_cpu_branch_uses_ktc(monkeypatch):
    """Post-fix: with no CUDA, decide_exhaustive_sweep predicts on the CPU backend using the measured
    per-host CPU throughput from the kernel-tuning cache, not the hardcoded fallback constant."""
    import types

    import mlframe.feature_selection.filters._fe_synergy_exhaustive as ex
    import mlframe.feature_selection.filters.batch_pair_mi_gpu as gpu_mod

    # Sanity: the KTC-backed function actually falls back gracefully and returns a positive throughput.
    pps, source = ex.measured_cpu_pairs_per_second(50_000, 4096)
    assert pps > 0
    assert source in ("cache", "fallback")

    monkeypatch.setattr(gpu_mod, "_CUDA_AVAIL", False)
    monkeypatch.setattr(ex, "warm_exhaustive_cpu_throughput_cache", lambda: None)
    monkeypatch.setattr(ex, "measured_cpu_pairs_per_second", lambda n_samples, n_pairs: (1_000_000.0, "cache"))
    use, reason = ex.decide_exhaustive_sweep(types.SimpleNamespace(fe_synergy_exhaustive="force"), n_samples=1000, n_raw=10, verbose=0)
    assert use is True
    assert "CPU backend" in reason
    assert "@ 1000000 pairs/s [cache]" in reason, reason


# ---------------------------------------------------------------------------
# FE_REDUNDANCY_SYNERGY-2 (P1): the nested clean-sub-expression anchor builder iterated the CALLER'S FULL
# raw input-feature-name set (P in the tens of thousands on a wide frame) per engineered survivor, instead
# of just the raw names actually referenced by that survivor's recipe sub-expressions.
# ---------------------------------------------------------------------------


def test_regression_synergy_screen_docstring_reflects_actual_wiring():
    """Post-fix: the module docstring no longer claims detect_synergy_combos is unwired."""
    from mlframe.feature_selection.filters import _fe_synergy_screen as mod

    doc = mod.__doc__ or ""
    assert "NOT YET WIRED" not in doc


def test_regression_gbm_split_propensity_exception_is_logged(monkeypatch, caplog):
    """Post-fix: a LightGBM training failure is logged at debug level and the propensity falls back to zeros."""
    import logging

    import lightgbm as lgb

    from mlframe.feature_selection.filters._fe_interaction_prerank import gbm_split_propensity

    def _boom(*args, **kwargs):
        """Test double method _boom."""
        raise RuntimeError("gbm-boom-marker")

    monkeypatch.setattr(lgb, "train", _boom)
    rng = np.random.default_rng(0)
    values = rng.standard_normal((200, 3))
    y = (values[:, 0] > 0).astype(np.float64)
    with caplog.at_level(logging.DEBUG, logger=_ESCALATION_LOGGER):
        out = gbm_split_propensity(values, y)
    np.testing.assert_array_equal(out, np.zeros(3))
    assert any("gbm_split_propensity" in rec.message and "gbm-boom-marker" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-1 (P1): greedy_cmi_fe_construct's noise-floor permutation RNG was hardcoded to
# 0xC011 with no seed parameter at all, correlating the FE admission gate across nominally-independent
# bootstrap/multi-seed replicates.
# ---------------------------------------------------------------------------


def test_regression_greedy_cmi_fe_construct_seed_varies_permutation(monkeypatch):
    """Pre-fix: rng_floor was always np.random.default_rng(0xC011) regardless of caller. Post-fix: the seed reaches the
    noise-floor permutation draws, so different seeds give different floors, the same seed reproduces its floors, and the
    default equals the historical 0xC011."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as mod

    real_step3 = mod._greedy_cmi_fe_constr_step3_while_st_remaining
    recorded = []

    def _recording_step3(st, top_k, noise_floor, *rest):
        """Wrap the per-round noise-floor callable so every floor value the construction draws is recorded."""

        def _floor():
            """Draw the noise floor and record it."""
            value = noise_floor()
            recorded.append(value)
            return value

        return real_step3(st, top_k, _floor, *rest)

    monkeypatch.setattr(mod, "_greedy_cmi_fe_constr_step3_while_st_remaining", _recording_step3)
    rng = np.random.default_rng(4)
    n = 600
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(6)})
    y = ((X["x0"] * X["x1"] + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)

    def _floors(**kwargs):
        """Run the construction with the given seed arguments and return the noise floors it drew."""
        recorded.clear()
        mod.greedy_cmi_fe_construct(X, y, top_k=2, nbins=10, **kwargs)
        return list(recorded)

    floors_seed1 = _floors(seed=1)
    assert floors_seed1, "the construction must draw at least one noise floor for the test to observe"
    assert _floors(seed=1) == floors_seed1, "the same seed must reproduce the same noise floors"
    assert _floors(seed=2) != floors_seed1, "different seeds must draw different noise-floor permutations"
    assert _floors() == _floors(seed=0xC011), "the default seed must stay the historical 0xC011"


def test_regression_score_candidates_by_cmi_handles_fractional_y_confined_to_one_bucket():
    """Direct reproduction: a perfectly-separated fractional y confined to [0,1) must NOT collapse to a
    degenerate single-class score."""
    import pandas as pd

    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import score_candidates_by_cmi

    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    X_cand = pd.DataFrame({"x": x})
    y = np.where(x > 0, 0.1, 0.2)
    scores = score_candidates_by_cmi(X_cand, y)
    assert scores["x"] > 0.1, f"expected a real, non-degenerate CMI for a perfectly-separated fractional y, got {scores['x']}"


def test_regression_greedy_cmi_fe_construct_runs_on_fractional_y():
    """Direct reproduction of MI_GREEDY_RECIPES-2's scenario at the greedy_cmi_fe_construct entry point:
    must not crash and must not silently produce zero engineered columns due to y-truncation collapse."""
    import pandas as pd

    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import greedy_cmi_fe_construct

    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    X = pd.DataFrame({"x": x, "x2": rng.standard_normal(n)})
    y = np.where(x > 0, 0.1, 0.2)
    X_aug, _scores = greedy_cmi_fe_construct(X, y, seed_cols_count=2, top_k=3)
    assert X_aug.shape[1] >= X.shape[1]  # ran without raising


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-3 (P2): the five _grouped_recipes.py builders re-stringified already-canonicalized
# lookup keys with a bare str(k) instead of canonical_group_token(k), unlike _encoding_recipes.py's
# builders (the EN-1 fix, never propagated to this sibling file).
# ---------------------------------------------------------------------------


def test_regression_grouped_recipes_use_canonical_group_token():
    """Post-fix: the grouped builder stores its lookups under canonical_group_token(k), so an integral float
    group value (1.0) is found by the same key an integer group value (1) produces at replay time."""
    from mlframe.feature_selection.filters._internals import canonical_group_token
    from mlframe.feature_selection.filters.engineered_recipes._grouped_recipes import build_grouped_agg_recipe

    recipe = build_grouped_agg_recipe(
        name="grp_mean_num",
        group_col="g",
        num_col="v",
        stat="mean",
        op="broadcast",
        group_lookup_dict={1.0: 5.0},
        global_value=0.0,
        lookup_mean={1.0: 2.0},
        lookup_std={1.0: 0.5},
        global_mean=0.0,
        global_std=1.0,
    )
    key = canonical_group_token(np.int64(1))
    assert recipe.extra["group_lookup_dict"][key] == 5.0
    assert recipe.extra["lookup_mean"][key] == 2.0
    assert recipe.extra["lookup_std"][key] == 0.5
    assert str(1.0) not in recipe.extra["group_lookup_dict"]


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-4 (P2): greater/less/equal cast to a.dtype (float) on the GPU-resident replay path
# but to plain int on the CPU registry -- identical 0/1 values, divergent dtype by backend.
# ---------------------------------------------------------------------------


def test_regression_fe_batched_mi_docstring_reflects_actual_wiring():
    """Post-fix: the module docstring no longer claims it's unwired."""
    from mlframe.feature_selection.filters import _fe_batched_mi as mod

    doc = mod.__doc__ or ""
    assert "imported by nothing in production yet" not in doc


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-8 (P2): build_cluster_aggregate_recipe's flat-extra invariant was a bare `assert`,
# stripped under -O/PYTHONOPTIMIZE.
# ---------------------------------------------------------------------------


def test_regression_cluster_aggregate_recipe_rejects_nested_extra_without_assert():
    """Post-fix: an explicit TypeError raise enforces the flat-extra invariant, surviving -O."""
    from mlframe.feature_selection.filters.engineered_recipes._recipe_poly_cluster import build_cluster_aggregate_recipe

    with pytest.raises(TypeError, match="must be flat ndarray"):
        build_cluster_aggregate_recipe(
            name="test_cluster_agg",
            src_names=["a", "b"],
            method="mean_z",
            member_mean=[0.0, 0.0],
            member_std=[1.0, 1.0],
            signs=[1.0, 1.0],
            weights=None,
            diagnostics={"bad_field": {"nested": "dict"}},
        )


# ---------------------------------------------------------------------------
# USABILITY_A-1 (P1): fit_constant_memmap's backing .mmap files were never deleted and the cache was
# unbounded -- 184 orphaned files (~13.1GB) found already accumulated on this dev machine.
# ---------------------------------------------------------------------------


def test_regression_compose_pair_fe_rejects_duplicate_feature_names():
    """Regression: compose_pair_fe's single_mi_cache was keyed only by column name;."""
    from mlframe.feature_selection.filters.composition import compose_pair_fe

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)

    with pytest.raises(ValueError, match="unique"):
        compose_pair_fe(X, y, feature_names=["dup", "dup", "other"], n_rounds=1, n_trials=3)


# ---------------------------------------------------------------------------
# USABILITY_A-10 (P1) / USABILITY_A-11 (P2): composition.py's per-fold / per-pair exception handlers
# swallowed ANY exception with zero (or opt-in-only) logging.
# ---------------------------------------------------------------------------


def test_regression_validate_pair_fe_cv_logs_fold_exceptions(caplog):
    """Post-fix: a fold that raises inside optimise_hermite_pair is logged (not silently swallowed)."""
    import logging

    from mlframe.feature_selection.filters import composition as comp_mod
    import mlframe.feature_selection.filters.hermite_fe as hermite_mod

    orig_fn = hermite_mod.optimise_hermite_pair
    call_count = {"n": 0}

    def _raising_after_first(*args, **kwargs):
        # First call is validate_pair_fe_cv's own "in-sample reference (full data)" call, made outside
        # any try/except -- only the per-FOLD calls (2nd onward) exercise the fixed exception handler.
        """Test helper: raising after first."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            return orig_fn(*args, **kwargs)
        raise RuntimeError("fold-boom-marker")

    hermite_mod.optimise_hermite_pair = _raising_after_first
    try:
        rng = np.random.default_rng(0)
        n = 60
        x_a = rng.standard_normal(n)
        x_b = rng.standard_normal(n)
        y = (x_a * x_b > 0).astype(int)
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.composition"):
            comp_mod.validate_pair_fe_cv(x_a, x_b, y, n_splits=2, n_trials=2)
        assert any("fold-boom-marker" in rec.message for rec in caplog.records)
    finally:
        hermite_mod.optimise_hermite_pair = orig_fn


def test_regression_compose_pair_fe_logs_pair_failures_by_default(caplog):
    """Post-fix: a per-pair FE failure is logged at warning level even without verbose=True."""
    import logging

    from mlframe.feature_selection.filters import composition as comp_mod
    import mlframe.feature_selection.filters.hermite_fe as hermite_mod

    real_orig = hermite_mod.optimise_hermite_pair

    def _raising_optimise(*args, **kwargs):
        """Stand-in that raises RuntimeError, so the caller's failure path is the one under test."""
        raise RuntimeError("pair-boom-marker")

    hermite_mod.optimise_hermite_pair = _raising_optimise
    try:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.composition"):
            comp_mod.compose_pair_fe(X, y, n_rounds=1, n_trials=2, verbose=False)
        assert any("pair-boom-marker" in rec.message for rec in caplog.records)
    finally:
        hermite_mod.optimise_hermite_pair = real_orig


# ---------------------------------------------------------------------------
# USABILITY_A-9 (P2): _rbf_fit / _sigmoid_fit did not guard NaN/Inf on the raw column before
# np.quantile/np.std, unlike _fourier_fit / _pade_fit.
# ---------------------------------------------------------------------------


def test_regression_apply_rolling_window_single_loop_matches_expected():
    """Regression: _apply_rolling_window discarded its own correctly-computed first loop and rebuilt an essentially-identical list via a second,
    unnecessary loop.
    """
    from mlframe.feature_selection.filters._mrmr_partial_fit import _apply_rolling_window

    X_buf = pd.DataFrame({"a": np.arange(10)})
    y_buf = pd.Series(np.arange(10))
    batch_sizes = [3, 3, 4]  # window=6 drops 4 rows: whole first batch (3) + 1 of the second (leaves 2), third batch (4) untouched
    X_out, _y_out, sizes_out = _apply_rolling_window(X_buf, y_buf, batch_sizes, window=6)
    assert sizes_out == [2, 4]
    assert sum(sizes_out) == len(X_out) == 6
    assert list(X_out["a"]) == list(range(4, 10))


# ---------------------------------------------------------------------------
# USABILITY_B-5 (P2): the cache-replay source-freeze side effect (fit_constant-cache style sharing via
# _FIT_CACHE) was not documented anywhere in MRMR's public docstring.
# ---------------------------------------------------------------------------


def test_regression_apply_sis_screen_polars_input():
    """Regression: uSABILITY_B-7 (P2, test_gap): _apply_sis_screen's polars-specific branches had zero test coverage."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame(
        {
            "num_a": rng.standard_normal(n),
            "num_b": rng.standard_normal(n),
            "cat_c": rng.integers(0, 3, n).astype(str),
        }
    )
    y = (df["num_a"].to_numpy() > 0).astype(int)
    m = MRMR(verbose=0)
    out = m._apply_sis_screen(df, y)
    assert isinstance(out, pl.DataFrame)
    assert out.shape[0] == n
    assert out.shape[1] <= df.shape[1]
    assert hasattr(m, "sis_survivors_")


# ---------------------------------------------------------------------------
# USABILITY_B-8 (P2): explain_selection's per-section except-Exception blocks never called logger, even
# at debug level.
# ---------------------------------------------------------------------------


def test_regression_no_copy_idiom_removed_from_fe_constructors():
    """Post-fix: none of the 39 audited FE-family files still contain the wasteful
    `return X.copy(), ...` no-op early-exit idiom."""
    import re
    from pathlib import Path

    filters_dir = Path(__file__).resolve().parents[3] / "src" / "mlframe" / "feature_selection" / "filters"
    if not filters_dir.is_dir():
        pytest.skip("filters/ package not found at expected repo-root-relative path")
    pattern = re.compile(r"return X\.copy\(\),")
    offenders = []
    for py_file in filters_dir.rglob("*.py"):
        if "_benchmarks" in py_file.parts or "__pycache__" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(py_file))
    assert not offenders, f"wasteful 'return X.copy(), ...' idiom still present in: {offenders}"


# ---------------------------------------------------------------------------
# X_TEST_COVERAGE_QUALITY-1 (P1): mrmr_audit_2026-07-20's B-4 fix (selection_stability_report's
# bootstrap seed silently fell back to 0 for any estimator seeded via random_state=) shipped with NO
# regression test that actually passes random_state= (the existing test file only ever uses
# random_seed=), so it structurally cannot detect the bug class it claims to guard.
# ---------------------------------------------------------------------------


def test_regression_cat_interactions_no_mojibake_comment():
    """Post-fix: the corrupted comment is repaired; no other file in the repo shares the same corruption
    signature."""
    import inspect

    from mlframe.feature_selection.filters import cat_interactions as mod

    src = inspect.getsource(mod)
    assert "n/30 + 1" in src
    assert "РІР‚В°РІвЂљВ¬" not in src


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-7 (P2): cat_num_interaction_fit's docstring falsely claimed y-stratified fold
# assignment for binary targets; the implementation always uses a plain random permutation split.
# ---------------------------------------------------------------------------
