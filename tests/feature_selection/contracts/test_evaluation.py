"""Unit + biz_value tests for ``mlframe.feature_selection.filters.evaluation``.

Public surface covered:
- ``get_candidate_name`` (name join)
- ``should_skip_candidate`` (skip rules)
- ``handle_best_candidate`` (best-gain bookkeeping, time-out detection)
- ``evaluate_gain`` (njit redundancy loop; covered indirectly through evaluate_candidate)
- ``evaluate_candidate`` (per-candidate scoring with cache wiring + algorithm switch)
- ``find_best_partial_gain`` (max-gain selection with skip filtering)

The screen/orchestrator wrapper ``evaluate_candidates`` is integration-level and is exercised by tests/feature_selection/test_screen.py; we don't duplicate that here.
"""
from __future__ import annotations

import time

import numba
import numpy as np
import pytest
from numba.core import types
from numba.typed import Dict as NumbaDict

from mlframe.feature_selection.filters._internals import LARGE_CONST
from mlframe.feature_selection.filters.evaluation import (
    evaluate_candidate,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from mlframe.feature_selection.filters.info_theory import merge_vars


# ================================================================================================
# Fixture factory: build a fully-populated evaluate_candidate kwargs dict.
# ================================================================================================


def _new_entropy_cache() -> NumbaDict:
    return NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)


def _new_cond_mi_cache() -> NumbaDict:
    return NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)


def _build_xor_factors(n: int = 2000, seed: int = 42) -> tuple:
    """Three integer columns plus an XOR label. ``factors_data`` packs x_0, x_1, x_2; the target is ``y = x_0 XOR x_1`` (x_2 is unrelated noise).
    XOR is the textbook synergy case: I(x_0;y) ~ 0 and I(x_1;y) ~ 0 individually, but I(x_0,x_1;y) is large.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, size=n).astype(np.int32)
    x1 = rng.integers(0, 2, size=n).astype(np.int32)
    x2 = rng.integers(0, 2, size=n).astype(np.int32)
    y_col = np.bitwise_xor(x0, x1).astype(np.int32)
    factors_data = np.column_stack([x0, x1, x2, y_col]).astype(np.int32)
    factors_nbins = np.array([2, 2, 2, 2], dtype=np.int64)
    factors_names = ["x0", "x1", "x2", "y"]
    return factors_data, factors_nbins, factors_names


def _make_eval_kwargs(factors_data, factors_nbins, factors_names, y_indices=(3,), cand_idx=0, X=(0,), selected_vars=None) -> dict:
    """Build a complete evaluate_candidate kwargs dict. Individual tests override only the field under test."""
    y_arr = np.asarray(y_indices, dtype=np.int64)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors_data, vars_indices=y_arr, var_is_nominal=None, factors_nbins=factors_nbins, dtype=np.int32,
    )
    n_candidates = factors_data.shape[1]
    expected_gains = np.zeros(n_candidates, dtype=np.float64)
    return dict(
        cand_idx=cand_idx,
        X=tuple(X),
        y=y_arr,
        nexisting=0,
        best_gain=-LARGE_CONST,
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        factors_names=factors_names,
        expected_gains=expected_gains,
        partial_gains={},
        selected_vars=list(selected_vars) if selected_vars else [],
        baseline_npermutations=0,  # skip permutation work in unit tests
        classes_y=classes_y,
        classes_y_safe=classes_y.copy(),
        freqs_y=freqs_y,
        freqs_y_safe=None,
        use_gpu=False,
        cached_MIs={},
        cached_confident_MIs={},
        cached_cond_MIs=_new_cond_mi_cache(),
        entropy_cache=_new_entropy_cache(),
        mrmr_relevance_algo="fleuret",
        mrmr_redundancy_algo="fleuret",
        max_veteranes_interactions_order=2,
        extra_knowledge_multipler=-1.0,
        sink_threshold=-1.0,
        dtype=np.int32,
        verbose=0,
        ndigits=5,
        use_simple_mode=False,  # exercise the conditional-MI path when selected_vars is non-empty
    )


@pytest.fixture
def xor_factors():
    return _build_xor_factors()


# ================================================================================================
# get_candidate_name
# ================================================================================================


def test_get_candidate_name_single():
    assert get_candidate_name([2], ["a", "b", "c", "d"]) == "c"


def test_get_candidate_name_multi():
    assert get_candidate_name([0, 2], ["a", "b", "c", "d"]) == "a-c"


# ================================================================================================
# should_skip_candidate
# ================================================================================================


def test_should_skip_failed_candidate():
    expected_gains = np.zeros(5, dtype=np.float64)
    skip, nexist = should_skip_candidate(
        cand_idx=2, X=(2,), interactions_order=1,
        failed_candidates={2}, added_candidates=set(),
        expected_gains=expected_gains, selected_vars=[], selected_interactions_vars=[],
    )
    assert skip is True
    assert nexist == 0


def test_should_skip_added_candidate():
    expected_gains = np.zeros(5, dtype=np.float64)
    skip, _ = should_skip_candidate(
        cand_idx=3, X=(3,), interactions_order=1,
        failed_candidates=set(), added_candidates={3},
        expected_gains=expected_gains, selected_vars=[], selected_interactions_vars=[],
    )
    assert skip is True


def test_should_skip_already_scored_candidate():
    # Non-zero expected_gains[cand_idx] also short-circuits.
    expected_gains = np.zeros(5, dtype=np.float64)
    expected_gains[1] = 0.42
    skip, _ = should_skip_candidate(
        cand_idx=1, X=(1,), interactions_order=1,
        failed_candidates=set(), added_candidates=set(),
        expected_gains=expected_gains, selected_vars=[], selected_interactions_vars=[],
    )
    assert skip is True


def test_should_not_skip_fresh_candidate():
    expected_gains = np.zeros(5, dtype=np.float64)
    skip, nexist = should_skip_candidate(
        cand_idx=4, X=(4,), interactions_order=1,
        failed_candidates={0, 1}, added_candidates={2},
        expected_gains=expected_gains, selected_vars=[], selected_interactions_vars=[],
    )
    assert skip is False
    assert nexist == 0


def test_should_skip_interaction_with_selected_subelement():
    # interactions_order > 1: a k-way candidate is skipped if any sub-element is already in selected_interactions_vars at this stage.
    expected_gains = np.zeros(5, dtype=np.float64)
    skip, _ = should_skip_candidate(
        cand_idx=4, X=(1, 2), interactions_order=2,
        failed_candidates=set(), added_candidates=set(),
        expected_gains=expected_gains, selected_vars=[], selected_interactions_vars=[1],
    )
    assert skip is True


def test_should_skip_interaction_only_unknown_skips_partial_overlap():
    # interactions_order > 1, only_unknown_interactions default True: any sub-element already in selected_vars triggers skip.
    expected_gains = np.zeros(5, dtype=np.float64)
    skip, nexist = should_skip_candidate(
        cand_idx=4, X=(1, 2), interactions_order=2,
        failed_candidates=set(), added_candidates=set(),
        expected_gains=expected_gains, selected_vars=[1], selected_interactions_vars=[],
        only_unknown_interactions=True,
    )
    assert skip is True
    assert nexist == 1


def test_should_skip_lineage_filter():
    # Engineered column 4 is built from parents {1, 2}; a candidate that pairs 4 with parent 2 must be skipped.
    expected_gains = np.zeros(6, dtype=np.float64)
    lineage = {4: frozenset({1, 2})}
    skip, _ = should_skip_candidate(
        cand_idx=5, X=(4, 2), interactions_order=2,
        failed_candidates=set(), added_candidates=set(),
        expected_gains=expected_gains, selected_vars=[], selected_interactions_vars=[],
        engineered_lineage=lineage,
    )
    assert skip is True


# ================================================================================================
# handle_best_candidate
# ================================================================================================


def test_handle_best_candidate_updates_when_gain_exceeds():
    new_best, new_cand, run_out = handle_best_candidate(
        current_gain=0.5, best_gain=0.2, X=(7,), best_candidate=(3,),
        factors_names=["a", "b", "c", "d", "e", "f", "g", "h"], verbose=0,
    )
    assert new_best == pytest.approx(0.5)
    assert new_cand == (7,)
    assert run_out is False


def test_handle_best_candidate_keeps_previous_when_gain_lower():
    new_best, new_cand, run_out = handle_best_candidate(
        current_gain=0.1, best_gain=0.5, X=(7,), best_candidate=(3,),
        factors_names=["a", "b", "c", "d", "e", "f", "g", "h"], verbose=0,
    )
    assert new_best == pytest.approx(0.5)
    assert new_cand == (3,)
    assert run_out is False


def test_handle_best_candidate_detects_runtime_exhaustion():
    # start_time deep in the past with a tiny budget must trip the timeout path.
    past = time.perf_counter() - 3600.0
    _, _, run_out = handle_best_candidate(
        current_gain=0.1, best_gain=0.5, X=(7,), best_candidate=(3,),
        factors_names=["a", "b", "c", "d", "e", "f", "g", "h"], verbose=0,
        max_runtime_mins=0.001, start_time=past,
    )
    assert run_out is True


# ================================================================================================
# find_best_partial_gain
# ================================================================================================


def test_find_best_partial_gain_returns_max():
    partial_gains = {0: (0.1, 0), 1: (0.5, 1), 2: (0.3, 2)}
    candidates = [(0,), (1,), (2,)]
    best_gain, best_key = find_best_partial_gain(
        partial_gains=partial_gains, failed_candidates=set(), added_candidates=set(),
        candidates=candidates, selected_vars=[],
    )
    assert best_key == 1
    assert best_gain == pytest.approx(0.5)


def test_find_best_partial_gain_skips_failed_and_added():
    partial_gains = {0: (0.9, 0), 1: (0.5, 1), 2: (0.7, 2)}
    candidates = [(0,), (1,), (2,)]
    # 0 (best) is failed, 2 (next) is added => 1 wins.
    best_gain, best_key = find_best_partial_gain(
        partial_gains=partial_gains, failed_candidates={0}, added_candidates={2},
        candidates=candidates, selected_vars=[],
    )
    assert best_key == 1
    assert best_gain == pytest.approx(0.5)


def test_find_best_partial_gain_skips_selected_subelement():
    # Candidate 0 references variable 5 which is already in selected_vars => skip; candidate 1 (variable 6) wins.
    partial_gains = {0: (0.9, 0), 1: (0.4, 1)}
    candidates = [(5,), (6,)]
    best_gain, best_key = find_best_partial_gain(
        partial_gains=partial_gains, failed_candidates=set(), added_candidates=set(),
        candidates=candidates, selected_vars=[5],
    )
    assert best_key == 1
    assert best_gain == pytest.approx(0.4)


def test_find_best_partial_gain_empty_returns_none():
    best_gain, best_key = find_best_partial_gain(
        partial_gains={}, failed_candidates=set(), added_candidates=set(),
        candidates=[], selected_vars=[],
    )
    assert best_key is None
    assert best_gain == -LARGE_CONST


# ================================================================================================
# evaluate_candidate -- cache wiring + GPU dispatch + algorithm switch
# ================================================================================================


def test_evaluate_candidate_returns_tuple_with_set(xor_factors):
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    current_gain, sink_reasons = evaluate_candidate(**kwargs)
    assert isinstance(current_gain, float)
    assert isinstance(sink_reasons, set)


def test_evaluate_candidate_cached_confident_short_circuits(xor_factors):
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    # ``cached_confident_MIs`` stores ``(bootstrapped_gain, confidence)`` tuples
    # (the producer is ``confirm_candidate`` -- see _confirm_predictor.py:410).
    # The short-circuit unpacks the tuple and returns ``bootstrapped_gain``.
    sentinel_gain = 1.2345
    kwargs["cached_confident_MIs"] = {(0,): (sentinel_gain, 0.95)}
    # selected_vars empty => current_gain == direct_gain == sentinel_gain.
    current_gain, _ = evaluate_candidate(**kwargs)
    assert current_gain == pytest.approx(sentinel_gain)


def test_evaluate_candidate_cached_confident_branch_routed_through_su_floor_scaling(xor_factors):
    """A confirmed candidate re-entering the ``cached_confident_MIs`` branch must be scored on the SAME scale as the
    fresh ``mi_direct`` else-path, NOT returned as raw MI.

    The branch is normally unreachable (a confirmed candidate's cand_idx is in added_/failed_candidates, so
    should_skip_candidate filters it out before re-entry). This pins the safety-net behavior if that invariant ever
    breaks: with ``mi_normalization='su'`` active, the cached bootstrapped gain is SU floor-scaled identically to the
    else-path -- otherwise a raw-MI-scale value would be compared against the SU-scale min_relevance_gain floor.

    Pre-fix: the branch returned the bootstrapped gain unmodified (raw MI), bypassing the SU floor-scaling.
    Post-fix: the branch routes through ``_su_normalize_relevance`` -- so under SU the returned gain is the SU-scaled
    value, and under the default (SU off) it stays exactly the bootstrapped gain (legacy byte-identical).
    """
    from mlframe.feature_selection.filters.evaluation import _su_normalize_relevance
    from mlframe.feature_selection.filters.info_theory import set_su_normalization

    factors_data, factors_nbins, factors_names = xor_factors
    sentinel_gain = 1.2345

    # Default (SU off): unchanged bootstrapped gain.
    set_su_normalization(False)
    try:
        kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
        kwargs["cached_confident_MIs"] = {(0,): (sentinel_gain, 0.95)}
        current_gain, _ = evaluate_candidate(**kwargs)
        assert current_gain == pytest.approx(sentinel_gain)

        # SU on: the branch SU-scales the bootstrapped gain, matching what the shared helper computes on the same X/y.
        set_su_normalization(True)
        expected_su = _su_normalize_relevance(
            sentinel_gain, (0,), kwargs["y"], factors_data, factors_nbins, np.int32,
        )
        # SU is a real rescale on this synthetic, so the routed value must differ from the raw sentinel AND equal the helper.
        assert expected_su != pytest.approx(sentinel_gain)
        kwargs2 = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
        kwargs2["cached_confident_MIs"] = {(0,): (sentinel_gain, 0.95)}
        current_gain_su, _ = evaluate_candidate(**kwargs2)
        assert current_gain_su == pytest.approx(expected_su)
    finally:
        set_su_normalization(False)


def test_evaluate_candidate_cached_mi_short_circuits(xor_factors):
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    sentinel = 0.789
    kwargs["cached_MIs"] = {(0,): sentinel}
    current_gain, _ = evaluate_candidate(**kwargs)
    assert current_gain == pytest.approx(sentinel)


def test_evaluate_candidate_populates_cached_MIs_on_miss(xor_factors):
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    assert (0,) not in kwargs["cached_MIs"]
    evaluate_candidate(**kwargs)
    # CPU path inserts the freshly-computed MI under tuple key X.
    assert (0,) in kwargs["cached_MIs"]
    assert isinstance(kwargs["cached_MIs"][(0,)], float)


def test_evaluate_candidate_cpu_dispatch_no_gpu_required(xor_factors):
    # ``use_gpu=False`` must not raise even when cupy is absent; this exercises the CPU mi_direct branch.
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    kwargs["use_gpu"] = False
    current_gain, _ = evaluate_candidate(**kwargs)
    assert current_gain >= 0.0


def test_evaluate_candidate_zero_direct_gain_returns_zero(xor_factors):
    # Force direct_gain=0 via cached_MIs sentinel; the function must short-circuit current_gain to 0.
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    kwargs["cached_MIs"] = {(0,): 0.0}
    current_gain, sink_reasons = evaluate_candidate(**kwargs)
    assert current_gain == 0
    assert sink_reasons == set()


def test_evaluate_candidate_fleuret_algo_runs_conditional_path(xor_factors):
    # With selected_vars non-empty and use_simple_mode=False the fleuret conditional-MI loop runs and writes cached_cond_MIs.
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names, cand_idx=1, X=(1,), selected_vars=[0])
    cond_cache = kwargs["cached_cond_MIs"]
    assert len(cond_cache) == 0
    evaluate_candidate(**kwargs)
    # fleuret writes at least one (X, Z) key into cached_cond_MIs.
    assert len(cond_cache) >= 1


def test_evaluate_candidate_partial_gains_early_exit(xor_factors):
    # When a prior partial gain is already <= best_gain the function short-circuits without running evaluate_gain.
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names, cand_idx=1, X=(1,), selected_vars=[0])
    kwargs["best_gain"] = 0.9
    kwargs["partial_gains"] = {1: (0.05, 0)}  # tiny partial gain, well below best.
    # Pre-seed direct gain so we don't trigger mi_direct path.
    kwargs["cached_MIs"] = {(1,): 0.5}
    current_gain, sink_reasons = evaluate_candidate(**kwargs)
    assert current_gain == pytest.approx(0.05)
    assert sink_reasons == set()


# ================================================================================================
# biz_value tests
# ================================================================================================


@pytest.mark.fast
def test_biz_evaluate_xor_synergy_beats_marginal(xor_factors):
    """XOR: I(x_1; y | x_0) should be substantial (~0.69 nats in the noiseless case) while I(x_1; y) ~ 0.
    Evaluating candidate x_1 with x_0 already selected (use_simple_mode=False, fleuret) must surface that synergy as a positive gain.
    """
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names, cand_idx=1, X=(1,), selected_vars=[0])
    current_gain, _ = evaluate_candidate(**kwargs)
    # Synergy gain must be materially above marginal (~0). 0.1 nats is a loose floor; noiseless XOR sits around ln(2) ~= 0.69.
    assert current_gain >= 0.1, f"expected XOR synergy gain >= 0.1 nats, got {current_gain}"


@pytest.mark.fast
def test_biz_cache_hit_speedup(xor_factors):
    """A second evaluate_candidate call on the same X should hit cached_MIs and skip the CPU mi_direct path, giving a measurable speedup.
    Floor at 2x because numba JIT warm-up + permutation budget (baseline_npermutations=0 here) keep absolute timings tiny and noisy.
    """
    factors_data, factors_nbins, factors_names = xor_factors

    # First call: cache is cold and mi_direct runs.
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    # Warm up numba once before timing so JIT compile doesn't dominate the cold call.
    evaluate_candidate(**_make_eval_kwargs(factors_data, factors_nbins, factors_names, X=(2,)))

    t0 = time.perf_counter()
    evaluate_candidate(**kwargs)
    cold = time.perf_counter() - t0
    assert (0,) in kwargs["cached_MIs"]

    # Second call: same kwargs dict, so cached_MIs already has (0,) populated.
    t1 = time.perf_counter()
    evaluate_candidate(**kwargs)
    hot = time.perf_counter() - t1

    # Guard against zero-division on absurdly fast hot path.
    assert hot >= 0.0
    if cold <= 0.0:
        pytest.skip("cold path too fast to measure reliably")
    speedup = cold / max(hot, 1e-9)
    assert speedup >= 2.0, f"expected >= 2x cache speedup, got cold={cold:.6f}s hot={hot:.6f}s ratio={speedup:.2f}"


# ================================================================================================
# Run Tests
# ================================================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
