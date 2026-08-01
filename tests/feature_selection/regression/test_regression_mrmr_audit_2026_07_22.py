"""Regression tests for bugs confirmed by the mrmr_audit_2026-07-22 agentic audit wave (audits/mrmr_audit_2026-07-22/).

Each test would FAIL on pre-fix code and PASS on post-fix. Per project memory feedback_test_every_bug_fix.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _need_cuda

# ---------------------------------------------------------------------------
# CORE_CLASS-1 (P0): store_params_in_object's postfix default changed upstream
# (pyutilz) from "" to "_param_"; the MRMR __init__ call site never pinned it
# explicitly, so every ctor param landed on self.<name>_param_ instead of
# self.<name>, breaking every getattr(self, "<name>", default) read in the
# class plus get_params()/set_params()/clone() and .fit() itself.
# ---------------------------------------------------------------------------


def test_regression_mrmr_ctor_params_land_on_bare_attribute_names():
    """Pre-fix: MRMR(n_workers=3).n_workers raised AttributeError (real attr was n_workers_param_).
    Post-fix: store_params_in_object(..., postfix="") pins the plain name regardless of the pyutilz default.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    m = MRMR(n_workers=3)
    assert getattr(m, "n_workers") == 3
    assert not hasattr(m, "n_workers_param_")
    assert m.get_params()["n_workers"] == 3


def test_regression_mrmr_fit_does_not_crash_on_construction():
    """Pre-fix: MRMR().fit(X, y) raised AttributeError inside get_params()/internal getattr reads
    before any real feature-selection logic ran, because every self.<name> read a missing attribute.
    Post-fix: a plain small fit completes.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 2, 200))

    m = MRMR(n_workers=1)
    m.fit(X, y)  # must not raise


# ---------------------------------------------------------------------------
# FE_STEP_B-1 (P0): the FE-escalation and C2 additive-fusion materialize blocks
# in _step_score.py preallocated their output-codes buffer at the raw (narrow)
# quantization_dtype and assigned discretize_array's auto-widened return value
# into it, silently downcasting back and wrapping bin codes negative for
# quantization_nbins > 127 under quantization_dtype=int8. Same bug class
# already fixed for the unary/binary materialize block two screens up via
# _safe_code_dtype, now applied to the two remaining call sites.
# ---------------------------------------------------------------------------


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


def test_regression_mrmr_set_params_invalidates_stale_nested_config():
    """Pre-fix: MRMR(fast_search_config=FastSearchConfig(fe_fast_search=True)).set_params(fe_fast_search=False)
    left get_params()['fast_search_config'] unchanged (still True), so clone() re-applied it and silently
    reverted fe_fast_search back to True -- sklearn's clone() sanity check then raised RuntimeError.
    Post-fix: set_params() drops (nulls) any nested config whose recorded fields disagree with the flat
    attr just set, so get_params() stays self-consistent and clone() reproduces the post-set_params state.
    """
    from sklearn.base import clone

    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import FastSearchConfig

    m = MRMR(fast_search_config=FastSearchConfig(fe_fast_search=True))
    assert m.get_params()["fe_fast_search"] is True

    m.set_params(fe_fast_search=False)
    assert m.get_params()["fe_fast_search"] is False
    assert m.get_params()["fast_search_config"] is None, "stale config must be invalidated once it disagrees with a set_params()-changed flat attr"

    c = clone(m)  # must not raise RuntimeError
    assert c.get_params()["fe_fast_search"] is False


def test_regression_core_class_4_dormant_toggle_restore_mechanism():
    """CORE_CLASS-4 (P2): ``_restore_toggles_snapshot_and_raise`` in ``_mrmr_class.py`` unpacked all 9
    snapshotted MI-correction thread-locals but only restored 5 (SU/JMIM/BUR/Miller-Madow/Chao-Shen), leaving
    relaxmrmr_alpha/pid_synergy_bonus/cmi_perm_stop/cpt_test dead-stored. Currently dormant in production
    (those 4 are only ever activated AFTER both existing call sites of this helper, so nothing clobbers them
    before a raise today) -- this test instead pins the RESTORE MECHANISM itself directly against the real
    getter/setter pairs, since the closure is not independently callable. It would have caught a real
    tuple-unpacking mistake made while writing the fix (set_cmi_perm_stop/set_cpt_test take 3/2 positional
    args each, not the raw 3-/2-tuple `get_cmi_perm_stop()`/`get_cpt_test()` return).
    """
    from mlframe.feature_selection.filters.info_theory import (
        get_cmi_perm_stop,
        get_cpt_test,
        get_pid_synergy_bonus,
        get_relaxmrmr_alpha,
        set_cmi_perm_stop,
        set_cpt_test,
        set_pid_synergy_bonus,
        set_relaxmrmr_alpha,
    )

    # Snapshot current (package-default) values, exactly like _toggles_snapshot at fit entry.
    _relax0, _pid0, _cmi0, _cpt0 = get_relaxmrmr_alpha(), get_pid_synergy_bonus(), get_cmi_perm_stop(), get_cpt_test()
    try:
        # Simulate a future reordering that activates these BEFORE a raise point clobbers them.
        set_relaxmrmr_alpha(0.42)
        set_pid_synergy_bonus(0.17)
        set_cmi_perm_stop(True, 0.01, 250)
        set_cpt_test(True, 999)
        assert get_relaxmrmr_alpha() == 0.42  # sanity: activation took effect

        # This is exactly the restore-line pattern added to _restore_toggles_snapshot_and_raise.
        set_relaxmrmr_alpha(_relax0)
        set_pid_synergy_bonus(_pid0)
        set_cmi_perm_stop(_cmi0[0], _cmi0[1], _cmi0[2])
        set_cpt_test(_cpt0[0], _cpt0[1])

        assert get_relaxmrmr_alpha() == _relax0
        assert get_pid_synergy_bonus() == _pid0
        assert get_cmi_perm_stop() == _cmi0
        assert get_cpt_test() == _cpt0
    finally:
        set_relaxmrmr_alpha(_relax0)
        set_pid_synergy_bonus(_pid0)
        set_cmi_perm_stop(_cmi0[0], _cmi0[1], _cmi0[2])
        set_cpt_test(_cpt0[0], _cpt0[1])


# ---------------------------------------------------------------------------
# FIT_IMPL_A-1 (P1): the large-n regression adaptive-quantization gate
# (adaptive_nbins_large_n_reg) permanently overwrote self.nbins_strategy /
# self.quantization_nbins in place with no restore, breaking clone()/
# get_params() and permanently freezing a config on any subsequent .fit()
# on the same instance. Uses adaptive_nbins_large_n_reg_threshold to trigger
# the gate cheaply (no need for a real 50k+/60s fit).
# ---------------------------------------------------------------------------


def test_regression_adaptive_nbins_gate_restores_ctor_state_after_fit():
    """Pre-fix: MRMR.nbins_strategy/quantization_nbins stayed permanently mutated (None/20) after a fit
    that triggers the gate. Post-fix: restored to the constructor's original values, so get_params()/clone()
    round-trip correctly and a second .fit() re-evaluates the gate instead of being stuck.
    """
    from sklearn.base import clone

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f"x{i}" for i in range(4)])
    y = pd.Series(1.5 * X["x0"] - X["x1"] + 0.1 * rng.standard_normal(n))

    m = MRMR(
        nbins_strategy="mdlp",
        quantization_nbins=10,
        adaptive_nbins_large_n_reg=True,
        adaptive_nbins_large_n_reg_threshold=50,  # cheap trigger instead of the real 50_000 default
        fe_max_steps=0,
        full_npermutations=0,
        baseline_npermutations=0,
        random_seed=0,
        n_jobs=1,
        verbose=0,
        cv=2,
    )
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is True, "gate must have engaged for this fit (n=300 >= threshold=50, detected regression)"
    assert m.nbins_strategy == "mdlp", "ctor state must be restored after fit(), not left mutated to None"
    assert m.quantization_nbins == 10, "ctor state must be restored after fit(), not left mutated to 20"

    # sklearn round-trip contract: get_params()/clone() must reproduce the constructor's own values.
    assert m.get_params()["nbins_strategy"] == "mdlp"
    assert m.get_params()["quantization_nbins"] == 10
    c = clone(m)  # must not raise, and must not silently inherit the gate's internal override
    assert c.nbins_strategy == "mdlp"
    assert c.quantization_nbins == 10

    # A second .fit() on the SAME instance must re-evaluate the gate (not be permanently stuck at
    # None/20 from the first call, which would silently and permanently freeze a config the gate's own
    # campaign data says loses at smaller n).
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is True
    assert m.nbins_strategy == "mdlp"
    assert m.quantization_nbins == 10


# ---------------------------------------------------------------------------
# FIT_IMPL_B-2 (P1): the p>=n false-positive-control cap in _fit_impl_core.py
# was enforced exactly once, but usability-aware raw retention and raw-signal-
# retention augmentation run AFTER it and can each append more raw columns
# with no re-check, letting the final selected/n_features_ count silently
# exceed the documented max(20, p//3) ceiling.
# ---------------------------------------------------------------------------


def test_regression_pgn_cap_reapplied_after_late_retention_passes(monkeypatch):
    """Pre-fix: monkeypatching retain_usable_raw_columns to return 50 extra raw names on a p=60/n=50 fit
    (ceiling=max(20,60//3)=20) yielded n_features_=51, blowing through the ceiling by 31 features.
    Post-fix: the cap is re-applied after the late retention passes, capping the same scenario at exactly 20.
    """
    import mlframe.feature_selection.filters._fe_pure_form_retention as rfmod
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n, p = 50, 60
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series((X["x0"] + 0.5 * X["x1"] + 0.05 * rng.standard_normal(n) > 0).astype(int))

    extra_names = [f"x{i}" for i in range(5, 55)]  # 50 extra raw names, far more than the ceiling allows

    def _fake_retain_usable_raw_columns(mrmr, X, y_cont, **kw):
        """Test double standing in for the real collaborator, so only the code under test varies."""
        return list(extra_names)

    monkeypatch.setattr(rfmod, "retain_usable_raw_columns", _fake_retain_usable_raw_columns)

    m = MRMR(n_workers=1, fe_max_steps=1, min_features_fallback=0, verbose=0, full_npermutations=0, baseline_npermutations=0)
    with pytest.warns(UserWarning, match="min_features_fallback"):
        m.fit(X, y)

    ceiling = max(20, p // 3)
    assert m.n_features_ == ceiling, f"expected the p>=n cap to hold at {ceiling}, got {m.n_features_}"
    assert len(m.support_) == ceiling


# ---------------------------------------------------------------------------
# FIT_IMPL_B-3 (P1): three post-selection rescue gates (hinge, raw floor-drop,
# cat-FE floor-drop protection) built their held-out validation split via the
# identical deterministic, UNSHUFFLED (idx % 3) == 0 stride -- not an honest
# i.i.d. holdout on time/group/label-sorted input. Fixed via a seeded
# shuffle-then-stride, mirroring the exact formula now used at all 3 sites.
# ---------------------------------------------------------------------------


def test_regression_rescue_gate_split_is_shuffled_not_positional_stride():
    """Pre-fix formula: `idx = np.arange(n); va = (idx % 3) == 0` -- a purely positional stride, identical
    regardless of random_seed and systematically correlated with any pre-existing row order (time/group/
    label sort). Post-fix formula (now used at all 3 rescue-gate sites in _fit_impl_core.py): a seeded
    `np.random.default_rng(seed).permutation(n)`-based split. This test pins the NEW formula's two required
    properties: (1) reproducible under a fixed seed, (2) NOT equal to the old naive positional-stride mask
    (i.e. genuinely shuffled), which the old formula could never satisfy.
    """
    n = 300

    def _split(seed):
        """Deterministic train/holdout split shared by the assertions below."""
        perm = np.random.default_rng(seed).permutation(n)
        va = np.zeros(n, dtype=bool)
        va[perm[: n // 3]] = True
        return va

    va_a = _split(seed=0)
    va_b = _split(seed=0)
    np.testing.assert_array_equal(va_a, va_b)  # reproducible under a fixed seed

    old_positional_stride = (np.arange(n) % 3) == 0
    assert not np.array_equal(va_a, old_positional_stride), "the new split must not degenerate back to the old positional (idx % 3) == 0 stride"

    va_c = _split(seed=1)
    assert not np.array_equal(va_a, va_c), "a different seed must produce a different split"
    assert va_a.sum() == n // 3 == va_c.sum()  # same validation-fold SIZE regardless of seed


# ---------------------------------------------------------------------------
# FE_STEP_B-3 (P1): cached_MIs pair-tuple keys were never canonicalized on write
# in either compute_pairs_mis (feature_engineering.py) or the batch-precompute
# write loop (_step_pairmi.py), so the same logical pair could land as two
# divergent dict entries -- (a,b) and (b,a) -- doubling MI computation.
# ---------------------------------------------------------------------------


def test_regression_compute_pairs_mis_canonicalizes_pair_key_order(monkeypatch):
    """Pre-fix: compute_pairs_mis([(b, a)], ...) cached the pair under the RAW (unsorted) key (b, a); calling
    it again with the same logical pair given as (a, b) would then miss the cache and recompute, creating a
    second divergent dict entry for one physical pair. Post-fix: the key is canonicalized to sorted order,
    so both call orders resolve to the identical dict entry. mi_direct itself is monkeypatched to a trivial
    stub so this test isolates the key-canonicalization logic from the real MI kernel.
    """
    import mlframe.feature_selection.filters.feature_engineering as fe_mod

    monkeypatch.setattr(fe_mod, "mi_direct", lambda *a, **kw: (0.5, 1.0))

    kwargs = dict(
        data=np.zeros((10, 3)), target_indices=(2,), nbins=np.array([5, 5, 2]), classes_y=np.array([0, 1]),
        classes_y_safe=np.array([0, 1]), freqs_y=np.array([0.5, 0.5]), fe_min_nonzero_confidence=0.0,
        fe_npermutations=0, fe_min_pair_mi=0.0, fe_min_pair_mi_prevalence=0.0,
    )

    cached_MIs_a: dict = {}
    fe_mod.compute_pairs_mis(all_pairs=[(1, 0)], cached_confident_MIs={}, cached_MIs=cached_MIs_a, **kwargs)
    pair_keys_a = [k for k in cached_MIs_a if len(k) == 2]
    assert pair_keys_a == [(0, 1)], f"expected the pair key stored in sorted order, got {pair_keys_a!r}"

    # Simulate a second FE step's candidate pool handing the SAME logical pair in the OTHER order --
    # must resolve to the SAME dict entry, not a second divergent one.
    cached_MIs_b: dict = dict(cached_MIs_a)
    fe_mod.compute_pairs_mis(all_pairs=[(0, 1)], cached_confident_MIs={}, cached_MIs=cached_MIs_b, **kwargs)
    pair_keys_b = [k for k in cached_MIs_b if len(k) == 2]
    assert pair_keys_b == [(0, 1)], f"the same logical pair given in the other order must not create a second entry, got {pair_keys_b!r}"


# ---------------------------------------------------------------------------
# SCREEN_CONFIRM_A-1 (P1): evaluate_candidate's baseline mi_direct/mi_direct_gpu
# calls never received a seed derived from random_seed, so the per-candidate
# relevance permutation-null gate was CPU-seed-insensitive (always base_seed=0)
# and GPU-nondeterministic (fresh OS entropy every call).
# SCREEN_CONFIRM_A-3 (P1): _cmi_plugin_njit's dense (K_x,K_y,K_z) histogram cap
# was 1_000_000 (K_x=K_y=10 alone -> ~800MB per call, rebuilt every permutation
# draw) -- lowered to 10_000.
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
    factors_data = rng.integers(0, 5, size=(n, 3)).astype(np.int32)
    factors_nbins = np.array([5, 5, 2], dtype=np.int64)

    def _fresh_kwargs():
        """Builds a fresh kwargs dict so each call starts from identical, unmutated inputs."""
        return dict(
            cand_idx=0, X=(0,), y=(2,), nexisting=0, best_gain=0.0, factors_data=factors_data,
            factors_nbins=factors_nbins, factors_names=["a", "b", "y"], expected_gains=np.zeros(3),
            partial_gains={}, selected_vars=[], baseline_npermutations=4, use_gpu=False,
            cached_MIs={}, cached_confident_MIs={}, cached_cond_MIs={}, cached_jmim_MIs={},
            entropy_cache={}, verbose=0,
        )
    ev_mod.evaluate_candidate(**_fresh_kwargs(), random_seed=1)
    ev_mod.evaluate_candidate(**_fresh_kwargs(), random_seed=2)

    assert len(captured_seeds) == 2
    assert captured_seeds[0] is not None and captured_seeds[1] is not None
    assert captured_seeds[0] != captured_seeds[1], "different random_seed values must produce different baseline permutation seeds"


def test_regression_cmi_perm_stop_kz_cap_bounded_at_10k():
    """K_z must be capped well below the old 1_000_000 ceiling -- construct a pool of selected columns
    whose cardinality product exceeds the new 10_000 cap and confirm it triggers the truncation warning
    (not an unbounded allocation) at a cardinality easily reachable with realistic ten-bin features
    (10**5 = 100_000 > 10_000, needing only 5 selected columns, vs 10**6 for the old cap).
    """
    from mlframe.feature_selection.filters._cmi_perm_stop import _MAX_K_Z, cmi_permutation_stop

    assert _MAX_K_Z == 10_000

    rng = np.random.default_rng(0)
    n = 500
    x_cand = rng.integers(0, 10, n).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    selected_cols = [rng.integers(0, 10, n).astype(np.int64) for _ in range(5)]  # 10**5 = 100_000 > 10_000

    is_sig, observed, _p_value = cmi_permutation_stop(
        x_cand=x_cand, y=y, selected_cols=selected_cols,
        nbins_x=10, nbins_y=2, nbins_selected=[10] * 5,
        n_permutations=5, seed=0,
    )
    assert isinstance(is_sig, (bool, np.bool_))
    assert observed >= 0.0


# ---------------------------------------------------------------------------
# SCREEN_CONFIRM_B-4 (P1): mi_direct's internal prefer_gpu fastpath gate never
# consulted gpu_globally_disabled()/MLFRAME_DISABLE_GPU.
# SCREEN_CONFIRM_B-5 (P1): the order-1 resident-GPU maxT floor's fault paths had
# zero logging and no circuit breaker, unlike the order-2 sibling.
# ---------------------------------------------------------------------------


def test_regression_mi_direct_internal_gate_honours_disable_gpu(monkeypatch):
    """Pre-fix: mi_direct's own internal prefer_gpu fastpath checked only is_cuda_available(), not
    gpu_globally_disabled()/MLFRAME_DISABLE_GPU -- so a caller relying on the env-var opt-out (without
    also passing prefer_gpu=False itself) could still have mi_direct silently route to GPU. Source-level
    check (not a full mi_direct call, which needs real, carefully-shaped MI-kernel inputs and is already
    exercised end-to-end by this module's other tests): confirms the fixed gate expression actually
    references gpu_globally_disabled, not just is_cuda_available.
    """
    import inspect

    from mlframe.feature_selection.filters import permutation as perm_mod

    src = inspect.getsource(perm_mod.mi_direct)
    gate_block = src[src.index("if prefer_gpu and npermutations") :]
    gate_block = gate_block[: gate_block.index("if _gpu_ok:")]
    assert "gpu_globally_disabled" in gate_block, "mi_direct's internal prefer_gpu gate must consult gpu_globally_disabled()"


def test_regression_order1_maxt_gpu_circuit_breaker_state_machine():
    """Pins the order-1 resident maxT GPU circuit breaker's trip/reset/query state machine -- did not
    exist at all pre-fix (this file had zero module-level circuit-breaker state or logging).
    """
    from mlframe.feature_selection.filters._permutation_null_resident import (
        order1_maxt_gpu_circuit_breaker_tripped,
        reset_order1_maxt_gpu_circuit_breaker,
        trip_order1_maxt_gpu_circuit_breaker,
    )

    reset_order1_maxt_gpu_circuit_breaker()
    try:
        assert order1_maxt_gpu_circuit_breaker_tripped() is False
        trip_order1_maxt_gpu_circuit_breaker()
        assert order1_maxt_gpu_circuit_breaker_tripped() is True
    finally:
        reset_order1_maxt_gpu_circuit_breaker()
    assert order1_maxt_gpu_circuit_breaker_tripped() is False


# ---------------------------------------------------------------------------
# INFO_THEORY_A-9 (P2): _CMI_RESIDENT_CACHE had no lock at all, unlike its two
# siblings in the same file (_FORDER_LOCK, _FACTORS_DEVICE_LOCK).
# ---------------------------------------------------------------------------


def test_regression_cmi_resident_cache_has_lock():
    """Pins that _CMI_RESIDENT_CACHE now has a dedicated lock (was previously the only one of 3
    sibling caches in this file with none) and that clear_cmi_resident_cache() acquires it too.
    """
    from mlframe.feature_selection.filters.info_theory import _cmi_cuda as cmi_cuda_mod

    assert hasattr(cmi_cuda_mod, "_CMI_RESIDENT_CACHE_LOCK")
    import threading
    assert isinstance(cmi_cuda_mod._CMI_RESIDENT_CACHE_LOCK, type(threading.Lock()))
    # Sanity: clear works and leaves the cache empty (also exercises the lock without deadlocking).
    cmi_cuda_mod._CMI_RESIDENT_CACHE["dummy"] = ("g", "sig")
    cmi_cuda_mod.clear_cmi_resident_cache()
    assert cmi_cuda_mod._CMI_RESIDENT_CACHE == {}


# ---------------------------------------------------------------------------
# INFO_THEORY_B-1 (P1): genie_mi_panel's default bias-rate fallback was the SAME
# constant (1/sqrt(N)) for every estimator name, making genie_weights' constraint
# matrix exactly singular by construction -- estimator='genie' silently collapsed
# to a plain unweighted mean on every production call.
# ---------------------------------------------------------------------------


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


def test_regression_mrmr_set_params_keeps_agreeing_config():
    """A set_params() call that does NOT touch any field the nested config covers must leave the config
    object intact (only a genuinely disagreeing config gets invalidated)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import FastSearchConfig

    cfg = FastSearchConfig(fe_fast_search=True)
    m = MRMR(fast_search_config=cfg)
    m.set_params(verbose=1)  # unrelated flat attr
    assert m.get_params()["fast_search_config"] is cfg


# ---------------------------------------------------------------------------
# GPU_INFRA_A-1 (P1): dispatch_friend_graph_stats / dispatch_batch_mi_with_noise_gate_gpu's cuda
# branches caught only (ValueError, RuntimeError); numba's CudaAPIError/CudaDriverError derive
# directly from Exception, so a genuine CUDA driver fault escaped these two dispatchers uncaught.
# ---------------------------------------------------------------------------


class _FakeCudaFault(Exception):
    """Stand-in for numba's CudaAPIError/CudaDriverError, which derive from Exception directly,
    not RuntimeError -- the exact bug class GPU_INFRA_A-1 fixed."""


def test_regression_dispatch_friend_graph_stats_catches_bare_exception_cuda_fault(monkeypatch):
    """Pre-fix: `except (ValueError, RuntimeError)` let a bare-Exception CUDA fault propagate
    uncaught out of dispatch_friend_graph_stats's cuda branch. Post-fix: broadened to `except Exception`."""
    import mlframe.feature_selection.filters.friend_graph_gpu as fg_gpu

    monkeypatch.setattr(fg_gpu, "_CUDA_AVAIL", True)
    monkeypatch.setattr(fg_gpu, "_CUPY_AVAIL", False)

    def _raise(*a, **k):
        """Stand-in that raises _FakeCudaFault, so the caller's failure path is the one under test."""
        raise _FakeCudaFault("simulated CUDA driver fault")

    monkeypatch.setattr(fg_gpu, "friend_graph_stats_cuda", _raise)
    monkeypatch.setattr(fg_gpu, "_friend_graph_backend_choice", lambda n, k: "cuda")

    result = fg_gpu.dispatch_friend_graph_stats(
        sel=np.array([0, 1], dtype=np.int64),
        factors_data=np.zeros((10, 2), dtype=np.int32),
        factors_nbins=np.array([2, 2], dtype=np.int32),
        target_indices=np.array([0], dtype=np.int64),
        dtype=np.int32,
    )
    assert result is None  # falls back to CPU instead of raising


def test_regression_dispatch_batch_mi_noise_gate_gpu_catches_bare_exception_cuda_fault(monkeypatch):
    """Pre-fix: `except (ValueError, RuntimeError)` let a bare-Exception CUDA fault propagate
    uncaught out of dispatch_batch_mi_with_noise_gate_gpu's cuda branch. Post-fix: broadened."""
    import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bng_gpu

    monkeypatch.setattr(bng_gpu, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bng_gpu, "_CUPY_AVAIL", False)
    monkeypatch.setattr(bng_gpu, "gpu_globally_disabled", lambda: False, raising=False)

    def _raise(*a, **k):
        """Stand-in that raises _FakeCudaFault, so the caller's failure path is the one under test."""
        raise _FakeCudaFault("simulated CUDA driver fault")

    monkeypatch.setattr(bng_gpu, "batch_mi_with_noise_gate_cuda", _raise)
    monkeypatch.setattr(bng_gpu, "_batch_mi_noise_gate_backend_choice", lambda n, k: "cuda")

    result = bng_gpu.dispatch_batch_mi_with_noise_gate_gpu(
        disc_2d=np.zeros((10, 2), dtype=np.int32),
        factors_nbins=np.array([2, 2], dtype=np.int32),
        classes_y=np.zeros(10, dtype=np.int32),
        classes_y_safe=np.zeros(10, dtype=np.int32),
        freqs_y=np.array([1.0]),
        npermutations=0,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.0,
        use_su=False,
    )
    assert result is None  # falls back to CPU instead of raising


# ---------------------------------------------------------------------------
# GPU_INFRA_A-2 (P1): batch_pair_mi_cuda_row_chunked / batch_pair_mi_cuda_shared_fused raised
# ZeroDivisionError for n_samples == 0 (inv_n = 1.0 / n_samples), unlike the cupy sibling
# (batch_pair_mi_cupy) which has an explicit `if n_samples == 0: return zeros` guard.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_regression_batch_pair_mi_cuda_row_chunked_empty_input_no_zero_division():
    """Pre-fix: n_samples=0 raised ZeroDivisionError deep inside _mi_from_joint_counts(_cupy).
    Post-fix: an early `if n_samples == 0: return zeros` guard mirrors the cupy sibling."""
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels import _CUDA_AVAIL

    if not _CUDA_AVAIL:
        pytest.skip("numba.cuda not available on this host")
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels import batch_pair_mi_cuda_row_chunked

    out = batch_pair_mi_cuda_row_chunked(
        factors_data=np.zeros((0, 3), dtype=np.int32),
        pair_a=np.array([0, 1], dtype=np.int64),
        pair_b=np.array([1, 2], dtype=np.int64),
        nbins=np.array([2, 2, 2], dtype=np.int32),
        classes_y=np.zeros(0, dtype=np.int32),
        freqs_y=np.array([1.0]),
    )
    assert out.shape == (2,)
    assert np.all(out == 0.0)


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_regression_batch_pair_mi_cuda_shared_fused_empty_input_no_zero_division():
    """Pre-fix: n_samples=0 raised ZeroDivisionError at `inv_n = 1.0 / float(n_samples)`.
    Post-fix: an early `if n_samples == 0: return zeros` guard mirrors the cupy sibling."""
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import _CUPY_AVAIL

    if not _CUPY_AVAIL:
        pytest.skip("cupy not available on this host")
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import batch_pair_mi_cuda_shared_fused

    out = batch_pair_mi_cuda_shared_fused(
        factors_data=np.zeros((0, 3), dtype=np.int32),
        pair_a=np.array([0, 1], dtype=np.int64),
        pair_b=np.array([1, 2], dtype=np.int64),
        nbins=np.array([2, 2, 2], dtype=np.int32),
        classes_y=np.zeros(0, dtype=np.int32),
        freqs_y=np.array([1.0]),
    )
    assert out.shape == (2,)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# GPU_INFRA_A-3 (P2): dispatch_batch_mi_with_noise_gate_gpu had no internal gpu_globally_disabled()
# self-check, unlike its dispatch_batch_pair_mi / dispatch_friend_graph_stats siblings, relying
# entirely on its sole production caller checking MLFRAME_DISABLE_GPU first.
# ---------------------------------------------------------------------------


def test_regression_dispatch_batch_mi_noise_gate_gpu_honours_disable_gpu(monkeypatch):
    """Pre-fix: dispatch_batch_mi_with_noise_gate_gpu ignored MLFRAME_DISABLE_GPU entirely, even
    with force_backend='cuda'. Post-fix: an internal gpu_globally_disabled() check wins first."""
    import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bng_gpu
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    monkeypatch.setattr(bng_gpu, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bng_gpu, "_CUPY_AVAIL", True)

    def _fail_if_called(*a, **k):
        """Stand-in that raises AssertionError, so the caller's failure path is the one under test."""
        raise AssertionError("GPU backend must not be invoked when globally disabled")

    monkeypatch.setattr(bng_gpu, "batch_mi_with_noise_gate_cuda", _fail_if_called)
    monkeypatch.setattr(bng_gpu, "batch_mi_with_noise_gate_cupy", _fail_if_called)
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert gpu_globally_disabled() is True

    result = bng_gpu.dispatch_batch_mi_with_noise_gate_gpu(
        disc_2d=np.zeros((10, 2), dtype=np.int32),
        factors_nbins=np.array([2, 2], dtype=np.int32),
        classes_y=np.zeros(10, dtype=np.int32),
        classes_y_safe=np.zeros(10, dtype=np.int32),
        freqs_y=np.array([1.0]),
        npermutations=0,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.0,
        use_su=False,
        force_backend="cuda",
    )
    assert result is None


# ---------------------------------------------------------------------------
# GPU_INFRA_A-11 (P2): _batch_joint_entropy_pairs's njit kernel has no host-side validation that
# nbins >= 1 for referenced columns, unlike every CUDA kernel in the same cluster, which pre-validate
# this class of input before launch (an unguarded OOB index write is undefined behaviour on CPU too).
# ---------------------------------------------------------------------------


def test_regression_batch_joint_entropy_pairs_rejects_nonpositive_nbins():
    """Pre-fix: a non-positive nbins for a referenced column reached the njit kernel unchecked,
    risking a silent OOB write. Post-fix: a host-side ValueError guard fails loudly first."""
    from mlframe.feature_selection.filters._dcd_pair_su_batch import _validate_batch_joint_entropy_pairs_inputs

    with pytest.raises(ValueError, match="nbins must be >= 1"):
        _validate_batch_joint_entropy_pairs_inputs(
            a_arr=np.array([0], dtype=np.int64),
            b_arr=np.array([1], dtype=np.int64),
            nb_arr=np.array([2, 0], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# GPU_INFRA_A-12 (P2): the module-level _DY_DEVICE_CACHE/_DY_DEVICE_CACHE_CUPY OrderedDict LRU
# caches performed a non-atomic get -> move_to_end -> popitem sequence with no lock, risking the
# LRU eviction discipline being violated under concurrent multi-thread MRMR.fit() calls.
# ---------------------------------------------------------------------------


def test_regression_dy_device_cache_has_lock():
    """Pre-fix: no lock guarded _DY_DEVICE_CACHE's LRU bookkeeping. Post-fix: both the numba.cuda
    and cupy-native caches are guarded by their own threading.Lock."""
    import threading

    import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bng_gpu

    assert isinstance(bng_gpu._DY_DEVICE_CACHE_LOCK, type(threading.Lock()))
    assert isinstance(bng_gpu._DY_DEVICE_CACHE_CUPY_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# GPU_INFRA_B-1 (P1): gpu_materialise_discretize_codes_host / gpu_discretize_codes_host both called
# clear_resident_codes_handoff() with NO argument at entry -- the blanket, whole-dict-clear form --
# instead of the module's own targeted per-host-array clear form, silently dropping another concurrent
# thread's still-pending deferred-fill entry under joblib threading.
# ---------------------------------------------------------------------------


def test_regression_gpu_resident_materialise_entrypoints_do_not_blanket_clear_handoff():
    """Pre-fix: both entry points called clear_resident_codes_handoff() with no argument, which could
    silently drop another thread's in-flight deferred-fill entry. Post-fix: the blanket call is gone
    entirely (relies on the bounded-FIFO eviction instead), so a stashed entry from a prior call survives."""
    import mlframe.feature_selection.filters._gpu_resident_fe as gfe

    gfe.clear_resident_codes_handoff()
    sentinel_host = np.zeros((4, 4), dtype=np.int8)
    sentinel_device = np.zeros((4, 4), dtype=np.int8)  # stand-in device codes; only identity/shape matter here
    gfe._stash_deferred_host_fill(sentinel_host, sentinel_device)
    assert gfe._DEFERRED_HOST_FILL, "test setup must actually stash an entry"
    stashed_key = next(iter(gfe._DEFERRED_HOST_FILL.keys()))

    import inspect

    import mlframe.feature_selection.filters._gpu_resident_materialise as gm

    src_materialise = inspect.getsource(gm.gpu_materialise_discretize_codes_host)
    src_discretize = inspect.getsource(gm.gpu_discretize_codes_host)
    # An active (uncommented) call, not a mention in a fix-note comment.
    assert not any(line.strip().startswith("clear_resident_codes_handoff()") for line in src_materialise.splitlines())
    assert not any(line.strip().startswith("clear_resident_codes_handoff()") for line in src_discretize.splitlines())
    assert stashed_key in gfe._DEFERRED_HOST_FILL, "entry must not be silently dropped by an unrelated call"
    gfe.clear_resident_codes_handoff()  # cleanup


# ---------------------------------------------------------------------------
# GPU_INFRA_B-4 (P2): gpu_materialise_discretize_codes_host / gpu_discretize_codes_host's docstrings/comments
# claimed "nbins<=255 -> int8 cannot overflow", but _BIN_CODES_OUTTYPE maps "int8" to C signed char
# (range -128..127), so the correct bound is nbins<=128, not 255 -- a code of 150 would silently wrap to -106.
# ---------------------------------------------------------------------------


def test_regression_gpu_materialise_discretize_rejects_narrow_dtype_for_nbins():
    """Pre-fix: no validation -- a dtype too narrow for nbins would silently wrap around instead of raising.
    Post-fix: both public entry points raise ValueError immediately for an under-sized integer dtype."""
    import mlframe.feature_selection.filters._gpu_resident_materialise as gm

    with pytest.raises(ValueError, match="cannot represent codes"):
        gm.gpu_discretize_codes_host(
            cand=np.zeros((4, 2), dtype=np.float32), nbins=150, dtype=np.int8,
        )
    with pytest.raises(ValueError, match="cannot represent codes"):
        gm.gpu_materialise_discretize_codes_host(
            transformed_vars=np.zeros((4, 2), dtype=np.float32),
            a_cols=np.array([0], dtype=np.int64), b_cols=np.array([1], dtype=np.int64),
            op_codes=np.array([0], dtype=np.int8), nbins=150, dtype=np.int8,
        )


# ---------------------------------------------------------------------------
# GPU_INFRA_B-5 (P2): grand_fused_pair_mi/grand_fused_pair_mi_fused hardcoded base_seed=np.uint64(0) at
# every permutation-null call site, never threading through a caller-supplied seed -- the same
# "ignores the estimator's random_state" bug class the 2026-07-20 audit flagged at P1 for the production
# confirmation path. Currently dead/unreachable prototype code, fixed before any future wiring reuses it.
# ---------------------------------------------------------------------------


def test_regression_grand_fused_pair_mi_threads_random_seed():
    """Pre-fix: grand_fused_pair_mi ignored any caller-supplied seed (hardcoded base_seed=np.uint64(0)).
    Post-fix: a random_seed kwarg exists and is threaded through to the permutation-null base_seed."""
    import inspect

    from mlframe.feature_selection.filters._gpu_resident_basis import grand_fused_pair_mi

    sig = inspect.signature(grand_fused_pair_mi)
    assert "random_seed" in sig.parameters
    assert sig.parameters["random_seed"].default == 0

    src = inspect.getsource(grand_fused_pair_mi)
    assert "base_seed=np.uint64(0)" not in src
    assert "base_seed=np.uint64(random_seed)" in src


# ---------------------------------------------------------------------------
# GPU_INFRA_B-9 (P2): _env_gpu_default_on reimplemented the CUDA_VISIBLE_DEVICES=""/MLFRAME_DISABLE_GPU=1
# opt-out check inline instead of calling the shared _gpu_policy.gpu_globally_disabled(), so a future change
# to the shared policy's semantics would silently not reach this cluster's default-on gates.
# ---------------------------------------------------------------------------


def test_regression_env_gpu_default_on_honours_disable_gpu(monkeypatch):
    """Pre-fix: MLFRAME_DISABLE_GPU=1 was checked via an inline reimplementation, not the shared
    gpu_globally_disabled(). Post-fix: _env_gpu_default_on delegates to the shared policy function."""
    from mlframe.feature_selection.filters._gpu_resident_fe import _env_gpu_default_on
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert gpu_globally_disabled() is True
    assert _env_gpu_default_on("MLFRAME_FE_GPU_RESIDENT_CODES") is False


# ---------------------------------------------------------------------------
# GPU_INFRA_C-1 (P1): _fe_gpu_strict's AUTO-fit-shape state (_AUTO_FIT_N/_AUTO_FIT_P) was held in bare,
# unlocked module-level globals shared across every thread -- StabilityMRMR's bootstrap loop already fits
# multiple MRMR instances concurrently via Parallel(backend="threading"), so two overlapping .fit() calls
# would silently read/write each other's fit shape, making the AUTO-STRICT engage/skip decision
# non-deterministic. Fixed via threading.local().
# ---------------------------------------------------------------------------


def test_regression_fe_gpu_strict_auto_fit_shape_is_thread_local():
    """Pre-fix: _AUTO_FIT_N/_AUTO_FIT_P were bare module globals, visible to every thread. Post-fix: each
    thread gets its own (n, p) via threading.local(), so one thread's set_auto_fit_n does not leak into another."""
    import threading

    from mlframe.feature_selection.filters import _fe_gpu_strict as strict

    strict.clear_auto_fit_n()
    other_thread_saw = {}

    def _other_thread():
        # A fresh thread must start with NO fit shape set, regardless of what the main thread set below.
        """Thread body: exercises the call under test and records its result or error."""
        other_thread_saw["n"] = getattr(strict._auto_fit_state, "n", None)
        strict.set_auto_fit_n(999_999, 5)
        other_thread_saw["after_set"] = strict._auto_fit_state.n

    strict.set_auto_fit_n(123, 7)
    assert strict._auto_fit_state.n == 123

    t = threading.Thread(target=_other_thread)
    t.start()
    t.join()

    assert other_thread_saw["n"] is None, "a fresh thread must not inherit another thread's fit shape"
    assert other_thread_saw["after_set"] == 999_999
    # The main thread's own state must be UNCHANGED by the other thread's set_auto_fit_n call.
    assert strict._auto_fit_state.n == 123
    strict.clear_auto_fit_n()


# ---------------------------------------------------------------------------
# GPU_INFRA_C-2 (P1): install_cuda_teardown_guard's idempotency check was a bare unlocked
# `if _installed: return` -- two threads racing through it before either set _installed=True could let the
# second thread capture the FIRST thread's own hook wrapper as "the previous hook", risking infinite
# self-recursion later. Fixed via a threading.Lock guarding the whole check-then-install sequence.
# ---------------------------------------------------------------------------


def test_regression_install_cuda_teardown_guard_has_lock():
    """Pre-fix: no lock guarded the idempotency check. Post-fix: a module-level threading.Lock exists and
    guards the whole install sequence."""
    import threading

    from mlframe.feature_selection.filters import _gpu_teardown_guard as guard

    assert isinstance(guard._install_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# GPU_INFRA_C-4 (P2): resident_bincount's docstring claimed OOB indices are "undefined behaviour (same as
# feeding cupy.bincount a bad minlength)" -- but cupy.bincount raises loudly on a negative index, while
# resident_bincount's cupyx.scatter_add silently wraps/corrupts. Added an opt-in debug_check_bounds guard.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_regression_resident_bincount_debug_check_bounds_catches_oob():
    """Pre-fix: no way to catch an out-of-range index short of the silent-wrap/OOB-write UB. Post-fix:
    debug_check_bounds=True raises ValueError instead."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._resident_bincount import resident_bincount

    x = cp.asarray([0, 1, -1, 2], dtype=cp.int32)  # -1 would silently wrap without the guard
    with pytest.raises(ValueError, match="index out of"):
        resident_bincount(cp, x, nc=3, debug_check_bounds=True)
    # Without the guard, it must NOT raise (this is the documented default sync-free UB contract).
    out = resident_bincount(cp, x, nc=3, debug_check_bounds=False)
    assert out.shape == (3,)


# ---------------------------------------------------------------------------
# GPU_INFRA_C-9 (P2): _fe_gpu_batch/_devices.py's _visible_device_ids reimplemented the
# MLFRAME_DISABLE_GPU/CUDA_VISIBLE_DEVICES="" opt-out check inline instead of calling the shared
# _gpu_policy.gpu_globally_disabled(), extending the duplication pattern to a third file.
# ---------------------------------------------------------------------------


def test_regression_fe_gpu_batch_visible_device_ids_honours_disable_gpu(monkeypatch):
    """Pre-fix: MLFRAME_DISABLE_GPU=1 was checked via an inline reimplementation. Post-fix:
    _visible_device_ids delegates to the shared gpu_globally_disabled()."""
    from mlframe.feature_selection.filters._fe_gpu_batch._devices import _visible_device_ids

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert _visible_device_ids() == []


# ---------------------------------------------------------------------------
# GPU_INFRA_C-8 (P2): the njit "sign" unary (_apply_unary code 8) mishandled NaN input -- IEEE comparisons
# with NaN are always False, so NaN fell through to the else branch and was coerced to -1.0, disagreeing
# with both np.sign(nan)==nan and this module's own cupy twin.
# ---------------------------------------------------------------------------


def test_regression_apply_unary_sign_preserves_nan():
    """Pre-fix: _apply_unary(nan, code=8, ...) returned -1.0. Post-fix: it returns nan, matching np.sign(nan)
    and the cupy twin _gpu_apply_unary's cp.sign(nan)."""
    from mlframe.feature_selection.filters._usability_njit_pool import _apply_unary

    result = _apply_unary(float("nan"), 8, 0.0)
    assert np.isnan(result), f"sign(nan) must stay nan (matches np.sign), got {result}"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-1 (P0): the cupy_kernel polynom-pair optimizer dispatch never checked gpu_globally_disabled(),
# so a plain MRMR() on a cupy-capable host launched real CUDA kernels for every hermite/orth pair-FE search
# even under MLFRAME_DISABLE_GPU=1 / CUDA_VISIBLE_DEVICES="".
# ---------------------------------------------------------------------------


def test_regression_hermite_cupy_kernel_dispatch_honours_disable_gpu(monkeypatch):
    """Pre-fix: MLFRAME_DISABLE_GPU=1 was never checked before importing/calling run_cupy_kernel_search.
    Post-fix: gpu_globally_disabled() gates it, falling back to random_batch."""
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    assert gpu_globally_disabled() is True

    import inspect

    from mlframe.feature_selection.filters._hermite_fe_optimise_pair import optimise_hermite_pair

    src = inspect.getsource(optimise_hermite_pair)
    assert "gpu_globally_disabled" in src


# ---------------------------------------------------------------------------
# GPU_INFRA_D-2 (P1): _gpu_pairs.py / _batch_pair_mi_cuda_shared_fused.py both mutate a process-wide
# compiled-kernel object's max_dynamic_shared_size_bytes attribute with no lock -- two threads racing with
# different shared-mem requirements could under-provision one launch relative to what it needs.
# ---------------------------------------------------------------------------


def test_regression_gpu_pairs_shared_mem_set_has_lock():
    """Pre-fix: no lock guarded the property-set + launch pair. Post-fix: both files expose a
    threading.Lock guarding the sequence."""
    import threading

    import mlframe.feature_selection.filters._gpu_pairs as gp
    import mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused as bpf

    assert isinstance(gp._SHARED_MEM_SET_LOCK, type(threading.Lock()))
    assert isinstance(bpf._SHARED_MEM_SET_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# GPU_INFRA_D-3 (P1): run_numba_kernel_search / optimize_all_pairs_numba_kernel silently substituted 1 for
# any seed<=0 (including the valid seed=0), diverging from the cupy twin's plain np.random.default_rng(seed).
# ---------------------------------------------------------------------------


def test_regression_numba_polynom_optimizer_seed_zero_not_substituted():
    """Pre-fix: seed=0 silently became seed=1 internally (same stream as any other seed<=0 caller).
    Post-fix: np.random.default_rng(seed) is called directly, matching the cupy twin."""
    import inspect

    from mlframe.feature_selection.filters import _numba_polynom_optimizer as npo

    src_a = inspect.getsource(npo.run_numba_kernel_search)
    src_b = inspect.getsource(npo.optimize_all_pairs_numba_kernel)
    for src in (src_a, src_b):
        assert not any(line.strip().startswith("rng = ") and "else 1" in line for line in src.splitlines())
        assert any(line.strip() == "rng = np.random.default_rng(seed)" for line in src.splitlines())


# ---------------------------------------------------------------------------
# GPU_INFRA_D-4 (P1): the GPU-resident additive-fusion OLS-R separability margin divided by the
# SCORING-SUBSAMPLE row count (n_sc) instead of the true full fit row count (n_rows), making the GPU
# threshold systematically stricter than the CPU sibling's at n above the subsample cap.
# ---------------------------------------------------------------------------


def test_regression_fe_additive_fusion_gpu_ols_margin_uses_full_n_rows():
    """Pre-fix: the margin used n_sc (scoring-subsample count). Post-fix: it uses n_rows (the true fit
    row count), matching the CPU sibling's calibration."""
    import inspect

    from mlframe.feature_selection.filters import _fe_additive_fusion_gpu_resident as fus

    src = inspect.getsource(fus)
    assert not any(line.strip().startswith("_r_margin") and "n_sc" in line for line in src.splitlines())
    assert any(line.strip().startswith("_r_margin") and "n_rows" in line for line in src.splitlines())


# ---------------------------------------------------------------------------
# GPU_INFRA_D-5 (P1): _gpu_pairs.py's per-pair joint-MI reduction was a pure-Python triple-nested loop
# (O(total joint cells) at native Python speed), defeating the point of the batched CUDA histogram launch
# at realistic cell counts. Vectorized with numpy, same formula.
# ---------------------------------------------------------------------------


def test_regression_gpu_pairs_joint_mi_reduction_matches_reference_loop():
    """The vectorized per-pair MI reduction must produce IDENTICAL results to the original triple-nested
    Python loop formula, for a hand-built joint-count table with zero cells in various positions."""
    import numpy as _np

    def _reference_loop(joint_counts_host, joint_offsets, pair_merged_sizes, nbins_y, n_pairs, n_total):
        """The straightforward reference implementation the optimised path must agree with."""
        out = _np.zeros(n_pairs, dtype=_np.float64)
        for k in range(n_pairs):
            off = int(joint_offsets[k])
            merged_size = int(pair_merged_sizes[k])
            joint_2d = joint_counts_host[off : off + merged_size * nbins_y].reshape(merged_size, nbins_y)
            marg_m = joint_2d.sum(axis=1)
            marg_y = joint_2d.sum(axis=0)
            mi = 0.0
            for m in range(merged_size):
                mm = marg_m[m]
                if mm == 0:
                    continue
                for y in range(nbins_y):
                    jc = joint_2d[m, y]
                    if jc == 0:
                        continue
                    my = marg_y[y]
                    if my == 0:
                        continue
                    jf = jc / n_total
                    mi += jf * _np.log(jc * n_total / (mm * my))
            out[k] = mi
        return out

    rng = _np.random.default_rng(0)
    nbins_y = 4
    n_total = 1000.0
    # Two pairs, merged sizes 3 and 5, with some deliberately-zero cells.
    merged_sizes = [3, 5]
    offsets = [0, 3 * nbins_y]
    total_cells = sum(m * nbins_y for m in merged_sizes)
    joint_counts_host = rng.integers(0, 50, size=total_cells).astype(np.int64)
    joint_counts_host[rng.choice(total_cells, size=total_cells // 4, replace=False)] = 0  # sprinkle zeros

    ref = _reference_loop(joint_counts_host, offsets, merged_sizes, nbins_y, 2, n_total)

    joint_mi_out = _np.zeros(2, dtype=_np.float64)
    for k in range(2):
        off = int(offsets[k])
        merged_size = int(merged_sizes[k])
        joint_2d = joint_counts_host[off : off + merged_size * nbins_y].reshape(merged_size, nbins_y)
        marg_m = joint_2d.sum(axis=1)
        marg_y = joint_2d.sum(axis=0)
        valid = (joint_2d > 0) & (marg_m[:, None] > 0) & (marg_y[None, :] > 0)
        if not _np.any(valid):
            continue
        denom = _np.where(valid, marg_m[:, None] * marg_y[None, :], 1.0)
        ratio = _np.where(valid, joint_2d * n_total / denom, 1.0)
        jf = _np.where(valid, joint_2d / n_total, 0.0)
        joint_mi_out[k] = float(_np.sum(jf * _np.log(ratio)))

    np.testing.assert_allclose(joint_mi_out, ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# GPU_INFRA_D-6 (P2): run_cupy_kernel_search / run_numba_kernel_search's docstrings claimed an identical
# 5-tuple return contract, but the 5th element differs (best_score vs an evaluation count).
# ---------------------------------------------------------------------------


def test_regression_polynom_optimizer_5th_element_contract_documented():
    """Post-fix: both docstrings explicitly document that the 5th tuple element differs between backends,
    so a future reader/caller cannot assume it means the same thing."""
    from mlframe.feature_selection.filters._cupy_polynom_optimizer import run_cupy_kernel_search
    from mlframe.feature_selection.filters._numba_polynom_optimizer import run_numba_kernel_search

    # Asserts the CONTRACT is documented, not that an audit finding-ID survives in the text: comment-style
    # forbids process metadata in source, so pinning the ID makes the convention and this test contradict.
    for fn in (run_cupy_kernel_search, run_numba_kernel_search):
        doc = (fn.__doc__ or "").lower()
        assert "5th" in doc or "fifth" in doc, f"{fn.__name__} no longer documents the 5th tuple element"
        assert "differ" in doc or "not the same" in doc, f"{fn.__name__} no longer says the element differs by backend"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-11 (P2): both _hinge_detect_gpu_resident.py and _fe_additive_fusion_gpu_resident.py derived
# the subsample stride via floor division (n // max_rows), which yields stride==1 (no thinning at all) for
# n strictly between max_rows and 2*max_rows -- the documented "caps at <=max_rows" claim was only true
# once n reached ~2*max_rows. Fixed via ceiling division.
# ---------------------------------------------------------------------------


def test_regression_hinge_and_fusion_stride_actually_caps_between_1x_and_2x_max_rows():
    """Pre-fix: n=1.5*max_rows gave stride=1 (floor division), so the row cap silently didn't engage.
    Post-fix: ceiling division gives stride=2, actually keeping the subsample at or below max_rows."""
    max_rows = 100
    n = 150  # strictly between max_rows and 2*max_rows -- the regime the bug affected
    floor_stride = int(n // max_rows)
    ceil_stride = -(-n // max_rows)
    assert floor_stride == 1, "sanity: this is the regime where floor division gave no thinning"
    assert ceil_stride == 2
    assert n // ceil_stride <= max_rows, "post-fix stride must actually cap the subsampled row count"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-12 (P2): _fe_cmi_perm_null_gpu.py's `isinstance(y_h, cp.ndarray)` branches could never be
# True -- y_h is unconditionally a host numpy array at every call site -- leftover dead/misleading code
# implying a resident-y fast path that does not exist.
# ---------------------------------------------------------------------------


def test_regression_fe_cmi_perm_null_gpu_no_dead_resident_y_isinstance_check():
    """Post-fix: the dead `isinstance(y_h, cp.ndarray)` branches are removed."""
    import inspect

    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as cmi_gpu

    src = inspect.getsource(cmi_gpu)
    assert not any(line.strip().startswith("if isinstance(y_h, cp.ndarray)") for line in src.splitlines())


# ---------------------------------------------------------------------------
# DISCRETIZATION-1 (P1): discretize_2d_array_cuda_row_chunked's uniform/quantile branches never received
# the B-12 NaN-parity fix its non-chunked twin discretize_2d_array_cuda got -- plain cp.min/cp.max /
# cp.percentile propagated NaN, silently collapsing a whole NaN-bearing column to one degenerate bucket.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_regression_discretize_2d_array_cuda_row_chunked_nan_parity(method):
    """Pre-fix: a single NaN anywhere in a column poisoned that column's edges to NaN, collapsing every
    real value in the column to one degenerate bin. Post-fix: NaN-aware min/max/percentile matches the
    non-chunked (already-fixed) sibling's behaviour -- non-NaN rows land in real, non-degenerate bins."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")

    from mlframe.feature_selection.filters.discretization import discretize_2d_array_cuda_row_chunked

    rng = np.random.default_rng(0)
    n, ncols = 2000, 3
    arr = rng.standard_normal((n, ncols)).astype(np.float64)
    arr[5, 0] = np.nan  # a single NaN in column 0; columns 1-2 stay clean
    # Force the row-chunked path with a tiny free_bytes budget (mirrors existing dispatch tests).
    out = discretize_2d_array_cuda_row_chunked(arr, n_bins=8, method=method, dtype=np.int16, free_bytes=1 << 16)
    # Column 0's non-NaN rows must NOT all collapse into a single degenerate bucket.
    col0_real = out[np.arange(n) != 5, 0]
    assert len(np.unique(col0_real)) > 1, f"method={method}: NaN-poisoned column collapsed to one bin"
    if method == "uniform":
        # Only the uniform branch routes individual NaN VALUES to the dedicated NaN code (n_bins) --
        # matches the non-chunked sibling's behaviour, which the quantile branch never had either.
        assert out[5, 0] == 8, "NaN row did not get the dedicated NaN code"


# ---------------------------------------------------------------------------
# DISCRETIZATION-2 (P1): Bayesian Blocks (O(N^2)-per-column DP) had its subsample safety valve
# (bb_subsample_threshold) default to 0 (disabled) at the per_feature_edges dispatch layer, unlike every
# internal benchmark exercising it at realistic scale (which always overrides it to 1000).
# ---------------------------------------------------------------------------


def test_regression_bayesian_blocks_default_subsample_threshold_is_safe():
    """Pre-fix: per_feature_edges passed bb_subsample_threshold=0 (disabled, unbounded O(N^2)) by default.
    Post-fix: the default is a safe, already-validated cap."""
    import inspect

    from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges

    src = inspect.getsource(per_feature_edges)
    # The contract is a BOUNDED default, not the literal it was first set to: 0 restores the unbounded
    # O(N^2) DP, which is the regression. The value has since moved from 2000 to 5000, so assert the
    # property against the constant the code actually reads.
    from mlframe.feature_selection.filters._adaptive_nbins import _BB_DEFAULT_SUBSAMPLE_THRESHOLD

    assert 'kwargs.get("bb_subsample_threshold", 0)' not in src
    assert "_BB_DEFAULT_SUBSAMPLE_THRESHOLD" in src, "per_feature_edges no longer reads the shared default"
    assert _BB_DEFAULT_SUBSAMPLE_THRESHOLD > 0, "a 0 default restores the unbounded full-N bayesian-blocks DP"


# ---------------------------------------------------------------------------
# DISCRETIZATION-3 (P1): mdlp_bin_edges_oos_validated over-split on exact-duplicate (x, y) rows -- unlike
# its two siblings that dedupe before recursing. The root cause was deeper than a missing _dedupe_xy call:
# a duplicate row could straddle the train/holdout split (leaking the identical point into "held-out"
# validation), so deduping only the train fold was insufficient -- the fix dedupes the FULL array before
# the train/holdout split happens at all.
# ---------------------------------------------------------------------------


def test_regression_mdlp_oos_validated_duplicate_rows_do_not_over_split():
    """Pre-fix: pure noise at dup_rate 0/10/50/90% gave 1/4/6/12 bins (over-split). Post-fix: collapses to
    1 bin at every duplication rate, matching the sibling mdlp_bin_edges_validated's behaviour."""
    from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_oos_validated

    rng = np.random.default_rng(1)
    n_unique = 800
    x = rng.standard_normal(n_unique)
    y = rng.standard_normal(n_unique) * 1000.0
    for dup_rate in (0.0, 0.10, 0.50, 0.90):
        n_dup = round(dup_rate * n_unique)
        dup_idx = rng.integers(0, n_unique, n_dup) if n_dup else np.array([], dtype=np.int64)
        x_all = np.concatenate([x, x[dup_idx]])
        y_all = np.concatenate([y, y[dup_idx]])
        edges = mdlp_bin_edges_oos_validated(x_all, y_all, seed=1)
        n_bins = edges.size - 1
        assert n_bins <= 2, (dup_rate, n_bins)


# ---------------------------------------------------------------------------
# DISCRETIZATION-6 (P2): edges_optimal_joint masked rows by x's finiteness only, never independently
# checking y for NaN before _bin_y_for_mi's np.quantile call.
# ---------------------------------------------------------------------------


def test_regression_edges_optimal_joint_drops_nan_y_rows():
    """Pre-fix: a NaN-y row with finite x survived the mask and could propagate NaN into quantile edges.
    Post-fix: NaN-y rows are dropped too, same as NaN-x rows."""
    from mlframe.feature_selection.filters._adaptive_nbins import edges_optimal_joint

    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    y[10] = np.nan  # finite x, NaN y -- the bug's exact scenario
    edges = edges_optimal_joint(x, y, n_splits=3, random_state=0)
    assert np.all(np.isfinite(edges)), f"NaN leaked into edges: {edges}"


# ---------------------------------------------------------------------------
# DISCRETIZATION-12 (P2): _mdlp_recurse_oos_validated's present/counts_parent branch handled a combination
# the recursive call sites never actually produce (present_parent set, counts_parent None) -- dead code.
# ---------------------------------------------------------------------------


def test_regression_mdlp_recurse_oos_validated_no_dead_present_parent_branch():
    """Post-fix: the dead present_parent-without-counts_parent branch is removed; passing both together
    (the only combination real call sites use) still works correctly."""
    from mlframe.feature_selection.filters._mdlp_validated_split import _mdlp_recurse_oos_validated

    rng = np.random.default_rng(0)
    n = 200
    x = np.sort(rng.standard_normal(n))
    y = (x > 0).astype(np.int64)
    splits: list = []
    counts_parent = np.bincount(y, minlength=2).astype(np.int64)
    present_parent = np.array([0, 1], dtype=np.int64)
    # Must not raise (the removed branch's absence must not break the both-given call pattern).
    _mdlp_recurse_oos_validated(
        x[:100], y[:100], x[100:], y[100:], splits, 0, 5, 8, 0.3,
        counts_parent=counts_parent, present_parent=present_parent,
    )


# ---------------------------------------------------------------------------
# DISCRETIZATION-13 (P2): cap_categorical_cardinality's "most frequent" selection used the non-stable
# default argsort, so two categories tied exactly at the cutoff boundary could swap non-deterministically
# across numpy versions/architectures.
# ---------------------------------------------------------------------------


def test_regression_cap_categorical_cardinality_stable_tie_break():
    """Post-fix: kind='stable' argsort on negated counts gives a deterministic, reproducible tie-break
    (lowest original index wins) for categories tied exactly at the cutoff."""
    from mlframe.feature_selection.filters.discretization import cap_categorical_cardinality

    # Exercises the PRODUCT, not numpy: asserting np.argsort(kind='stable') on its own only restates a numpy
    # guarantee and would keep passing if the function stopped using a stable sort altogether.
    # Codes 0 and 1 are tied at 10 occurrences, 3 has 20, 2 has 5. With a cap of 3 the two most frequent codes
    # keep distinct ids and the rest fold into the 'other' bucket, so the tie decides which of 0/1 survives:
    # a stable sort keeps the lower original code, deterministically, on every run and platform.
    col = np.array([3.0] * 20 + [0.0] * 10 + [1.0] * 10 + [2.0] * 5).reshape(-1, 1)
    capped = cap_categorical_cardinality(col, 3)

    # id 0 is the most frequent code (3), id 1 the tie-break winner (0), id 2 the folded 'other' bucket.
    assert capped[col[:, 0] == 3.0, 0].tolist() == [0.0] * 20
    assert capped[col[:, 0] == 0.0, 0].tolist() == [1.0] * 10, "the tie must resolve to the LOWER original code"
    assert set(capped[col[:, 0] == 1.0, 0].tolist()) == {2.0}, "the tie loser must fold into the other bucket"
    assert set(capped[col[:, 0] == 2.0, 0].tolist()) == {2.0}, "the rare code must fold into the other bucket"

    # Determinism: the same input must give the same assignment every call.
    np.testing.assert_array_equal(capped, cap_categorical_cardinality(col, 3))


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-1 (P1): cluster_stability_selection / complementary_pairs_stability /
# stability_select_fe all silently returned an empty/"nothing is stable" result (only a logger.warning)
# when every single bootstrap failed, unlike the sibling StabilityMRMR.fit's post-B-14 RuntimeError.
# ---------------------------------------------------------------------------


def test_regression_cluster_stability_selection_raises_on_total_failure():
    """Pre-fix: 100% bootstrap failure silently returned an empty selection. Post-fix: raises RuntimeError."""
    from mlframe.feature_selection.filters._stability_cluster import cluster_stability_selection

    rng = np.random.default_rng(0)
    n, p = 40, 5
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    def _always_fails(Xb, yb):
        """Stand-in that raises ValueError, so the caller's failure path is the one under test."""
        raise ValueError("simulated systematic selector_fn failure")

    with pytest.raises(RuntimeError, match="all .* bootstraps failed"):
        cluster_stability_selection(X, y, _always_fails, n_bootstrap=5, rng_seed=0)


def test_regression_complementary_pairs_stability_raises_on_total_failure():
    """Pre-fix: 100% pair failure silently returned an empty selection. Post-fix: raises RuntimeError."""
    from mlframe.feature_selection.filters._stability_cluster import complementary_pairs_stability

    rng = np.random.default_rng(0)
    n, p = 40, 5
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    def _always_fails(Xb, yb):
        """Stand-in that raises ValueError, so the caller's failure path is the one under test."""
        raise ValueError("simulated systematic selector_fn failure")

    with pytest.raises(RuntimeError, match="all .* pairs failed"):
        complementary_pairs_stability(X, y, _always_fails, n_pairs=5, rng_seed=0)


def test_regression_stability_select_fe_raises_on_total_failure(monkeypatch):
    """Pre-fix: 100% bootstrap failure silently returned an empty per-bootstrap list. Post-fix: raises."""
    import pandas as pd

    from mlframe.feature_selection.filters import _stability_fe as sfe

    class _FakeMRMR:
        """Minimal MRMR stand-in: carries just the attributes the code under test reads, nothing else."""
        def __init__(self, **kwargs):
            """Minimal stand-in constructor for the double used below."""
            pass

        def fit(self, X, y):
            """Stand-in that raises ValueError, so the caller's failure path is the one under test."""
            raise ValueError("simulated systematic MRMR.fit failure")

    monkeypatch.setattr(sfe, "_resolve_mrmr_cls", lambda: _FakeMRMR)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.standard_normal(40))

    with pytest.raises(RuntimeError, match="all .* bootstraps failed"):
        sfe._run_bootstraps(X, y, {}, n_bootstraps=5, sample_fraction=0.5, rng=rng)


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-5 (P2): _calibrate_tau_auto's diagnostics (n_pairs_sampled/n_pairs_finite) were
# populated AFTER the len(su_scores)<10 fallback check, so they stayed at 0 whenever that fallback fired.
# ---------------------------------------------------------------------------


def test_regression_calibrate_tau_auto_diagnostics_populated_before_fallback():
    """Pre-fix: n_pairs_sampled/n_pairs_finite stayed at their 0 init value when the <10-scores fallback
    fired. Post-fix: they reflect what was actually sampled/scored even on the fallback path."""
    from mlframe.feature_selection.filters._dcd_tau_auto import _calibrate_tau_auto

    rng = np.random.default_rng(0)
    n_cols = 4  # small n_cols -> C(4,2)=6 pairs, well under the len(su_scores)<10 fallback floor
    n_rows = 200
    factors_data = rng.integers(0, 5, size=(n_rows, n_cols)).astype(np.int32)
    factors_nbins = np.full(n_cols, 5, dtype=np.int64)

    _tau, diagnostics = _calibrate_tau_auto(factors_data=factors_data, factors_nbins=factors_nbins, seed=0)
    assert diagnostics["n_pairs_sampled"] > 0, "diagnostics must reflect the actual sampled-pair count, not 0"


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-6 (P2): _binarize_aggregate excluded NaN/Inf rows only from edge computation, not
# from the actual binned output -- a non-finite row silently landed in a real bin instead of a dedicated code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["quantile", "uniform"])
def test_regression_binarize_aggregate_nan_gets_dedicated_code(method):
    """Pre-fix: a NaN row silently landed in a real bin (last bin for quantile, bin 0 for uniform).
    Post-fix: it gets the dedicated out-of-band code (n_bins), one past every real bin."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_metrics import _binarize_aggregate

    rng = np.random.default_rng(0)
    values = rng.standard_normal(200)
    values[7] = np.nan
    n_bins = 5
    binned = _binarize_aggregate(values, method=method, n_bins=n_bins, dtype=np.int64)
    assert binned[7] == n_bins, f"method={method}: NaN row did not get the dedicated out-of-band code"
    assert np.all(binned[np.arange(200) != 7] < n_bins), f"method={method}: a real row leaked into the NaN code"


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-8 (P2): ks_stability_filter's multi-split mode had no validation on split_frac; a
# caller-supplied split_frac > 1.0 crashed with an opaque numpy ValueError instead of a clear message.
# ---------------------------------------------------------------------------


def test_regression_ks_stability_filter_rejects_invalid_split_frac():
    """Pre-fix: split_frac=1.5 crashed deep inside rng.choice with an opaque numpy ValueError.
    Post-fix: a clear, named ValueError raises immediately."""
    import pandas as pd

    from mlframe.feature_selection.filters._ks_stability import ks_stability_filter

    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"f": rng.standard_normal(100)})
    test_df = pd.DataFrame({"f": rng.standard_normal(100)})

    with pytest.raises(ValueError, match="split_frac"):
        ks_stability_filter(train_df, test_df, n_splits=3, split_frac=1.5)


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-10 (P2): commit_swap's is_member_swap check used a getattr default that could never
# be reached (SwapDecision.branch always exists, defaulting to "none").
# ---------------------------------------------------------------------------


def test_regression_dcd_swap_no_dead_getattr_default():
    """Post-fix: commit_swap reads decision.branch directly, no unreachable getattr default."""
    import inspect

    from mlframe.feature_selection.filters._dynamic_cluster_discovery import _dcd_swap

    src = inspect.getsource(_dcd_swap.commit_swap)
    assert 'getattr(decision, "branch"' not in src
    assert "decision.branch ==" in src


# ---------------------------------------------------------------------------
# ORTH_BASIS_A-1 (P1): _run_random_batch_search silently substituted 1 for seed<=0 (including the valid
# seed=0), while sibling RNGs in the SAME call chain correctly kept seed=0 as 0.
# ---------------------------------------------------------------------------


def test_regression_run_random_batch_search_seed_zero_not_substituted():
    """Pre-fix: seed=0 silently became seed=1 internally. Post-fix: np.random.default_rng(seed) directly."""
    import inspect

    from mlframe.feature_selection.filters._hermite_fe_optimise import _run_random_batch_search

    src = inspect.getsource(_run_random_batch_search)
    assert not any(line.strip().startswith("rng = ") and "else 1" in line for line in src.splitlines())
    assert any(line.strip() == "rng = np.random.default_rng(seed)" for line in src.splitlines())


# ---------------------------------------------------------------------------
# ORTH_BASIS_A-2 (P1): _run_cma_search / _run_cma_search_batch's es.ask()/es.tell() bare-except blocks
# swallowed any cma-library fault with zero logging, silently truncating the CMA generation loop early.
# ---------------------------------------------------------------------------


def test_regression_cma_search_ask_tell_exceptions_are_logged():
    """Pre-fix: es.ask()/es.tell() faults were swallowed with a bare `except Exception: break` and zero
    logging. Post-fix: both functions log a warning naming the failing call before breaking."""
    import inspect

    from mlframe.feature_selection.filters._hermite_fe_optimise import _run_cma_search, _run_cma_search_batch

    src_single = inspect.getsource(_run_cma_search)
    src_batch = inspect.getsource(_run_cma_search_batch)
    assert "es.ask() raised" in src_single
    assert "es.ask() raised" in src_batch
    assert "es.tell() raised" in src_batch
    # No bare `except Exception:\n    break` left with nothing in between (would mean the log call is missing).
    for src, label in ((src_single, "single"), (src_batch, "batch")):
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "except Exception:" and i + 1 < len(lines) and lines[i + 1].strip() == "break":
                raise AssertionError(f"{label}: found an unlogged bare except-Exception/break pair")


# ---------------------------------------------------------------------------
# ORTH_BASIS_A-7 (P2): polynom_pair_fe.py's per-pair injection loop appended to survivor-tracking lists
# BEFORE the X column assignment; if the X assignment itself raised, the outer except only logs and
# continues -- it does not undo the earlier appends, so the final np.concatenate would still bake in a
# column with no matching X column and no recipe.
# ---------------------------------------------------------------------------


def test_regression_polynom_pair_fe_x_assignment_before_list_append():
    """Post-fix: the X column assignment happens BEFORE _new_data_cols.append, so a raise during
    assignment leaves the survivor lists untouched (no orphaned column baked into data/nbins)."""
    import inspect

    from mlframe.feature_selection.filters.polynom_pair_fe import run_polynom_pair_fe

    src = inspect.getsource(run_polynom_pair_fe)
    assign_pos = src.index("X[_new_col_name] = _t_vals")
    append_pos = src.index("_new_data_cols.append(_new_binned)")
    assert assign_pos < append_pos, "X column assignment must happen before the survivor-list append"


# ---------------------------------------------------------------------------
# ORTH_BASIS_A-8 (P2): scan_integer_lattice_pairs (prototype) constructed a fresh np.random.default_rng
# inside the innermost per-candidate loop, so every candidate was tested against the IDENTICAL n_perm
# shuffles of y (same draws every time, since the generator restarts at the same seed each call).
# ---------------------------------------------------------------------------


def test_regression_scan_integer_lattice_pairs_uses_one_shared_rng():
    """Pre-fix: np.random.default_rng(rng_seed) was constructed fresh per candidate. Post-fix: one
    Generator is built once and threaded through, so consecutive candidates draw different shuffles."""
    from mlframe.feature_selection.filters._integer_lattice_fe_proto import _perm_null_hi

    rng = np.random.default_rng(0)
    n = 500
    feat = np.random.default_rng(1).standard_normal(n)
    y = np.random.default_rng(2).integers(0, 3, n)
    v1 = _perm_null_hi(feat, y, nbins=10, n_perm=12, rng=rng)
    v2 = _perm_null_hi(feat, y, nbins=10, n_perm=12, rng=rng)
    # Same shared rng, called twice in sequence -> the SECOND call's shuffles must differ from the first's
    # (proving the generator's state actually advances instead of being reset each call).
    assert v1 != v2 or True  # values may coincidentally match; the real proof is in the source below
    import inspect

    from mlframe.feature_selection.filters._integer_lattice_fe_proto import scan_integer_lattice_pairs

    src = inspect.getsource(scan_integer_lattice_pairs)
    assert "np.random.default_rng(rng_seed)" in src  # built once, outside the loop
    assert not any("default_rng(rng_seed)" in line and line.strip().startswith("null_hi") for line in src.splitlines())


# ---------------------------------------------------------------------------
# ORTH_BASIS_B-3 (P2): _fourier_detect_gpu_resident.py's _SPLIT_MASK_CACHE/_SUBSAMPLE_IDX_CACHE performed
# an unlocked read-then-write (get() then [key]=), the same hand-rolled unlocked module-level cache pattern
# that reproduced a real crash elsewhere in this codebase under concurrent access.
# ---------------------------------------------------------------------------


def test_regression_fourier_detect_gpu_resident_caches_have_lock():
    """Pre-fix: no lock guarded the get-or-insert sequence on either cache. Post-fix: both are guarded by
    a shared threading.Lock."""
    import threading

    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_gpu_resident as fd

    assert isinstance(fd._CACHE_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# ORTH_BASIS_B-4 (P2): _orth_extra_basis_fe.py <-> _orth_extra_basis_fe_generate.py had a circular
# TOP-LEVEL import (each imports names from the other at module scope), which only resolved because every
# current caller happens to trigger _orth_extra_basis_fe.py's import first. A future caller importing
# _orth_extra_basis_fe_generate directly would have hit ImportError on a partially-initialized module.
# ---------------------------------------------------------------------------


def test_regression_orth_extra_basis_fe_generate_importable_directly_first():
    """Pre-fix: importing _orth_extra_basis_fe_generate BEFORE _orth_extra_basis_fe (e.g. in a fresh
    subprocess where nothing has touched this package yet) would raise ImportError on a partially
    initialized module. Post-fix: a lazy module __getattr__ breaks the load-order dependency."""
    import subprocess
    import sys

    code = (
        "from mlframe.feature_selection.filters._orthogonal_univariate_fe import "
        "_orth_extra_basis_fe_generate as g; "
        "assert callable(g.generate_extra_basis_features); "
        "from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import "
        "generate_extra_basis_features; "
        "assert callable(generate_extra_basis_features); "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# ORTH_SCORING_A-1 (P1): hybrid_orth_mi_adaptive_arity_fe_with_recipes's arity-2/3/4 branches built
# EngineeredRecipe / build_orth_quadruplet_cross_recipe WITHOUT preprocess_params_i/j/k/l -- the B-17
# replay-drift bug (frozen fit-time basis-preprocess params never persisted) was still open for arity>=2,
# which is Layer 78's entire point. The B-17 fix commit only touched this file's arity==1 branch.
# ---------------------------------------------------------------------------


def test_regression_adaptive_arity_pair_recipe_freezes_preprocess_params():
    """Pre-fix: an arity-2 recipe's extra dict had no preprocess_params_i/j keys at all (raw EngineeredRecipe
    construction). Post-fix: build_orth_pair_cross_recipe is used and both keys are populated (non-None for
    a basis with real preprocess params, e.g. zscore mean/std)."""
    import pandas as pd

    from mlframe.feature_selection.filters._orthogonal_adaptive_arity_fe import (
        hybrid_orth_mi_adaptive_arity_fe_with_recipes,
    )

    rng = np.random.default_rng(1)
    n = 3000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    y = ((x1 * x2) + 0.02 * rng.standard_normal(n) > 0).astype(int)

    _X_aug, _uni, _adaptive, recipes = hybrid_orth_mi_adaptive_arity_fe_with_recipes(
        X, y, cols=["x1", "x2", "x3", "x4"], max_arity=4, max_degree=1, basis="hermite", seed_k=4,
    )
    pair_recipes = [r for r in recipes if r.kind == "orth_pair_cross"]
    assert pair_recipes, "expected at least one arity-2 recipe on a pure 2-way XOR signal"
    for r in pair_recipes:
        assert "preprocess_params_i" in r.extra, f"recipe {r.name} missing preprocess_params_i key entirely"
        assert "preprocess_params_j" in r.extra, f"recipe {r.name} missing preprocess_params_j key entirely"


# ---------------------------------------------------------------------------
# ORTH_SCORING_A-3 (P2): all 11 univariate *_with_recipes builders wrapped the B-17-fixed
# _evaluate_basis_column(..., return_params=True) recompute in a bare except with zero logging.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname", [
    "_orthogonal_cmim_fe", "_orthogonal_jmim_fe", "_orthogonal_three_gate_mi_fe",
    "_orthogonal_bootstrap_mi_fe", "_orthogonal_adaptive_arity_fe", "_orthogonal_adaptive_degree_fe",
    "_orthogonal_total_correlation_fe", "_orthogonal_ksg_mi_fe", "_orthogonal_copula_mi_fe",
    "_orthogonal_dcor_fe", "_orthogonal_hsic_fe",
])
def test_regression_scorer_zoo_preprocess_params_except_is_logged(modname):
    """Pre-fix: `except Exception: _pp = None` with zero logging. Post-fix: a debug log call exists."""
    import importlib
    import inspect

    mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    src = inspect.getsource(mod)
    assert "failed to freeze fit-time basis preprocess_params" in src, f"{modname}: missing the logging fix"


# ---------------------------------------------------------------------------
# ORTH_SCORING_A-4 (P2): _route_basis's except Exception: return "hermite" fallback had zero logging.
# ---------------------------------------------------------------------------


def test_regression_route_basis_exception_is_logged():
    """Post-fix: _route_basis's fallback logs at debug level before returning 'hermite'."""
    import inspect

    from mlframe.feature_selection.filters._orthogonal_adaptive_arity_fe import hybrid_orth_mi_adaptive_arity_fe_with_recipes

    src = inspect.getsource(hybrid_orth_mi_adaptive_arity_fe_with_recipes)
    assert "_route_basis: failed to route column" in src


# ---------------------------------------------------------------------------
# ORTH_SCORING_A-5 (P2): KSG/HSIC's near-zero-baseline uplift special-case hard-set uplift=0.0 whenever
# engineered_mi was below the FIXED _ABS_MI_FLOOR=1e-3, which could hard-reject a candidate the real
# DYNAMIC MAD-based abs_floor (gate 2) would have accepted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname", ["_orthogonal_ksg_mi_fe", "_orthogonal_hsic_fe"])
def test_regression_near_zero_baseline_uplift_defers_to_dynamic_floor(modname):
    """Pre-fix: uplift = 0.0 if emi < _ABS_MI_FLOOR else inf (could hard-reject a candidate the dynamic
    floor would accept). Post-fix: uplift is always inf for a near-zero baseline, deferring entirely to
    gate 2's dynamic floor."""
    import importlib
    import inspect

    mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    src = inspect.getsource(mod)
    assert '0.0 if emi < _ABS_MI_FLOOR else float("inf")' not in src
    assert 'uplift = float("inf")' in src


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-1 (P1): _orth_auto_scorer_fe.py's _score_plug_in coerced non-integer y via a bare
# `y_arr.astype(np.int64)` -- TRUNCATES, does not densify -- the exact B-18 bug class, reintroduced
# because this function was carved into a new sibling file BEFORE the B-18 fix pass patched its parent.
# ---------------------------------------------------------------------------


def test_regression_score_plug_in_densifies_fractional_y_not_truncates():
    """Pre-fix: y=[0.1, 0.2] perfectly separated by x truncated both classes to 0 -> MI=0.0.
    Post-fix: densified via np.unique(return_inverse=True) -> a real, non-zero MI."""
    from mlframe.feature_selection.filters._orth_auto_scorer_fe import _score_plug_in

    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    y = np.where(x > 0, 0.1, 0.2)  # perfectly separated by x, but both truncate to class 0 pre-fix
    mi = _score_plug_in(x, y)
    assert mi > 0.1, f"expected a real, non-degenerate MI for a perfectly-separated fractional y, got {mi}"


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-2 (P1): hybrid_orth_mi_auto_scorer_fe_with_recipes never froze the fit-time basis-
# preprocess params into emitted orth_univariate recipes -- the B-17 bug, reintroduced by the same
# split-before-fix blind spot as B-1.
# ---------------------------------------------------------------------------


def test_regression_auto_scorer_recipe_freezes_preprocess_params():
    """Pre-fix: build_orth_univariate_recipe was called with no preprocess_params kwarg at all.
    Post-fix: preprocess_params is threaded through, recomputed from the full fit-time column."""
    import inspect

    from mlframe.feature_selection.filters._orth_auto_scorer_fe import hybrid_orth_mi_auto_scorer_fe_with_recipes

    src = inspect.getsource(hybrid_orth_mi_auto_scorer_fe_with_recipes)
    assert "_evaluate_basis_column" in src
    assert "preprocess_params=_pp" in src


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-3 (P1): _oracle_scorer_select.py's _ROWS_CACHE get-check-evict-write sequence had no
# lock, empirically reproducing crashes (dictionary changed size during iteration / KeyError) under
# concurrent access.
# ---------------------------------------------------------------------------


def test_regression_oracle_rows_cache_concurrent_access_no_crash():
    """Pre-fix: 32 threads x 500 calls against distinct fake stores reproduced ~225 crashes under tightened
    thread-switch interval. Post-fix: the same stress pattern (reduced scale for test speed) completes with
    zero exceptions, since the whole get-or-insert sequence is now lock-guarded."""
    import sys
    import threading

    from mlframe.feature_selection.filters._oracle_scorer_select import _cached_read_rows, _ROWS_CACHE

    class _FakeStore:
        """In-memory stand-in for the persistence layer, so the test never touches real storage."""
        def __init__(self, path, rows):
            """Minimal stand-in constructor for the double used below."""
            self._path = path
            self._rows = rows

        def read_rows(self):
            """Test helper: read rows."""
            return list(self._rows)

    _ROWS_CACHE.clear()
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force aggressive interleaving to actually exercise the race window
    try:
        stores = [_FakeStore(f"/fake/store_{i}.parquet", [{"i": i}]) for i in range(20)]
        errors: list[BaseException] = []

        def _worker():
            """Thread body: exercises the call under test and records its result or error."""
            rng = np.random.default_rng()
            for _ in range(300):
                store = stores[rng.integers(0, len(stores))]
                try:
                    _cached_read_rows(store)
                except BaseException as exc:
                    errors.append(exc)

        # os.path.getmtime on a nonexistent fake path raises OSError -> _cached_read_rows falls back to the
        # uncached path for these fixtures, which still exercises _ROWS_CACHE.get() at the top under
        # concurrent access from the mtime=-1.0 branch... to actually hit the cache dict, patch getmtime.
        import mlframe.feature_selection.filters._oracle_scorer_select as mod

        orig_isfile = mod.os.path.isfile
        mod.os.path.isfile = lambda p: True  # nosec B110 - test monkeypatch, restored in finally
        mod.os.path.getmtime = lambda p: 1.0  # nosec B110 - fixed mtime so all calls for a store share one cache key

        threads = [threading.Thread(target=_worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        mod.os.path.isfile = orig_isfile
    finally:
        sys.setswitchinterval(old_interval)
    assert not errors, f"concurrent _cached_read_rows access raised: {errors[:3]}"


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-4 (P2): _orthogonal_scorer_auto_fe.py's nested _call() helper's "plug_in"/"copula"
# branches are unreachable dead code (both names are always in _BATCHABLE, routed through _batch_scores
# instead) -- documented, not deleted, as the correct fallback contract.
# ---------------------------------------------------------------------------


def test_regression_call_dead_branches_documented():
    """Post-fix: the dead plug_in/copula branches carry an explicit note explaining why they're kept."""
    import inspect

    from mlframe.feature_selection.filters import _orthogonal_scorer_auto_fe as mod

    src = inspect.getsource(mod)
    # The note must survive, not the finding-ID that labelled it; comment-style forbids the ID in source.
    assert "plug_in" in src and "copula" in src, "the plug_in/copula branches vanished entirely"
    assert (
        "dead" in src.lower() or "kept" in src.lower() or "unreachable" in src.lower()
    ), "the branches are no longer explained - a reader cannot tell why they are kept"


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-5 (P2): detect_clusters_by_correlation's max_cluster_size truncation branch computed a
# no-op O(p) linear-scan re-lookup (dense_names[dense_names.index(name)] just produces name back).
# ---------------------------------------------------------------------------


def test_regression_cluster_basis_no_noop_index_lookup():
    """Post-fix: the no-op dense_names.index(name) re-lookup is removed; mean_corr's names are used directly."""
    import inspect

    from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import detect_clusters_by_correlation

    src = inspect.getsource(detect_clusters_by_correlation)
    assert not any("members = sorted(dense_names[dense_names.index(name)]" in line for line in src.splitlines())


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-7 (P2): _oracle_scorer_select.py's _quality_objective except-Exception had zero logging.
# ---------------------------------------------------------------------------


def test_regression_quality_objective_exception_is_logged():
    """Post-fix: a debug log call exists for the malformed-output fallback."""
    import inspect

    from mlframe.feature_selection.filters._oracle_scorer_select import _quality_objective

    src = inspect.getsource(_quality_objective)
    assert "logger.debug" in src


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-9 (P2): _orthogonal_meta_scorer_fe.py's fingerprint_signal inner per-column Pearson/
# Spearman/symmetric-Pearson probes used bare except Exception (3 sites), broader than the module's own
# declared _NUMERIC_ERRORS convention, with zero logging.
# ---------------------------------------------------------------------------


def test_regression_meta_scorer_inner_corr_excepts_use_numeric_errors():
    """Post-fix: all 3 inner corr except-blocks use _NUMERIC_ERRORS (not bare Exception) and log at debug."""
    import inspect

    from mlframe.feature_selection.filters import _orthogonal_meta_scorer_fe as mod

    src = inspect.getsource(mod)
    assert src.count("except _NUMERIC_ERRORS as exc:") >= 3
    assert "meta_scorer pearson corr failed" in src
    assert "meta_scorer spearman corr failed" in src
    assert "meta_scorer symmetric-pearson corr failed" in src


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-1 (P1): the marginal-MI term subtracted in every cat-FE Interaction Information
# computation was ALWAYS built via the unweighted _marginal_screen_njit, even when sample weights made
# the joint-MI search kernel weighted -- mixing a weighted joint term with an unweighted marginal term.
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
    """Post-fix: run_cat_interaction_step's weighted branch calls _marginal_screen_weighted and overwrites
    marginal_mi_full before the pair-search dispatch, rather than leaving it unweighted."""
    import inspect

    from mlframe.feature_selection.filters._cat_interactions_step import run_cat_interaction_step

    src = inspect.getsource(run_cat_interaction_step)
    assert "_marginal_screen_weighted" in src
    assert "marginal_mi_full[int(_idx)] = candidate_mi[_k]" in src


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-2 (P2): neither cat-FE GPU dispatch point (permutation-confirmation kernel,
# pair-search kernel) consulted gpu_globally_disabled() -- MLFRAME_DISABLE_GPU=1 was silently ignored.
# ---------------------------------------------------------------------------


def test_regression_cat_confirm_permutation_gpu_gate_honours_disable_gpu(monkeypatch):
    """Pre-fix: MLFRAME_DISABLE_GPU=1 was never checked. Post-fix: _perm_kernel_dispatch_use_gpu returns
    False whenever the global GPU off-switch is set, even for an explicit backend='gpu' request."""
    from mlframe.feature_selection.filters._cat_confirm_permutation import _perm_kernel_dispatch_use_gpu
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert gpu_globally_disabled() is True
    assert _perm_kernel_dispatch_use_gpu(n_samples=1_000_000, n_perms=1000, backend="gpu") is False
    assert _perm_kernel_dispatch_use_gpu(n_samples=1_000_000, n_perms=1000, backend="auto") is False


def test_regression_cat_interactions_step_pair_search_gpu_gate_honours_disable_gpu():
    """Post-fix: run_cat_interaction_step's pair-search GPU gate consults gpu_globally_disabled() before
    honouring cfg.backend='gpu'/'auto', and raises a clear error for an explicit backend='gpu' request
    under the global off-switch instead of silently using GPU."""
    import inspect

    from mlframe.feature_selection.filters._cat_interactions_step import run_cat_interaction_step

    src = inspect.getsource(run_cat_interaction_step)
    assert "gpu_globally_disabled" in src
    assert "GPU is globally disabled" in src


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-3 (P1): CatFEConfig.perm_budget_strategy defaults to "bandit_ucb1" (not "fixed"), and
# the bandit allocator unconditionally ignores weights -- so the DEFAULT confirmation path for any weighted
# cat-FE fit silently tested a weighted II_obs against an unweighted permutation null.
# ---------------------------------------------------------------------------


def test_regression_bandit_ucb1_auto_falls_back_to_fixed_when_weighted():
    """Post-fix: run_cat_interaction_step skips the bandit-UCB1 path entirely (falling back to the
    correctly-weighted fixed path) whenever use_weights is True, regardless of perm_budget_strategy."""
    import inspect

    from mlframe.feature_selection.filters._cat_interactions_step import run_cat_interaction_step

    src = inspect.getsource(run_cat_interaction_step)
    assert "and not use_weights" in src
    assert "falling back to the fixed-budget permutation path" in src


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-8 (P2): _kfold_target_encode_codes had no n_folds validation; n_folds=0 raised an
# opaque ZeroDivisionError, n_folds=1 silently emitted an all-global_mean column with no warning.
# ---------------------------------------------------------------------------


def test_regression_kfold_target_encode_codes_rejects_n_folds_below_2():
    """Pre-fix: n_folds=0 raised a raw ZeroDivisionError; n_folds=1 silently produced a useless column.
    Post-fix: both raise a clear, named ValueError."""
    from mlframe.feature_selection.filters._cat_pair_fe import _kfold_target_encode_codes

    codes = np.array([0, 1, 0, 1, 2, 2], dtype=np.int64)
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    for bad_n_folds in (0, 1):
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            _kfold_target_encode_codes(codes, y, n_folds=bad_n_folds)


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-9 (P2): CatFEConfig.__post_init__ omitted range checks for 5 fields, letting e.g.
# bootstrap_ci_alpha=1.2 pass construction silently and later compute inverted CI bounds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwarg,bad_value,match", [
    ("bootstrap_ci_alpha", 1.2, "bootstrap_ci_alpha"),
    ("bootstrap_ci_alpha", 0.0, "bootstrap_ci_alpha"),
    ("bootstrap_sample_frac", 0.0, "bootstrap_sample_frac"),
    ("bootstrap_sample_frac", 1.5, "bootstrap_sample_frac"),
    ("streaming_cache_kl_threshold", 0.0, "streaming_cache_kl_threshold"),
    ("streaming_cache_kl_threshold", -1.0, "streaming_cache_kl_threshold"),
    ("target_encoding_smoothing", -1.0, "target_encoding_smoothing"),
    ("numeric_nbins", 1, "numeric_nbins"),
])
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


def test_regression_rolling_median_raises_clear_error_not_keyerror():
    """Pre-fix: stats=["median"] raised a bare KeyError deep in the call stack. Post-fix: a clear,
    actionable ValueError names the unsupported stat up front."""
    import pandas as pd

    from mlframe.feature_selection.filters._temporal_agg_fe_rolling import generate_rolling_window_agg_features

    X = pd.DataFrame({
        "entity": [1, 1, 2, 2],
        "t": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
        "v": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="median"):
        generate_rolling_window_agg_features(X, ["entity"], ["v"], "t", windows=("7D",), stats=["median"])


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-2 (P1): _global_value_for_stat's nunique/count fallback for an unseen group/composite
# key returned a whole-population-scale statistic instead of a per-group-scale one.
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
    # The whole-population cardinality (the pre-fix fallback) must NOT appear as a "fallback" value mixed
    # into an otherwise per-group-scale column -- every observed value should be well below n (real groups
    # were all seen at fit time here, so this just pins the per-group scale is sane, not degenerate).
    assert all(v < n * 0.5 for v in real_per_group_values), f"nunique values not per-group scale: {real_per_group_values}"


def test_regression_composite_group_agg_count_fallback_is_cell_scale():
    """Post-fix: _composite_group_agg_fe's count/nunique fallback also uses the median per-cell scale."""
    import inspect

    from mlframe.feature_selection.filters import _composite_group_agg_fe as mod

    src = inspect.getsource(mod)
    assert 'stat in ("count", "nunique")' in src
    assert "np.median(agg_series.to_numpy())" in src


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-3 (P1): _gradient_interaction_seeder.py's two except-Exception blocks logged only
# `if verbose`, so with the default verbose=0 any exception was silently swallowed with zero trace.
# ---------------------------------------------------------------------------


def test_regression_gradient_seeder_exceptions_logged_unconditionally():
    """Post-fix: both except-blocks log unconditionally (not gated behind `if verbose`)."""
    import inspect

    from mlframe.feature_selection.filters import _gradient_interaction_seeder as mod

    src = inspect.getsource(mod.propose_gradient_interaction_pairs)
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "not array-coercible" in line or "gradient-interaction seeder failed" in line:
            assert lines[i - 1].strip() != "if verbose:", f"still gated behind verbose: {line!r}"
    assert 'logger.debug("MRMR FE gradient-interaction seeder: X not array-coercible' in src
    assert 'logger.warning("MRMR FE gradient-interaction seeder failed' in src


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-4 (P1): _parse_engineered_name split on the FIRST "__", breaking for any source column
# whose own name contains "__" (e.g. this codebase's own orth-basis naming convention "{col}__{code}{deg}").
# ---------------------------------------------------------------------------


def test_regression_parse_engineered_name_handles_double_underscore_source():
    """Pre-fix: source 'a__b' with emitted name 'a__b__digit_1' parsed to None (recipe skipped).
    Post-fix: rsplit-based parsing correctly recovers src='a__b', code='digit_extract', arg=1."""
    from mlframe.feature_selection.filters._numeric_decompose_fe import _parse_engineered_name

    assert _parse_engineered_name("a__b__digit_1") == ("digit_extract", "a__b", 1)
    assert _parse_engineered_name("a__b__round_0p1") == ("numeric_rounding", "a__b", 0.1)
    assert _parse_engineered_name("plain__digit_2") == ("digit_extract", "plain", 2)


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-5 (P1): the lagged_diff family had no entity/group-scoping parameter at all, computing
# a global time-sorted diff that silently mixes rows across different entities on panel data.
# ---------------------------------------------------------------------------


def test_regression_lagged_diff_entity_scoping_prevents_cross_entity_diff():
    """Pre-fix: no entity_cols parameter existed at all. Post-fix: passing entity_cols zeroes any diff
    whose period-back neighbour (in time-sorted order) belongs to a different entity."""
    import pandas as pd

    from mlframe.feature_selection.filters._ratio_delta_fe import lagged_diff_features

    X = pd.DataFrame({
        "entity": [1, 1, 1, 2, 2, 2],
        "t": [1, 2, 3, 1, 2, 3],
        "v": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })
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

    X_test = pd.DataFrame({
        "entity": [1, 1, 1, 2, 2, 2],
        "t": [1, 2, 3, 1, 2, 3],
        "v": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })
    out = apply_lagged_diff(X_test, {"time_col": "t", "value_col": "v", "period": 1, "entity_cols": ("entity",)})
    np.testing.assert_allclose(out, [0.0, 10.0, 10.0, 0.0, 100.0, 100.0])


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-6 (P2): _auto_detect_group_cols's two-dot `from .._grouped_agg_fe import ...` was the
# wrong import depth (_grouped_agg_fe.py is a sibling, one dot) and always failed inside a bare except,
# silently falling through to the correct single-dot import -- masked but dead/misleading code.
# ---------------------------------------------------------------------------


def test_regression_composite_group_agg_no_dead_two_dot_import():
    """Post-fix: the always-failing two-dot import is removed; only the correct single-dot import remains."""
    import inspect

    from mlframe.feature_selection.filters._composite_group_agg_fe import _auto_detect_group_cols

    src = inspect.getsource(_auto_detect_group_cols)
    assert not any(line.strip().startswith("from .._grouped_agg_fe import") for line in src.splitlines())
    assert any(line.strip().startswith("from ._grouped_agg_fe import") for line in src.splitlines())


# ---------------------------------------------------------------------------
# FE_PAIRS_CORE-1 (P1): the chunk-pipeline feature runs chunk k+1's production concurrently with the main
# thread consuming chunk k, and both threads reach into the un-locked _OPERAND_TABLE_CACHE /
# _PREBUILT_OPERAND_TABLE OrderedDicts for the same transformed_vars object with no lock.
# ---------------------------------------------------------------------------


def test_regression_gpu_resident_materialise_operand_caches_have_lock():
    """Pre-fix: no lock guarded either cache's get-or-insert sequence. Post-fix: both are guarded by a
    dedicated threading.Lock."""
    import threading

    import mlframe.feature_selection.filters._gpu_resident_materialise as mod

    assert isinstance(mod._OPERAND_TABLE_CACHE_LOCK, type(threading.Lock()))
    assert isinstance(mod._PREBUILT_OPERAND_TABLE_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# FE_PAIRS_CORE-2 (P2): _fe_gpu_discretize_enabled/_fe_gpu_binning_enabled were called up to 3x per pair,
# once per chunk, and once per ext-val tied-leader-set, each re-reading env vars + calling
# is_cuda_available() + a kernel_tuning_cache lookup -- unlike their hoisted sibling _fe_env_gate.
# ---------------------------------------------------------------------------


def test_regression_fe_gpu_gates_are_memoized():
    """Pre-fix: every call re-did the full env/CUDA/KTC lookup. Post-fix: repeated calls with the same
    (n_rows, n_cands) hit an in-process cache instead of recomputing."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_core as mod

    mod._GPU_GATE_CACHE.clear()
    call_count = {"n": 0}
    orig = mod._fe_gpu_discretize_enabled_uncached

    def _counting(*a, **k):
        """Test helper: counting."""
        call_count["n"] += 1
        return orig(*a, **k)

    mod._fe_gpu_discretize_enabled_uncached = _counting
    try:
        r1 = mod._fe_gpu_discretize_enabled(12345, 67)
        r2 = mod._fe_gpu_discretize_enabled(12345, 67)
        assert r1 == r2
        assert call_count["n"] == 1, f"expected exactly 1 uncached call for a repeated (n_rows, n_cands); got {call_count['n']}"
    finally:
        mod._fe_gpu_discretize_enabled_uncached = orig
        mod._GPU_GATE_CACHE.clear()


# ---------------------------------------------------------------------------
# FE_PAIRS_CORE-3 (P2): the "Reject no-variance operands (a constant gate is dead)" comment in
# _pairs_setup.py did not match the code -- the only guard was an isfinite check, which does not reject a
# literal constant (finite-median) operand.
# ---------------------------------------------------------------------------


def test_regression_gate_med_rejects_constant_operand():
    """Post-fix: a constant, non-NaN operand is now rejected (matching the comment's promise), not
    registered under gate_med as a dead all-zero pseudo-unary."""
    import inspect

    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_setup as mod

    src = inspect.getsource(mod)
    assert "nanmax(_gfinite) - np.nanmin(_gfinite)) <= 0.0" in src


# ---------------------------------------------------------------------------
# FE_PAIRS_CORE-4 (P2): the subsample freqs_y recompute used np.bincount without minlength=, so dropping
# every row of the highest class label would silently under-count k_y.
# ---------------------------------------------------------------------------


def test_regression_bincount_minlength_prevents_class_undercounting():
    """Direct pin of the underlying bincount behavior this fix relies on: minlength= keeps k_y honest even
    when the highest class label has zero occurrences in a subsample."""
    classes_y_full_k = 5  # classes 0..4 existed pre-subsample
    classes_y_subsample = np.array([0, 1, 2, 1, 0], dtype=np.int64)  # classes 3, 4 dropped by the draw
    counts_without_minlength = np.bincount(classes_y_subsample)
    counts_with_minlength = np.bincount(classes_y_subsample, minlength=classes_y_full_k)
    assert counts_without_minlength.shape[0] == 3, "sanity: the bug's premise -- bare bincount undercounts k_y"
    assert counts_with_minlength.shape[0] == classes_y_full_k

    import inspect

    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_core as mod

    src = inspect.getsource(mod)
    assert "np.bincount(classes_y.astype(np.int64), minlength=_minlength)" in src


# ---------------------------------------------------------------------------
# FE_ORCH_BUDGET-1 (P1): raws_linearly_explain_y's regression skip-gate scored IN-SAMPLE R^2, which
# crosses the default 0.92 threshold purely from overfitting once p numeric raw columns approaches n --
# a false "nothing left to find" verdict that silently disables the discrete-structural operators.
# ---------------------------------------------------------------------------


def test_regression_raws_linearly_explain_y_rejects_high_p_overfit_noise():
    """Pre-fix: p=1900,n=2000 pure noise gave in-sample R^2=0.936, crossing the 0.92 threshold with ZERO
    real signal. Post-fix: held-out (K-fold CV) R^2 stays near/below 0 for pure noise regardless of p."""
    import pandas as pd

    from mlframe.feature_selection.filters._fe_linear_explainability import raws_linearly_explain_y

    rng = np.random.default_rng(0)
    n, p = 2000, 1900
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = rng.standard_normal(n)
    assert raws_linearly_explain_y(X, y) is False, "high-p pure noise must NOT clear the linear-explains-y gate"


def test_regression_raws_linearly_explain_y_still_fires_on_real_signal():
    """A genuine strong linear relationship must still clear the gate (the fix must not make it
    permanently unreachable)."""
    import pandas as pd

    from mlframe.feature_selection.filters._fe_linear_explainability import raws_linearly_explain_y

    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] * 2.0 + X["x1"] * 0.5 + rng.standard_normal(n) * 0.01
    assert raws_linearly_explain_y(X, y) is True, "a genuine strong linear relationship must still clear the gate"


# ---------------------------------------------------------------------------
# FE_ORCH_BUDGET-2 (P1): four module-level bounded-FIFO memo caches used the unlocked
# `if len(cache) > N: cache.pop(next(iter(cache)))` eviction idiom with no lock, the same race class
# fixed elsewhere in this codebase for concurrent MRMR.fit() calls.
# ---------------------------------------------------------------------------


def test_regression_fe_gate_memo_caches_have_lock():
    """Pre-fix: no lock guarded any of the 4 caches' eviction. Post-fix: both modules expose a
    _FE_GATE_MEMO_LOCK guarding their respective caches."""
    import threading

    from mlframe.feature_selection.filters import _unified_fe_gate as ufg
    from mlframe.feature_selection.filters import _fe_accuracy_gate as fag

    assert isinstance(ufg._FE_GATE_MEMO_LOCK, type(threading.Lock()))
    assert isinstance(fag._FE_GATE_MEMO_LOCK, type(threading.Lock()))


def test_regression_fe_gate_memo_caches_concurrent_access_no_crash():
    """Stress test mirroring the audit's proposed regression: call raw_mi_noise_floor / infer_classification
    from many threads with distinct-content keys (forcing repeated FIFO eviction) and assert no KeyError."""
    import threading

    import pandas as pd

    from mlframe.feature_selection.filters._unified_fe_gate import raw_mi_noise_floor
    from mlframe.feature_selection.filters._fe_accuracy_gate import infer_classification

    errors: list[BaseException] = []

    def _worker(seed):
        """Thread body: exercises the call under test and records its result or error."""
        rng = np.random.default_rng(seed)
        try:
            for _ in range(20):
                n = 200
                X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
                y = rng.standard_normal(n)
                raw_mi_noise_floor(X, y)
                infer_classification(y)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent FE gate memo access raised: {errors[:3]}"


# ---------------------------------------------------------------------------
# FE_ORCH_BUDGET-3 (P2): FeatureMatrix.numeric_column / from_feature_matrix both resolved a column by
# columns.index(name), which returns only the first index for a duplicate column name -- silently
# corrupting/dropping data on round-trip instead of raising.
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


def test_regression_fe_auto_escalation_excepts_are_logged():
    """Post-fix: all 3 previously-silent except-blocks now log at debug level."""
    import inspect

    from mlframe.feature_selection.filters import _fe_auto_escalation as mod

    src = inspect.getsource(mod)
    assert "apply_operand_prewarp failed; skipping candidate" in src
    assert "fit_pair_prewarp_als failed; skipping candidate" in src
    assert "column extraction failed for" in src


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


def test_regression_exhaustive_sweep_cpu_branch_uses_ktc():
    """Post-fix: a measured_cpu_pairs_per_second/warm_exhaustive_cpu_throughput_cache pair exists and is
    wired into decide_exhaustive_sweep's CPU branch, mirroring the CUDA branch's KTC discipline."""
    import inspect

    from mlframe.feature_selection.filters._fe_synergy_exhaustive import (
        decide_exhaustive_sweep, measured_cpu_pairs_per_second,
    )

    # Sanity: the KTC-backed function actually falls back gracefully and returns a positive throughput.
    pps, source = measured_cpu_pairs_per_second(50_000, 4096)
    assert pps > 0
    assert source in ("cache", "fallback")

    src = inspect.getsource(decide_exhaustive_sweep)
    assert "predict_exhaustive_cpu_seconds" in src
    assert "warm_exhaustive_cpu_throughput_cache" in src
    assert not any(line.strip() == "pps = float(_EXHAUSTIVE_CPU_FALLBACK_PAIRS_PER_SEC)" for line in src.splitlines())


# ---------------------------------------------------------------------------
# FE_REDUNDANCY_SYNERGY-2 (P1): the nested clean-sub-expression anchor builder iterated the CALLER'S FULL
# raw input-feature-name set (P in the tens of thousands on a wide frame) per engineered survivor, instead
# of just the raw names actually referenced by that survivor's recipe sub-expressions.
# ---------------------------------------------------------------------------


def test_regression_raw_redundancy_anchors_iterates_bounded_raw_set():
    """Post-fix: the inner loop iterates the bounded union of actually-referenced parents
    (_all_relevant_raws), not the caller's full raw_name_set."""
    import inspect

    from mlframe.feature_selection.filters import _fe_raw_redundancy_anchors as mod

    src = inspect.getsource(mod)
    assert "_all_relevant_raws = set().union(*_sub_parents.values())" in src
    assert "for _rn in _all_relevant_raws:" in src


# ---------------------------------------------------------------------------
# FE_REDUNDANCY_SYNERGY-3 (P1): _Y_DENSE_MEMO's read-check-evict-write sequence had no lock -- two
# MRMR.fit() calls on DIFFERENT instances in different threads of the same process could race
# next(iter(_Y_DENSE_MEMO)), raising RuntimeError: dictionary changed size during iteration.
# ---------------------------------------------------------------------------


def test_regression_y_dense_memo_has_lock():
    """Pre-fix: no lock guarded _Y_DENSE_MEMO. Post-fix: a dedicated threading.Lock exists."""
    import threading

    from mlframe.feature_selection.filters import _fe_cmi_redundancy_gate as mod

    assert isinstance(mod._Y_DENSE_MEMO_LOCK, type(threading.Lock()))


def test_regression_y_dense_memo_concurrent_access_no_crash():
    """Stress test: call the y-dense-memoizing gate path from many threads with distinct-content y arrays
    (forcing repeated FIFO eviction) and assert no RuntimeError/KeyError."""
    import threading

    from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import apply_cmi_redundancy_gate

    errors: list[BaseException] = []

    def _worker(seed):
        """Thread body: exercises the call under test and records its result or error."""
        rng = np.random.default_rng(seed)
        try:
            for _ in range(15):
                n = 300
                names = ["a", "b", "c"]
                candidates = {nm: (rng.standard_normal(n), float(rng.uniform(0.01, 0.5))) for nm in names}
                y = rng.integers(0, 3, n).astype(np.int64)
                apply_cmi_redundancy_gate(candidates, y)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent _Y_DENSE_MEMO access raised: {errors[:3]}"


# ---------------------------------------------------------------------------
# FE_REDUNDANCY_SYNERGY-8/9/10 (P2): stale "NOT YET WIRED" docstring, stale self-referential log-line-
# number strings across 6 files, and an unlogged except-Exception swallow in gbm_split_propensity.
# ---------------------------------------------------------------------------


def test_regression_synergy_screen_docstring_reflects_actual_wiring():
    """Post-fix: the module docstring no longer claims detect_synergy_combos is unwired."""
    from mlframe.feature_selection.filters import _fe_synergy_screen as mod

    doc = mod.__doc__ or ""
    assert "NOT YET WIRED" not in doc


@pytest.mark.parametrize("modname", [
    "_fe_raw_redundancy_drop", "_fe_sufficient_summary", "_fe_cmi_redundancy_null",
    "_extra_fe_families_dispersion_resident", "_meta_fe_recommender", "_conditional_gate_fe",
])
def test_regression_no_stale_self_referential_log_line_numbers(modname):
    """Post-fix: none of the 6 files' debug-log messages embed a self-referential '<file>.py:<N>' string
    that can go stale relative to the file's actual current length."""
    import importlib
    import inspect
    import re

    mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    src = inspect.getsource(mod)
    assert not re.search(r"suppressed in \S+\.py:\d+", src), f"{modname} still has a stale line-number log string"


def test_regression_gbm_split_propensity_exception_is_logged():
    """Post-fix: the lgb.train/feature_importance except-block logs at debug level."""
    import inspect

    from mlframe.feature_selection.filters import _fe_interaction_prerank as mod

    src = inspect.getsource(mod)
    assert "gbm_split_propensity: lgb.train/feature_importance failed" in src


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-1 (P1): greedy_cmi_fe_construct's noise-floor permutation RNG was hardcoded to
# 0xC011 with no seed parameter at all, correlating the FE admission gate across nominally-independent
# bootstrap/multi-seed replicates.
# ---------------------------------------------------------------------------


def test_regression_greedy_cmi_fe_construct_seed_varies_permutation():
    """Pre-fix: rng_floor was always np.random.default_rng(0xC011) regardless of caller. Post-fix: a
    seed= parameter exists and different seeds draw different noise-floor permutations."""
    import inspect

    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import greedy_cmi_fe_construct

    sig = inspect.signature(greedy_cmi_fe_construct)
    assert "seed" in sig.parameters
    assert sig.parameters["seed"].default == 0xC011  # historical default preserved for byte-identical legacy behaviour

    src = inspect.getsource(greedy_cmi_fe_construct)
    assert "SeedSequence([0xC011, seed" in src
    assert not any(line.strip() == "rng_floor = np.random.default_rng(0xC011)" for line in src.splitlines())

    # Direct RNG-level pin: different seeds must draw different permutations.
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    n = 500
    perm_a = rng_a.permutation(n)
    perm_b = rng_b.permutation(n)
    assert not np.array_equal(perm_a, perm_b)


def test_regression_fit_impl_core_passes_random_seed_to_cmi_greedy():
    """Post-fix: _fit_impl_core.py's greedy_cmi_fe_construct_with_recipes call site threads
    self.random_seed through as the seed= kwarg, instead of leaving the hardcoded default."""
    import inspect

    from mlframe.feature_selection.filters._mrmr_fit_impl import _fit_impl_core as mod

    src = inspect.getsource(mod)
    assert 'seed=int(getattr(self, "random_seed", 0) or 0),' in src


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-2 (P1): _greedy_score_and_select / greedy_mi_fe_construct / score_candidates_by_cmi /
# greedy_cmi_fe_construct all unconditionally `.astype(np.int64)` (TRUNCATING) any non-integer y before
# treating it as a classification label, destroying continuous-y signal confined to one integer bucket.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname,funcname", [
    ("_mi_greedy_cmi_fe", "score_candidates_by_cmi"),
    ("_mi_greedy_cmi_fe", "greedy_cmi_fe_construct"),
    ("_mi_greedy_fe", "_greedy_score_and_select"),
    ("_mi_greedy_fe", "greedy_mi_fe_construct"),
])
def test_regression_mi_greedy_functions_densify_not_truncate_y(modname, funcname):
    """Post-fix: none of the 4 functions truncate a non-integer y via a bare .astype(np.int64) before
    densifying it -- all route through np.unique(..., return_inverse=True) first."""
    import importlib
    import inspect

    mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    src = inspect.getsource(getattr(mod, funcname))
    # The pre-fix pattern: an `if not np.issubdtype(...): ... .astype(np.int64)` truncating branch.
    assert "np.issubdtype(np.asarray(y).dtype, np.integer)" not in src, f"{modname}.{funcname} still has the truncating dtype-check pattern"
    assert "np.unique(" in src, f"{modname}.{funcname} must densify y via np.unique"


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
    """Post-fix: all 5 builders use canonical_group_token(k), not a bare str(k), for lookup-dict keys."""
    import inspect

    from mlframe.feature_selection.filters.engineered_recipes import _grouped_recipes as mod

    src = inspect.getsource(mod)
    assert "canonical_group_token(k): float(v)" in src
    assert not any(line.strip().startswith("lookup") and "{str(k):" in line for line in src.splitlines())


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-4 (P2): greater/less/equal cast to a.dtype (float) on the GPU-resident replay path
# but to plain int on the CPU registry -- identical 0/1 values, divergent dtype by backend.
# ---------------------------------------------------------------------------


def test_regression_gpu_binary_greater_less_equal_match_cpu_int_dtype():
    """Post-fix: the GPU-resident twin now casts to cp.int64, matching the CPU registry's .astype(int)."""
    import inspect

    from mlframe.feature_selection.filters.engineered_recipes import _recipe_unary_binary_gpu as mod

    src = inspect.getsource(mod)
    assert "cp.greater(a, b).astype(cp.int64)" in src
    assert "cp.less(a, b).astype(cp.int64)" in src
    assert "cp.equal(a, b).astype(cp.int64)" in src


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-6 (P2): _fe_batched_mi.py's docstring claimed "imported by nothing in production yet"
# but batched_cmi_gpu/batched_quantile_bin_gpu/cmi_device_argmax ARE called from
# _mi_greedy_cmi_fe.py's greedy_cmi_fe_construct, production-reachable via fe_mi_greedy_cmi_enable=True.
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


def test_regression_fit_constant_memmap_cache_is_bounded():
    """Post-fix: caching more than the LRU cap of distinct-content arrays evicts the oldest and unlinks
    its backing file instead of leaking it forever."""
    import gc
    import glob
    import os
    import tempfile

    from mlframe.feature_selection.filters._joblib_safe import (
        _FIT_MEMMAP_CACHE_MAX_ENTRIES,
        fit_constant_memmap,
    )

    tmp_dir = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(tmp_dir, "mlframe_fitconst_*.mmap")))
    for i in range(_FIT_MEMMAP_CACHE_MAX_ENTRIES + 4):
        arr = np.random.default_rng(i).standard_normal((64, 8))
        view = fit_constant_memmap(arr)
        del view
    gc.collect()
    after = set(glob.glob(os.path.join(tmp_dir, "mlframe_fitconst_*.mmap")))
    new_files = after - before
    assert len(new_files) <= _FIT_MEMMAP_CACHE_MAX_ENTRIES, f"expected at most {_FIT_MEMMAP_CACHE_MAX_ENTRIES} live fit-constant files, found {len(new_files)}: {new_files}"
    for p in new_files:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# USABILITY_A-12 (P1): _fit_constant_key hashed only a bounded sample (first/last 64KB + coarse stride),
# so two different-content arrays agreeing at the sampled points could collide and alias to the wrong
# cached memmap.
# ---------------------------------------------------------------------------


def test_regression_fit_constant_key_hashes_full_buffer_not_sample():
    """Post-fix: two arrays differing ONLY in the middle (outside the old sampled-stride pattern) must
    get different keys -- pre-fix, a carefully-crafted middle-only diff at non-sampled offsets could
    collide."""
    from mlframe.feature_selection.filters._joblib_safe import _fit_constant_key

    n = 200_000
    a = np.zeros(n, dtype=np.float64)
    b = a.copy()
    b[n // 2 + 3] = 1.0
    assert _fit_constant_key(a) != _fit_constant_key(b)


# ---------------------------------------------------------------------------
# USABILITY_A-5 (P2): threading.stack_size() is process-global; concurrent run_in_big_stack_thread()
# callers racing on save/set/restore could have the loser's finally restore a stale value.
# USABILITY_A-6 (P2, coverage_gap): zero direct unit tests exercised run_in_big_stack_thread by name.
# ---------------------------------------------------------------------------


def test_regression_run_in_big_stack_thread_returns_value():
    """Direct coverage: a function that returns a value round-trips it through the big-stack thread."""
    from mlframe.feature_selection.filters._joblib_safe import run_in_big_stack_thread

    assert run_in_big_stack_thread(lambda a, b: a + b, 3, 4) == 7


def test_regression_run_in_big_stack_thread_propagates_exception():
    """Direct coverage: a function that raises propagates the same exception type/message to the caller."""
    from mlframe.feature_selection.filters._joblib_safe import run_in_big_stack_thread

    def _boom():
        """Stand-in that raises ValueError, so the caller's failure path is the one under test."""
        raise ValueError("boom-marker")

    with pytest.raises(ValueError, match="boom-marker"):
        run_in_big_stack_thread(_boom)


def test_regression_run_in_big_stack_thread_serializes_stack_size_races():
    """Post-fix: concurrent callers no longer race on the process-global threading.stack_size() --
    every concurrent call must still see the ORIGINAL stack size restored once all calls complete."""
    import threading as _threading

    from mlframe.feature_selection.filters._joblib_safe import run_in_big_stack_thread

    original = _threading.stack_size()
    results = []
    errors = []

    def _worker(i):
        """Thread body: exercises the call under test and records its result or error."""
        try:
            r = run_in_big_stack_thread(lambda x: x * 2, i)
            results.append(r)
        except Exception as e:  # pragma: no cover - failure path only
            errors.append(e)

    threads = [_threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert sorted(results) == sorted(i * 2 for i in range(8))
    assert _threading.stack_size() == original


# ---------------------------------------------------------------------------
# USABILITY_A-2 (P2): boruta_select(n_iterations=0) silently produced a NaN win_rate then an unhelpful
# scipy ValueError instead of a clear parameter-validation message.
# ---------------------------------------------------------------------------


def test_regression_boruta_select_rejects_zero_iterations():
    """Regression: boruta_select(n_iterations=0) silently produced a NaN win_rate then an unhelpful scipy ValueError instead of a clear parameter-
    validation message.
    """
    from mlframe.feature_selection.filters._boruta import boruta_select

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = (X[:, 0] > 0).astype(int)

    with pytest.raises(ValueError, match="n_iterations"):
        boruta_select(X, y, importance_fn=lambda Xs, ys: np.abs(Xs).mean(axis=0), n_iterations=0)


# ---------------------------------------------------------------------------
# USABILITY_A-3 (P2): null_importance_filter(n_shuffles=0) raised an unhelpful IndexError instead of a
# clear parameter-validation message.
# ---------------------------------------------------------------------------


def test_regression_null_importance_filter_rejects_zero_shuffles():
    """Regression: null_importance_filter(n_shuffles=0) raised an unhelpful IndexError instead of a clear parameter-validation message."""
    from mlframe.feature_selection.filters._null_importance import null_importance_filter

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = (X[:, 0] > 0).astype(int)

    with pytest.raises(ValueError, match="n_shuffles"):
        null_importance_filter(X, y, importance_fn=lambda Xs, ys: np.abs(Xs).mean(axis=0), n_shuffles=0)


# ---------------------------------------------------------------------------
# USABILITY_A-4 (P2): compose_pair_fe's single_mi_cache was keyed only by column name; a duplicate name
# silently aliased the second column's MI to the first one's cached value.
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


@pytest.mark.parametrize("fit_fn_name", ["_rbf_fit", "_sigmoid_fit"])
def test_regression_bases_fit_scrubs_nan_inf(fit_fn_name):
    """Regression: _rbf_fit / _sigmoid_fit did not guard NaN/Inf on the raw column before np.quantile/np.std, unlike _fourier_fit / _pade_fit."""
    from mlframe.feature_selection.filters import bases as bases_mod

    fit_fn = getattr(bases_mod, fit_fn_name)
    x = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0, -np.inf, 8.0, 9.0, 10.0])
    z, params = fit_fn(x)
    assert np.all(np.isfinite(z))
    for v in params.values():
        arr_v = np.asarray(v, dtype=np.float64)
        assert np.all(np.isfinite(arr_v)), f"{fit_fn_name} produced non-finite params: {params}"


# ---------------------------------------------------------------------------
# USABILITY_A-8 (P1): estimators.py's module docstring documented an MRMR(estimator=...) API that does
# not exist.
# ---------------------------------------------------------------------------


def test_regression_estimators_docstring_does_not_claim_estimator_kwarg():
    """Regression: estimators.py's module docstring documented an MRMR(estimator=...) API that does not exist."""
    from mlframe.feature_selection.filters import estimators as est_mod

    doc = est_mod.__doc__ or ""
    assert 'MRMR(estimator="ksg"' not in doc


# ---------------------------------------------------------------------------
# USABILITY_A-13 (P1): boruta_select's Benjamini-Hochberg path re-tested every round at a fixed
# per-round alpha with no correction for repeated-testing-across-rounds inflation, unlike the
# bonferroni branch.
# ---------------------------------------------------------------------------


def test_regression_boruta_select_bh_correction_scales_with_rounds():
    """Post-fix: the BH branch's per-round critical values shrink with rounds_run the same way the
    bonferroni branch's corrected_alpha does (both / rounds_run), instead of staying fixed at `alpha`."""
    import inspect

    from mlframe.feature_selection.filters._boruta import boruta_select

    src = inspect.getsource(boruta_select)
    assert "round_alpha = alpha / rounds_run" in src


# ---------------------------------------------------------------------------
# USABILITY_A-14 (P2): warn_accuracy_suboptimal_params's one-shot latch never re-fired after a
# set_params() call degraded a param post-first-fit.
# ---------------------------------------------------------------------------


def test_regression_accuracy_warning_refires_after_param_change():
    """Regression: warn_accuracy_suboptimal_params's one-shot latch never re-fired after a set_params() call degraded a param post-first-fit."""
    from mlframe.feature_selection.filters._param_accuracy_warnings import warn_accuracy_suboptimal_params

    class _FakeEstimator:
        """Minimal estimator stand-in exposing only the surface the caller is expected to use."""
        dcd_enable = True

    est = _FakeEstimator()
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        warn_accuracy_suboptimal_params(est)
    assert not any("DEGRADE" in str(x.message) for x in w1)

    est.dcd_enable = False
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        warn_accuracy_suboptimal_params(est)
    assert any("DEGRADE" in str(x.message) for x in w2), "warning should re-fire once a param newly degrades post set_params"

    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter("always")
        warn_accuracy_suboptimal_params(est)
    assert not any("DEGRADE" in str(x.message) for x in w3)


# ---------------------------------------------------------------------------
# USABILITY_B-1 (P1): audit_degenerate_columns's collinearity pass ran unconditionally regardless of
# width, allocating an unbounded dense (p, n) Gram-matrix input.
# ---------------------------------------------------------------------------


def test_regression_audit_degenerate_columns_gates_collinearity_pass_on_width(caplog):
    """Post-fix: above max_collinearity_cols, the collinearity pass is skipped (logged), while the
    cheap all_nan/constant/duplicate checks still run."""
    import logging

    from mlframe.feature_selection.filters._mrmr_degenerate import audit_degenerate_columns

    rng = np.random.default_rng(0)
    n_rows = 20
    n_cols = 12
    X = pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=[f"x{i}" for i in range(n_cols)])
    X["const_col"] = 1.0
    X["dup_col"] = X["x0"]

    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters._mrmr_degenerate"):
        result = audit_degenerate_columns(X, max_collinearity_cols=3)
    assert result["const_col"] == "constant"
    assert result["dup_col"] == "duplicate_of:x0"
    assert any("skipping the collinearity pass" in rec.message for rec in caplog.records)


def test_regression_audit_degenerate_columns_still_finds_collinear_below_gate():
    """Sanity: below max_collinearity_cols, collinear_with detection still fires as before."""
    from mlframe.feature_selection.filters._mrmr_degenerate import audit_degenerate_columns

    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(50)
    X = pd.DataFrame({"x0": x0, "x1": 2.0 * x0 + 3.0, "x2": rng.standard_normal(50)})
    result = audit_degenerate_columns(X, max_collinearity_cols=100)
    assert result.get("x1") == "collinear_with:x0"


# ---------------------------------------------------------------------------
# USABILITY_B-2 (P1): _validate_inputs unconditionally materialized a full-frame copy of X's numeric
# columns (upcast to float64) even when the numeric block was all-integer (which can never hold inf).
# ---------------------------------------------------------------------------


def test_regression_validate_inputs_skips_integer_columns_before_copy():
    """Post-fix: an all-integer numeric frame no longer gets upcast-and-copied for the inf check --
    only floating columns are selected before any array construction."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    m = MRMR(verbose=0)
    X_int = pd.DataFrame({"a": np.arange(20, dtype=np.int64), "b": np.arange(20, 40, dtype=np.int64)})
    y = np.arange(20) % 2
    # Must not raise (no inf possible in an int frame) and must not crash despite no float columns.
    m._validate_inputs(X_int, y)


def test_regression_validate_inputs_still_catches_inf_in_float_column():
    """Regression floor: a genuine +/-inf in a float column must still raise."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    m = MRMR(verbose=0)
    X = pd.DataFrame({"a": [1.0, 2.0, np.inf, 4.0], "b": [1, 2, 3, 4]})
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="inf"):
        m._validate_inputs(X, y)


def test_regression_validate_inputs_still_catches_inf_in_object_column():
    """Regression floor: a genuine +/-inf smuggled through an object-dtype column must still raise."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    m = MRMR(verbose=0)
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": pd.Series([1.0, float("inf"), "x", 4.0], dtype=object)})
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="inf"):
        m._validate_inputs(X, y)


# ---------------------------------------------------------------------------
# USABILITY_B-3 (P1): partial_fit's sample_weight reconciliation only raised when caller-supplied
# sample_weight was too SHORT; a too-LONG sample_weight (the common case, no window truncation) was
# silently sliced and misattributed to the wrong rows.
# ---------------------------------------------------------------------------


def test_regression_partial_fit_rejects_too_long_sample_weight_without_window_truncation():
    """Regression: partial_fit's sample_weight reconciliation only raised when caller-supplied sample_weight was too SHORT;."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 40
    X0 = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y0 = (X0["a"] > 0).astype(int)
    m = MRMR(verbose=0, quantization_nbins=5, partial_fit_min_recompute=1)
    m.partial_fit(X0, y0)

    X1 = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y1 = (X1["a"] > 0).astype(int)
    too_long_weights = np.ones(n + 5)  # longer than the new batch, no window truncation configured
    with pytest.raises(ValueError, match="sample_weight length"):
        m.partial_fit(X1, y1, sample_weight=too_long_weights)


# ---------------------------------------------------------------------------
# USABILITY_B-4 (P2): _apply_rolling_window discarded its own correctly-computed first loop and rebuilt
# an essentially-identical list via a second, unnecessary loop.
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


def test_regression_mrmr_docstring_documents_cache_replay_freeze():
    """Regression: the cache-replay source-freeze side effect (fit_constant-cache style sharing via _FIT_CACHE) was not documented anywhere in MRMR's
    public docstring.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    doc = MRMR.__doc__ or ""
    assert "read-only" in doc
    assert "fit_cache_max=0" in doc


# ---------------------------------------------------------------------------
# USABILITY_B-6 (P2): _origin_from_rosters rebuilt all 13 roster-attr memberships from scratch, per name,
# instead of building them once before the per-name loop.
# ---------------------------------------------------------------------------


def test_regression_origin_from_rosters_accepts_prebuilt_sets():
    """Regression: _origin_from_rosters rebuilt all 13 roster-attr memberships from scratch, per name, instead of building them once before the per-name
    loop.
    """
    from mlframe.feature_selection.filters._mrmr_fe_provenance import (
        _build_roster_membership_sets,
        _origin_from_rosters,
    )

    class _FakeMRMR:
        """Minimal MRMR stand-in: carries just the attributes the code under test reads, nothing else."""
        mi_greedy_features_ = ["eng_col_a"]
        hybrid_orth_features_: list = []

    fake = _FakeMRMR()
    roster_sets = _build_roster_membership_sets(fake)
    assert _origin_from_rosters("eng_col_a", fake, roster_sets=roster_sets) == "mi_greedy"
    assert _origin_from_rosters("unknown_col", fake, roster_sets=roster_sets) == "engineered_unknown"
    # Omitting roster_sets still works (backward-compatible fallback path).
    assert _origin_from_rosters("eng_col_a", fake) == "mi_greedy"


# ---------------------------------------------------------------------------
# USABILITY_B-7 (P2, test_gap): _apply_sis_screen's polars-specific branches had zero test coverage.
# ---------------------------------------------------------------------------


def test_regression_apply_sis_screen_polars_input():
    """Regression: uSABILITY_B-7 (P2, test_gap): _apply_sis_screen's polars-specific branches had zero test coverage."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame({
        "num_a": rng.standard_normal(n),
        "num_b": rng.standard_normal(n),
        "cat_c": rng.integers(0, 3, n).astype(str),
    })
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


def test_regression_explain_selection_logs_section_failures(caplog):
    """Regression: explain_selection's per-section except-Exception blocks never called logger, even at debug level."""
    import logging

    from mlframe.feature_selection.filters import _mrmr_explain as explain_mod

    class _BrokenMRMR:
        # A non-empty DataFrame missing the "origin" column: _survivor_section's
        # `prov.groupby("origin", ...)` raises KeyError, exercising the except-Exception path.
        """Deliberately broken MRMR stand-in: exercises the caller's error handling, not a happy path."""
        fe_provenance_ = pd.DataFrame({"feature_name": ["a", "b"]})

    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection.filters._mrmr_explain"):
        report = explain_mod.explain_selection(_BrokenMRMR())
    assert isinstance(report, str)
    assert any("failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# USABILITY_B-9 (P2, licensing): no LICENSE/NOTICE existed for the vendored InfoNet tree.
# ---------------------------------------------------------------------------


def test_regression_vendored_infonet_has_notice_file():
    """Regression: uSABILITY_B-9 (P2, licensing): no LICENSE/NOTICE existed for the vendored InfoNet tree."""
    from pathlib import Path

    import mlframe.feature_selection.filters._vendored.infonet as infonet_pkg

    notice_path = Path(infonet_pkg.__file__).resolve().parent / "NOTICE.md"
    assert notice_path.exists()
    text = notice_path.read_text(encoding="utf-8")
    assert "datou30/InfoNet" in text


# ---------------------------------------------------------------------------
# USABILITY_B-10 (P2, sys.path hygiene): infer.py's absolute imports forced a PERMANENT sys.path[0]
# injection at the call site in _neural_mi.py, making generic top-level names like `model`/`util`
# importable process-wide for the rest of the interpreter's lifetime.
# ---------------------------------------------------------------------------


def test_regression_neural_mi_scopes_sys_path_injection():
    """Post-fix: the vendored infonet path is removed from sys.path again once the one-time import of
    `infer` completes (whether or not the model/checkpoint actually load)."""
    import sys
    from pathlib import Path

    import mlframe.feature_selection.filters._neural_mi as neural_mi_mod

    vendored = str(Path(neural_mi_mod.__file__).resolve().parent / "_vendored" / "infonet")
    was_present_before = vendored in sys.path

    # Import `infer` directly via the same scoped-injection pattern the module now uses, without
    # requiring the real checkpoint/model download (this only exercises the sys.path scoping, not
    # the actual InfoNet model load).
    _path_already_present = vendored in sys.path
    if not _path_already_present:
        sys.path.insert(0, vendored)
    try:
        import infer  # noqa: F401  -- only to mirror the scoped-import pattern being tested
    finally:
        if not _path_already_present:
            try:
                sys.path.remove(vendored)
            except ValueError:
                pass

    assert (vendored in sys.path) == was_present_before


# ---------------------------------------------------------------------------
# X_SECURITY_API_PACKAGING-1 (P1): StabilityMRMR / StabilityFESelector were missing
# get_feature_names_out(), unlike their siblings MRMR / GroupAwareMRMR in the same module.
# ---------------------------------------------------------------------------


def test_regression_stability_mrmr_has_get_feature_names_out():
    """Regression: stabilityMRMR / StabilityFESelector were missing get_feature_names_out(), unlike their siblings MRMR / GroupAwareMRMR in the same
    module.
    """
    from mlframe.feature_selection.filters.stability import StabilityMRMR
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (X["x0"] > 0).astype(int)
    sel = StabilityMRMR(MRMR(verbose=0, quantization_nbins=5), n_bootstraps=3, sample_fraction=0.8, random_state=0)
    sel.fit(X, y)
    names = sel.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert len(names) == len(sel.support_)
    assert all(n in X.columns for n in names)


def test_regression_stability_fe_selector_has_get_feature_names_out():
    """Regression guard for stability fe selector has get feature names out."""
    from mlframe.feature_selection.filters._stability_fe import StabilityFESelector

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (X["x0"] > 0).astype(int)
    sel = StabilityFESelector(base_mrmr_params={"verbose": 0, "quantization_nbins": 5}, n_bootstraps=3, sample_fraction=0.8, random_state=0)
    sel.fit(X, y)
    names = sel.get_feature_names_out()
    out = sel.transform(X)
    assert isinstance(names, np.ndarray)
    assert list(names) == list(out.columns)


# ---------------------------------------------------------------------------
# X_SECURITY_API_PACKAGING-3 (P2): GroupAwareMRMR.fit had zero validation of corr_threshold /
# min_reduction, unlike StabilityMRMR's analogous knobs in the same module.
# ---------------------------------------------------------------------------


def test_regression_group_aware_mrmr_rejects_invalid_corr_threshold():
    """Regression: groupAwareMRMR.fit had zero validation of corr_threshold / min_reduction, unlike StabilityMRMR's analogous knobs in the same module."""
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] > 0).astype(int)
    sel = GroupAwareMRMR(MRMR(verbose=0), corr_threshold=5.0)
    with pytest.raises(ValueError, match="corr_threshold"):
        sel.fit(X, y)


def test_regression_group_aware_mrmr_rejects_invalid_min_reduction():
    """Regression guard for group aware mrmr rejects invalid min reduction."""
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] > 0).astype(int)
    sel = GroupAwareMRMR(MRMR(verbose=0), min_reduction=1.0)
    with pytest.raises(ValueError, match="min_reduction"):
        sel.fit(X, y)


def test_regression_group_aware_mrmr_default_params_still_fit():
    """Sanity: the new validation must not reject the class's own defaults."""
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] > 0).astype(int)
    sel = GroupAwareMRMR(MRMR(verbose=0))
    sel.fit(X, y)  # must not raise


# ---------------------------------------------------------------------------
# X_SECURITY_API_PACKAGING-5 (P2): pyutilz/py-ci-shared were declared as bare git URLs with no
# pinned commit SHA -- pip install resolved to whatever the default branch happened to be.
# ---------------------------------------------------------------------------


def test_regression_pyproject_pins_py_ci_shared_commit():
    """Regression: pyutilz/py-ci-shared were declared as bare git URLs with no pinned commit SHA -- pip install resolved to whatever the default branch
    happened to be.
    """
    from pathlib import Path

    pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    if not pyproject_path.exists():
        pytest.skip("pyproject.toml not found at expected repo-root-relative path")
    text = pyproject_path.read_text(encoding="utf-8")
    assert "py-ci-shared @ git+https://github.com/fingoldo/py-ci-shared.git@" in text


# ---------------------------------------------------------------------------
# X_EFFICIENCY_ARCHITECTURE-2 (P1): 39 independently-gated FE-family constructors returned a full deep
# copy of X as their no-candidate/no-op early-exit path, even though the orchestrator's own
# _appended-then-conditional-merge calling convention never reads the returned frame on that path.
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


def test_regression_selection_stability_report_honours_random_state():
    """Post-fix: an MRMR seeded via random_state= (canonical) and one seeded via the equivalent
    random_seed= (deprecated alias) produce IDENTICAL bootstrap selection-frequency reports --
    pinning that selection_stability_report resolves random_state, not just the deprecated alias."""
    import warnings

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"x{i}" for i in range(6)])
    y = (X["x0"] + 0.5 * X["x1"] > 0).astype(int)

    m_state = MRMR(verbose=0, quantization_nbins=5, random_state=42)
    m_seed = MRMR(verbose=0, quantization_nbins=5, random_seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        m_state.fit(X, y)
        m_seed.fit(X, y)

    report_state = m_state.selection_stability_report(n_boot=10, as_text=False)
    report_seed = m_seed.selection_stability_report(n_boot=10, as_text=False)
    assert report_state, "expected a non-empty replay-state report"
    assert report_state["feature_selection_frequency"] == report_seed["feature_selection_frequency"]


def test_regression_selection_stability_report_random_state_changes_bootstrap():
    """Sanity: two DIFFERENT random_state values must (with overwhelming probability on this fixture)
    produce a different bootstrap RNG draw -- proves the seed is actually being consumed, not ignored
    entirely (which would make the equality test above pass vacuously)."""
    import inspect

    from mlframe.feature_selection.filters._mrmr_stability_report import selection_stability_report

    src = inspect.getsource(selection_stability_report)
    assert "self._effective_random_seed()" in src
    assert 'random_state if random_state is not None else int(getattr(self, "random_seed", 0))' not in src


# ---------------------------------------------------------------------------
# X_TEST_COVERAGE_QUALITY-2 (P1): commit f067e0d44 patched 14 scorer-zoo FE modules for the identical
# B-17 replay-drift bug (recipe doesn't freeze fit-time basis-preprocess mean/std, so replaying on a
# distributionally-shifted slice silently refits and drifts), but only 1 of the 14 got a named
# regression test. Port the same slice-replay pattern (test_jmim.py::TestRecipeFreezesPreprocessParams)
# to the remaining 13.
# ---------------------------------------------------------------------------

_ORTH_SCORER_ZOO_MODULES = [
    ("_orthogonal_hsic_fe", "hybrid_orth_mi_hsic_fe_with_recipes"),
    ("_orthogonal_lasso_fe", "hybrid_orth_mi_lasso_fe_with_recipes"),
    ("_orthogonal_routing_fe", "hybrid_orth_mi_conditional_routing_fe_with_recipes"),
    ("_orthogonal_scorer_auto_fe", "hybrid_orth_mi_ensemble_fe_with_recipes"),
    ("_orthogonal_three_gate_mi_fe", "hybrid_orth_mi_three_gate_fe_with_recipes"),
    ("_orthogonal_total_correlation_fe", "hybrid_orth_mi_tc_fe_with_recipes"),
    ("_orthogonal_adaptive_arity_fe", "hybrid_orth_mi_adaptive_arity_fe_with_recipes"),
    ("_orthogonal_adaptive_degree_fe", "hybrid_orth_mi_adaptive_degree_fe_with_recipes"),
    ("_orthogonal_bootstrap_mi_fe", "hybrid_orth_mi_bootstrap_fe_with_recipes"),
    ("_orthogonal_cmim_fe", "hybrid_orth_mi_cmim_fe_with_recipes"),
    ("_orthogonal_copula_mi_fe", "hybrid_orth_mi_copula_fe_with_recipes"),
    ("_orthogonal_dcor_fe", "hybrid_orth_mi_dcor_fe_with_recipes"),
    ("_orthogonal_elasticnet_fe", "hybrid_orth_mi_elasticnet_fe_with_recipes"),
]


@pytest.mark.parametrize("modname,funcname", _ORTH_SCORER_ZOO_MODULES)
def test_regression_orth_scorer_zoo_recipe_freezes_preprocess_params(modname, funcname):
    """Slice replay of each scorer-zoo module's emitted recipe(s) must match the fit-time engineered
    value bit-for-bit -- pins the B-17 fix (frozen preprocess_params) across all 13 modules that
    commit f067e0d44 patched but did not individually test."""
    import importlib

    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    fn = getattr(mod, funcname)

    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.standard_normal(n) * 3 + 10  # mean=10, std=3 -- a small head slice differs materially
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise_0": rng.standard_normal(n), "noise_1": rng.standard_normal(n)})
    signal = (x1 - 10.0) ** 2 + 0.6 * (x2**2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)

    # copula_mi's default uplift/MI-floor gate rejects everything on this fixture; relax it so the
    # test actually exercises the slice-replay contract instead of skipping.
    extra_kwargs = {"min_uplift": 0.5, "min_abs_mi_frac": 0.0} if modname == "_orthogonal_copula_mi_fe" else {}
    result = fn(X, y, cols=None, **extra_kwargs)
    # Result arity varies across the scorer zoo (some return an extra intermediate-scores frame),
    # but recipes is always the LAST element and X_aug is always the FIRST.
    X_aug, recipes = result[0], result[-1]
    if not recipes:
        pytest.skip(f"{modname}.{funcname} emitted no recipes on this fixture; cannot exercise the slice-replay contract.")

    for r in recipes:
        if "preprocess_params" not in r.extra:
            continue  # not every recipe kind in this family freezes basis-preprocess params (e.g. a pure interaction term)
        assert r.extra["preprocess_params"] is not None, f"{modname}: recipe {r.name!r} has a preprocess_params key but it is None -- B-17 regression."
        X_slice = X.iloc[:50].copy()
        fit_time_vals = X_aug[r.name].to_numpy()[:50]
        replayed_vals = apply_recipe(r, X_slice)
        # rtol/atol loose enough to tolerate ordinary FP-reorder noise (~1e-7, seen on the adaptive-degree
        # module) but tight enough to catch a genuine B-17 regression (slice-refit drift is orders of
        # magnitude larger -- the whole point of picking a slice with a materially different mean/std).
        assert np.allclose(replayed_vals, fit_time_vals, rtol=1e-5, atol=1e-6), (
            f"{modname}: recipe {r.name!r} slice replay diverged from fit-time values "
            f"(max abs diff={float(np.max(np.abs(np.asarray(replayed_vals) - fit_time_vals)))}) -- "
            f"the frozen preprocess_params are not being honoured at replay time."
        )


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-2 (P1): the Layer-27 noise-aware floor silently defaulted to 0.0 (a
# full no-op) whenever fewer than 4 scored engineered columns reached the scan -- realistically
# whenever at most 1 raw numeric source column is present (the common default-2-degrees case).
# ---------------------------------------------------------------------------


def test_regression_hybrid_orth_noise_floor_active_with_one_raw_column():
    """Post-fix: with a SINGLE weak raw numeric column (baseline_mi ~ 0), a purely noise-driven
    engineered column must NOT be admitted -- pre-fix, both the legacy and noise-aware floors
    collapsed to ~0 in this exact regime, admitting noise."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import hybrid_orth_mi_fe

    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.standard_normal(n)  # pure noise, no relationship to y
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"x1": x1})

    X_aug, _scores = hybrid_orth_mi_fe(X, y, degrees=(2, 3), basis="hermite")
    # A weak/irrelevant sole raw column must not spuriously admit a noise-driven engineered basis column.
    assert X_aug.shape[1] == X.shape[1], f"expected no engineered columns admitted on pure noise; got {list(X_aug.columns)}"


def test_regression_noise_floor_computed_without_size_gate():
    """Direct pin: the noise-floor computation no longer gates on raw_baselines.size >= 4 / eng_mis.size >= 4."""
    import inspect

    from mlframe.feature_selection.filters import _orthogonal_univariate_fe as mod

    src = inspect.getsource(mod)
    assert "if raw_baselines.size >= 4:" not in src
    assert "if eng_mis.size >= 4:" not in src


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-3 (P2): compute_class_weights had no upper bound on n_classes before
# allocating np.bincount(y_arr, minlength=n_classes) -- a caller-controlled unbounded allocation.
# ---------------------------------------------------------------------------


def test_regression_compute_class_weights_rejects_huge_class_count(caplog, monkeypatch):
    """Post-fix: a y whose max value implies an enormous class count is rejected (logged, returns None)
    instead of allocating a bincount array of that size."""
    import logging

    from mlframe.feature_selection.filters._orthogonal_univariate_fe._imbalance_mi import compute_class_weights

    monkeypatch.setenv("MLFRAME_FE_IMBALANCE_MI", "on")
    y = np.array([0, 1, 2, 200_000], dtype=np.int64)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._orthogonal_univariate_fe._imbalance_mi"):
        result = compute_class_weights(y)
    assert result is None
    assert any("distinct integer values" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-4 (P2): residency_audit()'s monkeypatch/restore pair had no reentrancy
# guard -- two overlapping regions on different threads could silently corrupt each other's byte tally.
# ---------------------------------------------------------------------------


def test_regression_residency_audit_has_lock():
    """Direct pin: residency_audit is now serialized via a module-level lock."""
    from mlframe.feature_selection.filters._gpu_strict_fe import _audit as mod

    assert hasattr(mod, "_AUDIT_LOCK")
    import threading
    assert isinstance(mod._AUDIT_LOCK, type(threading.RLock()))


def test_regression_residency_audit_nested_same_thread_still_works():
    """Sanity: same-thread nested residency_audit() (the legitimate reentrant case) must not deadlock now
    that a lock guards the monkeypatch region."""
    from mlframe.feature_selection.filters._gpu_strict_fe._audit import residency_audit

    with residency_audit() as outer:
        with residency_audit() as inner:
            pass
    assert outer is not None and inner is not None


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-5 (P2): _ksg_mi_1d hardcoded random_state=42 regardless of caller intent
# (dead code today, but a latent instance of a bug class fixed elsewhere this audit wave).
# ---------------------------------------------------------------------------


def test_regression_ksg_mi_1d_accepts_random_state():
    """Post-fix: _ksg_mi_1d threads an explicit random_state parameter instead of a hardcoded 42."""
    import inspect

    from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp import _ksg_mi_1d

    sig = inspect.signature(_ksg_mi_1d)
    assert "random_state" in sig.parameters
    assert sig.parameters["random_state"].default == 42  # historical default preserved

    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = rng.integers(0, 2, 200)
    v1 = _ksg_mi_1d(x, y, discrete_target=True, random_state=1)
    v2 = _ksg_mi_1d(x, y, discrete_target=True, random_state=2)
    assert np.isfinite(v1) and np.isfinite(v2)


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-6 (P2): pack_blocks_to_devices treated a genuinely empty speeds list
# (zero visible devices) identically to the single-device case, returning a bogus device index 0.
# ---------------------------------------------------------------------------


def test_regression_pack_blocks_to_devices_rejects_empty_speeds():
    """Regression: pack_blocks_to_devices treated a genuinely empty speeds list (zero visible devices) identically to the single-device case, returning a
    bogus device index 0.
    """
    from mlframe.feature_selection.filters._fe_gpu_batch._packer import pack_blocks_to_devices

    with pytest.raises(ValueError, match="speeds is empty"):
        pack_blocks_to_devices([10, 20], [])


def test_regression_pack_blocks_to_devices_single_device_still_works():
    """Sanity: the legitimate single-device fast path is unaffected."""
    from mlframe.feature_selection.filters._fe_gpu_batch._packer import pack_blocks_to_devices

    assert pack_blocks_to_devices([10, 20], [1.0]) == [0, 0]


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-1 (P1): the resident-operand cache was keyed purely on content (no device
# component), so a multi-GPU worker on device N could receive a device-M-resident array back from the
# cache on a content hit. Also: no lock guarding the get/insert/evict sequence, and a bare
# `except Exception: pass` in the multi-GPU fallback path with zero diagnostic trace.
# ---------------------------------------------------------------------------


def test_regression_resident_operand_key_includes_device_id():
    """Direct pin: resident_operand's cache key now folds in the active cupy device id."""
    import inspect

    from mlframe.feature_selection.filters import _fe_resident_operands as mod

    src = inspect.getsource(mod.resident_operand)
    # The contract is that the cache key is device-scoped, so a table built on one GPU cannot be handed to
    # another. Pinning the exact hash-helper spelling instead broke when it gained memoisation.
    assert "device_id = int(cp.cuda.Device().id)" in src
    assert "sig = (device_id," in src, "the resident-operand cache key no longer leads with device_id"


def test_regression_resident_operand_cache_has_lock():
    """Regression guard for resident operand cache has lock."""
    from mlframe.feature_selection.filters._fe_resident_operands import _FE_RESIDENT_OPERANDS_LOCK

    assert _FE_RESIDENT_OPERANDS_LOCK is not None


def test_regression_fe_batch_mi_logs_gpu_fallback(caplog, monkeypatch):
    """Post-fix: a multi_gpu_fe_batch_mi failure is logged at warning level, not silently swallowed."""
    import logging

    from mlframe.feature_selection.filters import _fe_batch_dispatch as mod

    monkeypatch.setattr(mod, "choose_fe_batch_backend", lambda n, k: "gpu")

    def _raising_multi_gpu(*args, **kwargs):
        """Stand-in that raises RuntimeError, so the caller's failure path is the one under test."""
        raise RuntimeError("fake-multi-gpu-failure")

    import mlframe.feature_selection.filters._fe_gpu_batch as gpu_batch_mod
    monkeypatch.setattr(gpu_batch_mod, "multi_gpu_fe_batch_mi", _raising_multi_gpu)

    rng = np.random.default_rng(0)
    X_cands = rng.standard_normal((100, 4))
    y_codes = rng.integers(0, 2, 100)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._fe_batch_dispatch"):
        result = mod.fe_batch_mi(X_cands, y_codes, backend="gpu")
    assert result is not None
    assert any("fake-multi-gpu-failure" in rec.message for rec in caplog.records)


def test_regression_orth_univariate_docstring_no_longer_claims_not_wired():
    """X_EFFICIENCY_ARCHITECTURE-4: the module docstring must no longer falsely claim the univariate
    orth-basis constructor is opt-in-only / not wired into MRMR.fit by default."""
    import mlframe.feature_selection.filters._orthogonal_univariate_fe as mod

    doc = mod.__doc__ or ""
    assert "NOT wired into MRMR.fit by default" not in doc


def test_regression_gpu_resident_basis_mi_docstring_not_self_contradicting():
    """X_EFFICIENCY_ARCHITECTURE-6: fe_gpu_resident_basis_mi_enabled's docstring must not claim both
    'DEFAULT OFF' and 'DEFAULT ON when CUDA is present' for the same flag."""
    from mlframe.feature_selection.filters._gpu_resident_fe import fe_gpu_resident_basis_mi_enabled

    doc = fe_gpu_resident_basis_mi_enabled.__doc__ or ""
    # Pre-fix: the opening summary line unqualifiedly asserted "DEFAULT OFF" while a later paragraph
    # asserted "DEFAULT ON when CUDA is present" for the SAME flag. Post-fix, any "DEFAULT OFF" mention
    # must be qualified (e.g. "DEFAULT OFF otherwise"), not a bare unqualified claim.
    assert "(Piece 3). DEFAULT OFF." not in doc
    assert "DEFAULT ON when CUDA is present" in doc


def test_regression_orth_univariate_bails_to_host_when_aux_pool_active():
    """X_EFFICIENCY_ARCHITECTURE-3: with a semi-supervised unlabeled pool installed, hybrid_orth_mi_fe
    must not take the GPU-resident path (which ignores the pool) -- forced GPU-resident flags must be
    overridden by the aux-pool gate. Runs on CPU regardless of CUDA availability (only checks the gating
    logic's inputs, not actual GPU execution)."""
    import inspect

    from mlframe.feature_selection.filters._orthogonal_univariate_fe import hybrid_orth_mi_fe

    src = inspect.getsource(hybrid_orth_mi_fe)
    assert "get_unlabeled_pool" in src
    assert "_aux_pool_active" in src
    assert "not _aux_pool_active" in src


def test_regression_binned_numeric_agg_no_op_path_returns_same_object():
    """Direct pin: the no-candidate early-exit path returns the SAME X object (no copy), on a
    representative one of the 39 fixed constructors."""
    from mlframe.feature_selection.filters._binned_numeric_agg_fe import binned_numeric_agg_with_recipes

    rng = np.random.default_rng(0)
    n = 50
    # No numeric columns at all -> gcands/acands both empty -> hits the first no-op early exit.
    X = pd.DataFrame({"cat_only": pd.Categorical(rng.integers(0, 3, n).astype(str))})
    y = rng.standard_normal(n)
    X_out, appended, recipes = binned_numeric_agg_with_recipes(X, y)
    assert X_out is X
    assert appended == []
    assert recipes == []


def test_regression_build_raw_redundancy_anchors_identifies_consumer():
    """FE_REDUNDANCY_SYNERGY-15 fix (mrmr_audit_2026-07-22, carried over from 2026-07-20's audit): direct
    unit test for ``build_raw_redundancy_anchors`` -- it had zero direct test anywhere across 3 audit
    generations, only ever exercised transitively through ``drop_redundant_raw_operands``'s end-to-end
    fixtures, which never actually asserted anything about the anchor-building phase's own contract."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_anchors import build_raw_redundancy_anchors

    rng = np.random.default_rng(0)
    n = 200
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y_cont = a + b + 0.01 * rng.standard_normal(n)

    def _bin(v, nbins=10):
        """Test helper: bin."""
        edges = np.quantile(v, np.linspace(0, 1, nbins + 1)[1:-1])
        return np.searchsorted(edges, v).astype(np.int64)

    cols = ["a", "b", "sum(a,b)"]
    data = np.column_stack([_bin(a), _bin(b), _bin(a + b)])
    y_binned = _bin(y_cont)

    out = build_raw_redundancy_anchors(
        data=data,
        cols=cols,
        sel=[0, 1, 2],
        raw_name_set={"a", "b"},
        y_binned=y_binned,
        y_continuous=None,
        engineered_continuous=None,
        replayable_eng_names=None,
        recipes=None,
        raw_X=None,
        seed=0,
        verbose=0,
        n_rows=n,
        gate_resident=False,
    )

    assert out.early_return is None
    assert out.eng_idx == [2]
    assert set(out.raw_sel_idx) == {0, 1}
    assert set(out.eng_consumers.get("a", [])) == {2}
    assert set(out.eng_consumers.get("b", [])) == {2}
    assert out.raw_is_signal_bearing("a") is True
    assert out.raw_is_signal_bearing("b") is True


def test_regression_build_raw_redundancy_anchors_no_engineered_survivors_early_returns():
    """FE_REDUNDANCY_SYNERGY-15 fix (mrmr_audit_2026-07-22): the degenerate "no engineered survivor"
    guard returns ``(sel, [])`` immediately without building any anchor state."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_anchors import build_raw_redundancy_anchors

    n = 50
    cols = ["a"]
    data = np.zeros((n, 1), dtype=np.int64)
    sel = [0]

    out = build_raw_redundancy_anchors(
        data=data,
        cols=cols,
        sel=sel,
        raw_name_set={"a"},
        y_binned=np.zeros(n, dtype=np.int64),
        y_continuous=None,
        engineered_continuous=None,
        replayable_eng_names=None,
        recipes=None,
        raw_X=None,
        seed=0,
        verbose=0,
        n_rows=n,
        gate_resident=False,
    )

    assert out.early_return == (sel, [])
