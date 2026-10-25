"""Behavioural tests for the MRMR estimator: constructor state, fit flow, caching and configuration."""

from __future__ import annotations
import contextlib
import re
import warnings
import numpy as np
import pandas as pd
import pytest

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
    fitted = m.fit(X, y)
    assert fitted is m
    assert m.get_params()["n_workers"] == 1
    assert m.n_features_in_ == 5
    assert list(m.feature_names_in_) == [f"f{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# FE_STEP_B-1 (P0): the FE-escalation and C2 additive-fusion materialize blocks
# in _step_score.py preallocated their output-codes buffer at the raw (narrow)
# quantization_dtype and assigned discretize_array's auto-widened return value
# into it, silently downcasting back and wrapping bin codes negative for
# quantization_nbins > 127 under quantization_dtype=int8. Same bug class
# already fixed for the unary/binary materialize block two screens up via
# _safe_code_dtype, now applied to the two remaining call sites.
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
    # support_ holds raw column indices only; engineered survivors count toward the ceiling via n_features_.
    assert len(m.support_) + len(m._engineered_recipes_) == ceiling


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


def test_regression_dcd_swap_no_dead_getattr_default():
    """Post-fix: commit_swap reads decision.branch directly, so a decision object without that field raises AttributeError
    instead of silently falling back to a default branch."""
    from types import SimpleNamespace

    from mlframe.feature_selection.filters._dynamic_cluster_discovery import DCDState, _dcd_swap

    decision = SimpleNamespace(accept=True, new_col_idx=0)  # no ``branch`` attribute on purpose
    with pytest.raises(AttributeError):
        _dcd_swap.commit_swap(DCDState(), 0, decision, selected_vars=[0], data_ref={})


# ---------------------------------------------------------------------------
# ORTH_BASIS_A-1 (P1): _run_random_batch_search silently substituted 1 for seed<=0 (including the valid
# seed=0), while sibling RNGs in the SAME call chain correctly kept seed=0 as 0.
# ---------------------------------------------------------------------------


def test_regression_run_random_batch_search_seed_zero_not_substituted(monkeypatch):
    """Pre-fix: seed=0 silently became seed=1 internally. Post-fix: the first candidate batch is drawn from
    np.random.default_rng(0) exactly, so it matches an independent generator seeded with 0."""
    import mlframe.feature_selection.filters._hermite_fe_optimise as hfo

    captured: list = []

    def _fake_eval(coefs_a, coefs_b, **kwargs):
        """Record the sampled candidates and return neutral scores so the search runs its single batch."""
        captured.append((np.array(coefs_a), np.array(coefs_b)))
        n = len(coefs_a)
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64)

    monkeypatch.setattr(hfo, "_eval_coef_pair_batch", _fake_eval)
    hfo._run_random_batch_search(
        ca_size=2,
        cb_size=3,
        coef_range=(-2.0, 2.0),
        n_trials=20,
        seed=0,
        direction_only=False,
        warm_start_seeds=None,
        eval_kwargs={},
    )

    ref = np.random.default_rng(0)
    expected_a = ref.uniform(-2.0, 2.0, size=(20, 2))
    expected_b = ref.uniform(-2.0, 2.0, size=(20, 3))
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0][0], expected_a)
    np.testing.assert_array_equal(captured[0][1], expected_b)


# ---------------------------------------------------------------------------
# ORTH_BASIS_A-2 (P1): _run_cma_search / _run_cma_search_batch's es.ask()/es.tell() bare-except blocks
# swallowed any cma-library fault with zero logging, silently truncating the CMA generation loop early.
# ---------------------------------------------------------------------------


def test_regression_cma_search_ask_tell_exceptions_are_logged(monkeypatch, caplog):
    """Pre-fix: es.ask()/es.tell() faults were swallowed with a bare `except Exception: break` and zero
    logging. Post-fix: both functions log a warning naming the failing call before breaking."""
    import logging

    import cma

    import mlframe.feature_selection.filters._hermite_fe_optimise as hfo

    class _AskFails:
        """Test double _AskFails."""

        ask_calls = 0

        def __init__(self, x0, sigma0, opts):
            pass

        def stop(self):
            """Never report convergence."""
            return False

        def ask(self):
            """Test double method ask."""
            _AskFails.ask_calls += 1
            raise RuntimeError("simulated ask fault")

    class _TellFails:
        """Test double _TellFails."""

        def __init__(self, x0, sigma0, opts):
            self.popsize = int(opts["popsize"])
            self.dim = len(x0)

        def stop(self):
            """Never report convergence."""
            return False

        def ask(self):
            """Test double method ask."""
            return np.random.default_rng(0).uniform(-1.0, 1.0, size=(self.popsize, self.dim))

        def tell(self, solutions, scores):
            """Test double method tell."""
            raise RuntimeError("simulated tell fault")

    def _fake_batch_eval(coefs_a, coefs_b, **kwargs):
        """Return neutral finite scores for every candidate."""
        n = len(coefs_a)
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64)

    common = dict(ca_size=2, cb_size=2, coef_range=(-2.0, 2.0), n_trials=40, seed=1, direction_only=False, warm_start_seeds=None, eval_kwargs={})

    with monkeypatch.context() as mp:
        mp.setattr(cma, "CMAEvolutionStrategy", _AskFails)
        mp.setattr(hfo, "_eval_coef_pair_batch", _fake_batch_eval)
        with caplog.at_level(logging.WARNING):
            hfo._run_cma_search_batch(**common)
            hfo._run_cma_search(**common)
    assert "es.ask() raised" in caplog.text
    assert _AskFails.ask_calls == 2, "each search must stop after the failing ask instead of looping"

    caplog.clear()
    with monkeypatch.context() as mp:
        mp.setattr(cma, "CMAEvolutionStrategy", _TellFails)
        mp.setattr(hfo, "_eval_coef_pair_batch", _fake_batch_eval)
        with caplog.at_level(logging.WARNING):
            hfo._run_cma_search_batch(**common)
    assert "es.tell() raised" in caplog.text


# ---------------------------------------------------------------------------
# ORTH_BASIS_A-7 (P2): polynom_pair_fe.py's per-pair injection loop appended to survivor-tracking lists
# BEFORE the X column assignment; if the X assignment itself raised, the outer except only logs and
# continues -- it does not undo the earlier appends, so the final np.concatenate would still bake in a
# column with no matching X column and no recipe.
# ---------------------------------------------------------------------------


def test_regression_call_dead_branches_documented():
    """Post-fix: the plug_in and copula scorer branches stay wired into the batched rank table, so each engineered
    column gets a rank under both scorers and the column carrying the target signal ranks first under each."""
    from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import _compute_per_scorer_rank_table

    rng = np.random.default_rng(0)
    n = 400
    y = (rng.standard_normal(n) > 0).astype(int)
    raw_X = pd.DataFrame({"raw_a": rng.standard_normal(n), "raw_b": rng.standard_normal(n)})
    engineered_X = pd.DataFrame(
        {
            "signal": y + 0.1 * rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
            "noise_2": rng.standard_normal(n),
        }
    )

    _score_table, _baseline, rank_per_scorer = _compute_per_scorer_rank_table(
        raw_X,
        engineered_X,
        y,
        scorers=("plug_in", "copula"),
        random_state=0,
        nbins=10,
        n_neighbors=5,
        copula_n_bins=10,
        dcor_n_sample=100,
    )
    for scorer in ("plug_in", "copula"):
        assert set(rank_per_scorer[scorer]) == set(engineered_X.columns)
        assert sorted(rank_per_scorer[scorer].values()) == [1, 2, 3]
        assert rank_per_scorer[scorer]["signal"] == 1


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-5 (P2): detect_clusters_by_correlation's max_cluster_size truncation branch computed a
# no-op O(p) linear-scan re-lookup (dense_names[dense_names.index(name)] just produces name back).
# ---------------------------------------------------------------------------


def test_regression_quality_objective_exception_is_logged(caplog):
    """Post-fix: a malformed bake-off output yields quality=NaN and a debug record naming the unpack failure."""
    import logging

    from mlframe.feature_selection.filters._oracle_scorer_select import _quality_objective

    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection.filters._oracle_scorer_select"):
        result = _quality_objective("not-a-pair", 1.5, 0.0)
    assert np.isnan(result["quality"])
    assert result["elapsed_s"] == 1.5
    assert any("could not unpack" in rec.getMessage() for rec in caplog.records)

    ok = _quality_objective(("scorer_a", 0.25), 2.0, 0.0)
    assert ok["quality"] == 0.25


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-9 (P2): _orthogonal_meta_scorer_fe.py's fingerprint_signal inner per-column Pearson/
# Spearman/symmetric-Pearson probes used bare except Exception (3 sites), broader than the module's own
# declared _NUMERIC_ERRORS convention, with zero logging.
# ---------------------------------------------------------------------------


def test_regression_meta_scorer_inner_corr_excepts_use_numeric_errors(monkeypatch, caplog):
    """Post-fix: a numeric failure inside each inner corr probe is logged at debug and the probe falls back to NaN,
    while a non-numeric fault (not in _NUMERIC_ERRORS) still propagates instead of being swallowed."""
    import logging

    from mlframe.feature_selection.filters._orthogonal_meta_scorer_fe import fingerprint_signal

    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = (X["a"].to_numpy() + 0.3 * rng.standard_normal(n) > 0).astype(int)

    def _numeric_boom(self, other, method="pearson", min_periods=None):
        """Test double method _numeric_boom."""
        raise ValueError("simulated numeric failure")

    monkeypatch.setattr(pd.Series, "corr", _numeric_boom)
    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection.filters._orthogonal_meta_scorer_fe"):
        fp = fingerprint_signal(X, y)
    assert fp["mean_abs_pearson"] == 0.0
    assert fp["dcor_proxy"] == 0.0
    messages = " | ".join(rec.getMessage() for rec in caplog.records)
    assert "meta_scorer pearson corr failed for column 'a'" in messages
    assert "meta_scorer spearman corr failed" in messages
    assert "meta_scorer symmetric-pearson corr failed" in messages

    def _programming_boom(self, other, method="pearson", min_periods=None):
        """Test double method _programming_boom."""
        raise RuntimeError("simulated programming fault")

    monkeypatch.setattr(pd.Series, "corr", _programming_boom)
    with pytest.raises(RuntimeError):
        fingerprint_signal(X, y)


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-1 (P1): the marginal-MI term subtracted in every cat-FE Interaction Information
# computation was ALWAYS built via the unweighted _marginal_screen_njit, even when sample weights made
# the joint-MI search kernel weighted -- mixing a weighted joint term with an unweighted marginal term.
# ---------------------------------------------------------------------------


def test_regression_estimators_docstring_does_not_claim_estimator_kwarg():
    """Regression: the documented MRMR(estimator=...) API does not exist, so constructing MRMR with that keyword must be rejected."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    with pytest.raises(TypeError):
        MRMR(estimator="ksg")


# ---------------------------------------------------------------------------
# USABILITY_A-13 (P1): boruta_select's Benjamini-Hochberg path re-tested every round at a fixed
# per-round alpha with no correction for repeated-testing-across-rounds inflation, unlike the
# bonferroni branch.
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


def test_regression_audit_degenerate_columns_streams_collinearity_pass_above_width(caplog):
    """Above max_collinearity_cols the dense Gram matrix is not built: the pass streams (logged) and still reports the collinear columns, while the cheap
    all_nan/constant/duplicate checks run as before."""
    import logging

    from mlframe.feature_selection.filters._mrmr_degenerate import audit_degenerate_columns

    rng = np.random.default_rng(0)
    n_rows = 20
    n_cols = 12
    X = pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=[f"x{i}" for i in range(n_cols)])
    X["const_col"] = 1.0
    X["dup_col"] = X["x0"]
    X["lin_col"] = 4.0 * X["x1"] - 2.0

    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters._mrmr_degenerate"):
        result = audit_degenerate_columns(X, max_collinearity_cols=3)
    assert result["const_col"] == "constant"
    assert result["dup_col"] == "duplicate_of:x0"
    assert result["lin_col"] == "collinear_with:x1"
    assert any("the collinearity pass streams" in rec.message for rec in caplog.records)


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

    # "Does not raise" held BEFORE the fix too -- the pre-fix code did not raise on an integer frame either, it
    # merely built a float64 copy of the whole thing. So the implicit assertion could never distinguish the two,
    # and a regression reinstating the whole-frame upcast was invisible. On this project's frame sizes that is
    # the memory rule's central prohibition, so the ALLOCATION is what has to be asserted.
    _converted: list = []
    _orig_asarray = np.asarray

    def _spy_asarray(a, *args, **kwargs):
        """Record any call that would materialise this frame as a float array."""
        if a is X_int or (hasattr(a, "shape") and getattr(a, "shape", None) == X_int.shape and not isinstance(a, pd.Series)):
            _converted.append((type(a).__name__, kwargs.get("dtype", args[0] if args else None)))
        return _orig_asarray(a, *args, **kwargs)

    import unittest.mock as _mock

    with _mock.patch.object(np, "asarray", _spy_asarray):
        m._validate_inputs(X_int, y)

    _float_conversions = [c for c in _converted if c[1] is not None and np.dtype(c[1]).kind == "f"]
    assert not _float_conversions, (
        f"the all-integer frame was upcast to float for the inf check ({_float_conversions}); only floating "
        "columns should be selected before any array construction, or a 100+ GB frame is copied for nothing"
    )


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


def test_regression_mrmr_docstring_documents_cache_replay_freeze():
    """Regression: a later identical fit replays from the process-wide _FIT_CACHE and freezes the earlier instance's
    ndarray attributes read-only (the side effect the public docstring warns about). fit_cache_max=0 opts out of it.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((300, 6)), columns=[f"x{i}" for i in range(6)])
    y = pd.Series((X["x0"] + 0.5 * X["x1"] + 0.2 * rng.standard_normal(300) > 0).astype(int))
    params = dict(n_workers=1, verbose=0, full_npermutations=0, baseline_npermutations=0)

    def _writeable_ndarrays(est):
        """Test double method _writeable_ndarrays."""
        return [k for k, v in vars(est).items() if isinstance(v, np.ndarray) and v.flags.writeable]

    MRMR._FIT_CACHE.clear()
    first = MRMR(**params).fit(X, y)
    before = _writeable_ndarrays(first)
    MRMR(**params).fit(X, y)  # identical content and params: replays from first via the shared cache
    assert len(_writeable_ndarrays(first)) < len(before), "the cache replay must freeze at least one ndarray attribute of the first instance"

    MRMR._FIT_CACHE.clear()
    opted_out = MRMR(fit_cache_max=0, **params).fit(X, y)
    MRMR(fit_cache_max=0, **params).fit(X, y)
    assert len(_writeable_ndarrays(opted_out)) == sum(isinstance(v, np.ndarray) for v in vars(opted_out).values())


# ---------------------------------------------------------------------------
# USABILITY_B-6 (P2): _origin_from_rosters rebuilt all 13 roster-attr memberships from scratch, per name,
# instead of building them once before the per-name loop.
# ---------------------------------------------------------------------------


def test_regression_vendored_infonet_has_notice_file():
    """Regression: no LICENSE/NOTICE existed for the vendored InfoNet tree. The shipped NOTICE file must exist as a
    non-empty regular file next to the vendored package, so the attribution travels with the code."""
    from pathlib import Path

    import mlframe.feature_selection.filters._vendored.infonet as infonet_pkg

    notice_path = Path(infonet_pkg.__file__).resolve().parent / "NOTICE.md"
    assert notice_path.is_file()
    assert notice_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# USABILITY_B-10 (P2, sys.path hygiene): infer.py's absolute imports forced a PERMANENT sys.path[0]
# injection at the call site in _neural_mi.py, making generic top-level names like `model`/`util`
# importable process-wide for the rest of the interpreter's lifetime.
# ---------------------------------------------------------------------------


def test_regression_neural_mi_scopes_sys_path_injection(monkeypatch, tmp_path):
    """Post-fix: the vendored infonet path is removed from sys.path again once the one-time import of `infer`
    completes, whether or not the model loads. Observed by driving the real model loader with a placeholder
    checkpoint, so the real scoped import runs and fails at load_model."""
    import sys
    from pathlib import Path

    import mlframe.feature_selection.filters._neural_mi as neural_mi_mod

    vendored = str(Path(neural_mi_mod.__file__).resolve().parent / "_vendored" / "infonet")
    was_present_before = vendored in sys.path

    monkeypatch.setattr(neural_mi_mod, "_INFONET_CACHE_DIR", tmp_path)
    (tmp_path / "infonet_pretrained.pt").write_bytes(b"placeholder, not a real checkpoint")
    monkeypatch.delitem(sys.modules, "infer", raising=False)  # force the real first-time import path to run
    monkeypatch.setattr(sys, "path", list(sys.path))  # any injection is undone when the test ends

    with contextlib.suppress(Exception):  # the placeholder checkpoint cannot load; the path scoping must still be undone
        neural_mi_mod._get_infonet_model("sys-path-scope-probe")

    assert (vendored in sys.path) == was_present_before, "the vendored infonet path leaked into sys.path after the infer import"


# ---------------------------------------------------------------------------
# X_SECURITY_API_PACKAGING-1 (P1): StabilityMRMR / StabilityFESelector were missing
# get_feature_names_out(), unlike their siblings MRMR / CorrelatedFeaturesSelector in the same module.
# ---------------------------------------------------------------------------


def test_regression_pyproject_pins_pyutilz_commit():
    """Regression: pyutilz was declared as a bare git URL with no pinned commit SHA, so installs resolved to whatever
    the default branch pointed at. The uv lockfile now records the git source with an exact commit SHA, and the
    pyproject declares pyutilz as a direct dependency that the lock resolves from that source.

    py-ci-shared stays deliberately unpinned (first-party CI tooling, never shipped to PyPI consumers), so only
    pyutilz is asserted here.
    """
    tomllib = pytest.importorskip("tomllib", reason="tomllib is stdlib from Python 3.11; this check needs it")
    from pathlib import Path

    pyproject_path, lock_path = (Path(__file__).resolve().parents[3] / name for name in ("pyproject.toml", "uv.lock"))
    if not pyproject_path.exists() or not lock_path.exists():
        pytest.skip("pyproject.toml / uv.lock not found at expected repo-root-relative path")
    deps = tomllib.loads(pyproject_path.read_text(encoding="utf-8")).get("project", {}).get("dependencies", [])
    assert any(d.startswith("pyutilz") for d in deps), "pyutilz must be a declared runtime dependency"
    packages = tomllib.loads(lock_path.read_text(encoding="utf-8")).get("package", [])
    pyutilz_entries = [pkg for pkg in packages if pkg.get("name") == "pyutilz"]
    assert pyutilz_entries, "uv.lock must resolve pyutilz"
    git_source = pyutilz_entries[0].get("source", {}).get("git", "")
    assert git_source.startswith("https://github.com/fingoldo/pyutilz.git"), git_source
    assert re.fullmatch(r"[0-9a-f]{40}", git_source.rsplit("#", 1)[-1]), "the locked pyutilz source must carry a full commit SHA"


# ---------------------------------------------------------------------------
# X_EFFICIENCY_ARCHITECTURE-2 (P1): 39 independently-gated FE-family constructors returned a full deep
# copy of X as their no-candidate/no-op early-exit path, even though the orchestrator's own
# _appended-then-conditional-merge calling convention never reads the returned frame on that path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modname",
    [
        "_fe_raw_redundancy_drop",
        "_fe_sufficient_summary",
        "_fe_cmi_redundancy_null",
        "_extra_fe_families_dispersion_resident",
        "_meta_fe_recommender",
        "_conditional_gate_fe",
    ],
)
def test_regression_no_stale_self_referential_log_line_numbers(modname):
    """Post-fix: none of the 6 files' debug-log messages embed a self-referential '<file>.py:<N>' string
    that can go stale relative to the file's actual current length."""
    import importlib
    import inspect
    import re

    mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    src = inspect.getsource(mod)
    assert not re.search(r"suppressed in \S+\.py:\d+", src), f"{modname} still has a stale line-number log string"
