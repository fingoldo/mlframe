"""Regression tests for the stability-vote / rung / subsample / sufficient-summary / resident-operand / raw-redundancy audit fixes."""

import logging
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts

_LOGGER = "mlframe.feature_selection.filters.mrmr"


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Forget the throttle counts so every test sees its own first warning."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def test_quorum_folds_not_inflated_by_float_product():
    """0.7 * 10 is 7.000000000000001 in floating point; the quorum bar must stay 7 folds, not 8."""
    from mlframe.feature_selection.filters._fe_stability_vote import _quorum_folds

    assert _quorum_folds(0.7, 10) == 7
    assert _quorum_folds(0.6, 5) == 3
    assert _quorum_folds(0.5, 5) == 3


def test_stability_vote_fold_replay_failure_is_logged_at_warning(monkeypatch, caplog):
    """A recipe that cannot replay on any fold is left admitted; that must be visible at WARNING."""
    import mlframe.feature_selection.filters.engineered_recipes as er
    from mlframe.feature_selection.filters._fe_stability_vote import _vote_recipe_over_folds

    def _boom(recipe, X):
        """Replay stub that always fails."""
        raise RuntimeError("replay broke")

    monkeypatch.setattr(er, "apply_recipe", _boom)
    X = pd.DataFrame({"a": np.arange(20.0)})
    folds = [np.arange(10), np.arange(10, 20)]
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        passes, evaluated = _vote_recipe_over_folds("eng", SimpleNamespace(), folds, X, np.zeros(20, dtype=np.int64), None, None, False, {"a"}, 20, 4, {})
    assert (passes, evaluated) == (0, 0)
    assert any("failed to replay on a fold" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_rung_schedule_nan_pair_mi_does_not_cut_the_best_pair():
    """A NaN pair MI must rank last instead of poisoning the max and the ordering."""
    from mlframe.feature_selection.filters._fe_rung_schedule import apply_rung_schedule

    pms = [float("nan"), 0.9, 0.8, 0.1, 0.09, 0.08, 0.07, 0.06]
    pairs = {((i, i + 100), pm): i for i, pm in enumerate(pms)}
    kept, info = apply_rung_schedule(pairs, n_rows=1000, keep_frac=0.25, min_pairs=6)
    kept_pms = [k[1] for k in kept]
    assert 0.9 in kept_pms and 0.8 in kept_pms
    assert info["applied"] is True


def test_rung_schedule_invalid_env_keep_frac_warns(monkeypatch, caplog):
    """An unusable MLFRAME_FE_RUNG_KEEP_FRAC is reported instead of silently ignored."""
    from mlframe.feature_selection.filters import _fe_rung_schedule as rs

    monkeypatch.setenv("MLFRAME_FE_RUNG_KEEP_FRAC", "abc")
    with caplog.at_level(logging.WARNING, logger=rs.logger.name):
        assert rs._dispatch_keep_frac(1000, 10) == 1.0
    assert any("MLFRAME_FE_RUNG_KEEP_FRAC" in r.getMessage() for r in caplog.records)


def test_proportional_stratum_pick_allocation_is_exact_in_float32_mode(monkeypatch):
    """The per-stratum quota is integer arithmetic: 53/100 of 100 rows is 53, not 52 from a float32 product."""
    from mlframe.feature_selection.filters import _fe_subsample as fs

    monkeypatch.setattr(fs, "_crit_np_dtype", lambda: np.float32)
    members = [np.arange(53), np.arange(53, 100)]
    counts = np.array([53, 47], dtype=np.int64)
    picked = fs._proportional_stratum_pick(np.random.default_rng(0), members, counts, 100, 100, 1)
    assert picked.shape[0] == 100


def test_stratified_subsample_failure_is_logged_at_warning(monkeypatch, caplog):
    """A failed stratified draw falls back to uniform, which can drop a rare class; that fallback is logged at WARNING."""
    from mlframe.feature_selection.filters import _fe_subsample as fs

    def _boom(*args, **kwargs):
        """Kernel stub that always fails."""
        raise RuntimeError("kernel broke")

    monkeypatch.setattr(fs, "_strat_clf_kernel", _boom)
    y = np.array([0] * 90 + [1] * 10)
    with caplog.at_level(logging.WARNING, logger=fs._logger.name):
        idx = fs.stratified_subsample_idx(np.random.default_rng(0), y, 20, is_clf=True)
    assert idx.shape[0] == 20
    assert any("stratified draw failed" in r.getMessage() for r in caplog.records)


def test_shannon_entropy_is_double_precision_under_relaxed_dtype(monkeypatch):
    """Entropy differences feed an MI near 1e-5; the counts must be accumulated in float64 even when the criterion dtype is float32."""
    from mlframe.feature_selection.filters import _fe_sufficient_summary as ss

    monkeypatch.setattr(ss, "_crit_np_dtype", lambda: np.float32)
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 10, size=1_000_003)
    counts = np.bincount(codes).astype(np.float64)
    p = counts / counts.sum()
    ref = float(-(p * np.log(p)).sum())
    assert abs(ss._shannon_entropy_nats(codes) - ref) < 1e-12


def test_maxt_residual_mi_is_double_precision_under_relaxed_dtype(monkeypatch):
    """The per-raw MI(r; x) is scored in float64 so a float32 entropy difference cannot move it past the permutation floor."""
    from mlframe.feature_selection.filters import _fe_sufficient_summary as ss

    monkeypatch.setattr(ss, "_crit_np_dtype", lambda: np.float32)
    rng = np.random.default_rng(1)
    n = 200_000
    x = rng.integers(0, 10, size=n).astype(np.int64)
    r = rng.integers(0, 10, size=n).astype(np.int64)
    v = SimpleNamespace()
    ss._maxt_residual_test(v, [0], x[:, None], np.array([10]), r, n, 3, 0.95, False, 0)
    joint = np.bincount(x * 10 + r, minlength=100).astype(np.float64) / n
    hx = -(np.bincount(x) / n * np.log(np.bincount(x) / n)).sum()
    hr = -(np.bincount(r) / n * np.log(np.bincount(r) / n)).sum()
    hxr = -(joint[joint > 0] * np.log(joint[joint > 0])).sum()
    assert abs(v.per_raw_mi[0] - max(0.0, hx + hr - hxr)) < 1e-12


def test_shared_subsample_idx_is_recomputed_when_the_target_changes():
    """The shared row draw is cached per fit, not per row count: a refit with another target of the same length must not reuse a stale stratified draw."""
    from mlframe.feature_selection.filters._fe_sufficient_summary import _get_shared_fe_subsample_idx

    est = SimpleNamespace(_default_screen_subsample_n=lambda: 100, fe_subsample_stratify=True, random_seed=3, is_classification_=True, task="classification")
    y1 = np.zeros(1000)
    y1[:20] = 1.0
    y2 = np.zeros(1000)
    y2[-20:] = 1.0
    idx1 = _get_shared_fe_subsample_idx(est, y1, 1000)
    idx2 = _get_shared_fe_subsample_idx(est, y2, 1000)
    assert idx1 is not None and idx2 is not None
    assert (y2[idx2] == 1.0).sum() >= 2
    assert not np.array_equal(idx1, idx2)


def test_sufficient_summary_check_failure_is_logged_at_warning(caplog):
    """A plumbing failure disables the FE early stop; that is reported at WARNING, not debug."""
    from mlframe.feature_selection.filters._fe_sufficient_summary import check_sufficient_summary_for_mrmr

    est = SimpleNamespace(feature_names_in_=["a"])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        v = check_sufficient_summary_for_mrmr(est, data=None, nbins=np.array([2]), cols=["a"], selected_vars=[], target_indices=[0], X=None, y=np.zeros(3))
    assert v.reached is False
    assert any("sufficient-summary check failed" in r.getMessage() for r in caplog.records)


def test_resident_qbin_codes_cache_is_keyed_per_device(monkeypatch):
    """Bin codes cached for one GPU must not be handed to a thread working on another GPU."""
    from mlframe.feature_selection.filters import _fe_resident_operands as ro

    current = [0]
    fake_cp = SimpleNamespace(int64=np.int64, int16=np.int16, cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=current[0])))
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setattr(ro, "resident_operand", lambda host, role, dtype=None: host)
    monkeypatch.setattr(ro, "_disabled", lambda: False)
    calls = []

    def _compute(cp, xd, nbins):
        """Binner stub that records each real compute."""
        calls.append(current[0])
        return np.zeros(xd.shape[0], dtype=np.int64)

    ro._FE_RESIDENT_QBIN_CODES.clear()
    col = np.arange(50, dtype=np.float64)
    try:
        ro.resident_qbin_codes(col, 10, np.float64, _compute)
        current[0] = 1
        ro.resident_qbin_codes(col, 10, np.float64, _compute)
    finally:
        ro._FE_RESIDENT_QBIN_CODES.clear()
        ro.clear_hash_memo()
    assert calls == [0, 1]


def test_rank_transform_does_not_sort_twice(monkeypatch):
    """The average-rank transform derives ranks from np.unique alone; an argsort whose result is discarded is wasted work."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_helpers as h

    def _forbidden(*args, **kwargs):
        """argsort stub: the transform must not call it."""
        raise AssertionError("argsort result is unused")

    monkeypatch.setattr(np, "argsort", _forbidden)
    out = h._rank_transform(np.array([3.0, 1.0, 3.0, 2.0]))
    assert out.tolist() == [3.5, 1.0, 3.5, 2.0]


def test_join_support_device_failure_is_logged_at_warning(monkeypatch, caplog):
    """The device-to-host join fallback is a slower path and is reported at WARNING."""
    import mlframe.feature_selection.filters._mi_greedy_cmi_fe as mg
    from mlframe.feature_selection.filters._fe_raw_redundancy_drop import _join_support

    def _boom(*args, **kwargs):
        """Device join stub that always fails."""
        raise RuntimeError("device join broke")

    monkeypatch.setattr(mg, "_renumber_joint_gpu", _boom, raising=False)
    host = np.array([0, 1, 1, 0], dtype=np.int64)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _z, card, z_dev = _join_support([host], [object()])
    assert z_dev is None and card == 2
    assert any("conditioning-support join failed" in r.getMessage() for r in caplog.records)


def test_admission_ledger_cap_is_logged(caplog):
    """Admission records past the cap are dropped; the truncation is announced once at WARNING."""
    from mlframe.feature_selection.filters._fe_rejection_ledger import record_fe_admission

    est = SimpleNamespace(fe_rejection_ledger_cap=0)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        record_fe_admission(est, candidate="c")
    assert est._fe_admission_records_ == []
    assert any("admission ledger hit its record cap" in r.getMessage() for r in caplog.records)
