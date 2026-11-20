"""Audit chunk 15: silent slow-path fallbacks log at WARNING, compiled kernels reject mismatched lengths."""

import logging
import types

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_sufficient_summary_pairs as ssp
from mlframe.feature_selection.filters import _feature_engineering_mem_budget as mb
from mlframe.feature_selection.filters._fe_synergy_screen import detect_synergy_combos
from mlframe.feature_selection.filters._fe_usability_signal import abs_pearson
from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_abs_corr_gpu as acg
from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_analytic_gpu as ag
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _abs_corr_zerofill_njit
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core_helpers import _check_prospective__scores_same_rows_fall


def test_pair_residual_tuning_lookup_failure_warns(monkeypatch, caplog):
    """A broken kernel-tuning lookup is logged at WARNING, not debug."""
    monkeypatch.setattr(ssp, "_fused_gpu_available", lambda: True)
    monkeypatch.setattr(ssp._PRT_SPEC, "choose", lambda **kw: (_ for _ in ()).throw(RuntimeError("cache broken")))
    with caplog.at_level(logging.WARNING, logger=ssp.logger.name):
        ssp.pair_residual_backend_choice(2000, 10)
    assert any(r.levelno == logging.WARNING and "kernel-tuning lookup failed" in r.getMessage() for r in caplog.records)


def test_hoist_decline_warns(monkeypatch, caplog):
    """Declining the hoisted buffer (a slower recompute path) is logged at WARNING."""
    monkeypatch.setattr(mb, "_available_ram_bytes_cached", lambda: 4 * 2**30)
    monkeypatch.setattr(mb, "_FE_MIN_FREE_RAM_GB", 3.0)
    with caplog.at_level(logging.WARNING, logger=mb.logger.name):
        ok, _, _ = mb._decide_hoist_shared_buffer(2 * 2**30)
    assert not ok
    assert any(r.levelno == logging.WARNING and "declining" in r.getMessage() for r in caplog.records)


def test_synergy_screen_rejects_column_shorter_than_target():
    """A code column whose length differs from the target must raise, not feed the njit kernel."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(np.int64)
    cols = {0: rng.integers(0, 3, 200), 1: rng.integers(0, 3, 150)}
    with pytest.raises(ValueError, match="rows"):
        detect_synergy_combos(cols, y, [0, 1])


def test_synergy_screen_rejects_negative_codes():
    """A negative code would index the histogram from the end silently."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(np.int64)
    c0 = rng.integers(0, 3, 200)
    c0[3] = -1
    with pytest.raises(ValueError, match="negative"):
        detect_synergy_combos({0: c0, 1: rng.integers(0, 3, 200)}, y, [0, 1])


def test_abs_pearson_rejects_length_mismatch():
    """The njit |corr| kernel loops over y and reads v at the same index: a shorter v is an out-of-bounds read."""
    with pytest.raises(ValueError, match="length"):
        abs_pearson(np.arange(50, dtype=np.float64), np.arange(30, dtype=np.float64))


def test_zerofill_corr_rejects_length_mismatch():
    """The zero-fill |corr| kernel must not read past the end of the shorter operand."""
    with pytest.raises(ValueError, match="length"):
        _abs_corr_zerofill_njit(np.arange(50, dtype=np.float64), np.arange(30, dtype=np.float64))


def test_abs_corr_gpu_rejects_target_length_mismatch():
    """The device |corr| entry points check the target rows before any launch."""
    dev = types.SimpleNamespace(shape=(100,))
    with pytest.raises(ValueError, match="rows"):
        acg.abs_corr_finite_gpu(dev, np.zeros(60), np.ones(60, dtype=bool))
    with pytest.raises(ValueError, match="rows"):
        acg.abs_corr_finite_batch_gpu(types.SimpleNamespace(shape=(3, 100)), np.zeros(60), np.ones(60, dtype=bool))


def test_resident_observed_mi_rejects_target_length_mismatch():
    """The fused observed-MI kernel reads y at every code row: a shorter target is rejected up front."""
    dev = types.SimpleNamespace(shape=(100, 4))
    with pytest.raises(ValueError, match="rows"):
        ag.resident_observed_mi_and_bins(dev, np.zeros(60, dtype=np.int64), 2)


def test_shared_subsample_idx_with_negative_entries_is_rejected():
    """A negative row index wraps to the end of the frame silently; the shared draw must be refused."""
    idx = np.array([-1, 2, 5, 9])
    assert _check_prospective__scores_same_rows_fall(idx, 100, None) is None
    ok = _check_prospective__scores_same_rows_fall(np.array([1, 2, 5, 9]), 100, None)
    assert ok is not None and ok.tolist() == [1, 2, 5, 9]
