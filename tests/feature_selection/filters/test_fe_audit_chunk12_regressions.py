"""Regression tests for the FE helper audit (batched qbin, CMI null, content memo, batcher executors, frame ops, gain stats, budgets)."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


def test_device_seed_distinct_for_xor_colliding_pairs():
    """Pairs the old XOR fold mapped to one device RNG stream (seed 2 / salt 0 and seed 0 / salt 1) now get distinct seeds."""
    from mlframe.feature_selection.filters._fe_cmi_perm_null_gpu import _device_seed

    assert _device_seed(2, 0) != _device_seed(0, 1)
    assert _device_seed(7, 3) == _device_seed(7, 3)
    assert 0 <= _device_seed(2**40 + 5, 2**33 + 9) < 2**63


def test_stratified_support_stratum_ranks_are_float64():
    """The dense stratum rank is held in float64 whatever the criterion dtype, since float32 stops counting integers at 2**24."""
    from mlframe.feature_selection.filters._fe_cmi_redundancy_null import _NullInputs, _StratifiedSupport

    rng = np.random.default_rng(0)
    z = rng.integers(0, 5, 200).astype(np.int64)
    inp = _NullInputs(rng.integers(0, 4, 200).astype(np.int64), rng.integers(0, 2, 200).astype(np.int64), 0, 0)
    sup = _StratifiedSupport(inp, z)
    assert sup.z_rank.dtype == np.float64


def test_content_memo_distinguishes_long_lists_with_equal_head():
    """Two long list arguments that differ only after the first 200 repr characters must not share a memo entry."""
    from mlframe.feature_selection.filters._fe_content_memo import content_memo

    calls = []

    @content_memo()
    def f(values):
        """Return a one-row frame holding the last element."""
        calls.append(1)
        return pd.DataFrame({"last": [values[-1]]})

    a = [0.5] * 700 + [1.0]
    b = [0.5] * 700 + [2.0]
    assert f(a)["last"].iloc[0] == 1.0
    assert f(b)["last"].iloc[0] == 2.0
    assert len(calls) == 2


def test_cpu_fe_batch_mi_rejects_y_length_mismatch():
    """A target shorter than the candidate matrix is refused before it reaches the compiled kernel."""
    from mlframe.feature_selection.filters._fe_cpu_batch import cpu_fe_batch_mi

    with pytest.raises(ValueError, match="y_codes"):
        cpu_fe_batch_mi(np.random.default_rng(0).normal(size=(100, 3)), np.zeros(50, dtype=np.int64))


def test_gpu_fe_batch_mi_rejects_y_length_mismatch():
    """The device executor checks the target length before any device work."""
    from mlframe.feature_selection.filters._fe_gpu_batch._executor import gpu_fe_batch_mi

    with pytest.raises(ValueError, match="y_codes"):
        gpu_fe_batch_mi(np.zeros((100, 3)), np.zeros(50, dtype=np.int64))


def test_load_budgets_survives_null_value(tmp_path, monkeypatch):
    """A budget that serialised as null (NaN) makes the loader return None instead of raising TypeError."""
    from mlframe.feature_selection.filters import _fe_family_budget as fb

    monkeypatch.setattr(fb, "_BUDGET_CACHE_DIR", tmp_path)
    fb.persist_budgets({"a": float("nan"), "b": 1.0}, cache_key="k")
    assert fb.load_budgets(cache_key="k") is None


def test_held_out_codes_ignores_nan_in_fit_rows():
    """NaN among the fit rows no longer turns every edge into NaN; the finite rows keep their equal-frequency codes."""
    from mlframe.feature_selection.filters._fe_gain_stats import held_out_codes

    rng = np.random.default_rng(1)
    f = rng.normal(size=400)
    f[:5] = np.nan
    even = np.arange(400) % 2 == 0
    codes = held_out_codes(f, even, ~even, 10)
    assert len(np.unique(codes)) >= 8


def test_collect_and_score_keeps_list_matrix():
    """A candidate matrix given as nested lists is scored, not silently dropped for lacking ``.size``."""
    from mlframe.feature_selection.filters._fe_gpu_batch._collector import collect_and_score

    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(np.int64)
    out = collect_and_score([X.tolist(), X], y)
    assert out[0].shape == (2,) and np.allclose(out[0], out[1])


def test_fused_entropy_force_scopes_overlap_restore_env(monkeypatch):
    """Two overlapping forced-fused sweeps leave the environment switch exactly as it was found."""
    from mlframe.feature_selection.filters import _fe_cmi_fused_backend as fb

    monkeypatch.delenv(fb._FORCE_ENV, raising=False)
    fb._force_fused_enter()
    fb._force_fused_enter()
    fb._force_fused_exit()
    forced = os.environ.get(fb._FORCE_ENV) == "1"
    assert forced, f"{fb._FORCE_ENV} is not '1' while forced"
    fb._force_fused_exit()
    unset = fb._FORCE_ENV not in os.environ
    assert unset, f"{fb._FORCE_ENV} left set after the last exit"


def test_fused_entropy_backend_selection_failure_warns(monkeypatch, caplog):
    """A failing tuning-cache lookup is reported at WARNING, not only at debug."""
    from mlframe.feature_selection.filters import _fe_cmi_fused_backend as fb
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

    def boom(*a, **k):
        """Raise like a broken cache."""
        raise RuntimeError("cache broken")

    monkeypatch.setattr(KernelTuningCache, "load_or_create", boom)
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        assert fb.fused_entropy_backend(1000, 400) == "fused"
    assert any("selection failed" in r.getMessage() for r in caplog.records)


def test_swept_device_null_failure_warns(monkeypatch, caplog):
    """The swept device null falling back to the host table null is logged at WARNING."""
    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as pg
    from mlframe.feature_selection.filters import _fe_cmi_redundancy_null as rn

    monkeypatch.setattr(pg, "perm_null_on_device", lambda n: True)

    def boom(*a, **k):
        """Raise like a device failure."""
        raise RuntimeError("device down")

    monkeypatch.setattr(pg, "conditional_perm_null_gpu", boom)
    inp = SimpleNamespace(n_size=100, y=np.zeros(100, dtype=np.int64), resident_or_host=lambda: np.zeros(100, dtype=np.int64))
    sup = SimpleNamespace(z=np.zeros(100, dtype=np.int64), order=np.arange(100), z_rank=np.zeros(100))
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.mrmr"):
        assert rn._swept_device_null(inp, sup, 25, 0.95, 0, 0) is None
    assert any("swept device conditional perm-null failed" in r.getMessage() for r in caplog.records)
