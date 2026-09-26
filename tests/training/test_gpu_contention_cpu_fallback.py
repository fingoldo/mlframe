"""A target's boosters train on CPU while the GPU is busy with somebody else's work, and on GPU again once it is free.

A production run shared the card with an embedding job (util 100%, 8.1 of 8.2 GB taken): CatBoost ran at a sixth of its
early rate for two hours and hit the time budget. The GPU decision had looked only at the card's total memory.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training import _gpu_contention as gc_mod
from mlframe.training._gpu_contention import cb_device_index, gpu_busy_reason

OWN = 1000


def _snap(util=5.0, used=500.0, total=8192.0, procs=()):
    return {"gpus": [{"index": 0, "util_pct": util, "mem_used_mb": used, "mem_total_mb": total}], "processes": list(procs)}


def test_the_production_shape_is_busy():
    """Another process at full utilisation holding almost all of the card."""
    snap = _snap(util=100.0, used=8100.0, procs=[{"gpu": 0, "pid": 7320, "name": "python.exe", "mem_mb": 8000.0}])
    reason = gpu_busy_reason(snap, required_gb=1.0, own_pid=OWN)
    assert reason is not None and "free" in reason


def test_our_own_memory_does_not_count_against_us():
    """Our cupy pools and a finished CatBoost fit's buffers are reusable: 7 GB of our own is not contention."""
    snap = _snap(util=0.0, used=7300.0, procs=[{"gpu": 0, "pid": OWN, "name": "python.exe", "mem_mb": 7000.0}])
    assert gpu_busy_reason(snap, required_gb=2.0, own_pid=OWN) is None


def test_without_per_process_memory_all_used_memory_counts():
    """Windows WDDM reports no per-process VRAM; then the card's used memory is the only measure."""
    snap = _snap(util=0.0, used=7900.0, procs=[{"gpu": 0, "pid": 7320, "name": "python.exe", "mem_mb": None}])
    assert gpu_busy_reason(snap, required_gb=1.0, own_pid=OWN) is not None


def test_high_utilisation_alone_is_busy_and_the_threshold_is_configurable(monkeypatch):
    snap = _snap(util=85.0, used=1000.0)
    assert "85% busy" in gpu_busy_reason(snap, required_gb=0.5, own_pid=OWN)
    monkeypatch.setenv("MLFRAME_GPU_BUSY_UTIL_PCT", "90")
    assert gpu_busy_reason(snap, required_gb=0.5, own_pid=OWN) is None


def test_an_unreadable_card_is_not_evidence_of_contention():
    assert gpu_busy_reason(None, required_gb=1.0) is None
    assert gpu_busy_reason({"gpus": [], "processes": []}, required_gb=1.0) is None


@pytest.mark.parametrize("devices,expected", [(None, 0), ("0", 0), ("1:2", 1), ("3-5", 3), ([2], 2), ("GPU", 0)])
def test_catboost_devices_map_to_the_first_device(devices, expected):
    assert cb_device_index(devices) == expected


def _configure(monkeypatch, snapshot):
    """configure_training_params on a GPU-capable host whose card looks like ``snapshot`` right now."""
    from mlframe.training import _gpu_state_probe, _trainer_configure
    from mlframe.training._trainer_configure import configure_training_params

    monkeypatch.setattr(_trainer_configure, "_cached_gpu_info", lambda: [{"index": 0}])
    monkeypatch.setattr(_trainer_configure, "compute_total_gpus_ram", lambda gpus: {"gpu_max_ram_total": 8.0, "gpus_ram_total": 8.0})
    monkeypatch.setattr(_gpu_state_probe, "gpu_snapshot", lambda: snapshot)
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"f": rng.standard_normal(n)})
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    out = configure_training_params(
        df=df, train_df=df.iloc[:40], val_df=df.iloc[40:50], test_df=df.iloc[50:], target=y, train_target=y.iloc[:40],
        val_target=y.iloc[40:50], test_target=y.iloc[50:], train_idx=np.arange(40), val_idx=np.arange(40, 50),
        test_idx=np.arange(50, n), use_regression=False, prefer_gpu_configs=True, mlframe_models=["cb"], verbose=False,
        config_params={"iterations": 10},
    )
    return out


def _gpu_decision(monkeypatch, caplog, snapshot) -> str:
    """The ``data_fits_gpu_ram=...`` value configure_training_params logged: whether GPU configs were chosen.

    Read from the decision line rather than the CatBoost ``task_type``, which also depends on whether this host's
    CatBoost build can reach a GPU at all."""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _configure(monkeypatch, snapshot)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("data_fits_gpu_ram="))
    return line.split(",")[0].split("=")[1]


_BUSY = _snap(util=100.0, used=8100.0, procs=[{"gpu": 0, "pid": 7320, "name": "python.exe", "mem_mb": 8000.0}])


def test_a_busy_card_sends_the_target_to_cpu_and_a_free_one_back_to_gpu(monkeypatch, caplog):
    monkeypatch.setattr(gc_mod, "_LAST_BUSY", {"reason": None})
    assert _gpu_decision(monkeypatch, caplog, _BUSY) == "False"
    assert any("train on CPU" in r.getMessage() for r in caplog.records)
    assert _gpu_decision(monkeypatch, caplog, _snap()) == "True"
    assert any("free again" in r.getMessage() for r in caplog.records)


def test_the_check_can_be_switched_off(monkeypatch, caplog):
    monkeypatch.setenv("MLFRAME_GPU_CONTENTION_CPU_FALLBACK", "0")
    assert _gpu_decision(monkeypatch, caplog, _BUSY) == "True"
