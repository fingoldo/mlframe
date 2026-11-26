"""The stage ledger and the idle-device report: events from the FE timers, the dump at the end of a fit, and the join with a hardware timeline (synthetic data, no fit)."""

from __future__ import annotations

import json
import time

import pytest

from mlframe.feature_selection._benchmarks.profiling import stage_idle_report as R
from mlframe.feature_selection.filters import _fe_family_timing as T
from mlframe.feature_selection.filters import _stage_ledger as L


@pytest.fixture(autouse=True)
def _own_ledger_events(monkeypatch):
    """Every test records into its own event list, so the module's list is not shared between tests."""
    monkeypatch.setattr(L, "_EVENTS", [])


def test_the_ledger_is_off_and_empty_by_default(monkeypatch):
    """Without the environment variable no event is recorded and the timers behave as before."""
    monkeypatch.delenv(L.ENV, raising=False)
    L.reset()
    with T.fe_family_timer("a_family"):
        pass
    T.record_fe_family_wall("another", 0.01)
    assert not L.enabled() and L.snapshot() == []


class _SteppingClock:
    """An epoch clock that advances by half a second at every reading, so that a measured duration does not depend on the machine."""

    def __init__(self) -> None:
        """Start at a fixed epoch second."""
        self.now = 1_000_000.0

    def __call__(self) -> float:
        """One reading: the clock moves on first."""
        self.now += 0.5
        return self.now


def test_timers_become_events_with_epoch_times_device_and_input_size(monkeypatch):
    """A family timer, a direct wall record and a gate lap are events on the epoch clock; the labels given to the timer are kept."""
    monkeypatch.setenv(L.ENV, "1")
    L.reset()
    monkeypatch.setattr(time, "time", _SteppingClock())
    before = time.time()
    with T.fe_family_timer("the_family", device="gpu:0", in_gb=0.4):
        pass
    T.record_fe_family_wall("direct", 0.05)
    timer = T.FamilyLapTimer()
    timer.gate("lap_flag", True)
    timer.close()
    events = {e["name"]: e for e in L.snapshot()}
    assert set(events) == {"the_family", "direct", "gate:lap_flag"}
    family = events["the_family"]
    assert before <= family["t0"] < family["t1"] <= time.time() and family["t1"] - family["t0"] == 0.5
    assert family["device"] == "gpu:0" and family["in_gb"] == 0.4
    assert abs((events["direct"]["t1"] - events["direct"]["t0"]) - 0.05) < 1e-6
    assert events["gate:lap_flag"]["device"] is None
    L.reset()


def test_a_path_in_the_variable_dumps_the_events_at_the_end_of_a_fit(monkeypatch, tmp_path):
    """With a path as the value the events are appended as JSON lines when the fit's summary is logged, and the memory is cleared."""
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(L.ENV, str(path))
    L.reset()
    with T.fe_family_timer("first"):
        pass
    with T.fe_family_timer("second"):
        pass
    T.log_fe_family_summary()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["name"] for r in rows] == ["first", "second"] and L.snapshot() == []


def _sample(t, cores, gpu_util=0.0, gpu_free=3.0, ram_free=8.0):
    """A synthetic timeline sample: ``cores`` busy percentages, one GPU."""
    return {"t": t, "cpu": sum(cores) / len(cores), "cores": cores, "ram_free_gb": ram_free, "gpus": [{"index": 0, "util": gpu_util, "mem_free_gb": gpu_free, "mem_used_gb": 4.0 - gpu_free, "temp_c": 60.0}]}


def test_the_report_names_the_idle_device_before_a_stage_and_the_stage_it_followed():
    """Eight cores and one GPU: a GPU stage runs 0-10 s with the CPU nearly idle, a CPU stage runs 10-20 s with the GPU idle. Before the CPU stage the CPU stood idle (the GPU was busy), and it followed the GPU stage."""
    samples = [_sample(float(t), [5.0] * 8, gpu_util=80.0) for t in range(0, 10)] + [_sample(float(t), [90.0] * 8, gpu_util=2.0) for t in range(10, 20)]
    events = [
        {"name": "gpu_stage", "t0": 0.0, "t1": 10.0, "thread": 1, "device": "gpu:0", "in_gb": 1.0},
        {"name": "cpu_stage", "t0": 10.0, "t1": 20.0, "thread": 1, "device": None, "in_gb": 1.0},
        {"name": "tiny", "t0": 5.0, "t1": 5.2, "thread": 2, "device": None, "in_gb": None},
    ]
    rows = {r["name"]: r for r in R.analyse(events, samples, min_stage_s=1.0, lookback_s=5.0)}
    assert set(rows) == {"gpu_stage", "cpu_stage"}  # the 0.2 s event is below the minimum
    assert rows["gpu_stage"]["device"] == "gpu:0" and rows["gpu_stage"]["idle_before"] == []  # nothing was sampled before it
    assert rows["cpu_stage"]["device"] == "cpu" and rows["cpu_stage"]["after"] == "gpu_stage"
    assert len(rows["cpu_stage"]["idle_before"]) == 1 and rows["cpu_stage"]["idle_before"][0].startswith("cpu 7.6/8 cores idle")
    during_gpu = R.window_stats(samples, 0.0, 9.0)
    assert R.classify(during_gpu) == "gpu"
    assert R.classify(R.window_stats(samples, 10.0, 19.0)) == "cpu" and R.classify(None) == "unknown"


def test_free_memory_decides_whether_an_idle_device_could_have_taken_the_stage():
    """An idle GPU with 0.5 GB free cannot hold a 1.2 GB stage input; with 3 GB it can."""
    idle_gpu_small = R.window_stats([_sample(float(t), [90.0] * 8, gpu_util=1.0, gpu_free=0.5) for t in range(5)], 0.0, 4.0)
    idle_gpu_big = R.window_stats([_sample(float(t), [90.0] * 8, gpu_util=1.0, gpu_free=3.0) for t in range(5)], 0.0, 4.0)
    assert R.idle_before(idle_gpu_small, 1.2) == []
    assert len(R.idle_before(idle_gpu_big, 1.2)) == 1 and R.idle_before(idle_gpu_big, 1.2)[0].startswith("gpu0")


def test_the_timeline_summary_charges_the_idle_seconds_to_the_shortest_covering_stage():
    """Samples with the CPU and the GPU both idle are charged to the shortest stage that covers them; samples outside every stage go to ``(no stage)``."""
    samples = [_sample(float(t), [2.0] * 4, gpu_util=1.0) for t in range(0, 6)] + [_sample(float(t), [95.0] * 4, gpu_util=90.0) for t in range(6, 10)]
    events = [{"name": "outer", "t0": 0.0, "t1": 8.0}, {"name": "inner", "t0": 1.0, "t1": 4.0}]
    summary = R.idle_timeline(events, samples)
    assert abs(summary["idle_share"] - 0.6) < 1e-9
    seconds = dict(summary["by_stage"])
    assert seconds["inner"] > seconds["outer"] > 0.0 and "(no stage)" not in seconds
    assert "both devices idle" in R.render(R.analyse(events, samples), summary)
