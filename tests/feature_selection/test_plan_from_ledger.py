"""The instance of a measured fit: the cascades overlap only as far as the cores, the GPU and the interpreter lock they hold allow."""

import json

import pytest

from mlframe.feature_selection._benchmarks.profiling import plan_from_ledger as pl
from mlframe.feature_selection.filters._fe_plan import plan, violations


def _fit(tmp_path, *, cpu_percent, gpu_util):
    """A synthetic fit of 100 s: four cascades of 10 s each at ``cpu_percent`` of 8 logical cores and ``gpu_util`` percent of the GPU, then a 5 s gate and a 5 s discretisation, and 40 s of unnamed tail."""
    rows, t = [], 1000.0
    for name in pl.CASCADES:
        rows.append({"name": name, "t0": t, "t1": t + 10.0, "thread_name": "MainThread"})
        t += 10.0
    rows.append({"name": "stage:accuracy_gate", "t0": t, "t1": t + 5.0, "thread_name": "MainThread"})
    rows.append({"name": "stage:discretize_inputs", "t0": t + 5.0, "t1": t + 10.0, "thread_name": "MainThread"})
    rows.append({"name": "stage:fe_step", "t0": t + 10.0, "t1": t + 60.0, "thread_name": "MainThread"})
    ledger = tmp_path / "fit.ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    samples = [{"t": 1000.0 + s + 0.5, "cpu": cpu_percent, "gpus": [{"util": gpu_util}]} for s in range(0, 100)]
    timeline = tmp_path / "fit.timeline.jsonl"
    timeline.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    return pl.read_jsonl(str(ledger)), pl.read_jsonl(str(timeline))


def test_the_windows_demands_and_the_chain_of_a_measured_fit(tmp_path):
    """The windows of the named stages are found, the demand of a cascade is the mean of its samples, and the instance chains the stages after the cascades."""
    ledger, samples = _fit(tmp_path, cpu_percent=25.0, gpu_util=20.0)
    windows = pl.stage_windows(ledger)
    assert set(pl.CASCADES) <= set(windows) and windows["stage:fe_step"] == (1050.0, 1100.0)
    cores, share = pl.demand_of(samples, windows["stage:cascade_mid_a"], logical=8)
    assert cores == pytest.approx(2.0) and share == pytest.approx(0.2)
    instance, sequential, span = pl.build_instance(ledger, samples, logical=8, capacity_cores=6.0, gil=0.0)
    assert span == pytest.approx(100.0) and sequential == pytest.approx(100.0)
    after = {t.name: t.after for t in instance.tasks}
    assert after["stage:accuracy_gate"] == pl.CASCADES and after["stage:discretize_inputs"] == ("stage:accuracy_gate",)


def test_the_cascades_overlap_exactly_as_far_as_the_resources_allow(tmp_path):
    """Four cascades of one core each on 6 cores: all four overlap while the interpreter lock they hold sums to at most 1 (0.25 each), two at a time at 0.5 each, none at 1.0."""
    ledger, samples = _fit(tmp_path, cpu_percent=12.5, gpu_util=10.0)
    walls = {}
    for gil in (0.0, 0.25, 0.5, 1.0):
        instance, sequential, _ = pl.build_instance(ledger, samples, logical=8, capacity_cores=6.0, gil=gil)
        result = plan(instance, time_limit_s=1.0)
        assert violations(instance, result) == []
        walls[gil] = result.makespan
    assert sequential == pytest.approx(100.0)
    assert walls[0.0] == pytest.approx(70.0) and walls[0.25] == pytest.approx(70.0) and walls[0.5] == pytest.approx(80.0) and walls[1.0] == pytest.approx(100.0)
