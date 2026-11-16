"""The hardware-state helper of the benchmark harnesses: throttle decoding, the one-line rendering, the sample summary and the cool-down wait."""

from __future__ import annotations

from mlframe.feature_selection._benchmarks.profiling import hardware_state as hw


def _snap(temp=70.0, sm=1500.0, mask=0, util=10.0, cpu=2800.0):
    """A synthetic reading in the shape of ``snapshot()``."""
    return {"gpu_temp_c": temp, "gpu_sm_mhz": sm, "gpu_sm_max_mhz": 1911.0, "gpu_mem_mhz": 3504.0, "gpu_throttle_mask": mask, "gpu_util_pct": util, "cpu_load_pct": 20.0, "cpu_mhz": cpu, "cpu_max_mhz": 2808.0}


def test_throttle_mask_names_only_the_holding_reasons():
    """Idle (0x1) and application-clock (0x2) bits are not throttling; thermal and power-cap bits are."""
    assert hw.throttle_reasons(0x0) == []
    assert hw.throttle_reasons(0x3) == []
    assert hw.throttle_reasons(0x4 | 0x40) == ["sw_power_cap", "hw_thermal"]


def test_line_shows_temperature_clocks_and_throttle():
    """The one-line form carries the temperature, the SM clock against its maximum and the throttle reasons, and survives a missing GPU."""
    text = hw.line(_snap(temp=84, sm=139, mask=0x40))
    assert "84 C" in text and "139/1911" in text and "hw_thermal" in text and "CPU load" in text
    assert "GPU n/a" in hw.line({"cpu_load_pct": 5.0})


def test_sample_summary_reports_mean_max_and_throttle_seen():
    """Mean and maximum temperature, mean SM clock and the union of throttle reasons over the readings of a run."""
    text = hw.summarise_samples([_snap(temp=60, sm=1900), _snap(temp=80, sm=1100, mask=0x20)])
    assert "mean 70 C" in text and "max 80 C" in text and "mean 1500/1911" in text and "sw_thermal" in text
    assert hw.summarise_samples([]) == "hardware: no GPU readings"


def test_wait_until_cool_returns_when_the_card_is_cool_enough(monkeypatch):
    """The wait polls until the temperature is at or below the bound, and gives up at the timeout instead of blocking forever."""
    temps = iter([80.0, 71.0, 61.0])
    monkeypatch.setattr(hw, "snapshot", lambda: _snap(temp=next(temps)))
    monkeypatch.setattr(hw.time, "sleep", lambda s: None)
    assert hw.wait_until_cool(62.0, timeout_s=100.0, poll_s=0.0)["gpu_temp_c"] == 61.0
    monkeypatch.setattr(hw, "snapshot", lambda: _snap(temp=90.0))
    assert hw.wait_until_cool(62.0, timeout_s=0.0, poll_s=0.0)["gpu_temp_c"] == 90.0
