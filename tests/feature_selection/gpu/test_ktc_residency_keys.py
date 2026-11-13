"""The residency vocabulary of the kernel-tuning entries: scenario names and the placement read from the operands."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._ktc_fallback import RESIDENCIES, detect_residency, residency_entry_name


def test_entry_names_cover_the_four_scenarios():
    """Four distinct entry names, one per (in, out) pair."""
    names = {residency_entry_name("k", a, b) for a in RESIDENCIES for b in RESIDENCIES}
    assert names == {"k__host_host", "k__host_device", "k__device_host", "k__device_device"}
    with pytest.raises(ValueError):
        residency_entry_name("k", "host", "gpu")


def test_host_arrays_and_scalars_read_as_host():
    """Numpy operands (scalars ignored) are host; no array at all is host."""
    assert detect_residency(np.zeros(3), 5, None, "x") == "host"
    assert detect_residency() == "host"


def test_cupy_operands_read_as_device_and_a_mix_as_host():
    """All cupy arrays: device. One numpy operand among them: host, because the device path then pays that upload."""
    cp = pytest.importorskip("cupy")
    assert detect_residency(cp.zeros(3), cp.ones(2), 7) == "device"
    assert detect_residency(cp.zeros(3), np.ones(2)) == "host"


def test_residency_report_times_every_declared_scenario_and_derives_the_transfer_share(monkeypatch):
    """A fake sweep module whose device call costs 4 ms host-in and 1 ms resident reports a transfer share of 0.75."""
    import sys
    import time
    import types

    from mlframe.feature_selection.filters._ktc_fallback import residency_report

    m = types.ModuleType("fake_ktc_module")
    m._SWEEP_ROWS = [10]
    m._SWEEP_RESIDENCIES = [("host", "host"), ("device", "device")]
    m._sweep_inputs = lambda dims: ("host-data",)
    m._sweep_place = lambda inputs, residency: ("device-data",) if residency == "device" else inputs
    m._sweep_host = lambda x: time.sleep(0.002)
    m._sweep_device = lambda x: time.sleep(0.001 if x == "device-data" else 0.004)
    monkeypatch.setitem(sys.modules, "fake_ktc_module", m)
    rows = residency_report("fake_ktc_module", repeats=2)
    assert {r["scenario"] for r in rows} == {("host", "host"), ("device", "device")}
    share = rows[0]["transfer_share"]
    assert 0.6 < share < 0.85
    resident = next(r for r in rows if r["scenario"] == ("device", "device"))
    assert resident["host_s"] is None


def test_a_residency_sweep_places_the_inputs_and_hands_the_output_placement_to_both_backends(monkeypatch):
    """``host_device_sweep(residency=True)`` builds a tuner taking the two placement keys: inputs go through ``_sweep_place`` for the input side, both callables receive ``data_out_residency``."""
    import sys
    import types

    import pyutilz.dev.benchmarking as bench

    from mlframe.feature_selection.filters._ktc_fallback import host_device_sweep

    seen = {}

    def fake_grid(variants, axes, make_inputs, **kw):
        """Capture what the sweep would time instead of timing it."""
        inputs = make_inputs({"n_samples": 7})
        seen["inputs"] = inputs
        seen["results"] = {name: fn(*inputs) for name, fn in variants.items()}
        return [{"n_samples_max": 7, "backend_choice": "host"}]

    monkeypatch.setattr(bench, "sweep_backend_grid", fake_grid)
    m = types.ModuleType("fake_residency_sweep_module")
    m._SWEEP_ROWS = [7]
    m._sweep_inputs = lambda dims: (f"x{dims['n_samples']}",)
    m._sweep_place = lambda inputs, residency: tuple(f"{residency}:{a}" for a in inputs)
    monkeypatch.setitem(sys.modules, "fake_residency_sweep_module", m)
    tuner = host_device_sweep("fake_residency_sweep_module", lambda x, data_out_residency: ("h", x, data_out_residency), lambda x, data_out_residency: ("d", x, data_out_residency), residency=True)
    assert tuner(data_in_residency="device", data_out_residency="host") == [{"n_samples_max": 7, "backend_choice": "host"}]
    assert seen["inputs"] == ("device:x7",)
    assert seen["results"] == {"host": ("h", "device:x7", "host"), "device": ("d", "device:x7", "host")}
    assert host_device_sweep("fake_residency_sweep_module", lambda x: 1, lambda x: 2).__code__.co_argcount == 2
