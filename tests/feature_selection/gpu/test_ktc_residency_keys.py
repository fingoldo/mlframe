"""The residency report and the residency-aware sweep of the kernel-tuning entries (the scenario vocabulary and the placement read from the operands live in ``pyutilz.performance.kernel_tuning.residency``)."""

import numpy as np


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
    tuner = host_device_sweep(
        "fake_residency_sweep_module",
        lambda x, data_out_residency: ("h", x, data_out_residency),
        lambda x, data_out_residency: ("d", x, data_out_residency),
        residency=True,
    )
    assert tuner(data_in_residency="device", data_out_residency="host") == [{"n_samples_max": 7, "backend_choice": "host"}]
    assert seen["inputs"] == ("device:x7",)
    assert seen["results"] == {"host": ("h", "device:x7", "host"), "device": ("d", "device:x7", "host")}
    assert host_device_sweep("fake_residency_sweep_module", lambda x: 1, lambda x: 2).__code__.co_argcount == 2


def test_selection_view_counts_the_scores_and_marks_the_best_with_a_tie_band():
    """The shared view: the first entry is the number of scored candidates, then a 0/1 mark for those at or within the tie band of the ``top``-th best."""
    from mlframe.feature_selection.filters._ktc_fallback import selection_view

    scores = np.array([0.5, 0.1, 0.4, 0.3, 0.2])
    v = selection_view(scores, top=2, width=8)
    assert v[0] == 5 and list(v[1:6]) == [1.0, 0.0, 1.0, 0.0, 0.0] and v[6:].sum() == 0
    near = scores.copy()
    near[2] = 0.4 * (1 - 1e-5)
    assert np.array_equal(selection_view(near, top=2, width=8), v), "a value inside the tie band at the cut stays on the same side"
    assert selection_view(scores[:-1], top=2, width=8)[0] == 4
    assert selection_view(np.array([]), top=2, width=4)[0] == 0


def test_the_report_cli_prints_a_row_per_scenario(monkeypatch, capsys):
    """``python -m ..._ktc_fallback <module>`` prints one line per swept size and declared scenario."""
    import sys
    import types

    from mlframe.feature_selection.filters import _ktc_fallback as K

    m = types.ModuleType("fake_ktc_cli_module")
    m._SWEEP_ROWS = [10, 20]
    m._sweep_inputs = lambda dims: ("x",)
    m._sweep_host = lambda x: None
    m._sweep_device = lambda x: None
    monkeypatch.setitem(sys.modules, "fake_ktc_cli_module", m)
    assert K._main(["fake_ktc_cli_module", "1"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2 and all("scenario=('host', 'host')" in line for line in out)
    assert K._main([]) == 2
