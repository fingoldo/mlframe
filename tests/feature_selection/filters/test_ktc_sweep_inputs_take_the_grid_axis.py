"""The sweep input builders of the host/device tuners accept the dims the sweep grid passes (``n_samples``).

``host_device_sweep`` calls ``_sweep_inputs({"n_samples": n})``. Two tuners read ``dims["n_rows"]`` instead, raised ``KeyError`` inside the sweep, persisted no region and so always decided by their row marks.
The same grid names the axis ``n_samples`` when the tuner is asked: a spec registered with an ``n_rows`` axis and asked with ``choose(n_rows=...)`` raised ``TypeError`` out of the row-mark fallback on every call (caught,
so the row mark decided): six tuners never consulted their swept regions.
"""

from __future__ import annotations

import importlib

import pytest

_MODULES = ("_hinge_precheck_gpu", "_cmim_min_gpu", "_meta_fingerprint_gpu", "_grouped_quantile_gpu", "_fleuret_confirm_gpu", "_gain_stats_codes_gpu")


@pytest.mark.parametrize("name", _MODULES)
def test_sweep_inputs_build_from_the_grid_dims(name):
    """``_sweep_inputs`` of the module returns its inputs for the dims the grid passes, at the smallest swept row count."""
    module = importlib.import_module(f"mlframe.feature_selection.filters.{name}")
    inputs = module._sweep_inputs({"n_samples": int(min(module._SWEEP_ROWS))})
    assert isinstance(inputs, tuple) and inputs


@pytest.mark.parametrize("name", _MODULES)
def test_the_tuner_is_asked_with_the_axis_it_was_registered_with(name):
    """``_SPEC.choose(n_samples=...)`` answers ``host`` or ``device`` (no ``TypeError`` out of the fallback), and the spec's axes are the grid's ``n_samples``."""
    module = importlib.import_module(f"mlframe.feature_selection.filters.{name}")
    assert module._SPEC is not None
    assert str(module._SPEC.choose(n_samples=int(min(module._SWEEP_ROWS)))) in ("host", "device")
