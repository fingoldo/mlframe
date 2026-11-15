"""The sweep input builders of the host/device tuners accept the dims the sweep grid passes (``n_samples``).

``host_device_sweep`` calls ``_sweep_inputs({"n_samples": n})``. Two tuners read ``dims["n_rows"]`` instead, raised ``KeyError`` inside the sweep, persisted no region and so always decided by their row marks.
"""

from __future__ import annotations

import importlib

import pytest

_MODULES = ("_hinge_precheck_gpu", "_cmim_min_gpu", "_fleuret_confirm_gpu", "_gain_stats_codes_gpu")


@pytest.mark.parametrize("name", _MODULES)
def test_sweep_inputs_build_from_the_grid_dims(name):
    """``_sweep_inputs`` of the module returns its inputs for the dims the grid passes, at the smallest swept row count."""
    module = importlib.import_module(f"mlframe.feature_selection.filters.{name}")
    inputs = module._sweep_inputs({"n_samples": int(min(module._SWEEP_ROWS))})
    assert isinstance(inputs, tuple) and inputs
