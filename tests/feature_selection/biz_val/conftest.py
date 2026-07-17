"""Run every biz_val selection-quality test on BOTH FE backends -- opt-in via ``MLFRAME_TEST_STRICT_BOTH=1``.

The AUTO size-gated STRICT default (``MLFRAME_FE_GPU_STRICT`` unset) keeps small-n fits (these tests are n~2000) on
the CPU path, so the GPU-resident STRICT path would go untested by the default suite. Setting
``MLFRAME_TEST_STRICT_BOTH=1`` (a GPU CI job, or a local both-path vetting run) makes this autouse fixture
parametrize each biz_val test over the exact CPU path (``=0``) AND the STRICT GPU-resident path (``=1`` -- added only
when a CUDA device is usable), so a GPU regression cannot pass by exercising CPU alone.

Default (flag unset) is a SINGLE unparametrized run -- the suite is unchanged and does not double in runtime. A test
that asserts CPU-estimator MI MAGNITUDES (edge-binning STRICT gives different absolute MI -- selection converges at
large n, magnitude never does) opts out of the STRICT param with ``@pytest.mark.strict_cpu_only``.
"""

import os

import pytest


def _cuda_available() -> bool:
    """Cuda available."""
    try:
        from mlframe.feature_selection.filters._fe_gpu_strict import _cuda_usable

        return bool(_cuda_usable())
    except Exception:
        return False


def _strict_both_enabled() -> bool:
    """Strict both enabled."""
    return os.environ.get("MLFRAME_TEST_STRICT_BOTH", "").strip().lower() in ("1", "true", "on", "yes")


# Default: single CPU param (suite unchanged). Opt-in flag + CUDA => also run the STRICT GPU-resident path.
_STRICT_PARAMS = ["0", "1"] if (_strict_both_enabled() and _cuda_available()) else [None]


def pytest_configure(config):
    """Pytest configure."""
    config.addinivalue_line(
        "markers",
        "strict_cpu_only: run this biz_val test only on the CPU FE path -- it asserts CPU-estimator MI magnitudes "
        "that edge-binning STRICT does not reproduce (selection converges at large n, absolute MI does not).",
    )


@pytest.fixture(params=_STRICT_PARAMS, ids=lambda m: "default" if m is None else f"strict={m}", autouse=True)
def _biz_val_strict_both_paths(request, monkeypatch):
    """Biz val strict both paths."""
    mode = request.param
    if mode is None:
        return  # flag off: single unparametrized run, env untouched (AUTO default -> CPU at these small n)
    if mode == "1" and request.node.get_closest_marker("strict_cpu_only"):
        pytest.skip("strict_cpu_only: MI-magnitude contract is CPU-estimator-specific")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", mode)
