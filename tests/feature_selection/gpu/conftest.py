"""GPU test-suite fixtures.

Resets the process-lifetime device-probe memo in ``_fe_gpu_strict`` between tests so the GPU parity / selection-
equivalence tests are order-independent under a global ``MLFRAME_FE_GPU_STRICT_RESIDENT=1`` / ``MLFRAME_FE_GPU_STRICT=1``
suite run.

The STRICT *flag* itself is read live now (``fe_gpu_strict_enabled`` no longer caches the env value), so a test
that sets/unsets the flag no longer poisons a later test. The only remaining process-lifetime state is the immutable
CUDA-device probe ``_CUDA_USABLE_CACHE``; an opt-out test that runs under ``MLFRAME_DISABLE_GPU=1`` / ``CUDA_VISIBLE_DEVICES=""``
and happens to trigger the probe would freeze it to False for later parity tests. Resetting it at each boundary keeps the
probe honest regardless of test order (mirrors the ``_CC_MAJOR_CACHE`` reset precedent in
``test_gpu_cpu_mi_selection_equivalence``)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_fe_gpu_strict_device_probe():
    """Clear the cached CUDA-device probe before each GPU test so an opt-out test's transient
    ``MLFRAME_DISABLE_GPU`` / ``CUDA_VISIBLE_DEVICES`` cannot leak a stale False into a later parity test."""
    try:
        import mlframe.feature_selection.filters._fe_gpu_strict as _strict

        _strict._CUDA_USABLE_CACHE = None
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        pass
    yield
    try:
        import mlframe.feature_selection.filters._fe_gpu_strict as _strict

        _strict._CUDA_USABLE_CACHE = None
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        pass
