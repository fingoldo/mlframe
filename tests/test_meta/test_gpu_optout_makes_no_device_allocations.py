"""Meta-test: the global GPU opt-out must produce ZERO device allocations, not merely "prefer" the CPU.

``MLFRAME_DISABLE_GPU=1`` / ``CUDA_VISIBLE_DEVICES=""`` is the project's "no GPU on this run" contract.
The failure mode it exists for is silent: cupy's own detection ignores both, so a dispatch path gated only
on cupy PRESENCE still routes to the device, and the only symptom is wall-clock (a weak GPU spent ~37% of a
300k fit in cupy argsort + GPU-sync sleep with the CPU idle). Nothing raises, so no ordinary test catches it.

A structural gate ("every GPU module must import ``gpu_globally_disabled``") does not work here: most CUDA
modules are leaf kernels whose caller owns the decision, so the rule would be ~100 false positives and zero
signal. This asserts the observable contract instead - run a fit with every FE family enabled under the
opt-out and require that not one byte of device memory is allocated by any backend.

Both allocation routes are counted: cupy device memory (via ``cupy.cuda.set_allocator``, which sees every
cupy entry point regardless of which ``cp.*`` call made it) and numba's own device arrays (which bypass
cupy's allocator entirely).
"""

from __future__ import annotations

import traceback

import numpy as np
import pytest


@pytest.fixture
def device_alloc_counter(monkeypatch):
    """Yield a one-element list counting every device allocation attempted during the block.

    Hooks the allocator rather than individual array constructors: ``cp.zeros``/``cp.asarray``/``cp.empty``
    and every kernel-internal temporary all funnel through it, so a path that reaches the device cannot slip
    past by using an API the test did not think to patch.
    """
    calls: list[int] = []
    hooked_any = False

    # One-time CAPABILITY probes, not dispatch: mlframe's startup check for a broken cupy install, and
    # numba's "can this device actually compile a kernel" probe (device presence does not imply NVVM
    # support). Both must run to decide what the machine can do at all, they allocate a handful of bytes
    # once per process, and neither routes any of the fit's work to the GPU.
    _PROBE_FRAMES = frozenset({"_disable_broken_cupy", "_ensure_gpu_runtime_configured", "numba_cuda_can_compile"})

    def _is_probe() -> bool:
        """True when the current stack is one of the documented capability probes."""
        return any(fr.name in _PROBE_FRAMES for fr in traceback.extract_stack())

    try:
        import cupy as cp
    except Exception:
        cp = None
    if cp is not None:
        prev = cp.cuda.get_allocator()

        def _counting_alloc(size):
            """Record the allocation, then satisfy it normally so the fit under test still works."""
            if not _is_probe():
                calls.append(int(size))
            return prev(size)

        cp.cuda.set_allocator(_counting_alloc)
        hooked_any = True

    try:
        from numba import cuda as _nbcuda
    except Exception:
        _nbcuda = None
    if _nbcuda is not None:
        for _name in ("to_device", "device_array", "device_array_like", "pinned_array"):
            _orig = getattr(_nbcuda, _name, None)
            if _orig is None:
                continue

            def _make(orig):
                """Bind ``orig`` per-name so the wrapper does not close over the loop variable."""

                def _wrapped(*a, **kw):
                    """Count this device-array construction unless it comes from a capability probe."""
                    if not _is_probe():
                        calls.append(-1)
                    return orig(*a, **kw)

                return _wrapped

            monkeypatch.setattr(_nbcuda, _name, _make(_orig), raising=False)
        hooked_any = True

    if not hooked_any:
        pytest.skip("neither cupy nor numba.cuda importable - nothing could allocate on a device anyway")

    try:
        yield calls
    finally:
        if cp is not None:
            cp.cuda.set_allocator(prev)


def test_gpu_optout_allocates_nothing_on_device(monkeypatch, device_alloc_counter):
    """A full MRMR fit with every FE family on allocates zero device memory under the opt-out."""
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")

    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    assert gpu_globally_disabled(), "opt-out not in effect - the rest of this test would prove nothing"

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(20260726)
    n = 4000
    X = rng.normal(size=(n, 12))
    # A target the FE families can actually earn signal on, so the GPU-eligible scan paths really run:
    # a pure-noise target can short-circuit early and never reach a dispatcher.
    y = ((X[:, 0] * X[:, 1] + np.sin(3.0 * X[:, 2]) + 0.3 * rng.normal(size=n)) > 0).astype(np.int64)

    sel = MRMR(verbose=0, random_seed=0, fe_max_steps=2)
    sel.fit(X, y)

    assert getattr(sel, "support_", None) is not None, "fit did not complete - allocation count is meaningless"
    assert not device_alloc_counter, (
        f"{len(device_alloc_counter)} device allocation(s) during a fit with MLFRAME_DISABLE_GPU=1. "
        "The opt-out is a hard contract, not a preference: some dispatch path is gated on cupy/numba "
        "PRESENCE instead of consulting gpu_globally_disabled(). Find it and add the gate - the symptom "
        "in production is silent wall-clock loss on a weak or busy GPU, never an exception."
    )


def test_cuda_visible_devices_empty_is_honoured_like_the_explicit_flag(monkeypatch):
    """``CUDA_VISIBLE_DEVICES=""`` alone must disable the GPU - it is the documented convention.

    Cheap companion to the allocation test: the expensive one sets the explicit flag, so without this a
    regression that honoured only ``MLFRAME_DISABLE_GPU`` would pass everything.
    """
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_globally_disabled()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert not gpu_globally_disabled(), "a real device list must NOT be read as an opt-out"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert not gpu_globally_disabled(), "an unset variable is 'no preference', not an opt-out"
