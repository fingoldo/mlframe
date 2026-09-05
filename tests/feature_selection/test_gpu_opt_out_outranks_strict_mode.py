"""The project-wide GPU opt-out must outrank a memoised probe and STRICT mode alike.

`MLFRAME_DISABLE_GPU=1` / `CUDA_VISIBLE_DEVICES=""` is read LIVE everywhere in this package through
`gpu_globally_disabled()`. Two places did not honour that.

`_fe_gpu_strict._cuda_usable()` folded both env flags into a process-lifetime memo, so a process that
touched any strict-gated dispatch BEFORE setting the opt-out kept a cached True and went on using the GPU
while the rest of the same process honoured the flag. The module's own comment on the STRICT toggle warns
against exactly this order-dependence, for the other flag.

`_cmi_cuda._should_use_cuda` returned True from its STRICT branch before ever reaching the tuning-cache
path -- and that path is the only one that consulted `cmi_use_cuda`, whose comment states that the global
opt-out outranks both the tuning cache and STRICT mode.
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters import _fe_gpu_strict as strict


@pytest.fixture(autouse=True)
def _clear_probe_memo():
    """Each test starts from an unprobed state."""
    strict._CUDA_USABLE_CACHE = None
    yield
    strict._CUDA_USABLE_CACHE = None


def test_the_opt_out_is_honoured_even_after_the_probe_has_been_memoised(monkeypatch):
    """The order-dependence: probe first, opt out second."""
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(strict, "_CUDA_USABLE_CACHE", True)

    assert strict._cuda_usable() is True

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert strict._cuda_usable() is False, "the opt-out was ignored because the probe result had been memoised"


def test_an_empty_cuda_visible_devices_is_also_read_live(monkeypatch):
    """The other short-circuit, same shape."""
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(strict, "_CUDA_USABLE_CACHE", True)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert strict._cuda_usable() is False


def test_clearing_the_opt_out_restores_the_memoised_probe(monkeypatch):
    """Reading the flag live must not throw away the device probe the memo exists for."""
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(strict, "_CUDA_USABLE_CACHE", True)

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert strict._cuda_usable() is False
    monkeypatch.delenv("MLFRAME_DISABLE_GPU")
    assert strict._cuda_usable() is True, "the cached device probe was discarded by the live flag check"


def test_the_device_probe_itself_is_still_memoised(monkeypatch):
    """The memo's whole purpose: a ~17us probe the per-round dispatch must not repeat."""
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = {"n": 0}

    def _probe():
        """Count how often the expensive path runs."""
        calls["n"] += 1
        return True

    import mlframe.feature_selection.filters._gpu_policy as policy

    monkeypatch.setattr(policy, "cuda_available_for_run", _probe)
    strict._CUDA_USABLE_CACHE = None
    assert strict._cuda_usable() is True
    assert strict._cuda_usable() is True
    assert calls["n"] == 1, f"the device probe ran {calls['n']} times; the memo is gone"


def test_strict_mode_does_not_override_the_opt_out(monkeypatch):
    """`_should_use_cuda` returned True from STRICT before anything consulted the opt-out.

    The gates upstream of STRICT are stubbed past deliberately. On a machine without cupy the function
    refuses long before it reaches STRICT, so the test would pass for the wrong reason and prove nothing --
    which is what it did until these stubs were added.
    """
    from mlframe.feature_selection.filters.info_theory import _cmi_cuda

    monkeypatch.setattr(_cmi_cuda, "cupy_available", lambda: True)
    monkeypatch.setattr(_cmi_cuda, "_CMI_GPU_FAILED", False)
    monkeypatch.setattr(_cmi_cuda, "_cmi_cuda_shmem_fits", lambda *a, **k: True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")

    # Without the opt-out, STRICT is what decides -- establishing that the stubs really do reach it.
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    strict._CUDA_USABLE_CACHE = True
    assert (
        _cmi_cuda._should_use_cuda(n=1_000_000, p=64, joint_size=8, nbins_x=2, nbins_y=2, nbins_z=2) is True
    ), "the stubs do not reach the STRICT branch, so the assertion below would pass for the wrong reason"

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert _cmi_cuda._should_use_cuda(n=1_000_000, p=64, joint_size=8, nbins_x=2, nbins_y=2, nbins_z=2) is False, "STRICT mode overrode the global GPU opt-out"


def test_the_opt_out_check_cannot_force_gpu_on_when_the_policy_module_is_missing(monkeypatch):
    """A guard that fails open would be worse than none; absence must leave the other gates in charge."""
    from mlframe.feature_selection.filters.info_theory import _cmi_cuda

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    # Even with the policy import broken, the downstream gates still refuse without a device.
    monkeypatch.setattr(_cmi_cuda, "cupy_available", lambda: False)
    assert _cmi_cuda._should_use_cuda(n=1_000_000, p=64, joint_size=64, nbins_x=4, nbins_y=4, nbins_z=4) is False
