"""Both KTC tuners must time the call shape production actually runs.

The calibration and inference tuners hoisted `cp.asarray` out of the timed region and returned a device
array, so the persisted crossover came from a GPU-resident workload -- while every production caller passes
a host array and pays the upload on entry and the download on exit. The measurement therefore systematically
over-selected `cupy`. The inference tuner was biased the other way on the CPU side as well: its njit lambdas
re-copied the input each iteration, an allocation `apply_logical_constraints` never makes.

The sibling `votenrank/_confidence_gated_blend_ktc_dispatch.py` already measured this correctly, and its own
docstring records what the difference is worth: "cupy resident 0.8 ms / cupy e2e 8.5 ms (host input: H2D
transfer dominates -- slower than njit_parallel)".

These tests substitute a fake `cupy` whose operations are numpy's, so the tuners run to completion with no
device present and the transfer calls can simply be counted: one transfer per timed invocation means it sits
inside the region, a single transfer for the whole measurement means it was hoisted back out.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


class _FakeCupy:
    """A `cupy` stand-in backed by numpy, counting the host/device transfers it is asked for."""

    def __init__(self):
        """Start with no transfers recorded."""
        self.h2d = 0
        self.d2h = 0
        self.log = np.log
        self.exp = np.exp
        self.clip = np.clip
        self.where = np.where

    def asarray(self, a, *args, **kwargs):
        """Stand in for the host-to-device upload."""
        self.h2d += 1
        return np.array(a, copy=True)

    def asnumpy(self, a):
        """Stand in for the device-to-host download."""
        self.d2h += 1
        return np.asarray(a)


@pytest.fixture
def fake_cupy(monkeypatch):
    """Install the fake as the importable `cupy` for the duration of one test."""
    fake = _FakeCupy()
    mod = types.ModuleType("cupy")
    for name in ("asarray", "asnumpy", "log", "exp", "clip", "where"):
        setattr(mod, name, getattr(fake, name))
    mod.cuda = types.SimpleNamespace(Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "cupy", mod)
    return fake


def test_the_calibration_tuner_measures_the_transfer_on_every_iteration(fake_cupy):
    """One upload and one download per timed call, not one for the whole measurement."""
    from mlframe.calibration._ktc_dispatch import _make_tuner

    _make_tuner(n=256, k=3)()
    assert fake_cupy.h2d > 1, f"the upload was hoisted out of the timed region ({fake_cupy.h2d} for the whole measurement)"
    assert fake_cupy.d2h == fake_cupy.h2d, f"{fake_cupy.h2d} uploads against {fake_cupy.d2h} downloads: the tuner is not returning to host"


def test_the_inference_tuner_measures_the_transfer_on_every_iteration(fake_cupy):
    """Same contract for `apply_logical_constraints`, whose `_apply_cupy` is also host-in, host-out."""
    from mlframe.inference._ktc_dispatch import _make_tuner

    _make_tuner(n=256, n_labels=6, n_rules=2)()
    assert fake_cupy.h2d > 1, f"the upload was hoisted out of the timed region ({fake_cupy.h2d} for the whole measurement)"
    assert fake_cupy.d2h == fake_cupy.h2d, f"{fake_cupy.h2d} uploads against {fake_cupy.d2h} downloads: the tuner is not returning to host"


def test_the_inference_tuner_gives_the_njit_side_a_restored_buffer(fake_cupy):
    """The in-place kernels must see violating data every iteration, without being charged for the restore.

    This one pins a contract rather than catching the old behaviour: the previous form restored the buffer
    too, just inside the timed region via `preds.copy()`. It guards the naive fix -- dropping the copy and
    passing the same buffer straight through, which would measure a converged no-op from the second call on.
    """
    import mlframe.inference.logical_constraints as lc
    from mlframe.inference._ktc_dispatch import _make_tuner

    seen_already_satisfied = []

    def _spy(out, rules_arr):
        """Record whether this call received data that still violates the rules."""
        violating = any(bool(np.any(out[:, int(c)] > out[:, int(p)])) for c, p in rules_arr)
        seen_already_satisfied.append(not violating)
        for c, p in rules_arr:
            c, p = int(c), int(p)
            v = out[:, c] > out[:, p]
            out[v, c], out[v, p] = out[v, p], out[v, c].copy()
        return out

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lc, "_apply_njit", _spy)
    monkey.setattr(lc, "_apply_njit_parallel", _spy)
    monkey.setattr(lc, "_NUMBA_AVAILABLE", True)
    try:
        _make_tuner(n=256, n_labels=6, n_rules=2)()
    finally:
        monkey.undo()

    assert seen_already_satisfied, "the njit side was never measured"
    assert not any(seen_already_satisfied), "a timed call ran on already-constrained data, measuring a no-op"


def test_the_measure_helper_runs_setup_outside_the_timed_region():
    """`setup` exists precisely so the restore is not charged to the backend being measured."""
    from mlframe._ktc_dispatch_shared import measure_backend

    import time

    calls = {"setup": 0, "fn": 0}

    def _setup():
        """A deliberately slow restore that must not land in the reported time."""
        calls["setup"] += 1
        time.sleep(0.02)

    def _fn():
        """The measured work, which is far cheaper than the restore."""
        calls["fn"] += 1

    elapsed = measure_backend(_fn, n_iters=3, setup=_setup)
    assert calls["setup"] == calls["fn"] == 4, f"setup/fn ran {calls['setup']}/{calls['fn']} times, expected one setup per call including the warmup"
    assert elapsed < 20.0, f"the untimed setup's 20ms leaked into the reported {elapsed:.2f}ms"


def test_the_measure_helper_still_works_without_a_setup():
    """The parameter is optional; every other tuner in the repo calls it positionally as before."""
    from mlframe._ktc_dispatch_shared import measure_backend

    assert measure_backend(lambda: None) >= 0.0
