"""mi_direct_gpu honours an explicit max_failed instead of fanning out to the batched kernel that ignores it."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import gpu as gpu_mod


@pytest.fixture
def _paths(monkeypatch):
    """Replace both permutation back ends with markers so no GPU is needed."""
    monkeypatch.setattr(gpu_mod, "_ensure_kernels_inited", lambda: None)
    monkeypatch.setattr(gpu_mod, "_pin_device_if_needed", lambda: None)
    monkeypatch.setattr(gpu_mod, "mi_direct_gpu_batched", lambda **k: ("batched",))
    seen = {}

    def _loop(*args):
        """Record the max_failed the per-iteration loop received."""
        seen["max_failed"] = args[7]
        return args[6], 1.0, 0, 0.0, 1

    monkeypatch.setattr(gpu_mod, "_gpu_single_perm_loop", _loop)
    return seen


def _call(max_failed):
    """Run mi_direct_gpu on a dependent pair with 64 permutations."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, size=500)
    data = np.column_stack([a, (a + rng.integers(0, 2, size=500)) % 3]).astype(np.int32)
    return gpu_mod.mi_direct_gpu(
        data, x=(0,), y=(1,), factors_nbins=np.array([3, 3]), npermutations=64, max_failed=max_failed, min_nonzero_confidence=0.95,
    )


def test_explicit_max_failed_stays_on_per_iteration_loop(_paths):
    """A max_failed that differs from the batched path's derived cap must reach the loop."""
    assert _call(1) != ("batched",)
    assert _paths["max_failed"] == 1


@pytest.mark.parametrize("mf", [None, 3])
def test_default_or_equal_max_failed_still_fans_out(_paths, mf):
    """None, or a cap equal to the derived one, keeps the batched fast path."""
    assert _call(mf) == ("batched",)
