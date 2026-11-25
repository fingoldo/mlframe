"""The device family gate with ``decide``: the flags of the batches inside the window are the full test's, and the null draws stop once those are settled."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _family_maxt_gate_device as device

if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

N = 4000
BINS = 6


class _Batch:
    """A one-column batch that holds its device codes, the way the lazy columns of a pair family do."""

    def __init__(self, codes):
        """Keep the ``(n, 1)`` device column."""
        self.dev = cp.asarray(codes.reshape(-1, 1))
        self.shape = self.dev.shape


def _family(seed: int, graded: int = 14, noise: int = 400, top: float = 0.45, bottom: float = 0.04):
    """``(columns, y)``: ``graded`` columns whose dependence on the class falls from ``top`` to ``bottom``, then ``noise`` independent columns."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N).astype(np.int64)
    cols = []
    for strength in np.linspace(top, bottom, graded):
        copy = rng.random(N) < strength
        cols.append(np.where(copy, y * 3 + rng.integers(0, 3, N) // 3, rng.integers(0, BINS, N)).astype(np.int8))
    cols += [rng.integers(0, BINS, N).astype(np.int8) for _ in range(noise)]
    return cols, y


@pytest.mark.parametrize("seed", [0, 7])
def test_the_device_window_gives_the_full_flags_to_the_leaders_with_fewer_draws(seed, monkeypatch):
    """Leaders' flags equal the full test's; the number of null permutations drawn does not grow, and shrinks when the columns near the threshold are far below the leaders."""
    cols, y = _family(seed)
    batches = [_Batch(c) for c in cols]
    drawn = []
    original = device._FamilyOnDevice.null_max

    def counting(self, m, rng):
        """The wrapped null draw, counted."""
        drawn.append(int(m))
        return original(self, m, rng)

    monkeypatch.setattr(device._FamilyOnDevice, "null_max", counting)
    full = device.family_pass_batches_device(batches, y, 0.01, 1999, seed)
    drawn_full = sum(drawn)
    drawn.clear()
    window = np.zeros(len(cols), dtype=bool)
    window[:2] = True
    decided = device.family_pass_batches_device(batches, y, 0.01, 1999, seed, window)
    assert full is not None and decided is not None
    assert decided[:2] == full[:2]
    assert all(decided[i] or not full[i] for i in range(len(cols)) if not window[i])
    assert sum(drawn) <= drawn_full
    if seed == 7:
        assert drawn_full > 0 and sum(drawn) < drawn_full


def test_a_window_without_an_undecided_batch_draws_no_permutation(monkeypatch):
    """When the triage settles every batch of the window, the null is never drawn."""
    cols, y = _family(3, graded=3, noise=100, top=0.9, bottom=0.8)
    batches = [_Batch(c) for c in cols]
    drawn = []
    original = device._FamilyOnDevice.null_max
    monkeypatch.setattr(device._FamilyOnDevice, "null_max", lambda self, m, rng: drawn.append(int(m)) or original(self, m, rng))
    window = np.zeros(len(cols), dtype=bool)
    window[:3] = True
    flags = device.family_pass_batches_device(batches, y, 0.01, 1999, 0, window)
    assert flags is not None and all(flags[:3])
    assert drawn == []
