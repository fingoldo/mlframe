"""The device family gate packs its code batches one by one at a column offset instead of concatenating them first.

Concatenating the batches copied the whole ``(n, k)`` code matrix (57 MB a call, 2.4 GB over a default 100000 x 100 fit) only so that one launch could read it. The bit sets, level counts, observed MI and the
permutation maxima must equal those of the concatenated matrix, for batches of different width and of different integer type, whatever the memory-budget group size.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._family_maxt_gate_device import _FamilyOnDevice


def _batches(n=3000, widths=(5, 1, 9), seed=0, dtypes=(np.int8, np.int8, np.int8)):
    """Device code batches of the given widths and integer types, 3 to 8 levels per column, and a 3-class target."""
    rng = np.random.default_rng(seed)
    devs = [cp.asarray(rng.integers(0, int(rng.integers(3, 9)), (n, w)).astype(dt)) for w, dt in zip(widths, dtypes)]
    y = rng.integers(0, 3, n).astype(np.int64)
    return devs, y


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dtypes", [(np.int8, np.int8, np.int8), (np.int8, np.int16, np.int8)])
@pytest.mark.parametrize("free_bytes", [None, 140_000])
def test_batches_packed_at_offsets_equal_the_concatenated_matrix(seed, dtypes, free_bytes, monkeypatch):
    """Observed MI, level counts and the next permutation maxima agree, for one memory group and for two (a free-memory figure that makes the group smaller than the 15 columns but still admits them)."""
    if free_bytes is not None:
        monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (free_bytes, 4 << 30))
    devs, y = _batches(seed=seed, dtypes=dtypes)
    split = _FamilyOnDevice(cp, devs, y, 3)
    whole = _FamilyOnDevice(cp, [cp.concatenate([d.astype(cp.int16) for d in devs], axis=1)], y, 3)
    np.testing.assert_array_equal(split.nx_occ, whole.nx_occ)
    np.testing.assert_allclose(split.observed(), whole.observed(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(split.null_max(6, np.random.default_rng(4)), whole.null_max(6, np.random.default_rng(4)), rtol=0, atol=1e-12)


def test_the_code_matrix_is_not_concatenated(monkeypatch):
    """No two-dimensional ``cp.concatenate`` (the old whole-matrix copy) runs when the batches are packed."""
    devs, y = _batches()
    seen = []
    real = cp.concatenate

    def spy(tup, axis=0, *a, **k):
        """Record the arity, dimensionality and axis of each concatenate, then run it."""
        seen.append((len(tup), getattr(tup[0], "ndim", 0), axis))
        return real(tup, axis, *a, **k)

    monkeypatch.setattr(cp, "concatenate", spy)
    _FamilyOnDevice(cp, devs, y, 3)
    assert not any(ndim == 2 for _n, ndim, _axis in seen), seen


def test_many_small_batches_make_one_launch_group():
    """Sixty batches of three columns fit one memory group: one bit-set group (one null launch and one maximum per permutation block), not sixty."""
    devs, y = _batches(n=2000, widths=(3,) * 60, seed=2, dtypes=(np.int8,) * 60)
    fam = _FamilyOnDevice(cp, devs, y, 3)
    assert len(fam.groups) == 1 and fam.groups[0][:2] == (0, 180)
    whole = _FamilyOnDevice(cp, [cp.concatenate(devs, axis=1)], y, 3)
    np.testing.assert_array_equal(fam.nx_occ, whole.nx_occ)
    np.testing.assert_allclose(fam.observed(), whole.observed(), rtol=0, atol=1e-12)


def test_a_group_boundary_inside_a_batch_is_packed_from_both_sides(monkeypatch):
    """A memory budget that ends a group in the middle of a batch splits that batch over two groups with the right column offsets."""
    devs, y = _batches(n=3000, widths=(7, 11, 4), seed=3)
    cap = max(int(d.max()) for d in devs) + 1
    column_bytes = cap * ((3000 + 63) // 64) * 8
    free = 3 * column_bytes * 22 + 3  # admits the 22 columns; the group is then 3/4 of them (16), which ends inside the second batch (columns 7 to 17)
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (free, 4 << 30))
    fam = _FamilyOnDevice(cp, devs, y, 3)
    assert len(fam.groups) >= 2
    whole = _FamilyOnDevice(cp, [cp.concatenate(devs, axis=1)], y, 3)
    np.testing.assert_allclose(fam.observed(), whole.observed(), rtol=0, atol=1e-12)
