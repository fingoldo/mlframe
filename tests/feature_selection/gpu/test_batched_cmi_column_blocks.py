"""``batched_cmi_gpu`` joins column blocks into exactly the result of the one-shot call, in every return form."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _device_blocks
from mlframe.feature_selection.filters import _fe_batched_mi as M


def _data(seed=0, n=3000, k=23, kx=6):
    """Candidate codes, a binary target that depends on two of them, and a 9-class support."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, kx, (n, k)).astype(np.int64)
    y = ((X[:, 0] + X[:, 1] + rng.integers(0, 4, n)) > 6).astype(np.int64)
    z = rng.integers(0, 9, n).astype(np.int64)
    return X, y, z


def _both(monkeypatch, X, y, z, **kw):
    """The one-shot result and the result with the block forced to 7 columns."""
    once = M._batched_cmi_gpu_once(X, y, z, **kw)
    monkeypatch.setattr(_device_blocks, "first_block", lambda *_a, **_k: 7)
    blocked = M.batched_cmi_gpu(X, y, z, **kw)
    return once, blocked


@pytest.mark.parametrize("z_kind", ["none", "support"])
def test_values_equal_the_one_shot_call(monkeypatch, z_kind):
    """Marginal and conditional CMI of 23 columns in blocks of 7 equal the one-shot vector."""
    X, y, z = _data()
    once, blocked = _both(monkeypatch, X, y, None if z_kind == "none" else z)
    assert blocked.shape == once.shape and np.array_equal(once, blocked)


def test_cards_and_resident_forms_join_in_order(monkeypatch):
    """The ``return_cards`` form (host and resident) joins the per-column cards and keeps the shared scalars."""
    X, y, z = _data(1)
    for dev in (False, True):
        once, blocked = _both(monkeypatch, X, y, z, return_cards=True, return_device=dev)
        for a, b in zip(once, blocked):
            assert np.array_equal(cp.asnumpy(a) if hasattr(a, "get") else np.asarray(a), cp.asnumpy(b) if hasattr(b, "get") else np.asarray(b))


def test_device_input_and_precomputed_terms_are_accepted(monkeypatch):
    """A resident candidate matrix is sliced on the device, and a shared (x, z) count rides through every block."""
    X, y, z = _data(2)
    once, blocked = _both(monkeypatch, cp.asarray(X), y, z)
    assert np.array_equal(once, blocked)
    once, blocked = _both(monkeypatch, X, y, z, precomp_xz=(1.5, 40))
    assert np.array_equal(once, blocked)


def test_a_matrix_that_fits_is_one_call(monkeypatch):
    """With room for every column the wrapper makes a single one-shot call."""
    X, y, z = _data(3)
    calls = []
    real = M._batched_cmi_gpu_once
    monkeypatch.setattr(M, "_batched_cmi_gpu_once", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    monkeypatch.setattr(_device_blocks, "first_block", lambda *_a, **_k: 10**6)
    M.batched_cmi_gpu(X, y, z)
    assert len(calls) == 1
