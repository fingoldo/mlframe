"""The block entropy of the pooled pair null equals the per-pair bincount loop it replaced (codes of unequal cardinality, a constant row, a row using the whole range), within double rounding, for blocks of one pair and many."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _permutation_null_pair_resident as m
from tests.feature_selection.gpu._gpu_support import has_device


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp


def _loop(cp, cls_x, cards, inv_n):
    """The per-pair loop."""
    out = cp.empty(cls_x.shape[0])
    for p in range(cls_x.shape[0]):
        cnt = cp.bincount(cls_x[p], minlength=int(cards[p]))[: int(cards[p])].astype(cp.float64)
        px = cnt * inv_n
        nz = px > 0
        out[p] = -cp.sum(cp.where(nz, px * cp.log(cp.where(nz, px, 1.0)), 0.0))
    return out


@pytest.mark.parametrize("n,n_pairs", [(500, 1), (4000, 17), (20_000, 60)])
def test_equals_the_loop(n, n_pairs):
    """Same entropies for all pairs and for a prefix of them."""
    cp = _cp()
    rng = np.random.default_rng(0)
    cards = rng.integers(2, 120, n_pairs)
    cls = np.stack([rng.integers(0, c, n) for c in cards]).astype(np.int64)
    cls[0] = 0
    cards[-1] = 150
    cls[-1] = rng.integers(0, 150, n)
    cls_x = cp.asarray(cls)
    inv_n = 1.0 / n
    want = cp.asnumpy(_loop(cp, cls_x, cards, inv_n))
    got = cp.asnumpy(m._joint_code_entropies(cp, cls_x, int(cards.max()), inv_n))
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)
    small = cp.asnumpy(m._joint_code_entropies(cp, cls_x[:3], int(cards.max()), inv_n))
    np.testing.assert_allclose(small, want[:3], rtol=1e-13, atol=1e-13)
