"""Device family gate: null maxima stay double precision and the permutation block honours the index budget (run on a numpy stand-in for cupy)."""

import numpy as np

from mlframe.feature_selection.filters import _family_maxt_gate as G
from mlframe.feature_selection.filters import _family_maxt_gate_device as D


class _FakeFam(D._FamilyOnDevice):
    """A device family whose ``_parts`` are host blocks, so ``null_max`` runs without a card."""

    def __init__(self, n: int, values: np.ndarray) -> None:
        """Set only what ``null_max`` reads."""
        self.cp = np
        np.asarray  # noqa: B018
        self.y = np.zeros(n, dtype=np.int64)
        self.n, self.n_y, self.words = n, 2, (n + 63) // 64
        self._values = values

    def _parts(self, cls_host):
        """One ``(B, 1)`` block holding the preset value for each permutation row."""
        yield np.full((cls_host.shape[0], 1), self._values)


class _NpAsCupy:
    """numpy with the one cupy function ``null_max`` needs beyond numpy's own."""

    maximum = staticmethod(np.maximum)
    asnumpy = staticmethod(np.asarray)


def test_null_max_keeps_double_precision_when_the_criterion_dtype_is_float32(monkeypatch):
    """A null maximum just above 1.0 must not be rounded to float32 (1.0) before it is compared with a float64 observed value."""
    monkeypatch.setattr(G, "_permutation_block", lambda rng, y, m: np.zeros((m, y.shape[0]), dtype=np.int32))
    fam = _FakeFam(64, 1.0 + 1e-9)
    fam.cp = _NpAsCupy
    got = fam.null_max(3, np.random.default_rng(0))
    assert np.all(got == 1.0 + 1e-9)


def test_null_max_draws_permutations_in_blocks_within_the_index_budget(monkeypatch):
    """With a tiny index budget the device null draws small blocks, as the host null does, instead of a fixed 256."""
    sizes = []

    def _blk(rng, y, m):
        """Record the block size asked for."""
        sizes.append(m)
        return np.zeros((m, y.shape[0]), dtype=np.int32)

    monkeypatch.setattr(G, "_permutation_block", _blk)
    monkeypatch.setattr(G, "_PERM_BLOCK_BUDGET_BYTES", 64 * 2 * 10)  # 10 permutations of 64 rows at 2 bytes
    fam = _FakeFam(64, 0.5)
    fam.cp = _NpAsCupy
    out = fam.null_max(25, np.random.default_rng(0))
    assert out.shape == (25,)
    assert sizes == [10, 10, 5]
