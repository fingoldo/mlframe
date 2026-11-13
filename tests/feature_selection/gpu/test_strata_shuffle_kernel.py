"""The per-stratum Fisher-Yates kernel replacing the flat key sort in the conditional permutation null."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as pn
from mlframe.feature_selection.filters._strata_shuffle_gpu import MIN_STRATA, strata_shuffle

pytestmark = pytest.mark.gpu


def _layout(n, nrank, kx, seed=0):
    """A z-sorted layout: dense stratum ranks, starts and an x column of ``kx`` values."""
    rng = np.random.default_rng(seed)
    z = np.sort(rng.integers(0, nrank, n))
    _, rank = np.unique(z, return_inverse=True)
    x = rng.integers(0, kx, n).astype(np.int64)
    return cp.asarray(x), cp.asarray(rank.astype(np.float64)), int(rank.max()) + 1, rank


def test_each_stratum_keeps_its_multiset():
    """Every permutation is a rearrangement inside each stratum: the per-stratum value counts never change."""
    x, zr, nr, rank = _layout(20000, 5000, 6)
    out = cp.asnumpy(strata_shuffle(cp, cp.random.RandomState(3), x, zr, 7, nr))
    xh = cp.asnumpy(x)
    for p in range(out.shape[0]):
        assert np.array_equal(np.bincount(rank * 6 + out[p], minlength=nr * 6), np.bincount(rank * 6 + xh, minlength=nr * 6))


def test_shuffles_move_values_and_differ_between_permutations():
    """The draw is a real shuffle: most rows change and two permutations are not the same."""
    x, zr, nr, _ = _layout(20000, 2000, 50)
    out = cp.asnumpy(strata_shuffle(cp, cp.random.RandomState(4), x, zr, 4, nr))
    assert (out[0] != cp.asnumpy(x)).mean() > 0.5
    assert (out[0] != out[1]).mean() > 0.5


def test_position_is_uniform_inside_a_stratum():
    """A value of a 10-row stratum lands on every position about equally often (a chi-square style bound)."""
    n_str, m = 3000, 10
    x = cp.asarray(np.tile(np.arange(m), n_str).astype(np.int64))
    zr = cp.asarray(np.repeat(np.arange(n_str), m).astype(np.float64))
    out = cp.asnumpy(strata_shuffle(cp, cp.random.RandomState(5), x, zr, 40, n_str)).reshape(40, n_str, m)
    counts = np.stack([(out == v).sum(axis=(0, 1)) for v in range(m)])
    expect = 40 * n_str / m
    assert np.abs(counts - expect).max() < 5 * np.sqrt(expect)


def test_flat_shuffle_routes_to_kernel_above_the_gate(monkeypatch):
    """With ``MLFRAME_STRATA_SHUFFLE_KERNEL=1`` and enough strata ``_flat_within_stratum_shuffle`` uses the kernel; by default it keeps the sort."""
    x, zr, nr, _ = _layout(20000, 5000, 6)
    assert nr >= MIN_STRATA
    seen = []
    real = pn.strata_shuffle
    monkeypatch.setattr(pn, "strata_shuffle", lambda *a, **k: seen.append(1) or real(*a, **k))
    monkeypatch.setenv("MLFRAME_STRATA_SHUFFLE_KERNEL", "1")
    pn._flat_within_stratum_shuffle(cp, cp.random.RandomState(1), x, zr[:, None], 5, 6, nr)
    assert seen
    monkeypatch.delenv("MLFRAME_STRATA_SHUFFLE_KERNEL")
    seen.clear()
    assert pn._flat_within_stratum_shuffle(cp, cp.random.RandomState(1), x, zr[:, None], 5, 6, nr).shape == (5, 20000)
    assert not seen
