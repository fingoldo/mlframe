"""The parallel kernel behind the pooled interaction-information null floor gives the floor of the numpy per-pair loop (agreement to rounding)."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _interaction_information as II


def _pool(seed, n=3000, m=9, k_y=3):
    """Random ordinal codes with different bin counts, a target, and the pair pool of all column pairs."""
    rng = np.random.default_rng(seed)
    nbins = rng.integers(3, 9, m).astype(np.int64)
    data = np.stack([rng.integers(0, int(b), n) for b in nbins], axis=1).astype(np.int32)
    y = rng.integers(0, k_y, n).astype(np.int64)
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    pa = np.array([p[0] for p in pairs], dtype=np.int64)
    pb = np.array([p[1] for p in pairs], dtype=np.int64)
    freqs = np.bincount(y, minlength=k_y) / n
    return data, nbins, pa, pb, y, freqs


@pytest.mark.parametrize("miller_madow", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_kernel_floor_equals_the_loop_floor(monkeypatch, seed, miller_madow):
    """Same shuffles, same pool: the kernel and the numpy loop return the same floor."""
    data, nbins, pa, pb, y, freqs = _pool(seed)
    kw = dict(factors_data=data, nbins=nbins, pair_a=pa, pair_b=pb, classes_y=y, freqs_y=freqs, marginal_mi_a=np.zeros(len(pa)), marginal_mi_b=np.zeros(len(pa)), n_permutations=12, random_seed=5, miller_madow=miller_madow)
    monkeypatch.setattr(II, "_KERNEL_ENABLED", False)
    want = II.pooled_pair_ii_null_floor(**kw)
    monkeypatch.setattr(II, "_KERNEL_ENABLED", True)
    got = II.pooled_pair_ii_null_floor(**kw)
    assert got == pytest.approx(want, rel=0, abs=1e-12)
