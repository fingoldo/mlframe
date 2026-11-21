"""The batched baseline and residue-grid scoring of the pairwise-modular scan gives the per-combiner values, and the scan returns the same hits."""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _pairwise_modular_fe as pm
from mlframe.feature_selection.filters import _pairwise_modular_resident as res


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    """The device-born modular path is on under the strict-resident mode."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")


def _data(n=6000, seed=0):
    """Integer columns with a parity-of-sum-mod-3 target and a few noise columns."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f"i{j}": r.integers(0, 12, n) for j in range(6)})
    y = ((X["i0"] + X["i1"]) % 3 == 0).astype(int).to_numpy()
    return X, y


def test_batched_values_equal_per_combiner_values():
    """Baselines and grids of a chunk of combiners equal the single-combiner functions' values."""
    X, y = _data()
    combiners = [np.asarray(pm._combine([X[a].to_numpy(), X[b].to_numpy()], op)) for a, b in (("i0", "i1"), ("i2", "i3"), ("i0", "i4")) for op in pm._PAIR_OPS]
    mods = [2, 3, 4, 5, 7, 9, 12, 16]
    out = res.combiner_bases_and_grids_resident(combiners, y, mods, nbins=12, rank_binning=True)
    assert out is not None
    bases, grids = out
    for i, c in enumerate(combiners):
        one_grid = res.residue_grid_mis_resident(c, y, mods, nbins=12)
        one_base = res.combiner_mi_resident(c, y, nbins=12, rank_binning=True, modulus=0)
        np.testing.assert_array_equal(grids[i], one_grid)
        assert bases[i] == one_base


def test_scan_hits_equal_with_and_without_the_batch(monkeypatch):
    """``cheap_modular_scan`` returns the same hits (operator, columns, modulus, MIs) with the batched scoring and with the per-combiner path."""
    X, y = _data(seed=1)
    got = pm.cheap_modular_scan(X, y, max_pairs=4, nbins=12, seed=0)
    monkeypatch.setattr(res, "combiner_bases_and_grids_resident", lambda *a, **k: None)
    want = pm.cheap_modular_scan(X, y, max_pairs=4, nbins=12, seed=0)
    assert len(got) == len(want) > 0
    for g, w in zip(got, want):
        assert (g.op, g.cols, g.modulus) == (w.op, w.cols, w.modulus)
        assert g.residue_mi == w.residue_mi and g.baseline_mi == w.baseline_mi and g.null_hi == w.null_hi
