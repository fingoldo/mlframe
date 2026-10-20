"""The residue grid scores every modulus from one broadcast residue block and returns exactly what the per-modulus build returned."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _strict_resident(monkeypatch):
    """The device-born modular path is on only under the strict-resident switch."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")


def _per_modulus_reference(c, y, mods, nbins):
    """The previous formulation: one residue column per modulus (mod, cast), stacked and transposed per group of equal effective bin count."""
    from mlframe.feature_selection.filters._fe_resident_operands import resident_operand
    from mlframe.feature_selection.filters._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    cf = np.ascontiguousarray(np.asarray(c, dtype=np.int64)).ravel()
    cg = resident_operand(cf, ("modular_combiner", cf.shape[0]), dtype=np.int64)
    yi = np.ascontiguousarray(encode_y_for_classif_mi(y))
    yd = resident_operand(yi, "y_mi_classif", dtype=np.int64)
    ymin = int(yi.min())
    ncls = int(yi.max()) - ymin + 1
    out = np.empty(len(mods), dtype=np.float64)
    groups = {}
    for idx, k in enumerate(mods):
        groups.setdefault(max(int(nbins), int(k)), []).append(idx)
    for eff, idxs in groups.items():
        mat = cp.ascontiguousarray(cp.stack([(cg % int(mods[i])).astype(cp.float64) for i in idxs], axis=1))
        mis = _plugin_mi_classif_batch_cuda_resident(mat, yd, int(eff), y_min=ymin, n_classes=ncls, relax_binning=True)
        for j, i in enumerate(idxs):
            out[i] = float(np.asarray(mis, dtype=np.float64)[j])
    return out


@pytest.mark.parametrize(
    "n,classes,mods,nbins",
    [
        (2000, 2, list(range(2, 19)), 12),
        (3000, 4, [2, 3, 5, 7, 11, 13, 17, 19, 23], 8),
        (1500, 3, [9, 5, 2, 14, 3, 14], 6),
        (999, 2, [2], 12),
    ],
)
def test_block_build_is_bitwise_the_per_modulus_build(n, classes, mods, nbins):
    """Residue grids with consecutive, scattered, repeated and single moduli, binary and multiclass labels, agree to the last bit."""
    from mlframe.feature_selection.filters._pairwise_modular_resident import residue_grid_mis_resident

    rng = np.random.default_rng(n)
    c = rng.integers(0, 10**6, n).astype(np.int64)
    y = rng.integers(0, classes, n).astype(np.int64)
    got = residue_grid_mis_resident(c, y, mods, nbins=nbins)
    ref = _per_modulus_reference(c, y, mods, nbins)
    assert got is not None
    assert got.view(np.uint64).tolist() == ref.view(np.uint64).tolist()


def test_empty_moduli_and_the_off_switch_keep_their_contract(monkeypatch):
    """No moduli give an empty float64 array; without the strict-resident switch the host path is taken (``None``)."""
    from mlframe.feature_selection.filters._pairwise_modular_resident import residue_grid_mis_resident

    y = np.zeros(50, dtype=np.int64)
    c = np.arange(50, dtype=np.int64)
    empty = residue_grid_mis_resident(c, y, [], nbins=8)
    assert empty is not None and empty.shape == (0,) and empty.dtype == np.float64
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    assert residue_grid_mis_resident(c, y, [2, 3], nbins=8) is None
