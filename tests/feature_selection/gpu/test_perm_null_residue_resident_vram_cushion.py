"""``perm_null_residue_mis_resident`` checks the VRAM cushion before allocating its (n, n_perm) code matrix, instead of
retrying a doomed allocation on every candidate once the device pool is pinned at its limit (GPU11-13)."""

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


def _fixture(seed=0, n=4000, n_perm=6, k=5):
    """A residue column, a 3-class target, and n_perm index permutations - the shapes perm_null_residue_mis_resident expects."""
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    rng = np.random.default_rng(seed)
    c = rng.integers(0, 50, size=n).astype(np.float64)
    r = np.mod(c, k).astype(np.float64)
    y = rng.integers(0, 3, size=n)
    yi = encode_y_for_classif_mi(y)
    perms = [rng.permutation(n) for _ in range(n_perm)]
    return r, y, perms, yi


def test_a_refused_cushion_returns_none_without_touching_cupy(monkeypatch):
    """When fe_gpu_has_vram_cushion declines, the function returns None immediately - no RawModule/rank-bin call, no OutOfMemoryError retry."""
    import mlframe.feature_selection.filters._pairwise_modular_resident as pmr

    r, y, perms, _ = _fixture()
    # fe_gpu_has_vram_cushion is imported inside the function body (lazy import), so patch the module it is imported FROM.
    import mlframe.feature_selection.filters._fe_gpu_vram as vram_mod

    called = {"n": 0}

    def _declining(*a, **k):
        """Stand-in for fe_gpu_has_vram_cushion that always declines, counting its own calls."""
        called["n"] += 1
        return False

    monkeypatch.setattr(vram_mod, "fe_gpu_has_vram_cushion", _declining)
    out = pmr.perm_null_residue_mis_resident(r, y, perms, eff_nbins=12, rank_binning=True)
    assert out is None
    assert called["n"] == 1, "the cushion check must run exactly once per call, before any allocation"


def test_a_granted_cushion_matches_the_host_per_permutation_loop():
    """With the cushion available, the resident batch matches the host loop's MI values exactly (the code-reorder identity is bit-identical, not approximate)."""
    import mlframe.feature_selection.filters._pairwise_modular_fe as pm
    import mlframe.feature_selection.filters._pairwise_modular_resident as pmr
    from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

    r, y, perms, yi = _fixture()
    got = pmr.perm_null_residue_mis_resident(r, y, perms, eff_nbins=12, rank_binning=True)
    assert got is not None, "the device path must run on this fixture (small, well under any real cushion)"
    ref = np.array([pm._mi(r, yi[p], nbins=12) for p in perms], dtype=_crit_np_dtype())
    assert np.array_equal(np.asarray(got, dtype=_crit_np_dtype()), ref), f"resident batch diverged from the host loop: {got} vs {ref}"
