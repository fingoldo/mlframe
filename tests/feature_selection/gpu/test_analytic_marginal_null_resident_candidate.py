"""The marginal analytic CMI null of a resident candidate counts its occupied classes on the card and gives the host-candidate answer."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters._fe_cmi_redundancy_null import _conditional_perm_null
from tests.feature_selection.gpu._gpu_support import has_device

pytestmark = pytest.mark.gpu


def _case(n=60_000, kx=9, ky=3, seed=0):
    """A candidate code column with one empty class (so occupancy differs from the declared bins) and a target."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, kx, n).astype(np.int64)
    x[x == 4] = 5
    y = rng.integers(0, ky, n).astype(np.int64)
    return x, y


def test_resident_and_host_candidates_give_the_same_null():
    """Device occupancy count equals the host one: the (floor, mean) pair is identical for a device and a host column, and the empty class is not counted."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    x, y = _case()
    host = _conditional_perm_null(x, y, None, n_permutations=8, quantile=0.95, seed=1, salt=2)
    dev = _conditional_perm_null(cp.asarray(x), y, None, n_permutations=8, quantile=0.95, seed=1, salt=2)
    assert host[1] > 0
    assert dev == host


def test_resident_candidate_is_not_copied_back_whole(monkeypatch):
    """No transfer of the candidate's size crosses to the host while the marginal analytic null runs on a resident column."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    x, y = _case()
    d = cp.asarray(x)
    sizes = []
    real = cp.ndarray.get

    def spy(self, *a, **k):
        """Record the byte size of every D2H copy."""
        sizes.append(int(self.nbytes))
        return real(self, *a, **k)

    monkeypatch.setattr(cp.ndarray, "get", spy)
    _conditional_perm_null(d, y, None, n_permutations=8, quantile=0.95, seed=1, salt=2)
    assert max(sizes, default=0) < x.nbytes, f"a column-sized D2H copy happened: {sorted(sizes)[-3:]}"
