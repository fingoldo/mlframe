"""The contiguous-column route of the batched pair joint entropy gives the strided kernel's values."""
import numpy as np
import pytest

from mlframe.feature_selection.filters import _dcd_pair_su_batch as M

pytestmark = pytest.mark.skipif(not M._HAVE_NUMBA, reason="needs numba")


def _case(n=3000, k_cols=40, k_pairs=50, seed=0):
    """A code matrix, pair lists sharing an anchor column and the per-column bin counts."""
    rng = np.random.default_rng(seed)
    nb = rng.integers(2, 9, k_cols).astype(np.int64)
    fd = np.stack([rng.integers(0, b, n) for b in nb], axis=1).astype(np.int32)
    a = np.zeros(k_pairs, np.int64)
    b = rng.integers(1, k_cols, k_pairs).astype(np.int64)
    return fd, a, b, nb


@pytest.mark.parametrize("k_pairs", [5, 50, 200])
def test_values_equal_the_strided_kernel(k_pairs):
    """Forced through each route, the helper's table equals the strided kernel's to the last bit."""
    fd, a, b, nb = _case(k_pairs=k_pairs)
    expected = M._batch_joint_entropy_pairs(fd, a, b, nb)
    used, inverse = np.unique(np.concatenate([a, b]), return_inverse=True)
    cols = np.ascontiguousarray(fd[:, used].T)
    got = M._batch_joint_entropy_columns(cols, inverse[:k_pairs], inverse[k_pairs:], nb[a], nb[b])
    assert np.array_equal(got, expected)
    assert np.array_equal(M.joint_entropies_of_pairs(fd, a, b, nb), expected)


def test_the_gate_picks_the_contiguous_route_only_where_measured(monkeypatch):
    """Few pairs on a narrow matrix stay strided; many pairs, or a wide matrix with enough pairs, take the copy."""
    calls = []
    real = M._batch_joint_entropy_columns
    monkeypatch.setattr(M, "_batch_joint_entropy_columns", lambda *a: calls.append(1) or real(*a))
    fd, a, b, nb = _case(k_cols=40, k_pairs=30)
    M.joint_entropies_of_pairs(fd, a, b, nb)
    assert not calls
    fd, a, b, nb = _case(k_cols=40, k_pairs=160)
    M.joint_entropies_of_pairs(fd, a, b, nb)
    assert len(calls) == 1
    fd, a, b, nb = _case(k_cols=210, k_pairs=40)
    M.joint_entropies_of_pairs(fd, a, b, nb)
    assert len(calls) == 2
