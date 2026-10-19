"""The one-launch device kernel for the pair maxT floor reproduces the CPU permutation-batched kernel and declines what it cannot hold."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def _pool(n=3000, p=30, k=8, ky=5, bins=7, seed=0):
    """A code matrix, all its pairs, a target and K shuffles of it, in the types the floor passes around."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, bins, (n, p)).astype(np.int8)
    ia, ib = np.triu_indices(p, 1)
    y = rng.integers(0, ky, n).astype(np.int64)
    cur = y.copy()
    perms = np.empty((k, n), dtype=np.int64)
    for i in range(k):
        rng.shuffle(cur)
        perms[i] = cur
    return data, ia.astype(np.int64), ib.astype(np.int64), np.full(p, bins, dtype=np.int64), perms, np.bincount(y, minlength=ky).astype(np.float64) / n


@pytest.mark.parametrize("n,p,ky,bins", [(3000, 30, 5, 7), (2000, 20, 2, 2), (1500, 12, 4, 20)])
def test_matrix_matches_the_cpu_kernel(n, p, ky, bins):
    """The ``(K, n_pairs)`` MI matrix of the device kernel agrees with the CPU kernel to the last bits of a double."""
    from mlframe.feature_selection.filters._pair_mi_perm_cuda import pair_mi_perm_matrix_cuda
    from mlframe.feature_selection.filters.info_theory import batch_pair_mi_perm_batched
    from mlframe.feature_selection.filters.info_theory.shared import joint_cardinality_cap

    data, pa, pb, nb, perms, fy = _pool(n=n, p=p, ky=ky, bins=bins)
    cap = joint_cardinality_cap()
    expected = batch_pair_mi_perm_batched(data, pa, pb, nb, perms, fy, cap)
    got = pair_mi_perm_matrix_cuda(data, pa, pb, nb, perms, fy, cap)
    assert got is not None and got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)


def test_pairs_above_the_joint_cardinality_cap_are_zero_as_on_the_cpu():
    """A pair whose joint cardinality exceeds the cap gets zeros for every shuffle, like the CPU kernel."""
    from mlframe.feature_selection.filters._pair_mi_perm_cuda import pair_mi_perm_matrix_cuda
    from mlframe.feature_selection.filters.info_theory import batch_pair_mi_perm_batched

    data, pa, pb, nb, perms, fy = _pool(n=1500, p=8, ky=4, bins=6)
    cap = 100  # 6 * 6 * 4 = 144 > 100: every pair is over the cap
    expected = batch_pair_mi_perm_batched(data, pa, pb, nb, perms, fy, cap)
    got = pair_mi_perm_matrix_cuda(data, pa, pb, nb, perms, fy, cap)
    assert got is not None
    assert float(np.abs(expected).max()) == 0.0 and float(np.abs(got).max()) == 0.0


def test_codes_beyond_a_byte_or_beyond_nbins_make_the_kernel_decline():
    """Codes that do not fit a byte, or reach past a column's ``nbins``, would index outside the histogram, so the kernel declines."""
    from mlframe.feature_selection.filters._pair_mi_perm_cuda import pair_mi_perm_matrix_cuda
    from mlframe.feature_selection.filters.info_theory.shared import joint_cardinality_cap

    data, pa, pb, nb, perms, fy = _pool(n=1000, p=6, ky=3, bins=5)
    cap = joint_cardinality_cap()
    wide = data.astype(np.int32)
    wide[0, 0] = 300
    assert pair_mi_perm_matrix_cuda(wide, pa, pb, nb, perms, fy, cap) is None
    over = data.copy()
    over[0, 0] = 5  # nbins is 5, so codes must stay below it
    assert pair_mi_perm_matrix_cuda(over, pa, pb, nb, perms, fy, cap) is None


def test_floor_is_the_same_through_both_backends_and_device_only_declines_on_the_cpu_choice(monkeypatch):
    """The floor read from the device matrix equals the floor read from the CPU matrix, and ``device_only`` raises when the host prefers the CPU kernel."""
    from mlframe.feature_selection.filters import _pair_mi_perm_cuda as mod
    from mlframe.feature_selection.filters._permutation_null import pooled_pair_permutation_null_joint_mi_floor

    data, pa, pb, nb, perms, fy = _pool(n=4000, p=40, ky=5, bins=8)
    y = perms[0]
    kw = dict(factors_data=data, nbins=nb, pair_a=pa, pair_b=pb, classes_y=y, freqs_y=fy, n_permutations=12, random_seed=3)
    monkeypatch.setattr(mod, "floor_backend_choice", lambda n, n_pairs: "cuda")
    on_device = pooled_pair_permutation_null_joint_mi_floor(**kw)
    monkeypatch.setattr(mod, "floor_backend_choice", lambda n, n_pairs: "cpu")
    on_cpu = pooled_pair_permutation_null_joint_mi_floor(**kw)
    assert on_device == pytest.approx(on_cpu, abs=1e-12)
    with pytest.raises(mod.DeviceKernelDeclinedError):
        pooled_pair_permutation_null_joint_mi_floor(device_only=True, **kw)
