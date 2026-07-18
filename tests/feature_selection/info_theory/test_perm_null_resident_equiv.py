"""iter16: the resident-GPU twin of the maxT permutation-null pooled gain floor is selection-equivalent
to the njit kernel it replaces.

The resident cupy path (``_permutation_null_resident.pooled_gain_floor_perms_cupy``) and the host njit
kernel (``_permutation_null._pooled_gain_floor_perms_njit``) compute the per-shuffle MAX corrected marginal
MI from the SAME integer contingency tables; they differ only in the FP reduction order of the per-cell
``-p*log(p)`` accumulation. This pins their per-shuffle outputs to fp64 round-off (and the 0.95-quantile
FLOOR -- the value the caller actually uses -- to full agreement), so the downstream gain-floor selection
is unchanged whichever path the KTC crossover picks.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._permutation_null import _pooled_gain_floor_perms_njit


def _make_inputs(n, ncand, nperm, nbins_x=16, nbins_y=10, seed=0):
    """Make inputs."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, nbins_x, size=(ncand, n)).astype(np.int64)
    y = rng.integers(0, nbins_y, size=n).astype(np.int32)
    scaled_flat = np.concatenate([(x[j] * nbins_y).astype(np.int32) for j in range(ncand)])
    offsets = np.arange(ncand + 1, dtype=np.int64) * n
    joint_card = np.full(ncand, nbins_x * nbins_y, dtype=np.int64)
    inv_n = 1.0 / n
    yc = np.bincount(y, minlength=nbins_y).astype(np.float64)
    py = yc[yc > 0] * inv_n
    h_y = float(-(py * np.log(py)).sum())
    h_x = np.empty(ncand, dtype=np.float64)
    for j in range(ncand):
        xc = np.bincount(x[j], minlength=nbins_x).astype(np.float64)
        px = xc[xc > 0] * inv_n
        h_x[j] = float(-(px * np.log(px)).sum())
    mm_bias = np.full(ncand, (nbins_x - 1) * (nbins_y - 1) / (2.0 * n), dtype=np.float64)
    yp = y.copy()
    y_perms = np.empty((nperm, n), dtype=np.int32)
    for k in range(nperm):
        rng.shuffle(yp)
        y_perms[k] = yp
    return (scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n)


@pytest.mark.parametrize("n,ncand,nperm", [(5000, 8, 25), (8000, 24, 40)])
def test_resident_permnull_floor_selection_equivalent(n, ncand, nperm):
    """Resident permnull floor selection equivalent."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters._permutation_null_resident import pooled_gain_floor_perms_cupy

    args = _make_inputs(n, ncand, nperm)
    m_cpu = _pooled_gain_floor_perms_njit(*args)
    m_gpu = pooled_gain_floor_perms_cupy(*args)
    # per-shuffle maxes agree to fp64 round-off (only the entropy reduction ORDER differs)
    assert np.max(np.abs(m_cpu - m_gpu)) < 1e-12
    # the FLOOR the caller uses (0.95 quantile) is identical to many digits -> selection unchanged
    assert abs(np.quantile(m_cpu, 0.95) - np.quantile(m_gpu, 0.95)) < 1e-12


def test_gpu_device_shuffle_gen_valid_permutations():
    """The device argsort-keys shuffle generator (KTC-gated for big-VRAM hosts) yields, per row, a TRUE
    permutation of the target codes (identical sorted multiset) and distinct rows -- a valid uniform null
    born on the device and fed resident to the floor with no host gen / no (nperm,n) H2D."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._permutation_null_resident import gen_target_shuffles_cupy

    y = np.random.default_rng(0).integers(0, 7, size=8000).astype(np.int32)
    out = gen_target_shuffles_cupy(y, 32, np.int32, 42)
    if out is None:
        pytest.skip("device shuffle-gen unavailable on this host (cupy/VRAM)")
    h = cp.asnumpy(out)
    assert h.shape == (32, y.shape[0])
    ys = np.sort(y)
    assert all(np.array_equal(np.sort(h[k]), ys) for k in range(h.shape[0])), "each row must be a permutation of y"
    assert not np.array_equal(h[0], h[1]), "distinct permutations across rows"
