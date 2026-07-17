"""Selection-equivalence parity for the resident-GPU order-2 pooled-max joint-MI permutation-null floor.

``_permutation_null_pair_resident.pooled_pair_permutation_null_joint_mi_floor_cupy`` is the device twin of the
CPU njit ``_permutation_null.pooled_pair_permutation_null_joint_mi_floor``. The floor is the 0.95-quantile of a
RANDOM permutation null, so the contract is SELECTION-EQUIVALENCE, not byte-identity: the device RNG stream
differs from numpy's (acceptable), but

  * the floor must be reproducible per seed,
  * the device floor must track the host floor closely enough that the gate decision ``pair_mi >= floor`` is
    IDENTICAL on a fixture (a genuine synergy pair clears both; noise pairs clear neither), and
  * degenerate pools no-op to 0.0 like the CPU floor.

Auto-skips when cupy is unavailable (CI without GPU).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._permutation_null import (
    pooled_pair_permutation_null_joint_mi_floor,
)
from mlframe.feature_selection.filters._permutation_null_pair_resident import (
    pooled_pair_permutation_null_joint_mi_floor_cupy,
)
from mlframe.feature_selection.filters.info_theory import batch_pair_mi_prange


def _make_pool(n, n_features, nbins_val, n_classes_y, seed, synergy=False):
    rng = np.random.default_rng(seed)
    fd = rng.integers(0, nbins_val, size=(n, n_features)).astype(np.int32)
    nb = np.full(n_features, nbins_val, dtype=np.int32)
    if synergy:
        y = (fd[:, 0] ^ fd[:, 1]) % n_classes_y
    else:
        y = rng.integers(0, n_classes_y, size=n)
    y = y.astype(np.int64)
    fy = np.bincount(y, minlength=n_classes_y).astype(np.float64) / n
    pairs = list(combinations(range(n_features), 2))
    pa = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    pb = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    return fd, nb, pa, pb, y, fy


@pytest.mark.parametrize("mm_debias", [False, True])
@pytest.mark.parametrize("n_classes_y", [2, 4])
def test_device_floor_tracks_host_floor(mm_debias, n_classes_y):
    """Device floor is on the same scale as the host floor (noise null) -- within a generous relative band that
    the RNG-stream difference over K=25 draws produces, far below the synergy gap the gate cares about."""
    fd, nb, pa, pb, y, fy = _make_pool(20000, 12, 8, n_classes_y, seed=7)
    host = pooled_pair_permutation_null_joint_mi_floor(
        factors_data=fd,
        nbins=nb,
        pair_a=pa,
        pair_b=pb,
        classes_y=y,
        freqs_y=fy,
        n_permutations=25,
        quantile=0.95,
        random_seed=42,
        mm_debias=mm_debias,
    )
    dev = pooled_pair_permutation_null_joint_mi_floor_cupy(
        factors_data=fd,
        pair_a=pa,
        pair_b=pb,
        nbins=nb,
        classes_y=y,
        freqs_y=fy,
        n_permutations=25,
        quantile=0.95,
        random_seed=42,
        mm_debias=mm_debias,
    )
    assert dev is not None
    assert dev > 0.0 and host > 0.0
    # Both are 0.95-quantiles of a pure-noise max-null; the device RNG draws a different stream, so allow a
    # wide relative band (the noise floor is ~1e-3 and the synergy signal is ~1.4 -> ~200x, see the gate test).
    assert abs(dev - host) / host < 0.30, f"device floor {dev:.3e} too far from host floor {host:.3e}"


def test_device_floor_reproducible_per_seed():
    """Same seed -> identical device floor (the device RNG is seeded from random_seed)."""
    fd, nb, pa, pb, y, fy = _make_pool(20000, 12, 8, 4, seed=7)
    d1 = pooled_pair_permutation_null_joint_mi_floor_cupy(
        factors_data=fd,
        pair_a=pa,
        pair_b=pb,
        nbins=nb,
        classes_y=y,
        freqs_y=fy,
        random_seed=123,
    )
    d2 = pooled_pair_permutation_null_joint_mi_floor_cupy(
        factors_data=fd,
        pair_a=pa,
        pair_b=pb,
        nbins=nb,
        classes_y=y,
        freqs_y=fy,
        random_seed=123,
    )
    assert d1 == d2


@pytest.mark.parametrize("mm_debias", [False, True])
def test_gate_decisions_identical_on_synergy_fixture(mm_debias):
    """The gate decision ``pair_mi >= floor`` is IDENTICAL host-vs-device on a fixture with one true XOR synergy
    pair: the synergy joint MI is ~200x the noise floor, so the floor's RNG-noise difference never flips the
    admit/reject decision for ANY pair."""
    fd, nb, pa, pb, y, fy = _make_pool(20000, 12, 8, 4, seed=11, synergy=True)
    obs = batch_pair_mi_prange(fd, pa, pb, nb, y, fy)
    bias = None
    if mm_debias:
        from mlframe.feature_selection.filters._permutation_null import pairwise_mm_joint_bias

        bias = pairwise_mm_joint_bias(fd, pa, pb, nb, int(fy.shape[0]))
        obs = obs - bias
    host = pooled_pair_permutation_null_joint_mi_floor(
        factors_data=fd,
        nbins=nb,
        pair_a=pa,
        pair_b=pb,
        classes_y=y,
        freqs_y=fy,
        random_seed=42,
        mm_debias=mm_debias,
    )
    dev = pooled_pair_permutation_null_joint_mi_floor_cupy(
        factors_data=fd,
        pair_a=pa,
        pair_b=pb,
        nbins=nb,
        classes_y=y,
        freqs_y=fy,
        random_seed=42,
        mm_debias=mm_debias,
        mm_bias=bias,
    )
    assert dev is not None
    gate_host = obs >= host
    gate_dev = obs >= dev
    assert np.array_equal(gate_host, gate_dev), (
        f"gate flip: host_floor={host:.4e} dev_floor={dev:.4e}; pass host={int(gate_host.sum())} dev={int(gate_dev.sum())}"
    )
    assert gate_host.sum() >= 1  # the XOR pair must clear the floor on both


@pytest.mark.parametrize(
    "n,npairs_ok",
    [(4, True), (20000, False)],
)
def test_degenerate_pools_noop(n, npairs_ok):
    """Same degenerate-pool no-op (0.0) as the CPU floor: n<8 or <2 pairs or single-class y."""
    rng = np.random.default_rng(0)
    if npairs_ok:
        # n too small -> 0.0 regardless of pairs
        fd = rng.integers(0, 8, size=(n, 4)).astype(np.int32)
        nb = np.full(4, 8, dtype=np.int32)
        pa = np.array([0, 1, 2], dtype=np.int64)
        pb = np.array([1, 2, 3], dtype=np.int64)
    else:
        # single-class y -> 0.0
        fd = rng.integers(0, 8, size=(n, 4)).astype(np.int32)
        nb = np.full(4, 8, dtype=np.int32)
        pa = np.array([0, 1, 2], dtype=np.int64)
        pb = np.array([1, 2, 3], dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    fy = np.array([1.0], dtype=np.float64)
    dev = pooled_pair_permutation_null_joint_mi_floor_cupy(
        factors_data=fd,
        pair_a=pa,
        pair_b=pb,
        nbins=nb,
        classes_y=y,
        freqs_y=fy,
    )
    assert dev == 0.0
