"""Identity gate for the prange-parallel order-1 maxT permutation-null kernel.

``_pooled_gain_floor_perms_njit`` parallelises its K-shuffle loop over ``prange``.
The result must be BIT-IDENTICAL to the serial scan: each shuffle ``k`` writes only
``maxes[k]`` from read-only shared inputs with a private ``counts`` scratch, so no FP
reduction-order changes within a shuffle. This pins max|diff|=0.0 and an exactly-equal
0.95-quantile floor against a serial reference re-implementation, so a future "drop the
parallel flag" or a scratch-sharing regression is caught.
"""

from __future__ import annotations

import numba
import numpy as np

from mlframe.feature_selection.filters._permutation_null import (
    _pooled_gain_floor_perms_njit,
    pooled_permutation_null_gain_floor,
)


@numba.njit(cache=True)
def _serial_reference(scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n):
    nperm = y_perms.shape[0]
    n = y_perms.shape[1]
    ncand = offsets.shape[0] - 1
    maxes = np.empty(nperm, dtype=np.float64)
    max_jc = 0
    for j in range(ncand):
        if joint_card[j] > max_jc:
            max_jc = joint_card[j]
    counts = np.empty(max_jc, dtype=np.float64)
    for k in range(nperm):
        yp = y_perms[k]
        best = 0.0
        for j in range(ncand):
            jc = joint_card[j]
            for t in range(jc):
                counts[t] = 0.0
            s0 = offsets[j]
            for i in range(n):
                counts[scaled_flat[s0 + i] + yp[i]] += 1.0
            h_xy = 0.0
            for t in range(jc):
                c = counts[t]
                if c > 0.0:
                    p = c * inv_n
                    h_xy -= p * np.log(p)
            mi = h_x[j] + h_y - h_xy - mm_bias[j]
            if mi > best:
                best = mi
        maxes[k] = best
    return maxes


def _make_inputs(n, n_cand, nperm, nbins_x=12, nbins_y=10, seed=0):
    rng = np.random.default_rng(seed)
    inv_n = 1.0 / n
    scaled, joint_card, h_x, mm_bias = [], [], [], []
    for _ in range(n_cand):
        xc = rng.integers(0, nbins_x, size=n).astype(np.int64)
        xcounts = np.bincount(xc, minlength=nbins_x).astype(np.float64)
        px = xcounts[xcounts > 0] * inv_n
        scaled.append((xc * nbins_y).astype(np.int32))
        joint_card.append(nbins_x * nbins_y)
        h_x.append(float(-(px * np.log(px)).sum()))
        mm_bias.append((nbins_x - 1) * (nbins_y - 1) / (2.0 * n))
    y_codes = rng.integers(0, nbins_y, size=n).astype(np.int32)
    ycounts = np.bincount(y_codes, minlength=nbins_y).astype(np.float64)
    py = ycounts[ycounts > 0] * inv_n
    h_y = float(-(py * np.log(py)).sum())
    y_perm = y_codes.copy()
    y_perms = np.empty((nperm, n), dtype=np.int32)
    for k in range(nperm):
        rng.shuffle(y_perm)
        y_perms[k] = y_perm
    scaled_flat = np.concatenate(scaled).astype(np.int32)
    offsets = np.arange(n_cand + 1, dtype=np.int64) * n
    return (
        scaled_flat,
        offsets,
        np.asarray(joint_card, dtype=np.int64),
        np.asarray(h_x, dtype=np.float64),
        np.asarray(mm_bias, dtype=np.float64),
        h_y,
        y_perms,
        inv_n,
    )


def test_prange_kernel_bit_identical_to_serial():
    for n, n_cand, nperm in [(2000, 40, 50), (5000, 80, 64)]:
        args = _make_inputs(n, n_cand, nperm)
        out_parallel = _pooled_gain_floor_perms_njit(*args)
        out_serial = _serial_reference(*args)
        assert np.max(np.abs(out_parallel - out_serial)) == 0.0, f"prange kernel diverged from serial at n={n} p={n_cand} K={nperm}"
        assert float(np.quantile(out_parallel, 0.95)) == float(np.quantile(out_serial, 0.95))


def test_public_floor_deterministic_and_nonnegative():
    rng = np.random.default_rng(7)
    n, p, nbins = 3000, 40, 10
    cols = [rng.integers(0, nbins, size=n).astype(np.int32) for _ in range(p)]
    y = rng.integers(0, nbins, size=n).astype(np.int32)
    factors_data = np.column_stack([*cols, y]).astype(np.int32)
    factors_nbins = np.array([nbins] * (p + 1), dtype=np.int64)
    cand = list(range(p))
    floor_a = pooled_permutation_null_gain_floor(
        factors_data,
        factors_nbins,
        cand,
        p,
        n_permutations=64,
        random_seed=123,
    )
    floor_b = pooled_permutation_null_gain_floor(
        factors_data,
        factors_nbins,
        cand,
        p,
        n_permutations=64,
        random_seed=123,
    )
    assert floor_a == floor_b  # same seed -> identical floor regardless of thread interleave
    assert floor_a >= 0.0
