"""Idea-#9 (backlog #4): the order-1 maxT permutation-null floor uses the
EFFECTIVE OCCUPIED bin count (not nominal cardinality) in its Miller-Madow bias.

Heavy-tailed engineered columns (a**2/b) bin to far fewer occupied cells than
their nominal cardinality. The nominal-K MM term ``(nb_x-1)(nb_y-1)/2n`` charges
bias for empty cells that contribute no plug-in inflation, OVER-correcting the
estimator. Occupied-K is the statistically correct Miller-Madow and tracks the
true (null=0) MI several-fold tighter on such columns.
"""

import numpy as np

from mlframe.feature_selection.filters._permutation_null import (
    pooled_permutation_null_gain_floor,
)


def _disc(v, nb):
    e = np.linspace(v.min(), v.max(), nb + 1)
    return np.clip(np.digitize(v, e[1:-1]), 0, nb - 1).astype(np.int64)


def _occupied_mm_residual(use_occupied: bool, n: int, nb: int, seeds: int = 20) -> float:
    """Mean |MM-corrected MI| of a heavy-tailed col against an INDEPENDENT target.

    True MI is 0 (independence); a well-corrected estimator returns ~0. Returns
    the mean absolute residual -- smaller is more accurate.
    """
    res = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        xc = _disc(a**2 / b, nb)  # heavy-tailed engineered col
        yc = _disc(rng.normal(size=n), nb)  # independent target -> true MI 0
        inv = 1.0 / n
        xb = np.bincount(xc).astype(float)
        px = xb[xb > 0] * inv
        yb = np.bincount(yc).astype(float)
        py = yb[yb > 0] * inv
        hx = -(px * np.log(px)).sum()
        hy = -(py * np.log(py)).sum()
        j = xc * nb + yc
        jc = np.bincount(j).astype(float)
        pj = jc[jc > 0] * inv
        hxy = -(pj * np.log(pj)).sum()
        mi_plug = hx + hy - hxy
        kx = len(px) if use_occupied else nb
        ky = len(py) if use_occupied else nb
        res.append(abs(mi_plug - (kx - 1) * (ky - 1) / (2.0 * n)))
    return float(np.mean(res))


def test_occupied_mm_more_accurate_than_nominal_on_heavy_tail():
    """Occupied-K MM residual is materially smaller than nominal-K on a**2/b."""
    for n in (2000, 5000):
        nom = _occupied_mm_residual(False, n, 16)
        occ = _occupied_mm_residual(True, n, 16)
        assert occ < nom, f"n={n}: occupied {occ:.5f} not < nominal {nom:.5f}"
        # decisive: at least 2x tighter (measured ~3.1-3.2x)
        assert occ < 0.5 * nom, f"n={n}: occupied {occ:.5f} not <= 0.5*nominal {nom:.5f}"


def test_floor_stays_nonneg_and_finite_with_occupied_mm():
    """The integrated floor is non-negative and finite (no negative-entropy leak)."""
    n, nb = 4000, 16
    rng = np.random.default_rng(1)
    cols = [_disc(rng.normal(size=n) ** 2 / rng.normal(size=n), nb) for _ in range(8)]
    y = _disc(rng.normal(size=n), nb)
    data = np.column_stack(cols + [y]).astype(np.int64)
    nbins = np.array([nb] * 9)
    f = pooled_permutation_null_gain_floor(data, nbins, np.arange(8), 8, n_permutations=25, random_seed=0)
    assert np.isfinite(f) and f >= 0.0


def _legacy_gain_floor_python(
    factors_data,
    factors_nbins,
    candidate_indices,
    y_index,
    *,
    n_permutations=25,
    quantile=0.95,
    cardinality_bias_correction=True,
    random_seed=None,
):
    """Pure-Python reference for the order-1 maxT gain floor (the pre-njit body).

    Pins :func:`pooled_permutation_null_gain_floor`'s fused-njit per-shuffle MI pass
    to the per-shuffle ``np.bincount`` loop it replaced -- the floor must stay bit-identical
    (FP reduction-order ~1e-15) so the njit fusion never moves a ``gain >= floor`` decision.
    """
    n = int(factors_data.shape[0])
    y_idx = int(y_index)
    nbins_y = int(factors_nbins[y_idx])
    inv_n = 1.0 / n
    y_codes = np.ascontiguousarray(factors_data[:, y_idx]).astype(np.int64)
    yc = np.bincount(y_codes, minlength=nbins_y).astype(np.float64)
    py = yc[yc > 0] * inv_n
    h_y = float(-(py * np.log(py)).sum())
    ky = int(py.shape[0])
    sc, jc, hx, mm = [], [], [], []
    for c in candidate_indices:
        ci = int(c)
        if ci == y_idx:
            continue
        nb = int(factors_nbins[ci])
        if nb < 2:
            continue
        xc = np.ascontiguousarray(factors_data[:, ci]).astype(np.int64)
        xx = np.bincount(xc, minlength=nb).astype(np.float64)
        px = xx[xx > 0] * inv_n
        kx = int(px.shape[0])
        sc.append(xc * nbins_y)
        jc.append(nb * nbins_y)
        hx.append(float(-(px * np.log(px)).sum()))
        mm.append(((kx - 1) * (ky - 1) / (2.0 * n)) if cardinality_bias_correction else 0.0)
    if len(sc) < 2:
        return 0.0
    rng = np.random.default_rng(random_seed)
    yp = y_codes.copy()
    maxes = np.empty(n_permutations)
    for k in range(n_permutations):
        rng.shuffle(yp)
        best = 0.0
        for j in range(len(sc)):
            b = np.bincount(sc[j] + yp, minlength=jc[j]).astype(np.float64)
            pj = b[b > 0] * inv_n
            mi = hx[j] + h_y - (-(pj * np.log(pj)).sum()) - mm[j]
            if mi > best:
                best = mi
        maxes[k] = best
    return float(np.quantile(maxes, quantile))


def test_njit_gain_floor_bit_identical_to_legacy_python_body():
    """The fused-njit floor matches the per-shuffle Python bincount loop to FP reduction-order across pool shapes/cardinalities."""
    rng = np.random.default_rng(5)
    for _ in range(6):
        n = int(rng.integers(500, 3000))
        p = int(rng.integers(3, 200))
        fd = np.empty((n, p + 1), dtype=np.int64)
        fnb = np.empty(p + 1, dtype=np.int64)
        nby = int(rng.integers(2, 12))
        fd[:, p] = rng.integers(0, nby, n)
        fnb[p] = nby
        for j in range(p):
            nb = int(rng.integers(2, 17))
            fd[:, j] = rng.integers(0, nb, n)
            fnb[j] = nb
        ci = np.arange(p)
        seed = int(rng.integers(0, 1000))
        # Pin K explicitly on BOTH paths: the reference and the njit floor must use the same permutation count for the bit-identity comparison (the function default is 200).
        ref = _legacy_gain_floor_python(fd, fnb, ci, p, n_permutations=200, random_seed=seed)
        got = pooled_permutation_null_gain_floor(fd, fnb, ci, p, n_permutations=200, random_seed=seed)
        assert abs(ref - got) <= 1e-12, f"n={n} p={p}: legacy {ref:.15g} vs njit {got:.15g}"
