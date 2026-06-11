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
        xc = _disc(a ** 2 / b, nb)            # heavy-tailed engineered col
        yc = _disc(rng.normal(size=n), nb)    # independent target -> true MI 0
        inv = 1.0 / n
        xb = np.bincount(xc).astype(float); px = xb[xb > 0] * inv
        yb = np.bincount(yc).astype(float); py = yb[yb > 0] * inv
        hx = -(px * np.log(px)).sum(); hy = -(py * np.log(py)).sum()
        j = xc * nb + yc
        jc = np.bincount(j).astype(float); pj = jc[jc > 0] * inv
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
    cols = [
        _disc(rng.normal(size=n) ** 2 / rng.normal(size=n), nb) for _ in range(8)
    ]
    y = _disc(rng.normal(size=n), nb)
    data = np.column_stack(cols + [y]).astype(np.int64)
    nbins = np.array([nb] * 9)
    f = pooled_permutation_null_gain_floor(
        data, nbins, np.arange(8), 8, n_permutations=25, random_seed=0
    )
    assert np.isfinite(f) and f >= 0.0
