"""The neighbour-count walk of the Ross k-NN MI gives the counts of the binary-search bounds it replaced."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._ksg_cd_fast import _psi_table, ksg_mi_discrete_target_columns


def _reference_mi(x: np.ndarray, d: np.ndarray, n_classes: int, n_neighbors: int) -> float:
    """Ross MI of one prepared column with ``np.searchsorted`` bounds and the same exact-distance trim, in plain numpy."""
    n = x.shape[0]
    psi = _psi_table(n)
    counts = np.bincount(d, minlength=n_classes)
    order = np.argsort(x, kind="quicksort")
    keep = counts[d[order]] > 1
    xs, ds = x[order][keep], d[order][keep]
    n_kept = xs.size
    sum_k = sum_cnt = sum_m = 0.0
    for j in range(n_kept):
        c = ds[j]
        cls = xs[ds == c]
        cnt = counts[c]
        k = min(n_neighbors, cnt - 1)
        i = int(np.flatnonzero(ds == c).tolist().index(j))
        v = xs[j]
        lo, hi, r = i - 1, i + 1, 0.0
        for _ in range(k):
            dl = v - cls[lo] if lo >= 0 else np.inf
            dh = cls[hi] - v if hi < cnt else np.inf
            if dl <= dh:
                r, lo = dl, lo - 1
            else:
                r, hi = dh, hi + 1
        lo_m = int(np.searchsorted(xs, v - r, side="left"))
        hi_m = int(np.searchsorted(xs, v + r, side="right"))
        while lo_m < hi_m and v - xs[lo_m] >= r:
            lo_m += 1
        while hi_m > lo_m and xs[hi_m - 1] - v >= r:
            hi_m -= 1
        sum_k += psi[k]
        sum_cnt += psi[cnt]
        sum_m += psi[hi_m - lo_m]
    mi = psi[n_kept] + (sum_k - sum_cnt - sum_m) / n_kept
    return max(mi, 0.0)


@pytest.mark.parametrize("kind", ["continuous", "rounded", "few_levels"])
def test_walk_matches_binary_search_bounds(kind):
    """Same MI as the searchsorted reference for continuous, heavily tied and few-level columns (the noise leaves near-ties a binary search can round onto)."""
    rng = np.random.default_rng(3)
    n = 1500
    if kind == "continuous":
        x = rng.standard_normal(n)
    elif kind == "rounded":
        x = np.round(rng.standard_normal(n) * 3)
    else:
        x = rng.integers(0, 4, n).astype(float)
    y = rng.integers(0, 3, n)
    got = ksg_mi_discrete_target_columns(x[:, None], y, n_neighbors=3, random_state=5)[0]
    # the same preparation as ksg_mi_discrete_target_columns: scale, then the broadcast Kraskov noise
    xp = x / (x.std() or 1.0)
    xp = xp + 1e-10 * max(1.0, float(np.mean(np.abs(xp)))) * np.random.RandomState(5).standard_normal(size=n)
    _, d = np.unique(y, return_inverse=True)
    want = _reference_mi(xp, d.astype(np.int64), 3, 3)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-15)
