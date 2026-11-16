"""Within-stratum permutation null of the stratified CMI drawn as contingency tables, without touching the rows.

``_cmi_strata_njit`` draws a null value by shuffling x inside every stratum of ``s`` (``(n_perm, n)`` source positions) and counting an ``(strata, x levels, y levels)`` table from the
shuffled rows: two passes over the n rows with random gathers per permutation. But the statistic reads only the table, and the table of ``x`` permuted uniformly inside a stratum, with ``y`` fixed,
is a random contingency table with the stratum's own margins (the x level counts and the y level counts of that stratum) drawn from the multivariate hypergeometric distribution: the same
distribution the shuffles sample. Here each stratum's table is drawn directly, row of the table by row, with one hypergeometric draw per cell (an inversion started from the mode, so a few
dozen steps), then the same ``_cmi_from_counts`` reads the CMI. A permutation costs O(strata x x-levels x y-levels) instead of O(n), and the margins are counted once for all permutations.

The draw of permutation ``p`` is a function of ``seed + p`` alone (its own xorshift stream), so any chunking of the null is the same data and the result does not depend on the thread schedule. The
values are NOT the numbers the shuffle path draws (another random stream) but the same null distribution; ``MLFRAME_TRIPLET_NULL_SHUFFLE=1`` keeps the shuffle path.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._numba_utils import _cmi_from_counts
from ._numba_utils_shuffle import next_state, seed_state


@njit(cache=True, nogil=True)
def _uniform(x):
    """``(u, next_state)``: a uniform in [0, 1) from the 64-bit xorshift state (the multiply-shift output of ``bounded``, 53 bits)."""
    x = next_state(x)
    r = (x * np.uint64(2685821657736338717)) >> np.uint64(11)
    return float(r) * (1.0 / 9007199254740992.0), x


@njit(cache=True, nogil=True)
def _hypergeom(x, good, bad, draws, lf):
    """``(k, next_state)``: one draw of the number of ``good`` items among ``draws`` drawn without replacement from ``good + bad``, by inversion from the mode.

    ``lf`` is the table of log factorials. The probability mass is enumerated mode first, then alternately one step up and one step down; the draw is the first value at which the running mass
    passes the uniform, which samples the exact distribution for any fixed enumeration order."""
    total = good + bad
    lo = max(0, draws - bad)
    hi = min(draws, good)
    if lo >= hi:
        return lo, x
    u, x = _uniform(x)
    mode = ((draws + 1) * (good + 1)) // (total + 2)
    mode = min(max(mode, lo), hi)
    logp = lf[good] - lf[mode] - lf[good - mode] + lf[bad] - lf[draws - mode] - lf[bad - draws + mode] - (lf[total] - lf[draws] - lf[total - draws])
    pm = math.exp(logp)
    cum = pm
    if u < cum:
        return mode, x
    up = mode
    down = mode
    pu = pm
    pd = pm
    while up < hi or down > lo:
        if up < hi:
            pu = pu * (good - up) * (draws - up) / ((up + 1) * (bad - draws + up + 1))
            up += 1
            cum += pu
            if u < cum:
                return up, x
        if down > lo:
            pd = pd * down * (bad - draws + down) / ((good - down + 1) * (draws - down + 1))
            down -= 1
            cum += pd
            if u < cum:
                return down, x
    return up, x


@njit(cache=True, nogil=True)
def _draw_table(x, row_m, col_m, lf, out):
    """Fill ``out`` (x levels, y levels) with a random contingency table of margins ``row_m`` (x level counts) and ``col_m`` (y level counts) of one stratum; returns the next state."""
    n_x = row_m.shape[0]
    n_y = col_m.shape[0]
    rem = col_m.copy()
    for i in range(n_x - 1):
        draws = row_m[i]
        pop = 0
        for j in range(n_y):
            pop += rem[j]
        for j in range(n_y - 1):
            good = rem[j]
            k, x = _hypergeom(x, good, pop - good, draws, lf)
            out[i, j] = k
            draws -= k
            rem[j] -= k
            pop -= good
        out[i, n_y - 1] = draws
        rem[n_y - 1] -= draws
    for j in range(n_y):
        out[n_x - 1, j] = rem[j]
    return x


@njit(cache=True, nogil=True)
def _stratum_margins(x_codes, y_codes, s_codes, n_s, n_x, n_y):
    """``(row_m, col_m)``: per stratum the x level counts ``(n_s, n_x)`` and the y level counts ``(n_s, n_y)``."""
    row_m = np.zeros((n_s, n_x), dtype=np.int64)
    col_m = np.zeros((n_s, n_y), dtype=np.int64)
    for i in range(x_codes.shape[0]):
        row_m[s_codes[i], x_codes[i]] += 1
        col_m[s_codes[i], y_codes[i]] += 1
    return row_m, col_m


@njit(cache=True, nogil=True)
def _occupied_terms(counts):
    """``(k_xyz - k_xz - k_yz + k_z)``: the occupied-cell counts of the ``(strata, x levels, y levels)`` table that make the Miller-Madow bias of the plug-in CMI (the plug-in is high by this over ``2 n``)."""
    n_s, n_x, n_y = counts.shape
    k_xyz = 0
    k_xz = 0
    k_yz = 0
    k_z = 0
    for sv in range(n_s):
        total = 0
        for a in range(n_x):
            row = 0
            for b in range(n_y):
                c = counts[sv, a, b]
                if c > 0:
                    k_xyz += 1
                row += c
            if row > 0:
                k_xz += 1
            total += row
        if total > 0:
            k_z += 1
        for b in range(n_y):
            col = 0
            for a in range(n_x):
                col += counts[sv, a, b]
            if col > 0:
                k_yz += 1
    return k_xyz - k_xz - k_yz + k_z


@njit(cache=True, parallel=True, nogil=True)
def _null_njit(row_m, col_m, n, n_perm, seed, min_rows, lf, debias):
    """The stratified CMI of ``n_perm`` random within-stratum permutation tables (permutation ``p`` seeded ``seed + p``); with ``debias`` the Miller-Madow corrected CMI clamped at zero."""
    n_s, n_x = row_m.shape
    n_y = col_m.shape[1]
    out = np.zeros(n_perm, dtype=np.float64)  # Intentional float64: statistic vector (null values), exact permutation ties
    for p in prange(n_perm):
        counts = np.zeros((n_s, n_x, n_y), dtype=np.int64)
        x = seed_state(seed + p)
        tab = np.zeros((n_x, n_y), dtype=np.int64)
        for sv in range(n_s):
            tab[:, :] = 0
            m = 0
            for a in range(n_x):
                m += row_m[sv, a]
            if m == 0:
                continue
            x = _draw_table(x, row_m[sv], col_m[sv], lf, tab)
            counts[sv, :, :] = tab
        v = _cmi_from_counts(counts, n, min_rows)
        if debias:
            v = max(v - _occupied_terms(counts) / (2.0 * n), 0.0)
        out[p] = v
    return out


def _log_factorials(n: int) -> np.ndarray:
    """``log(k!)`` for ``k = 0 .. n``."""
    lf = np.zeros(n + 1, dtype=np.float64)  # Intentional float64: log-factorial table, exact to the last bits
    np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)), out=lf[1:])  # Intentional float64: log-factorial table
    return lf


class StratifiedNull:
    """The margins of ``(x, y, s)`` (dense non-negative codes) and the log-factorial table, built once (``debias``: the Miller-Madow corrected CMI of the device scorers instead of the plug-in); :meth:`chunk` draws permutations ``first .. first + count - 1`` of the null seeded by ``seed``."""

    def __init__(self, x_codes: np.ndarray, y_codes: np.ndarray, s_codes: np.ndarray, min_rows: int, debias: bool = False) -> None:
        self.n = int(x_codes.shape[0])
        self.min_rows = int(min_rows)
        self.debias = bool(debias)
        self.row_m, self.col_m = _stratum_margins(x_codes, y_codes, s_codes, int(s_codes.max()) + 1, int(x_codes.max()) + 1, int(y_codes.max()) + 1)
        self.lf = _log_factorials(self.n)

    def chunk(self, first: int, count: int, seed: int) -> np.ndarray:
        """The stratified CMI of permutations ``first .. first + count - 1`` (seeded ``seed + p``)."""
        from mlframe._numba_parallel_guard import parallel_kernel_entry

        with parallel_kernel_entry():
            out: np.ndarray = _null_njit(self.row_m, self.col_m, self.n, int(count), int(seed) + int(first), self.min_rows, self.lf, self.debias)
        return out


__all__ = ["StratifiedNull"]
