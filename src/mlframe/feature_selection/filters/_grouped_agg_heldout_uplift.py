"""Held-out marginal redundancy test of a grouped aggregate against its source (the flagged alternative of the uplift gate).

The aggregate of a source that is only a noisy copy of its group key carries information the source lacks, and the conditional form of the gate rejects it (``cmi - MI(source)``, reviewed in
reporting-audit and MR11-54/86). This form compares the aggregate with the BEST partition of its source into as many intervals as the aggregate has levels, the partition chosen on one half of
the rows and the comparison made on the other half, so the partition's own optimism does not count as a gain. The test statistic is the difference of the two Miller-Madow-corrected plug-in MIs
divided by its paired standard error. It is opt-in (``score_grouped_agg_by_cmi_uplift(..., heldout_uplift=True)``); the default keeps the conditional gate.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ._fe_gain_stats import plugin_mi_of_codes, pointwise_mi

_FINE_BINS = 1024  # fine equal-frequency bins of the source the partition is cut from
_MAX_LEVELS = 64  # cap of the intervals of the partition (the dynamic programme costs levels x fine bins squared x classes)


@njit(cache=True, nogil=True)
def _best_interval_cuts(cnt: np.ndarray, n_levels: int) -> np.ndarray:
    """Cut points (fine-bin indices, ascending) of the partition of the ordered fine bins into ``n_levels`` contiguous intervals that maximises the plug-in MI with the classes.

    ``cnt`` is ``(fine bins, classes)``. With the class totals fixed the MI is a sum over intervals of ``sum_c m_c / n * log(m_c n / (m n_c))``, so the dynamic programme over interval ends is exact.
    """
    f, ky = cnt.shape
    pre = np.zeros((f + 1, ky), dtype=np.int64)
    for i in range(f):
        for c in range(ky):
            pre[i + 1, c] = pre[i, c] + cnt[i, c]
    n = float(pre[f].sum())
    levels = min(n_levels, f)
    neg = -1.0e300
    best = np.full((levels + 1, f + 1), neg)
    arg = np.zeros((levels + 1, f + 1), dtype=np.int64)
    best[0, 0] = 0.0
    for lv in range(1, levels + 1):
        for j in range(lv, f + 1):
            top = neg
            top_i = lv - 1
            for i in range(lv - 1, j):
                if best[lv - 1, i] <= neg:
                    continue
                m_all = 0.0
                for c in range(ky):
                    m_all += pre[j, c] - pre[i, c]
                v = best[lv - 1, i]
                if m_all > 0.0:
                    for c in range(ky):
                        m = float(pre[j, c] - pre[i, c])
                        if m > 0.0:
                            v += m / n * np.log(m * n / (m_all * float(pre[f, c])))
                if v > top:
                    top = v
                    top_i = i
            best[lv, j] = top
            arg[lv, j] = top_i
    cuts = np.zeros(max(levels - 1, 0), dtype=np.int64)
    j = f
    for lv in range(levels, 1, -1):
        j = arg[lv, j]
        cuts[lv - 2] = j
    return cuts


def _mm_mi(codes: np.ndarray, k: int, y: np.ndarray, ky: int) -> float:
    """Plug-in MI with the Miller-Madow bias removed, ``(occupied codes - 1)(occupied classes - 1) / 2n``."""
    n = max(1, int(codes.shape[0]))
    kx = int(np.count_nonzero(np.bincount(codes, minlength=int(k))))
    kc = int(np.count_nonzero(np.bincount(y, minlength=int(ky))))
    return plugin_mi_of_codes(codes, int(k), y, int(ky)) - float(max(kx - 1, 0) * max(kc - 1, 0)) / (2.0 * n)


def heldout_marginal_gain(agg: np.ndarray, source: np.ndarray, y_bin: np.ndarray, fit_rows: np.ndarray, eval_rows: np.ndarray) -> tuple:
    """``(gain, z)`` of the aggregate over the best same-cardinality partition of its source, the partition fitted on ``fit_rows`` and both sides measured on ``eval_rows``.

    ``gain`` is ``MI_mm(agg) - MI_mm(partition)`` on the evaluation rows and ``z`` its value divided by the standard error of the paired per-row difference of pointwise MI. Returns
    ``(nan, nan)`` when the aggregate has fewer than two levels or a side is empty.
    """
    from ._mi_greedy_cmi_fe import quantile_bin_columns

    agg_f = np.asarray(agg, dtype=np.float64)  # Intentional float64: statistic column
    lv = min(int(np.unique(agg_f[np.isfinite(agg_f)]).size), _MAX_LEVELS)
    if lv < 2 or fit_rows.size == 0 or eval_rows.size == 0:
        return float("nan"), float("nan")
    ky = int(y_bin.max()) + 1
    fine = quantile_bin_columns(np.asarray(source, dtype=np.float64)[:, None], _FINE_BINS)[0]  # Intentional float64: source bins
    cnt = np.zeros((int(fine.max()) + 1, ky), dtype=np.int64)
    np.add.at(cnt, (fine[fit_rows], y_bin[fit_rows]), 1)
    cuts = _best_interval_cuts(cnt, lv)
    # interval of each fine bin, then of each evaluation row through its fine bin
    part_of_fine = np.searchsorted(cuts, np.arange(cnt.shape[0]), side="right").astype(np.int64)
    part_ev = part_of_fine[fine[eval_rows]]
    a = quantile_bin_columns(agg_f[eval_rows][:, None], lv)[0]
    yb = y_bin[eval_rows].astype(np.int64)
    ka, kb = int(a.max()) + 1, int(part_ev.max()) + 1
    gain = _mm_mi(a, ka, yb, ky) - _mm_mi(part_ev, kb, yb, ky)
    d = pointwise_mi(a, ka, yb, ky) - pointwise_mi(part_ev, kb, yb, ky)
    se = float(d.std() / np.sqrt(len(d))) if len(d) > 1 else 0.0
    # a paired difference with no spread (identical per-row values, or a degenerate table) carries no evidence either way: z is 0, not infinite
    return gain, (gain / se if se > 1e-12 else 0.0)
