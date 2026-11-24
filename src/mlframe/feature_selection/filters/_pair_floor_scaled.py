"""The order-2 maxT floor of a large pool from a row subsample, with the scaling error measured on the data instead of assumed.

The mm-debiased floor (``pooled_pair_permutation_null_joint_mi_floor(..., mm_debias=True)``) is the 0.95 quantile, over K label shuffles, of the largest Miller-Madow-debiased joint MI of the pool. It takes every row (the
caller debiases the observed pair MI on the full rows, so the floor must be on the same scale) and costs ``pairs x K x rows`` counter updates: 107 s of a 400 s fit at 100000 x 1000. Under the null the debiased plug-in MI of a
pair is ``(chi2 - df) / (2 n)`` to first order, so the floor of ``n_sub`` rows times ``n_sub / n`` estimates the floor of ``n`` rows. That holds while the cells of the pair tables hold enough rows for the chi-square
approximation and is too high (the floor is overstated 1.2 to 1.9 times at 32 bins and 100000 rows, ``floor_scaling_experiment.py``) where they do not, which would reject weak real pairs.

Instead of a fixed rule on rows per cell the error is measured on this very data. A random block of about 0.5 percent of the pairs gets BOTH floors, the exact one on all rows and the scaled one from the subsample, with 40 shuffles
each, in two independent halves. ``rho`` = exact / scaled of the block:

* ``rho`` inside ``[ACCEPT_LOW, ACCEPT_HIGH]``: the scaling is accepted as it is;
* ``rho`` outside it but inside ``[CORRECT_LOW, CORRECT_HIGH]`` and the two halves agree: the scaled floor is multiplied by ``rho``;
* otherwise (the halves disagree, ``rho`` is far from 1, a floor is zero, the pool or the rows are small, the subsample would not be much smaller than the data): ``None`` and the caller computes the exact floor.

The subsample is ``n_sub = max(MIN_SUB_ROWS, ROWS_PER_CELL x cells of a typical pair table)`` rows spread evenly over the data (not the head of it: the rows may be ordered), and it must be at most ``MAX_SUB_SHARE`` of the rows.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: The scaling is not tried below these sizes. Measured (GTX 1050 Ti, 16 bins, 4 classes): the one-launch device kernel costs about ``1.3 us + 0.023 us x rows / 1000`` per (pair, shuffle), and the calibration block runs on the
#: host kernel, 8 to 10 times slower per update than the device one. At 100000 rows and 500000 pairs the subsample of 20000 rows costs 22 s against 48.7 s for every row, and the block 11 s: 1.1 to 1.3 times faster in all (not worth an
#: approximation); at 400000 rows and 45000 pairs the scaled route is as slow as the exact one (15 s, the block costs what the subsample saves). From 250000 rows and 200000 pairs the exact floor takes minutes and the subsample route
#: about a third of that (modelled 100 s -> 35 s at 300000 rows, 300 s -> 56 s at 1000000).
MIN_ROWS = 250_000
MIN_PAIRS = 200_000
#: Rows of the subsample: at least this many, and this many per cell of a typical pair table (the chi-square regime starts around 12 rows per cell).
MIN_SUB_ROWS = 20_000
ROWS_PER_CELL = 12
#: The subsample may be at most this share of the rows (a smaller saving is not worth the calibration and the approximation).
MAX_SUB_SHARE = 0.2
#: The calibration block: about 0.5 percent of the pairs, at least and at most this many, and the number of shuffles of its two floors (more than the pool's K: the ratio of two quantiles of K maxima is noisy).
BLOCK_MIN_PAIRS = 1_000
BLOCK_MAX_PAIRS = 3_000
BLOCK_SHARE = 0.005
BLOCK_SHUFFLES = 40
#: The calibration block may cost at most this share of the exact floor (it runs on the host kernel: ``CPU_BLOCK_PENALTY`` times slower per update than the device one).
MAX_BLOCK_COST = 0.15
CPU_BLOCK_PENALTY = 8.0
#: The ratio exact / scaled is accepted as it is inside the first window, corrected by itself inside the second, and the exact floor is taken outside it.
ACCEPT_LOW, ACCEPT_HIGH = 0.88, 1.12
CORRECT_LOW, CORRECT_HIGH = 0.5, 2.0
#: The ratios of the two halves of the block may differ by this share (the noise of one ratio is about 7 percent).
HALVES_AGREE = 0.2


def scaling_enabled() -> bool:
    """On unless ``MLFRAME_FE_PAIR_FLOOR_SCALED`` is ``0``."""
    return os.environ.get("MLFRAME_FE_PAIR_FLOOR_SCALED", "1").strip() != "0"


def _ratio(
    exact_floor: Callable[..., float], data: np.ndarray, sub: np.ndarray, y: np.ndarray, y_sub: np.ndarray, fy: np.ndarray, fy_sub: np.ndarray, nb: np.ndarray, pa: np.ndarray, pb: np.ndarray, quantile: float, seed: int
) -> Optional[float]:
    """exact floor / scaled floor of the pairs ``(pa, pb)`` on the rows of ``data`` and of its subsample ``sub``, or ``None`` when either is not positive."""
    n, n_sub = int(data.shape[0]), int(sub.shape[0])
    exact = exact_floor(data, nb, pa, pb, y, fy, n_permutations=BLOCK_SHUFFLES, quantile=quantile, random_seed=seed, mm_debias=True, scaled=False)
    scaled = exact_floor(sub, nb, pa, pb, y_sub, fy_sub, n_permutations=BLOCK_SHUFFLES, quantile=quantile, random_seed=seed, mm_debias=True, scaled=False) * (n_sub / n)
    if not (exact > 0.0 and scaled > 0.0):
        return None
    return float(exact / scaled)


def scaled_pair_floor(
    exact_floor: Callable[..., float],
    factors_data: np.ndarray,
    nbins: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    *,
    n_permutations: int,
    quantile: float,
    random_seed: Any,
    device_only: bool = False,
    k_joint_of_pairs: "Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None" = None,
) -> "tuple[Optional[float], dict]":
    """``(floor, info)``: the mm-debiased floor of the pool from a row subsample checked against the exact floor of a pair block, or ``(None, info)`` when the exact floor has to be computed.

    ``exact_floor`` is the exact function (called with ``scaled=False``); ``k_joint_of_pairs(data, pa, pb, nbins)`` counts the occupied joint cells of pairs. ``info['why']`` names the reason of a ``None``, and
    ``info`` carries ``n_sub``, ``rho`` and ``factor`` when the calibration ran.
    """
    info: dict = {}
    n, n_pairs = int(factors_data.shape[0]), int(pair_a.shape[0])
    k_y = int(np.asarray(freqs_y).shape[0])
    if not scaling_enabled():
        return None, {"why": "disabled"}
    if n < MIN_ROWS or n_pairs < MIN_PAIRS:
        return None, {"why": "small pool or few rows"}
    seed = int(random_seed) if random_seed is not None else 0
    rng = np.random.default_rng(seed + 7919)
    m = int(min(BLOCK_MAX_PAIRS, max(BLOCK_MIN_PAIRS, round(BLOCK_SHARE * n_pairs))))
    block_cost = CPU_BLOCK_PENALTY * (BLOCK_SHUFFLES * m) / (float(n_permutations) * n_pairs)
    if block_cost > MAX_BLOCK_COST:
        return None, {"why": "the calibration block costs too much against the pool (%d of %d pairs)" % (m, n_pairs)}
    block = np.sort(rng.choice(n_pairs, size=m, replace=False))
    pa, pb = np.ascontiguousarray(pair_a, dtype=np.int64), np.ascontiguousarray(pair_b, dtype=np.int64)
    nb = np.ascontiguousarray(nbins)
    if k_joint_of_pairs is None:
        return None, {"why": "no cell counter"}
    cells = float(np.median(k_joint_of_pairs(factors_data, pa[block], pb[block], nb))) * k_y
    n_sub = int(max(MIN_SUB_ROWS, ROWS_PER_CELL * cells))
    info["n_sub"] = n_sub
    if n_sub > MAX_SUB_SHARE * n:
        info["why"] = "the subsample would not be much smaller than the data (%d of %d rows)" % (n_sub, n)
        return None, info
    rows = np.linspace(0, n - 1, n_sub).astype(np.int64)
    sub = np.ascontiguousarray(factors_data[rows])
    y = np.asarray(classes_y)
    y_sub = np.ascontiguousarray(y[rows])
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)  # Intentional float64: statistic vector
    fy_sub = np.bincount(y_sub.astype(np.int64), minlength=k_y).astype(np.float64) / n_sub  # Intentional float64: statistic vector
    cal_seed = seed + 104729
    halves = [block[0::2], block[1::2]]
    rhos = [_ratio(exact_floor, factors_data, sub, y, y_sub, fy, fy_sub, nb, pa[h], pb[h], quantile, cal_seed) for h in halves]
    info["rho_halves"] = rhos
    if any(r is None for r in rhos):
        info["why"] = "a floor of the calibration block is not positive"
        return None, info
    rho = float(np.sqrt(rhos[0] * rhos[1]))
    info["rho"] = rho
    if abs(rhos[0] / rhos[1] - 1.0) > HALVES_AGREE:
        info["why"] = "the two halves of the calibration block disagree (%.2f and %.2f)" % (rhos[0], rhos[1])
        return None, info
    if ACCEPT_LOW <= rho <= ACCEPT_HIGH:
        factor = 1.0
    elif CORRECT_LOW <= rho <= CORRECT_HIGH:
        factor = rho
    else:
        info["why"] = "the scaling error is too large to correct (rho %.2f)" % rho
        return None, info
    info["factor"] = factor
    # the subsample floor of the whole pool, with the same shuffles count as the exact one
    pool_floor = exact_floor(sub, nb, pa, pb, y_sub, fy_sub, n_permutations=int(n_permutations), quantile=quantile, random_seed=random_seed, mm_debias=True, device_only=device_only, scaled=False)
    return float(pool_floor * (n_sub / n) * factor), info
