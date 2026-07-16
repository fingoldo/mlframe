"""``ma_crossover_features``: MACD-style pairwise moving-average difference/sign/vote features.

Source: av_top3_rampaging_datahulk_minihack2017.md -- "difference between three day moving average and five
day moving average," "difference between five day and ten day," plus binary sign features "comparison of
3-day MA with other MAs (values 1, 0, -1)" summed into a composite crossover score. A classic technical-
analysis pattern (MACD-like): for every (short, long) window pair, the signed difference captures trend
direction/momentum, and the sign captures a discrete regime (uptrend/downtrend/flat) that's more robust to
noise than the raw difference magnitude; summing signs across many pairs gives a consensus "how many
timeframes agree on the current trend direction" score.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.feature_engineering.grouped import per_group_shift


def ma_crossover_features(
    moving_averages: Dict[int, pd.Series],
    column_prefix: str = "ma_crossover",
    *,
    group_ids: Optional[np.ndarray] = None,
    long_window_weight_power: float = 0.0,
) -> pd.DataFrame:
    """Pairwise MA differences, sign votes, a summed crossover score, and its row-over-row delta.

    Parameters
    ----------
    moving_averages
        ``{window_size: moving_average_series}`` -- precomputed rolling means (or EWMAs/Hull MAs) at
        different window sizes, all aligned to the same index. At least 2 entries required.
    column_prefix
        Prefix for emitted column names.
    group_ids
        Optional 1-D array aligned to the moving-average index. When given, rows are treated as
        multiple independent time-ordered entities (e.g. multiple instruments/tickers concatenated
        into one frame): ``vote_sum_delta`` is computed via a group-aware first difference
        (``per_group_shift``) that resets at each group boundary, instead of a naive ``diff()`` that
        would otherwise leak the last row of one entity into the first row of the next.
    long_window_weight_power
        Opt-in (default ``0.0`` -- every pair weighted equally, bit-identical to the original
        equal-weight vote). When ``> 0``, each pair's vote is weighted by ``short ** power`` (the
        pair's SHORT window size raised to this power) before summing into ``vote_sum``/
        ``vote_sum_delta``. A pair's reliability is set by its noisier, SHORT leg -- a pair like
        ``(3, 90)`` still flips on almost every single-bar wiggle in the 3-period MA even though its
        partner is long, so weighting by the long leg alone (an earlier, buggy version of this
        parameter) still let noise-anchored cross pairs dominate the vote. Weighting by the short leg
        penalizes every pair anchored to a noisy short MA regardless of its partner's length, which is
        what actually suppresses false flips; genuinely long-vs-long pairs (both legs large) end up with
        the highest weight, as intended. Typical values: ``1.0``-``2.0``.

    Returns
    -------
    pd.DataFrame
        For every ``(short, long)`` window pair (``short < long``): ``{prefix}_diff_{short}_{long}`` (signed
        difference) and ``{prefix}_vote_{short}_{long}`` (``sign(diff)`` in ``{-1, 0, 1}``), plus
        ``{prefix}_vote_sum`` (sum of all vote columns -- a consensus trend-direction score) and
        ``{prefix}_vote_sum_delta`` (row-over-row change in ``vote_sum`` -- momentum of the crossover
        consensus itself: a large positive value means many pairs are simultaneously flipping toward an
        uptrend RIGHT NOW, an accelerating regime shift that the slowly-drifting ``vote_sum`` level alone
        doesn't directly surface).
    """
    windows: Sequence[int] = sorted(moving_averages.keys())
    if len(windows) < 2:
        raise ValueError("ma_crossover_features: need at least 2 moving averages to compute crossover pairs.")

    out: Dict[str, np.ndarray] = {}
    vote_sum: Optional[np.ndarray] = None
    for short, long in combinations(windows, 2):
        diff = (moving_averages[short] - moving_averages[long]).to_numpy()
        vote = np.sign(diff)
        out[f"{column_prefix}_diff_{short}_{long}"] = diff
        out[f"{column_prefix}_vote_{short}_{long}"] = vote
        # accumulate the running sum incrementally (treating NaN as 0, matching pandas' default skipna sum)
        # instead of a separate np.stack + np.nansum pass over all pairs -- avoids materializing an extra
        # (n_rows, n_pairs) array just to reduce it away; measured as the dominant cProfile cost otherwise.
        vote_contribution = np.where(np.isnan(vote), 0.0, vote)
        if long_window_weight_power:
            # weight by the SHORT leg's window size, not the long leg's: a pair like (3, 90) still has a
            # noisy short MA driving most of its sign flips, so weighting on `long` alone (its prior
            # formulation) gave noise-dominated cross pairs the same huge weight as genuinely reliable
            # long-vs-long pairs. Weighting on `short` penalizes every pair anchored to a noisy short MA,
            # regardless of how long its partner is, which is what actually suppresses false flips.
            vote_contribution = vote_contribution * (float(short) ** long_window_weight_power)
        vote_sum = vote_contribution if vote_sum is None else vote_sum + vote_contribution

    assert vote_sum is not None  # guaranteed: len(windows) >= 2 checked above -> at least one pair iterated.
    out[f"{column_prefix}_vote_sum"] = vote_sum
    if group_ids is None:
        vote_sum_delta = np.empty_like(vote_sum)
        vote_sum_delta[0] = np.nan
        vote_sum_delta[1:] = vote_sum[1:] - vote_sum[:-1]
    else:
        prev = per_group_shift(vote_sum, np.asarray(group_ids), n=1)
        vote_sum_delta = vote_sum - prev
    out[f"{column_prefix}_vote_sum_delta"] = vote_sum_delta
    return pd.DataFrame(out, index=next(iter(moving_averages.values())).index)


__all__ = ["ma_crossover_features"]
