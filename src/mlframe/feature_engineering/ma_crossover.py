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


def ma_crossover_features(moving_averages: Dict[int, pd.Series], column_prefix: str = "ma_crossover") -> pd.DataFrame:
    """Pairwise MA differences, sign votes, and a summed crossover score.

    Parameters
    ----------
    moving_averages
        ``{window_size: moving_average_series}`` -- precomputed rolling means (or EWMAs/Hull MAs) at
        different window sizes, all aligned to the same index. At least 2 entries required.
    column_prefix
        Prefix for emitted column names.

    Returns
    -------
    pd.DataFrame
        For every ``(short, long)`` window pair (``short < long``): ``{prefix}_diff_{short}_{long}`` (signed
        difference) and ``{prefix}_vote_{short}_{long}`` (``sign(diff)`` in ``{-1, 0, 1}``), plus a single
        ``{prefix}_vote_sum`` column (sum of all vote columns -- a consensus trend-direction score).
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
        vote_sum = vote_contribution if vote_sum is None else vote_sum + vote_contribution

    assert vote_sum is not None  # guaranteed: len(windows) >= 2 checked above -> at least one pair iterated.
    out[f"{column_prefix}_vote_sum"] = vote_sum
    return pd.DataFrame(out, index=next(iter(moving_averages.values())).index)


__all__ = ["ma_crossover_features"]
