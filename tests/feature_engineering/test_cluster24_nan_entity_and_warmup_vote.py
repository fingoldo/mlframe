"""Cluster 24 regressions: a NaN entity is aggregated, not a crash, and warm-up rows get a NaN consensus, not 0 (audit 2026-11-20, findings 24-F40 and 24-F35).

Windowed edge aggregation listed NaN as an entity but the grouping dropped it, so the lookup raised KeyError. The MA crossover consensus
summed NaN votes as zero, so a row where every moving average is still warming up read as a neutral consensus.
"""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.ma_crossover import ma_crossover_features
from mlframe.feature_engineering.windowed_edge_diff import windowed_edge_aggregate_diff


def test_nan_entity_gets_its_own_row_instead_of_a_key_error():
    """A NaN entity is aggregated over its own records, and the other entities are unchanged (24-F40)."""
    df = pd.DataFrame(
        {
            "e": ["a", np.nan, "a", np.nan, "a"],
            "t": [1, 1, 2, 2, 3],
            "v": [1.0, 10.0, 2.0, 20.0, 3.0],
        }
    )
    out = windowed_edge_aggregate_diff(df, "e", "t", "v", n=1, agg="mean")
    assert list(out["e"].astype(object).fillna("NAN")) == ["a", "NAN"]
    a_row = out.loc[out["e"] == "a"].iloc[0]
    nan_row = out.loc[out["e"].isna()].iloc[0]
    assert a_row["v_first1_mean"] == 1.0 and a_row["v_last1_mean"] == 3.0
    assert nan_row["v_first1_mean"] == 10.0 and nan_row["v_last1_mean"] == 20.0


def test_warmup_row_with_no_vote_has_nan_consensus():
    """A row where every MA pair is still warming up has vote_sum NaN; a row with a real vote keeps its sum (24-F35)."""
    idx = pd.RangeIndex(4)
    mas = {
        3: pd.Series([np.nan, np.nan, 1.0, 2.0], index=idx),
        5: pd.Series([np.nan, np.nan, 0.5, 1.0], index=idx),
    }
    out = ma_crossover_features(mas, "ma")
    vote_sum = out["ma_vote_sum"].to_numpy()
    assert np.isnan(vote_sum[0]) and np.isnan(vote_sum[1])
    assert vote_sum[2] == 1.0 and vote_sum[3] == 1.0
