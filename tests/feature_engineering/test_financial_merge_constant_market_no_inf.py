"""Regression test: ``merge_perticker_and_wholemarket_features`` ranking must not emit inf/NaN for a market-constant column.

The bug (fixed): the rank ``(col - wm_min) / (wm_max - wm_min)`` divides by zero when the whole market is constant on that column (``wm_max == wm_min``), producing
inf/NaN. The fix wraps the ranking expression in ``pllib.clean_numeric`` (matching the other ranking sites), which maps non-finite results to the nans_filler.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlframe.feature_engineering.financial import merge_perticker_and_wholemarket_features


pytestmark = pytest.mark.fast


def test_merge_rankings_constant_market_column_no_inf_nan():
    perticker = pl.DataFrame(
        {
            "date": [1, 2, 3, 4],
            "feat": [1.0, 2.0, 3.0, 4.0],
        }
    )
    # Market is CONSTANT on `feat`: wm_min == wm_max -> denominator zero in the rank expr.
    wholemarket = pl.DataFrame(
        {
            "date": [1, 2, 3, 4],
            "feat_wm_min": [5.0, 5.0, 5.0, 5.0],
            "feat_wm_max": [5.0, 5.0, 5.0, 5.0],
        }
    )

    out = merge_perticker_and_wholemarket_features(perticker, wholemarket, timestamp_column="date")
    rnk = out["feat_wm_rnk"].to_numpy()
    assert np.isfinite(rnk).all(), (
        f"feat_wm_rnk contains inf/NaN ({rnk}) for a market-constant column; the rank expression must be wrapped in clean_numeric to guard the zero denominator."
    )
