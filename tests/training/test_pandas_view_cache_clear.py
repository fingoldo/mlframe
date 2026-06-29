"""clear_pandas_view_cache must drop the single-entry get_pandas_view_of_polars_df memo so its Arrow-backed pandas
view stops pinning the polars buffers when ctx frames are released (prod: 8 GB expected, 0 reclaimed)."""
from __future__ import annotations

import polars as pl

import mlframe.training.utils as U


def test_clear_empties_the_memo_after_a_view_is_cached():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    U.get_pandas_view_of_polars_df(df)
    # The memo now retains the last view (it pins the polars buffers zero-copy).
    assert U._PD_VIEW_LAST_CACHE["result"] is not None
    U.clear_pandas_view_cache()
    assert U._PD_VIEW_LAST_CACHE["id_key"] is None
    assert U._PD_VIEW_LAST_CACHE["result"] is None
