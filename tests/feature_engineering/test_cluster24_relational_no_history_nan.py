"""Cluster 24 regression: a parent with no eligible history keeps NaN in the chained aggregates, not zero (audit 2026-11-20, finding 24-F56).

stack_relational_chain zero-filled the new columns at every hop, so a parent with no child history got mean=0, which reads as a real zero average.
The parent with a history keeps its finite values; the one without keeps NaN.
"""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.relational_dfs import ChildTableSpec, RelationalHop, stack_relational_chain


def test_parent_without_history_is_nan_not_zero():
    """p2 has no hop rows, so its new chained columns are NaN; p1, which has history, stays finite (24-F56)."""
    parents = pd.DataFrame({"pid": ["p1", "p2"], "cutoff": [100.0, 100.0]})
    hop_df = pd.DataFrame({"hid": ["h1"], "pid": ["p1"], "t": [1.0]})
    leaf_df = pd.DataFrame({"hid": ["h1"], "t": [1.0], "val": [5.0]})
    hop = RelationalHop(df=hop_df, id_col="hid", time_col="t", foreign_key_col="pid", value_cols={}, prefix="h")
    leaf = ChildTableSpec(child_df=leaf_df, foreign_key_col="hid", time_col="t", value_cols={"val": ["mean", "sum"]}, prefix="leaf")

    out = stack_relational_chain(parents, parent_id_col="pid", cutoff_col="cutoff", hops=[hop], leaf_specs=[leaf], prefix="l2")
    new_cols = [c for c in out.columns if c not in parents.columns]
    assert new_cols, "the chain produced no feature columns"
    p1 = out.loc[out["pid"] == "p1", new_cols].to_numpy(dtype=float)
    p2 = out.loc[out["pid"] == "p2", new_cols].to_numpy(dtype=float)
    assert np.isfinite(p1).any()
    assert np.isnan(p2).all()
