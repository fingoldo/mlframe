"""Regression sensor for FU-w3a-1.

Before the fix the polars default-candidate-column branch in detect_group_column_candidates excluded ALL numeric columns via _is_numeric_column, while the pandas branch kept low-cardinality integer columns via the int-as-cat heuristic. A frame with only an int64 gid column therefore returned [] under polars and the gid candidate under pandas; the test pins parity between the two flavours when callers rely on default candidate discovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from mlframe.training.composite.discovery.auto_detect import detect_group_column_candidates


def _make_gid_frames(seed: int = 11):
    rng = np.random.default_rng(seed)
    group_ids = np.tile(np.arange(50, dtype=np.int64), 30)
    rng.shuffle(group_ids)
    pd_df = pd.DataFrame({"gid": group_ids})
    pl_df = pl.DataFrame({"gid": group_ids})
    return pd_df, pl_df


def test_default_candidates_polars_matches_pandas_for_lowcard_int_gid():
    pd_df, pl_df = _make_gid_frames()
    pd_names = {n for n, _ in detect_group_column_candidates(pd_df)}
    pl_names = {n for n, _ in detect_group_column_candidates(pl_df)}
    assert "gid" in pd_names, "pandas branch already kept low-card int gid via int-as-cat heuristic"
    assert "gid" in pl_names, "polars default-candidates must mirror pandas int-as-cat: low-card int columns are valid group keys (FU-w3a-1)"
    assert pd_names == pl_names, f"polars / pandas default-candidates diverged: pd={pd_names}, pl={pl_names}"


def test_default_candidates_polars_excludes_highcard_int():
    n = 1000
    rng = np.random.default_rng(13)
    big_ids = rng.integers(0, n, size=n, dtype=np.int64)
    pl_df = pl.DataFrame({"bigid": big_ids, "f": rng.normal(size=n)})
    names = {nm for nm, _ in detect_group_column_candidates(pl_df, max_unique=50)}
    assert "bigid" not in names, "high-cardinality int column must be excluded by max_unique gate"
    assert "f" not in names, "float column must remain excluded under default candidates"
