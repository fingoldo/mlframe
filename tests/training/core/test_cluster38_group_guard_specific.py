"""Cluster 38 regression: the group-aware pre-screen skip depends on the group column, not on any protected column (audit 2026-11-20, finding 38-F9).

Target names always fill the protected set, so a guard on the whole set never fired once a target existed. The pre-screen is skipped when the
split uses groups and the group column itself is not protected, and runs when it is.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.configs import FeatureSelectionConfig
from mlframe.training.core import _phase_train_one_target_pre_screen as ps


def _ctx(df, *, group_name, cat_features):
    """A suite context with the pre-screen enabled, a group-aware split and a categorical column so the protected set is not empty."""
    group_ids = pd.Series(np.arange(len(df)) % 5, name=group_name) if group_name else None
    return SimpleNamespace(
        feature_selection_config=FeatureSelectionConfig(),
        _pre_screen_done=False,
        _pre_screen_dropped_cols=None,
        target_by_type={},
        cat_features=list(cat_features),
        text_features=[],
        embedding_features=[],
        group_id_col=group_name,
        group_ids_raw=group_ids,
        ts_field=None,
        extractor=None,
        features_and_targets_extractor=None,
        split_config=SimpleNamespace(use_groups=True),
        metadata=None,
        verbose=0,
        filtered_train_df=df,
        filtered_val_df=None,
        train_df_pd=df,
        val_df_pd=None,
        test_df_pd=None,
        train_df_polars=None,
        val_df_polars=None,
        test_df_polars=None,
    )


def _frame(n=200):
    """A small frame with a categorical column that is always protected."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"c": rng.choice(["a", "b", "c"], n), "x": rng.normal(size=n), "y": rng.normal(size=n)})


def test_group_aware_split_with_unprotected_group_column_skips_the_pre_screen(caplog):
    """With groups in use and the group column not protected, the pre-screen is skipped and says so (38-F9)."""
    df = _frame()
    ctx = _ctx(df, group_name=None, cat_features=["c"])
    with caplog.at_level(logging.WARNING):
        ps._maybe_run_unsupervised_pre_screen(ctx, [])
    assert ctx._pre_screen_done is True
    assert ctx._pre_screen_dropped_cols == []
    assert any("skipping the unsupervised pre-screen" in r.getMessage() for r in caplog.records)


def test_group_column_that_is_protected_lets_the_pre_screen_run(caplog):
    """Control: when the group column is named and protected, the pre-screen runs instead of skipping (38-F9)."""
    df = _frame()
    ctx = _ctx(df, group_name="g", cat_features=["c"])
    df["g"] = ctx.group_ids_raw.to_numpy()
    ctx.filtered_train_df = df
    ctx.train_df_pd = df
    with caplog.at_level(logging.WARNING):
        ps._maybe_run_unsupervised_pre_screen(ctx, [])
    assert not any("skipping the unsupervised pre-screen" in r.getMessage() for r in caplog.records)
