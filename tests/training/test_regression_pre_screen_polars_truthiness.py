"""Regression: the unsupervised pre-screen must not choke on a polars-only train frame.

Pre-fix the train-frame selection used ``ctx.train_df_polars or ctx.train_df_pd``. ``bool(pl.DataFrame)``
raises a TypeError (ambiguous truthiness); the outer except then swallowed it, set
``_pre_screen_dropped_cols=[]`` and latched ``_pre_screen_done=True`` -> the polars-only pre-screen was
silently skipped (no columns ever dropped).
"""

from __future__ import annotations

import types

import pytest

pl = pytest.importorskip("polars")


class _FSCfg:
    """Groups tests covering f s cfg."""
    pre_screen_unsupervised = True
    pre_screen_variance_threshold = 0.0
    pre_screen_null_fraction_threshold = 0.99


def _make_ctx(df):
    """Make ctx."""
    return types.SimpleNamespace(
        feature_selection_config=_FSCfg(),
        _pre_screen_done=False,
        _pre_screen_dropped_cols=None,
        filtered_train_df=None,
        train_df_polars=df,
        train_df_pd=None,
        filtered_val_df=None,
        val_df_polars=None,
        test_df_polars=None,
        val_df_pd=None,
        test_df_pd=None,
        target_by_type={"regression": {"y": None}},
        cat_features=None,
        text_features=None,
        embedding_features=None,
        group_id_col=None,
        ts_field=None,
        extractor=None,
        features_and_targets_extractor=None,
        split_config=None,
        metadata={},
        verbose=False,
    )


def test_pre_screen_runs_on_polars_only_frame_and_drops_constant_col():
    """Pre screen runs on polars only frame and drops constant col."""
    from mlframe.training.core._phase_train_one_target_pre_screen import _maybe_run_unsupervised_pre_screen

    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "const_col": [7.0, 7.0, 7.0, 7.0, 7.0],  # variance 0 -> droppable
            "informative": [0.1, 0.9, 0.2, 0.8, 0.3],
        }
    )
    ctx = _make_ctx(df)

    _maybe_run_unsupervised_pre_screen(ctx, {"y": None})

    assert ctx._pre_screen_done is True
    # Pre-fix: TypeError -> swallowed -> empty list. Post-fix: the constant column is found and dropped.
    assert ctx._pre_screen_dropped_cols, "polars-only pre-screen must compute drops, not silently skip"
    assert "const_col" in ctx._pre_screen_dropped_cols
    assert "y" not in ctx._pre_screen_dropped_cols  # protected target
